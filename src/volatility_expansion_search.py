from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, evaluate_position_with_cost
from src.excess_reversion_search import cap_trades_per_day, same_hour_random_position
from src.hmm_lab import _lab_cfg, _target_symbol, build_lab_folds, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table


VARIANT_FLAGS: dict[str, dict[str, bool]] = {
    "compression_breakout": {"volume": False, "cross_asset": False},
    "compression_volume": {"volume": True, "cross_asset": False},
    "compression_cross_asset": {"volume": False, "cross_asset": True},
    "compression_volume_cross_asset": {"volume": True, "cross_asset": True},
}

REQUIRED_FEATURE_COLUMNS = sorted(
    {
        "target_open_next",
        "target_ret_3",
        "target_range_ratio_2_8",
        "target_rv_4_rel_by_bar",
        "target_breaks_roll_high_4",
        "target_breaks_roll_low_4",
        "target_breakout_margin_roll_high_4_atr",
        "target_breakdown_margin_roll_low_4_atr",
        "target_close_location_bar",
        "target_rel_volume_by_bar",
        "target_rel_volume_accel_2",
        "target_minutes_from_open",
        "target_minutes_to_close",
        "positive_index_count_2",
        "positive_sector_count_2",
        "index_above_vwap_count",
        "sector_above_vwap_count",
        "spread_credit_12",
        "risk_on_score",
        "risk_off_score",
        "intraday_stress_score",
        "cross_asset_vol_expansion_score",
    }
)


def _search_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("volatility_expansion_search", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "volatility_expansion_search.md"


def _json_dumps(values: dict[str, Any]) -> str:
    serializable = {}
    for key, value in sorted(values.items()):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            continue
        serializable[str(key)] = value
    return json.dumps(serializable, sort_keys=True)


def _json_loads(value: str | float | int | None) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    return dict(json.loads(str(value)))


def _clean_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].replace([np.inf, -np.inf], np.nan).astype(float)


def _bool_series(frame: pd.DataFrame, value: bool) -> pd.Series:
    return pd.Series(bool(value), index=frame.index, dtype=bool)


def _bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return _bool_series(frame, False)
    return frame[column].fillna(False).astype(bool)


def _quantile(frame: pd.DataFrame, column: str, q: float, default: float = np.nan) -> float:
    values = _clean_series(frame, column).dropna()
    if values.empty:
        return float(default)
    return float(values.quantile(float(q)))


def _positive_quantile(frame: pd.DataFrame, columns: list[str], q: float, default: float = np.nan) -> float:
    parts = []
    for column in columns:
        if column in frame:
            values = _clean_series(frame, column).dropna()
            parts.append(values[values > 0.0])
    if not parts:
        return float(default)
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return float(default)
    return float(combined.quantile(float(q)))


def _cost_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = names if names is not None else [str(value) for value in _search_cfg(config).get("cost_scenarios", ["ibkr_tiered_10000", "bps_2", "bps_5"])]
    by_name = {str(scenario["cost_scenario"]): scenario for scenario in cost_scenarios(config)}
    return [by_name[name] for name in wanted if name in by_name]


def _candidate_id(row: pd.Series | dict[str, Any]) -> str:
    return "__".join(
        [
            f"fold{int(row['fold'])}",
            str(row["variant"]),
            str(row["side"]),
            f"h{int(row['horizon_bars'])}",
            f"cq{float(row['compression_quantile']):g}",
            f"rvq{float(row['rv_compression_quantile']):g}",
            f"mq{float(row['breakout_margin_quantile']):g}",
            f"volq{float(row['volume_quantile']):g}",
            str(row["vol_filter_name"]),
            str(row.get("hour_filter_name", "all")),
        ]
    )


def _stable_seed(candidate_id: str, split: str, bucket: str) -> int:
    digest = hashlib.sha256(f"{candidate_id}::{split}::{bucket}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def load_feature_frame(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    path = features_input_path(config, target_symbol, feature_config)
    features = pd.read_parquet(path)
    return features.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)


def _add_prior_compression_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(["horizon_bars", "session", "bar_index"], kind="stable").copy()
    group = output.groupby(["horizon_bars", "session"], sort=False)
    for column, new_column in [
        ("target_range_ratio_2_8", "prior_target_range_ratio_2_8"),
        ("target_rv_4_rel_by_bar", "prior_target_rv_4_rel_by_bar"),
        ("target_rel_volume_by_bar", "prior_target_rel_volume_by_bar"),
    ]:
        if column in output:
            output[new_column] = group[column].shift(1)
    return output.sort_index(kind="stable")


def build_split_dataset(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _search_cfg(config)
    horizons = [int(value) for value in cfg.get("horizons", [2, 3, 4])]
    features = load_feature_frame(config, target_symbol)
    forward_returns = build_forward_returns(features, horizons)
    indexed = features.reset_index(names="source_index")
    available = [
        "source_index",
        *[column for column in REQUIRED_FEATURE_COLUMNS if column in indexed.columns and column not in forward_returns.columns],
    ]
    merged = forward_returns.merge(indexed.loc[:, available], on="source_index", how="left", validate="many_to_one")
    merged = _add_prior_compression_columns(merged)

    folds = build_lab_folds(features, config)
    splits: list[pd.DataFrame] = []
    for fold in folds:
        split_sessions = {
            "train": set(fold.train_sessions),
            "validation": set(fold.validation_sessions),
            "test": set(fold.test_sessions),
        }
        for split, sessions in split_sessions.items():
            part = merged[merged["session"].isin(sessions)].copy()
            if part.empty:
                continue
            part.insert(0, "split", split)
            part.insert(0, "fold", int(fold.fold))
            splits.append(part)
    if not splits:
        return pd.DataFrame()
    output = pd.concat(splits, ignore_index=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"])
    output["proposed_label"] = "no_hmm"
    output["feature_set"] = "volatility_expansion"
    output["n_states"] = 0
    output["seed"] = 0
    return output


def train_thresholds(train_frame: pd.DataFrame, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    cfg = _search_cfg(config)
    thresholds = {
        "range_compression_max": _quantile(train_frame, "prior_target_range_ratio_2_8", float(spec["compression_quantile"]), np.inf),
        "rv_compression_max": _quantile(train_frame, "prior_target_rv_4_rel_by_bar", float(spec["rv_compression_quantile"]), np.inf),
        "breakout_margin_min": _positive_quantile(
            train_frame,
            ["target_breakout_margin_roll_high_4_atr", "target_breakdown_margin_roll_low_4_atr"],
            float(spec["breakout_margin_quantile"]),
            0.0,
        ),
        "volume_accel_min": _quantile(train_frame, "target_rel_volume_accel_2", float(spec["volume_quantile"]), -np.inf),
        "relative_volume_min": _quantile(train_frame, "target_rel_volume_by_bar", float(spec["volume_quantile"]), -np.inf),
        "positive_index_low": _quantile(train_frame, "positive_index_count_2", float(cfg.get("breadth_low_quantile", 0.35)), -np.inf),
        "positive_index_high": _quantile(train_frame, "positive_index_count_2", float(cfg.get("breadth_high_quantile", 0.65)), np.inf),
        "positive_sector_low": _quantile(train_frame, "positive_sector_count_2", float(cfg.get("breadth_low_quantile", 0.35)), -np.inf),
        "positive_sector_high": _quantile(train_frame, "positive_sector_count_2", float(cfg.get("breadth_high_quantile", 0.65)), np.inf),
        "index_above_vwap_low": _quantile(train_frame, "index_above_vwap_count", float(cfg.get("breadth_low_quantile", 0.35)), -np.inf),
        "index_above_vwap_high": _quantile(train_frame, "index_above_vwap_count", float(cfg.get("breadth_high_quantile", 0.65)), np.inf),
        "sector_above_vwap_low": _quantile(train_frame, "sector_above_vwap_count", float(cfg.get("breadth_low_quantile", 0.35)), -np.inf),
        "sector_above_vwap_high": _quantile(train_frame, "sector_above_vwap_count", float(cfg.get("breadth_high_quantile", 0.65)), np.inf),
        "credit_low": _quantile(train_frame, "spread_credit_12", float(cfg.get("credit_low_quantile", 0.35)), -np.inf),
        "credit_high": _quantile(train_frame, "spread_credit_12", float(cfg.get("credit_high_quantile", 0.65)), np.inf),
        "risk_on_high": _quantile(train_frame, "risk_on_score", float(cfg.get("risk_high_quantile", 0.70)), np.inf),
        "risk_off_high": _quantile(train_frame, "risk_off_score", float(cfg.get("risk_high_quantile", 0.70)), np.inf),
        "stress_high": _quantile(train_frame, "intraday_stress_score", float(cfg.get("risk_high_quantile", 0.70)), np.inf),
        "vol_expansion_high": _quantile(train_frame, "cross_asset_vol_expansion_score", float(cfg.get("vol_high_quantile", 0.70)), np.inf),
    }
    if not np.isfinite(thresholds["range_compression_max"]) or not np.isfinite(thresholds["rv_compression_max"]):
        return {}
    thresholds["min_minutes_from_open"] = float(cfg.get("min_minutes_from_open", 45))
    thresholds["min_minutes_to_close"] = float(cfg.get("min_minutes_to_close", 30))
    thresholds["close_location_long_min"] = float(cfg.get("close_location_long_min", 0.65))
    thresholds["close_location_short_max"] = float(cfg.get("close_location_short_max", 0.35))
    return thresholds


def _time_mask(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    from_open = _clean_series(frame, "target_minutes_from_open", np.inf)
    to_close = _clean_series(frame, "target_minutes_to_close", np.inf)
    return from_open.ge(float(thresholds.get("min_minutes_from_open", 0.0))) & to_close.ge(float(thresholds.get("min_minutes_to_close", 0.0)))


def breakout_position(frame: pd.DataFrame, spec: dict[str, Any], thresholds: dict[str, Any]) -> pd.Series:
    long_mask = (
        _time_mask(frame, thresholds)
        & _bool_col(frame, "target_breaks_roll_high_4")
        & _clean_series(frame, "target_breakout_margin_roll_high_4_atr", 0.0).ge(float(thresholds.get("breakout_margin_min", 0.0)))
    )
    short_mask = (
        _time_mask(frame, thresholds)
        & _bool_col(frame, "target_breaks_roll_low_4")
        & _clean_series(frame, "target_breakdown_margin_roll_low_4_atr", 0.0).ge(float(thresholds.get("breakout_margin_min", 0.0)))
    )
    side = str(spec.get("side", "both"))
    if side == "long":
        short_mask = _bool_series(frame, False)
    elif side == "short":
        long_mask = _bool_series(frame, False)
    elif side != "both":
        raise ValueError(f"Unsupported side: {side}")
    position = pd.Series(0.0, index=frame.index)
    position.loc[long_mask] = 1.0
    position.loc[short_mask] = -1.0
    return position


def _compression_mask(frame: pd.DataFrame, thresholds: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    range_ok = _clean_series(frame, "prior_target_range_ratio_2_8").le(float(thresholds["range_compression_max"]))
    rv_ok = _clean_series(frame, "prior_target_rv_4_rel_by_bar").le(float(thresholds["rv_compression_max"]))
    mode = str(_search_cfg(config).get("compression_mode", "all"))
    if mode == "all":
        return range_ok & rv_ok
    if mode == "any":
        return range_ok | rv_ok
    raise ValueError(f"Unsupported compression_mode: {mode}")


def _close_location_mask(frame: pd.DataFrame, position: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    close_location = _clean_series(frame, "target_close_location_bar")
    return ((position > 0.0) & close_location.ge(float(thresholds["close_location_long_min"]))) | (
        (position < 0.0) & close_location.le(float(thresholds["close_location_short_max"]))
    )


def _volume_expansion_mask(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    accel = _clean_series(frame, "target_rel_volume_accel_2")
    rel_volume = _clean_series(frame, "target_rel_volume_by_bar")
    return accel.ge(float(thresholds.get("volume_accel_min", -np.inf))) & rel_volume.ge(float(thresholds.get("relative_volume_min", -np.inf)))


def _cross_asset_confirm_mask(frame: pd.DataFrame, position: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    positive_index = _clean_series(frame, "positive_index_count_2")
    positive_sector = _clean_series(frame, "positive_sector_count_2")
    index_vwap = _clean_series(frame, "index_above_vwap_count")
    sector_vwap = _clean_series(frame, "sector_above_vwap_count")
    credit = _clean_series(frame, "spread_credit_12")
    risk_on = _clean_series(frame, "risk_on_score")
    risk_off = _clean_series(frame, "risk_off_score")
    long_confirm = (
        (
            positive_index.ge(float(thresholds.get("positive_index_high", np.inf)))
            | positive_sector.ge(float(thresholds.get("positive_sector_high", np.inf)))
            | index_vwap.ge(float(thresholds.get("index_above_vwap_high", np.inf)))
            | sector_vwap.ge(float(thresholds.get("sector_above_vwap_high", np.inf)))
            | risk_on.ge(float(thresholds.get("risk_on_high", np.inf)))
        )
        & credit.ge(float(thresholds.get("credit_low", -np.inf)))
    )
    short_confirm = (
        (
            positive_index.le(float(thresholds.get("positive_index_low", -np.inf)))
            | positive_sector.le(float(thresholds.get("positive_sector_low", -np.inf)))
            | index_vwap.le(float(thresholds.get("index_above_vwap_low", -np.inf)))
            | sector_vwap.le(float(thresholds.get("sector_above_vwap_low", -np.inf)))
            | risk_off.ge(float(thresholds.get("risk_off_high", np.inf)))
        )
        & credit.le(float(thresholds.get("credit_high", np.inf)))
    )
    return ((position > 0.0) & long_confirm) | ((position < 0.0) & short_confirm)


def _vol_filter_mask(frame: pd.DataFrame, thresholds: dict[str, Any], vol_filter_name: str) -> pd.Series:
    if vol_filter_name == "none":
        return _bool_series(frame, True)
    if vol_filter_name == "exclude_stress":
        return _clean_series(frame, "intraday_stress_score").lt(float(thresholds.get("stress_high", np.inf)))
    if vol_filter_name == "exclude_cross_asset_vol_extreme":
        return _clean_series(frame, "cross_asset_vol_expansion_score").lt(float(thresholds.get("vol_expansion_high", np.inf)))
    raise ValueError(f"Unsupported volatility filter: {vol_filter_name}")


def _hour_filter_mask(frame: pd.DataFrame, hour_filter_name: str) -> pd.Series:
    if hour_filter_name in {"", "all", "none"}:
        return _bool_series(frame, True)
    hours = _clean_series(frame, "hour", np.nan).astype("Int64")
    if hour_filter_name.startswith("hour_"):
        return hours.eq(int(hour_filter_name.removeprefix("hour_"))).fillna(False)
    if hour_filter_name.startswith("hours_"):
        values = [int(value) for value in hour_filter_name.removeprefix("hours_").split("_") if value]
        if len(values) == 2 and values[0] < values[1]:
            values = list(range(values[0], values[1] + 1))
        return hours.isin(values).fillna(False)
    if hour_filter_name.startswith("exclude_"):
        return ~hours.eq(int(hour_filter_name.removeprefix("exclude_"))).fillna(False)
    raise ValueError(f"Unsupported hour filter: {hour_filter_name}")


def _apply_hour_filter(position: pd.Series, frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    hour_filter_name = str(spec.get("hour_filter_name", "all"))
    if hour_filter_name in {"", "all", "none"}:
        return position
    return position.where(_hour_filter_mask(frame, hour_filter_name), 0.0)


def compression_only_position(frame: pd.DataFrame, spec: dict[str, Any], thresholds: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    compressed = _time_mask(frame, thresholds) & _compression_mask(frame, thresholds, config)
    recent = _clean_series(frame, "target_ret_3", 0.0)
    position = pd.Series(0.0, index=frame.index)
    position.loc[compressed & recent.gt(0.0)] = 1.0
    position.loc[compressed & recent.lt(0.0)] = -1.0
    side = str(spec.get("side", "both"))
    if side == "long":
        position = position.where(position > 0.0, 0.0)
    elif side == "short":
        position = position.where(position < 0.0, 0.0)
    position = _apply_hour_filter(position, frame, spec)
    return cap_trades_per_day(position, frame, int(spec.get("max_trades_per_day", _search_cfg(config).get("max_trades_per_day", 2))))


def expansion_position(frame: pd.DataFrame, spec: dict[str, Any], thresholds: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    flags = VARIANT_FLAGS[str(spec["variant"])]
    position = breakout_position(frame, spec, thresholds)
    active = (
        position.abs().gt(0.0)
        & _compression_mask(frame, thresholds, config)
        & _close_location_mask(frame, position, thresholds)
        & _vol_filter_mask(frame, thresholds, str(spec.get("vol_filter_name", "none")))
        & _hour_filter_mask(frame, str(spec.get("hour_filter_name", "all")))
    )
    if flags["volume"]:
        active &= _volume_expansion_mask(frame, thresholds)
    if flags["cross_asset"]:
        active &= _cross_asset_confirm_mask(frame, position, thresholds)
    position = position.where(active, 0.0)
    return cap_trades_per_day(position, frame, int(spec.get("max_trades_per_day", _search_cfg(config).get("max_trades_per_day", 2))))


def _control_positions(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    thresholds: dict[str, Any],
    config: dict[str, Any],
    candidate_id: str,
    split: str,
) -> dict[str, pd.Series]:
    alpha = expansion_position(frame, spec, thresholds, config)
    breakout_only_position = _apply_hour_filter(breakout_position(frame, spec, thresholds), frame, spec)
    breakout_only = cap_trades_per_day(
        breakout_only_position,
        frame,
        int(spec.get("max_trades_per_day", _search_cfg(config).get("max_trades_per_day", 2))),
    )
    compression_only = compression_only_position(frame, spec, thresholds, config)
    random_control = same_hour_random_position(frame, alpha, _stable_seed(candidate_id, split, "same_hour_random_control"))
    return {
        "alpha_signal": alpha,
        "breakout_only_control": breakout_only,
        "compression_only_control": compression_only,
        "same_hour_random_control": random_control,
        "inverted_signal": -alpha,
        "always_flat": pd.Series(0.0, index=frame.index),
    }


def generate_candidate_specs(train_frame: pd.DataFrame, fold: int, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _search_cfg(config)
    rows: list[dict[str, Any]] = []
    for variant in [str(value) for value in cfg.get("variants", list(VARIANT_FLAGS))]:
        if variant not in VARIANT_FLAGS:
            continue
        for side in [str(value) for value in cfg.get("sides", ["long", "short"])]:
            for compression_q in [float(value) for value in cfg.get("compression_quantiles", [0.25, 0.35])]:
                for rv_q in [float(value) for value in cfg.get("rv_compression_quantiles", [0.35])]:
                    for margin_q in [float(value) for value in cfg.get("breakout_margin_quantiles", [0.4, 0.6])]:
                        for volume_q in [float(value) for value in cfg.get("volume_quantiles", [0.55])]:
                            for vol_filter in [str(value) for value in cfg.get("vol_filters", ["none"])]:
                                for hour_filter in [str(value) for value in cfg.get("hour_filters", ["all"])]:
                                    spec = {
                                        "fold": int(fold),
                                        "variant": variant,
                                        "side": side,
                                        "horizon_bars": int(horizon),
                                        "compression_quantile": compression_q,
                                        "rv_compression_quantile": rv_q,
                                        "breakout_margin_quantile": margin_q,
                                        "volume_quantile": volume_q,
                                        "vol_filter_name": vol_filter,
                                        "hour_filter_name": hour_filter,
                                        "max_trades_per_day": int(cfg.get("max_trades_per_day", 2)),
                                    }
                                    thresholds = train_thresholds(train_frame, spec, config)
                                    if not thresholds:
                                        continue
                                    spec["thresholds_json"] = _json_dumps(thresholds)
                                    spec["candidate_id"] = _candidate_id(spec)
                                    rows.append(spec)
    return pd.DataFrame(rows).drop_duplicates("candidate_id").reset_index(drop=True) if rows else pd.DataFrame()


def evaluate_candidate(
    frame: pd.DataFrame,
    split: str,
    spec: pd.Series | dict[str, Any],
    scenario: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    spec_dict = dict(spec)
    thresholds = _json_loads(spec_dict["thresholds_json"])
    candidate_id = str(spec_dict.get("candidate_id") or _candidate_id(spec_dict))
    positions = _control_positions(frame, spec_dict, thresholds, config, candidate_id, split)
    rows: list[dict[str, Any]] = []
    for bucket, position in positions.items():
        row = {
            **{
                key: spec_dict.get(key, "all" if key == "hour_filter_name" else None)
                for key in [
                    "fold",
                    "variant",
                    "side",
                    "horizon_bars",
                    "compression_quantile",
                    "rv_compression_quantile",
                    "breakout_margin_quantile",
                    "volume_quantile",
                    "vol_filter_name",
                    "hour_filter_name",
                ]
            },
            "split": split,
            "bucket": bucket,
            "thresholds_json": str(spec_dict["thresholds_json"]),
            "candidate_id": candidate_id,
            "cost_scenario": scenario["cost_scenario"],
            "cost_kind": scenario["cost_kind"],
            "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
            "notional_usd": float(scenario.get("notional_usd", np.nan)),
        }
        rows.append({**row, **evaluate_position_with_cost(frame, position, scenario)})
    return add_deltas(pd.DataFrame(rows))


def add_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = ["candidate_id", "split", "cost_scenario"]
    signal = rows[rows["bucket"].eq("alpha_signal")].copy()
    metrics = ["net_return", "daily_sharpe", "max_drawdown", "turnover", "avg_trade_net", "profit_factor"]
    controls = [
        ("breakout_only_control", "breakout"),
        ("compression_only_control", "compression"),
        ("same_hour_random_control", "random"),
        ("inverted_signal", "inverted"),
    ]
    for control_bucket, suffix in controls:
        control = rows[rows["bucket"].eq(control_bucket)].loc[:, [*keys, *metrics]].rename(
            columns={metric: f"{suffix}_{metric}" for metric in metrics}
        )
        signal = signal.merge(control, on=keys, how="left", validate="one_to_one")
    signal["net_delta_vs_breakout_only"] = signal["net_return"] - signal["breakout_net_return"]
    signal["net_delta_vs_compression_only"] = signal["net_return"] - signal["compression_net_return"]
    signal["net_delta_vs_random"] = signal["net_return"] - signal["random_net_return"]
    signal["net_delta_vs_inverted"] = signal["net_return"] - signal["inverted_net_return"]
    signal["daily_sharpe_delta_vs_breakout_only"] = signal["daily_sharpe"] - signal["breakout_daily_sharpe"]
    signal["daily_sharpe_delta_vs_random"] = signal["daily_sharpe"] - signal["random_daily_sharpe"]
    signal["drawdown_reduction_vs_breakout_only"] = signal["breakout_max_drawdown"] - signal["max_drawdown"]
    signal["drawdown_reduction_vs_compression_only"] = signal["compression_max_drawdown"] - signal["max_drawdown"]
    signal["drawdown_reduction_vs_random"] = signal["random_max_drawdown"] - signal["max_drawdown"]
    control_metric_cols = [f"{suffix}_{metric}" for _, suffix in controls for metric in metrics]
    delta_cols = [column for column in control_metric_cols if column in signal.columns]
    delta_cols.extend(
        [
            column
            for column in signal.columns
            if column.startswith(("net_delta_", "daily_sharpe_delta_", "drawdown_reduction_"))
        ]
    )
    return rows.merge(signal.loc[:, [*keys, *delta_cols]], on=keys, how="left", validate="many_to_one")


def classify_validation_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _search_cfg(config)
    if row["bucket"] != "alpha_signal":
        return "control"
    if int(row["trades"]) < int(cfg.get("min_trades", 35)):
        return "rejected_insufficient_trades"
    if float(row["turnover"]) > float(cfg.get("max_turnover", 2.5)):
        return "rejected_high_turnover"
    if float(row["net_return"]) <= 0.0 or float(row["avg_trade_net"]) < float(cfg.get("min_avg_trade_net", 0.0)):
        return "rejected_negative_edge"
    if float(row["profit_factor"]) < float(cfg.get("min_profit_factor", 1.05)):
        return "rejected_weak_profit_factor"
    if float(row["daily_sharpe"]) < float(cfg.get("min_daily_sharpe", 0.3)):
        return "rejected_weak_sharpe"
    if float(row["max_drawdown"]) > float(cfg.get("max_drawdown", 0.20)):
        return "rejected_drawdown"
    if float(row["top_day_abs_net_share"]) > float(cfg.get("max_top_day_abs_net_share", 0.35)):
        return "rejected_concentrated"
    if bool(cfg.get("require_random_improvement", True)) and float(row.get("net_delta_vs_random", -np.inf)) <= 0.0:
        return "rejected_no_random_edge"
    if bool(cfg.get("require_inverted_improvement", True)) and float(row.get("net_delta_vs_inverted", -np.inf)) <= 0.0:
        return "rejected_inverted_not_worse"
    if bool(cfg.get("require_breakout_improvement", True)):
        if float(row.get("net_delta_vs_breakout_only", -np.inf)) <= 0.0 and float(row.get("drawdown_reduction_vs_breakout_only", -np.inf)) <= 0.0:
            return "rejected_no_breakout_improvement"
    if bool(cfg.get("require_compression_improvement", False)):
        if float(row.get("net_delta_vs_compression_only", -np.inf)) <= 0.0 and float(row.get("drawdown_reduction_vs_compression_only", -np.inf)) <= 0.0:
            return "rejected_no_compression_improvement"
    return "volatility_expansion_validation_candidate"


def add_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["candidate_status"] = rows.apply(lambda row: classify_validation_row(row, config), axis=1)
    return rows


def validation_grid(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _search_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    scenarios = _cost_scenarios(config, [primary])
    rows: list[pd.DataFrame] = []
    validation = dataset[dataset["split"].eq(str(cfg.get("candidate_split", "validation")))]
    for (fold, horizon), validation_frame in validation.groupby(["fold", "horizon_bars"], sort=False):
        train_frame = dataset[
            dataset["split"].eq("train") & dataset["fold"].eq(int(fold)) & dataset["horizon_bars"].eq(int(horizon))
        ].copy()
        if train_frame.empty or validation_frame.empty:
            continue
        specs = generate_candidate_specs(train_frame, int(fold), int(horizon), config)
        for _, spec in specs.iterrows():
            for scenario in scenarios:
                rows.append(evaluate_candidate(validation_frame.copy(), "validation", spec, scenario, config))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(result, config) if not result.empty else result


def select_specs(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = _search_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    candidates = validation[
        validation["bucket"].eq("alpha_signal")
        & validation["cost_scenario"].eq(primary)
        & validation["candidate_status"].eq("volatility_expansion_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("alpha_signal") & validation["cost_scenario"].eq(primary)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["utility_score"] = (
        candidates["daily_sharpe"].fillna(0.0)
        + 75.0 * candidates["avg_trade_net"].fillna(0.0)
        + 0.25 * candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        + candidates["net_delta_vs_breakout_only"].fillna(0.0)
        + candidates["net_delta_vs_random"].fillna(0.0)
        - 0.03 * candidates["turnover"].fillna(0.0)
    )
    candidates = candidates.sort_values(
        ["candidate_status", "utility_score", "net_return", "avg_trade_net"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    cols = [
        "candidate_id",
        "fold",
        "variant",
        "side",
        "horizon_bars",
        "compression_quantile",
        "rv_compression_quantile",
        "breakout_margin_quantile",
        "volume_quantile",
        "vol_filter_name",
        "hour_filter_name",
        "thresholds_json",
        "candidate_status",
        "utility_score",
    ]
    return candidates.drop_duplicates("candidate_id").loc[:, cols].head(int(cfg.get("max_selected", 120))).reset_index(drop=True)


def evaluate_selected_on_split(dataset: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    scenarios = _cost_scenarios(config)
    rows: list[pd.DataFrame] = []
    for _, spec in specs.iterrows():
        frame = dataset[
            dataset["split"].eq(split)
            & dataset["fold"].eq(int(spec["fold"]))
            & dataset["horizon_bars"].eq(int(spec["horizon_bars"]))
        ].copy()
        if frame.empty:
            continue
        for scenario in scenarios:
            rows.append(evaluate_candidate(frame, split, spec, scenario, config))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(result, config) if not result.empty else result


def _primary_row(frame: pd.DataFrame, candidate_id: str, cost_scenario: str) -> pd.Series:
    selected = frame[frame["candidate_id"].eq(candidate_id) & frame["bucket"].eq("alpha_signal") & frame["cost_scenario"].eq(cost_scenario)]
    return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)


def _row_passes(row: pd.Series, config: dict[str, Any], *, require_sharpe: bool) -> bool:
    cfg = _search_cfg(config)
    if row.empty:
        return False
    checks = [
        int(row["trades"]) >= int(cfg.get("min_trades", 35)),
        float(row["net_return"]) > 0.0,
        float(row["avg_trade_net"]) >= float(cfg.get("min_avg_trade_net", 0.0)),
        float(row["profit_factor"]) >= float(cfg.get("min_profit_factor", 1.05)),
        float(row["max_drawdown"]) <= float(cfg.get("max_drawdown", 0.20)),
        float(row["top_day_abs_net_share"]) <= float(cfg.get("max_top_day_abs_net_share", 0.35)),
        float(row["turnover"]) <= float(cfg.get("max_turnover", 2.5)),
        float(row.get("net_delta_vs_random", -np.inf)) > 0.0,
        float(row.get("net_delta_vs_inverted", -np.inf)) > 0.0,
    ]
    if require_sharpe:
        checks.append(float(row["daily_sharpe"]) >= float(cfg.get("min_daily_sharpe", 0.3)))
    return all(checks)


def decision_table(validation: pd.DataFrame, test: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _search_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        candidate_id = str(spec["candidate_id"])
        val_row = _primary_row(validation, candidate_id, primary)
        primary_row = _primary_row(test, candidate_id, primary)
        conservative_row = _primary_row(test, candidate_id, conservative)
        stress_row = _primary_row(test, candidate_id, stress)
        validation_ok = val_row.get("candidate_status", "") == "volatility_expansion_validation_candidate"
        primary_ok = _row_passes(primary_row, config, require_sharpe=True)
        conservative_ok = _row_passes(conservative_row, config, require_sharpe=False)
        stress_ok = _row_passes(stress_row, config, require_sharpe=False)
        if not validation_ok:
            decision = "rejected_validation_failed"
        elif primary_ok and conservative_ok and stress_ok:
            decision = "accepted_candidate"
        elif primary_ok and conservative_ok:
            decision = "cost_fragile"
        elif primary_ok:
            decision = "cost_fragile"
        elif not primary_row.empty and primary_row.get("net_return", -np.inf) > 0.0:
            decision = "research_candidate"
        else:
            decision = "rejected"
        rows.append(
            {
                **{
                    col: spec.get(col, "all" if col == "hour_filter_name" else np.nan)
                    for col in ["candidate_id", "fold", "variant", "side", "horizon_bars", "vol_filter_name", "hour_filter_name"]
                },
                "validation_status": val_row.get("candidate_status", ""),
                "decision": decision,
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_sharpe_primary": primary_row.get("daily_sharpe", np.nan),
                "test_profit_factor_primary": primary_row.get("profit_factor", np.nan),
                "test_avg_trade_net_primary": primary_row.get("avg_trade_net", np.nan),
                "test_trades_primary": primary_row.get("trades", np.nan),
                "test_turnover_primary": primary_row.get("turnover", np.nan),
                "test_top_day_abs_net_share_primary": primary_row.get("top_day_abs_net_share", np.nan),
                "test_net_delta_vs_breakout_primary": primary_row.get("net_delta_vs_breakout_only", np.nan),
                "test_net_delta_vs_compression_primary": primary_row.get("net_delta_vs_compression_only", np.nan),
                "test_net_delta_vs_random_primary": primary_row.get("net_delta_vs_random", np.nan),
                "test_net_delta_vs_inverted_primary": primary_row.get("net_delta_vs_inverted", np.nan),
                "test_net_conservative": conservative_row.get("net_return", np.nan),
                "test_net_stress": stress_row.get("net_return", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "test_sharpe_primary", "test_net_primary"], ascending=[True, False, False], kind="stable")


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    specs: pd.DataFrame,
    decisions: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _search_cfg(config)
    val_alpha = validation[validation["bucket"].eq("alpha_signal")] if not validation.empty else pd.DataFrame()
    test_alpha = test[test["bucket"].eq("alpha_signal")] if not test.empty else pd.DataFrame()
    val_counts = val_alpha["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="rows") if not val_alpha.empty else pd.DataFrame()
    decision_counts = decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    family_summary = (
        test_alpha.groupby(["cost_scenario", "variant", "side", "horizon_bars", "hour_filter_name"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            median_net_return=("net_return", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0.0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_trades=("trades", "median"),
            median_net_delta_vs_breakout=("net_delta_vs_breakout_only", "median"),
            median_net_delta_vs_random=("net_delta_vs_random", "median"),
        )
        .sort_values(["cost_scenario", "median_net_return"], ascending=[True, False], kind="stable")
        if not test_alpha.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one volatility-expansion candidate passes the configured IBKR-aware gates."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No volatility-expansion candidate is accepted under the configured IBKR-aware gates."
    )
    return f"""# Volatility Expansion Search - {target_symbol.upper()}

## Scope

- HMM states are not used.
- Thresholds are computed from train only for each fold and horizon.
- Compression uses prior bar values shifted within session.
- Signal source: compression, current breakout, close-location confirmation, optional volume/cross-asset confirmation.
- Horizons: `{cfg.get("horizons", [2, 3, 4])}`
- Max trades per day: `{cfg.get("max_trades_per_day", 2)}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`

## Validation Status Counts

{_markdown_table(val_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Family Test Summary

{_markdown_table(family_summary, max_rows=int(cfg.get("report_top_rows", 100)))}

## Top Decisions

{_markdown_table(decisions.head(int(cfg.get("report_top_rows", 100))) if not decisions.empty else decisions, max_rows=int(cfg.get("report_top_rows", 100)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    dataset = build_split_dataset(config, target)
    validation = validation_grid(dataset, config) if not dataset.empty else pd.DataFrame()
    specs = select_specs(validation, config)
    selected_validation = validation[validation["candidate_id"].isin(specs["candidate_id"])] if not specs.empty else pd.DataFrame()
    test = evaluate_selected_on_split(dataset, specs, str(_search_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()
    decisions = decision_table(selected_validation, test, specs, config)
    outputs = {
        "volatility_expansion_validation": results_dir / "volatility_expansion_validation.parquet",
        "volatility_expansion_selected_validation": results_dir / "volatility_expansion_selected_validation.parquet",
        "volatility_expansion_test": results_dir / "volatility_expansion_test.parquet",
        "volatility_expansion_selected_specs": results_dir / "volatility_expansion_selected_specs.parquet",
        "volatility_expansion_decisions": results_dir / "volatility_expansion_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["volatility_expansion_validation"], index=False)
    selected_validation.to_parquet(outputs["volatility_expansion_selected_validation"], index=False)
    test.to_parquet(outputs["volatility_expansion_test"], index=False)
    specs.to_parquet(outputs["volatility_expansion_selected_specs"], index=False)
    decisions.to_parquet(outputs["volatility_expansion_decisions"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, validation, test, specs, decisions, outputs), encoding="utf-8")
    return report_path, outputs["volatility_expansion_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search train-calibrated volatility compression-expansion candidates.")
    parser.add_argument("--config", default="configs/hmm_lab_15min_expansion.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Volatility expansion report written to: {report_path}")
    print(f"Volatility expansion decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

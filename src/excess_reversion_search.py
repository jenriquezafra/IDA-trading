from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, evaluate_position_with_cost
from src.hmm_lab import _lab_cfg, _target_symbol, build_lab_folds, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table


VARIANT_FLAGS: dict[str, dict[str, bool]] = {
    "extension_only": {"exhaustion": False, "divergence": False, "risk_filter": False},
    "extension_exhaustion": {"exhaustion": True, "divergence": False, "risk_filter": False},
    "extension_divergence": {"exhaustion": False, "divergence": True, "risk_filter": False},
    "extension_exhaustion_divergence": {"exhaustion": True, "divergence": True, "risk_filter": False},
    "extension_exhaustion_risk_filter": {"exhaustion": True, "divergence": False, "risk_filter": True},
    "extension_exhaustion_divergence_risk_filter": {"exhaustion": True, "divergence": True, "risk_filter": True},
}

REQUIRED_FEATURE_COLUMNS = sorted(
    {
        "target_open_next",
        "target_dist_vwap_atr",
        "target_dist_open",
        "target_range_ratio_2_8",
        "target_rel_volume_accel_2",
        "target_close_location_bar",
        "target_minutes_from_open",
        "target_minutes_to_close",
        "target_rv_4_rel_by_bar",
        "positive_index_count_2",
        "positive_sector_count_2",
        "sector_above_vwap_count",
        "index_above_vwap_count",
        "spread_credit_12",
        "spread_equity_bonds_12",
        "cross_asset_vol_expansion_score",
        "intraday_stress_score",
        "risk_on_score",
        "risk_off_score",
        "target_ret_3",
    }
)


def _search_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("excess_reversion_search", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "excess_reversion_search.md"


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


def _quantile(frame: pd.DataFrame, column: str, q: float, default: float = np.nan) -> float:
    values = _clean_series(frame, column).dropna()
    if values.empty:
        return float(default)
    return float(values.quantile(float(q)))


def _abs_quantile(frame: pd.DataFrame, column: str, q: float, default: float = np.nan) -> float:
    values = _clean_series(frame, column).dropna().abs()
    values = values[values > 0.0]
    if values.empty:
        return float(default)
    return float(values.quantile(float(q)))


def _cost_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = names if names is not None else [str(value) for value in _search_cfg(config).get("cost_scenarios", ["ibkr_tiered_10000", "bps_2", "bps_5"])]
    by_name = {str(scenario["cost_scenario"]): scenario for scenario in cost_scenarios(config)}
    return [by_name[name] for name in wanted if name in by_name]


def _candidate_id(row: pd.Series | dict[str, Any]) -> str:
    parts = [
        f"fold{int(row['fold'])}",
        str(row["variant"]),
        str(row["side"]),
        f"h{int(row['horizon_bars'])}",
        f"vq{float(row['vwap_quantile']):g}",
        f"oq{float(row['open_quantile']):g}",
        f"rq{float(row['range_quantile']):g}",
        f"volq{float(row['volume_accel_quantile']):g}",
        f"cl{float(row['close_location_short_max']):g}-{float(row['close_location_long_min']):g}",
        str(row["risk_filter_name"]),
    ]
    return "__".join(parts)


def _stable_seed(candidate_id: str, split: str, bucket: str) -> int:
    digest = hashlib.sha256(f"{candidate_id}::{split}::{bucket}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def load_feature_frame(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    path = features_input_path(config, target_symbol, feature_config)
    features = pd.read_parquet(path)
    return features.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)


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
    output["feature_set"] = "excess_reversion"
    output["n_states"] = 0
    output["seed"] = 0
    return output


def train_thresholds(train_frame: pd.DataFrame, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    cfg = _search_cfg(config)
    breadth_low_q = float(cfg.get("breadth_low_quantile", 0.35))
    breadth_high_q = float(cfg.get("breadth_high_quantile", 0.65))
    credit_low_q = float(cfg.get("credit_low_quantile", 0.35))
    credit_high_q = float(cfg.get("credit_high_quantile", 0.65))
    risk_high_q = float(cfg.get("risk_high_quantile", 0.70))
    vol_high_q = float(cfg.get("vol_high_quantile", 0.70))

    thresholds = {
        "vwap_abs_min": _abs_quantile(train_frame, "target_dist_vwap_atr", float(spec["vwap_quantile"])),
        "open_abs_min": _abs_quantile(train_frame, "target_dist_open", float(spec["open_quantile"])),
        "range_ratio_min": _quantile(train_frame, "target_range_ratio_2_8", float(spec["range_quantile"]), -np.inf),
        "volume_accel_max": _quantile(train_frame, "target_rel_volume_accel_2", float(spec["volume_accel_quantile"]), np.inf),
        "positive_index_low": _quantile(train_frame, "positive_index_count_2", breadth_low_q, -np.inf),
        "positive_index_high": _quantile(train_frame, "positive_index_count_2", breadth_high_q, np.inf),
        "positive_sector_low": _quantile(train_frame, "positive_sector_count_2", breadth_low_q, -np.inf),
        "positive_sector_high": _quantile(train_frame, "positive_sector_count_2", breadth_high_q, np.inf),
        "index_above_vwap_low": _quantile(train_frame, "index_above_vwap_count", breadth_low_q, -np.inf),
        "index_above_vwap_high": _quantile(train_frame, "index_above_vwap_count", breadth_high_q, np.inf),
        "sector_above_vwap_low": _quantile(train_frame, "sector_above_vwap_count", breadth_low_q, -np.inf),
        "sector_above_vwap_high": _quantile(train_frame, "sector_above_vwap_count", breadth_high_q, np.inf),
        "spread_credit_low": _quantile(train_frame, "spread_credit_12", credit_low_q, -np.inf),
        "spread_credit_high": _quantile(train_frame, "spread_credit_12", credit_high_q, np.inf),
        "spread_equity_bonds_low": _quantile(train_frame, "spread_equity_bonds_12", credit_low_q, -np.inf),
        "risk_on_high": _quantile(train_frame, "risk_on_score", risk_high_q, np.inf),
        "risk_off_high": _quantile(train_frame, "risk_off_score", risk_high_q, np.inf),
        "stress_high": _quantile(train_frame, "intraday_stress_score", risk_high_q, np.inf),
        "vol_expansion_high": _quantile(train_frame, "cross_asset_vol_expansion_score", vol_high_q, np.inf),
        "target_rv_rel_high": _quantile(train_frame, "target_rv_4_rel_by_bar", vol_high_q, np.inf),
    }
    if not np.isfinite(thresholds["vwap_abs_min"]) or not np.isfinite(thresholds["open_abs_min"]):
        return {}
    thresholds["min_minutes_from_open"] = float(spec.get("min_minutes_from_open", cfg.get("min_minutes_from_open", 45)))
    thresholds["min_minutes_to_close"] = float(spec.get("min_minutes_to_close", cfg.get("min_minutes_to_close", 30)))
    thresholds["close_location_short_max"] = float(spec["close_location_short_max"])
    thresholds["close_location_long_min"] = float(spec["close_location_long_min"])
    return thresholds


def _time_mask(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    from_open = _clean_series(frame, "target_minutes_from_open", np.inf)
    to_close = _clean_series(frame, "target_minutes_to_close", np.inf)
    return from_open.ge(float(thresholds.get("min_minutes_from_open", 0.0))) & to_close.ge(float(thresholds.get("min_minutes_to_close", 0.0)))


def extension_position(frame: pd.DataFrame, spec: dict[str, Any], thresholds: dict[str, Any]) -> pd.Series:
    vwap = _clean_series(frame, "target_dist_vwap_atr")
    dist_open = _clean_series(frame, "target_dist_open")
    range_ratio = _clean_series(frame, "target_range_ratio_2_8", np.inf)
    range_ok = range_ratio.ge(float(thresholds.get("range_ratio_min", -np.inf))) if bool(spec.get("use_range_filter", True)) else _bool_series(frame, True)
    active = _time_mask(frame, thresholds) & range_ok

    short_mask = active & vwap.ge(float(thresholds["vwap_abs_min"])) & dist_open.ge(float(thresholds["open_abs_min"]))
    long_mask = active & vwap.le(-float(thresholds["vwap_abs_min"])) & dist_open.le(-float(thresholds["open_abs_min"]))
    side = str(spec.get("side", "both"))
    if side == "long":
        short_mask = _bool_series(frame, False)
    elif side == "short":
        long_mask = _bool_series(frame, False)
    elif side != "both":
        raise ValueError(f"Unsupported side: {side}")

    position = pd.Series(0.0, index=frame.index)
    position.loc[short_mask] = -1.0
    position.loc[long_mask] = 1.0
    return position


def _exhaustion_mask(frame: pd.DataFrame, position: pd.Series, thresholds: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    cfg = _search_cfg(config)
    close_location = _clean_series(frame, "target_close_location_bar")
    volume_accel = _clean_series(frame, "target_rel_volume_accel_2")
    short_close_ok = close_location.le(float(thresholds["close_location_short_max"]))
    long_close_ok = close_location.ge(float(thresholds["close_location_long_min"]))
    volume_ok = volume_accel.le(float(thresholds.get("volume_accel_max", np.inf)))
    mode = str(cfg.get("exhaustion_mode", "any"))
    if mode == "all":
        short_ok = short_close_ok & volume_ok
        long_ok = long_close_ok & volume_ok
    elif mode == "any":
        short_ok = short_close_ok | volume_ok
        long_ok = long_close_ok | volume_ok
    else:
        raise ValueError(f"Unsupported exhaustion_mode: {mode}")
    return ((position < 0.0) & short_ok) | ((position > 0.0) & long_ok)


def _divergence_mask(frame: pd.DataFrame, position: pd.Series, thresholds: dict[str, Any]) -> pd.Series:
    positive_index = _clean_series(frame, "positive_index_count_2")
    positive_sector = _clean_series(frame, "positive_sector_count_2")
    index_above_vwap = _clean_series(frame, "index_above_vwap_count")
    sector_above_vwap = _clean_series(frame, "sector_above_vwap_count")
    credit = _clean_series(frame, "spread_credit_12")
    risk_off = _clean_series(frame, "risk_off_score")

    upside_not_confirmed = (
        positive_index.le(float(thresholds.get("positive_index_high", np.inf)))
        | positive_sector.le(float(thresholds.get("positive_sector_high", np.inf)))
        | index_above_vwap.le(float(thresholds.get("index_above_vwap_high", np.inf)))
        | sector_above_vwap.le(float(thresholds.get("sector_above_vwap_high", np.inf)))
        | credit.le(float(thresholds.get("spread_credit_high", np.inf)))
    )
    downside_not_confirmed = (
        positive_index.ge(float(thresholds.get("positive_index_low", -np.inf)))
        | positive_sector.ge(float(thresholds.get("positive_sector_low", -np.inf)))
        | index_above_vwap.ge(float(thresholds.get("index_above_vwap_low", -np.inf)))
        | sector_above_vwap.ge(float(thresholds.get("sector_above_vwap_low", -np.inf)))
        | credit.ge(float(thresholds.get("spread_credit_low", -np.inf)))
        | risk_off.le(float(thresholds.get("risk_off_high", np.inf)))
    )
    return ((position < 0.0) & upside_not_confirmed) | ((position > 0.0) & downside_not_confirmed)


def risk_off_mask(frame: pd.DataFrame, thresholds: dict[str, Any], risk_filter_name: str) -> pd.Series:
    positive_index = _clean_series(frame, "positive_index_count_2")
    positive_sector = _clean_series(frame, "positive_sector_count_2")
    index_above_vwap = _clean_series(frame, "index_above_vwap_count")
    sector_above_vwap = _clean_series(frame, "sector_above_vwap_count")
    credit = _clean_series(frame, "spread_credit_12")
    equity_bonds = _clean_series(frame, "spread_equity_bonds_12")
    risk_off = _clean_series(frame, "risk_off_score")
    stress = _clean_series(frame, "intraday_stress_score")
    vol_expansion = _clean_series(frame, "cross_asset_vol_expansion_score")
    target_rv = _clean_series(frame, "target_rv_4_rel_by_bar")

    breadth_weak = (
        positive_index.le(float(thresholds.get("positive_index_low", -np.inf)))
        & positive_sector.le(float(thresholds.get("positive_sector_low", -np.inf)))
    ) | (
        index_above_vwap.le(float(thresholds.get("index_above_vwap_low", -np.inf)))
        & sector_above_vwap.le(float(thresholds.get("sector_above_vwap_low", -np.inf)))
    )
    credit_rates_stress = (
        credit.le(float(thresholds.get("spread_credit_low", -np.inf)))
        | equity_bonds.le(float(thresholds.get("spread_equity_bonds_low", -np.inf)))
        | risk_off.ge(float(thresholds.get("risk_off_high", np.inf)))
    )
    volatility_expansion = (
        stress.ge(float(thresholds.get("stress_high", np.inf)))
        | vol_expansion.ge(float(thresholds.get("vol_expansion_high", np.inf)))
        | target_rv.ge(float(thresholds.get("target_rv_rel_high", np.inf)))
    )
    if risk_filter_name == "none":
        return _bool_series(frame, False)
    if risk_filter_name == "risk_off_1":
        return breadth_weak
    if risk_filter_name == "risk_off_2":
        return breadth_weak & credit_rates_stress
    if risk_filter_name == "risk_off_3":
        return breadth_weak & credit_rates_stress & volatility_expansion
    raise ValueError(f"Unsupported risk filter: {risk_filter_name}")


def _risk_on_extreme_mask(frame: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    return (
        _clean_series(frame, "risk_on_score").ge(float(thresholds.get("risk_on_high", np.inf)))
        & _clean_series(frame, "positive_index_count_2").ge(float(thresholds.get("positive_index_high", np.inf)))
        & _clean_series(frame, "positive_sector_count_2").ge(float(thresholds.get("positive_sector_high", np.inf)))
    )


def _risk_filter_mask(frame: pd.DataFrame, position: pd.Series, thresholds: dict[str, Any], risk_filter_name: str) -> pd.Series:
    blocked_long = risk_off_mask(frame, thresholds, risk_filter_name)
    blocked_short = _risk_on_extreme_mask(frame, thresholds)
    stress_block = _clean_series(frame, "intraday_stress_score").ge(float(thresholds.get("stress_high", np.inf)))
    blocked = ((position > 0.0) & (blocked_long | stress_block)) | ((position < 0.0) & (blocked_short | stress_block))
    return ~blocked


def cap_trades_per_day(position: pd.Series, frame: pd.DataFrame, max_trades_per_day: int) -> pd.Series:
    capped = pd.Series(0.0, index=position.index)
    active = position.abs() > 0.0
    if max_trades_per_day <= 0 or not active.any():
        return capped
    order = frame.loc[active, ["session", "bar_index"]].copy()
    order["_idx"] = order.index
    order = order.sort_values(["session", "bar_index"], kind="stable")
    keep = order.groupby("session", sort=False).head(int(max_trades_per_day))["_idx"]
    capped.loc[keep] = position.loc[keep]
    return capped


def excess_reversion_position(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    thresholds: dict[str, Any],
    config: dict[str, Any],
    *,
    use_exhaustion: bool | None = None,
    use_divergence: bool | None = None,
    use_risk_filter: bool | None = None,
) -> pd.Series:
    flags = VARIANT_FLAGS[str(spec["variant"])]
    use_exhaustion = flags["exhaustion"] if use_exhaustion is None else bool(use_exhaustion)
    use_divergence = flags["divergence"] if use_divergence is None else bool(use_divergence)
    use_risk_filter = flags["risk_filter"] if use_risk_filter is None else bool(use_risk_filter)

    position = extension_position(frame, spec, thresholds)
    active = position.abs() > 0.0
    if use_exhaustion:
        active &= _exhaustion_mask(frame, position, thresholds, config)
    if use_divergence:
        active &= _divergence_mask(frame, position, thresholds)
    if use_risk_filter:
        active &= _risk_filter_mask(frame, position, thresholds, str(spec.get("risk_filter_name", "none")))
    position = position.where(active, 0.0)
    return cap_trades_per_day(position, frame, int(spec.get("max_trades_per_day", _search_cfg(config).get("max_trades_per_day", 2))))


def same_hour_random_position(frame: pd.DataFrame, template_position: pd.Series, seed: int) -> pd.Series:
    rng = np.random.default_rng(int(seed))
    random_position = pd.Series(0.0, index=frame.index)
    used: set[Any] = set()
    active = template_position[template_position.abs() > 0.0]
    if active.empty:
        return random_position
    active_frame = frame.loc[active.index, ["hour"]].copy()
    active_frame["direction"] = np.sign(active).astype(int)
    for (hour, direction), group in active_frame.groupby(["hour", "direction"], sort=False):
        eligible = frame.index[frame["hour"].eq(hour)]
        eligible = [idx for idx in eligible if idx not in used]
        if not eligible:
            continue
        count = min(len(group), len(eligible))
        chosen = rng.choice(np.array(eligible, dtype=object), size=count, replace=False)
        for idx in chosen.tolist():
            used.add(idx)
            random_position.loc[idx] = float(direction)
    return random_position


def _control_positions(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    thresholds: dict[str, Any],
    config: dict[str, Any],
    candidate_id: str,
    split: str,
) -> dict[str, pd.Series]:
    alpha = excess_reversion_position(frame, spec, thresholds, config)
    extension_only = cap_trades_per_day(
        extension_position(frame, spec, thresholds),
        frame,
        int(spec.get("max_trades_per_day", _search_cfg(config).get("max_trades_per_day", 2))),
    )
    target_only = excess_reversion_position(
        frame,
        spec,
        thresholds,
        config,
        use_exhaustion=VARIANT_FLAGS[str(spec["variant"])]["exhaustion"],
        use_divergence=False,
        use_risk_filter=False,
    )
    random_control = same_hour_random_position(frame, alpha, _stable_seed(candidate_id, split, "same_hour_random_control"))
    return {
        "alpha_signal": alpha,
        "extension_only_control": extension_only,
        "target_only_control": target_only,
        "same_hour_random_control": random_control,
        "inverted_signal": -alpha,
        "always_flat": pd.Series(0.0, index=frame.index),
    }


def generate_candidate_specs(train_frame: pd.DataFrame, fold: int, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _search_cfg(config)
    variants = [str(value) for value in cfg.get("variants", list(VARIANT_FLAGS))]
    sides = [str(value) for value in cfg.get("sides", ["both", "long", "short"])]
    vwap_quantiles = [float(value) for value in cfg.get("vwap_quantiles", [0.80, 0.90])]
    open_quantiles = [float(value) for value in cfg.get("open_quantiles", [0.75, 0.85])]
    range_quantiles = [float(value) for value in cfg.get("range_quantiles", [0.65, 0.75])]
    volume_quantiles = [float(value) for value in cfg.get("volume_accel_quantiles", [0.50, 0.67])]
    short_close = [float(value) for value in cfg.get("close_location_short_max", [0.55, 0.65])]
    long_close = [float(value) for value in cfg.get("close_location_long_min", [0.35, 0.45])]
    risk_filters = [str(value) for value in cfg.get("risk_filters", ["risk_off_1", "risk_off_2", "risk_off_3"])]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant not in VARIANT_FLAGS:
            continue
        filter_names = risk_filters if VARIANT_FLAGS[variant]["risk_filter"] else ["none"]
        for side in sides:
            for vwap_q in vwap_quantiles:
                for open_q in open_quantiles:
                    for range_q in range_quantiles:
                        for volume_q in volume_quantiles:
                            for short_cl in short_close:
                                for long_cl in long_close:
                                    for filter_name in filter_names:
                                        spec = {
                                            "fold": int(fold),
                                            "variant": variant,
                                            "side": side,
                                            "horizon_bars": int(horizon),
                                            "vwap_quantile": float(vwap_q),
                                            "open_quantile": float(open_q),
                                            "range_quantile": float(range_q),
                                            "volume_accel_quantile": float(volume_q),
                                            "close_location_short_max": float(short_cl),
                                            "close_location_long_min": float(long_cl),
                                            "risk_filter_name": filter_name,
                                            "use_range_filter": bool(cfg.get("use_range_filter", True)),
                                            "min_minutes_from_open": float(cfg.get("min_minutes_from_open", 45)),
                                            "min_minutes_to_close": float(cfg.get("min_minutes_to_close", 30)),
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
            **{key: spec_dict[key] for key in ["fold", "variant", "side", "horizon_bars", "vwap_quantile", "open_quantile", "range_quantile", "volume_accel_quantile", "close_location_short_max", "close_location_long_min", "risk_filter_name"]},
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
        ("extension_only_control", "extension"),
        ("target_only_control", "target_only"),
        ("same_hour_random_control", "random"),
        ("inverted_signal", "inverted"),
    ]
    for control_bucket, suffix in controls:
        control = rows[rows["bucket"].eq(control_bucket)].loc[:, [*keys, *metrics]].rename(
            columns={metric: f"{suffix}_{metric}" for metric in metrics}
        )
        signal = signal.merge(control, on=keys, how="left", validate="one_to_one")
    signal["net_delta_vs_extension_only"] = signal["net_return"] - signal["extension_net_return"]
    signal["net_delta_vs_target_only"] = signal["net_return"] - signal["target_only_net_return"]
    signal["net_delta_vs_random"] = signal["net_return"] - signal["random_net_return"]
    signal["net_delta_vs_inverted"] = signal["net_return"] - signal["inverted_net_return"]
    signal["daily_sharpe_delta_vs_extension_only"] = signal["daily_sharpe"] - signal["extension_daily_sharpe"]
    signal["daily_sharpe_delta_vs_random"] = signal["daily_sharpe"] - signal["random_daily_sharpe"]
    signal["drawdown_reduction_vs_extension_only"] = signal["extension_max_drawdown"] - signal["max_drawdown"]
    signal["drawdown_reduction_vs_random"] = signal["random_max_drawdown"] - signal["max_drawdown"]
    delta_cols = [
        column
        for column in signal.columns
        if column.startswith(("extension_", "target_only_", "random_", "inverted_", "net_delta_", "daily_sharpe_delta_", "drawdown_reduction_"))
    ]
    return rows.merge(signal.loc[:, [*keys, *delta_cols]], on=keys, how="left", validate="many_to_one")


def classify_validation_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _search_cfg(config)
    if row["bucket"] != "alpha_signal":
        return "control"
    if int(row["trades"]) < int(cfg.get("min_trades", 40)):
        return "rejected_insufficient_trades"
    if float(row["turnover"]) > float(cfg.get("max_turnover", 2.5)):
        return "rejected_high_turnover"
    if float(row["net_return"]) <= 0.0 or float(row["avg_trade_net"]) <= 0.0:
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
    if row["variant"] != "extension_only" and bool(cfg.get("require_extension_improvement", True)):
        if float(row.get("net_delta_vs_extension_only", -np.inf)) <= 0.0 and float(row.get("drawdown_reduction_vs_extension_only", -np.inf)) <= 0.0:
            return "rejected_no_extension_improvement"
    return "excess_reversion_validation_candidate"


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
    for (fold, horizon), validation_frame in dataset[dataset["split"].eq(str(cfg.get("candidate_split", "validation")))].groupby(
        ["fold", "horizon_bars"],
        sort=False,
    ):
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
        & validation["candidate_status"].eq("excess_reversion_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("alpha_signal") & validation["cost_scenario"].eq(primary)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["utility_score"] = (
        candidates["daily_sharpe"].fillna(0.0)
        + 75.0 * candidates["avg_trade_net"].fillna(0.0)
        + 0.25 * candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        + candidates["net_delta_vs_random"].fillna(0.0)
        + candidates["net_delta_vs_extension_only"].fillna(0.0)
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
        "vwap_quantile",
        "open_quantile",
        "range_quantile",
        "volume_accel_quantile",
        "close_location_short_max",
        "close_location_long_min",
        "risk_filter_name",
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
        int(row["trades"]) >= int(cfg.get("min_trades", 40)),
        float(row["net_return"]) > 0.0,
        float(row["avg_trade_net"]) > 0.0,
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
        primary_ok = _row_passes(primary_row, config, require_sharpe=True)
        conservative_ok = _row_passes(conservative_row, config, require_sharpe=False)
        if primary_ok and conservative_ok:
            decision = "accepted_candidate"
        elif primary_ok:
            decision = "cost_fragile"
        elif not primary_row.empty and primary_row.get("net_return", -np.inf) > 0.0:
            decision = "research_candidate"
        else:
            decision = "rejected"
        rows.append(
            {
                **{col: spec[col] for col in ["candidate_id", "fold", "variant", "side", "horizon_bars", "risk_filter_name"]},
                "validation_status": val_row.get("candidate_status", ""),
                "decision": decision,
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_sharpe_primary": primary_row.get("daily_sharpe", np.nan),
                "test_profit_factor_primary": primary_row.get("profit_factor", np.nan),
                "test_avg_trade_net_primary": primary_row.get("avg_trade_net", np.nan),
                "test_trades_primary": primary_row.get("trades", np.nan),
                "test_turnover_primary": primary_row.get("turnover", np.nan),
                "test_top_day_abs_net_share_primary": primary_row.get("top_day_abs_net_share", np.nan),
                "test_net_delta_vs_extension_primary": primary_row.get("net_delta_vs_extension_only", np.nan),
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
        test_alpha.groupby(["cost_scenario", "variant", "side", "horizon_bars"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            median_net_return=("net_return", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0.0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_trades=("trades", "median"),
            median_net_delta_vs_extension=("net_delta_vs_extension_only", "median"),
            median_net_delta_vs_random=("net_delta_vs_random", "median"),
        )
        .sort_values(["cost_scenario", "median_net_return"], ascending=[True, False], kind="stable")
        if not test_alpha.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one excess-reversion candidate passes the configured IBKR-aware gates."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No excess-reversion candidate is accepted under the configured IBKR-aware gates."
    )
    return f"""# Excess Reversion Search - {target_symbol.upper()}

## Scope

- HMM states are not used.
- Thresholds are computed from train only for each fold and horizon.
- Validation selects frozen rule specs; test only confirms.
- Primary timeframe: `{config.get("lab", {}).get("timeframe", config.get("project", {}).get("frequency", "15min"))}`
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
        "excess_reversion_validation": results_dir / "excess_reversion_validation.parquet",
        "excess_reversion_selected_validation": results_dir / "excess_reversion_selected_validation.parquet",
        "excess_reversion_test": results_dir / "excess_reversion_test.parquet",
        "excess_reversion_selected_specs": results_dir / "excess_reversion_selected_specs.parquet",
        "excess_reversion_decisions": results_dir / "excess_reversion_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["excess_reversion_validation"], index=False)
    selected_validation.to_parquet(outputs["excess_reversion_selected_validation"], index=False)
    test.to_parquet(outputs["excess_reversion_test"], index=False)
    specs.to_parquet(outputs["excess_reversion_selected_specs"], index=False)
    decisions.to_parquet(outputs["excess_reversion_decisions"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, validation, test, specs, decisions, outputs), encoding="utf-8")
    return report_path, outputs["excess_reversion_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search train-calibrated intraday excess mean-reversion candidates.")
    parser.add_argument("--config", default="configs/hmm_lab_15min_excess_reversion.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Excess reversion report written to: {report_path}")
    print(f"Excess reversion decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

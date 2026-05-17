from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import max_drawdown, scenario_cost_return
from src.hmm_lab import _lab_cfg, _target_symbol, build_lab_folds, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios


SIGNAL_COLUMNS = [
    "target_open_next",
    "target_overnight_ret",
    "target_abs_overnight_ret",
    "target_gap_fill_progress",
    "target_dist_open",
    "target_dist_vwap_atr",
    "target_range_ratio_6_24",
    "target_rv_12_rel_by_bar",
    "target_rel_volume_by_bar",
    "target_rel_cum_volume_by_bar",
    "target_rel_volume_accel_2",
    "target_close_location_bar",
    "target_bar_efficiency",
    "target_upper_wick_ratio",
    "target_lower_wick_ratio",
    "target_above_or_6_high",
    "target_below_or_6_low",
    "target_breakout_margin_or_6_high_atr",
    "target_breakdown_margin_or_6_low_atr",
    "target_above_or_6_high_persist_2",
    "target_below_or_6_low_persist_2",
    "target_breakout_attempt_count_or_6_high",
    "target_breakout_attempt_count_or_6_low",
    "target_failed_breakout_high_12",
    "target_failed_breakout_low_12",
    "target_breaks_roll_high_12",
    "target_breaks_roll_low_12",
    "target_first_60m",
    "target_lunch",
    "target_last_60m",
    "target_minutes_from_open",
    "target_minutes_to_close",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_6",
    "positive_sector_count_6",
    "positive_index_count_open",
    "positive_sector_count_open",
    "index_above_vwap_count",
    "sector_above_vwap_count",
    "relopen_QQQ_SPY",
    "relopen_IWM_SPY",
    "risk_on_open_confirmation",
]

DEFAULT_SIGNAL_COLUMN_MAP = {
    "dist_vwap": "target_dist_vwap_atr",
    "dist_open": "target_dist_open",
    "minutes_from_open": "target_minutes_from_open",
    "range_ratio": "target_range_ratio_6_24",
    "rv_rel": "target_rv_12_rel_by_bar",
    "opening_high": "target_above_or_6_high",
    "opening_low": "target_below_or_6_low",
    "breakout_margin_high": "target_breakout_margin_or_6_high_atr",
    "breakout_margin_low": "target_breakdown_margin_or_6_low_atr",
    "opening_high_persistence": "target_above_or_6_high_persist_2",
    "opening_low_persistence": "target_below_or_6_low_persist_2",
    "opening_high_attempts": "target_breakout_attempt_count_or_6_high",
    "opening_low_attempts": "target_breakout_attempt_count_or_6_low",
    "rel_volume_accel": "target_rel_volume_accel_2",
    "failed_breakout_high": "target_failed_breakout_high_12",
    "failed_breakout_low": "target_failed_breakout_low_12",
    "breaks_roll_high": "target_breaks_roll_high_12",
    "breaks_roll_low": "target_breaks_roll_low_12",
    "positive_index_count": "positive_index_count_6",
    "positive_sector_count": "positive_sector_count_6",
}


def _search_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_search", {})


def _signal_column_map(config: dict[str, Any] | None = None, override: dict[str, str] | None = None) -> dict[str, str]:
    mapping = dict(DEFAULT_SIGNAL_COLUMN_MAP)
    if config is not None:
        mapping.update({str(key): str(value) for key, value in _search_cfg(config).get("column_map", {}).items()})
    if override:
        mapping.update({str(key): str(value) for key, value in override.items()})
    return mapping


def _mapped_column(column_map: dict[str, str], key: str) -> str:
    return column_map.get(key, DEFAULT_SIGNAL_COLUMN_MAP[key])


def _configured_signal_columns(config: dict[str, Any]) -> list[str]:
    configured = _search_cfg(config).get("signal_columns")
    base = [str(value) for value in configured] if configured else list(SIGNAL_COLUMNS)
    mapped = list(_signal_column_map(config).values())
    seen: set[str] = set()
    output: list[str] = []
    for column in [*base, *mapped]:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def _combined_cfg(config: dict[str, Any]) -> dict[str, Any]:
    combined = dict(config.get("operable_candidate_search", {}))
    combined.update(_search_cfg(config))
    return combined


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "setup_signal_search.md"


def _json_dumps(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True)


def _json_loads(value: str | float | int | None) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    return json.loads(str(value))


def _clean(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _quantile(frame: pd.DataFrame, column: str, q: float, default: float) -> float:
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna().astype(float) if column in frame else pd.Series(dtype=float)
    return float(values.quantile(float(q))) if not values.empty else float(default)


def _optional_quantile(frame: pd.DataFrame, column: str, q: float | None, default: float) -> float | None:
    return None if q is None else _quantile(frame, column, float(q), default)


def _add_optional_threshold(params: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        params[key] = value


def _candidate_id(row: dict[str, Any] | pd.Series) -> str:
    raw = f"{row['family']}|{row['direction']}|fold{int(row['fold'])}|h{int(row['horizon_bars'])}|{row['params_json']}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"fold{int(row['fold'])}__{row['family']}__{row['direction']}__h{int(row['horizon_bars'])}__{digest}"


def _feature_config(config: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))


def load_feature_frame(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = _feature_config(config)
    return pd.read_parquet(features_input_path(config, target_symbol, feature_config)).sort_values(["session", "bar_index"], kind="stable")


def build_signal_dataset(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    features = load_feature_frame(config, target_symbol)
    horizons = [int(value) for value in _search_cfg(config).get("horizons", [12, 24])]
    forward = build_forward_returns(features, horizons)
    if forward.empty:
        return pd.DataFrame()
    indexed = features.reset_index(names="source_index")
    available = ["source_index", *[column for column in _configured_signal_columns(config) if column in indexed.columns and column not in forward.columns]]
    merged = forward.merge(indexed.loc[:, available], on="source_index", how="left", validate="many_to_one")
    if "target_open_next" not in merged:
        merged["target_open_next"] = merged["entry_px"].astype(float)

    folds = build_lab_folds(features, config)
    parts = []
    for fold in folds:
        sessions = {"validation": set(fold.validation_sessions), "test": set(fold.test_sessions)}
        for split, split_sessions in sessions.items():
            part = merged[merged["session"].isin(split_sessions)].copy()
            if part.empty:
                continue
            part.insert(0, "split", split)
            part.insert(0, "fold", int(fold.fold))
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    output = pd.concat(parts, ignore_index=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"])
    output["month"] = output["timestamp"].dt.strftime("%Y-%m")
    return output


def _base_mask(frame: pd.DataFrame, family: str, params: dict[str, Any], column_map: dict[str, str] | None = None) -> pd.Series:
    columns = _signal_column_map(override=column_map)
    if family == "opening_range_breakout":
        if params["direction"] == "long":
            return _flag(frame, _mapped_column(columns, "opening_high")) | _flag(frame, _mapped_column(columns, "breaks_roll_high"))
        return _flag(frame, _mapped_column(columns, "opening_low")) | _flag(frame, _mapped_column(columns, "breaks_roll_low"))
    if family == "opening_bias_followthrough":
        attempts_key = "opening_high_attempts" if params["direction"] == "long" else "opening_low_attempts"
        return _clean(frame, _mapped_column(columns, attempts_key), 0.0).ge(float(params.get("bias_attempts_min", 1.0)))
    if family == "gap_fill":
        gap = _clean(frame, "target_overnight_ret", np.nan)
        return gap.le(float(params["gap_low"])) if params["direction"] == "long" else gap.ge(float(params["gap_high"]))
    if family == "vwap_stretch_reversal":
        dist = _clean(frame, "target_dist_vwap_atr", np.nan)
        return dist.le(-float(params["stretch_abs"])) if params["direction"] == "long" else dist.ge(float(params["stretch_abs"]))
    if family == "failed_breakout":
        return _flag(frame, _mapped_column(columns, "failed_breakout_low")) if params["direction"] == "long" else _flag(frame, _mapped_column(columns, "failed_breakout_high"))
    if family == "compression_breakout":
        compression = _clean(frame, _mapped_column(columns, "range_ratio"), np.inf).le(float(params["range_ratio_max"]))
        if params["direction"] == "long":
            return compression & (_flag(frame, _mapped_column(columns, "breaks_roll_high")) | _flag(frame, _mapped_column(columns, "opening_high")))
        return compression & (_flag(frame, _mapped_column(columns, "breaks_roll_low")) | _flag(frame, _mapped_column(columns, "opening_low")))
    if family == "breakdown_short_risk_off":
        return _flag(frame, _mapped_column(columns, "breaks_roll_low")) | _flag(frame, _mapped_column(columns, "opening_low"))
    raise ValueError(f"Unsupported setup family: {family}")


def signal_mask(frame: pd.DataFrame, family: str, params: dict[str, Any], column_map: dict[str, str] | None = None) -> pd.Series:
    columns = _signal_column_map(override=column_map)
    direction = str(params["direction"])
    rel_volume = _clean(frame, "target_rel_volume_by_bar", 0.0)
    close_location = _clean(frame, "target_close_location_bar", 0.5)
    dist_vwap = _clean(frame, _mapped_column(columns, "dist_vwap"), 0.0)
    dist_open = _clean(frame, _mapped_column(columns, "dist_open"), 0.0)
    range_ratio = _clean(frame, _mapped_column(columns, "range_ratio"), np.inf)
    upper_wick = _clean(frame, "target_upper_wick_ratio", 0.0)
    lower_wick = _clean(frame, "target_lower_wick_ratio", 0.0)
    risk_off = _clean(frame, "risk_off_score", 0.0)
    index_count = _clean(frame, _mapped_column(columns, "positive_index_count"), np.nan)
    sector_count = _clean(frame, _mapped_column(columns, "positive_sector_count"), np.nan)

    mask = _base_mask(frame, family, params, columns)
    if family == "opening_range_breakout":
        mask &= rel_volume.ge(float(params["rel_volume_min"]))
        if direction == "long":
            mask &= dist_vwap.ge(float(params["vwap_min"]))
            mask &= close_location.ge(float(params["close_location_min"]))
            if "breakout_margin_min" in params:
                mask &= _clean(frame, _mapped_column(columns, "breakout_margin_high"), -np.inf).ge(float(params["breakout_margin_min"]))
            if "opening_persistence_min" in params:
                mask &= _clean(frame, _mapped_column(columns, "opening_high_persistence"), 0.0).ge(float(params["opening_persistence_min"]))
            if "opening_attempts_max" in params:
                mask &= _clean(frame, _mapped_column(columns, "opening_high_attempts"), np.inf).le(float(params["opening_attempts_max"]))
            if "max_upper_wick" in params:
                mask &= upper_wick.le(float(params["max_upper_wick"]))
        else:
            mask &= dist_vwap.le(-float(params["vwap_min"]))
            mask &= close_location.le(1.0 - float(params["close_location_min"]))
            if "breakout_margin_min" in params:
                mask &= _clean(frame, _mapped_column(columns, "breakout_margin_low"), -np.inf).ge(float(params["breakout_margin_min"]))
            if "opening_persistence_min" in params:
                mask &= _clean(frame, _mapped_column(columns, "opening_low_persistence"), 0.0).ge(float(params["opening_persistence_min"]))
            if "opening_attempts_max" in params:
                mask &= _clean(frame, _mapped_column(columns, "opening_low_attempts"), np.inf).le(float(params["opening_attempts_max"]))
            if "max_lower_wick" in params:
                mask &= lower_wick.le(float(params["max_lower_wick"]))
        if "rel_cum_volume_min" in params:
            mask &= _clean(frame, "target_rel_cum_volume_by_bar", 0.0).ge(float(params["rel_cum_volume_min"]))
        if "rel_volume_accel_min" in params:
            mask &= _clean(frame, _mapped_column(columns, "rel_volume_accel"), 0.0).ge(float(params["rel_volume_accel_min"]))
        if "rv_rel_min" in params:
            mask &= _clean(frame, _mapped_column(columns, "rv_rel"), 0.0).ge(float(params["rv_rel_min"]))
        if "range_ratio_max" in params:
            mask &= range_ratio.le(float(params["range_ratio_max"]))
        if "positive_index_min" in params:
            mask &= index_count.ge(float(params["positive_index_min"]))
        if "positive_sector_min" in params:
            mask &= sector_count.ge(float(params["positive_sector_min"]))
        if "positive_index_open_min" in params:
            mask &= _clean(frame, "positive_index_count_open", np.nan).ge(float(params["positive_index_open_min"]))
        if "positive_sector_open_min" in params:
            mask &= _clean(frame, "positive_sector_count_open", np.nan).ge(float(params["positive_sector_open_min"]))
        if "index_above_vwap_min" in params:
            mask &= _clean(frame, "index_above_vwap_count", np.nan).ge(float(params["index_above_vwap_min"]))
        if "sector_above_vwap_min" in params:
            mask &= _clean(frame, "sector_above_vwap_count", np.nan).ge(float(params["sector_above_vwap_min"]))
        if "relopen_qqq_spy_min" in params:
            mask &= _clean(frame, "relopen_QQQ_SPY", np.nan).ge(float(params["relopen_qqq_spy_min"]))
        if "relopen_iwm_spy_min" in params:
            mask &= _clean(frame, "relopen_IWM_SPY", np.nan).ge(float(params["relopen_iwm_spy_min"]))
        if "risk_on_open_min" in params:
            mask &= _clean(frame, "risk_on_open_confirmation", np.nan).ge(float(params["risk_on_open_min"]))
        if "risk_on_min" in params:
            mask &= _clean(frame, "risk_on_score", np.nan).ge(float(params["risk_on_min"]))
        if "risk_off_max" in params:
            mask &= risk_off.le(float(params["risk_off_max"]))
        if "min_minutes_to_close" in params:
            mask &= _clean(frame, "target_minutes_to_close", np.nan).ge(float(params["min_minutes_to_close"]))
    elif family == "opening_bias_followthrough":
        mask &= rel_volume.ge(float(params["rel_volume_min"]))
        if "min_minutes_from_open" in params:
            mask &= _clean(frame, _mapped_column(columns, "minutes_from_open"), np.nan).ge(float(params["min_minutes_from_open"]))
        if "min_minutes_to_close" in params:
            mask &= _clean(frame, "target_minutes_to_close", np.nan).ge(float(params["min_minutes_to_close"]))
        if "rel_volume_accel_min" in params:
            mask &= _clean(frame, _mapped_column(columns, "rel_volume_accel"), 0.0).ge(float(params["rel_volume_accel_min"]))
        if "range_ratio_max" in params:
            mask &= range_ratio.le(float(params["range_ratio_max"]))
        if "positive_index_min" in params:
            mask &= index_count.ge(float(params["positive_index_min"]))
        if "positive_sector_min" in params:
            mask &= sector_count.ge(float(params["positive_sector_min"]))
        if "positive_index_open_min" in params:
            mask &= _clean(frame, "positive_index_count_open", np.nan).ge(float(params["positive_index_open_min"]))
        if "positive_sector_open_min" in params:
            mask &= _clean(frame, "positive_sector_count_open", np.nan).ge(float(params["positive_sector_open_min"]))
        if "index_above_vwap_min" in params:
            mask &= _clean(frame, "index_above_vwap_count", np.nan).ge(float(params["index_above_vwap_min"]))
        if "sector_above_vwap_min" in params:
            mask &= _clean(frame, "sector_above_vwap_count", np.nan).ge(float(params["sector_above_vwap_min"]))
        if "relopen_qqq_spy_min" in params:
            mask &= _clean(frame, "relopen_QQQ_SPY", np.nan).ge(float(params["relopen_qqq_spy_min"]))
        if "relopen_iwm_spy_min" in params:
            mask &= _clean(frame, "relopen_IWM_SPY", np.nan).ge(float(params["relopen_iwm_spy_min"]))
        if "risk_on_open_min" in params:
            mask &= _clean(frame, "risk_on_open_confirmation", np.nan).ge(float(params["risk_on_open_min"]))
        if "risk_on_min" in params:
            mask &= _clean(frame, "risk_on_score", np.nan).ge(float(params["risk_on_min"]))
        if "risk_off_max" in params:
            mask &= risk_off.le(float(params["risk_off_max"]))
        if direction == "long":
            mask &= dist_open.ge(float(params.get("dist_open_min", 0.0)))
            mask &= dist_vwap.ge(float(params["vwap_floor"]))
            if "vwap_ceiling" in params:
                mask &= dist_vwap.le(float(params["vwap_ceiling"]))
            mask &= close_location.ge(float(params["close_location_min"]))
            if "max_upper_wick" in params:
                mask &= upper_wick.le(float(params["max_upper_wick"]))
        else:
            mask &= dist_open.le(-float(params.get("dist_open_min", 0.0)))
            mask &= dist_vwap.le(-float(params["vwap_floor"]))
            if "vwap_ceiling" in params:
                mask &= dist_vwap.ge(-float(params["vwap_ceiling"]))
            mask &= close_location.le(1.0 - float(params["close_location_min"]))
            if "max_lower_wick" in params:
                mask &= lower_wick.le(float(params["max_lower_wick"]))
    elif family == "gap_fill":
        mask &= _clean(frame, "target_gap_fill_progress", -np.inf).ge(float(params["gap_fill_progress_min"]))
        if direction == "long":
            mask &= dist_vwap.ge(float(params["vwap_floor"]))
        else:
            mask &= dist_vwap.le(-float(params["vwap_floor"]))
    elif family == "vwap_stretch_reversal":
        mask &= range_ratio.le(float(params["range_ratio_max"]))
        if direction == "long":
            mask &= lower_wick.ge(float(params["wick_min"]))
            mask &= close_location.ge(float(params["close_location_min"]))
        else:
            mask &= upper_wick.ge(float(params["wick_min"]))
            mask &= close_location.le(1.0 - float(params["close_location_min"]))
    elif family == "failed_breakout":
        mask &= rel_volume.ge(float(params["rel_volume_min"]))
        if direction == "long":
            mask &= lower_wick.ge(float(params["wick_min"]))
            mask &= close_location.ge(float(params["close_location_min"]))
        else:
            mask &= upper_wick.ge(float(params["wick_min"]))
            mask &= close_location.le(1.0 - float(params["close_location_min"]))
    elif family == "compression_breakout":
        mask &= rel_volume.ge(float(params["rel_volume_min"]))
        if direction == "long":
            mask &= close_location.ge(float(params["close_location_min"]))
            mask &= dist_vwap.ge(float(params["vwap_min"]))
        else:
            mask &= close_location.le(1.0 - float(params["close_location_min"]))
            mask &= dist_vwap.le(-float(params["vwap_min"]))
    elif family == "breakdown_short_risk_off":
        mask &= rel_volume.ge(float(params["rel_volume_min"]))
        mask &= risk_off.ge(float(params["risk_off_min"]))
        mask &= index_count.le(float(params["positive_index_max"]))
        mask &= sector_count.le(float(params["positive_sector_max"]))
        mask &= dist_vwap.le(-float(params["vwap_min"]))
    return mask.fillna(False).astype(bool)


def generate_family_specs(frame: pd.DataFrame, config: dict[str, Any], fold: int, horizon: int) -> pd.DataFrame:
    cfg = _search_cfg(config)
    columns = _signal_column_map(config)
    rel_qs = [float(value) for value in cfg.get("rel_volume_quantiles", [0.50, 0.67, 0.75])]
    wick_qs = [float(value) for value in cfg.get("wick_quantiles", [0.60, 0.75])]
    range_qs = [float(value) for value in cfg.get("range_ratio_quantiles", [0.33, 0.50])]
    close_locations = [float(value) for value in cfg.get("close_location_thresholds", [0.55, 0.70])]
    vwap_mins = [float(value) for value in cfg.get("vwap_abs_mins", [0.0, 0.25, 0.50])]
    progress_mins = [float(value) for value in cfg.get("gap_fill_progress_mins", [0.20, 0.50])]
    families = set(str(value) for value in cfg.get("families", []))
    if not families:
        families = {
            "opening_range_breakout",
            "opening_bias_followthrough",
            "gap_fill",
            "vwap_stretch_reversal",
            "failed_breakout",
            "compression_breakout",
            "breakdown_short_risk_off",
        }

    thresholds = {
        "rel_volume": {q: _quantile(frame, "target_rel_volume_by_bar", q, 1.0) for q in rel_qs},
        "rel_cum_volume": {
            q: _quantile(frame, "target_rel_cum_volume_by_bar", q, 1.0)
            for q in [float(value) for value in cfg.get("rel_cum_volume_quantiles", [0.50, 0.67])]
        },
        "rel_volume_accel": {
            q: _quantile(frame, _mapped_column(columns, "rel_volume_accel"), q, 1.0)
            for q in [float(value) for value in cfg.get("rel_volume_accel_quantiles", [0.50, 0.67])]
        },
        "breakout_margin_high": {
            q: _quantile(frame, _mapped_column(columns, "breakout_margin_high"), q, 0.0)
            for q in [float(value) for value in cfg.get("breakout_margin_quantiles", [0.50, 0.67])]
        },
        "breakout_margin_low": {
            q: _quantile(frame, _mapped_column(columns, "breakout_margin_low"), q, 0.0)
            for q in [float(value) for value in cfg.get("breakout_margin_quantiles", [0.50, 0.67])]
        },
        "rv_rel": {
            q: _quantile(frame, _mapped_column(columns, "rv_rel"), q, 1.0)
            for q in [float(value) for value in cfg.get("rv_rel_quantiles", [0.50, 0.67])]
        },
        "upper_wick": {q: _quantile(frame, "target_upper_wick_ratio", q, 0.4) for q in wick_qs},
        "lower_wick": {q: _quantile(frame, "target_lower_wick_ratio", q, 0.4) for q in wick_qs},
        "range_ratio": {q: _quantile(frame, _mapped_column(columns, "range_ratio"), q, 1.0) for q in range_qs},
        "stretch_abs": {
            q: abs(_quantile(frame, "target_dist_vwap_atr", q, 1.0)) for q in [0.15, 0.20, 0.80, 0.85]
        },
        "gap_low": _quantile(frame, "target_overnight_ret", float(cfg.get("gap_low_quantile", 0.25)), -0.002),
        "gap_high": _quantile(frame, "target_overnight_ret", float(cfg.get("gap_high_quantile", 0.75)), 0.002),
        "risk_off": {
            q: _quantile(frame, "risk_off_score", q, 0.0) for q in [float(value) for value in cfg.get("risk_off_quantiles", [0.60, 0.75])]
        },
        "positive_index_low": _quantile(frame, _mapped_column(columns, "positive_index_count"), float(cfg.get("breadth_low_quantile", 0.33)), 1.0),
        "positive_sector_low": _quantile(frame, _mapped_column(columns, "positive_sector_count"), float(cfg.get("breadth_low_quantile", 0.33)), 2.0),
    }

    rows: list[dict[str, Any]] = []

    def add(family: str, direction: str, params: dict[str, Any]) -> None:
        if family not in families:
            return
        payload = {"direction": direction, **params}
        row = {
            "fold": int(fold),
            "family": family,
            "direction": direction,
            "horizon_bars": int(horizon),
            "params_json": _json_dumps(payload),
            "column_map_json": _json_dumps(columns),
        }
        row["candidate_id"] = _candidate_id(row)
        rows.append(row)

    for direction in ["long", "short"]:
        for rel_q in rel_qs:
            for close_location in close_locations:
                for vwap_min in vwap_mins:
                    add(
                        "opening_range_breakout",
                        direction,
                        {
                            "rel_volume_min": thresholds["rel_volume"][rel_q],
                            "rel_volume_q": rel_q,
                            "close_location_min": close_location,
                            "vwap_min": vwap_min,
                        },
                    )

    opening_confirmation_cfg = cfg.get("opening_breakout_confirmations", {})
    if bool(opening_confirmation_cfg.get("enabled", False)):
        directions = [str(value) for value in opening_confirmation_cfg.get("directions", ["long"])]
        filter_sets = list(opening_confirmation_cfg.get("filter_sets", []))
        if not filter_sets:
            filter_sets = [{}]
        for direction in directions:
            for rel_q in [float(value) for value in opening_confirmation_cfg.get("rel_volume_quantiles", rel_qs)]:
                for close_location in [float(value) for value in opening_confirmation_cfg.get("close_location_thresholds", close_locations)]:
                    for vwap_min in [float(value) for value in opening_confirmation_cfg.get("vwap_abs_mins", [0.0])]:
                        for filter_set in filter_sets:
                            params = {
                                "rel_volume_min": thresholds["rel_volume"][rel_q],
                                "rel_volume_q": rel_q,
                                "close_location_min": close_location,
                                "vwap_min": vwap_min,
                                "filter_set": str(filter_set.get("name", "confirmation")),
                            }
                            margin_q = filter_set.get("breakout_margin_q")
                            if margin_q is not None:
                                margin_bucket = "breakout_margin_high" if direction == "long" else "breakout_margin_low"
                                params["breakout_margin_min"] = thresholds[margin_bucket][float(margin_q)]
                                params["breakout_margin_q"] = float(margin_q)
                            rel_cum_q = filter_set.get("rel_cum_volume_q")
                            _add_optional_threshold(params, "rel_cum_volume_min", _optional_quantile(frame, "target_rel_cum_volume_by_bar", rel_cum_q, 1.0))
                            if rel_cum_q is not None:
                                params["rel_cum_volume_q"] = float(rel_cum_q)
                            accel_q = filter_set.get("rel_volume_accel_q")
                            _add_optional_threshold(
                                params,
                                "rel_volume_accel_min",
                                thresholds["rel_volume_accel"].get(float(accel_q)) if accel_q is not None else None,
                            )
                            if accel_q is not None:
                                params["rel_volume_accel_q"] = float(accel_q)
                            rv_q = filter_set.get("rv_rel_q")
                            _add_optional_threshold(params, "rv_rel_min", thresholds["rv_rel"].get(float(rv_q)) if rv_q is not None else None)
                            if rv_q is not None:
                                params["rv_rel_q"] = float(rv_q)
                            range_q = filter_set.get("range_ratio_q")
                            _add_optional_threshold(
                                params,
                                "range_ratio_max",
                                thresholds["range_ratio"].get(float(range_q)) if range_q is not None else None,
                            )
                            if range_q is not None:
                                params["range_ratio_q"] = float(range_q)
                            for key in [
                                "positive_index_min",
                                "positive_sector_min",
                                "positive_index_open_min",
                                "positive_sector_open_min",
                                "index_above_vwap_min",
                                "sector_above_vwap_min",
                                "relopen_qqq_spy_min",
                                "relopen_iwm_spy_min",
                                "risk_on_open_min",
                                "risk_on_min",
                                "risk_off_max",
                                "opening_persistence_min",
                                "opening_attempts_max",
                                "max_upper_wick",
                                "max_lower_wick",
                                "min_minutes_to_close",
                            ]:
                                _add_optional_threshold(params, key, filter_set.get(key))
                            add("opening_range_breakout", direction, params)

    opening_bias_cfg = cfg.get("opening_bias_followthrough", {})
    if bool(opening_bias_cfg.get("enabled", False)) or "opening_bias_followthrough" in families:
        directions = [str(value) for value in opening_bias_cfg.get("directions", ["long"])]
        filter_sets = list(opening_bias_cfg.get("filter_sets", []))
        if not filter_sets:
            filter_sets = [{"name": "liquid_bias"}]
        for direction in directions:
            for rel_q in [float(value) for value in opening_bias_cfg.get("rel_volume_quantiles", rel_qs)]:
                for close_location in [float(value) for value in opening_bias_cfg.get("close_location_thresholds", close_locations)]:
                    for vwap_floor in [float(value) for value in opening_bias_cfg.get("vwap_floors", [-0.25, 0.0])]:
                        for vwap_ceiling in [float(value) for value in opening_bias_cfg.get("vwap_ceilings", [0.75, 1.25])]:
                            if vwap_ceiling < vwap_floor:
                                continue
                            for dist_open_min in [float(value) for value in opening_bias_cfg.get("dist_open_mins", [0.0])]:
                                for filter_set in filter_sets:
                                    params = {
                                        "rel_volume_min": thresholds["rel_volume"].get(
                                            rel_q, _quantile(frame, "target_rel_volume_by_bar", rel_q, 1.0)
                                        ),
                                        "rel_volume_q": rel_q,
                                        "close_location_min": close_location,
                                        "vwap_floor": vwap_floor,
                                        "vwap_ceiling": vwap_ceiling,
                                        "dist_open_min": dist_open_min,
                                        "filter_set": str(filter_set.get("name", "liquid_bias")),
                                    }
                                    for key in [
                                        "bias_attempts_min",
                                        "positive_index_min",
                                        "positive_sector_min",
                                        "positive_index_open_min",
                                        "positive_sector_open_min",
                                        "index_above_vwap_min",
                                        "sector_above_vwap_min",
                                        "relopen_qqq_spy_min",
                                        "relopen_iwm_spy_min",
                                        "risk_on_open_min",
                                        "risk_on_min",
                                        "risk_off_max",
                                        "max_upper_wick",
                                        "max_lower_wick",
                                        "min_minutes_from_open",
                                        "min_minutes_to_close",
                                    ]:
                                        _add_optional_threshold(params, key, filter_set.get(key))
                                    accel_q = filter_set.get("rel_volume_accel_q")
                                    _add_optional_threshold(
                                        params,
                                        "rel_volume_accel_min",
                                        thresholds["rel_volume_accel"].get(float(accel_q)) if accel_q is not None else None,
                                    )
                                    if accel_q is not None:
                                        params["rel_volume_accel_q"] = float(accel_q)
                                    range_q = filter_set.get("range_ratio_q")
                                    _add_optional_threshold(
                                        params,
                                        "range_ratio_max",
                                        thresholds["range_ratio"].get(float(range_q)) if range_q is not None else None,
                                    )
                                    if range_q is not None:
                                        params["range_ratio_q"] = float(range_q)
                                    add("opening_bias_followthrough", direction, params)

    for direction in ["long", "short"]:
        for progress in progress_mins:
            for vwap_floor in [-0.25, 0.0, 0.25]:
                add(
                    "gap_fill",
                    direction,
                    {
                        "gap_low": thresholds["gap_low"],
                        "gap_high": thresholds["gap_high"],
                        "gap_fill_progress_min": progress,
                        "vwap_floor": vwap_floor,
                    },
                )

    for direction in ["long", "short"]:
        stretch_values = [thresholds["stretch_abs"][0.15], thresholds["stretch_abs"][0.20]]
        if direction == "short":
            stretch_values = [thresholds["stretch_abs"][0.80], thresholds["stretch_abs"][0.85]]
        for stretch in stretch_values:
            for wick_q in wick_qs:
                for range_q in range_qs:
                    add(
                        "vwap_stretch_reversal",
                        direction,
                        {
                            "stretch_abs": abs(stretch),
                            "wick_min": thresholds["lower_wick" if direction == "long" else "upper_wick"][wick_q],
                            "wick_q": wick_q,
                            "range_ratio_max": thresholds["range_ratio"][range_q],
                            "range_ratio_q": range_q,
                            "close_location_min": 0.50,
                        },
                    )

    for direction in ["long", "short"]:
        for rel_q in rel_qs:
            for wick_q in wick_qs:
                add(
                    "failed_breakout",
                    direction,
                    {
                        "rel_volume_min": thresholds["rel_volume"][rel_q],
                        "rel_volume_q": rel_q,
                        "wick_min": thresholds["lower_wick" if direction == "long" else "upper_wick"][wick_q],
                        "wick_q": wick_q,
                        "close_location_min": 0.50,
                    },
                )

    for direction in ["long", "short"]:
        for rel_q in rel_qs:
            for range_q in range_qs:
                for vwap_min in [0.0, 0.25]:
                    add(
                        "compression_breakout",
                        direction,
                        {
                            "rel_volume_min": thresholds["rel_volume"][rel_q],
                            "rel_volume_q": rel_q,
                            "range_ratio_max": thresholds["range_ratio"][range_q],
                            "range_ratio_q": range_q,
                            "close_location_min": 0.55,
                            "vwap_min": vwap_min,
                        },
                    )

    for rel_q in rel_qs:
        for risk_q, risk_min in thresholds["risk_off"].items():
            for vwap_min in [0.0, 0.25]:
                add(
                    "breakdown_short_risk_off",
                    "short",
                    {
                        "rel_volume_min": thresholds["rel_volume"][rel_q],
                        "rel_volume_q": rel_q,
                        "risk_off_min": risk_min,
                        "risk_off_q": risk_q,
                        "positive_index_max": thresholds["positive_index_low"],
                        "positive_sector_max": thresholds["positive_sector_low"],
                        "vwap_min": vwap_min,
                    },
                )

    return pd.DataFrame(rows).drop_duplicates("candidate_id").reset_index(drop=True)


def _profit_factor(active_net: pd.Series) -> float:
    gross_profit = active_net[active_net > 0.0].sum()
    gross_loss = -active_net[active_net < 0.0].sum()
    if gross_loss == 0.0:
        return np.inf if gross_profit > 0.0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def _top_abs_share(values: pd.Series) -> float:
    denom = values.abs().sum()
    if denom == 0.0 or np.isnan(denom):
        return np.nan
    return float(values.abs().max() / denom)


def evaluate_position(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> dict[str, float | int]:
    position = position.astype(float).fillna(0.0)
    active = position.abs() > 0.0
    gross = position * frame["fwd_ret"].astype(float)
    cost = scenario_cost_return(frame, position, scenario)
    net = gross - cost
    active_net = net[active]
    daily = net.groupby(frame["session"]).sum()
    monthly = net.groupby(frame["month"]).sum() if "month" in frame else pd.Series(dtype=float)
    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "gross_return": float(gross.sum()),
        "cost_return": float(cost.sum()),
        "net_return": float(net.sum()),
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "median_trade_net": float(active_net.median()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0.0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": max_drawdown(net),
        "top_day_abs_net_share": _top_abs_share(daily),
        "top_month_abs_net_share": _top_abs_share(monthly) if not monthly.empty else np.nan,
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    spec: pd.Series,
    split: str,
    scenario: dict[str, Any],
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    params = _json_loads(spec["params_json"])
    spec_map = _json_loads(spec.get("column_map_json")) if "column_map_json" in spec else {}
    columns = _signal_column_map(override=spec_map or column_map)
    direction = 1.0 if str(spec["direction"]) == "long" else -1.0
    signal = signal_mask(frame, str(spec["family"]), params, columns)
    base = _base_mask(frame, str(spec["family"]), params, columns)
    rows = []
    for bucket, mask in [("setup_signal", signal), ("base_segment_control", base), ("always_flat", pd.Series(False, index=frame.index))]:
        position = pd.Series(0.0, index=frame.index)
        position.loc[mask] = direction
        row = {
            "candidate_id": spec["candidate_id"],
            "fold": int(spec["fold"]),
            "split": split,
            "family": spec["family"],
            "direction": spec["direction"],
            "horizon_bars": int(spec["horizon_bars"]),
            "params_json": spec["params_json"],
            "column_map_json": _json_dumps(columns),
            "bucket": bucket,
            "cost_scenario": scenario["cost_scenario"],
            "cost_kind": scenario["cost_kind"],
        }
        rows.append({**row, **evaluate_position(frame, position, scenario)})
    result = pd.DataFrame(rows)
    signal_row = result[result["bucket"].eq("setup_signal")].iloc[0]
    control_row = result[result["bucket"].eq("base_segment_control")].iloc[0]
    result["net_delta_vs_base_segment"] = signal_row["net_return"] - control_row["net_return"]
    result["sharpe_delta_vs_base_segment"] = signal_row["daily_sharpe"] - control_row["daily_sharpe"]
    return result


def validation_grid(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg}, [primary])
    rows = []
    validation = dataset[dataset["split"].eq("validation")]
    for (fold, horizon), frame in validation.groupby(["fold", "horizon_bars"], sort=False):
        specs = generate_family_specs(frame, config, int(fold), int(horizon))
        for _, spec in specs.iterrows():
            for scenario in scenarios:
                rows.append(evaluate_candidate(frame, spec, "validation", scenario))
    return add_validation_status(pd.concat(rows, ignore_index=True), config) if rows else pd.DataFrame()


def classify_validation(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _combined_cfg(config)
    if row["bucket"] != "setup_signal":
        return "control"
    if int(row["trades"]) < int(cfg.get("min_trades", 40)):
        return "rejected_insufficient_trades"
    if float(row["net_return"]) <= 0.0 or float(row["avg_trade_net"]) <= 0.0:
        return "rejected_negative_edge"
    if float(row["profit_factor"]) < float(cfg.get("min_profit_factor", 1.05)):
        return "rejected_weak_profit_factor"
    if float(row["daily_sharpe"]) < float(cfg.get("min_daily_sharpe", 0.30)):
        return "rejected_weak_sharpe"
    if float(row["max_drawdown"]) > float(cfg.get("max_drawdown", 0.30)):
        return "rejected_drawdown"
    if float(row["top_day_abs_net_share"]) > float(cfg.get("max_top_day_abs_net_share", 0.30)):
        return "rejected_day_concentration"
    if float(row["top_month_abs_net_share"]) > float(cfg.get("max_top_month_abs_net_share", 0.50)):
        return "rejected_month_concentration"
    if bool(cfg.get("require_base_segment_improvement", True)) and float(row["net_delta_vs_base_segment"]) <= 0.0:
        return "rejected_no_base_segment_improvement"
    return "setup_validation_candidate"


def add_validation_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["candidate_status"] = rows.apply(lambda row: classify_validation(row, config), axis=1)
    return rows


def select_specs(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    candidates = validation[
        validation["bucket"].eq("setup_signal")
        & validation["cost_scenario"].eq(primary)
        & validation["candidate_status"].eq("setup_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("setup_signal") & validation["cost_scenario"].eq(primary)].copy()
    candidates["utility_score"] = (
        candidates["daily_sharpe"].fillna(0.0)
        + 100.0 * candidates["avg_trade_net"].fillna(0.0)
        + 0.25 * candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        + candidates["net_delta_vs_base_segment"].fillna(0.0)
        - 0.25 * candidates["top_month_abs_net_share"].fillna(1.0)
    )
    candidates = candidates.sort_values(
        ["candidate_status", "utility_score", "net_return"],
        ascending=[True, False, False],
        kind="stable",
    )
    candidates = candidates.drop_duplicates(["fold", "family", "direction", "horizon_bars", "params_json"])
    cols = ["candidate_id", "fold", "family", "direction", "horizon_bars", "params_json", "candidate_status", "utility_score"]
    if "column_map_json" in candidates.columns:
        cols.insert(6, "column_map_json")
    return candidates.loc[:, cols].head(int(cfg.get("max_selected", 120))).reset_index(drop=True)


def evaluate_selected_on_split(dataset: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg})
    rows = []
    for _, spec in specs.iterrows():
        frame = dataset[
            dataset["split"].eq(split)
            & dataset["fold"].eq(int(spec["fold"]))
            & dataset["horizon_bars"].eq(int(spec["horizon_bars"]))
        ].copy()
        if frame.empty:
            continue
        for scenario in scenarios:
            rows.append(evaluate_candidate(frame, spec, split, scenario))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def decision_table(validation: pd.DataFrame, test: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    rows = []
    for _, spec in specs.iterrows():
        cid = spec["candidate_id"]
        val = validation[
            validation["candidate_id"].eq(cid) & validation["bucket"].eq("setup_signal") & validation["cost_scenario"].eq(primary)
        ]
        primary_row = test[test["candidate_id"].eq(cid) & test["bucket"].eq("setup_signal") & test["cost_scenario"].eq(primary)]
        conservative_row = test[test["candidate_id"].eq(cid) & test["bucket"].eq("setup_signal") & test["cost_scenario"].eq(conservative)]
        stress_row = test[test["candidate_id"].eq(cid) & test["bucket"].eq("setup_signal") & test["cost_scenario"].eq(stress)]
        val_row = val.iloc[0] if not val.empty else pd.Series(dtype=object)
        test_row = primary_row.iloc[0] if not primary_row.empty else pd.Series(dtype=object)
        cons_row = conservative_row.iloc[0] if not conservative_row.empty else pd.Series(dtype=object)
        stress_item = stress_row.iloc[0] if not stress_row.empty else pd.Series(dtype=object)

        primary_ok = bool(
            not test_row.empty
            and int(test_row["trades"]) >= int(cfg.get("min_trades", 40))
            and float(test_row["net_return"]) > 0.0
            and float(test_row["avg_trade_net"]) > 0.0
            and float(test_row["profit_factor"]) >= float(cfg.get("min_profit_factor", 1.05))
            and float(test_row["max_drawdown"]) <= float(cfg.get("max_drawdown", 0.30))
            and float(test_row["top_day_abs_net_share"]) <= float(cfg.get("max_top_day_abs_net_share", 0.30))
            and float(test_row["top_month_abs_net_share"]) <= float(cfg.get("max_top_month_abs_net_share", 0.50))
        )
        conservative_ok = bool(not cons_row.empty and cons_row["net_return"] > 0.0 and cons_row["avg_trade_net"] > 0.0)
        stress_ok = bool(not stress_item.empty and stress_item["net_return"] >= 0.0 and stress_item["avg_trade_net"] >= 0.0)
        if primary_ok and conservative_ok and stress_ok:
            decision = "accepted_candidate"
        elif primary_ok and conservative_ok:
            decision = "cost_fragile"
        elif not test_row.empty and test_row["net_return"] > 0.0:
            decision = "research_candidate"
        else:
            decision = "rejected"

        rows.append(
            {
                **{col: spec[col] for col in ["candidate_id", "fold", "family", "direction", "horizon_bars", "params_json"]},
                "validation_status": val_row.get("candidate_status", ""),
                "decision": decision,
                "test_net_primary": test_row.get("net_return", np.nan),
                "test_sharpe_primary": test_row.get("daily_sharpe", np.nan),
                "test_profit_factor_primary": test_row.get("profit_factor", np.nan),
                "test_avg_trade_net_primary": test_row.get("avg_trade_net", np.nan),
                "test_trades_primary": test_row.get("trades", np.nan),
                "test_top_day_abs_net_share_primary": test_row.get("top_day_abs_net_share", np.nan),
                "test_top_month_abs_net_share_primary": test_row.get("top_month_abs_net_share", np.nan),
                "test_net_delta_vs_base_segment_primary": test_row.get("net_delta_vs_base_segment", np.nan),
                "test_net_conservative": cons_row.get("net_return", np.nan),
                "test_net_stress": stress_item.get("net_return", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "test_avg_trade_net_primary", "test_net_primary"], ascending=[True, False, False], kind="stable")


def family_stability(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    rows = []
    for (family, direction, horizon), group in decisions.groupby(["family", "direction", "horizon_bars"], sort=False):
        ranked = group.sort_values(["test_avg_trade_net_primary", "test_net_primary"], ascending=[False, False], kind="stable")
        best_by_fold = ranked.drop_duplicates("fold", keep="first")
        primary_positive_by_fold = group.assign(
            primary_positive=group["test_net_primary"].gt(0.0) & group["test_avg_trade_net_primary"].gt(0.0)
        ).groupby("fold")["primary_positive"].any()
        stress_nonnegative_by_fold = group.assign(stress_nonnegative=group["test_net_stress"].ge(0.0)).groupby("fold")[
            "stress_nonnegative"
        ].any()
        accepted_by_fold = group.assign(accepted=group["decision"].eq("accepted_candidate")).groupby("fold")["accepted"].any()
        research_by_fold = group.assign(
            research_or_better=group["decision"].isin(["accepted_candidate", "cost_fragile", "research_candidate"])
        ).groupby("fold")["research_or_better"].any()
        rows.append(
            {
                "family": family,
                "direction": direction,
                "horizon_bars": int(horizon),
                "folds_present": int(group["fold"].nunique()),
                "accepted_folds": int(accepted_by_fold.sum()),
                "research_or_better_folds": int(research_by_fold.sum()),
                "primary_positive_folds": int(primary_positive_by_fold.sum()),
                "stress_nonnegative_folds": int(stress_nonnegative_by_fold.sum()),
                "min_best_fold_avg_trade_net": float(best_by_fold["test_avg_trade_net_primary"].min()),
                "median_best_fold_avg_trade_net": float(best_by_fold["test_avg_trade_net_primary"].median()),
                "min_best_fold_net_stress": float(best_by_fold["test_net_stress"].min()),
                "stable_family": bool(
                    group["fold"].nunique() >= 2 and primary_positive_by_fold.all() and stress_nonnegative_by_fold.all()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_family", "primary_positive_folds", "median_best_fold_avg_trade_net"],
        ascending=[False, False, False],
        kind="stable",
    )


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    specs: pd.DataFrame,
    decisions: pd.DataFrame,
    stability: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _combined_cfg(config)
    val_alpha = validation[validation["bucket"].eq("setup_signal")] if not validation.empty else pd.DataFrame()
    test_alpha = test[test["bucket"].eq("setup_signal")] if not test.empty else pd.DataFrame()
    val_counts = val_alpha["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="rows") if not val_alpha.empty else pd.DataFrame()
    decision_counts = decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    family_summary = (
        test_alpha.groupby(["cost_scenario", "family", "direction", "horizon_bars"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            median_net_return=("net_return", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0.0).mean())),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_net_delta_vs_base=("net_delta_vs_base_segment", "median"),
        )
        .sort_values(["cost_scenario", "median_avg_trade_net"], ascending=[True, False], kind="stable")
        if not test_alpha.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one setup signal is accepted under the configured costs."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No setup signal is accepted under the configured costs."
    )
    return f"""# Setup Signal Search - {target_symbol.upper()}

## Scope

- Searches interpretable SPY setup signals without HMM.
- Long and short are separate candidates.
- Thresholds are frozen on validation and then evaluated on test.
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`.

## Validation Status Counts

{_markdown_table(val_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Family Stability

{_markdown_table(stability, max_rows=int(cfg.get("report_top_rows", 100)))}

## Family Test Summary

{_markdown_table(family_summary, max_rows=int(cfg.get("report_top_rows", 100)))}

## Top Decisions

{_markdown_table(decisions.head(int(cfg.get("report_top_rows", 100))) if not decisions.empty else decisions)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    dataset = build_signal_dataset(config, target)
    validation = validation_grid(dataset, config) if not dataset.empty else pd.DataFrame()
    specs = select_specs(validation, config)
    selected_validation = validation[validation["candidate_id"].isin(specs["candidate_id"])] if not specs.empty else pd.DataFrame()
    test = evaluate_selected_on_split(dataset, specs, "test", config) if not specs.empty else pd.DataFrame()
    decisions = decision_table(selected_validation, test, specs, config)
    stability = family_stability(decisions)

    results_dir = results_output_dir(config, target)
    outputs = {
        "setup_signal_validation": results_dir / "setup_signal_validation.parquet",
        "setup_signal_selected_validation": results_dir / "setup_signal_selected_validation.parquet",
        "setup_signal_selected_specs": results_dir / "setup_signal_selected_specs.parquet",
        "setup_signal_test": results_dir / "setup_signal_test.parquet",
        "setup_signal_decisions": results_dir / "setup_signal_decisions.parquet",
        "setup_signal_family_stability": results_dir / "setup_signal_family_stability.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["setup_signal_validation"], index=False)
    selected_validation.to_parquet(outputs["setup_signal_selected_validation"], index=False)
    specs.to_parquet(outputs["setup_signal_selected_specs"], index=False)
    test.to_parquet(outputs["setup_signal_test"], index=False)
    decisions.to_parquet(outputs["setup_signal_decisions"], index=False)
    stability.to_parquet(outputs["setup_signal_family_stability"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, validation, test, specs, decisions, stability, outputs), encoding="utf-8")
    return report_path, outputs["setup_signal_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search interpretable setup signals before HMM overlays.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Setup signal search report written to: {report_path}")
    print(f"Setup signal decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

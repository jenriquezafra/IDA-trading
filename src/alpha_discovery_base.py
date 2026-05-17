from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import evaluate_position_with_cost
from src.hmm_lab import _lab_cfg, _target_symbol, build_lab_folds, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios


ALPHA_SPECS: dict[str, dict[str, Any]] = {
    "m6_base": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ()},
    "m6_long_only": {"column": "target_ret_6", "mode": "long_positive", "base_variant": "m6_base", "gates": ()},
    "m6_short_only": {"column": "target_ret_6", "mode": "short_negative", "base_variant": "m6_base", "gates": ()},
    "m6_ret12_confirm": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("ret12_confirm",)},
    "m6_vwap_confirm": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("vwap_confirm",)},
    "m6_ret12_vwap": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("ret12_confirm", "vwap_confirm")},
    "m6_low_chop": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("low_chop",)},
    "m6_low_stress": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("low_stress",)},
    "m6_directional_risk": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("directional_risk",)},
    "m6_breadth": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("cross_asset_breadth",)},
    "m6_efficiency": {"column": "target_ret_6", "mode": "signed", "base_variant": "m6_base", "gates": ("efficiency_confirm",)},
    "vwap_breakout": {"column": "target_dist_vwap_atr", "mode": "signed", "base_variant": "vwap_breakout", "gates": ()},
    "vwap_reversion": {"column": "target_dist_vwap_atr", "mode": "inverse", "base_variant": "vwap_reversion", "gates": ()},
    "gap_breakout": {"column": "target_dist_open", "mode": "signed", "base_variant": "gap_breakout", "gates": ()},
    "gap_reversion": {"column": "target_dist_open", "mode": "inverse", "base_variant": "gap_reversion", "gates": ()},
    "risk_on_long": {"column": "risk_on_score", "mode": "long_positive", "base_variant": "risk_on_long", "gates": ()},
    "risk_off_short": {"column": "risk_off_score", "mode": "short_positive", "base_variant": "risk_off_short", "gates": ()},
    "growth_defensive_momentum": {"column": "spread_growth_defensive_12", "mode": "signed", "base_variant": "growth_defensive_momentum", "gates": ()},
    "cyclicals_defensive_momentum": {"column": "spread_cyclicals_defensive_12", "mode": "signed", "base_variant": "cyclicals_defensive_momentum", "gates": ()},
    "equity_bonds_momentum": {"column": "spread_equity_bonds_12", "mode": "signed", "base_variant": "equity_bonds_momentum", "gates": ()},
    "credit_momentum": {"column": "spread_credit_12", "mode": "signed", "base_variant": "credit_momentum", "gates": ()},
}


REQUIRED_FEATURE_COLUMNS = sorted(
    {
        "target_open_next",
        "target_ret_6",
        "target_ret_12",
        "target_dist_vwap_atr",
        "target_dist_open",
        "target_signed_efficiency_12",
        "target_range_ratio_6_24",
        "risk_on_score",
        "risk_off_score",
        "chop_score",
        "intraday_stress_score",
        "positive_index_count_6",
        "positive_sector_count_6",
        "spread_growth_defensive_12",
        "spread_cyclicals_defensive_12",
        "spread_equity_bonds_12",
        "spread_credit_12",
        *[str(spec["column"]) for spec in ALPHA_SPECS.values()],
    }
)


def _discovery_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("alpha_discovery_base", {})


def _combined_cfg(config: dict[str, Any]) -> dict[str, Any]:
    combined = dict(config.get("operable_candidate_search", {}))
    combined.update(_discovery_cfg(config))
    return combined


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "alpha_discovery_base.md"


def _json_dumps(values: dict[str, float]) -> str:
    return json.dumps({key: float(value) for key, value in sorted(values.items()) if np.isfinite(float(value))}, sort_keys=True)


def _json_loads(value: str | float | int | None) -> dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    raw = json.loads(str(value))
    return {str(key): float(item) for key, item in raw.items()}


def _clean_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _candidate_key(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"fold{int(row['fold'])}__{row['alpha_variant']}__h{int(row['horizon_bars'])}"
        f"__thr{float(row['threshold']):g}"
    )


def load_feature_frame(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config))
    return features.sort_values(["session", "bar_index"], kind="stable")


def build_split_dataset(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _discovery_cfg(config)
    horizons = [int(value) for value in cfg.get("horizons", [12, 24])]
    features = load_feature_frame(config, target_symbol)
    forward_returns = build_forward_returns(features, horizons)
    indexed = features.reset_index(names="source_index")
    available = ["source_index", *[col for col in REQUIRED_FEATURE_COLUMNS if col in indexed.columns and col not in forward_returns.columns]]
    merged = forward_returns.merge(indexed.loc[:, available], on="source_index", how="left", validate="many_to_one")
    folds = build_lab_folds(features, config)
    splits = []
    for fold in folds:
        sessions = {
            "validation": set(fold.validation_sessions),
            "test": set(fold.test_sessions),
        }
        for split, split_sessions in sessions.items():
            part = merged[merged["session"].isin(split_sessions)].copy()
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
    output["feature_set"] = "base_alpha"
    output["n_states"] = 0
    output["seed"] = 0
    return output


def thresholds_for_variant(frame: pd.DataFrame, alpha_variant: str, quantiles: list[float]) -> list[float]:
    spec = ALPHA_SPECS[alpha_variant]
    values = _clean_series(frame, str(spec["column"]), np.nan).dropna()
    if values.empty:
        return []
    mode = str(spec["mode"])
    source = values.abs() if mode in {"signed", "inverse", "short_negative"} else values[values > 0]
    source = source[source > 0]
    if source.empty:
        return []
    thresholds = [float(source.quantile(float(q))) for q in quantiles]
    return sorted(set(round(value, 12) for value in thresholds if np.isfinite(value) and value >= 0))


def frozen_gates(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    cfg = _discovery_cfg(config)

    def quantile(column: str, q: float, default: float) -> float:
        values = _clean_series(frame, column, np.nan).dropna()
        return float(values.quantile(q)) if not values.empty else default

    def abs_quantile(column: str, q: float, default: float) -> float:
        values = _clean_series(frame, column, np.nan).dropna().abs()
        return float(values.quantile(q)) if not values.empty else default

    breadth_q = float(cfg.get("breadth_high_quantile", 0.65))
    return {
        "abs_efficiency_min": abs_quantile("target_signed_efficiency_12", float(cfg.get("efficiency_abs_quantile", 0.50)), 0.0),
        "range_ratio_max": quantile("target_range_ratio_6_24", float(cfg.get("range_ratio_max_quantile", 0.75)), np.inf),
        "chop_score_max": quantile("chop_score", float(cfg.get("chop_max_quantile", 0.70)), np.inf),
        "stress_score_max": quantile("intraday_stress_score", float(cfg.get("stress_max_quantile", 0.70)), np.inf),
        "risk_on_min": quantile("risk_on_score", float(cfg.get("risk_score_quantile", 0.55)), -np.inf),
        "risk_off_min": quantile("risk_off_score", float(cfg.get("risk_score_quantile", 0.55)), -np.inf),
        "positive_index_high": quantile("positive_index_count_6", breadth_q, np.inf),
        "positive_index_low": quantile("positive_index_count_6", 1.0 - breadth_q, -np.inf),
        "positive_sector_high": quantile("positive_sector_count_6", breadth_q, np.inf),
        "positive_sector_low": quantile("positive_sector_count_6", 1.0 - breadth_q, -np.inf),
    }


def alpha_position(frame: pd.DataFrame, alpha_variant: str, threshold: float, gates: dict[str, float] | None = None) -> pd.Series:
    if alpha_variant not in ALPHA_SPECS:
        raise ValueError(f"Unsupported alpha variant: {alpha_variant}")
    gates = gates or {}
    spec = ALPHA_SPECS[alpha_variant]
    values = _clean_series(frame, str(spec["column"]))
    threshold = float(threshold)
    position = pd.Series(0.0, index=frame.index)
    mode = str(spec["mode"])
    if mode == "signed":
        position.loc[values > threshold] = 1.0
        position.loc[values < -threshold] = -1.0
    elif mode == "inverse":
        position.loc[values > threshold] = -1.0
        position.loc[values < -threshold] = 1.0
    elif mode == "long_positive":
        position.loc[values > threshold] = 1.0
    elif mode == "short_positive":
        position.loc[values > threshold] = -1.0
    elif mode == "short_negative":
        position.loc[values < -threshold] = -1.0
    else:
        raise ValueError(f"Unsupported alpha mode: {mode}")

    active = position.ne(0.0)
    direction = position.copy()
    for gate in spec.get("gates", ()):
        if gate == "ret12_confirm":
            ret12 = _clean_series(frame, "target_ret_12")
            active &= np.sign(ret12).eq(direction) & ret12.ne(0.0)
        elif gate == "vwap_confirm":
            dist_vwap = _clean_series(frame, "target_dist_vwap_atr")
            active &= np.sign(dist_vwap).eq(direction) & dist_vwap.ne(0.0)
        elif gate == "efficiency_confirm":
            efficiency = _clean_series(frame, "target_signed_efficiency_12")
            active &= np.sign(efficiency).eq(direction) & efficiency.abs().ge(float(gates.get("abs_efficiency_min", 0.0)))
        elif gate == "low_chop":
            active &= _clean_series(frame, "chop_score").le(float(gates.get("chop_score_max", np.inf)))
        elif gate == "low_stress":
            active &= _clean_series(frame, "intraday_stress_score").le(float(gates.get("stress_score_max", np.inf)))
        elif gate == "low_range":
            active &= _clean_series(frame, "target_range_ratio_6_24").le(float(gates.get("range_ratio_max", np.inf)))
        elif gate == "directional_risk":
            risk_on = _clean_series(frame, "risk_on_score")
            risk_off = _clean_series(frame, "risk_off_score")
            active &= (direction.gt(0) & risk_on.ge(float(gates.get("risk_on_min", -np.inf)))) | (
                direction.lt(0) & risk_off.ge(float(gates.get("risk_off_min", -np.inf)))
            )
        elif gate == "cross_asset_breadth":
            index_count = _clean_series(frame, "positive_index_count_6")
            sector_count = _clean_series(frame, "positive_sector_count_6")
            active &= (
                direction.gt(0)
                & index_count.ge(float(gates.get("positive_index_high", np.inf)))
                & sector_count.ge(float(gates.get("positive_sector_high", np.inf)))
            ) | (
                direction.lt(0)
                & index_count.le(float(gates.get("positive_index_low", -np.inf)))
                & sector_count.le(float(gates.get("positive_sector_low", -np.inf)))
            )
        else:
            raise ValueError(f"Unsupported alpha gate: {gate}")
    return position.where(active, 0.0).astype(float)


def selected_hours(frame: pd.DataFrame, position: pd.Series) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in frame.loc[position.abs() > 0, "hour"].dropna().unique().tolist()))


def add_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = ["candidate_id", "split", "cost_scenario"]
    signal = rows[rows["bucket"].eq("alpha_signal")].copy()
    for control_bucket, suffix in [("base_control", "base"), ("same_hour_control", "same_hour"), ("always_flat", "flat")]:
        control = rows[rows["bucket"].eq(control_bucket)].loc[:, [*keys, "net_return", "daily_sharpe", "max_drawdown", "turnover"]].rename(
            columns={metric: f"{suffix}_{metric}" for metric in ["net_return", "daily_sharpe", "max_drawdown", "turnover"]}
        )
        signal = signal.merge(control, on=keys, how="left", validate="one_to_one")
    signal["net_delta_vs_base"] = signal["net_return"] - signal["base_net_return"]
    signal["net_delta_vs_same_hour"] = signal["net_return"] - signal["same_hour_net_return"]
    signal["daily_sharpe_delta_vs_base"] = signal["daily_sharpe"] - signal["base_daily_sharpe"]
    signal["daily_sharpe_delta_vs_same_hour"] = signal["daily_sharpe"] - signal["same_hour_daily_sharpe"]
    signal["drawdown_reduction_vs_base"] = signal["base_max_drawdown"] - signal["max_drawdown"]
    signal["drawdown_reduction_vs_same_hour"] = signal["same_hour_max_drawdown"] - signal["max_drawdown"]
    signal["turnover_reduction_vs_base"] = signal["base_turnover"] - signal["turnover"]
    delta_cols = [
        "net_delta_vs_base",
        "net_delta_vs_same_hour",
        "daily_sharpe_delta_vs_base",
        "daily_sharpe_delta_vs_same_hour",
        "drawdown_reduction_vs_base",
        "drawdown_reduction_vs_same_hour",
        "turnover_reduction_vs_base",
    ]
    return rows.merge(signal.loc[:, [*keys, *delta_cols]], on=keys, how="left", validate="many_to_one")


def evaluate_candidate(
    frame: pd.DataFrame,
    split: str,
    fold: int,
    alpha_variant: str,
    horizon: int,
    threshold: float,
    scenario: dict[str, Any],
    gates: dict[str, float],
    hours: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    position = alpha_position(frame, alpha_variant, threshold, gates)
    base_variant = str(ALPHA_SPECS[alpha_variant].get("base_variant", alpha_variant))
    base_position = alpha_position(frame, base_variant, threshold, gates)
    active_hours = hours if hours is not None else selected_hours(frame, position)
    hour_mult = frame["hour"].isin(active_hours).astype(float) if active_hours else pd.Series(0.0, index=frame.index)
    gates_json = _json_dumps(gates)
    rows = []
    for bucket, bucket_position in (
        ("alpha_signal", position),
        ("base_control", base_position),
        ("same_hour_control", base_position * hour_mult),
        ("always_flat", pd.Series(0.0, index=frame.index)),
    ):
        row = {
            "fold": int(fold),
            "split": split,
            "alpha_variant": alpha_variant,
            "base_variant": base_variant,
            "bucket": bucket,
            "horizon_bars": int(horizon),
            "threshold": float(threshold),
            "selected_hours": ",".join(str(hour) for hour in active_hours),
            "gates_json": gates_json,
            "cost_scenario": scenario["cost_scenario"],
            "cost_kind": scenario["cost_kind"],
            "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
            "notional_usd": float(scenario.get("notional_usd", np.nan)),
        }
        row["candidate_id"] = _candidate_key(row)
        rows.append({**row, **evaluate_position_with_cost(frame, bucket_position, scenario)})
    return add_deltas(pd.DataFrame(rows))


def classify_validation_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _combined_cfg(config)
    if row["bucket"] != "alpha_signal":
        return "control"
    if int(row["trades"]) < int(cfg.get("min_trades", 30)):
        return "rejected_insufficient_trades"
    if float(row["turnover"]) > float(cfg.get("max_turnover", 4.0)):
        return "rejected_high_turnover"
    if float(row["net_return"]) <= 0 or float(row["avg_trade_net"]) <= 0:
        return "rejected_negative_edge"
    if float(row["profit_factor"]) < float(cfg.get("min_profit_factor", 1.10)):
        return "rejected_weak_profit_factor"
    if float(row["daily_sharpe"]) < float(cfg.get("min_daily_sharpe", 1.0)):
        return "rejected_weak_sharpe"
    if float(row["max_drawdown"]) > float(cfg.get("max_drawdown", 0.12)):
        return "rejected_drawdown"
    if float(row["top_day_abs_net_share"]) > float(cfg.get("max_top_day_abs_net_share", 0.35)):
        return "rejected_concentrated"
    if row["alpha_variant"] != row["base_variant"]:
        if float(row["net_delta_vs_base"]) <= 0 and float(row["drawdown_reduction_vs_base"]) <= 0:
            return "rejected_no_base_improvement"
        if float(row["net_delta_vs_same_hour"]) <= 0 and float(row["drawdown_reduction_vs_same_hour"]) <= 0:
            return "rejected_no_same_hour_edge"
    return "alpha_base_validation_candidate"


def add_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["candidate_status"] = rows.apply(lambda row: classify_validation_row(row, config), axis=1)
    return rows


def validation_grid(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _discovery_cfg(config)
    variants = [str(value) for value in cfg.get("variants", list(ALPHA_SPECS))]
    quantiles = [float(value) for value in cfg.get("threshold_quantiles", [0.75, 0.85, 0.95])]
    primary = str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": _combined_cfg(config)}, [primary])
    rows = []
    validation = dataset[dataset["split"].eq(str(cfg.get("candidate_split", "validation")))].copy()
    for (fold, horizon), frame in validation.groupby(["fold", "horizon_bars"], sort=False):
        gates = frozen_gates(frame, config)
        for variant in variants:
            if variant not in ALPHA_SPECS:
                continue
            for threshold in thresholds_for_variant(frame, variant, quantiles):
                for scenario in scenarios:
                    rows.append(evaluate_candidate(frame, "validation", int(fold), variant, int(horizon), threshold, scenario, gates))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(result, config) if not result.empty else result


def select_specs(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    candidates = validation[
        validation["bucket"].eq("alpha_signal")
        & validation["cost_scenario"].eq(primary)
        & validation["candidate_status"].eq("alpha_base_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("alpha_signal") & validation["cost_scenario"].eq(primary)].copy()
    candidates["utility_score"] = (
        candidates["daily_sharpe"].fillna(0.0)
        + 60.0 * candidates["avg_trade_net"].fillna(0.0)
        + 0.4 * candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        + candidates["drawdown_reduction_vs_base"].fillna(0.0)
        + candidates["net_delta_vs_base"].fillna(0.0)
        - 0.04 * candidates["turnover"].fillna(0.0)
    )
    candidates = candidates.sort_values(
        ["candidate_status", "utility_score", "net_return", "avg_trade_net"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    candidates = candidates.drop_duplicates(["fold", "alpha_variant", "horizon_bars", "threshold"])
    cols = [
        "candidate_id",
        "fold",
        "alpha_variant",
        "base_variant",
        "horizon_bars",
        "threshold",
        "selected_hours",
        "gates_json",
        "candidate_status",
        "utility_score",
    ]
    return candidates.loc[:, cols].head(int(cfg.get("max_selected", 80))).reset_index(drop=True)


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
        hours = tuple(int(value) for value in str(spec["selected_hours"]).split(",") if value != "")
        gates = _json_loads(spec.get("gates_json", "{}"))
        for scenario in scenarios:
            rows.append(
                evaluate_candidate(
                    frame,
                    split,
                    int(spec["fold"]),
                    str(spec["alpha_variant"]),
                    int(spec["horizon_bars"]),
                    float(spec["threshold"]),
                    scenario,
                    gates,
                    hours,
                )
            )
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(result, config) if not result.empty else result


def decision_table(validation: pd.DataFrame, test: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    rows = []
    for _, spec in specs.iterrows():
        candidate_id = spec["candidate_id"]
        val = validation[validation["candidate_id"].eq(candidate_id) & validation["bucket"].eq("alpha_signal") & validation["cost_scenario"].eq(primary)]
        tst_primary = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("alpha_signal") & test["cost_scenario"].eq(primary)]
        tst_conservative = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("alpha_signal") & test["cost_scenario"].eq(conservative)]
        tst_stress = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("alpha_signal") & test["cost_scenario"].eq(stress)]
        val_row = val.iloc[0] if not val.empty else pd.Series(dtype=object)
        primary_row = tst_primary.iloc[0] if not tst_primary.empty else pd.Series(dtype=object)
        conservative_row = tst_conservative.iloc[0] if not tst_conservative.empty else pd.Series(dtype=object)
        stress_row = tst_stress.iloc[0] if not tst_stress.empty else pd.Series(dtype=object)
        incrementality_ok = bool(
            spec["alpha_variant"] == spec["base_variant"]
            or (
                not primary_row.empty
                and primary_row.get("net_delta_vs_base", -np.inf) > 0
                and primary_row.get("net_delta_vs_same_hour", -np.inf) > 0
            )
        )
        primary_ok = bool(
            not primary_row.empty
            and int(primary_row["trades"]) >= int(cfg.get("min_trades", 30))
            and primary_row["net_return"] > 0
            and primary_row["avg_trade_net"] > 0
            and primary_row["profit_factor"] >= float(cfg.get("min_profit_factor", 1.10))
            and primary_row["daily_sharpe"] >= float(cfg.get("min_daily_sharpe", 1.0))
            and primary_row["max_drawdown"] <= float(cfg.get("max_drawdown", 0.12))
            and primary_row["top_day_abs_net_share"] <= float(cfg.get("max_top_day_abs_net_share", 0.35))
            and primary_row["turnover"] <= float(cfg.get("max_turnover", 4.0))
            and incrementality_ok
        )
        conservative_ok = bool(
            not conservative_row.empty
            and int(conservative_row["trades"]) >= int(cfg.get("min_trades", 30))
            and conservative_row["net_return"] > 0
            and conservative_row["avg_trade_net"] > 0
            and conservative_row["profit_factor"] >= float(cfg.get("min_profit_factor", 1.10))
            and conservative_row["max_drawdown"] <= float(cfg.get("max_drawdown", 0.12))
            and conservative_row["top_day_abs_net_share"] <= float(cfg.get("max_top_day_abs_net_share", 0.35))
        )
        if primary_ok and conservative_ok:
            decision = "accepted_candidate"
        elif primary_ok and not conservative_ok:
            decision = "cost_fragile"
        elif not primary_row.empty and primary_row["net_return"] > 0:
            decision = "research_candidate"
        else:
            decision = "rejected"
        rows.append(
            {
                **{col: spec[col] for col in ["candidate_id", "fold", "alpha_variant", "base_variant", "horizon_bars", "threshold"]},
                "validation_status": val_row.get("candidate_status", ""),
                "decision": decision,
                "incrementality_ok": incrementality_ok,
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_sharpe_primary": primary_row.get("daily_sharpe", np.nan),
                "test_profit_factor_primary": primary_row.get("profit_factor", np.nan),
                "test_avg_trade_net_primary": primary_row.get("avg_trade_net", np.nan),
                "test_trades_primary": primary_row.get("trades", np.nan),
                "test_turnover_primary": primary_row.get("turnover", np.nan),
                "test_top_day_abs_net_share_primary": primary_row.get("top_day_abs_net_share", np.nan),
                "test_net_delta_vs_base_primary": primary_row.get("net_delta_vs_base", np.nan),
                "test_net_delta_vs_same_hour_primary": primary_row.get("net_delta_vs_same_hour", np.nan),
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
    cfg = _combined_cfg(config)
    val_alpha = validation[validation["bucket"].eq("alpha_signal")] if not validation.empty else pd.DataFrame()
    test_alpha = test[test["bucket"].eq("alpha_signal")] if not test.empty else pd.DataFrame()
    val_counts = val_alpha["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="rows") if not val_alpha.empty else pd.DataFrame()
    decision_counts = decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    family_summary = (
        test_alpha.groupby(["cost_scenario", "alpha_variant", "horizon_bars"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            median_net_return=("net_return", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_trades=("trades", "median"),
            median_net_delta_vs_base=("net_delta_vs_base", "median"),
            median_net_delta_vs_same_hour=("net_delta_vs_same_hour", "median"),
        )
        .sort_values(["cost_scenario", "median_net_return"], ascending=[True, False], kind="stable")
        if not test_alpha.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one base alpha candidate passes the configured IBKR-aware gates."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No base alpha candidate is accepted under the configured IBKR-aware gates."
    )
    return f"""# Alpha Discovery Base - {target_symbol.upper()}

## Scope

- HMM states are not used for selection or filtering.
- Splits are rebuilt from the existing walk-forward configuration.
- Variants: `{cfg.get("variants", list(ALPHA_SPECS))}`
- Horizons: `{cfg.get("horizons", [12, 24])}`
- Threshold quantiles: `{cfg.get("threshold_quantiles", [0.75, 0.85, 0.95])}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Selection uses validation only; test is applied after freezing thresholds, gates, and active hours.

## Validation Status Counts

{_markdown_table(val_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Family Test Summary

{_markdown_table(family_summary, max_rows=int(cfg.get("report_top_rows", 80)))}

## Top Decisions

{_markdown_table(decisions.head(int(cfg.get("report_top_rows", 80))) if not decisions.empty else decisions, max_rows=int(cfg.get("report_top_rows", 80)))}

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
    test = evaluate_selected_on_split(dataset, specs, str(_discovery_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()
    decisions = decision_table(selected_validation, test, specs, config)
    outputs = {
        "alpha_discovery_validation": results_dir / "alpha_discovery_validation.parquet",
        "alpha_discovery_selected_validation": results_dir / "alpha_discovery_selected_validation.parquet",
        "alpha_discovery_test": results_dir / "alpha_discovery_test.parquet",
        "alpha_discovery_selected_specs": results_dir / "alpha_discovery_selected_specs.parquet",
        "alpha_discovery_decisions": results_dir / "alpha_discovery_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["alpha_discovery_validation"], index=False)
    selected_validation.to_parquet(outputs["alpha_discovery_selected_validation"], index=False)
    test.to_parquet(outputs["alpha_discovery_test"], index=False)
    specs.to_parquet(outputs["alpha_discovery_selected_specs"], index=False)
    decisions.to_parquet(outputs["alpha_discovery_decisions"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, validation, test, specs, decisions, outputs), encoding="utf-8")
    return report_path, outputs["alpha_discovery_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover base alpha candidates without HMM regime filters.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Alpha discovery report written to: {report_path}")
    print(f"Alpha discovery decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import evaluate_position_with_cost
from src.hmm_lab import _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_risk_filter import filter_multiplier, same_hour_multiplier, split_combo_frame
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import (
    _copy_config_with_horizons,
    _path_from_template,
    available_cost_scenarios,
    build_search_dataset,
    classify_validation_row,
    load_search_combos,
)


ALPHA_VARIANTS: dict[str, tuple[str, ...]] = {
    "m6_base": (),
    "m6_ret12_confirm": ("ret12_confirm",),
    "m6_vwap_confirm": ("vwap_confirm",),
    "m6_efficiency_confirm": ("efficiency_confirm",),
    "m6_ret12_vwap": ("ret12_confirm", "vwap_confirm"),
    "m6_ret12_low_chop": ("ret12_confirm", "low_chop"),
    "m6_ret12_low_stress": ("ret12_confirm", "low_stress"),
    "m6_ret12_directional_risk": ("ret12_confirm", "directional_risk"),
    "m6_ret12_cross_asset_breadth": ("ret12_confirm", "cross_asset_breadth"),
    "m6_ret12_vwap_low_stress": ("ret12_confirm", "vwap_confirm", "low_stress"),
    "m6_ret12_vwap_directional_risk": ("ret12_confirm", "vwap_confirm", "directional_risk"),
    "m6_ret12_vwap_breadth": ("ret12_confirm", "vwap_confirm", "cross_asset_breadth"),
}


EXTRA_FEATURE_COLS = [
    "source_index",
    "target_signed_efficiency_12",
    "target_dir_persistence_12",
    "target_intraday_runup",
    "target_intraday_drawdown",
    "positive_index_count_6",
    "positive_sector_count_6",
    "sector_above_vwap_count",
    "sector_rel_strength_count_12",
    "market_range_ratio_6_24",
    "cross_asset_vol_expansion_score",
    "defensive_rotation_score",
    "narrow_rally_score",
]


def _alpha_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("operable_alpha_refinement", {})


def _combined_cfg(config: dict[str, Any]) -> dict[str, Any]:
    combined = dict(config.get("operable_candidate_search", {}))
    combined.update(_alpha_cfg(config))
    return combined


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "operable_alpha_refinement.md"


def _candidate_key(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}__fold{int(row['fold'])}"
        f"__{row['alpha_variant']}__{row['filter_name']}__h{int(row['horizon_bars'])}__thr{float(row['threshold']):g}"
    )


def _json_dumps(values: dict[str, float]) -> str:
    return json.dumps({key: float(value) for key, value in sorted(values.items()) if np.isfinite(float(value))}, sort_keys=True)


def _json_loads(value: str | float | int | None) -> dict[str, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    raw = json.loads(str(value))
    return {str(key): float(item) for key, item in raw.items()}


def load_refinement_combos(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _alpha_cfg(config)
    triage_path = _path_from_template(str(cfg.get("triage_source", "results/{target_symbol}/operable_candidate_triage.parquet")), target_symbol)
    combo_cols = ["feature_set", "n_states", "seed", "fold"]
    if triage_path.exists():
        triage = pd.read_parquet(triage_path)
        focus_actions = {str(value) for value in cfg.get("focus_next_actions", ["promote_family_to_feature_refinement"])}
        focus_strategies = {str(value) for value in cfg.get("focus_strategies", ["momentum_ret_6"])}
        selected = triage[triage["next_action"].isin(focus_actions) & triage["strategy"].isin(focus_strategies)].copy()
        if not selected.empty:
            selected = selected.sort_values(["failure_count", "research_score"], ascending=[True, False], kind="stable")
            combos = selected.loc[:, combo_cols].drop_duplicates()
        else:
            combos = pd.DataFrame(columns=combo_cols)
    else:
        combos = pd.DataFrame(columns=combo_cols)

    if combos.empty:
        combos = load_search_combos(config, target_symbol)

    max_combos = int(cfg.get("max_combos", 12))
    return combos.head(max_combos).reset_index(drop=True)


def enrich_alpha_features(merged: pd.DataFrame, config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    if merged.empty:
        return merged
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config)).reset_index(names="source_index")
    available = [col for col in EXTRA_FEATURE_COLS if col in features.columns and (col == "source_index" or col not in merged.columns)]
    if len(available) <= 1:
        return merged
    return merged.merge(features.loc[:, available], on="source_index", how="left", validate="many_to_one")


def build_alpha_dataset(config: dict[str, Any], target_symbol: str, combos: pd.DataFrame) -> pd.DataFrame:
    cfg = _alpha_cfg(config)
    horizons = [int(value) for value in cfg.get("horizons", [24])]
    build_config = dict(config)
    build_config["operable_candidate_search"] = dict(config.get("operable_candidate_search", {}))
    build_config["operable_candidate_search"]["horizons"] = horizons
    build_config = _copy_config_with_horizons(build_config, horizons)
    return enrich_alpha_features(build_search_dataset(build_config, target_symbol, combos), config, target_symbol)


def _clean_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[column].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def alpha_thresholds(frame: pd.DataFrame, quantiles: list[float]) -> list[float]:
    values = _clean_series(frame, "target_ret_6").abs()
    values = values[values > 0]
    if values.empty:
        return []
    thresholds = [float(values.quantile(float(q))) for q in quantiles]
    return sorted(set(round(value, 12) for value in thresholds if np.isfinite(value) and value >= 0))


def confirmation_gates(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    cfg = _alpha_cfg(config)

    def quantile(column: str, q: float, default: float) -> float:
        values = _clean_series(frame, column, np.nan).dropna()
        return float(values.quantile(q)) if not values.empty else default

    def abs_quantile(column: str, q: float, default: float) -> float:
        values = _clean_series(frame, column, np.nan).dropna().abs()
        return float(values.quantile(q)) if not values.empty else default

    breadth_high_q = float(cfg.get("breadth_high_quantile", 0.65))
    return {
        "abs_efficiency_min": abs_quantile("target_signed_efficiency_12", float(cfg.get("efficiency_abs_quantile", 0.50)), 0.0),
        "chop_score_max": quantile("chop_score", float(cfg.get("chop_max_quantile", 0.70)), np.inf),
        "stress_score_max": quantile("intraday_stress_score", float(cfg.get("stress_max_quantile", 0.70)), np.inf),
        "risk_on_min": quantile("risk_on_score", float(cfg.get("risk_score_quantile", 0.55)), -np.inf),
        "risk_off_min": quantile("risk_off_score", float(cfg.get("risk_score_quantile", 0.55)), -np.inf),
        "positive_index_high": quantile("positive_index_count_6", breadth_high_q, np.inf),
        "positive_index_low": quantile("positive_index_count_6", 1.0 - breadth_high_q, -np.inf),
        "positive_sector_high": quantile("positive_sector_count_6", breadth_high_q, np.inf),
        "positive_sector_low": quantile("positive_sector_count_6", 1.0 - breadth_high_q, -np.inf),
    }


def alpha_position(frame: pd.DataFrame, alpha_variant: str, threshold: float, gates: dict[str, float] | None = None) -> pd.Series:
    if alpha_variant not in ALPHA_VARIANTS:
        raise ValueError(f"Unsupported alpha variant: {alpha_variant}")
    gates = gates or {}
    ret6 = _clean_series(frame, "target_ret_6")
    direction = pd.Series(0.0, index=frame.index)
    direction.loc[ret6 > float(threshold)] = 1.0
    direction.loc[ret6 < -float(threshold)] = -1.0
    active = direction.ne(0.0)

    checks = ALPHA_VARIANTS[alpha_variant]
    if "ret12_confirm" in checks:
        ret12 = _clean_series(frame, "target_ret_12")
        active &= np.sign(ret12).eq(direction) & ret12.ne(0.0)
    if "vwap_confirm" in checks:
        dist_vwap = _clean_series(frame, "target_dist_vwap_atr")
        active &= np.sign(dist_vwap).eq(direction) & dist_vwap.ne(0.0)
    if "efficiency_confirm" in checks:
        efficiency = _clean_series(frame, "target_signed_efficiency_12")
        active &= np.sign(efficiency).eq(direction) & efficiency.abs().ge(float(gates.get("abs_efficiency_min", 0.0)))
    if "low_chop" in checks:
        active &= _clean_series(frame, "chop_score").le(float(gates.get("chop_score_max", np.inf)))
    if "low_stress" in checks:
        active &= _clean_series(frame, "intraday_stress_score").le(float(gates.get("stress_score_max", np.inf)))
    if "directional_risk" in checks:
        risk_on = _clean_series(frame, "risk_on_score")
        risk_off = _clean_series(frame, "risk_off_score")
        long_ok = direction.gt(0) & risk_on.ge(float(gates.get("risk_on_min", -np.inf)))
        short_ok = direction.lt(0) & risk_off.ge(float(gates.get("risk_off_min", -np.inf)))
        active &= long_ok | short_ok
    if "cross_asset_breadth" in checks:
        index_count = _clean_series(frame, "positive_index_count_6")
        sector_count = _clean_series(frame, "positive_sector_count_6")
        long_ok = direction.gt(0) & index_count.ge(float(gates.get("positive_index_high", np.inf))) & sector_count.ge(float(gates.get("positive_sector_high", np.inf)))
        short_ok = direction.lt(0) & index_count.le(float(gates.get("positive_index_low", -np.inf))) & sector_count.le(float(gates.get("positive_sector_low", -np.inf)))
        active &= long_ok | short_ok

    return direction.where(active, 0.0).astype(float)


def _selected_hours(frame: pd.DataFrame, multiplier: pd.Series) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in frame.loc[multiplier > 0, "hour"].dropna().unique().tolist()))


def add_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = ["candidate_id", "split", "cost_scenario"]
    hmm = rows[rows["bucket"].eq("hmm_filter")].copy()
    for control_bucket, suffix in [("base_no_hmm", "base"), ("same_hour_control", "same_hour"), ("always_flat", "flat")]:
        control = rows[rows["bucket"].eq(control_bucket)].loc[:, [*keys, "net_return", "daily_sharpe", "max_drawdown", "turnover"]].rename(
            columns={metric: f"{suffix}_{metric}" for metric in ["net_return", "daily_sharpe", "max_drawdown", "turnover"]}
        )
        hmm = hmm.merge(control, on=keys, how="left", validate="one_to_one")
    hmm["net_delta_vs_base"] = hmm["net_return"] - hmm["base_net_return"]
    hmm["net_delta_vs_same_hour"] = hmm["net_return"] - hmm["same_hour_net_return"]
    hmm["daily_sharpe_delta_vs_base"] = hmm["daily_sharpe"] - hmm["base_daily_sharpe"]
    hmm["daily_sharpe_delta_vs_same_hour"] = hmm["daily_sharpe"] - hmm["same_hour_daily_sharpe"]
    hmm["drawdown_reduction_vs_base"] = hmm["base_max_drawdown"] - hmm["max_drawdown"]
    hmm["drawdown_reduction_vs_same_hour"] = hmm["same_hour_max_drawdown"] - hmm["max_drawdown"]
    hmm["turnover_reduction_vs_base"] = hmm["base_turnover"] - hmm["turnover"]
    delta_cols = [
        "net_delta_vs_base",
        "net_delta_vs_same_hour",
        "daily_sharpe_delta_vs_base",
        "daily_sharpe_delta_vs_same_hour",
        "drawdown_reduction_vs_base",
        "drawdown_reduction_vs_same_hour",
        "turnover_reduction_vs_base",
    ]
    return rows.merge(hmm.loc[:, [*keys, *delta_cols]], on=keys, how="left", validate="many_to_one")


def evaluate_alpha_candidate(
    frame: pd.DataFrame,
    combo: pd.Series,
    split: str,
    alpha_variant: str,
    filter_name: str,
    horizon: int,
    threshold: float,
    scenario: dict[str, Any],
    gates: dict[str, float],
    selected_hours: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    base = alpha_position(frame, alpha_variant, threshold, gates)
    hmm_mult = filter_multiplier(frame, filter_name)
    hours = selected_hours if selected_hours is not None else _selected_hours(frame, hmm_mult)
    hour_mult = same_hour_multiplier(frame, hours)
    gates_json = _json_dumps(gates)
    rows = []
    for bucket, position in (
        ("base_no_hmm", base),
        ("hmm_filter", base * hmm_mult),
        ("same_hour_control", base * hour_mult),
        ("always_flat", pd.Series(0.0, index=frame.index)),
    ):
        row = {
            "feature_set": combo["feature_set"],
            "n_states": int(combo["n_states"]),
            "seed": int(combo["seed"]),
            "fold": int(combo["fold"]),
            "split": split,
            "alpha_variant": alpha_variant,
            "strategy": alpha_variant,
            "filter_name": filter_name,
            "bucket": bucket,
            "horizon_bars": int(horizon),
            "threshold": float(threshold),
            "selected_hours": ",".join(str(hour) for hour in hours),
            "gates_json": gates_json,
            "cost_scenario": scenario["cost_scenario"],
            "cost_kind": scenario["cost_kind"],
            "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
            "notional_usd": float(scenario.get("notional_usd", np.nan)),
        }
        row["candidate_id"] = _candidate_key(row)
        rows.append({**row, **evaluate_position_with_cost(frame, position, scenario)})
    return add_deltas(pd.DataFrame(rows))


def classify_alpha_validation_row(row: pd.Series, config: dict[str, Any]) -> str:
    wrapped = {"operable_candidate_search": _combined_cfg(config)}
    status = classify_validation_row(row, wrapped)
    return "alpha_validation_candidate" if status == "operable_validation_candidate" else status


def add_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["candidate_status"] = rows.apply(lambda row: classify_alpha_validation_row(row, config), axis=1)
    return rows


def validation_grid(merged: pd.DataFrame, combos: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _alpha_cfg(config)
    split = str(cfg.get("candidate_split", "validation"))
    horizons = [int(value) for value in cfg.get("horizons", [24])]
    variants = [str(value) for value in cfg.get("variants", list(ALPHA_VARIANTS))]
    filters = [str(value) for value in cfg.get("filters", ["only_risk_on"])]
    quantiles = [float(value) for value in cfg.get("threshold_quantiles", [0.80, 0.90, 0.95])]
    primary = str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": _combined_cfg(config)}, [primary])
    rows = []
    for _, combo in combos.iterrows():
        for horizon in horizons:
            frame = split_combo_frame(merged, combo, split, horizon)
            if frame.empty:
                continue
            gates = confirmation_gates(frame, config)
            for threshold in alpha_thresholds(frame, quantiles):
                for variant in variants:
                    if variant not in ALPHA_VARIANTS:
                        continue
                    for filter_name in filters:
                        for scenario in scenarios:
                            rows.append(evaluate_alpha_candidate(frame, combo, split, variant, filter_name, horizon, threshold, scenario, gates))
    grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(grid, config) if not grid.empty else grid


def select_specs(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    candidates = validation[
        validation["bucket"].eq("hmm_filter")
        & validation["cost_scenario"].eq(primary)
        & validation["candidate_status"].eq("alpha_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("hmm_filter") & validation["cost_scenario"].eq(primary)].copy()
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
    candidates = candidates.drop_duplicates(["feature_set", "n_states", "seed", "fold", "alpha_variant", "filter_name", "horizon_bars", "threshold"])
    max_selected = int(cfg.get("max_selected", 80))
    cols = [
        "candidate_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "alpha_variant",
        "strategy",
        "filter_name",
        "horizon_bars",
        "threshold",
        "selected_hours",
        "gates_json",
        "candidate_status",
        "utility_score",
    ]
    return candidates.loc[:, cols].head(max_selected).reset_index(drop=True)


def evaluate_selected_on_split(merged: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg})
    rows = []
    for _, spec in specs.iterrows():
        frame = split_combo_frame(merged, spec, split, int(spec["horizon_bars"]))
        if frame.empty:
            continue
        hours = tuple(int(value) for value in str(spec["selected_hours"]).split(",") if value != "")
        gates = _json_loads(spec.get("gates_json", "{}"))
        for scenario in scenarios:
            rows.append(
                evaluate_alpha_candidate(
                    frame,
                    spec,
                    split,
                    str(spec["alpha_variant"]),
                    str(spec["filter_name"]),
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
        val = validation[validation["candidate_id"].eq(candidate_id) & validation["bucket"].eq("hmm_filter") & validation["cost_scenario"].eq(primary)]
        tst_primary = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("hmm_filter") & test["cost_scenario"].eq(primary)]
        tst_conservative = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("hmm_filter") & test["cost_scenario"].eq(conservative)]
        tst_stress = test[test["candidate_id"].eq(candidate_id) & test["bucket"].eq("hmm_filter") & test["cost_scenario"].eq(stress)]
        val_row = val.iloc[0] if not val.empty else pd.Series(dtype=object)
        primary_row = tst_primary.iloc[0] if not tst_primary.empty else pd.Series(dtype=object)
        conservative_row = tst_conservative.iloc[0] if not tst_conservative.empty else pd.Series(dtype=object)
        stress_row = tst_stress.iloc[0] if not tst_stress.empty else pd.Series(dtype=object)
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
                **{
                    col: spec[col]
                    for col in [
                        "candidate_id",
                        "feature_set",
                        "n_states",
                        "seed",
                        "fold",
                        "alpha_variant",
                        "filter_name",
                        "horizon_bars",
                        "threshold",
                    ]
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
    combos: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    specs: pd.DataFrame,
    decisions: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _combined_cfg(config)
    val_hmm = validation[validation["bucket"].eq("hmm_filter")] if not validation.empty else pd.DataFrame()
    test_hmm = test[test["bucket"].eq("hmm_filter")] if not test.empty else pd.DataFrame()
    val_counts = val_hmm["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="rows") if not val_hmm.empty else pd.DataFrame()
    decision_counts = decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    family_summary = (
        test_hmm.groupby(["cost_scenario", "alpha_variant", "filter_name"], as_index=False)
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
        if not test_hmm.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one refined alpha candidate passes the configured IBKR-aware gates."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No refined alpha candidate is accepted under the configured IBKR-aware gates."
    )
    return f"""# Operable Alpha Refinement - {target_symbol.upper()}

## Scope

- Focus family: `momentum_ret_6` around `only_risk_on`, horizon 24.
- HMMs are reused; no HMM retraining is performed.
- Combos evaluated: `{len(combos)}`
- Variants: `{cfg.get("variants", list(ALPHA_VARIANTS))}`
- Filters: `{cfg.get("filters", ["only_risk_on"])}`
- Horizons: `{cfg.get("horizons", [24])}`
- Threshold quantiles: `{cfg.get("threshold_quantiles", [0.80, 0.90, 0.95])}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Selection uses validation only; test is applied after freezing selected candidates and confirmation gates.

## Validation Status Counts

{_markdown_table(val_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Family Test Summary

{_markdown_table(family_summary, max_rows=int(cfg.get("report_top_rows", 60)))}

## Top Decisions

{_markdown_table(decisions.head(int(cfg.get("report_top_rows", 60))) if not decisions.empty else decisions, max_rows=int(cfg.get("report_top_rows", 60)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    combos = load_refinement_combos(config, target)
    merged = build_alpha_dataset(config, target, combos) if not combos.empty else pd.DataFrame()
    validation = validation_grid(merged, combos, config) if not merged.empty else pd.DataFrame()
    specs = select_specs(validation, config)
    selected_validation = validation[validation["candidate_id"].isin(specs["candidate_id"])] if not specs.empty else pd.DataFrame()
    test = evaluate_selected_on_split(merged, specs, str(_alpha_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()
    decisions = decision_table(selected_validation, test, specs, config)
    outputs = {
        "operable_alpha_validation": results_dir / "operable_alpha_validation.parquet",
        "operable_alpha_selected_validation": results_dir / "operable_alpha_selected_validation.parquet",
        "operable_alpha_test": results_dir / "operable_alpha_test.parquet",
        "operable_alpha_selected_specs": results_dir / "operable_alpha_selected_specs.parquet",
        "operable_alpha_decisions": results_dir / "operable_alpha_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["operable_alpha_validation"], index=False)
    selected_validation.to_parquet(outputs["operable_alpha_selected_validation"], index=False)
    test.to_parquet(outputs["operable_alpha_test"], index=False)
    specs.to_parquet(outputs["operable_alpha_selected_specs"], index=False)
    decisions.to_parquet(outputs["operable_alpha_decisions"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, combos, validation, test, specs, decisions, outputs), encoding="utf-8")
    return report_path, outputs["operable_alpha_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine positive operable research alpha families without retraining HMMs.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Operable alpha refinement report written to: {report_path}")
    print(f"Operable alpha refinement decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

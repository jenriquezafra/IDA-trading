from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, evaluate_position_with_cost
from src.hmm_lab import _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_risk_filter import build_filter_dataset, filter_multiplier, same_hour_multiplier, split_combo_frame
from src.hmm_state_interpretability_cross_asset import _markdown_table


SIGNAL_SPECS = {
    "momentum_ret_6": ("target_ret_6", "signed"),
    "momentum_ret_12": ("target_ret_12", "signed"),
    "reversion_ret_6": ("target_ret_6", "inverse"),
    "reversion_ret_12": ("target_ret_12", "inverse"),
    "vwap_reversion": ("target_dist_vwap_atr", "inverse"),
    "vwap_breakout": ("target_dist_vwap_atr", "signed"),
    "supervised_score": ("supervised_score", "signed"),
    "risk_on_long": ("risk_on_score", "long_positive"),
    "risk_off_short": ("risk_off_score", "short_positive"),
    "stress_short": ("intraday_stress_score", "short_positive"),
}


def _search_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("operable_candidate_search", {})


def _path_from_template(template: str, target_symbol: str) -> Path:
    return Path(template.format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "operable_candidate_search.md"


def _copy_config_with_horizons(config: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    copied = dict(config)
    copied["hmm_risk_filter"] = dict(config.get("hmm_risk_filter", {}))
    copied["hmm_risk_filter"]["horizons"] = horizons
    return copied


def load_search_combos(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _search_cfg(config)
    source = _path_from_template(str(cfg.get("combo_source", "results/{target_symbol}/hmm_feature_lab_cross_asset.parquet")), target_symbol)
    frame = pd.read_parquet(source)
    frame = frame[frame["status"].eq("ok")].copy() if "status" in frame else frame
    feature_sets = cfg.get("feature_sets")
    if feature_sets:
        frame = frame[frame["feature_set"].isin([str(value) for value in feature_sets])].copy()
    combos = frame.loc[:, ["feature_set", "n_states", "seed", "fold"]].drop_duplicates().sort_values(
        ["feature_set", "n_states", "seed", "fold"], kind="stable"
    )
    priority_source = cfg.get("preferred_combo_source")
    if priority_source:
        priority_path = _path_from_template(str(priority_source), target_symbol)
        combo_cols = ["feature_set", "n_states", "seed", "fold"]
        if priority_path.exists():
            preferred = pd.read_parquet(priority_path)
            if set(combo_cols).issubset(preferred.columns):
                preferred = preferred.loc[:, combo_cols].drop_duplicates().reset_index(drop=True)
                preferred["_priority_rank"] = np.arange(len(preferred))
                combos = combos.merge(preferred, on=combo_cols, how="left")
                combos["_priority_group"] = combos["_priority_rank"].isna().astype(int)
                combos["_priority_rank"] = combos["_priority_rank"].fillna(len(preferred))
                combos = combos.sort_values(
                    ["_priority_group", "_priority_rank", "feature_set", "n_states", "seed", "fold"],
                    kind="stable",
                ).drop(columns=["_priority_group", "_priority_rank"])
    max_combos = cfg.get("max_combos")
    if max_combos:
        combos = combos.head(int(max_combos)).copy()
    return combos.reset_index(drop=True)


def enrich_search_dataset(merged: pd.DataFrame, config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config)).reset_index(names="source_index")
    extra_cols = [
        "source_index",
        "target_open_next",
        "target_ret_1",
        "target_ret_3",
        "target_ret_6",
        "target_ret_12",
        "target_ret_24",
        "target_dist_vwap_atr",
        "target_vwap_slope_12",
        "target_range_ratio_6_24",
        "target_pos_session_range",
        "target_dist_open",
        "risk_on_score",
        "risk_off_score",
        "chop_score",
        "intraday_stress_score",
    ]
    available = [col for col in extra_cols if col in features.columns and (col == "source_index" or col not in merged.columns)]
    if len(available) <= 1:
        return merged
    return merged.merge(features.loc[:, available], on="source_index", how="left", validate="many_to_one")


def build_search_dataset(config: dict[str, Any], target_symbol: str, combos: pd.DataFrame) -> pd.DataFrame:
    horizons = [int(value) for value in _search_cfg(config).get("horizons", [6, 12, 24])]
    build_config = _copy_config_with_horizons(config, horizons)
    return enrich_search_dataset(build_filter_dataset(build_config, target_symbol, combos), config, target_symbol)


def available_cost_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    cfg = _search_cfg(config)
    wanted = names if names is not None else [str(value) for value in cfg.get("cost_scenarios", ["bps_2", "ibkr_tiered_10000", "bps_5"])]
    by_name = {str(scenario["cost_scenario"]): scenario for scenario in cost_scenarios(config)}
    return [by_name[name] for name in wanted if name in by_name]


def position_for_signal(frame: pd.DataFrame, strategy: str, threshold: float) -> pd.Series:
    if strategy not in SIGNAL_SPECS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    column, mode = SIGNAL_SPECS[strategy]
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    values = frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    threshold = float(threshold)
    position = pd.Series(0.0, index=frame.index)
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
    else:
        raise ValueError(f"Unsupported signal mode: {mode}")
    return position


def thresholds_for_signal(frame: pd.DataFrame, strategy: str, quantiles: list[float]) -> list[float]:
    if strategy not in SIGNAL_SPECS:
        return []
    column, mode = SIGNAL_SPECS[strategy]
    if column not in frame:
        return []
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if values.empty:
        return []
    source = values.abs() if mode in {"signed", "inverse"} else values[values > 0]
    if source.empty:
        return []
    thresholds = [float(source.quantile(float(q))) for q in quantiles]
    thresholds = [value for value in thresholds if np.isfinite(value) and value >= 0]
    return sorted(set(round(value, 12) for value in thresholds))


def _selected_hours(frame: pd.DataFrame, multiplier: pd.Series) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in frame.loc[multiplier > 0, "hour"].dropna().unique().tolist()))


def _candidate_key(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}__fold{int(row['fold'])}"
        f"__{row['strategy']}__{row['filter_name']}__h{int(row['horizon_bars'])}__thr{float(row['threshold']):g}"
    )


def evaluate_candidate(
    frame: pd.DataFrame,
    combo: pd.Series,
    split: str,
    strategy: str,
    filter_name: str,
    horizon: int,
    threshold: float,
    scenario: dict[str, Any],
    selected_hours: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    base = position_for_signal(frame, strategy, threshold)
    hmm_mult = filter_multiplier(frame, filter_name)
    hours = selected_hours if selected_hours is not None else _selected_hours(frame, hmm_mult)
    hour_mult = same_hour_multiplier(frame, hours)
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
            "strategy": strategy,
            "filter_name": filter_name,
            "bucket": bucket,
            "horizon_bars": int(horizon),
            "threshold": float(threshold),
            "selected_hours": ",".join(str(hour) for hour in hours),
            "cost_scenario": scenario["cost_scenario"],
            "cost_kind": scenario["cost_kind"],
            "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
            "notional_usd": float(scenario.get("notional_usd", np.nan)),
        }
        row["candidate_id"] = _candidate_key(row)
        rows.append({**row, **evaluate_position_with_cost(frame, position, scenario)})
    return add_deltas(pd.DataFrame(rows))


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


def classify_validation_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _search_cfg(config)
    if row["bucket"] != "hmm_filter":
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
    if float(row["net_delta_vs_base"]) <= 0 and float(row["drawdown_reduction_vs_base"]) <= 0:
        return "rejected_no_base_improvement"
    if float(row["net_delta_vs_same_hour"]) <= 0 and float(row["drawdown_reduction_vs_same_hour"]) <= 0:
        return "rejected_no_same_hour_edge"
    return "operable_validation_candidate"


def add_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["candidate_status"] = rows.apply(lambda row: classify_validation_row(row, config), axis=1)
    return rows


def validation_grid(merged: pd.DataFrame, combos: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _search_cfg(config)
    split = str(cfg.get("candidate_split", "validation"))
    horizons = [int(value) for value in cfg.get("horizons", [6, 12, 24])]
    strategies = [str(value) for value in cfg.get("strategies", list(SIGNAL_SPECS))]
    filters = [str(value) for value in cfg.get("filters", ["only_risk_on", "exclude_risk_off", "exclude_stress"])]
    quantiles = [float(value) for value in cfg.get("threshold_quantiles", [0.7, 0.8, 0.9, 0.95])]
    validation_costs = cfg.get("validation_cost_scenarios", [str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))])
    scenarios = available_cost_scenarios(config, [str(value) for value in validation_costs])
    progress_every = int(cfg.get("progress_every_combos", 0))
    rows = []
    for combo_idx, combo in combos.iterrows():
        for horizon in horizons:
            frame = split_combo_frame(merged, combo, split, horizon)
            if frame.empty:
                continue
            for strategy in strategies:
                thresholds = thresholds_for_signal(frame, strategy, quantiles)
                if not thresholds:
                    continue
                for threshold in thresholds:
                    for filter_name in filters:
                        for scenario in scenarios:
                            rows.append(evaluate_candidate(frame, combo, split, strategy, filter_name, horizon, threshold, scenario))
        if progress_every > 0 and (int(combo_idx) + 1) % progress_every == 0:
            print(f"Validated {int(combo_idx) + 1}/{len(combos)} combo specs", flush=True)
    grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(grid, config) if not grid.empty else grid


def select_specs(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = _search_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    candidates = validation[
        validation["bucket"].eq("hmm_filter")
        & validation["cost_scenario"].eq(primary)
        & validation["candidate_status"].eq("operable_validation_candidate")
    ].copy()
    if candidates.empty:
        candidates = validation[validation["bucket"].eq("hmm_filter") & validation["cost_scenario"].eq(primary)].copy()
    candidates["utility_score"] = (
        candidates["daily_sharpe"].fillna(0.0)
        + 50.0 * candidates["avg_trade_net"].fillna(0.0)
        + 0.5 * candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        + candidates["drawdown_reduction_vs_base"].fillna(0.0)
        - 0.05 * candidates["turnover"].fillna(0.0)
    )
    candidates = candidates.sort_values(
        ["candidate_status", "utility_score", "net_return", "avg_trade_net"],
        ascending=[True, False, False, False],
        kind="stable",
    )
    candidates = candidates.drop_duplicates(["feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "threshold"])
    max_selected = int(cfg.get("max_selected", 80))
    cols = [
        "candidate_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "threshold",
        "selected_hours",
        "candidate_status",
        "utility_score",
    ]
    return candidates.loc[:, cols].head(max_selected).reset_index(drop=True)


def evaluate_selected_on_split(merged: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    rows = []
    scenarios = available_cost_scenarios(config)
    for _, spec in specs.iterrows():
        frame = split_combo_frame(merged, spec, split, int(spec["horizon_bars"]))
        if frame.empty:
            continue
        hours = tuple(int(value) for value in str(spec["selected_hours"]).split(",") if value != "")
        for scenario in scenarios:
            rows.append(
                evaluate_candidate(
                    frame,
                    spec,
                    split,
                    str(spec["strategy"]),
                    str(spec["filter_name"]),
                    int(spec["horizon_bars"]),
                    float(spec["threshold"]),
                    scenario,
                    hours,
                )
            )
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_status(result, config) if not result.empty else result


def decision_table(validation: pd.DataFrame, test: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _search_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    rows = []
    for _, spec in specs.iterrows():
        candidate_id = spec["candidate_id"]
        val = validation[
            validation["candidate_id"].eq(candidate_id)
            & validation["bucket"].eq("hmm_filter")
            & validation["cost_scenario"].eq(primary)
        ]
        tst_primary = test[
            test["candidate_id"].eq(candidate_id)
            & test["bucket"].eq("hmm_filter")
            & test["cost_scenario"].eq(primary)
        ]
        tst_conservative = test[
            test["candidate_id"].eq(candidate_id)
            & test["bucket"].eq("hmm_filter")
            & test["cost_scenario"].eq(conservative)
        ]
        val_row = val.iloc[0] if not val.empty else pd.Series(dtype=object)
        primary_row = tst_primary.iloc[0] if not tst_primary.empty else pd.Series(dtype=object)
        conservative_row = tst_conservative.iloc[0] if not tst_conservative.empty else pd.Series(dtype=object)
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
                **{col: spec[col] for col in ["candidate_id", "feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "threshold"]},
                "validation_status": val_row.get("candidate_status", ""),
                "decision": decision,
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_sharpe_primary": primary_row.get("daily_sharpe", np.nan),
                "test_profit_factor_primary": primary_row.get("profit_factor", np.nan),
                "test_avg_trade_net_primary": primary_row.get("avg_trade_net", np.nan),
                "test_turnover_primary": primary_row.get("turnover", np.nan),
                "test_net_conservative": conservative_row.get("net_return", np.nan),
                "test_avg_trade_net_conservative": conservative_row.get("avg_trade_net", np.nan),
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
    cfg = _search_cfg(config)
    val_hmm = validation[validation["bucket"].eq("hmm_filter")] if not validation.empty else pd.DataFrame()
    test_hmm = test[test["bucket"].eq("hmm_filter")] if not test.empty else pd.DataFrame()
    val_counts = val_hmm["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="rows") if not val_hmm.empty else pd.DataFrame()
    decision_counts = decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    test_summary = (
        test_hmm.groupby(["cost_scenario", "strategy", "filter_name"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            median_net_return=("net_return", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_turnover=("turnover", "median"),
        )
        .sort_values(["cost_scenario", "median_net_return"], ascending=[True, False], kind="stable")
        if not test_hmm.empty
        else pd.DataFrame()
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one candidate passes the configured IBKR-aware gates."
        if not decisions.empty and decisions["decision"].eq("accepted_candidate").any()
        else "No selected candidate is accepted under the configured IBKR-aware gates."
    )
    return f"""# Operable Candidate Search - {target_symbol.upper()}

## Scope

- HMMs are reused from the existing lab; no HMM retraining is performed.
- Combos evaluated: `{len(combos)}`
- Horizons: `{cfg.get("horizons", [6, 12, 24])}`
- Strategies: `{cfg.get("strategies", list(SIGNAL_SPECS))}`
- Filters: `{cfg.get("filters", ["only_risk_on", "exclude_risk_off", "exclude_stress"])}`
- Threshold quantiles: `{cfg.get("threshold_quantiles", [0.7, 0.8, 0.9, 0.95])}`
- Cost scenarios: `{cfg.get("cost_scenarios", ["bps_2", "ibkr_tiered_10000", "bps_5"])}`
- Validation cost scenarios: `{cfg.get("validation_cost_scenarios", [cfg.get("primary_cost_scenario", "ibkr_tiered_10000")])}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Selection uses validation only; test is applied after freezing selected candidates.

## Validation Status Counts

{_markdown_table(val_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Selected Test Summary

{_markdown_table(test_summary, max_rows=int(cfg.get("report_top_rows", 60)))}

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
    combos = load_search_combos(config, target)
    merged = build_search_dataset(config, target, combos) if not combos.empty else pd.DataFrame()
    validation = validation_grid(merged, combos, config) if not merged.empty else pd.DataFrame()
    specs = select_specs(validation, config)
    selected_validation = validation[validation["candidate_id"].isin(specs["candidate_id"])] if not specs.empty else pd.DataFrame()
    test = evaluate_selected_on_split(merged, specs, str(_search_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()
    decisions = decision_table(selected_validation, test, specs, config)
    outputs = {
        "operable_candidate_validation": results_dir / "operable_candidate_validation.parquet",
        "operable_candidate_selected_validation": results_dir / "operable_candidate_selected_validation.parquet",
        "operable_candidate_test": results_dir / "operable_candidate_test.parquet",
        "operable_candidate_selected_specs": results_dir / "operable_candidate_selected_specs.parquet",
        "operable_candidate_decisions": results_dir / "operable_candidate_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["operable_candidate_validation"], index=False)
    selected_validation.to_parquet(outputs["operable_candidate_selected_validation"], index=False)
    test.to_parquet(outputs["operable_candidate_test"], index=False)
    specs.to_parquet(outputs["operable_candidate_selected_specs"], index=False)
    decisions.to_parquet(outputs["operable_candidate_decisions"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, combos, validation, test, specs, decisions, outputs), encoding="utf-8")
    return report_path, outputs["operable_candidate_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for lower-turnover operable candidates under IBKR-aware costs.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Operable candidate search report written to: {report_path}")
    print(f"Operable candidate decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

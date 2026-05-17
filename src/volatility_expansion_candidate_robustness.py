from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, evaluate_position_with_cost, scenario_cost_return
from src.excess_reversion_search import same_hour_random_position
from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.volatility_expansion_search import _control_positions, _json_loads, _search_cfg, build_split_dataset


def _robust_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("volatility_expansion_candidate_robustness", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return (
        Path(config.get("paths", {}).get("reports_dir", "reports"))
        / target_symbol.upper()
        / "volatility_expansion_candidate_robustness.md"
    )


def default_input_paths(config: dict[str, Any], target_symbol: str) -> dict[str, Path]:
    cfg = _robust_cfg(config)
    results_dir = results_output_dir(config, target_symbol)
    return {
        "selected_specs": Path(str(cfg.get("selected_specs", results_dir / "volatility_expansion_selected_specs.parquet")).format(target_symbol=target_symbol.upper(), target=target_symbol.upper())),
        "decisions": Path(str(cfg.get("decisions", results_dir / "volatility_expansion_decisions.parquet")).format(target_symbol=target_symbol.upper(), target=target_symbol.upper())),
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _scenario_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    by_name = {str(scenario["cost_scenario"]): scenario for scenario in cost_scenarios(config)}
    if name not in by_name:
        raise ValueError(f"Cost scenario not found: {name}")
    return by_name[name]


def robustness_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = _robust_cfg(config)
    scenarios: list[dict[str, Any]] = []
    for cost_bps in cfg.get("cost_bps_grid", [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]):
        value = float(cost_bps)
        scenarios.append({"cost_scenario": f"bps_{value:g}", "cost_kind": "bps", "cost_bps": value})
    include_ibkr = bool(cfg.get("include_ibkr_scenarios", True))
    if include_ibkr:
        scenarios.extend([scenario for scenario in cost_scenarios(config) if scenario["cost_kind"] == "ibkr"])
    deduped: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        deduped[str(scenario["cost_scenario"])] = scenario
    return list(deduped.values())


def select_frozen_specs(specs: pd.DataFrame, decisions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty or decisions.empty:
        return pd.DataFrame()
    cfg = _robust_cfg(config)
    configured_ids = cfg.get("candidate_ids", cfg.get("candidate_id"))
    if configured_ids:
        if isinstance(configured_ids, str):
            configured_ids = [configured_ids]
        wanted = pd.Index([str(value) for value in configured_ids])
        eligible = decisions[decisions["candidate_id"].isin(wanted)].copy()
    else:
        decision_label = str(cfg.get("candidate_decision", "accepted_candidate"))
        eligible = decisions[decisions["decision"].eq(decision_label)].copy()
    if eligible.empty:
        return pd.DataFrame()
    sort_cols = [column for column in ["test_net_primary", "test_sharpe_primary", "test_avg_trade_net_primary"] if column in eligible.columns]
    if sort_cols:
        eligible = eligible.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    max_candidates = int(cfg.get("max_candidates", 3))
    selected_ids = eligible["candidate_id"].drop_duplicates().head(max_candidates)
    rank = {candidate_id: idx for idx, candidate_id in enumerate(selected_ids)}
    selected = specs[specs["candidate_id"].isin(selected_ids)].copy()
    if selected.empty:
        return selected
    selected["_rank"] = selected["candidate_id"].map(rank).fillna(len(rank)).astype(int)
    metric_cols = [
        "candidate_id",
        "decision",
        "test_net_primary",
        "test_sharpe_primary",
        "test_profit_factor_primary",
        "test_avg_trade_net_primary",
        "test_trades_primary",
        "test_net_conservative",
        "test_net_stress",
        "test_net_delta_vs_random_primary",
        "test_net_delta_vs_breakout_primary",
    ]
    selected = selected.merge(
        eligible.loc[:, [column for column in metric_cols if column in eligible.columns]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    return selected.sort_values("_rank", kind="stable").drop(columns=["_rank"]).reset_index(drop=True)


def _candidate_split_frame(dataset: pd.DataFrame, spec: pd.Series | dict[str, Any], split: str) -> pd.DataFrame:
    return dataset[
        dataset["split"].eq(split)
        & dataset["fold"].eq(int(spec["fold"]))
        & dataset["horizon_bars"].eq(int(spec["horizon_bars"]))
    ].copy()


def _alpha_position(frame: pd.DataFrame, spec: pd.Series | dict[str, Any], config: dict[str, Any], split: str) -> pd.Series:
    spec_dict = dict(spec)
    positions = _control_positions(
        frame,
        spec_dict,
        _json_loads(spec_dict["thresholds_json"]),
        config,
        str(spec_dict["candidate_id"]),
        split,
    )
    return positions["alpha_signal"]


def build_cost_curve(dataset: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    cfg = _robust_cfg(config)
    splits = [str(value) for value in cfg.get("splits", ["validation", "test"])]
    scenarios = robustness_scenarios(config)
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        for split in splits:
            frame = _candidate_split_frame(dataset, spec, split)
            if frame.empty:
                continue
            position = _alpha_position(frame, spec, config, split)
            for scenario in scenarios:
                metadata = {
                    "candidate_id": spec["candidate_id"],
                    "split": split,
                    "fold": int(spec["fold"]),
                    "variant": spec.get("variant", ""),
                    "side": spec.get("side", ""),
                    "horizon_bars": int(spec["horizon_bars"]),
                    "vol_filter_name": spec.get("vol_filter_name", "none"),
                    "hour_filter_name": spec.get("hour_filter_name", "all"),
                    "cost_scenario": scenario["cost_scenario"],
                    "cost_kind": scenario["cost_kind"],
                    "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                    "notional_usd": float(scenario.get("notional_usd", np.nan)),
                }
                rows.append({**metadata, **evaluate_position_with_cost(frame, position, scenario)})
    return pd.DataFrame(rows)


def build_trade_frame(dataset: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    cfg = _robust_cfg(config)
    split_names = [str(value) for value in cfg.get("splits", ["validation", "test"])]
    scenario_names = list(
        dict.fromkeys(
            [
                str(cfg.get("primary_cost_scenario", _search_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))),
                str(cfg.get("stress_cost_scenario", _search_cfg(config).get("stress_cost_scenario", "bps_5"))),
            ]
        )
    )
    scenarios = [_scenario_by_name(config, name) for name in scenario_names]
    rows: list[pd.DataFrame] = []
    for _, spec in specs.iterrows():
        for split in split_names:
            frame = _candidate_split_frame(dataset, spec, split)
            if frame.empty:
                continue
            position = _alpha_position(frame, spec, config, split)
            active = position.abs().gt(0.0)
            if not active.any():
                continue
            gross = position * frame["fwd_ret"].astype(float)
            for scenario in scenarios:
                cost = scenario_cost_return(frame, position, scenario)
                trades = pd.DataFrame(
                    {
                        "candidate_id": spec["candidate_id"],
                        "split": split,
                        "fold": int(spec["fold"]),
                        "variant": spec.get("variant", ""),
                        "side": spec.get("side", ""),
                        "horizon_bars": int(spec["horizon_bars"]),
                        "vol_filter_name": spec.get("vol_filter_name", "none"),
                        "hour_filter_name": spec.get("hour_filter_name", "all"),
                        "cost_scenario": scenario["cost_scenario"],
                        "cost_kind": scenario["cost_kind"],
                        "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                        "notional_usd": float(scenario.get("notional_usd", np.nan)),
                        "timestamp": pd.to_datetime(frame["timestamp"]),
                        "session": frame["session"].values,
                        "hour": frame["hour"].astype(int).values,
                        "gross_return": gross.values,
                        "cost_return": cost.values,
                        "net_return": (gross - cost).values,
                    },
                    index=frame.index,
                ).loc[active]
                trades["month"] = trades["timestamp"].dt.strftime("%Y-%m")
                rows.append(trades.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_periods(trades: pd.DataFrame, period_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_id", "split", "cost_scenario", period_col]
    for key, group in trades.groupby(group_cols, sort=False):
        candidate_id, split, cost_scenario, period_value = key
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "cost_scenario": cost_scenario,
                period_col: period_value,
                "trades": int(len(group)),
                "gross_return": float(group["gross_return"].sum()),
                "cost_return": float(group["cost_return"].sum()),
                "net_return": float(group["net_return"].sum()),
                "avg_trade_net": float(group["net_return"].mean()) if len(group) else 0.0,
                "win_rate": float(group["net_return"].gt(0.0).mean()) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def leave_one_period_summary(trades: pd.DataFrame, period_col: str = "month") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_id", "split", "cost_scenario"]
    for key, group in trades.groupby(group_cols, sort=False):
        candidate_id, split, cost_scenario = key
        total = float(group["net_return"].sum())
        periods = group.groupby(period_col, sort=False)["net_return"].sum()
        for period, net in periods.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": split,
                    "cost_scenario": cost_scenario,
                    "removed_period": period,
                    "period_net": float(net),
                    "net_without_period": float(total - net),
                    "total_net": total,
                    "period_share_of_abs_net": float(abs(net) / periods.abs().sum()) if periods.abs().sum() > 0.0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_trade_distribution(trades: pd.DataFrame, *, samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_id", "split", "cost_scenario"]
    for key, group in trades.groupby(group_cols, sort=False):
        candidate_id, split, cost_scenario = key
        values = group["net_return"].astype(float).to_numpy()
        if values.size == 0:
            continue
        sampled = rng.choice(values, size=(int(samples), values.size), replace=True)
        totals = sampled.sum(axis=1)
        avgs = sampled.mean(axis=1)
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": split,
                "cost_scenario": cost_scenario,
                "trades": int(values.size),
                "samples": int(samples),
                "observed_total_net": float(values.sum()),
                "observed_avg_trade_net": float(values.mean()),
                "prob_total_net_positive": float(np.mean(totals > 0.0)),
                "total_net_p05": float(np.quantile(totals, 0.05)),
                "total_net_p50": float(np.quantile(totals, 0.50)),
                "total_net_p95": float(np.quantile(totals, 0.95)),
                "avg_trade_p05": float(np.quantile(avgs, 0.05)),
                "avg_trade_p50": float(np.quantile(avgs, 0.50)),
                "avg_trade_p95": float(np.quantile(avgs, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def build_random_control_distribution(dataset: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.empty or specs.empty:
        return pd.DataFrame(), pd.DataFrame()
    cfg = _robust_cfg(config)
    splits = [str(value) for value in cfg.get("random_control_splits", ["test"])]
    scenario_names = [str(value) for value in cfg.get("random_control_cost_scenarios", [cfg.get("primary_cost_scenario", _search_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))])]
    scenarios = [_scenario_by_name(config, name) for name in scenario_names]
    runs = int(cfg.get("random_control_runs", 500))
    seed_prefix = str(cfg.get("random_seed", 20260503))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        for split in splits:
            frame = _candidate_split_frame(dataset, spec, split)
            if frame.empty:
                continue
            alpha = _alpha_position(frame, spec, config, split)
            for scenario in scenarios:
                alpha_metrics = evaluate_position_with_cost(frame, alpha, scenario)
                for run_idx in range(runs):
                    seed = _stable_seed(seed_prefix, str(spec["candidate_id"]), split, str(run_idx))
                    random_position = same_hour_random_position(frame, alpha, seed)
                    random_metrics = evaluate_position_with_cost(frame, random_position, scenario)
                    rows.append(
                        {
                            "candidate_id": spec["candidate_id"],
                            "split": split,
                            "cost_scenario": scenario["cost_scenario"],
                            "run": int(run_idx),
                            "alpha_net_return": float(alpha_metrics["net_return"]),
                            "alpha_avg_trade_net": float(alpha_metrics["avg_trade_net"]),
                            "alpha_trades": int(alpha_metrics["trades"]),
                            "random_net_return": float(random_metrics["net_return"]),
                            "random_avg_trade_net": float(random_metrics["avg_trade_net"]),
                            "random_trades": int(random_metrics["trades"]),
                            "random_beats_alpha": bool(float(random_metrics["net_return"]) >= float(alpha_metrics["net_return"])),
                        }
                    )
    distribution = pd.DataFrame(rows)
    if distribution.empty:
        return distribution, pd.DataFrame()
    summary = (
        distribution.groupby(["candidate_id", "split", "cost_scenario"], as_index=False)
        .agg(
            alpha_net_return=("alpha_net_return", "first"),
            alpha_avg_trade_net=("alpha_avg_trade_net", "first"),
            alpha_trades=("alpha_trades", "first"),
            random_runs=("run", "count"),
            random_net_mean=("random_net_return", "mean"),
            random_net_p05=("random_net_return", lambda values: float(np.quantile(values, 0.05))),
            random_net_p50=("random_net_return", lambda values: float(np.quantile(values, 0.50))),
            random_net_p95=("random_net_return", lambda values: float(np.quantile(values, 0.95))),
            random_net_max=("random_net_return", "max"),
            prob_random_beats_alpha=("random_beats_alpha", "mean"),
        )
        .reset_index(drop=True)
    )
    return distribution, summary


def _metric_row(frame: pd.DataFrame, candidate_id: str, split: str, cost_scenario: str) -> pd.Series:
    rows = frame[frame["candidate_id"].eq(candidate_id) & frame["split"].eq(split) & frame["cost_scenario"].eq(cost_scenario)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _positive_month_rate(monthly: pd.DataFrame, candidate_id: str, split: str, cost_scenario: str) -> float:
    rows = monthly[monthly["candidate_id"].eq(candidate_id) & monthly["split"].eq(split) & monthly["cost_scenario"].eq(cost_scenario)]
    if rows.empty:
        return np.nan
    return float(rows["net_return"].gt(0.0).mean())


def _min_leave_one_net(leave_one: pd.DataFrame, candidate_id: str, split: str, cost_scenario: str) -> float:
    rows = leave_one[leave_one["candidate_id"].eq(candidate_id) & leave_one["split"].eq(split) & leave_one["cost_scenario"].eq(cost_scenario)]
    if rows.empty:
        return np.nan
    return float(rows["net_without_period"].min())


def label_robustness(
    specs: pd.DataFrame,
    cost_curve: pd.DataFrame,
    monthly: pd.DataFrame,
    leave_one_month: pd.DataFrame,
    bootstrap: pd.DataFrame,
    random_summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    cfg = _robust_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", _search_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000")))
    stress = str(cfg.get("stress_cost_scenario", _search_cfg(config).get("stress_cost_scenario", "bps_5")))
    test_split = str(cfg.get("test_split", _search_cfg(config).get("test_split", "test")))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        candidate_id = str(spec["candidate_id"])
        primary_row = _metric_row(cost_curve, candidate_id, test_split, primary)
        stress_row = _metric_row(cost_curve, candidate_id, test_split, stress)
        gross_avg = float(primary_row.get("gross_return", np.nan)) / float(primary_row.get("trades", np.nan)) if float(primary_row.get("trades", 0) or 0) > 0 else np.nan
        breakeven_cost_bps = gross_avg * 10_000.0 if np.isfinite(gross_avg) else np.nan
        primary_bootstrap = _metric_row(bootstrap, candidate_id, test_split, primary)
        stress_bootstrap = _metric_row(bootstrap, candidate_id, test_split, stress)
        random_row = _metric_row(random_summary, candidate_id, test_split, primary)
        positive_month_rate = _positive_month_rate(monthly, candidate_id, test_split, primary)
        min_leave_one_month_net = _min_leave_one_net(leave_one_month, candidate_id, test_split, primary)
        checks = {
            "primary_net_positive": float(primary_row.get("net_return", np.nan)) > 0.0,
            "stress_net_positive": float(stress_row.get("net_return", np.nan)) > 0.0,
            "min_trades": int(primary_row.get("trades", 0) or 0) >= int(cfg.get("min_trades", _search_cfg(config).get("min_trades", 20))),
            "breakeven_cost": breakeven_cost_bps >= float(cfg.get("min_breakeven_cost_bps", 6.0)),
            "positive_month_rate": positive_month_rate >= float(cfg.get("min_positive_month_rate", 0.50)),
            "leave_one_month_positive": min_leave_one_month_net > 0.0,
            "bootstrap_primary": float(primary_bootstrap.get("prob_total_net_positive", np.nan)) >= float(cfg.get("min_bootstrap_primary_prob_positive", 0.75)),
            "bootstrap_stress": float(stress_bootstrap.get("prob_total_net_positive", np.nan)) >= float(cfg.get("min_bootstrap_stress_prob_positive", 0.55)),
            "random_control": float(random_row.get("prob_random_beats_alpha", np.nan)) <= float(cfg.get("max_random_beats_alpha_rate", 0.10)),
        }
        failed = [name for name, ok in checks.items() if not bool(ok)]
        economic_core = all(
            bool(checks[name])
            for name in [
                "primary_net_positive",
                "stress_net_positive",
                "min_trades",
                "breakeven_cost",
            ]
        )
        if not failed:
            status = "robustness_candidate"
        elif economic_core:
            status = "robustness_provisional"
        else:
            status = "robustness_failed"
        rows.append(
            {
                "candidate_id": candidate_id,
                "robustness_status": status,
                "fold": spec.get("fold", np.nan),
                "variant": spec.get("variant", ""),
                "side": spec.get("side", ""),
                "horizon_bars": spec.get("horizon_bars", np.nan),
                "vol_filter_name": spec.get("vol_filter_name", ""),
                "hour_filter_name": spec.get("hour_filter_name", ""),
                "candidate_decision": spec.get("decision", ""),
                "failed_checks": ", ".join(failed),
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_avg_trade_primary": primary_row.get("avg_trade_net", np.nan),
                "test_trades_primary": primary_row.get("trades", np.nan),
                "test_net_stress": stress_row.get("net_return", np.nan),
                "test_avg_trade_stress": stress_row.get("avg_trade_net", np.nan),
                "breakeven_cost_bps": breakeven_cost_bps,
                "positive_month_rate": positive_month_rate,
                "min_leave_one_month_net": min_leave_one_month_net,
                "bootstrap_primary_prob_positive": primary_bootstrap.get("prob_total_net_positive", np.nan),
                "bootstrap_stress_prob_positive": stress_bootstrap.get("prob_total_net_positive", np.nan),
                "prob_random_beats_alpha": random_row.get("prob_random_beats_alpha", np.nan),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    target_symbol: str,
    specs: pd.DataFrame,
    decisions: pd.DataFrame,
    cost_curve: pd.DataFrame,
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    leave_one_month: pd.DataFrame,
    bootstrap: pd.DataFrame,
    random_summary: pd.DataFrame,
    robustness_decisions: pd.DataFrame,
    outputs: dict[str, Path],
    *,
    max_rows: int = 80,
) -> str:
    output_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    status_counts = (
        robustness_decisions["robustness_status"].value_counts().rename_axis("robustness_status").reset_index(name="rows")
        if not robustness_decisions.empty
        else pd.DataFrame()
    )
    candidate_cols = [
        "candidate_id",
        "decision",
        "test_net_primary",
        "test_sharpe_primary",
        "test_avg_trade_net_primary",
        "test_trades_primary",
        "test_net_stress",
    ]
    cost_focus = cost_curve[cost_curve["split"].eq("test")].copy() if not cost_curve.empty else pd.DataFrame()
    monthly_focus = monthly[monthly["split"].eq("test")].copy() if not monthly.empty else pd.DataFrame()
    hourly_focus = hourly[hourly["split"].eq("test")].copy() if not hourly.empty else pd.DataFrame()
    conclusion = (
        "The frozen candidate passes the robustness block and can advance to non-optimized holdout/paper-style validation."
        if not robustness_decisions.empty and robustness_decisions["robustness_status"].eq("robustness_candidate").all()
        else "The frozen candidate remains provisional; do not promote to real-money without the next holdout/paper-style validation block."
    )
    return f"""# Volatility Expansion Candidate Robustness - {target_symbol.upper()}

## Scope

- No threshold or feature reoptimization.
- Uses frozen selected specs and frozen train-derived thresholds.
- Evaluates accepted candidates only unless `candidate_id` is configured.
- Stress tests: cost curve, month/hour distribution, leave-one-month, trade bootstrap and same-hour random-control distribution.

## Robustness Status

{_markdown_table(status_counts, max_rows=max_rows)}

## Frozen Candidates

{_markdown_table(specs.loc[:, [column for column in candidate_cols if column in specs.columns]], max_rows=max_rows)}

## Robustness Decision

{_markdown_table(robustness_decisions, max_rows=max_rows)}

## Test Cost Curve

{_markdown_table(cost_focus, max_rows=max_rows)}

## Test Monthly Summary

{_markdown_table(monthly_focus, max_rows=max_rows)}

## Test Hourly Summary

{_markdown_table(hourly_focus, max_rows=max_rows)}

## Leave-One-Month Test

{_markdown_table(leave_one_month[leave_one_month["split"].eq("test")] if not leave_one_month.empty else leave_one_month, max_rows=max_rows)}

## Bootstrap

{_markdown_table(bootstrap, max_rows=max_rows)}

## Random Control Distribution

{_markdown_table(random_summary, max_rows=max_rows)}

## Outputs

{output_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    cfg = _robust_cfg(config)
    paths = default_input_paths(config, target)
    specs_all = pd.read_parquet(paths["selected_specs"])
    decisions = pd.read_parquet(paths["decisions"])
    specs = select_frozen_specs(specs_all, decisions, config)
    dataset = build_split_dataset(config, target)
    cost_curve = build_cost_curve(dataset, specs, config)
    trades = build_trade_frame(dataset, specs, config)
    monthly = summarize_periods(trades, "month")
    hourly = summarize_periods(trades, "hour")
    leave_one_month = leave_one_period_summary(trades, "month")
    bootstrap = bootstrap_trade_distribution(
        trades,
        samples=int(cfg.get("bootstrap_samples", 2000)),
        seed=int(cfg.get("bootstrap_seed", 20260503)),
    )
    random_distribution, random_summary = build_random_control_distribution(dataset, specs, config)
    robustness_decisions = label_robustness(specs, cost_curve, monthly, leave_one_month, bootstrap, random_summary, config)

    results_dir = results_output_dir(config, target)
    outputs = {
        "volatility_expansion_robustness_cost_curve": results_dir / "volatility_expansion_robustness_cost_curve.parquet",
        "volatility_expansion_robustness_trades": results_dir / "volatility_expansion_robustness_trades.parquet",
        "volatility_expansion_robustness_monthly": results_dir / "volatility_expansion_robustness_monthly.parquet",
        "volatility_expansion_robustness_hourly": results_dir / "volatility_expansion_robustness_hourly.parquet",
        "volatility_expansion_robustness_leave_one_month": results_dir / "volatility_expansion_robustness_leave_one_month.parquet",
        "volatility_expansion_robustness_bootstrap": results_dir / "volatility_expansion_robustness_bootstrap.parquet",
        "volatility_expansion_robustness_random_distribution": results_dir / "volatility_expansion_robustness_random_distribution.parquet",
        "volatility_expansion_robustness_random_summary": results_dir / "volatility_expansion_robustness_random_summary.parquet",
        "volatility_expansion_robustness_decisions": results_dir / "volatility_expansion_robustness_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    cost_curve.to_parquet(outputs["volatility_expansion_robustness_cost_curve"], index=False)
    trades.to_parquet(outputs["volatility_expansion_robustness_trades"], index=False)
    monthly.to_parquet(outputs["volatility_expansion_robustness_monthly"], index=False)
    hourly.to_parquet(outputs["volatility_expansion_robustness_hourly"], index=False)
    leave_one_month.to_parquet(outputs["volatility_expansion_robustness_leave_one_month"], index=False)
    bootstrap.to_parquet(outputs["volatility_expansion_robustness_bootstrap"], index=False)
    random_distribution.to_parquet(outputs["volatility_expansion_robustness_random_distribution"], index=False)
    random_summary.to_parquet(outputs["volatility_expansion_robustness_random_summary"], index=False)
    robustness_decisions.to_parquet(outputs["volatility_expansion_robustness_decisions"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            target,
            specs,
            decisions,
            cost_curve,
            monthly,
            hourly,
            leave_one_month,
            bootstrap,
            random_summary,
            robustness_decisions,
            outputs,
            max_rows=int(cfg.get("report_top_rows", 80)),
        ),
        encoding="utf-8",
    )
    return report_path, outputs["volatility_expansion_robustness_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness checks for frozen volatility-expansion accepted candidates.")
    parser.add_argument("--config", default="configs/hmm_lab_15min_expansion_frequency_repair.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Volatility expansion candidate robustness report written to: {report_path}")
    print(f"Volatility expansion candidate robustness decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

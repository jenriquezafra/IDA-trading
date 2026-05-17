from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, evaluate_position_with_cost, scenario_cost_return
from src.excess_reversion_search import same_hour_random_position
from src.hmm_lab import _target_symbol, build_lab_folds, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.volatility_expansion_candidate_robustness import (
    bootstrap_trade_distribution,
    default_input_paths,
    leave_one_period_summary,
    robustness_scenarios,
    select_frozen_specs,
    summarize_periods,
)
from src.volatility_expansion_search import (
    REQUIRED_FEATURE_COLUMNS,
    _add_prior_compression_columns,
    _control_positions,
    _json_loads,
    _search_cfg,
    load_feature_frame,
)


def _holdout_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("volatility_expansion_holdout", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "volatility_expansion_holdout.md"


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _scenario_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    by_name = {str(scenario["cost_scenario"]): scenario for scenario in cost_scenarios(config)}
    if name not in by_name:
        raise ValueError(f"Cost scenario not found: {name}")
    return by_name[name]


def build_full_candidate_dataset(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _holdout_cfg(config)
    horizons = [int(value) for value in cfg.get("horizons", _search_cfg(config).get("horizons", [4]))]
    features = load_feature_frame(config, target_symbol)
    forward_returns = build_forward_returns(features, horizons)
    indexed = features.reset_index(names="source_index")
    available = [
        "source_index",
        *[column for column in REQUIRED_FEATURE_COLUMNS if column in indexed.columns and column not in forward_returns.columns],
    ]
    merged = forward_returns.merge(indexed.loc[:, available], on="source_index", how="left", validate="many_to_one")
    merged = _add_prior_compression_columns(merged)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged["split"] = "full"
    merged["proposed_label"] = "no_hmm"
    merged["feature_set"] = "volatility_expansion"
    merged["n_states"] = 0
    merged["seed"] = 0
    return merged.sort_values(["session", "bar_index", "horizon_bars"], kind="stable").reset_index(drop=True)


def fold_test_boundary(features: pd.DataFrame, config: dict[str, Any], fold_id: int) -> tuple[str, str]:
    folds = build_lab_folds(features, config)
    matches = [fold for fold in folds if int(fold.fold) == int(fold_id)]
    if not matches:
        raise ValueError(f"Fold not found: {fold_id}")
    fold = matches[0]
    if not fold.test_sessions:
        raise ValueError(f"Fold has no test sessions: {fold_id}")
    return str(min(fold.test_sessions)), str(max(fold.test_sessions))


def holdout_frame_for_spec(dataset: pd.DataFrame, features: pd.DataFrame, spec: pd.Series | dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    cfg = _holdout_cfg(config)
    _, default_start_after = fold_test_boundary(features, config, int(spec["fold"]))
    start_after = str(cfg.get("start_after_session", default_start_after))
    end_session = cfg.get("end_session")
    frame = dataset[
        dataset["horizon_bars"].eq(int(spec["horizon_bars"]))
        & dataset["session"].astype(str).gt(start_after)
    ].copy()
    if end_session:
        frame = frame[frame["session"].astype(str).le(str(end_session))].copy()
    frame["split"] = "holdout"
    frame["fold"] = int(spec["fold"])
    return frame.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)


def _alpha_position(frame: pd.DataFrame, spec: pd.Series | dict[str, Any], config: dict[str, Any]) -> pd.Series:
    spec_dict = dict(spec)
    positions = _control_positions(
        frame,
        spec_dict,
        _json_loads(spec_dict["thresholds_json"]),
        config,
        str(spec_dict["candidate_id"]),
        "holdout",
    )
    return positions["alpha_signal"]


def evaluate_holdout_cost_curve(dataset: pd.DataFrame, features: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        frame = holdout_frame_for_spec(dataset, features, spec, config)
        position = _alpha_position(frame, spec, config) if not frame.empty else pd.Series(dtype=float)
        for scenario in robustness_scenarios(config):
            metadata = {
                "candidate_id": spec["candidate_id"],
                "split": "holdout",
                "fold": int(spec["fold"]),
                "variant": spec.get("variant", ""),
                "side": spec.get("side", ""),
                "horizon_bars": int(spec["horizon_bars"]),
                "vol_filter_name": spec.get("vol_filter_name", "none"),
                "hour_filter_name": spec.get("hour_filter_name", "all"),
                "holdout_start_session": str(frame["session"].min()) if not frame.empty else "",
                "holdout_end_session": str(frame["session"].max()) if not frame.empty else "",
                "holdout_sessions": int(frame["session"].nunique()) if not frame.empty else 0,
                "cost_scenario": scenario["cost_scenario"],
                "cost_kind": scenario["cost_kind"],
                "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                "notional_usd": float(scenario.get("notional_usd", np.nan)),
            }
            metrics = evaluate_position_with_cost(frame, position, scenario) if not frame.empty else {}
            rows.append({**metadata, **metrics})
    return pd.DataFrame(rows)


def build_holdout_trades(dataset: pd.DataFrame, features: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    cfg = _holdout_cfg(config)
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
        frame = holdout_frame_for_spec(dataset, features, spec, config)
        if frame.empty:
            continue
        position = _alpha_position(frame, spec, config)
        active = position.abs().gt(0.0)
        if not active.any():
            continue
        gross = position * frame["fwd_ret"].astype(float)
        for scenario in scenarios:
            cost = scenario_cost_return(frame, position, scenario)
            trades = pd.DataFrame(
                {
                    "candidate_id": spec["candidate_id"],
                    "split": "holdout",
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


def build_holdout_random_distribution(dataset: pd.DataFrame, features: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.empty or specs.empty:
        return pd.DataFrame(), pd.DataFrame()
    cfg = _holdout_cfg(config)
    scenario_names = [str(value) for value in cfg.get("random_control_cost_scenarios", [cfg.get("primary_cost_scenario", _search_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))])]
    scenarios = [_scenario_by_name(config, name) for name in scenario_names]
    runs = int(cfg.get("random_control_runs", 500))
    seed_prefix = str(cfg.get("random_seed", 20260503))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        frame = holdout_frame_for_spec(dataset, features, spec, config)
        if frame.empty:
            continue
        alpha = _alpha_position(frame, spec, config)
        for scenario in scenarios:
            alpha_metrics = evaluate_position_with_cost(frame, alpha, scenario)
            for run_idx in range(runs):
                seed = _stable_seed(seed_prefix, str(spec["candidate_id"]), "holdout", str(run_idx))
                random_position = same_hour_random_position(frame, alpha, seed)
                random_metrics = evaluate_position_with_cost(frame, random_position, scenario)
                rows.append(
                    {
                        "candidate_id": spec["candidate_id"],
                        "split": "holdout",
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
    summary = distribution.groupby(["candidate_id", "split", "cost_scenario"], as_index=False).agg(
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
    return distribution, summary


def _metric_row(frame: pd.DataFrame, candidate_id: str, cost_scenario: str) -> pd.Series:
    if frame.empty or not {"candidate_id", "split", "cost_scenario"}.issubset(frame.columns):
        return pd.Series(dtype=object)
    rows = frame[frame["candidate_id"].eq(candidate_id) & frame["split"].eq("holdout") & frame["cost_scenario"].eq(cost_scenario)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _positive_month_rate(monthly: pd.DataFrame, candidate_id: str, cost_scenario: str) -> float:
    if monthly.empty or not {"candidate_id", "split", "cost_scenario", "net_return"}.issubset(monthly.columns):
        return np.nan
    rows = monthly[monthly["candidate_id"].eq(candidate_id) & monthly["split"].eq("holdout") & monthly["cost_scenario"].eq(cost_scenario)]
    if rows.empty:
        return np.nan
    return float(rows["net_return"].gt(0.0).mean())


def _min_leave_one_net(leave_one: pd.DataFrame, candidate_id: str, cost_scenario: str) -> float:
    if leave_one.empty or not {"candidate_id", "split", "cost_scenario", "net_without_period"}.issubset(leave_one.columns):
        return np.nan
    rows = leave_one[leave_one["candidate_id"].eq(candidate_id) & leave_one["split"].eq("holdout") & leave_one["cost_scenario"].eq(cost_scenario)]
    if rows.empty:
        return np.nan
    return float(rows["net_without_period"].min())


def label_holdout(
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
    cfg = _holdout_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", _search_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000")))
    stress = str(cfg.get("stress_cost_scenario", _search_cfg(config).get("stress_cost_scenario", "bps_5")))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        candidate_id = str(spec["candidate_id"])
        primary_row = _metric_row(cost_curve, candidate_id, primary)
        stress_row = _metric_row(cost_curve, candidate_id, stress)
        gross_avg = float(primary_row.get("gross_return", np.nan)) / float(primary_row.get("trades", np.nan)) if float(primary_row.get("trades", 0) or 0) > 0 else np.nan
        breakeven_cost_bps = gross_avg * 10_000.0 if np.isfinite(gross_avg) else np.nan
        primary_bootstrap = _metric_row(bootstrap, candidate_id, primary)
        stress_bootstrap = _metric_row(bootstrap, candidate_id, stress)
        random_row = _metric_row(random_summary, candidate_id, primary)
        positive_month_rate = _positive_month_rate(monthly, candidate_id, primary)
        min_leave_one_month_net = _min_leave_one_net(leave_one_month, candidate_id, primary)
        checks = {
            "primary_net_positive": float(primary_row.get("net_return", np.nan)) > 0.0,
            "stress_net_positive": float(stress_row.get("net_return", np.nan)) > 0.0,
            "min_trades": int(primary_row.get("trades", 0) or 0) >= int(cfg.get("min_trades", 30)),
            "min_avg_trade": float(primary_row.get("avg_trade_net", np.nan)) >= float(cfg.get("min_avg_trade_net", 0.00015)),
            "breakeven_cost": breakeven_cost_bps >= float(cfg.get("min_breakeven_cost_bps", 6.0)),
            "positive_month_rate": positive_month_rate >= float(cfg.get("min_positive_month_rate", 0.50)),
            "leave_one_month_positive": min_leave_one_month_net > 0.0,
            "bootstrap_primary": float(primary_bootstrap.get("prob_total_net_positive", np.nan)) >= float(cfg.get("min_bootstrap_primary_prob_positive", 0.75)),
            "bootstrap_stress": float(stress_bootstrap.get("prob_total_net_positive", np.nan)) >= float(cfg.get("min_bootstrap_stress_prob_positive", 0.55)),
            "random_control": float(random_row.get("prob_random_beats_alpha", np.nan)) <= float(cfg.get("max_random_beats_alpha_rate", 0.10)),
        }
        failed = [name for name, ok in checks.items() if not bool(ok)]
        economic_core = all(bool(checks[name]) for name in ["primary_net_positive", "stress_net_positive", "min_trades", "min_avg_trade", "breakeven_cost"])
        if not failed:
            status = "holdout_pass"
        elif economic_core:
            status = "holdout_provisional"
        else:
            status = "holdout_failed"
        rows.append(
            {
                "candidate_id": candidate_id,
                "holdout_status": status,
                "fold": spec.get("fold", np.nan),
                "variant": spec.get("variant", ""),
                "side": spec.get("side", ""),
                "horizon_bars": spec.get("horizon_bars", np.nan),
                "vol_filter_name": spec.get("vol_filter_name", ""),
                "hour_filter_name": spec.get("hour_filter_name", ""),
                "candidate_decision": spec.get("decision", ""),
                "failed_checks": ", ".join(failed),
                "holdout_start_session": primary_row.get("holdout_start_session", ""),
                "holdout_end_session": primary_row.get("holdout_end_session", ""),
                "holdout_sessions": primary_row.get("holdout_sessions", np.nan),
                "holdout_net_primary": primary_row.get("net_return", np.nan),
                "holdout_avg_trade_primary": primary_row.get("avg_trade_net", np.nan),
                "holdout_trades_primary": primary_row.get("trades", np.nan),
                "holdout_net_stress": stress_row.get("net_return", np.nan),
                "holdout_avg_trade_stress": stress_row.get("avg_trade_net", np.nan),
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
    holdout_decisions: pd.DataFrame,
    cost_curve: pd.DataFrame,
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    leave_one_month: pd.DataFrame,
    bootstrap: pd.DataFrame,
    random_summary: pd.DataFrame,
    outputs: dict[str, Path],
    *,
    max_rows: int = 80,
) -> str:
    status_counts = (
        holdout_decisions["holdout_status"].value_counts().rename_axis("holdout_status").reset_index(name="rows")
        if not holdout_decisions.empty
        else pd.DataFrame()
    )
    candidate_cols = [
        "candidate_id",
        "decision",
        "test_net_primary",
        "test_avg_trade_net_primary",
        "test_trades_primary",
        "test_net_stress",
    ]
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "The frozen candidate passes posterior holdout. Next step is paper-style simulation with execution assumptions frozen."
        if not holdout_decisions.empty and holdout_decisions["holdout_status"].eq("holdout_pass").all()
        else "The frozen candidate does not pass posterior holdout cleanly. Do not promote without a new, validation-defined hypothesis."
    )
    return f"""# Volatility Expansion Posterior Holdout - {target_symbol.upper()}

## Scope

- Same accepted candidate.
- Same frozen `thresholds_json`.
- No search, no reoptimization and no test-derived hour filter.
- Holdout starts after the candidate fold test window.

## Holdout Status

{_markdown_table(status_counts, max_rows=max_rows)}

## Frozen Candidate

{_markdown_table(specs.loc[:, [column for column in candidate_cols if column in specs.columns]], max_rows=max_rows)}

## Holdout Decision

{_markdown_table(holdout_decisions, max_rows=max_rows)}

## Holdout Cost Curve

{_markdown_table(cost_curve, max_rows=max_rows)}

## Holdout Monthly Summary

{_markdown_table(monthly, max_rows=max_rows)}

## Holdout Hourly Summary

{_markdown_table(hourly, max_rows=max_rows)}

## Holdout Leave-One-Month

{_markdown_table(leave_one_month, max_rows=max_rows)}

## Holdout Bootstrap

{_markdown_table(bootstrap, max_rows=max_rows)}

## Holdout Random Control Distribution

{_markdown_table(random_summary, max_rows=max_rows)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    cfg = _holdout_cfg(config)
    paths = default_input_paths(config, target)
    specs_all = pd.read_parquet(paths["selected_specs"])
    decisions = pd.read_parquet(paths["decisions"])
    specs = select_frozen_specs(specs_all, decisions, config)
    features = load_feature_frame(config, target)
    dataset = build_full_candidate_dataset(config, target)
    cost_curve = evaluate_holdout_cost_curve(dataset, features, specs, config)
    trades = build_holdout_trades(dataset, features, specs, config)
    monthly = summarize_periods(trades, "month")
    hourly = summarize_periods(trades, "hour")
    leave_one_month = leave_one_period_summary(trades, "month")
    bootstrap = bootstrap_trade_distribution(
        trades,
        samples=int(cfg.get("bootstrap_samples", 2000)),
        seed=int(cfg.get("bootstrap_seed", 20260503)),
    )
    random_distribution, random_summary = build_holdout_random_distribution(dataset, features, specs, config)
    holdout_decisions = label_holdout(specs, cost_curve, monthly, leave_one_month, bootstrap, random_summary, config)

    results_dir = results_output_dir(config, target)
    outputs = {
        "volatility_expansion_holdout_cost_curve": results_dir / "volatility_expansion_holdout_cost_curve.parquet",
        "volatility_expansion_holdout_trades": results_dir / "volatility_expansion_holdout_trades.parquet",
        "volatility_expansion_holdout_monthly": results_dir / "volatility_expansion_holdout_monthly.parquet",
        "volatility_expansion_holdout_hourly": results_dir / "volatility_expansion_holdout_hourly.parquet",
        "volatility_expansion_holdout_leave_one_month": results_dir / "volatility_expansion_holdout_leave_one_month.parquet",
        "volatility_expansion_holdout_bootstrap": results_dir / "volatility_expansion_holdout_bootstrap.parquet",
        "volatility_expansion_holdout_random_distribution": results_dir / "volatility_expansion_holdout_random_distribution.parquet",
        "volatility_expansion_holdout_random_summary": results_dir / "volatility_expansion_holdout_random_summary.parquet",
        "volatility_expansion_holdout_decisions": results_dir / "volatility_expansion_holdout_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    cost_curve.to_parquet(outputs["volatility_expansion_holdout_cost_curve"], index=False)
    trades.to_parquet(outputs["volatility_expansion_holdout_trades"], index=False)
    monthly.to_parquet(outputs["volatility_expansion_holdout_monthly"], index=False)
    hourly.to_parquet(outputs["volatility_expansion_holdout_hourly"], index=False)
    leave_one_month.to_parquet(outputs["volatility_expansion_holdout_leave_one_month"], index=False)
    bootstrap.to_parquet(outputs["volatility_expansion_holdout_bootstrap"], index=False)
    random_distribution.to_parquet(outputs["volatility_expansion_holdout_random_distribution"], index=False)
    random_summary.to_parquet(outputs["volatility_expansion_holdout_random_summary"], index=False)
    holdout_decisions.to_parquet(outputs["volatility_expansion_holdout_decisions"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            target,
            specs,
            holdout_decisions,
            cost_curve,
            monthly,
            hourly,
            leave_one_month,
            bootstrap,
            random_summary,
            outputs,
            max_rows=int(cfg.get("report_top_rows", 80)),
        ),
        encoding="utf-8",
    )
    return report_path, outputs["volatility_expansion_holdout_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen volatility-expansion candidates on posterior holdout.")
    parser.add_argument("--config", default="configs/hmm_lab_15min_expansion_frequency_repair.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Volatility expansion holdout report written to: {report_path}")
    print(f"Volatility expansion holdout decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

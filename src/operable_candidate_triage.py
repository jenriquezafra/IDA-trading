from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import _path_from_template, _search_cfg


METRIC_COLS = [
    "rows",
    "trades",
    "exposure",
    "turnover",
    "gross_return",
    "total_cost",
    "effective_cost_bps",
    "net_return",
    "avg_trade_net",
    "profit_factor",
    "daily_sharpe",
    "max_drawdown",
    "drawdown_duration_bars",
    "drawdown_duration_days",
    "worst_day_net",
    "worst_month_net",
    "top_day_abs_net_share",
    "top_month_abs_net_share",
    "top_hour_abs_net_share",
    "top_state_abs_net_share",
    "net_delta_vs_base",
    "net_delta_vs_same_hour",
    "daily_sharpe_delta_vs_base",
    "daily_sharpe_delta_vs_same_hour",
    "drawdown_reduction_vs_base",
    "drawdown_reduction_vs_same_hour",
    "turnover_reduction_vs_base",
]


def _triage_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("operable_candidate_triage", {})


def _combined_cfg(config: dict[str, Any]) -> dict[str, Any]:
    combined = dict(_search_cfg(config))
    combined.update(_triage_cfg(config))
    return combined


def _result_path(config: dict[str, Any], target_symbol: str, name: str) -> Path:
    return results_output_dir(config, target_symbol) / name


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    return reports_dir / target_symbol.upper() / "operable_candidate_triage.md"


def _input_path(config: dict[str, Any], target_symbol: str, key: str, default_name: str) -> Path:
    cfg = _triage_cfg(config)
    default = f"results/{{target_symbol}}/{default_name}"
    return _path_from_template(str(cfg.get(key, default)), target_symbol)


def load_inputs(config: dict[str, Any], target_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_parquet(_input_path(config, target_symbol, "decisions", "operable_candidate_decisions.parquet"))
    validation = pd.read_parquet(_input_path(config, target_symbol, "selected_validation", "operable_candidate_selected_validation.parquet"))
    test = pd.read_parquet(_input_path(config, target_symbol, "test", "operable_candidate_test.parquet"))
    return decisions, validation, test


def _prefixed_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = ["candidate_id", *[col for col in METRIC_COLS if col in frame.columns]]
    output = frame.loc[:, cols].copy()
    return output.rename(columns={col: f"{prefix}_{col}" for col in output.columns if col != "candidate_id"})


def _scenario_bucket_metrics(frame: pd.DataFrame, cost_scenario: str, bucket: str, prefix: str) -> pd.DataFrame:
    selected = frame[frame["cost_scenario"].eq(cost_scenario) & frame["bucket"].eq(bucket)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["candidate_id"])
    return _prefixed_metrics(selected, prefix)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _value(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    return float(value) if _finite(value) else default


def gate_failures(row: pd.Series, config: dict[str, Any]) -> list[str]:
    cfg = _combined_cfg(config)
    failures: list[str] = []
    trades = _value(row, "primary_trades")
    net_return = _value(row, "primary_net_return")
    avg_trade = _value(row, "primary_avg_trade_net")
    profit_factor = _value(row, "primary_profit_factor")
    sharpe = _value(row, "primary_daily_sharpe")
    drawdown = _value(row, "primary_max_drawdown")
    concentration = _value(row, "primary_top_day_abs_net_share")
    turnover = _value(row, "primary_turnover")
    net_delta_base = _value(row, "primary_net_delta_vs_base")
    net_delta_same_hour = _value(row, "primary_net_delta_vs_same_hour")
    drawdown_base = _value(row, "primary_drawdown_reduction_vs_base")
    drawdown_same_hour = _value(row, "primary_drawdown_reduction_vs_same_hour")
    conservative_net = _value(row, "conservative_net_return")
    stress_net = _value(row, "stress_net_return")
    validation_net = _value(row, "validation_net_return")

    if not _finite(net_return):
        return ["missing_primary_test_metrics"]
    if trades < float(cfg.get("min_trades", 30)):
        failures.append("insufficient_test_trades")
    if net_return <= 0 or avg_trade <= 0:
        failures.append("nonpositive_test_edge")
    if profit_factor < float(cfg.get("min_profit_factor", 1.10)):
        failures.append("weak_test_profit_factor")
    if sharpe < float(cfg.get("min_daily_sharpe", 1.0)):
        failures.append("weak_test_sharpe")
    if drawdown > float(cfg.get("max_drawdown", 0.12)):
        failures.append("excessive_test_drawdown")
    if concentration > float(cfg.get("max_top_day_abs_net_share", 0.35)):
        failures.append("concentrated_test_pnl")
    if turnover > float(cfg.get("max_turnover", 4.0)):
        failures.append("high_test_turnover")
    if net_delta_base <= 0:
        failures.append("hmm_underperforms_base_return")
    if net_delta_same_hour <= 0:
        failures.append("hmm_underperforms_same_hour_return")
    if net_delta_base <= 0 and drawdown_base <= 0:
        failures.append("no_base_incrementality")
    if net_delta_same_hour <= 0 and drawdown_same_hour <= 0:
        failures.append("no_same_hour_incrementality")
    if _finite(conservative_net) and conservative_net <= 0:
        failures.append("fails_conservative_cost")
    if _finite(stress_net) and stress_net <= 0:
        failures.append("fails_5bps_stress_cost")
    if _finite(validation_net) and validation_net > 0 and net_return < 0.25 * validation_net:
        failures.append("validation_to_test_decay")
    return failures


def triage_label(row: pd.Series, config: dict[str, Any]) -> str:
    failures = set(str(row.get("failure_reasons", "")).split(","))
    decision = str(row.get("decision", ""))
    net_return = _value(row, "primary_net_return")
    profit_factor = _value(row, "primary_profit_factor")
    cfg = _combined_cfg(config)
    if decision == "accepted_candidate":
        return "accepted_candidate"
    if "missing_primary_test_metrics" in failures:
        return "missing_test_metrics"
    if net_return <= 0 or "nonpositive_test_edge" in failures:
        return "negative_oos_edge"
    if {"insufficient_test_trades", "concentrated_test_pnl"} & failures:
        return "too_sparse_or_concentrated"
    if {"no_base_incrementality", "no_same_hour_incrementality"} & failures:
        return "no_hmm_incrementality"
    if "weak_test_sharpe" in failures and profit_factor >= float(cfg.get("min_profit_factor", 1.10)):
        return "positive_but_low_sharpe"
    if "fails_conservative_cost" in failures:
        return "cost_fragile"
    return "research_followup"


def next_action(row: pd.Series) -> str:
    label = str(row.get("triage_label", ""))
    if label == "negative_oos_edge":
        return "drop_candidate"
    if label == "too_sparse_or_concentrated":
        return "do_not_freeze_collect_more_evidence"
    if label == "no_hmm_incrementality":
        return "drop_hmm_filter_variant"
    if label == "positive_but_low_sharpe":
        return "promote_family_to_feature_refinement"
    if label == "cost_fragile":
        return "redesign_for_lower_turnover"
    if label == "research_followup":
        return "manual_review"
    return "hold"


def build_triage_frame(decisions: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))

    frame = decisions.copy()
    joins = [
        _scenario_bucket_metrics(test, primary, "hmm_filter", "primary"),
        _scenario_bucket_metrics(test, conservative, "hmm_filter", "conservative"),
        _scenario_bucket_metrics(test, stress, "hmm_filter", "stress"),
        _scenario_bucket_metrics(test, primary, "base_no_hmm", "base"),
        _scenario_bucket_metrics(test, primary, "same_hour_control", "same_hour"),
        _scenario_bucket_metrics(validation, primary, "hmm_filter", "validation"),
    ]
    for join in joins:
        if not join.empty:
            frame = frame.merge(join, on="candidate_id", how="left", validate="one_to_one")

    failure_lists = frame.apply(lambda row: gate_failures(row, config), axis=1)
    frame["failure_reasons"] = failure_lists.map(lambda values: ",".join(values))
    frame["failure_count"] = failure_lists.map(len)
    frame["triage_label"] = frame.apply(lambda row: triage_label(row, config), axis=1)
    frame["next_action"] = frame.apply(next_action, axis=1)
    frame["research_score"] = (
        frame["primary_net_return"].fillna(-1.0)
        + 0.02 * frame["primary_daily_sharpe"].replace([np.inf, -np.inf], np.nan).fillna(-5.0)
        + 0.01 * frame["primary_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=5.0)
        + 10.0 * frame["primary_avg_trade_net"].fillna(-0.01)
        + frame["primary_drawdown_reduction_vs_base"].fillna(0.0)
        - 0.01 * frame["failure_count"].fillna(0)
    )
    return frame.sort_values(["decision", "failure_count", "research_score"], ascending=[True, True, False], kind="stable").reset_index(drop=True)


def failure_counts(triage: pd.DataFrame) -> pd.DataFrame:
    if triage.empty:
        return pd.DataFrame(columns=["failure_reason", "rows"])
    exploded = triage.assign(failure_reason=triage["failure_reasons"].str.split(",")).explode("failure_reason")
    exploded = exploded[exploded["failure_reason"].astype(str).ne("")]
    return exploded["failure_reason"].value_counts().rename_axis("failure_reason").reset_index(name="rows")


def family_summary(triage: pd.DataFrame) -> pd.DataFrame:
    if triage.empty:
        return pd.DataFrame()
    return (
        triage.groupby(["strategy", "filter_name", "horizon_bars"], as_index=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            research_candidates=("decision", lambda values: int((values == "research_candidate").sum())),
            median_net_return=("primary_net_return", "median"),
            positive_net_rate=("primary_net_return", lambda values: float((values > 0).mean())),
            median_daily_sharpe=("primary_daily_sharpe", "median"),
            median_profit_factor=("primary_profit_factor", "median"),
            median_avg_trade_net=("primary_avg_trade_net", "median"),
            median_trades=("primary_trades", "median"),
            median_top_day_abs_net_share=("primary_top_day_abs_net_share", "median"),
            median_net_delta_vs_base=("primary_net_delta_vs_base", "median"),
            median_net_delta_vs_same_hour=("primary_net_delta_vs_same_hour", "median"),
        )
        .sort_values(["research_candidates", "median_net_return"], ascending=[False, False], kind="stable")
    )


def recommendation(triage: pd.DataFrame, counts: pd.DataFrame) -> str:
    accepted = int((triage["decision"] == "accepted_candidate").sum()) if not triage.empty else 0
    research = triage[triage["decision"].eq("research_candidate")].copy() if not triage.empty else pd.DataFrame()
    if accepted > 0:
        return "At least one candidate can move to final freeze, but review failure reasons for neighboring candidates first."
    if research.empty:
        return "No research candidates remain. Do not run final freeze; move to new alpha/features before another search."
    top_reasons = set(counts.head(4)["failure_reason"].astype(str).tolist()) if not counts.empty else set()
    if {"weak_test_sharpe", "hmm_underperforms_base_return", "hmm_underperforms_same_hour_return"} & top_reasons:
        return (
            "Do not freeze a strategy yet. The next useful block is feature/alpha refinement around the positive research families, "
            "then rerun the operable search with the same IBKR gates."
        )
    return "Do not freeze yet; investigate the research candidates with the lowest failure count before changing the final walk-forward."


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    triage: pd.DataFrame,
    counts: pd.DataFrame,
    families: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _combined_cfg(config)
    report_top_rows = int(cfg.get("report_top_rows", 60))
    decision_counts = triage["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not triage.empty else pd.DataFrame()
    label_counts = triage["triage_label"].value_counts().rename_axis("triage_label").reset_index(name="rows") if not triage.empty else pd.DataFrame()
    research_cols = [
        "candidate_id",
        "strategy",
        "filter_name",
        "horizon_bars",
        "triage_label",
        "failure_count",
        "primary_net_return",
        "primary_daily_sharpe",
        "primary_profit_factor",
        "primary_avg_trade_net",
        "primary_trades",
        "primary_top_day_abs_net_share",
        "primary_net_delta_vs_base",
        "primary_net_delta_vs_same_hour",
        "conservative_net_return",
        "stress_net_return",
        "next_action",
    ]
    top_research = (
        triage[triage["decision"].eq("research_candidate")]
        .sort_values(["failure_count", "research_score"], ascending=[True, False], kind="stable")
        .loc[:, [col for col in research_cols if col in triage.columns]]
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    return f"""# Operable Candidate Triage - {target_symbol.upper()}

## Scope

- Input decisions: `{len(triage)}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Conservative cost scenario: `{cfg.get("conservative_cost_scenario", "bps_2")}`
- Stress cost scenario: `{cfg.get("stress_cost_scenario", "bps_5")}`
- Gates reused from `operable_candidate_search`.

## Decision Counts

{_markdown_table(decision_counts)}

## Triage Label Counts

{_markdown_table(label_counts)}

## Failure Reason Counts

{_markdown_table(counts)}

## Family Summary

{_markdown_table(families, max_rows=report_top_rows)}

## Top Research Candidates

{_markdown_table(top_research, max_rows=report_top_rows)}

## Recommendation

{recommendation(triage, counts)}

## Outputs

{outputs_text}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    decisions, validation, test = load_inputs(config, target)
    triage = build_triage_frame(decisions, validation, test, config)
    counts = failure_counts(triage)
    families = family_summary(triage)
    results_dir = results_output_dir(config, target)
    outputs = {
        "operable_candidate_triage": _result_path(config, target, "operable_candidate_triage.parquet"),
        "operable_candidate_failure_counts": _result_path(config, target, "operable_candidate_failure_counts.parquet"),
        "operable_candidate_family_triage": _result_path(config, target, "operable_candidate_family_triage.parquet"),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    triage.to_parquet(outputs["operable_candidate_triage"], index=False)
    counts.to_parquet(outputs["operable_candidate_failure_counts"], index=False)
    families.to_parquet(outputs["operable_candidate_family_triage"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, triage, counts, families, outputs), encoding="utf-8")
    return report_path, outputs["operable_candidate_triage"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage operable candidate research results and explain rejection reasons.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, triage_path = run(args.config, args.target)
    print(f"Operable candidate triage report written to: {report_path}")
    print(f"Operable candidate triage table written to: {triage_path}")


if __name__ == "__main__":
    main()

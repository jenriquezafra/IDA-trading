from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios
from src.setup_signal_diagnostics import reconstruct_bar_returns, summarize_returns
from src.setup_signal_search import (
    _combined_cfg,
    build_signal_dataset,
    evaluate_selected_on_split,
    validation_grid,
)


SPEC_COLUMNS = [
    "candidate_id",
    "fold",
    "family",
    "direction",
    "horizon_bars",
    "params_json",
    "column_map_json",
    "anti_status",
    "anti_score",
]


def _anti_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_anti_concentration", {})


def _float_value(row: pd.Series, key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if np.isnan(result) else result


def configured_focus(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _anti_cfg(config)
    fallback = config.get("setup_signal_diagnostics", {})
    return {
        "family": str(cfg.get("family", fallback.get("family", "breakdown_short_risk_off"))),
        "direction": str(cfg.get("direction", fallback.get("direction", "short"))),
        "horizon_bars": int(cfg.get("horizon_bars", fallback.get("horizon_bars", 4))),
    }


def focused_search_config(config: dict[str, Any]) -> dict[str, Any]:
    focused = copy.deepcopy(config)
    cfg = _anti_cfg(config)
    focus = configured_focus(config)
    search = dict(focused.get("setup_signal_search", {}))
    search["families"] = [focus["family"]]
    search["horizons"] = [focus["horizon_bars"]]
    for key in ["rel_volume_quantiles", "risk_off_quantiles", "vwap_abs_mins", "breadth_low_quantile"]:
        if key in cfg:
            search[key] = cfg[key]
    focused["setup_signal_search"] = search
    return focused


def primary_cost_name(config: dict[str, Any]) -> str:
    return str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))


def conservative_cost_name(config: dict[str, Any]) -> str:
    return str(_combined_cfg(config).get("conservative_cost_scenario", "bps_2"))


def stress_cost_name(config: dict[str, Any]) -> str:
    return str(_combined_cfg(config).get("stress_cost_scenario", "bps_5"))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "setup_signal_anti_concentration.md"


def candidate_monthly_metrics(bar_returns: pd.DataFrame) -> pd.DataFrame:
    if bar_returns.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "fold",
                "validation_months",
                "positive_months",
                "positive_month_rate",
                "median_month_net",
                "worst_month_net",
                "leave_one_month_min_net",
                "top_month_abs_net_share_rebuilt",
            ]
        )
    validation = bar_returns[bar_returns["split"].eq("validation")].copy()
    if validation.empty:
        return pd.DataFrame()
    monthly = (
        validation.groupby(["candidate_id", "fold", "month"], as_index=False, sort=False)
        .agg(trades=("net_return", "size"), net_return=("net_return", "sum"))
        .reset_index(drop=True)
    )
    rows = []
    for (candidate_id, fold), group in monthly.groupby(["candidate_id", "fold"], sort=False):
        month_net = group["net_return"].astype(float)
        total_abs = float(month_net.abs().sum())
        total_net = float(month_net.sum())
        strongest_month_net = float(month_net.max()) if not month_net.empty else np.nan
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold": int(fold),
                "validation_months": int(group["month"].nunique()),
                "positive_months": int(month_net.gt(0.0).sum()),
                "positive_month_rate": float(month_net.gt(0.0).mean()) if len(month_net) else np.nan,
                "median_month_net": float(month_net.median()) if len(month_net) else np.nan,
                "worst_month_net": float(month_net.min()) if len(month_net) else np.nan,
                "leave_one_month_min_net": total_net - strongest_month_net,
                "top_month_abs_net_share_rebuilt": float(month_net.abs().max() / total_abs) if total_abs > 0.0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def classify_anti_concentration(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _anti_cfg(config)
    if int(_float_value(row, "trades", 0.0)) < int(cfg.get("min_trades", 40)):
        return "rejected_insufficient_trades"
    if _float_value(row, "net_return", 0.0) <= 0.0 or _float_value(row, "avg_trade_net", 0.0) <= 0.0:
        return "rejected_negative_edge"
    if _float_value(row, "profit_factor", 0.0) < float(cfg.get("min_profit_factor", 1.05)):
        return "rejected_weak_profit_factor"
    if _float_value(row, "daily_sharpe", -np.inf) < float(cfg.get("min_daily_sharpe", 0.30)):
        return "rejected_weak_sharpe"
    if _float_value(row, "top_day_abs_net_share", np.inf) > float(cfg.get("max_top_day_abs_net_share", 0.30)):
        return "rejected_day_concentration"
    if int(_float_value(row, "validation_months", 0.0)) < int(cfg.get("min_months", 4)):
        return "rejected_insufficient_months"
    if int(_float_value(row, "positive_months", 0.0)) < int(cfg.get("min_positive_months", 3)):
        return "rejected_few_positive_months"
    if _float_value(row, "positive_month_rate", 0.0) < float(cfg.get("min_positive_month_rate", 0.50)):
        return "rejected_low_positive_month_rate"
    top_month = _float_value(row, "top_month_abs_net_share_rebuilt", _float_value(row, "top_month_abs_net_share", np.inf))
    if top_month > float(cfg.get("max_top_month_abs_net_share", 0.45)):
        return "rejected_month_concentration"
    if _float_value(row, "leave_one_month_min_net", -np.inf) <= float(cfg.get("min_leave_one_month_net", 0.0)):
        return "rejected_top_month_dependency"
    if _float_value(row, "median_month_net", -np.inf) < float(cfg.get("min_median_month_net", -np.inf)):
        return "rejected_weak_median_month"
    return "anti_concentration_candidate"


def add_anti_scores(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    output = rows.copy()
    pf = output["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    output["anti_score"] = (
        output["daily_sharpe"].fillna(0.0)
        + 100.0 * output["avg_trade_net"].fillna(0.0)
        + 0.20 * pf
        + 2.0 * output["positive_month_rate"].fillna(0.0)
        + output["leave_one_month_min_net"].fillna(-1.0)
        - output["top_month_abs_net_share_rebuilt"].fillna(1.0)
    )
    output["anti_status"] = output.apply(lambda row: classify_anti_concentration(row, config), axis=1)
    output["anti_status_rank"] = output["anti_status"].ne("anti_concentration_candidate").astype(int)
    return output.sort_values(["fold", "anti_status_rank", "anti_score", "net_return"], ascending=[True, True, False, False], kind="stable")


def rank_validation_candidates(validation: pd.DataFrame, bar_returns: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    focus = configured_focus(config)
    primary = primary_cost_name(config)
    setup = validation[
        validation["bucket"].eq("setup_signal")
        & validation["cost_scenario"].eq(primary)
        & validation["family"].eq(focus["family"])
        & validation["direction"].eq(focus["direction"])
        & validation["horizon_bars"].astype(int).eq(focus["horizon_bars"])
    ].copy()
    if setup.empty:
        return pd.DataFrame()
    metrics = candidate_monthly_metrics(bar_returns)
    ranked = setup.merge(metrics, on=["candidate_id", "fold"], how="left", validate="one_to_one")
    return add_anti_scores(ranked, config).reset_index(drop=True)


def select_specs(ranked_validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if ranked_validation.empty:
        return pd.DataFrame(columns=SPEC_COLUMNS)
    cfg = _anti_cfg(config)
    selected = ranked_validation[ranked_validation["anti_status"].eq("anti_concentration_candidate")].copy()
    if selected.empty:
        return pd.DataFrame(columns=SPEC_COLUMNS)
    selected = selected.sort_values(["fold", "anti_score", "net_return"], ascending=[True, False, False], kind="stable")
    selected = selected.groupby("fold", group_keys=False, sort=False).head(int(cfg.get("max_selected_per_fold", 5)))
    cols = [column for column in SPEC_COLUMNS if column in selected.columns]
    return selected.loc[:, cols].reset_index(drop=True)


def _primary_passes(row: pd.Series, config: dict[str, Any]) -> bool:
    cfg = _anti_cfg(config)
    if row.empty:
        return False
    return bool(
        int(row.get("trades", 0)) >= int(cfg.get("min_trades", 40))
        and float(row.get("net_return", np.nan)) > 0.0
        and float(row.get("avg_trade_net", np.nan)) > 0.0
        and float(row.get("profit_factor", np.nan)) >= float(cfg.get("min_profit_factor", 1.05))
        and float(row.get("daily_sharpe", np.nan)) >= float(cfg.get("min_daily_sharpe", 0.30))
        and float(row.get("top_day_abs_net_share", np.inf)) <= float(cfg.get("max_top_day_abs_net_share", 0.30))
        and float(row.get("top_month_abs_net_share", np.inf)) <= float(cfg.get("max_top_month_abs_net_share", 0.45))
    )


def _setup_row(frame: pd.DataFrame, candidate_id: str, cost_scenario: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    rows = frame[
        frame["candidate_id"].eq(candidate_id) & frame["bucket"].eq("setup_signal") & frame["cost_scenario"].eq(cost_scenario)
    ]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def anti_decision_table(ranked_validation: pd.DataFrame, test: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    rows = []
    primary = primary_cost_name(config)
    conservative = conservative_cost_name(config)
    stress = stress_cost_name(config)
    for _, spec in specs.iterrows():
        candidate_id = str(spec["candidate_id"])
        validation_row = ranked_validation[ranked_validation["candidate_id"].eq(candidate_id)].iloc[0]
        primary_row = _setup_row(test, candidate_id, primary)
        conservative_row = _setup_row(test, candidate_id, conservative)
        stress_row = _setup_row(test, candidate_id, stress)
        primary_ok = _primary_passes(primary_row, config)
        conservative_ok = bool(
            not conservative_row.empty
            and float(conservative_row.get("net_return", np.nan)) > 0.0
            and float(conservative_row.get("avg_trade_net", np.nan)) > 0.0
        )
        stress_ok = bool(
            not stress_row.empty
            and float(stress_row.get("net_return", np.nan)) >= 0.0
            and float(stress_row.get("avg_trade_net", np.nan)) >= 0.0
        )
        if primary_ok and conservative_ok and stress_ok:
            decision = "accepted_candidate"
        elif primary_ok and conservative_ok:
            decision = "cost_fragile"
        elif not primary_row.empty and float(primary_row.get("net_return", np.nan)) > 0.0:
            decision = "research_candidate"
        else:
            decision = "rejected"
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold": int(spec["fold"]),
                "family": spec["family"],
                "direction": spec["direction"],
                "horizon_bars": int(spec["horizon_bars"]),
                "params_json": spec["params_json"],
                "validation_anti_status": validation_row.get("anti_status", ""),
                "validation_net": validation_row.get("net_return", np.nan),
                "validation_leave_one_month_min_net": validation_row.get("leave_one_month_min_net", np.nan),
                "validation_positive_month_rate": validation_row.get("positive_month_rate", np.nan),
                "validation_top_month_abs_share": validation_row.get("top_month_abs_net_share_rebuilt", np.nan),
                "decision": decision,
                "test_net_primary": primary_row.get("net_return", np.nan),
                "test_avg_trade_net_primary": primary_row.get("avg_trade_net", np.nan),
                "test_trades_primary": primary_row.get("trades", np.nan),
                "test_profit_factor_primary": primary_row.get("profit_factor", np.nan),
                "test_daily_sharpe_primary": primary_row.get("daily_sharpe", np.nan),
                "test_top_month_abs_net_share_primary": primary_row.get("top_month_abs_net_share", np.nan),
                "test_net_conservative": conservative_row.get("net_return", np.nan),
                "test_net_stress": stress_row.get("net_return", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "test_avg_trade_net_primary"], ascending=[True, False], kind="stable")


def render_report(
    target_symbol: str,
    ranked_validation: pd.DataFrame,
    selected_specs: pd.DataFrame,
    test: pd.DataFrame,
    decisions: pd.DataFrame,
    monthly: pd.DataFrame,
    outputs: dict[str, Path],
    config: dict[str, Any],
) -> str:
    cfg = _anti_cfg(config)
    focus = configured_focus(config)
    validation_counts = (
        ranked_validation["anti_status"].value_counts().rename_axis("anti_status").reset_index(name="rows")
        if not ranked_validation.empty
        else pd.DataFrame()
    )
    decision_counts = (
        decisions["decision"].value_counts().rename_axis("decision").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    )
    top_validation = ranked_validation.head(int(cfg.get("report_top_rows", 80))) if not ranked_validation.empty else ranked_validation
    setup_test = (
        test[test["bucket"].eq("setup_signal")]
        .sort_values(["fold", "cost_scenario", "net_return"], ascending=[True, True, False], kind="stable")
        .head(int(cfg.get("report_top_rows", 80)))
        if not test.empty
        else test
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    if decisions.empty:
        conclusion = "No validation candidate survived the anti-concentration rules. The family should be closed unless a new hypothesis changes the signal definition."
    elif decisions["decision"].eq("accepted_candidate").any():
        conclusion = "At least one anti-concentration candidate passed test checks. It can move to the next review gate."
    else:
        conclusion = "Anti-concentration produced validation candidates, but none became an accepted test candidate. Do not use HMM to rescue this family."

    return f"""# Setup Signal Anti-Concentration - {target_symbol.upper()}

## Scope

- Family: `{focus["family"]}`.
- Direction: `{focus["direction"]}`.
- Horizon bars: `{focus["horizon_bars"]}`.
- Selection uses validation only.
- Anti-concentration rules:
  - min months: `{cfg.get("min_months", 4)}`;
  - min positive months: `{cfg.get("min_positive_months", 3)}`;
  - max top month absolute share: `{cfg.get("max_top_month_abs_net_share", 0.45)}`;
  - min leave-one-month net: `{cfg.get("min_leave_one_month_net", 0.0)}`.

## Validation Status Counts

{_markdown_table(validation_counts)}

## Decision Counts

{_markdown_table(decision_counts)}

## Selected Specs

{_markdown_table(selected_specs, max_rows=int(cfg.get("report_top_rows", 80)))}

## Top Validation Candidates

{_markdown_table(top_validation, max_rows=int(cfg.get("report_top_rows", 80)))}

## Test Evaluation

{_markdown_table(setup_test, max_rows=int(cfg.get("report_top_rows", 80)))}

## Decisions

{_markdown_table(decisions, max_rows=int(cfg.get("report_top_rows", 80)))}

## Monthly Validation Reconstruction

{_markdown_table(monthly, max_rows=int(cfg.get("report_top_rows", 80)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    focused_config = focused_search_config(config)
    dataset = build_signal_dataset(focused_config, target)
    validation = validation_grid(dataset, focused_config) if not dataset.empty else pd.DataFrame()
    primary_rows = validation[validation["bucket"].eq("setup_signal") & validation["cost_scenario"].eq(primary_cost_name(focused_config))]
    scenarios = available_cost_scenarios(
        {**focused_config, "operable_candidate_search": _combined_cfg(focused_config)},
        [primary_cost_name(focused_config)],
    )
    primary_scenario = scenarios[0]
    bar_returns = (
        reconstruct_bar_returns(dataset, primary_rows, primary_scenario, splits=("validation",))
        if not primary_rows.empty
        else pd.DataFrame()
    )
    ranked_validation = rank_validation_candidates(validation, bar_returns, config)
    selected_specs = select_specs(ranked_validation, config)
    test = evaluate_selected_on_split(dataset, selected_specs, "test", focused_config) if not selected_specs.empty else pd.DataFrame()
    decisions = anti_decision_table(ranked_validation, test, selected_specs, config)
    monthly = summarize_returns(bar_returns, ["split", "fold", "candidate_id", "month"]) if not bar_returns.empty else pd.DataFrame()

    results_dir = results_output_dir(config, target)
    outputs = {
        "setup_signal_anti_validation": results_dir / "setup_signal_anti_validation.parquet",
        "setup_signal_anti_bar_returns": results_dir / "setup_signal_anti_bar_returns.parquet",
        "setup_signal_anti_ranked_validation": results_dir / "setup_signal_anti_ranked_validation.parquet",
        "setup_signal_anti_selected_specs": results_dir / "setup_signal_anti_selected_specs.parquet",
        "setup_signal_anti_test": results_dir / "setup_signal_anti_test.parquet",
        "setup_signal_anti_decisions": results_dir / "setup_signal_anti_decisions.parquet",
        "setup_signal_anti_monthly": results_dir / "setup_signal_anti_monthly.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(outputs["setup_signal_anti_validation"], index=False)
    bar_returns.to_parquet(outputs["setup_signal_anti_bar_returns"], index=False)
    ranked_validation.to_parquet(outputs["setup_signal_anti_ranked_validation"], index=False)
    selected_specs.to_parquet(outputs["setup_signal_anti_selected_specs"], index=False)
    test.to_parquet(outputs["setup_signal_anti_test"], index=False)
    decisions.to_parquet(outputs["setup_signal_anti_decisions"], index=False)
    monthly.to_parquet(outputs["setup_signal_anti_monthly"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(target, ranked_validation, selected_specs, test, decisions, monthly, outputs, config),
        encoding="utf-8",
    )
    return report_path, outputs["setup_signal_anti_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one validation-only anti-concentration refinement for a setup family.")
    parser.add_argument("--config", default="configs/hmm_lab_15min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Setup signal anti-concentration report written to: {report_path}")
    print(f"Setup signal anti-concentration decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

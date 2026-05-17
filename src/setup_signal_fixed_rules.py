from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios
from src.setup_signal_search import (
    _base_mask,
    _json_dumps,
    _signal_column_map,
    build_signal_dataset,
    evaluate_position,
    signal_mask,
)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_fixed_rules", {})


def _targets(config: dict[str, Any]) -> list[str]:
    configured = _cfg(config).get("targets")
    if configured:
        return [str(value).upper() for value in configured]
    return [_target_symbol(config)]


def _rule_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for raw in _cfg(config).get("rules", []):
        params = {str(key): value for key, value in dict(raw.get("params", {})).items()}
        direction = str(raw.get("direction", params.get("direction", "long")))
        params["direction"] = direction
        specs.append(
            {
                "rule_name": str(raw["name"]),
                "family": str(raw["family"]),
                "direction": direction,
                "horizon_bars": int(raw["horizon_bars"]),
                "params": params,
                "column_map": {str(k): str(v) for k, v in dict(raw.get("column_map", {})).items()},
            }
        )
    return specs


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    fixed = _cfg(config)
    results_dir = Path(fixed.get("results_dir", Path(config.get("paths", {}).get("results_dir", "results")) / "_fixed_rules"))
    reports_dir = Path(fixed.get("reports_dir", Path(config.get("paths", {}).get("reports_dir", "reports")) / "_fixed_rules"))
    return {
        "fold_metrics": results_dir / "setup_signal_fixed_rule_fold_metrics.parquet",
        "target_summary": results_dir / "setup_signal_fixed_rule_target_summary.parquet",
        "rule_summary": results_dir / "setup_signal_fixed_rule_summary.parquet",
        "report": reports_dir / "setup_signal_fixed_rule_summary.md",
    }


def evaluate_fixed_rules_for_target(config: dict[str, Any], target: str) -> pd.DataFrame:
    dataset = build_signal_dataset(config, target)
    if dataset.empty:
        return pd.DataFrame()
    rules = _rule_specs(config)
    if not rules:
        return pd.DataFrame()
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": dict(config.get("setup_signal_search", {}))})
    rows: list[dict[str, Any]] = []
    for rule in rules:
        horizon = int(rule["horizon_bars"])
        params = dict(rule["params"])
        columns = _signal_column_map(config, rule["column_map"])
        direction = 1.0 if rule["direction"] == "long" else -1.0
        for keys, group in dataset[dataset["horizon_bars"].eq(horizon)].groupby(["fold", "split"], sort=False):
            fold, split = keys
            signal = signal_mask(group, rule["family"], params, columns)
            base = _base_mask(group, rule["family"], params, columns)
            for bucket, mask in [
                ("fixed_rule", signal),
                ("base_segment_control", base),
                ("always_flat", pd.Series(False, index=group.index)),
            ]:
                position = pd.Series(0.0, index=group.index)
                position.loc[mask] = direction
                for scenario in scenarios:
                    rows.append(
                        {
                            "target": target.upper(),
                            "rule_name": rule["rule_name"],
                            "family": rule["family"],
                            "direction": rule["direction"],
                            "horizon_bars": horizon,
                            "params_json": _json_dumps(params),
                            "column_map_json": _json_dumps(columns),
                            "fold": int(fold),
                            "split": str(split),
                            "bucket": bucket,
                            "cost_scenario": str(scenario["cost_scenario"]),
                            "cost_kind": str(scenario["cost_kind"]),
                            **evaluate_position(group, position, scenario),
                        }
                    )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    rule_rows = metrics[metrics["bucket"].eq("fixed_rule")].copy()
    control_rows = metrics[metrics["bucket"].eq("base_segment_control")].copy()
    control = control_rows.set_index(["target", "rule_name", "fold", "split", "cost_scenario"])
    deltas = []
    for _, row in rule_rows.iterrows():
        key = (row["target"], row["rule_name"], int(row["fold"]), row["split"], row["cost_scenario"])
        control_row = control.loc[key] if key in control.index else pd.Series(dtype=object)
        deltas.append(
            {
                "net_delta_vs_base_segment": float(row["net_return"] - control_row.get("net_return", 0.0)),
                "avg_trade_net_delta_vs_base_segment": float(row["avg_trade_net"] - control_row.get("avg_trade_net", 0.0)),
            }
        )
    metrics.loc[rule_rows.index, ["net_delta_vs_base_segment", "avg_trade_net_delta_vs_base_segment"]] = pd.DataFrame(
        deltas, index=rule_rows.index
    )
    return metrics.reset_index(drop=True)


def aggregate_target_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    grouped = (
        fold_metrics[fold_metrics["bucket"].eq("fixed_rule")]
        .groupby(
            [
                "target",
                "rule_name",
                "family",
                "direction",
                "horizon_bars",
                "params_json",
                "split",
                "cost_scenario",
                "cost_kind",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            trades=("trades", "sum"),
            exposure=("exposure", "mean"),
            gross_return=("gross_return", "sum"),
            cost_return=("cost_return", "sum"),
            net_return=("net_return", "sum"),
            avg_trade_net_mean=("avg_trade_net", "mean"),
            min_fold_avg_trade_net=("avg_trade_net", "min"),
            hit_rate_mean=("hit_rate", "mean"),
            profit_factor_median=("profit_factor", "median"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_day_abs_net_share_max=("top_day_abs_net_share", "max"),
            top_month_abs_net_share_max=("top_month_abs_net_share", "max"),
            net_delta_vs_base_segment=("net_delta_vs_base_segment", "sum"),
        )
    )
    grouped["avg_trade_net_pooled"] = grouped["net_return"] / grouped["trades"].replace(0, np.nan)
    grouped["positive_fold_share"] = grouped["positive_folds"] / grouped["folds"].replace(0, np.nan)
    return grouped.sort_values(["split", "cost_scenario", "net_return"], ascending=[True, True, False], kind="stable").reset_index(drop=True)


def aggregate_rule_summary(target_summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if target_summary.empty:
        return pd.DataFrame()
    fixed = _cfg(config)
    primary = str(fixed.get("primary_cost_scenario", config.get("setup_signal_search", {}).get("primary_cost_scenario", "ibkr_tiered_10000")))
    conservative = str(fixed.get("conservative_cost_scenario", config.get("setup_signal_search", {}).get("conservative_cost_scenario", "bps_2")))
    stress = str(fixed.get("stress_cost_scenario", config.get("setup_signal_search", {}).get("stress_cost_scenario", "bps_5")))
    min_targets = int(fixed.get("min_positive_targets", min(3, len(_targets(config)))))
    rows: list[dict[str, Any]] = []
    test = target_summary[target_summary["split"].eq("test")].copy()
    for keys, group in test.groupby(["rule_name", "family", "direction", "horizon_bars", "params_json"], sort=False):
        rule_name, family, direction, horizon, params_json = keys
        primary_rows = group[group["cost_scenario"].eq(primary)]
        conservative_rows = group[group["cost_scenario"].eq(conservative)]
        stress_rows = group[group["cost_scenario"].eq(stress)]

        def _positive_targets(frame: pd.DataFrame) -> int:
            return int((frame["net_return"].gt(0.0) & frame["avg_trade_net_pooled"].gt(0.0)).sum())

        rows.append(
            {
                "rule_name": rule_name,
                "family": family,
                "direction": direction,
                "horizon_bars": int(horizon),
                "params_json": params_json,
                "targets": int(primary_rows["target"].nunique()),
                "primary_positive_targets": _positive_targets(primary_rows),
                "conservative_positive_targets": _positive_targets(conservative_rows),
                "stress_nonnegative_targets": int(
                    (stress_rows["net_return"].ge(0.0) & stress_rows["avg_trade_net_pooled"].ge(0.0)).sum()
                ),
                "median_primary_net_return": float(primary_rows["net_return"].median()) if not primary_rows.empty else np.nan,
                "median_primary_avg_trade_net_bps": float(primary_rows["avg_trade_net_pooled"].median() * 10_000.0)
                if not primary_rows.empty
                else np.nan,
                "min_primary_avg_trade_net_bps": float(primary_rows["avg_trade_net_pooled"].min() * 10_000.0)
                if not primary_rows.empty
                else np.nan,
                "median_stress_net_return": float(stress_rows["net_return"].median()) if not stress_rows.empty else np.nan,
                "median_primary_sharpe": float(primary_rows["daily_sharpe_mean"].median()) if not primary_rows.empty else np.nan,
                "max_primary_drawdown": float(primary_rows["max_drawdown_max"].max()) if not primary_rows.empty else np.nan,
                "best_targets_primary": ", ".join(
                    primary_rows.loc[primary_rows["net_return"].gt(0.0), "target"].astype(str).sort_values().tolist()
                ),
                "promotable_family": bool(
                    _positive_targets(primary_rows) >= min_targets
                    and _positive_targets(conservative_rows) >= min_targets
                    and int((stress_rows["net_return"].ge(0.0) & stress_rows["avg_trade_net_pooled"].ge(0.0)).sum()) >= min_targets
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "promotable_family",
            "stress_nonnegative_targets",
            "primary_positive_targets",
            "median_primary_avg_trade_net_bps",
        ],
        ascending=[False, False, False, False],
        kind="stable",
    )


def render_report(
    config: dict[str, Any],
    target_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    fixed = _cfg(config)
    targets = ", ".join(_targets(config))
    rule_names = ", ".join(rule["rule_name"] for rule in _rule_specs(config))
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    target_test = target_summary[target_summary["split"].eq("test")] if not target_summary.empty else pd.DataFrame()
    target_test_primary = (
        target_test[target_test["cost_scenario"].eq(str(fixed.get("primary_cost_scenario", "ibkr_tiered_10000")))]
        if not target_test.empty
        else pd.DataFrame()
    )
    conclusion = (
        "At least one frozen rule passes the configured multi-asset gate."
        if not rule_summary.empty and rule_summary["promotable_family"].any()
        else "No frozen rule passes the configured multi-asset gate."
    )
    return f"""# H9 Fixed Rule Evaluation

## Scope

- Targets: `{targets}`
- Frozen rules: `{rule_names}`
- Primary cost: `{fixed.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Conservative cost: `{fixed.get("conservative_cost_scenario", "bps_2")}`
- Stress cost: `{fixed.get("stress_cost_scenario", "bps_5")}`

## Rule Summary

{_markdown_table(rule_summary, max_rows=int(fixed.get("report_top_rows", 80)))}

## Target Test Summary - Primary Cost

{_markdown_table(target_test_primary, max_rows=int(fixed.get("report_top_rows", 120)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    parts = [evaluate_fixed_rules_for_target(config, target) for target in _targets(config)]
    fold_metrics = pd.concat([part for part in parts if not part.empty], ignore_index=True, sort=False) if parts else pd.DataFrame()
    target_summary = aggregate_target_summary(fold_metrics)
    rule_summary = aggregate_rule_summary(target_summary, config)
    outputs = _output_paths(config)
    outputs["fold_metrics"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_parquet(outputs["fold_metrics"], index=False)
    target_summary.to_parquet(outputs["target_summary"], index=False)
    rule_summary.to_parquet(outputs["rule_summary"], index=False)
    outputs["report"].write_text(render_report(config, target_summary, rule_summary, outputs), encoding="utf-8")
    return outputs["report"], outputs["rule_summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen setup rules across targets and costs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report_path, summary_path = run(args.config)
    print(f"H9 fixed-rule report written to: {report_path}")
    print(f"H9 fixed-rule summary written to: {summary_path}")


if __name__ == "__main__":
    main()

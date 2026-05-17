from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table


DECISION_RANK = {
    "accepted_candidate": 0,
    "cost_fragile": 1,
    "research_candidate": 2,
    "rejected": 3,
}


def _multiasset_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_multiasset", {})


def _targets(config: dict[str, Any]) -> list[str]:
    configured = _multiasset_cfg(config).get("targets")
    if configured:
        return [str(symbol).upper() for symbol in configured]
    return [_target_symbol(config)]


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    cfg = _multiasset_cfg(config)
    results_dir = Path(cfg.get("results_dir", Path(config.get("paths", {}).get("results_dir", "results")) / "_multiasset"))
    reports_dir = Path(cfg.get("reports_dir", Path(config.get("paths", {}).get("reports_dir", "reports")) / "_multiasset"))
    return {
        "multiasset_decisions": results_dir / "setup_signal_multiasset_decisions.parquet",
        "multiasset_best_by_target_family": results_dir / "setup_signal_multiasset_best_by_target_family.parquet",
        "multiasset_family_summary": results_dir / "setup_signal_multiasset_family_summary.parquet",
        "multiasset_report": reports_dir / "setup_signal_multiasset_summary.md",
    }


def load_target_decisions(config: dict[str, Any], targets: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for target in targets or _targets(config):
        path = results_output_dir(config, target) / "setup_signal_decisions.parquet"
        if not path.exists():
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "target": target,
                            "candidate_id": "",
                            "family": "",
                            "direction": "",
                            "horizon_bars": np.nan,
                            "decision": "missing_results",
                        }
                    ]
                )
            )
            continue
        decisions = pd.read_parquet(path)
        if decisions.empty:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "target": target,
                            "candidate_id": "",
                            "family": "",
                            "direction": "",
                            "horizon_bars": np.nan,
                            "decision": "no_candidates",
                        }
                    ]
                )
            )
            continue
        decisions = decisions.copy()
        decisions.insert(0, "target", target)
        rows.append(decisions)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def best_by_target_family(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    required = {"target", "family", "direction", "horizon_bars", "decision"}
    valid = decisions.dropna(subset=["horizon_bars"]).copy() if required.issubset(decisions.columns) else pd.DataFrame()
    valid = valid[valid["family"].astype(str).ne("")]
    if valid.empty:
        return pd.DataFrame()
    valid["decision_rank"] = valid["decision"].map(DECISION_RANK).fillna(9).astype(int)
    valid["primary_positive"] = valid["test_net_primary"].gt(0.0) & valid["test_avg_trade_net_primary"].gt(0.0)
    valid["stress_nonnegative"] = valid["test_net_stress"].ge(0.0)
    valid["research_or_better"] = valid["decision"].isin(["accepted_candidate", "cost_fragile", "research_candidate"])
    return (
        valid.sort_values(
            [
                "target",
                "family",
                "direction",
                "horizon_bars",
                "decision_rank",
                "test_avg_trade_net_primary",
                "test_net_primary",
            ],
            ascending=[True, True, True, True, True, False, False],
            kind="stable",
        )
        .drop_duplicates(["target", "family", "direction", "horizon_bars"], keep="first")
        .reset_index(drop=True)
    )


def family_summary(best: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    cfg = _multiasset_cfg(config)
    target_count = len(_targets(config))
    min_targets = int(cfg.get("min_targets", min(3, target_count)))
    min_research_targets = int(cfg.get("min_research_targets", min_targets))
    min_primary_positive_targets = int(cfg.get("min_primary_positive_targets", min_targets))
    min_stress_nonnegative_targets = int(cfg.get("min_stress_nonnegative_targets", min_targets))
    rows: list[dict[str, Any]] = []
    for keys, group in best.groupby(["family", "direction", "horizon_bars"], sort=False):
        family, direction, horizon = keys
        primary = group["primary_positive"].fillna(False)
        stress = group["stress_nonnegative"].fillna(False)
        research = group["research_or_better"].fillna(False)
        accepted = group["decision"].eq("accepted_candidate")
        cost_fragile = group["decision"].eq("cost_fragile")
        rows.append(
            {
                "family": family,
                "direction": direction,
                "horizon_bars": int(horizon),
                "targets_present": int(group["target"].nunique()),
                "accepted_targets": int(accepted.sum()),
                "cost_fragile_or_better_targets": int((accepted | cost_fragile).sum()),
                "research_or_better_targets": int(research.sum()),
                "primary_positive_targets": int(primary.sum()),
                "stress_nonnegative_targets": int(stress.sum()),
                "median_test_net_primary": float(group["test_net_primary"].median()),
                "median_test_avg_trade_net_primary": float(group["test_avg_trade_net_primary"].median()),
                "min_target_test_avg_trade_net_primary": float(group["test_avg_trade_net_primary"].min()),
                "median_test_net_stress": float(group["test_net_stress"].median()),
                "best_targets": ", ".join(group.loc[research, "target"].astype(str).sort_values().tolist()),
                "stable_family": bool(
                    group["target"].nunique() >= min_targets
                    and int(research.sum()) >= min_research_targets
                    and int(primary.sum()) >= min_primary_positive_targets
                    and int(stress.sum()) >= min_stress_nonnegative_targets
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "stable_family",
            "stress_nonnegative_targets",
            "primary_positive_targets",
            "research_or_better_targets",
            "median_test_avg_trade_net_primary",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    )


def render_report(
    config: dict[str, Any],
    decisions: pd.DataFrame,
    best: pd.DataFrame,
    summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _multiasset_cfg(config)
    targets = ", ".join(_targets(config))
    decision_counts = (
        decisions.groupby(["target", "decision"], as_index=False).size().rename(columns={"size": "rows"})
        if not decisions.empty and {"target", "decision"}.issubset(decisions.columns)
        else pd.DataFrame()
    )
    top = (
        decisions.assign(decision_rank=decisions["decision"].map(DECISION_RANK).fillna(9).astype(int))
        .sort_values(["decision_rank", "test_avg_trade_net_primary", "test_net_primary"], ascending=[True, False, False], kind="stable")
        .head(int(cfg.get("report_top_rows", 80)))
        if not decisions.empty and "test_avg_trade_net_primary" in decisions.columns
        else decisions
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one setup family is stable across the configured multi-asset gate."
        if not summary.empty and summary["stable_family"].any()
        else "No setup family passes the configured multi-asset stability gate."
    )
    return f"""# H9 Setup-First Multi-Asset Summary

## Scope

- Targets: `{targets}`
- Search output root: `{config.get("paths", {}).get("results_dir", "results")}`
- Multi-asset stability uses best candidate per target/family/direction/horizon.

## Decision Counts By Target

{_markdown_table(decision_counts)}

## Multi-Asset Family Summary

{_markdown_table(summary, max_rows=int(cfg.get("report_top_rows", 80)))}

## Best By Target And Family

{_markdown_table(best, max_rows=int(cfg.get("report_top_rows", 80)))}

## Top Decisions

{_markdown_table(top, max_rows=int(cfg.get("report_top_rows", 80)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    targets = _targets(config)
    decisions = load_target_decisions(config, targets)
    best = best_by_target_family(decisions)
    summary = family_summary(best, config)
    outputs = _output_paths(config)
    outputs["multiasset_decisions"].parent.mkdir(parents=True, exist_ok=True)
    outputs["multiasset_report"].parent.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(outputs["multiasset_decisions"], index=False)
    best.to_parquet(outputs["multiasset_best_by_target_family"], index=False)
    summary.to_parquet(outputs["multiasset_family_summary"], index=False)
    outputs["multiasset_report"].write_text(render_report(config, decisions, best, summary, outputs), encoding="utf-8")
    return outputs["multiasset_report"], outputs["multiasset_family_summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate setup-first search results across multiple target symbols.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report_path, summary_path = run(args.config)
    print(f"H9 multi-asset summary report written to: {report_path}")
    print(f"H9 multi-asset family summary written to: {summary_path}")


if __name__ == "__main__":
    main()

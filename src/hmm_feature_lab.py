from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.hmm_state_economics import (
    classify_candidate,
    run_fold_state_economics,
    _prepare_horizon_labels,
)
from src.walkforward import build_monthly_folds


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _feature_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    lab_cfg = config.get("hmm_feature_lab", {})
    sets = lab_cfg.get("feature_sets", [])
    if not sets:
        sets = [{"name": "current_default", "columns": list(config["hmm"]["feature_columns"])}]
    return [{"name": str(item["name"]), "columns": list(item["columns"])} for item in sets]


def _config_for_feature_set(config: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    lab_cfg = config.get("hmm_feature_lab", {})
    copied = deepcopy(config)
    copied["hmm"]["feature_columns"] = list(columns)
    copied["hmm"]["n_states"] = int(lab_cfg.get("n_states", copied["hmm"].get("n_states", 4)))
    copied["hmm"]["n_iter"] = int(lab_cfg.get("n_iter", copied["hmm"].get("n_iter", 200)))
    copied.setdefault("robustness", {})
    copied["robustness"]["horizons"] = [int(value) for value in lab_cfg.get("horizons", copied["robustness"].get("horizons", [2]))]
    copied["robustness"]["cost_bps"] = [float(value) for value in lab_cfg.get("cost_bps", copied["robustness"].get("cost_bps", [1.0]))]
    return copied


def _validate_feature_sets(features: pd.DataFrame, feature_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    available = set(features.columns)
    for item in feature_sets:
        missing = sorted(set(item["columns"]) - available)
        rows.append(
            {
                "feature_set": item["name"],
                "n_features": len(item["columns"]),
                "columns": ",".join(item["columns"]),
                "missing_columns": ",".join(missing),
                "status": "ready" if not missing else "missing_columns",
            }
        )
    return rows


def _select_folds(labels: pd.DataFrame, config: dict[str, Any]) -> list[Any]:
    folds = build_monthly_folds(labels, config)
    max_folds = config.get("hmm_feature_lab", {}).get("max_folds")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    return folds


def run_feature_set(
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
    folds: list[Any],
    base_config: dict[str, Any],
    feature_set: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_config = _config_for_feature_set(base_config, feature_set["columns"])
    rows: list[dict[str, Any]] = []
    hour_frames: list[pd.DataFrame] = []
    for fold in folds:
        fold_rows, fold_hours = run_fold_state_economics(features, labels_by_horizon, fold, set_config)
        for row in fold_rows:
            row["feature_set"] = feature_set["name"]
            row["feature_columns"] = ",".join(feature_set["columns"])
        for frame in fold_hours:
            annotated = frame.copy()
            annotated["feature_set"] = feature_set["name"]
            hour_frames.append(annotated)
        rows.extend(fold_rows)
    metrics = pd.DataFrame(rows)
    hours = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    return metrics, hours


def aggregate_feature_set_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics[(metrics["split"] == "validation") & (metrics["action"] != "flat")].copy()
    if active.empty:
        return pd.DataFrame()
    grouped = (
        active.groupby(["feature_set", "horizon_bars", "cost_bps", "hmm_state", "action"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_trades=("trades", "sum"),
            total_net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            median_profit_factor=("profit_factor", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            negative_folds=("net_return", lambda values: int((values < 0).sum())),
            avg_state_frequency=("state_frequency", "mean"),
            avg_persistence=("persistence", "mean"),
        )
        .sort_values(["total_net_return", "avg_trade_net", "total_trades"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    grouped["candidate_status"] = grouped.apply(classify_candidate, axis=1)
    return grouped


def summarize_feature_sets(metrics: pd.DataFrame, ranking: pd.DataFrame, validations: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return validations.copy()
    non_random = ranking[ranking["action"] != "random_symmetric"].copy()
    best_rows = (
        non_random.sort_values(["feature_set", "total_net_return", "avg_trade_net"], ascending=[True, False, False])
        .groupby("feature_set", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    candidates = non_random[non_random["candidate_status"] == "candidate"].groupby("feature_set").size().rename("candidate_count")
    state_quality = (
        metrics[(metrics["split"] == "validation") & (metrics["action"] == "flat")]
        .groupby("feature_set", as_index=False)
        .agg(
            avg_state_frequency=("state_frequency", "mean"),
            avg_persistence=("persistence", "mean"),
            avg_mean_duration=("mean_duration", "mean"),
        )
    )
    summary = validations.merge(
        best_rows[
            [
                "feature_set",
                "horizon_bars",
                "cost_bps",
                "hmm_state",
                "action",
                "total_trades",
                "total_net_return",
                "avg_trade_net",
                "median_profit_factor",
                "median_daily_sharpe",
                "candidate_status",
            ]
        ].rename(columns={col: f"best_{col}" for col in best_rows.columns if col != "feature_set"}),
        on="feature_set",
        how="left",
    )
    summary = summary.merge(candidates.reset_index(), on="feature_set", how="left")
    summary = summary.merge(state_quality, on="feature_set", how="left")
    summary["candidate_count"] = summary["candidate_count"].fillna(0).astype(int)
    return summary


def candidate_holdout_summary(metrics: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or ranking.empty:
        return pd.DataFrame()
    keys = ["feature_set", "horizon_bars", "cost_bps", "hmm_state", "action"]
    candidates = ranking.loc[ranking["candidate_status"] == "candidate", keys].drop_duplicates()
    if candidates.empty:
        return pd.DataFrame()
    selected = metrics.merge(candidates, on=keys, how="inner")
    grouped = (
        selected.groupby(keys + ["split"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_trades=("trades", "sum"),
            total_net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            median_profit_factor=("profit_factor", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            negative_folds=("net_return", lambda values: int((values < 0).sum())),
            avg_state_frequency=("state_frequency", "mean"),
            avg_persistence=("persistence", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["candidate_status"] = grouped.apply(classify_candidate, axis=1)
    split_order = {"validation": 0, "test": 1}
    grouped["_split_order"] = grouped["split"].map(split_order).fillna(9).astype(int)
    grouped = grouped.sort_values(keys + ["_split_order"]).drop(columns=["_split_order"]).reset_index(drop=True)
    return grouped


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_lab_report(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    holdout: pd.DataFrame,
    validations: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    lab_cfg = config.get("hmm_feature_lab", {})
    top_cols = [
        "feature_set",
        "horizon_bars",
        "cost_bps",
        "hmm_state",
        "action",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "candidate_status",
    ]
    holdout_cols = [
        "feature_set",
        "horizon_bars",
        "cost_bps",
        "hmm_state",
        "action",
        "split",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "candidate_status",
    ]
    candidates = ranking[ranking["candidate_status"] == "candidate"] if not ranking.empty else pd.DataFrame()
    test_candidates = holdout[(holdout["split"] == "test") & (holdout["candidate_status"] == "candidate")] if not holdout.empty else pd.DataFrame()
    if candidates.empty:
        conclusion = "No configured feature set produced a validation candidate under the lab filters."
    elif test_candidates.empty:
        conclusion = "Validation candidates were found, but none passed the same candidate filters on test. Treat them as hypotheses for redesign/stability work, not as accepted edge."
    else:
        conclusion = "At least one validation candidate also passed the same filters on test. It still needs full seed/K stability and a final frozen OOS evaluation."
    return f"""# HMM Feature Lab

## Scope

- Feature sets: {len(validations)}
- Max folds: `{lab_cfg.get("max_folds")}`
- Horizons: `{lab_cfg.get("horizons")}`
- Costs bps: `{lab_cfg.get("cost_bps")}`
- HMM K: `{lab_cfg.get("n_states", config["hmm"]["n_states"])}`
- HMM n_iter: `{lab_cfg.get("n_iter", config["hmm"].get("n_iter", 200))}`

## Feature Set Summary

{_markdown_table(summary)}

## Top Validation Rankings

{_markdown_table(ranking.loc[:, top_cols] if not ranking.empty else ranking, max_rows=30)}

## Validation Candidate Holdout Sanity

{_markdown_table(holdout.loc[:, holdout_cols] if not holdout.empty else holdout)}

## Feature Set Validation

{_markdown_table(validations)}

## Outputs

- `{lab_cfg.get("output_dir", "reports/hmm_feature_lab")}/feature_set_metrics.parquet`
- `{lab_cfg.get("output_dir", "reports/hmm_feature_lab")}/feature_set_ranking.parquet`
- `{lab_cfg.get("output_dir", "reports/hmm_feature_lab")}/feature_set_summary.parquet`
- `{lab_cfg.get("output_dir", "reports/hmm_feature_lab")}/feature_set_holdout.parquet`
- `{lab_cfg.get("output_dir", "reports/hmm_feature_lab")}/feature_set_hour_distribution.parquet`

## Conclusion

{conclusion}

This lab is for iterative feature-set screening. Do not accept a feature set from this report alone; promote only interpretable sets to full seed/K/fold stability tests and frozen OOS evaluation.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    lab_cfg = config.get("hmm_feature_lab", {})
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    feature_sets = _feature_sets(config)
    validations = pd.DataFrame(_validate_feature_sets(features, feature_sets))
    ready_sets = [item for item in feature_sets if validations.loc[validations["feature_set"] == item["name"], "status"].iloc[0] == "ready"]

    horizons = [int(value) for value in lab_cfg.get("horizons", [config["labeling"]["horizon_bars"]])]
    labels_by_horizon = _prepare_horizon_labels(features, config, horizons)
    folds = _select_folds(labels, config)

    metric_frames: list[pd.DataFrame] = []
    hour_frames: list[pd.DataFrame] = []
    for feature_set in ready_sets:
        metrics, hours = run_feature_set(features, labels_by_horizon, folds, config, feature_set)
        metric_frames.append(metrics)
        if not hours.empty:
            hour_frames.append(hours)

    all_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    all_hours = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    ranking = aggregate_feature_set_ranking(all_metrics) if not all_metrics.empty else pd.DataFrame()
    holdout = candidate_holdout_summary(all_metrics, ranking) if not all_metrics.empty else pd.DataFrame()
    summary = summarize_feature_sets(all_metrics, ranking, validations) if not all_metrics.empty else validations

    output_dir = Path(lab_cfg.get("output_dir", "reports/hmm_feature_lab"))
    report_path = Path(lab_cfg.get("report_file", "reports/hmm_feature_lab.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics.to_parquet(output_dir / "feature_set_metrics.parquet", index=False)
    ranking.to_parquet(output_dir / "feature_set_ranking.parquet", index=False)
    summary.to_parquet(output_dir / "feature_set_summary.parquet", index=False)
    holdout.to_parquet(output_dir / "feature_set_holdout.parquet", index=False)
    all_hours.to_parquet(output_dir / "feature_set_hour_distribution.parquet", index=False)
    report_path.write_text(render_lab_report(summary, ranking, holdout, validations, config), encoding="utf-8")
    return output_dir / "feature_set_summary.parquet", report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative HMM feature-set lab.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    summary_path, report_path = run(args.config)
    print(f"HMM feature lab summary written to: {summary_path}")
    print(f"HMM feature lab report written to: {report_path}")


if __name__ == "__main__":
    main()

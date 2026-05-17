from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.hmm_state_economics import classify_candidate, run_fold_state_economics, _prepare_horizon_labels
from src.walkforward import build_monthly_folds


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _feature_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    stability_cfg = config.get("hmm_stability", {})
    sets = stability_cfg.get("feature_sets", [])
    if not sets:
        lab_sets = config.get("hmm_feature_lab", {}).get("feature_sets", [])
        sets = [item for item in lab_sets if item.get("name") in {"rich_extreme_reversion", "minimal_vwap_location"}]
    return [{"name": str(item["name"]), "columns": list(item["columns"])} for item in sets]


def _validate_feature_sets(features: pd.DataFrame, feature_sets: list[dict[str, Any]]) -> pd.DataFrame:
    available = set(features.columns)
    rows = []
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
    return pd.DataFrame(rows)


def _combo_config(base_config: dict[str, Any], columns: list[str], n_states: int, seed: int) -> dict[str, Any]:
    stability_cfg = base_config.get("hmm_stability", {})
    copied = deepcopy(base_config)
    copied["hmm"]["feature_columns"] = list(columns)
    copied["hmm"]["n_states"] = int(n_states)
    copied["hmm"]["random_state"] = int(seed)
    copied["hmm"]["n_iter"] = int(stability_cfg.get("n_iter", copied["hmm"].get("n_iter", 200)))
    copied.setdefault("robustness", {})
    copied["robustness"]["horizons"] = [int(value) for value in stability_cfg.get("horizons", [6])]
    copied["robustness"]["cost_bps"] = [float(value) for value in stability_cfg.get("cost_bps", [1.0, 2.0])]
    return copied


def _select_folds(labels: pd.DataFrame, config: dict[str, Any]) -> list[Any]:
    folds = build_monthly_folds(labels, config)
    max_folds = config.get("hmm_stability", {}).get("max_folds")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    return folds


def run_stability_combo(
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
    folds: list[Any],
    base_config: dict[str, Any],
    feature_set: dict[str, Any],
    n_states: int,
    seed: int,
) -> pd.DataFrame:
    combo_config = _combo_config(base_config, feature_set["columns"], n_states, seed)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_rows, _ = run_fold_state_economics(features, labels_by_horizon, fold, combo_config)
        for row in fold_rows:
            row["feature_set"] = feature_set["name"]
            row["feature_columns"] = ",".join(feature_set["columns"])
            row["n_states"] = int(n_states)
            row["seed"] = int(seed)
        rows.extend(fold_rows)
    return pd.DataFrame(rows)


def aggregate_stability_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics[(metrics["split"] == "validation") & (metrics["action"] != "flat")].copy()
    if active.empty:
        return pd.DataFrame()
    grouped = (
        active.groupby(["feature_set", "n_states", "seed", "horizon_bars", "cost_bps", "hmm_state", "action"], as_index=False)
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


def candidate_holdout_summary(metrics: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or ranking.empty:
        return pd.DataFrame()
    keys = ["feature_set", "n_states", "seed", "horizon_bars", "cost_bps", "hmm_state", "action"]
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
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(keys + ["_split_order"]).drop(columns=["_split_order"]).reset_index(drop=True)


def summarize_combos(ranking: pd.DataFrame, holdout: pd.DataFrame, validations: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    stability_cfg = config.get("hmm_stability", {})
    combo_index = pd.MultiIndex.from_product(
        [
            validations.loc[validations["status"] == "ready", "feature_set"].tolist(),
            [int(value) for value in stability_cfg.get("n_states", [3, 4, 5])],
            [int(value) for value in stability_cfg.get("seeds", config["hmm"].get("stability_seeds", [42]))],
            [float(value) for value in stability_cfg.get("cost_bps", [1.0, 2.0])],
        ],
        names=["feature_set", "n_states", "seed", "cost_bps"],
    ).to_frame(index=False)
    if ranking.empty:
        combo_index["validation_candidates"] = 0
        combo_index["test_candidates"] = 0
        return combo_index

    validation_candidates = (
        ranking[ranking["candidate_status"] == "candidate"]
        .groupby(["feature_set", "n_states", "seed", "cost_bps"])
        .size()
        .rename("validation_candidates")
    )
    if holdout.empty:
        test_candidates = pd.Series(dtype=int, name="test_candidates")
    else:
        test_candidates = (
            holdout[(holdout["split"] == "test") & (holdout["candidate_status"] == "candidate")]
            .groupby(["feature_set", "n_states", "seed", "cost_bps"])
            .size()
            .rename("test_candidates")
        )
    ranking_for_best = ranking.copy()
    ranking_for_best["_is_candidate"] = (ranking_for_best["candidate_status"] == "candidate").astype(int)
    best_validation = (
        ranking_for_best.sort_values(
            ["feature_set", "n_states", "seed", "cost_bps", "_is_candidate", "total_net_return"],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(["feature_set", "n_states", "seed", "cost_bps"], as_index=False)
        .head(1)
        .loc[
            :,
            [
                "feature_set",
                "n_states",
                "seed",
                "cost_bps",
                "horizon_bars",
                "hmm_state",
                "action",
                "total_trades",
                "total_net_return",
                "avg_trade_net",
                "median_profit_factor",
                "median_daily_sharpe",
                "candidate_status",
            ],
        ]
        .rename(columns={col: f"best_validation_{col}" for col in ["horizon_bars", "hmm_state", "action", "total_trades", "total_net_return", "avg_trade_net", "median_profit_factor", "median_daily_sharpe", "candidate_status"]})
    )
    summary = combo_index.merge(validation_candidates.reset_index(), on=["feature_set", "n_states", "seed", "cost_bps"], how="left")
    summary = summary.merge(test_candidates.reset_index(), on=["feature_set", "n_states", "seed", "cost_bps"], how="left")
    summary = summary.merge(best_validation, on=["feature_set", "n_states", "seed", "cost_bps"], how="left")
    summary["validation_candidates"] = summary["validation_candidates"].fillna(0).astype(int)
    summary["test_candidates"] = summary["test_candidates"].fillna(0).astype(int)
    return summary


def summarize_feature_sets(combo_summary: pd.DataFrame) -> pd.DataFrame:
    if combo_summary.empty:
        return pd.DataFrame()
    return (
        combo_summary.groupby(["feature_set", "cost_bps"], as_index=False)
        .agg(
            combos=("seed", "size"),
            combos_with_validation_candidate=("validation_candidates", lambda values: int((values > 0).sum())),
            combos_with_test_candidate=("test_candidates", lambda values: int((values > 0).sum())),
            total_validation_candidates=("validation_candidates", "sum"),
            total_test_candidates=("test_candidates", "sum"),
        )
        .assign(
            validation_combo_rate=lambda frame: frame["combos_with_validation_candidate"] / frame["combos"],
            test_combo_rate=lambda frame: frame["combos_with_test_candidate"] / frame["combos"],
        )
        .sort_values(["cost_bps", "test_combo_rate", "validation_combo_rate"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if value in (np.inf, -np.inf):
        return "inf" if value == np.inf else "-inf"
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


def render_report(
    feature_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    holdout: pd.DataFrame,
    validations: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    stability_cfg = config.get("hmm_stability", {})
    top_cols = [
        "feature_set",
        "n_states",
        "seed",
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
    combo_cols = [
        "feature_set",
        "n_states",
        "seed",
        "cost_bps",
        "validation_candidates",
        "test_candidates",
        "best_validation_action",
        "best_validation_total_net_return",
        "best_validation_candidate_status",
    ]
    holdout_cols = [
        "feature_set",
        "n_states",
        "seed",
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
    stable = feature_summary[
        (feature_summary["cost_bps"] == min(feature_summary["cost_bps"]) if not feature_summary.empty else False)
        & (feature_summary["combos_with_test_candidate"] >= 2)
    ] if not feature_summary.empty else pd.DataFrame()
    conclusion = (
        "At least one feature set has test candidates across multiple K/seed combinations. Promote only those rows to deeper interpretation and frozen evaluation."
        if not stable.empty
        else "No feature set produced test candidates across multiple K/seed combinations. Treat current candidates as fragile until redesigned or retested."
    )
    return f"""# HMM Stability

## Scope

- Feature sets: {len(validations)}
- K grid: `{stability_cfg.get("n_states")}`
- Seeds: `{stability_cfg.get("seeds")}`
- Max folds: `{stability_cfg.get("max_folds")}`
- Horizons: `{stability_cfg.get("horizons")}`
- Costs bps: `{stability_cfg.get("cost_bps")}`

## Feature Set Stability Summary

{_markdown_table(feature_summary)}

## Combo Summary

{_markdown_table(combo_summary.loc[:, combo_cols] if not combo_summary.empty else combo_summary, max_rows=60)}

## Top Validation Rankings

{_markdown_table(ranking.loc[:, top_cols] if not ranking.empty else ranking, max_rows=40)}

## Validation Candidate Holdout Sanity

{_markdown_table(holdout.loc[:, holdout_cols] if not holdout.empty else holdout, max_rows=80)}

## Feature Set Validation

{_markdown_table(validations)}

## Conclusion

{conclusion}

State ids are not compared directly across K/seed because HMM state labels are permutation-dependent. This report asks whether an economically similar state/action candidate reappears across independent fits.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    stability_cfg = config.get("hmm_stability", {})
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    feature_sets = _feature_sets(config)
    validations = _validate_feature_sets(features, feature_sets)
    ready_sets = [item for item in feature_sets if validations.loc[validations["feature_set"] == item["name"], "status"].iloc[0] == "ready"]
    horizons = [int(value) for value in stability_cfg.get("horizons", [6])]
    labels_by_horizon = _prepare_horizon_labels(features, config, horizons)
    folds = _select_folds(labels, config)

    metric_frames: list[pd.DataFrame] = []
    for feature_set in ready_sets:
        for n_states in [int(value) for value in stability_cfg.get("n_states", [3, 4, 5])]:
            for seed in [int(value) for value in stability_cfg.get("seeds", config["hmm"].get("stability_seeds", [42]))]:
                metric_frames.append(run_stability_combo(features, labels_by_horizon, folds, config, feature_set, n_states, seed))

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    ranking = aggregate_stability_ranking(metrics)
    holdout = candidate_holdout_summary(metrics, ranking)
    combo_summary = summarize_combos(ranking, holdout, validations, config)
    feature_summary = summarize_feature_sets(combo_summary)

    output_dir = Path(stability_cfg.get("output_dir", "reports/hmm_stability"))
    report_path = Path(stability_cfg.get("report_file", "reports/hmm_stability.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_dir / "stability_metrics.parquet", index=False)
    ranking.to_parquet(output_dir / "stability_ranking.parquet", index=False)
    holdout.to_parquet(output_dir / "stability_holdout.parquet", index=False)
    combo_summary.to_parquet(output_dir / "stability_combo_summary.parquet", index=False)
    feature_summary.to_parquet(output_dir / "stability_feature_summary.parquet", index=False)
    report_path.write_text(render_report(feature_summary, combo_summary, ranking, holdout, validations, config), encoding="utf-8")
    return output_dir / "stability_feature_summary.parquet", report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMM feature-set stability over K and seed grids.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    summary_path, report_path = run(args.config)
    print(f"HMM stability summary written to: {summary_path}")
    print(f"HMM stability report written to: {report_path}")


if __name__ == "__main__":
    main()

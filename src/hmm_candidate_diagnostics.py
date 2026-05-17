from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.hmm_state_economics import ACTIONS, _prepare_horizon_labels, evaluate_state_action, classify_candidate
from src.walkforward import _fold_hmm_features, build_monthly_folds


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _feature_set_lookup(config: dict[str, Any]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for section in ("hmm_feature_lab", "hmm_stability", "hmm_candidate_diagnostics"):
        for item in config.get(section, {}).get("feature_sets", []):
            lookup[str(item["name"])] = list(item["columns"])
    return lookup


def candidate_id(row: pd.Series) -> str:
    return (
        f"{row['feature_set']}"
        f"__k{int(row['n_states'])}"
        f"__seed{int(row['seed'])}"
        f"__state{int(row['hmm_state'])}"
        f"__{row['action']}"
        f"__h{int(row['horizon_bars'])}"
        f"__c{float(row['cost_bps']):g}"
    )


def select_surviving_candidates(holdout: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    diag_cfg = config.get("hmm_candidate_diagnostics", {})
    feature_sets = set(diag_cfg.get("feature_sets_to_inspect", ["rich_extreme_reversion"]))
    candidate_status = str(diag_cfg.get("candidate_status", "candidate"))
    split = str(diag_cfg.get("split", "test"))
    cost_filter = diag_cfg.get("cost_bps")

    selected = holdout[
        (holdout["feature_set"].isin(feature_sets))
        & (holdout["split"] == split)
        & (holdout["candidate_status"] == candidate_status)
    ].copy()
    if cost_filter is not None:
        costs = {float(value) for value in cost_filter}
        selected = selected[selected["cost_bps"].astype(float).isin(costs)].copy()
    selected["candidate_id"] = selected.apply(candidate_id, axis=1)
    return selected.sort_values(["feature_set", "n_states", "seed", "hmm_state", "action"]).reset_index(drop=True)


def _config_for_candidate(config: dict[str, Any], candidate: pd.Series, feature_columns: list[str]) -> dict[str, Any]:
    diag_cfg = config.get("hmm_candidate_diagnostics", {})
    copied = deepcopy(config)
    copied["hmm"]["feature_columns"] = list(feature_columns)
    copied["hmm"]["n_states"] = int(candidate["n_states"])
    copied["hmm"]["random_state"] = int(candidate["seed"])
    copied["hmm"]["n_iter"] = int(diag_cfg.get("n_iter", copied["hmm"].get("n_iter", 200)))
    copied.setdefault("robustness", {})
    copied["robustness"]["horizons"] = [int(candidate["horizon_bars"])]
    copied["robustness"]["cost_bps"] = [float(candidate["cost_bps"])]
    return copied


def _target_splits(fold) -> list[tuple[str, list[str]]]:
    return [("validation", fold.validation_sessions), ("test", fold.test_sessions)]


def feature_profile_rows(
    frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    feature_columns: list[str],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        split_values = frame[feature].replace([np.inf, -np.inf], np.nan).dropna()
        state_values = state_frame[feature].replace([np.inf, -np.inf], np.nan).dropna()
        split_std = float(split_values.std(ddof=0)) if len(split_values) else np.nan
        split_mean = float(split_values.mean()) if len(split_values) else np.nan
        state_mean = float(state_values.mean()) if len(state_values) else np.nan
        state_z = (state_mean - split_mean) / split_std if split_std and not np.isnan(split_std) else np.nan
        rows.append(
            {
                **metadata,
                "feature": feature,
                "state_rows": int(len(state_frame)),
                "split_rows": int(len(frame)),
                "state_mean": state_mean,
                "split_mean": split_mean,
                "split_std": split_std,
                "state_z": float(state_z) if not pd.isna(state_z) else np.nan,
            }
        )
    return rows


def hour_distribution_rows(frame: pd.DataFrame, state_frame: pd.DataFrame, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if state_frame.empty:
        return []
    state_hours = state_frame["timestamp"].dt.hour.value_counts().sort_index()
    split_hours = frame["timestamp"].dt.hour.value_counts().sort_index()
    state_total = int(state_hours.sum())
    split_total = int(split_hours.sum())
    rows = []
    for hour in sorted(set(state_hours.index).union(set(split_hours.index))):
        state_rows = int(state_hours.get(hour, 0))
        split_rows = int(split_hours.get(hour, 0))
        state_pct = state_rows / state_total if state_total else 0.0
        split_pct = split_rows / split_total if split_total else 0.0
        rows.append(
            {
                **metadata,
                "hour": int(hour),
                "state_rows": state_rows,
                "split_rows": split_rows,
                "state_pct": float(state_pct),
                "split_pct": float(split_pct),
                "hour_lift": float(state_pct / split_pct) if split_pct else np.nan,
            }
        )
    return rows


def _action_metric_row(frame: pd.DataFrame, action: str, cost_bps: float, metadata: dict[str, Any]) -> dict[str, Any]:
    return {**metadata, "action": action, **evaluate_state_action(frame, action, cost_bps)}


def _clock_control_frames(frame: pd.DataFrame, state_frame: pd.DataFrame, target_state: int) -> dict[str, pd.DataFrame]:
    if state_frame.empty:
        return {
            "candidate_state": state_frame,
            "same_hours_all_states": state_frame,
            "same_hours_ex_state": state_frame,
            "full_split_all_states": frame,
        }
    state_hours = set(state_frame["timestamp"].dt.hour.unique().tolist())
    same_hours = frame[frame["timestamp"].dt.hour.isin(state_hours)].copy()
    return {
        "candidate_state": state_frame,
        "same_hours_all_states": same_hours,
        "same_hours_ex_state": same_hours[same_hours["hmm_state"] != target_state].copy(),
        "full_split_all_states": frame,
    }


def build_candidate_diagnostics(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    diag_cfg = config.get("hmm_candidate_diagnostics", {})
    holdout_path = Path(diag_cfg.get("candidate_source", "reports/hmm_stability/stability_holdout.parquet"))
    if not holdout_path.exists():
        raise FileNotFoundError(f"Candidate source not found: {holdout_path}")

    holdout = pd.read_parquet(holdout_path)
    candidates = select_surviving_candidates(holdout, config)
    feature_lookup = _feature_set_lookup(config)
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    folds = build_monthly_folds(labels, config)
    max_folds = diag_cfg.get("max_folds")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    horizons = sorted(candidates["horizon_bars"].astype(int).unique().tolist()) if not candidates.empty else []
    labels_by_horizon = _prepare_horizon_labels(features, config, horizons) if horizons else {}

    feature_profile: list[dict[str, Any]] = []
    hour_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        feature_columns = feature_lookup.get(str(candidate["feature_set"]))
        if not feature_columns:
            raise ValueError(f"Feature set columns not found for {candidate['feature_set']}")
        candidate_config = _config_for_candidate(config, candidate, feature_columns)
        target_state = int(candidate["hmm_state"])
        target_action = str(candidate["action"])
        horizon = int(candidate["horizon_bars"])
        cost_bps = float(candidate["cost_bps"])
        cid = str(candidate["candidate_id"])

        for fold in folds:
            filtered_hmm, _, _ = _fold_hmm_features(features, fold, candidate_config)
            fold_sessions = fold.train_sessions + fold.validation_sessions + fold.test_sessions
            fold_labels = labels_by_horizon[horizon][labels_by_horizon[horizon]["session"].isin(fold_sessions)].copy()
            hmm_cols = ["timestamp", "session", "bar_index", "hmm_state", "hmm_entropy", "hmm_max_prob"]
            merged = fold_labels.merge(filtered_hmm[hmm_cols], on=["timestamp", "session", "bar_index"], how="inner", validate="one_to_one")

            for split, sessions in _target_splits(fold):
                split_frame = merged[merged["session"].isin(sessions)].copy()
                state_frame = split_frame[split_frame["hmm_state"] == target_state].copy()
                metadata = {
                    "candidate_id": cid,
                    "feature_set": candidate["feature_set"],
                    "n_states": int(candidate["n_states"]),
                    "seed": int(candidate["seed"]),
                    "horizon_bars": horizon,
                    "cost_bps": cost_bps,
                    "hmm_state": target_state,
                    "target_action": target_action,
                    "fold": int(fold.fold),
                    "split": split,
                }
                feature_profile.extend(feature_profile_rows(split_frame, state_frame, feature_columns, metadata))
                hour_rows.extend(hour_distribution_rows(split_frame, state_frame, metadata))

                for action in ACTIONS:
                    action_rows.append(_action_metric_row(state_frame, action, cost_bps, metadata))

                for bucket, bucket_frame in _clock_control_frames(split_frame, state_frame, target_state).items():
                    clock_rows.append(_action_metric_row(bucket_frame, target_action, cost_bps, {**metadata, "bucket": bucket}))

    return {
        "candidates": candidates,
        "feature_profile_by_fold": pd.DataFrame(feature_profile),
        "hour_distribution_by_fold": pd.DataFrame(hour_rows),
        "state_action_by_fold": pd.DataFrame(action_rows),
        "clock_control_by_fold": pd.DataFrame(clock_rows),
    }


def summarize_feature_profile(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    grouped = (
        profile.groupby(["candidate_id", "feature_set", "n_states", "seed", "hmm_state", "target_action", "split", "feature"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            avg_state_mean=("state_mean", "mean"),
            avg_split_mean=("split_mean", "mean"),
            avg_state_z=("state_z", "mean"),
            median_state_z=("state_z", "median"),
            std_state_z=("state_z", "std"),
            positive_z_folds=("state_z", lambda values: int((values > 0).sum())),
            negative_z_folds=("state_z", lambda values: int((values < 0).sum())),
        )
        .reset_index(drop=True)
    )
    grouped["abs_avg_state_z"] = grouped["avg_state_z"].abs()
    return grouped.sort_values(["candidate_id", "split", "abs_avg_state_z"], ascending=[True, True, False]).reset_index(drop=True)


def summarize_fold_metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_trades=("trades", "sum"),
            total_net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            median_profit_factor=("profit_factor", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            negative_folds=("net_return", lambda values: int((values < 0).sum())),
        )
        .reset_index(drop=True)
    )
    grouped["candidate_status"] = grouped.apply(classify_candidate, axis=1)
    return grouped


def summarize_hour_distribution(hour_distribution: pd.DataFrame) -> pd.DataFrame:
    if hour_distribution.empty:
        return pd.DataFrame()
    grouped = (
        hour_distribution.groupby(["candidate_id", "feature_set", "n_states", "seed", "hmm_state", "target_action", "split", "hour"], as_index=False)
        .agg(state_rows=("state_rows", "sum"), split_rows=("split_rows", "sum"))
    )
    totals = grouped.groupby(["candidate_id", "split"], as_index=False).agg(total_state_rows=("state_rows", "sum"), total_split_rows=("split_rows", "sum"))
    grouped = grouped.merge(totals, on=["candidate_id", "split"], how="left")
    grouped["state_pct"] = grouped["state_rows"] / grouped["total_state_rows"]
    grouped["split_pct"] = grouped["split_rows"] / grouped["total_split_rows"]
    grouped["hour_lift"] = grouped["state_pct"] / grouped["split_pct"].replace(0, np.nan)
    return grouped.drop(columns=["total_state_rows", "total_split_rows"]).sort_values(["candidate_id", "split", "hour"]).reset_index(drop=True)


def summarize_time_concentration(hour_summary: pd.DataFrame) -> pd.DataFrame:
    if hour_summary.empty:
        return pd.DataFrame()
    top = (
        hour_summary.sort_values(["candidate_id", "split", "state_pct"], ascending=[True, True, False])
        .groupby(["candidate_id", "split"], as_index=False)
        .head(1)
        .loc[:, ["candidate_id", "split", "hour", "state_pct", "hour_lift"]]
        .rename(columns={"hour": "top_hour", "state_pct": "top_hour_state_pct", "hour_lift": "top_hour_lift"})
    )
    entropy = hour_summary.copy()
    entropy["entropy_component"] = -entropy["state_pct"] * np.log(np.clip(entropy["state_pct"], 1e-12, 1.0))
    entropy_summary = entropy.groupby(["candidate_id", "split"], as_index=False).agg(hour_entropy=("entropy_component", "sum"))
    entropy_summary["normalized_hour_entropy"] = entropy_summary["hour_entropy"] / np.log(hour_summary["hour"].nunique())
    return top.merge(entropy_summary, on=["candidate_id", "split"], how="left")


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


def render_report(outputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> str:
    candidates = outputs["candidates"]
    feature_summary = outputs["feature_profile_summary"]
    action_summary = outputs["state_action_summary"]
    clock_summary = outputs["clock_control_summary"]
    hour_summary = outputs["hour_summary"]
    time_concentration = outputs["time_concentration"]
    target_fold_metrics = outputs["target_fold_metrics"]

    candidate_cols = [
        "candidate_id",
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
    profile_cols = [
        "candidate_id",
        "split",
        "feature",
        "avg_state_z",
        "median_state_z",
        "std_state_z",
        "positive_z_folds",
        "negative_z_folds",
    ]
    action_cols = [
        "candidate_id",
        "split",
        "target_action",
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
    clock_cols = [
        "candidate_id",
        "split",
        "bucket",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "candidate_status",
    ]
    hour_cols = ["candidate_id", "split", "top_hour", "top_hour_state_pct", "top_hour_lift", "normalized_hour_entropy"]
    fold_cols = [
        "candidate_id",
        "split",
        "fold",
        "trades",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "hit_ratio",
    ]

    test_clock = clock_summary[(clock_summary["split"] == "test") & (clock_summary["bucket"].isin(["candidate_state", "same_hours_ex_state"]))]
    state_candidates = test_clock[(test_clock["bucket"] == "candidate_state") & (test_clock["candidate_status"] == "candidate")]
    clock_candidates = test_clock[(test_clock["bucket"] == "same_hours_ex_state") & (test_clock["candidate_status"] == "candidate")]
    if state_candidates.empty:
        conclusion = "The surviving stability rows did not remain candidates in the reconstructed diagnostics."
    elif clock_candidates.empty:
        conclusion = "The surviving rows remain candidates, and the same-hour ex-state control does not. This supports a regime-conditioned effect, but it is still cost-fragile."
    else:
        conclusion = "The same-hour ex-state control also shows candidate behavior. The edge may be partly time-of-day rather than regime-specific."

    return f"""# HMM Candidate Diagnostics

## Scope

- Candidate source: `{config.get("hmm_candidate_diagnostics", {}).get("candidate_source", "reports/hmm_stability/stability_holdout.parquet")}`
- Candidates inspected: {len(candidates)}
- Feature sets: `{config.get("hmm_candidate_diagnostics", {}).get("feature_sets_to_inspect", ["rich_extreme_reversion"])}`
- Splits reconstructed: validation and test

## Candidate Rows

{_markdown_table(candidates.loc[:, candidate_cols] if not candidates.empty else candidates)}

## Feature Profile Top Z-Scores

{_markdown_table(feature_summary.loc[:, profile_cols] if not feature_summary.empty else feature_summary, max_rows=40)}

## Time Concentration

{_markdown_table(time_concentration.loc[:, hour_cols] if not time_concentration.empty else time_concentration)}

## Clock Control

{_markdown_table(clock_summary.loc[:, clock_cols] if not clock_summary.empty else clock_summary, max_rows=60)}

## Target Fold Performance

{_markdown_table(target_fold_metrics.loc[:, fold_cols] if not target_fold_metrics.empty else target_fold_metrics, max_rows=60)}

## Action Comparison Inside Candidate State

{_markdown_table(action_summary.loc[:, action_cols] if not action_summary.empty else action_summary, max_rows=72)}

## Hour Distribution

{_markdown_table(hour_summary.sort_values(["candidate_id", "split", "state_pct"], ascending=[True, True, False]).head(40))}

## Outputs

- `reports/hmm_candidate_diagnostics/candidates.parquet`
- `reports/hmm_candidate_diagnostics/feature_profile_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/feature_profile_summary.parquet`
- `reports/hmm_candidate_diagnostics/hour_distribution_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/hour_summary.parquet`
- `reports/hmm_candidate_diagnostics/time_concentration.parquet`
- `reports/hmm_candidate_diagnostics/state_action_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/state_action_summary.parquet`
- `reports/hmm_candidate_diagnostics/clock_control_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/clock_control_summary.parquet`
- `reports/hmm_candidate_diagnostics/target_fold_metrics.parquet`

## Conclusion

{conclusion}

State ids are inspected within each fold/model fit. The report does not assume that nominal state ids are comparable across different K/seed combinations.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    raw_outputs = build_candidate_diagnostics(config)
    feature_summary = summarize_feature_profile(raw_outputs["feature_profile_by_fold"])
    hour_summary = summarize_hour_distribution(raw_outputs["hour_distribution_by_fold"])
    time_concentration = summarize_time_concentration(hour_summary)
    action_summary = summarize_fold_metrics(
        raw_outputs["state_action_by_fold"],
        ["candidate_id", "feature_set", "n_states", "seed", "hmm_state", "target_action", "split", "action"],
    )
    clock_summary = summarize_fold_metrics(
        raw_outputs["clock_control_by_fold"],
        ["candidate_id", "feature_set", "n_states", "seed", "hmm_state", "target_action", "split", "bucket"],
    )
    target_fold_metrics = raw_outputs["clock_control_by_fold"][
        raw_outputs["clock_control_by_fold"]["bucket"].eq("candidate_state")
    ].copy()
    outputs = {
        **raw_outputs,
        "feature_profile_summary": feature_summary,
        "hour_summary": hour_summary,
        "time_concentration": time_concentration,
        "state_action_summary": action_summary,
        "clock_control_summary": clock_summary,
        "target_fold_metrics": target_fold_metrics,
    }

    diag_cfg = config.get("hmm_candidate_diagnostics", {})
    output_dir = Path(diag_cfg.get("output_dir", "reports/hmm_candidate_diagnostics"))
    report_path = Path(diag_cfg.get("report_file", "reports/hmm_candidate_diagnostics.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    report_path.write_text(render_report(outputs, config), encoding="utf-8")
    return output_dir / "candidates.parquet", report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect surviving HMM stability candidates.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    candidates_path, report_path = run(args.config)
    print(f"HMM candidate diagnostics written to: {candidates_path}")
    print(f"HMM candidate diagnostics report written to: {report_path}")


if __name__ == "__main__":
    main()

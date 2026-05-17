from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.hmm_lab import FULL_CORE_TOKEN, _feature_set_version, _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import (
    _format_value,
    _known_symbols,
    _markdown_table,
    assign_economic_label,
    feature_tickers,
)


PROFILE_KEYS = ["feature_set", "n_states", "seed", "fold", "split", "hmm_state"]
STATE_KEYS = ["feature_set", "n_states", "seed", "fold", "hmm_state"]


def _stability_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hmm_state_stability", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "hmm_stability_cross_asset.md"


def _feature_sets(config: dict[str, Any], feature_config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in _lab_cfg(config).get("feature_sets", []):
        columns = item.get("columns", [])
        if columns == FULL_CORE_TOKEN:
            columns = feature_config.get("hmm_feature_columns", [])
        output.append(
            {
                "name": str(item["name"]),
                "description": str(item.get("description", "")),
                "columns": [str(column) for column in columns],
            }
        )
    return output


def _cosine(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).fillna(0.0)
    left_values = joined["left"].to_numpy(dtype=float)
    right_values = joined["right"].to_numpy(dtype=float)
    denom = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denom == 0.0:
        return np.nan
    return float(np.dot(left_values, right_values) / denom)


def build_feature_profiles(features: pd.DataFrame, posteriors: pd.DataFrame, feature_sets: list[dict[str, Any]]) -> pd.DataFrame:
    feature_source = features.reset_index(names="source_index")
    rows: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        columns = [column for column in feature_set["columns"] if column in feature_source.columns]
        if not columns:
            continue
        selected = posteriors[posteriors["feature_set"] == feature_set["name"]].copy()
        if selected.empty:
            continue
        merged = selected.merge(feature_source[["source_index", *columns]], on="source_index", how="left", validate="many_to_one")
        for split_key, split_frame in merged.groupby(["feature_set", "n_states", "seed", "fold", "split"], sort=False):
            split_values = split_frame[columns].replace([np.inf, -np.inf], np.nan)
            split_mean = split_values.mean()
            split_std = split_values.std(ddof=0)
            for state, state_frame in split_frame.groupby("hmm_state", sort=True):
                state_values = state_frame[columns].replace([np.inf, -np.inf], np.nan)
                state_mean = state_values.mean()
                state_median = state_values.median()
                state_p10 = state_values.quantile(0.10)
                state_p90 = state_values.quantile(0.90)
                for feature in columns:
                    std = float(split_std[feature]) if feature in split_std else np.nan
                    mean = float(split_mean[feature]) if feature in split_mean else np.nan
                    current_mean = float(state_mean[feature]) if feature in state_mean else np.nan
                    state_z = (current_mean - mean) / std if std and not np.isnan(std) else np.nan
                    rows.append(
                        {
                            "feature_set": str(split_key[0]),
                            "n_states": int(split_key[1]),
                            "seed": int(split_key[2]),
                            "fold": int(split_key[3]),
                            "split": str(split_key[4]),
                            "hmm_state": int(state),
                            "local_state_id": _local_state_id(str(split_key[0]), int(split_key[1]), int(split_key[2]), int(split_key[3]), int(state)),
                            "feature": feature,
                            "state_rows": int(len(state_frame)),
                            "split_rows": int(len(split_frame)),
                            "state_mean": current_mean,
                            "state_median": float(state_median[feature]),
                            "state_p10": float(state_p10[feature]),
                            "state_p90": float(state_p90[feature]),
                            "split_mean": mean,
                            "split_std": std,
                            "state_z": float(state_z) if not pd.isna(state_z) else np.nan,
                            "abs_state_z": float(abs(state_z)) if not pd.isna(state_z) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def _local_state_id(feature_set: str, n_states: int, seed: int, fold: int, state: int) -> str:
    return f"{feature_set}__k{int(n_states)}__seed{int(seed)}__fold{int(fold)}__state{int(state)}"


def build_state_summary(posteriors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = posteriors.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for key, split_frame in frame.groupby(["feature_set", "n_states", "seed", "fold", "split"], sort=False):
        total = len(split_frame)
        for state, state_frame in split_frame.groupby("hmm_state", sort=True):
            hours = state_frame["timestamp"].dt.hour.value_counts(normalize=True)
            sessions = state_frame["session"].value_counts(normalize=True)
            rows.append(
                {
                    "feature_set": str(key[0]),
                    "n_states": int(key[1]),
                    "seed": int(key[2]),
                    "fold": int(key[3]),
                    "split": str(key[4]),
                    "hmm_state": int(state),
                    "local_state_id": _local_state_id(str(key[0]), int(key[1]), int(key[2]), int(key[3]), int(state)),
                    "state_rows": int(len(state_frame)),
                    "split_rows": int(total),
                    "state_frequency": float(len(state_frame) / total) if total else 0.0,
                    "mean_hmm_entropy": float(state_frame["hmm_entropy"].mean()),
                    "mean_hmm_max_prob": float(state_frame["hmm_max_prob"].mean()),
                    "top_hour": int(hours.index[0]) if not hours.empty else None,
                    "top_hour_pct": float(hours.iloc[0]) if not hours.empty else np.nan,
                    "top_session": str(sessions.index[0]) if not sessions.empty else "",
                    "top_session_pct": float(sessions.iloc[0]) if not sessions.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_state_names(profiles: pd.DataFrame, state_summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split = str(_stability_cfg(config).get("split", "validation"))
    cfg = config.get("hmm_state_interpretability", {})
    validation_profiles = profiles[profiles["split"] == split].copy()
    train_profiles = profiles[profiles["split"] == "train"].copy()
    train_labels: dict[tuple[Any, ...], str] = {}
    for key, group in train_profiles.groupby(STATE_KEYS, sort=False):
        train_labels[key] = str(assign_economic_label(group, cfg)["proposed_label"])

    rows: list[dict[str, Any]] = []
    for key, group in validation_profiles.groupby(STATE_KEYS, sort=False):
        assigned = assign_economic_label(group, cfg)
        summary_row = state_summary[
            (state_summary["feature_set"] == key[0])
            & (state_summary["n_states"].astype(int) == int(key[1]))
            & (state_summary["seed"].astype(int) == int(key[2]))
            & (state_summary["fold"].astype(int) == int(key[3]))
            & (state_summary["hmm_state"].astype(int) == int(key[4]))
            & (state_summary["split"] == split)
        ]
        summary_data = summary_row.iloc[0].to_dict() if not summary_row.empty else {}
        reference_label = train_labels.get(key, "")
        rows.append(
            {
                "feature_set": str(key[0]),
                "n_states": int(key[1]),
                "seed": int(key[2]),
                "fold": int(key[3]),
                "hmm_state": int(key[4]),
                "local_state_id": _local_state_id(str(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4])),
                "split": split,
                "proposed_label": assigned["proposed_label"],
                "reference_label": reference_label,
                "train_validation_label_match": bool(reference_label == assigned["proposed_label"]) if reference_label else False,
                "best_score": assigned["best_score"],
                "score_margin": assigned["score_margin"],
                "profile_strength": assigned["profile_strength"],
                "top_positive_features": assigned["top_positive_features"],
                "top_negative_features": assigned["top_negative_features"],
                "state_frequency": float(summary_data.get("state_frequency", np.nan)),
                "top_hour": summary_data.get("top_hour"),
                "top_hour_pct": float(summary_data.get("top_hour_pct", np.nan)),
                "top_session": summary_data.get("top_session", ""),
                "top_session_pct": float(summary_data.get("top_session_pct", np.nan)),
                "mean_hmm_entropy": float(summary_data.get("mean_hmm_entropy", np.nan)),
                "mean_hmm_max_prob": float(summary_data.get("mean_hmm_max_prob", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _feature_ticker_map(features: list[str], feature_config: dict[str, Any], target_symbol: str) -> dict[str, tuple[str, ...]]:
    known = _known_symbols(feature_config, target_symbol)
    return {feature: feature_tickers(feature, known, target_symbol) for feature in features}


def build_ticker_dependency(
    profiles: pd.DataFrame,
    feature_config: dict[str, Any],
    target_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = "validation"
    validation_profiles = profiles[profiles["split"] == split].copy()
    feature_map = _feature_ticker_map(validation_profiles["feature"].unique().tolist(), feature_config, target_symbol)
    dependency_rows: list[dict[str, Any]] = []
    leave_one_rows: list[dict[str, Any]] = []
    for key, group in validation_profiles.groupby(STATE_KEYS, sort=False):
        group = group.dropna(subset=["abs_state_z"]).copy()
        total_abs = float(group["abs_state_z"].sum())
        ticker_scores: dict[str, float] = {}
        for _, row in group.iterrows():
            tickers = feature_map.get(str(row["feature"]), ())
            if not tickers:
                continue
            contribution = float(row["abs_state_z"]) / len(tickers)
            for ticker in tickers:
                ticker_scores[ticker] = ticker_scores.get(ticker, 0.0) + contribution
        top_ticker = max(ticker_scores, key=ticker_scores.get) if ticker_scores else ""
        top_share = ticker_scores.get(top_ticker, 0.0) / total_abs if total_abs else 0.0
        local_state_id = _local_state_id(str(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4]))
        dependency_rows.append(
            {
                "feature_set": str(key[0]),
                "n_states": int(key[1]),
                "seed": int(key[2]),
                "fold": int(key[3]),
                "hmm_state": int(key[4]),
                "local_state_id": local_state_id,
                "top_ticker": top_ticker,
                "top_ticker_abs_z_share": float(top_share),
                "total_feature_abs_z": total_abs,
                "ticker_count": int(len(ticker_scores)),
            }
        )
        vector = group.set_index("feature")["state_z"].fillna(0.0)
        norm = float(np.linalg.norm(vector.to_numpy()))
        for ticker in sorted(ticker_scores, key=ticker_scores.get, reverse=True):
            removed_features = [feature for feature, tickers in feature_map.items() if ticker in tickers and feature in vector.index]
            reduced = vector.copy()
            reduced.loc[removed_features] = 0.0
            reduced_norm = float(np.linalg.norm(reduced.to_numpy()))
            cosine = float(np.dot(vector.to_numpy(), reduced.to_numpy()) / (norm * reduced_norm)) if norm and reduced_norm else np.nan
            removed_share = float(group[group["feature"].isin(removed_features)]["abs_state_z"].sum() / total_abs) if total_abs else 0.0
            leave_one_rows.append(
                {
                    "feature_set": str(key[0]),
                    "n_states": int(key[1]),
                    "seed": int(key[2]),
                    "fold": int(key[3]),
                    "hmm_state": int(key[4]),
                    "local_state_id": local_state_id,
                    "ticker_removed": ticker,
                    "removed_feature_count": int(len(removed_features)),
                    "removed_abs_z_share": removed_share,
                    "profile_cosine_after_removal": cosine,
                }
            )
    return pd.DataFrame(dependency_rows), pd.DataFrame(leave_one_rows)


def align_seed_states(profiles: pd.DataFrame, state_names: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split = str(_stability_cfg(config).get("split", "validation"))
    reference_seed_rule = str(_stability_cfg(config).get("reference_seed", "min"))
    validation = profiles[profiles["split"] == split].copy()
    labels = state_names.set_index(["feature_set", "n_states", "seed", "fold", "hmm_state"])["proposed_label"].to_dict()
    rows: list[dict[str, Any]] = []

    for key, group in validation.groupby(["feature_set", "n_states", "fold"], sort=False):
        seeds = sorted(int(seed) for seed in group["seed"].unique())
        if not seeds:
            continue
        reference_seed = min(seeds) if reference_seed_rule == "min" else int(reference_seed_rule)
        if reference_seed not in seeds:
            reference_seed = seeds[0]
        reference = group[group["seed"].astype(int) == reference_seed]
        ref_vectors = {
            int(state): state_group.set_index("feature")["state_z"].fillna(0.0)
            for state, state_group in reference.groupby("hmm_state", sort=True)
        }
        ref_states = sorted(ref_vectors)
        for seed in seeds:
            current = group[group["seed"].astype(int) == seed]
            cur_vectors = {
                int(state): state_group.set_index("feature")["state_z"].fillna(0.0)
                for state, state_group in current.groupby("hmm_state", sort=True)
            }
            cur_states = sorted(cur_vectors)
            if not ref_states or not cur_states:
                continue
            similarity = np.array([[ _cosine(cur_vectors[cur_state], ref_vectors[ref_state]) for ref_state in ref_states] for cur_state in cur_states])
            cost = -np.nan_to_num(similarity, nan=-1.0)
            row_idx, col_idx = linear_sum_assignment(cost)
            for row_pos, col_pos in zip(row_idx, col_idx):
                cur_state = cur_states[row_pos]
                ref_state = ref_states[col_pos]
                current_label = labels.get((str(key[0]), int(key[1]), int(seed), int(key[2]), int(cur_state)), "")
                anchor_label = labels.get((str(key[0]), int(key[1]), int(reference_seed), int(key[2]), int(ref_state)), "")
                rows.append(
                    {
                        "feature_set": str(key[0]),
                        "n_states": int(key[1]),
                        "fold": int(key[2]),
                        "reference_seed": int(reference_seed),
                        "seed": int(seed),
                        "hmm_state": int(cur_state),
                        "anchor_hmm_state": int(ref_state),
                        "local_state_id": _local_state_id(str(key[0]), int(key[1]), int(seed), int(key[2]), int(cur_state)),
                        "anchor_state_id": _local_state_id(str(key[0]), int(key[1]), int(reference_seed), int(key[2]), int(ref_state)),
                        "profile_cosine": float(similarity[row_pos, col_pos]),
                        "proposed_label": current_label,
                        "anchor_label": anchor_label,
                        "label_match": bool(current_label == anchor_label),
                    }
                )
    return pd.DataFrame(rows)


def build_period_regime_occupancy(
    features: pd.DataFrame,
    posteriors: pd.DataFrame,
) -> pd.DataFrame:
    source = features.reset_index(names="source_index")
    cols = ["source_index", "target_ret_12", "target_range_ratio_6_24"]
    available = [column for column in cols if column in source.columns]
    frame = posteriors.merge(source[available], on="source_index", how="left", validate="many_to_one")
    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(None)
    frame["year"] = timestamps.dt.year.astype(str)

    vol = frame["target_range_ratio_6_24"].replace([np.inf, -np.inf], np.nan)
    low_vol, high_vol = vol.quantile([0.33, 0.67])
    frame["vol_regime"] = np.select([vol <= low_vol, vol >= high_vol], ["low_vol", "high_vol"], default="mid_vol")

    trend = frame["target_ret_12"].replace([np.inf, -np.inf], np.nan)
    low_trend, high_trend = trend.quantile([0.33, 0.67])
    frame["trend_regime"] = np.select([trend <= low_trend, trend >= high_trend], ["down_past_12", "up_past_12"], default="flat_past_12")

    rows = []
    for dimension in ("year", "vol_regime", "trend_regime"):
        counts = (
            frame.groupby(["feature_set", "n_states", "seed", "fold", "split", "hmm_state", dimension], as_index=False)
            .size()
            .rename(columns={"size": "state_rows", dimension: "period_bucket"})
        )
        totals = (
            frame.groupby(["feature_set", "n_states", "seed", "fold", "split", dimension], as_index=False)
            .size()
            .rename(columns={"size": "bucket_rows", dimension: "period_bucket"})
        )
        out = counts.merge(totals, on=["feature_set", "n_states", "seed", "fold", "split", "period_bucket"], how="left")
        out["period_dimension"] = dimension
        out["state_frequency"] = out["state_rows"] / out["bucket_rows"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def build_stability_grid(
    state_names: pd.DataFrame,
    alignment: pd.DataFrame,
    ticker_dependency: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    period_occupancy: pd.DataFrame,
    config: dict[str, Any],
    target_symbol: str,
) -> pd.DataFrame:
    cfg = _stability_cfg(config)
    split = str(cfg.get("split", "validation"))
    states = state_names.copy()
    states = states.merge(
        ticker_dependency[["local_state_id", "top_ticker", "top_ticker_abs_z_share", "ticker_count"]],
        on="local_state_id",
        how="left",
    )
    worst_loo = (
        leave_one_out.sort_values("removed_abs_z_share", ascending=False)
        .groupby("local_state_id", as_index=False)
        .head(1)
        .loc[:, ["local_state_id", "ticker_removed", "removed_abs_z_share", "profile_cosine_after_removal"]]
        .rename(
            columns={
                "ticker_removed": "worst_leave_one_ticker",
                "removed_abs_z_share": "worst_leave_one_removed_abs_z_share",
                "profile_cosine_after_removal": "worst_leave_one_profile_cosine",
            }
        )
    )
    states = states.merge(worst_loo, on="local_state_id", how="left")
    non_reference_alignment = alignment[alignment["seed"].astype(int) != alignment["reference_seed"].astype(int)].copy()
    alignment_summary = (
        non_reference_alignment.groupby(["feature_set", "n_states", "fold", "proposed_label"], as_index=False)
        .agg(mean_seed_alignment_cosine=("profile_cosine", "mean"), min_seed_alignment_cosine=("profile_cosine", "min"), seed_label_match_rate=("label_match", "mean"))
        if not non_reference_alignment.empty
        else pd.DataFrame(columns=["feature_set", "n_states", "fold", "proposed_label", "mean_seed_alignment_cosine", "min_seed_alignment_cosine", "seed_label_match_rate"])
    )

    grouped = (
        states.groupby(["feature_set", "n_states", "proposed_label"], as_index=False)
        .agg(
            local_states=("local_state_id", "nunique"),
            seeds_present=("seed", "nunique"),
            folds_present=("fold", "nunique"),
            avg_state_frequency=("state_frequency", "mean"),
            min_state_frequency=("state_frequency", "min"),
            avg_profile_strength=("profile_strength", "mean"),
            train_validation_match_rate=("train_validation_label_match", "mean"),
            avg_top_hour_pct=("top_hour_pct", "mean"),
            max_top_hour_pct=("top_hour_pct", "max"),
            max_top_session_pct=("top_session_pct", "max"),
            top_ticker_share_max=("top_ticker_abs_z_share", "max"),
            non_target_top_ticker_share_max=("top_ticker_abs_z_share", lambda values: float(values[states.loc[values.index, "top_ticker"] != target_symbol.upper()].max()) if (states.loc[values.index, "top_ticker"] != target_symbol.upper()).any() else 0.0),
            worst_leave_one_removed_abs_z_share=("worst_leave_one_removed_abs_z_share", "max"),
            worst_leave_one_profile_cosine=("worst_leave_one_profile_cosine", "min"),
        )
    )
    align_by_label = (
        alignment_summary.groupby(["feature_set", "n_states", "proposed_label"], as_index=False)
        .agg(
            mean_seed_alignment_cosine=("mean_seed_alignment_cosine", "mean"),
            min_seed_alignment_cosine=("min_seed_alignment_cosine", "min"),
            seed_label_match_rate=("seed_label_match_rate", "mean"),
        )
        if not alignment_summary.empty
        else pd.DataFrame(columns=["feature_set", "n_states", "proposed_label"])
    )
    grouped = grouped.merge(align_by_label, on=["feature_set", "n_states", "proposed_label"], how="left")

    period_split = period_occupancy[period_occupancy["split"] == split].copy()
    period_split["local_state_id"] = period_split.apply(
        lambda row: _local_state_id(row["feature_set"], int(row["n_states"]), int(row["seed"]), int(row["fold"]), int(row["hmm_state"])),
        axis=1,
    )
    period_labels = period_split.merge(states[["local_state_id", "proposed_label"]], on="local_state_id", how="left")
    period_counts = (
        period_labels.groupby(["feature_set", "n_states", "proposed_label", "period_dimension"], as_index=False)
        .agg(period_buckets_present=("period_bucket", "nunique"), avg_period_frequency=("state_frequency", "mean"))
    )
    for dimension in ("year", "vol_regime", "trend_regime"):
        dim = period_counts[period_counts["period_dimension"] == dimension].drop(columns=["period_dimension"])
        dim = dim.rename(
            columns={
                "period_buckets_present": f"{dimension}_buckets_present",
                "avg_period_frequency": f"{dimension}_avg_frequency",
            }
        )
        grouped = grouped.merge(dim, on=["feature_set", "n_states", "proposed_label"], how="left")

    grouped["status"] = grouped.apply(lambda row: classify_stability_row(row, cfg), axis=1)
    grouped["k_present_for_label"] = grouped.groupby(["feature_set", "proposed_label"])["n_states"].transform("nunique")
    return grouped.sort_values(["status", "feature_set", "proposed_label", "n_states"]).reset_index(drop=True)


def classify_stability_row(row: pd.Series, cfg: dict[str, Any]) -> str:
    if str(row["proposed_label"]) == "uninterpretable_noise":
        return "rejected_uninterpretable"
    if int(row["seeds_present"]) < int(cfg.get("min_seeds_present", 2)):
        return "rejected_single_seed"
    if int(row["folds_present"]) < int(cfg.get("min_folds_present", 2)):
        return "rejected_single_fold"
    if float(row["avg_state_frequency"]) < float(cfg.get("min_avg_state_frequency", 0.02)):
        return "rejected_low_occupancy"
    if float(row.get("max_top_hour_pct", 0.0)) > float(cfg.get("max_top_hour_pct", 0.35)):
        return "rejected_time_concentration"
    if float(row.get("non_target_top_ticker_share_max", 0.0)) > float(cfg.get("max_non_target_ticker_share", 0.6)):
        return "rejected_non_target_ticker_dependency"
    min_cosine = row.get("min_seed_alignment_cosine", np.nan)
    if not pd.isna(min_cosine) and float(min_cosine) < float(cfg.get("min_seed_profile_cosine", 0.70)):
        return "rejected_seed_instability"
    return "stable_profile_candidate"


def summarize_label_stability(stability_grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if stability_grid.empty:
        return pd.DataFrame()
    cfg = _stability_cfg(config)
    grouped = (
        stability_grid.groupby(["feature_set", "proposed_label"], as_index=False)
        .agg(
            k_present=("n_states", "nunique"),
            stable_k=("status", lambda values: int((values == "stable_profile_candidate").sum())),
            local_states=("local_states", "sum"),
            avg_state_frequency=("avg_state_frequency", "mean"),
            min_seed_alignment_cosine=("min_seed_alignment_cosine", "min"),
            max_top_hour_pct=("max_top_hour_pct", "max"),
            worst_leave_one_removed_abs_z_share=("worst_leave_one_removed_abs_z_share", "max"),
            statuses=("status", lambda values: ",".join(sorted(set(str(value) for value in values)))),
        )
        .reset_index(drop=True)
    )
    grouped["k_stability_status"] = np.where(
        grouped["k_present"] >= int(cfg.get("min_k_present", 2)),
        "multi_k",
        "single_k",
    )
    return grouped.sort_values(["stable_k", "k_present", "avg_state_frequency"], ascending=[False, False, False]).reset_index(drop=True)


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    stability_grid: pd.DataFrame,
    alignment: pd.DataFrame,
    label_summary: pd.DataFrame,
    period_occupancy: pd.DataFrame,
    ticker_dependency: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _stability_cfg(config)
    status_counts = stability_grid["status"].value_counts().rename_axis("status").reset_index(name="rows") if not stability_grid.empty else pd.DataFrame()
    top_cols = [
        "feature_set",
        "n_states",
        "proposed_label",
        "status",
        "local_states",
        "seeds_present",
        "folds_present",
        "k_present_for_label",
        "avg_state_frequency",
        "min_seed_alignment_cosine",
        "seed_label_match_rate",
        "max_top_hour_pct",
        "non_target_top_ticker_share_max",
        "worst_leave_one_removed_abs_z_share",
        "year_buckets_present",
        "vol_regime_buckets_present",
        "trend_regime_buckets_present",
    ]
    stable = stability_grid[stability_grid["status"] == "stable_profile_candidate"] if not stability_grid.empty else pd.DataFrame()
    alignment_summary = (
        alignment.groupby(["feature_set", "n_states"], as_index=False)
        .agg(mean_profile_cosine=("profile_cosine", "mean"), min_profile_cosine=("profile_cosine", "min"), label_match_rate=("label_match", "mean"))
        .sort_values(["mean_profile_cosine"], ascending=False)
        if not alignment.empty
        else pd.DataFrame()
    )
    period_summary = (
        period_occupancy[period_occupancy["split"] == str(cfg.get("split", "validation"))]
        .groupby(["period_dimension"], as_index=False)
        .agg(buckets=("period_bucket", "nunique"), rows=("state_rows", "sum"))
        if not period_occupancy.empty
        else pd.DataFrame()
    )
    dependency_cols = ["local_state_id", "top_ticker", "top_ticker_abs_z_share", "ticker_count", "total_feature_abs_z"]
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "There are stable profile candidates across seeds and folds. Treat them as candidates for economic diagnostics, not as accepted trading states."
        if not stable.empty
        else "No state profile passed the configured stability gates across seeds and folds. Redesign features or relax gates only with documented rationale."
    )
    return f"""# HMM Stability Cross-Asset - {target_symbol.upper()}

## Scope

- Feature version: `{_feature_set_version(config)}`
- Split used for stability labels: `{cfg.get("split", "validation")}`
- K grid: `{_lab_cfg(config).get("k_values")}`
- Seeds: `{_lab_cfg(config).get("seeds")}`
- Feature sets: `{len(_lab_cfg(config).get("feature_sets", []))}`
- PnL diagnostic: `{cfg.get("pnl_diagnostic", "deferred_to_block_12")}`

State ids are aligned by feature-profile cosine similarity, not by HMM numeric state id. PnL is intentionally deferred to block 12.

## Status Counts

{_markdown_table(status_counts)}

## Stable Profile Candidates

{_markdown_table(stable.loc[:, [column for column in top_cols if column in stable.columns]], max_rows=int(cfg.get("top_rows", 30)))}

## Full Stability Grid

{_markdown_table(stability_grid.loc[:, [column for column in top_cols if column in stability_grid.columns]], max_rows=int(cfg.get("top_rows", 30)))}

## Label Stability Across K

{_markdown_table(label_summary, max_rows=int(cfg.get("top_rows", 30)))}

## Seed Alignment Summary

{_markdown_table(alignment_summary, max_rows=40)}

## Period / Regime Coverage

{_markdown_table(period_summary)}

## Ticker Dependency Sample

{_markdown_table(ticker_dependency.loc[:, dependency_cols] if not ticker_dependency.empty else ticker_dependency, max_rows=40)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    feature_config_path = Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml"))
    feature_config = load_yaml(feature_config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)

    features = pd.read_parquet(features_input_path(config, target, feature_config))
    posteriors = pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet")
    feature_sets = _feature_sets(config, feature_config)

    profiles = build_feature_profiles(features, posteriors, feature_sets)
    state_summary = build_state_summary(posteriors)
    state_names = build_state_names(profiles, state_summary, config)
    ticker_dependency, leave_one_out = build_ticker_dependency(profiles, feature_config, target)
    alignment = align_seed_states(profiles, state_names, config)
    period_occupancy = build_period_regime_occupancy(features, posteriors)
    stability_grid = build_stability_grid(state_names, alignment, ticker_dependency, leave_one_out, period_occupancy, config, target)
    label_summary = summarize_label_stability(stability_grid, config)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "state_stability_grid": results_dir / "state_stability_grid.parquet",
        "state_alignment_map": results_dir / "state_alignment_map.parquet",
        "state_profile_grid": results_dir / "state_profile_grid.parquet",
        "state_name_grid": results_dir / "state_name_grid.parquet",
        "state_label_stability": results_dir / "state_label_stability.parquet",
        "state_period_regime_occupancy": results_dir / "state_period_regime_occupancy.parquet",
        "state_ticker_dependency": results_dir / "state_ticker_dependency.parquet",
        "state_leave_one_ticker_out": results_dir / "state_leave_one_ticker_out.parquet",
    }
    outputs["state_stability_grid"].parent.mkdir(parents=True, exist_ok=True)
    stability_grid.to_parquet(outputs["state_stability_grid"], index=False)
    alignment.to_parquet(outputs["state_alignment_map"], index=False)
    profiles.to_parquet(outputs["state_profile_grid"], index=False)
    state_names.to_parquet(outputs["state_name_grid"], index=False)
    label_summary.to_parquet(outputs["state_label_stability"], index=False)
    period_occupancy.to_parquet(outputs["state_period_regime_occupancy"], index=False)
    ticker_dependency.to_parquet(outputs["state_ticker_dependency"], index=False)
    leave_one_out.to_parquet(outputs["state_leave_one_ticker_out"], index=False)
    report_path.write_text(
        render_report(config, target, stability_grid, alignment, label_summary, period_occupancy, ticker_dependency, outputs),
        encoding="utf-8",
    )
    return report_path, outputs["state_stability_grid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cross-asset HMM state stability across K, seeds, folds and periods.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, grid_path = run(args.config, args.target)
    print(f"HMM state stability report written to: {report_path}")
    print(f"State stability grid written to: {grid_path}")


if __name__ == "__main__":
    main()

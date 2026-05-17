from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.hmm_lab import (
    FULL_CORE_TOKEN,
    _feature_set_version,
    _lab_cfg,
    _target_symbol,
    features_input_path,
    load_yaml,
    report_output_path as lab_report_output_path,
    results_output_dir,
)


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
STATE_LABELS = (
    "risk_on_trend",
    "risk_off_stress",
    "defensive_rotation",
    "tech_led_narrow_rally",
    "chop_neutral",
    "high_volatility_expansion",
    "uninterpretable_noise",
)


HYPOTHESIS_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on_trend": {
        "risk_on_score": 1.0,
        "target_ret_12": 0.8,
        "target_ret_6": 0.5,
        "spread_credit_12": 0.7,
        "spread_equity_bonds_12": 0.7,
        "spread_equity_gold_12": 0.4,
        "relret_HYG_LQD_12": 0.6,
        "risk_off_score": -0.8,
        "intraday_stress_score": -0.4,
        "cross_asset_vol_expansion_score": -0.3,
    },
    "risk_off_stress": {
        "risk_off_score": 1.0,
        "intraday_stress_score": 0.9,
        "cross_asset_vol_expansion_score": 0.7,
        "market_range_ratio_6_24": 0.7,
        "relret_TLT_SPY_12": 0.5,
        "relret_IEF_SPY_12": 0.5,
        "relret_GLD_SPY_12": 0.5,
        "target_ret_12": -0.8,
        "target_ret_6": -0.5,
        "risk_on_score": -0.8,
        "spread_credit_12": -0.6,
        "spread_equity_bonds_12": -0.6,
        "spread_equity_gold_12": -0.4,
    },
    "defensive_rotation": {
        "relret_TLT_SPY_12": 0.7,
        "relret_IEF_SPY_12": 0.7,
        "relret_GLD_SPY_12": 0.6,
        "risk_off_score": 0.5,
        "spread_equity_bonds_12": -0.7,
        "spread_equity_gold_12": -0.6,
        "risk_on_score": -0.4,
        "target_ret_12": -0.3,
    },
    "tech_led_narrow_rally": {
        "narrow_rally_score": 1.0,
        "relret_QQQ_SPY_6": 0.8,
        "relret_XLK_SPY_6": 0.8,
        "spread_tech_broad_12": 0.7,
        "leadership_concentration_score_12": 0.6,
        "relret_IWM_SPY_6": -0.5,
    },
    "chop_neutral": {
        "chop_score": 1.0,
        "target_signed_efficiency_12": -0.4,
        "target_dir_persistence_12": -0.4,
        "target_dist_open": -0.2,
        "target_dist_vwap_atr": -0.2,
        "cross_asset_signal_conflict_score": 0.6,
    },
    "high_volatility_expansion": {
        "target_range_ratio_6_24": 0.9,
        "market_range_ratio_6_24": 0.9,
        "sector_range_dispersion_12": 0.7,
        "cross_asset_vol_expansion_score": 0.9,
        "intraday_stress_score": 0.7,
        "risk_off_score": 0.3,
    },
}


AGGREGATE_TICKER_MAP: dict[str, tuple[str, ...]] = {
    "spread_equity_bonds": ("SPY", "QQQ", "IWM", "DIA", "TLT", "IEF"),
    "spread_equity_gold": ("SPY", "GLD"),
    "spread_credit": ("HYG", "LQD"),
    "market_range_ratio": ("SPY", "QQQ", "IWM", "DIA"),
    "sector_range_dispersion": ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU"),
    "cross_asset_vol_expansion_score": ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU"),
    "intraday_stress_score": ("SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "HYG", "LQD"),
    "risk_on_score": ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLY", "HYG", "LQD", "TLT", "IEF"),
    "risk_off_score": ("SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "HYG", "LQD"),
    "defensive_rotation_score": ("QQQ", "XLK", "XLY", "IWM", "XLF", "XLE", "XLP", "XLV", "XLU", "TLT", "IEF", "GLD"),
    "narrow_rally_score": ("SPY", "QQQ", "IWM", "XLK"),
    "chop_score": ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU"),
}


def _interp_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hmm_state_interpretability", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "hmm_state_interpretability_cross_asset.md"


def reports_output_dir(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper()


def figures_output_dir(config: dict[str, Any], target_symbol: str) -> Path:
    return reports_output_dir(config, target_symbol) / "figures" / "state_interpretability"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
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


def _feature_set_columns(config: dict[str, Any], feature_config: dict[str, Any], feature_set: str) -> list[str]:
    for item in _lab_cfg(config).get("feature_sets", []):
        if str(item["name"]) != feature_set:
            continue
        columns = item.get("columns", [])
        if columns == FULL_CORE_TOKEN:
            return [str(column) for column in feature_config.get("hmm_feature_columns", [])]
        return [str(column) for column in columns]
    raise ValueError(f"Feature set not found in hmm_lab config: {feature_set}")


def select_interpretability_combo(summary: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    cfg = _interp_cfg(config)
    selected = summary.copy()
    feature_set = str(cfg.get("selected_feature_set", "auto"))
    n_states = cfg.get("selected_n_states", "auto")
    seed = cfg.get("selected_seed", "auto")

    if feature_set != "auto":
        selected = selected[selected["feature_set"] == feature_set]
    if n_states != "auto":
        selected = selected[selected["n_states"].astype(int) == int(n_states)]
    if seed != "auto":
        selected = selected[selected["seed"].astype(int) == int(seed)]

    if selected.empty:
        raise ValueError("No HMM lab summary row matches hmm_state_interpretability selection")

    if "validation_rank" in selected:
        selected = selected.sort_values("validation_rank", kind="stable")
    else:
        selected = selected.sort_values("validation_weighted_avg_loglik_per_feature", ascending=False, kind="stable")

    rank = int(cfg.get("selected_rank", 1))
    if feature_set == "auto" and n_states == "auto" and seed == "auto":
        selected = selected[selected["validation_rank"].astype(int) == rank] if "validation_rank" in selected else selected.iloc[[rank - 1]]
        if selected.empty:
            raise ValueError(f"No HMM lab summary row found for validation rank {rank}")
    return selected.iloc[0]


def load_selected_state_frame(
    features: pd.DataFrame,
    posteriors: pd.DataFrame,
    combo: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    selected = posteriors[
        (posteriors["feature_set"] == combo["feature_set"])
        & (posteriors["n_states"].astype(int) == int(combo["n_states"]))
        & (posteriors["seed"].astype(int) == int(combo["seed"]))
    ].copy()
    if selected.empty:
        raise ValueError("Selected HMM combo has no posterior rows")

    feature_frame = features.reset_index(names="source_index").loc[:, ["source_index", *feature_columns]]
    merged = selected.merge(feature_frame, on="source_index", how="left", validate="many_to_one")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged


def build_feature_profiles(state_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), split_frame in state_frame.groupby(["fold", "split"], sort=False):
        split_values = split_frame[feature_columns].replace([np.inf, -np.inf], np.nan)
        split_mean = split_values.mean()
        split_std = split_values.std(ddof=0)
        for state, group in split_frame.groupby("hmm_state", sort=True):
            state_values = group[feature_columns].replace([np.inf, -np.inf], np.nan)
            state_mean = state_values.mean()
            state_median = state_values.median()
            state_q10 = state_values.quantile(0.10)
            state_q90 = state_values.quantile(0.90)
            for feature in feature_columns:
                std = float(split_std[feature]) if feature in split_std else np.nan
                mean = float(split_mean[feature]) if feature in split_mean else np.nan
                current_mean = float(state_mean[feature]) if feature in state_mean else np.nan
                state_z = (current_mean - mean) / std if std and not np.isnan(std) else np.nan
                rows.append(
                    {
                        "fold": int(fold),
                        "split": str(split),
                        "hmm_state": int(state),
                        "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                        "feature": feature,
                        "state_rows": int(len(group)),
                        "split_rows": int(len(split_frame)),
                        "state_mean": current_mean,
                        "state_median": float(state_median[feature]),
                        "state_p10": float(state_q10[feature]),
                        "state_p90": float(state_q90[feature]),
                        "split_mean": mean,
                        "split_std": std,
                        "state_z": float(state_z) if not pd.isna(state_z) else np.nan,
                        "abs_state_z": float(abs(state_z)) if not pd.isna(state_z) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _state_runs(states: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for session, group in pd.DataFrame({"state": states.astype(int), "session": sessions}).groupby("session", sort=False):
        run_state: int | None = None
        run_length = 0
        for state in group["state"]:
            state_int = int(state)
            if run_state is None or state_int != run_state:
                if run_state is not None:
                    rows.append({"hmm_state": run_state, "duration": run_length})
                run_state = state_int
                run_length = 1
            else:
                run_length += 1
        if run_state is not None:
            rows.append({"hmm_state": run_state, "duration": run_length})
    return pd.DataFrame(rows)


def build_state_summary(state_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), split_frame in state_frame.groupby(["fold", "split"], sort=False):
        total = len(split_frame)
        durations = _state_runs(split_frame["hmm_state"], split_frame["session"])
        duration_stats = (
            durations.groupby("hmm_state", as_index=False).agg(mean_duration=("duration", "mean"), runs=("duration", "size"))
            if not durations.empty
            else pd.DataFrame(columns=["hmm_state", "mean_duration", "runs"])
        )
        for state, group in split_frame.groupby("hmm_state", sort=True):
            hours = group["timestamp"].dt.hour.value_counts(normalize=True)
            sessions = group["session"].value_counts(normalize=True)
            duration_row = duration_stats[duration_stats["hmm_state"] == int(state)]
            rows.append(
                {
                    "fold": int(fold),
                    "split": str(split),
                    "hmm_state": int(state),
                    "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                    "rows": int(len(group)),
                    "split_rows": int(total),
                    "state_frequency": float(len(group) / total) if total else 0.0,
                    "mean_duration": float(duration_row["mean_duration"].iloc[0]) if not duration_row.empty else 0.0,
                    "runs": int(duration_row["runs"].iloc[0]) if not duration_row.empty else 0,
                    "top_hour": int(hours.index[0]) if not hours.empty else None,
                    "top_hour_pct": float(hours.iloc[0]) if not hours.empty else np.nan,
                    "top_session": str(sessions.index[0]) if not sessions.empty else "",
                    "top_session_pct": float(sessions.iloc[0]) if not sessions.empty else np.nan,
                    "mean_hmm_entropy": float(group["hmm_entropy"].mean()),
                    "mean_hmm_max_prob": float(group["hmm_max_prob"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_hour_distribution(state_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), split_frame in state_frame.groupby(["fold", "split"], sort=False):
        split_hours = split_frame["timestamp"].dt.hour.value_counts().sort_index()
        split_total = int(split_hours.sum())
        for state, group in split_frame.groupby("hmm_state", sort=True):
            state_hours = group["timestamp"].dt.hour.value_counts().sort_index()
            state_total = int(state_hours.sum())
            for hour in sorted(set(split_hours.index).union(set(state_hours.index))):
                state_rows = int(state_hours.get(hour, 0))
                split_rows = int(split_hours.get(hour, 0))
                state_pct = state_rows / state_total if state_total else 0.0
                split_pct = split_rows / split_total if split_total else 0.0
                rows.append(
                    {
                        "fold": int(fold),
                        "split": str(split),
                        "hmm_state": int(state),
                        "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                        "hour": int(hour),
                        "state_rows": state_rows,
                        "split_rows": split_rows,
                        "state_pct": float(state_pct),
                        "split_pct": float(split_pct),
                        "hour_lift": float(state_pct / split_pct) if split_pct else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_period_occupancy(state_frame: pd.DataFrame) -> pd.DataFrame:
    frame = state_frame.copy()
    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(None)
    frame["month"] = timestamps.dt.to_period("M").astype(str)
    counts = frame.groupby(["fold", "split", "hmm_state", "month"], as_index=False).size().rename(columns={"size": "state_rows"})
    totals = frame.groupby(["fold", "split", "month"], as_index=False).size().rename(columns={"size": "month_rows"})
    out = counts.merge(totals, on=["fold", "split", "month"], how="left")
    out["fold_state_id"] = out.apply(lambda row: f"fold{int(row['fold'])}_state{int(row['hmm_state'])}", axis=1)
    out["state_frequency"] = out["state_rows"] / out["month_rows"]
    return out


def _known_symbols(feature_config: dict[str, Any], target_symbol: str) -> list[str]:
    symbols = {target_symbol.upper()}
    for values in feature_config.get("groups", {}).values():
        symbols.update(str(symbol).upper() for symbol in values)
    return sorted(symbols, key=len, reverse=True)


def feature_tickers(feature: str, known_symbols: list[str], target_symbol: str) -> tuple[str, ...]:
    if feature.startswith("target_"):
        return (target_symbol.upper(),)
    for prefix, symbols in AGGREGATE_TICKER_MAP.items():
        if feature.startswith(prefix):
            return tuple(symbol for symbol in symbols if symbol in known_symbols)
    tokens = set(re.split(r"[^A-Z0-9]+", feature.upper()))
    matches = tuple(symbol for symbol in known_symbols if symbol in tokens)
    return matches


def build_ticker_dependency(
    validation_profiles: pd.DataFrame,
    feature_config: dict[str, Any],
    target_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    known_symbols = _known_symbols(feature_config, target_symbol)
    dependency_rows: list[dict[str, Any]] = []
    leave_one_rows: list[dict[str, Any]] = []
    feature_map = {feature: feature_tickers(feature, known_symbols, target_symbol) for feature in validation_profiles["feature"].unique()}

    for (fold, state), group in validation_profiles.groupby(["fold", "hmm_state"], sort=True):
        group = group.dropna(subset=["abs_state_z"]).copy()
        total_abs = float(group["abs_state_z"].sum())
        ticker_scores = {symbol: 0.0 for symbol in known_symbols}
        for _, row in group.iterrows():
            tickers = feature_map.get(str(row["feature"]), ())
            if not tickers:
                continue
            contribution = float(row["abs_state_z"]) / len(tickers)
            for ticker in tickers:
                ticker_scores[ticker] += contribution
        nonzero = {ticker: score for ticker, score in ticker_scores.items() if score > 0}
        top_ticker = max(nonzero, key=nonzero.get) if nonzero else ""
        top_score = nonzero.get(top_ticker, 0.0)
        share = top_score / total_abs if total_abs else 0.0
        dependency_rows.append(
            {
                "fold": int(fold),
                "hmm_state": int(state),
                "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                "top_ticker": top_ticker,
                "top_ticker_abs_z_share": float(share),
                "total_feature_abs_z": total_abs,
                "ticker_count": int(len(nonzero)),
            }
        )

        full_vector = group.set_index("feature")["state_z"].fillna(0.0)
        full_norm = float(np.linalg.norm(full_vector.to_numpy()))
        for ticker in sorted(nonzero, key=nonzero.get, reverse=True):
            removed_features = [feature for feature, tickers in feature_map.items() if ticker in tickers and feature in full_vector.index]
            reduced = full_vector.copy()
            reduced.loc[removed_features] = 0.0
            reduced_norm = float(np.linalg.norm(reduced.to_numpy()))
            cosine = float(np.dot(full_vector.to_numpy(), reduced.to_numpy()) / (full_norm * reduced_norm)) if full_norm and reduced_norm else np.nan
            removed_share = float(group[group["feature"].isin(removed_features)]["abs_state_z"].sum() / total_abs) if total_abs else 0.0
            leave_one_rows.append(
                {
                    "fold": int(fold),
                    "hmm_state": int(state),
                    "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                    "ticker_removed": ticker,
                    "removed_feature_count": int(len(removed_features)),
                    "removed_abs_z_share": removed_share,
                    "profile_cosine_after_removal": cosine,
                }
            )
    return pd.DataFrame(dependency_rows), pd.DataFrame(leave_one_rows)


def _score_hypotheses(profile: pd.DataFrame) -> dict[str, float]:
    z = profile.set_index("feature")["state_z"].to_dict()
    scores: dict[str, float] = {}
    for label, weights in HYPOTHESIS_WEIGHTS.items():
        terms = []
        for feature, weight in weights.items():
            if feature in z and not pd.isna(z[feature]):
                terms.append(float(weight) * float(z[feature]))
        scores[label] = float(np.mean(terms)) if terms else np.nan

    directional = [feature for feature in ("target_ret_3", "target_ret_6", "target_ret_12", "risk_on_score", "risk_off_score") if feature in z]
    if "chop_score" not in z and directional:
        scores["chop_neutral"] = -float(np.mean([abs(float(z[feature])) for feature in directional if not pd.isna(z[feature])]))
    return scores


def assign_economic_label(profile: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    scores = _score_hypotheses(profile)
    valid_scores = {label: score for label, score in scores.items() if not pd.isna(score)}
    if not valid_scores:
        return {
            "proposed_label": "uninterpretable_noise",
            "best_score": np.nan,
            "second_score": np.nan,
            "score_margin": np.nan,
            "profile_strength": 0.0,
            "top_positive_features": "",
            "top_negative_features": "",
        }

    ranked = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else np.nan
    margin = best_score - second_score if not pd.isna(second_score) else np.nan
    top = profile.dropna(subset=["state_z"]).sort_values("state_z", ascending=False)
    bottom = profile.dropna(subset=["state_z"]).sort_values("state_z", ascending=True)
    profile_strength = float(profile["abs_state_z"].max()) if not profile["abs_state_z"].dropna().empty else 0.0
    if profile_strength < float(cfg.get("min_abs_z_clear", 0.5)) or best_score < float(cfg.get("min_score_partial", 0.25)):
        best_label = "uninterpretable_noise"
    return {
        "proposed_label": best_label,
        "best_score": float(best_score),
        "second_score": float(second_score) if not pd.isna(second_score) else np.nan,
        "score_margin": float(margin) if not pd.isna(margin) else np.nan,
        "profile_strength": profile_strength,
        "top_positive_features": ", ".join(f"{row.feature}:{row.state_z:.2f}" for row in top.head(int(cfg.get("top_features", 5))).itertuples()),
        "top_negative_features": ", ".join(f"{row.feature}:{row.state_z:.2f}" for row in bottom.head(int(cfg.get("top_features", 5))).itertuples()),
        **{f"score_{label}": score for label, score in scores.items()},
    }


def build_state_names(
    profiles: pd.DataFrame,
    state_summary: pd.DataFrame,
    ticker_dependency: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    config: dict[str, Any],
    target_symbol: str,
) -> pd.DataFrame:
    cfg = _interp_cfg(config)
    naming_split = str(cfg.get("naming_split", "validation"))
    reference_split = str(cfg.get("reference_split", "train"))
    rows: list[dict[str, Any]] = []

    label_by_reference: dict[tuple[int, int], str] = {}
    reference_profiles = profiles[profiles["split"] == reference_split]
    for (fold, state), group in reference_profiles.groupby(["fold", "hmm_state"], sort=True):
        label_by_reference[(int(fold), int(state))] = str(assign_economic_label(group, cfg)["proposed_label"])

    naming_profiles = profiles[profiles["split"] == naming_split]
    for (fold, state), group in naming_profiles.groupby(["fold", "hmm_state"], sort=True):
        assigned = assign_economic_label(group, cfg)
        summary_row = state_summary[
            (state_summary["fold"].astype(int) == int(fold))
            & (state_summary["hmm_state"].astype(int) == int(state))
            & (state_summary["split"] == naming_split)
        ]
        dep_row = ticker_dependency[
            (ticker_dependency["fold"].astype(int) == int(fold)) & (ticker_dependency["hmm_state"].astype(int) == int(state))
        ]
        loo = leave_one_out[
            (leave_one_out["fold"].astype(int) == int(fold)) & (leave_one_out["hmm_state"].astype(int) == int(state))
        ].sort_values("removed_abs_z_share", ascending=False)
        summary_data = summary_row.iloc[0].to_dict() if not summary_row.empty else {}
        dep_data = dep_row.iloc[0].to_dict() if not dep_row.empty else {}
        reference_label = label_by_reference.get((int(fold), int(state)), "")

        time_dominated = float(summary_data.get("top_hour_pct", 0.0)) > float(cfg.get("max_hour_pct", 0.35))
        session_dominated = float(summary_data.get("top_session_pct", 0.0)) > float(cfg.get("max_session_pct", 0.08))
        ticker_dominated = (
            str(dep_data.get("top_ticker", "")) != target_symbol.upper()
            and float(dep_data.get("top_ticker_abs_z_share", 0.0)) > float(cfg.get("max_non_target_ticker_share", 0.6))
        )
        weak_profile = (
            assigned["proposed_label"] == "uninterpretable_noise"
            or float(assigned.get("best_score", 0.0)) < float(cfg.get("min_score_partial", 0.25))
        )
        train_validation_match = reference_label == assigned["proposed_label"] if reference_label else False

        if weak_profile or time_dominated or session_dominated:
            interpretability = "not_interpretable"
        elif (
            float(assigned.get("best_score", 0.0)) >= float(cfg.get("min_score_clear", 0.45))
            and float(assigned.get("score_margin", 0.0)) >= float(cfg.get("min_score_margin_clear", 0.15))
            and train_validation_match
            and not ticker_dominated
        ):
            interpretability = "interpretable"
        else:
            interpretability = "partially_interpretable"

        rows.append(
            {
                "fold": int(fold),
                "hmm_state": int(state),
                "fold_state_id": f"fold{int(fold)}_state{int(state)}",
                "naming_split": naming_split,
                "reference_split": reference_split,
                "proposed_label": assigned["proposed_label"],
                "reference_label": reference_label,
                "train_validation_label_match": bool(train_validation_match),
                "interpretability": interpretability,
                "risk_time_dominated": bool(time_dominated),
                "risk_session_dominated": bool(session_dominated),
                "risk_ticker_dominated": bool(ticker_dominated),
                "state_frequency": float(summary_data.get("state_frequency", np.nan)),
                "mean_duration": float(summary_data.get("mean_duration", np.nan)),
                "top_hour": summary_data.get("top_hour"),
                "top_hour_pct": float(summary_data.get("top_hour_pct", np.nan)),
                "top_session": summary_data.get("top_session", ""),
                "top_session_pct": float(summary_data.get("top_session_pct", np.nan)),
                "top_ticker": dep_data.get("top_ticker", ""),
                "top_ticker_abs_z_share": float(dep_data.get("top_ticker_abs_z_share", np.nan)),
                "worst_leave_one_ticker": str(loo["ticker_removed"].iloc[0]) if not loo.empty else "",
                "worst_leave_one_removed_abs_z_share": float(loo["removed_abs_z_share"].iloc[0]) if not loo.empty else np.nan,
                **assigned,
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "hmm_state"]).reset_index(drop=True)


def _plot_heatmap(profiles: pd.DataFrame, state_names: pd.DataFrame, output_path: Path) -> None:
    naming = profiles[profiles["split"] == "validation"].copy()
    if naming.empty:
        return
    matrix = naming.pivot_table(index="fold_state_id", columns="feature", values="state_z", aggfunc="mean")
    ordered_states = state_names["fold_state_id"].tolist()
    matrix = matrix.reindex([state for state in ordered_states if state in matrix.index])
    labels = state_names.set_index("fold_state_id").reindex(matrix.index)["proposed_label"].fillna("").tolist()

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(matrix.columns)), max(4, 0.35 * len(matrix.index))))
    im = ax.imshow(matrix.fillna(0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{idx} | {label}" for idx, label in zip(matrix.index, labels)], fontsize=8)
    ax.set_title("Validation feature z-score by fold-local state")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_hour_distribution(hour_distribution: pd.DataFrame, state_names: pd.DataFrame, output_path: Path) -> None:
    validation = hour_distribution[hour_distribution["split"] == "validation"].copy()
    if validation.empty:
        return
    pivot = validation.pivot_table(index="hour", columns="fold_state_id", values="state_pct", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    for column in pivot.columns:
        label = state_names.set_index("fold_state_id").get("proposed_label", pd.Series()).get(column, "")
        ax.plot(pivot.index, pivot[column], marker="o", linewidth=1, label=f"{column} {label}")
    ax.set_title("Validation hour distribution by fold-local state")
    ax.set_xlabel("Hour")
    ax.set_ylabel("State row share")
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_fold_label_distribution(state_names: pd.DataFrame, output_path: Path) -> None:
    if state_names.empty:
        return
    pivot = state_names.pivot_table(index="proposed_label", columns="fold", values="state_frequency", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Validation occupancy by proposed label and fold")
    ax.set_ylabel("Total state frequency")
    ax.set_xlabel("Proposed label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_transition_matrix(transitions: pd.DataFrame, output_path: Path) -> None:
    validation = transitions[transitions["split"] == "validation"].copy()
    if validation.empty:
        return
    matrix = validation.pivot_table(index="from_state", columns="to_state", values="transition_probability", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=max(1e-6, float(matrix.to_numpy().max())))
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Validation transition matrix, averaged across folds")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_period_occupancy(period_occupancy: pd.DataFrame, state_names: pd.DataFrame, output_path: Path) -> None:
    validation = period_occupancy[period_occupancy["split"] == "validation"].copy()
    if validation.empty:
        return
    validation = validation.merge(state_names[["fold_state_id", "proposed_label"]], on="fold_state_id", how="left")
    grouped = validation.groupby(["month", "proposed_label"], as_index=False)["state_frequency"].mean()
    pivot = grouped.pivot_table(index="month", columns="proposed_label", values="state_frequency", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(ax=ax, marker="o")
    ax.set_title("Validation occupancy by month and proposed label")
    ax.set_ylabel("Average state frequency")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_visualizations(
    profiles: pd.DataFrame,
    hour_distribution: pd.DataFrame,
    period_occupancy: pd.DataFrame,
    transitions: pd.DataFrame,
    state_names: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_heatmap": output_dir / "feature_state_heatmap.png",
        "hour_distribution": output_dir / "hour_distribution.png",
        "fold_distribution": output_dir / "fold_label_distribution.png",
        "transition_matrix": output_dir / "transition_matrix.png",
        "period_occupancy": output_dir / "period_occupancy.png",
    }
    _plot_heatmap(profiles, state_names, paths["feature_heatmap"])
    _plot_hour_distribution(hour_distribution, state_names, paths["hour_distribution"])
    _plot_fold_label_distribution(state_names, paths["fold_distribution"])
    _plot_transition_matrix(transitions, paths["transition_matrix"])
    _plot_period_occupancy(period_occupancy, state_names, paths["period_occupancy"])
    return paths


def state_names_to_yaml(
    config: dict[str, Any],
    target_symbol: str,
    combo: pd.Series,
    state_names: pd.DataFrame,
) -> str:
    payload = {
        "target_symbol": target_symbol.upper(),
        "feature_set_version": _feature_set_version(config),
        "selection_rule": "validation_rank_only_no_pnl",
        "selected_combo": {
            "feature_set": str(combo["feature_set"]),
            "n_states": int(combo["n_states"]),
            "seed": int(combo["seed"]),
            "validation_rank": int(combo.get("validation_rank", 0)),
        },
        "naming_split": str(_interp_cfg(config).get("naming_split", "validation")),
        "state_names": [
            {
                "fold": int(row.fold),
                "hmm_state": int(row.hmm_state),
                "fold_state_id": row.fold_state_id,
                "proposed_label": row.proposed_label,
                "reference_label": row.reference_label,
                "interpretability": row.interpretability,
                "top_positive_features": row.top_positive_features,
                "top_negative_features": row.top_negative_features,
                "risks": [
                    risk
                    for risk, active in {
                        "time_dominated": bool(row.risk_time_dominated),
                        "session_dominated": bool(row.risk_session_dominated),
                        "ticker_dominated": bool(row.risk_ticker_dominated),
                        "train_validation_label_mismatch": not bool(row.train_validation_label_match),
                    }.items()
                    if active
                ],
            }
            for row in state_names.itertuples()
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    combo: pd.Series,
    state_names: pd.DataFrame,
    profiles: pd.DataFrame,
    ticker_dependency: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    hour_distribution: pd.DataFrame,
    output_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> str:
    label_counts = state_names["interpretability"].value_counts().rename_axis("interpretability").reset_index(name="states")
    label_summary = (
        state_names.groupby(["proposed_label", "interpretability"], as_index=False)
        .agg(states=("fold_state_id", "count"), avg_frequency=("state_frequency", "mean"), avg_duration=("mean_duration", "mean"))
        .sort_values(["interpretability", "states"], ascending=[True, False])
    )
    state_cols = [
        "fold_state_id",
        "proposed_label",
        "reference_label",
        "train_validation_label_match",
        "interpretability",
        "state_frequency",
        "mean_duration",
        "top_hour",
        "top_hour_pct",
        "top_session_pct",
        "top_ticker",
        "top_ticker_abs_z_share",
        "worst_leave_one_ticker",
        "worst_leave_one_removed_abs_z_share",
        "best_score",
        "score_margin",
        "top_positive_features",
        "top_negative_features",
    ]
    dep_cols = ["fold_state_id", "top_ticker", "top_ticker_abs_z_share", "ticker_count", "total_feature_abs_z"]
    loo_cols = ["fold_state_id", "ticker_removed", "removed_feature_count", "removed_abs_z_share", "profile_cosine_after_removal"]
    top_profile = (
        profiles[profiles["split"] == str(_interp_cfg(config).get("naming_split", "validation"))]
        .sort_values(["fold", "hmm_state", "abs_state_z"], ascending=[True, True, False])
        .groupby(["fold_state_id"], as_index=False)
        .head(int(_interp_cfg(config).get("top_features", 5)))
        .loc[:, ["fold_state_id", "feature", "state_z", "state_mean", "state_median"]]
    )
    hour_summary = (
        hour_distribution[hour_distribution["split"] == str(_interp_cfg(config).get("naming_split", "validation"))]
        .sort_values(["fold", "hmm_state", "state_pct"], ascending=[True, True, False])
        .groupby("fold_state_id", as_index=False)
        .head(1)
        .loc[:, ["fold_state_id", "hour", "state_pct", "hour_lift"]]
    )
    figures = "\n".join(f"- {name}: `{path}`" for name, path in figure_paths.items())
    outputs = "\n".join(f"- {name}: `{path}`" for name, path in output_paths.items())
    interpretable_count = int(state_names["interpretability"].eq("interpretable").sum())
    partial_count = int(state_names["interpretability"].eq("partially_interpretable").sum())
    conclusion = (
        "Advance to stability with caution: at least 2-3 fold-local states are interpretable or partially interpretable before PnL."
        if interpretable_count + partial_count >= 3
        else "Do not advance as accepted: too few states are interpretable before PnL."
    )
    return f"""# HMM State Interpretability Cross-Asset - {target_symbol.upper()}

## Scope

- Selected only from validation diagnostics: rank `{int(combo.get("validation_rank", 0))}`
- Feature set: `{combo["feature_set"]}`
- K: `{int(combo["n_states"])}`
- Seed: `{int(combo["seed"])}`
- Naming split: `{_interp_cfg(config).get("naming_split", "validation")}`
- Reference split for stability sanity: `{_interp_cfg(config).get("reference_split", "train")}`
- PnL/forward returns used: `no`

State IDs are interpreted as fold-local in this report. Cross-fold/seed state alignment belongs to the next block.

## Interpretability Counts

{_markdown_table(label_counts)}

## Label Summary

{_markdown_table(label_summary)}

## Proposed State Names

{_markdown_table(state_names.loc[:, state_cols], max_rows=40)}

## Strongest Feature Evidence

{_markdown_table(top_profile, max_rows=80)}

## Time Concentration

{_markdown_table(hour_summary, max_rows=40)}

## Ticker Dependency

{_markdown_table(ticker_dependency.loc[:, dep_cols], max_rows=40)}

## Leave-One-Ticker-Out Profile Sensitivity

{_markdown_table(leave_one_out.sort_values(["fold", "hmm_state", "removed_abs_z_share"], ascending=[True, True, False]).loc[:, loo_cols], max_rows=60)}

## Outputs

{outputs}

## Figures

{figures}

## Conclusion

{conclusion}

No state name in this report is based on PnL. If later economics contradict a name, rename only with feature-profile evidence.
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    feature_config_path = Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml"))
    feature_config = load_yaml(feature_config_path)
    target = _target_symbol(config, target_symbol)

    results_dir = results_output_dir(config, target)
    summary = pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_summary.parquet")
    posteriors = pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet")
    transitions_all = pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_transitions.parquet")
    combo = select_interpretability_combo(summary, config)
    feature_columns = _feature_set_columns(config, feature_config, str(combo["feature_set"]))
    features = pd.read_parquet(features_input_path(config, target, feature_config))
    selected_frame = load_selected_state_frame(features, posteriors, combo, feature_columns)

    profiles = build_feature_profiles(selected_frame, feature_columns)
    state_summary = build_state_summary(selected_frame)
    hour_distribution = build_hour_distribution(selected_frame)
    period_occupancy = build_period_occupancy(selected_frame)
    validation_profiles = profiles[profiles["split"] == str(_interp_cfg(config).get("naming_split", "validation"))].copy()
    ticker_dependency, leave_one_out = build_ticker_dependency(validation_profiles, feature_config, target)
    state_names = build_state_names(profiles, state_summary, ticker_dependency, leave_one_out, config, target)
    transitions = transitions_all[
        (transitions_all["feature_set"] == combo["feature_set"])
        & (transitions_all["n_states"].astype(int) == int(combo["n_states"]))
        & (transitions_all["seed"].astype(int) == int(combo["seed"]))
    ].copy()

    report_dir = reports_output_dir(config, target)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "state_feature_profiles": report_dir / "state_feature_profiles.parquet",
        "state_summary": report_dir / "state_interpretability_summary.parquet",
        "state_names": report_dir / "state_names_pre_pnl.yaml",
        "hour_distribution": report_dir / "state_hour_distribution_cross_asset.parquet",
        "period_occupancy": report_dir / "state_period_occupancy_cross_asset.parquet",
        "ticker_dependency": report_dir / "state_ticker_dependency.parquet",
        "leave_one_ticker_out": report_dir / "state_leave_one_ticker_out.parquet",
        "transitions": report_dir / "state_transition_matrix_cross_asset.parquet",
    }
    profiles.to_parquet(output_paths["state_feature_profiles"], index=False)
    state_summary.to_parquet(output_paths["state_summary"], index=False)
    hour_distribution.to_parquet(output_paths["hour_distribution"], index=False)
    period_occupancy.to_parquet(output_paths["period_occupancy"], index=False)
    ticker_dependency.to_parquet(output_paths["ticker_dependency"], index=False)
    leave_one_out.to_parquet(output_paths["leave_one_ticker_out"], index=False)
    transitions.to_parquet(output_paths["transitions"], index=False)
    output_paths["state_names"].write_text(state_names_to_yaml(config, target, combo, state_names), encoding="utf-8")

    figure_paths = write_visualizations(profiles, hour_distribution, period_occupancy, transitions, state_names, figures_output_dir(config, target))
    report_path = report_output_path(config, target)
    report_path.write_text(
        render_report(config, target, combo, state_names, profiles, ticker_dependency, leave_one_out, hour_distribution, output_paths, figure_paths),
        encoding="utf-8",
    )
    return report_path, output_paths["state_feature_profiles"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and name cross-asset HMM states before PnL.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, profiles_path = run(args.config, args.target)
    print(f"HMM state interpretability report written to: {report_path}")
    print(f"State feature profiles written to: {profiles_path}")


if __name__ == "__main__":
    main()

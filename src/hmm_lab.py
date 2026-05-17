from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.hmm_filter import filtered_probabilities


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
FULL_CORE_TOKEN = "__hmm_feature_columns__"


@dataclass(frozen=True)
class LabFold:
    fold: int
    train_months: list[str]
    validation_months: list[str]
    test_months: list[str]
    train_sessions: list[str]
    validation_sessions: list[str]
    test_sessions: list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _lab_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hmm_lab", {})


def _target_symbol(config: dict[str, Any], target_symbol: str | None = None) -> str:
    return (target_symbol or config.get("lab", {}).get("target_symbol", "SPY")).upper()


def _timeframe(config: dict[str, Any]) -> str:
    return str(config.get("lab", {}).get("timeframe", config.get("project", {}).get("frequency", "5min")))


def _universe_id(config: dict[str, Any]) -> str:
    return str(config.get("lab", {}).get("universe_id", "core_cross_asset_v1"))


def _feature_set_version(config: dict[str, Any], feature_config: dict[str, Any] | None = None) -> str:
    lab_version = _lab_cfg(config).get("feature_set_version")
    if lab_version:
        return str(lab_version)
    if feature_config:
        return str(feature_config.get("feature_set_version", "cross_asset_v1"))
    return "cross_asset_v1"


def features_input_path(config: dict[str, Any], target_symbol: str, feature_config: dict[str, Any] | None = None) -> Path:
    features_dir = Path(config.get("paths", {}).get("features_dir", "data/features"))
    return features_dir / target_symbol.upper() / _timeframe(config) / _universe_id(config) / _feature_set_version(config, feature_config) / "features.parquet"


def model_output_root(config: dict[str, Any], target_symbol: str, feature_config: dict[str, Any] | None = None) -> Path:
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    return models_dir / target_symbol.upper() / _timeframe(config) / _universe_id(config) / _feature_set_version(config, feature_config) / "hmm_lab"


def results_output_dir(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("results_dir", "results")) / target_symbol.upper()


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "hmm_feature_lab_cross_asset.md"


def _session_months(frame: pd.DataFrame) -> pd.DataFrame:
    sessions = frame[["session"]].drop_duplicates().copy()
    sessions["session_date"] = pd.to_datetime(sessions["session"])
    sessions["month"] = sessions["session_date"].dt.to_period("M").astype(str)
    return sessions


def build_lab_folds(frame: pd.DataFrame, config: dict[str, Any]) -> list[LabFold]:
    wf_cfg = _lab_cfg(config).get("walk_forward", config.get("walkforward", {}))
    train_months = int(wf_cfg.get("train_months", 24))
    validation_months = int(wf_cfg.get("validation_months", 6))
    test_months = int(wf_cfg.get("test_months", 6))
    step_months = int(wf_cfg.get("step_months", 6))
    max_folds = _lab_cfg(config).get("max_folds")

    session_months = _session_months(frame)
    months = sorted(session_months["month"].unique().tolist())
    window = train_months + validation_months + test_months
    folds: list[LabFold] = []

    fold_id = 0
    for start in range(0, max(len(months) - window + 1, 0), step_months):
        train = months[start : start + train_months]
        validation = months[start + train_months : start + train_months + validation_months]
        test = months[start + train_months + validation_months : start + window]
        if len(train) != train_months or len(validation) != validation_months or len(test) != test_months:
            continue
        folds.append(
            LabFold(
                fold=fold_id,
                train_months=train,
                validation_months=validation,
                test_months=test,
                train_sessions=session_months.loc[session_months["month"].isin(train), "session"].tolist(),
                validation_sessions=session_months.loc[session_months["month"].isin(validation), "session"].tolist(),
                test_sessions=session_months.loc[session_months["month"].isin(test), "session"].tolist(),
            )
        )
        fold_id += 1
        if max_folds is not None and len(folds) >= int(max_folds):
            break
    return folds


def _configured_k_values(config: dict[str, Any]) -> list[int]:
    values = [int(value) for value in _lab_cfg(config).get("k_values", [3, 4, 5, 6])]
    if not values:
        raise ValueError("hmm_lab.k_values must contain at least one K")
    if max(values) > 7 and not bool(_lab_cfg(config).get("allow_k_above_7", False)):
        raise ValueError("HMM K above 7 requires hmm_lab.allow_k_above_7: true")
    return values


def _configured_seeds(config: dict[str, Any]) -> list[int]:
    values = [int(value) for value in _lab_cfg(config).get("seeds", [42])]
    if not values:
        raise ValueError("hmm_lab.seeds must contain at least one seed")
    return values


def _feature_set_columns(raw_columns: Any, feature_config: dict[str, Any]) -> list[str]:
    if raw_columns == FULL_CORE_TOKEN:
        return [str(column) for column in feature_config.get("hmm_feature_columns", [])]
    return [str(column) for column in raw_columns]


def resolve_feature_sets(config: dict[str, Any], feature_config: dict[str, Any], features: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    configured_sets = _lab_cfg(config).get("feature_sets", [])
    if not configured_sets:
        configured_sets = [{"name": "cross_asset_full_core", "columns": FULL_CORE_TOKEN}]

    rows: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    available = set(features.columns)
    for item in configured_sets:
        columns = _feature_set_columns(item.get("columns", []), feature_config)
        missing = sorted(set(columns) - available)
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        status = "ready" if not missing and not duplicates else "missing_columns" if missing else "duplicate_columns"
        row = {
            "feature_set": str(item["name"]),
            "description": str(item.get("description", "")),
            "n_features": len(columns),
            "columns": ",".join(columns),
            "missing_columns": ",".join(missing),
            "duplicate_columns": ",".join(duplicates),
            "status": status,
        }
        rows.append(row)
        if status == "ready":
            ready.append({"name": row["feature_set"], "description": row["description"], "columns": columns})
    return ready, pd.DataFrame(rows)


def prepare_hmm_frame(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required = set(INDEX_COLUMNS) | set(feature_columns)
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Features data is missing required HMM columns: {missing}")

    frame = features.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=False).rename(columns={"index": "source_index"})
    numeric = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    valid_mask = numeric.notna().all(axis=1)
    frame.loc[:, feature_columns] = numeric
    return frame.loc[valid_mask].reset_index(drop=True)


def _lengths_by_session(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("session", sort=False).size().astype(int).tolist()


def _fit_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    n_states: int,
    seed: int,
    config: dict[str, Any],
) -> tuple[GaussianHMM, StandardScaler]:
    lab_cfg = _lab_cfg(config)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_frame[feature_columns].to_numpy())
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type=str(lab_cfg.get("covariance_type", "diag")),
        n_iter=int(lab_cfg.get("n_iter", 80)),
        random_state=int(seed),
        min_covar=float(lab_cfg.get("min_covar", 1e-4)),
    )
    model.fit(train_x, lengths=_lengths_by_session(train_frame))
    return model, scaler


def _state_runs(states: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for _, group in pd.DataFrame({"state": states.astype(int), "session": sessions}).groupby("session", sort=False):
        run_state: int | None = None
        run_length = 0
        for state in group["state"]:
            if run_state is None or int(state) != run_state:
                if run_state is not None:
                    rows.append({"hmm_state": run_state, "duration": run_length})
                run_state = int(state)
                run_length = 1
            else:
                run_length += 1
        if run_state is not None:
            rows.append({"hmm_state": run_state, "duration": run_length})
    return pd.DataFrame(rows)


def _annotate_split(
    model: GaussianHMM,
    scaler: StandardScaler,
    split_frame: pd.DataFrame,
    feature_columns: list[str],
    n_states: int,
) -> tuple[pd.DataFrame, float, float]:
    if split_frame.empty:
        return pd.DataFrame(), np.nan, np.nan
    observations = scaler.transform(split_frame[feature_columns].to_numpy())
    probabilities = filtered_probabilities(model, observations, split_frame["session"])
    total_loglik = float(model.score(observations, lengths=_lengths_by_session(split_frame)))
    avg_loglik = total_loglik / len(split_frame)

    annotated = split_frame[["source_index", *INDEX_COLUMNS]].copy()
    for state in range(n_states):
        annotated[f"hmm_p{state}"] = probabilities[:, state]
    annotated["hmm_state"] = probabilities.argmax(axis=1).astype(int)
    annotated["hmm_max_prob"] = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))).sum(axis=1)
    annotated["hmm_entropy"] = entropy / np.log(n_states)
    return annotated, total_loglik, avg_loglik


def _state_occupancy(annotated: pd.DataFrame, n_states: int) -> pd.DataFrame:
    if annotated.empty:
        return pd.DataFrame(columns=["hmm_state", "state_rows", "state_frequency", "mean_duration", "runs"])
    counts = annotated["hmm_state"].value_counts().reindex(range(n_states), fill_value=0).rename_axis("hmm_state").reset_index(name="state_rows")
    counts["state_frequency"] = counts["state_rows"] / len(annotated)
    durations = _state_runs(annotated["hmm_state"], annotated["session"])
    if durations.empty:
        counts["mean_duration"] = 0.0
        counts["runs"] = 0
        return counts
    duration_stats = durations.groupby("hmm_state", as_index=False).agg(mean_duration=("duration", "mean"), runs=("duration", "size"))
    return counts.merge(duration_stats, on="hmm_state", how="left").fillna({"mean_duration": 0.0, "runs": 0})


def _transition_rows(annotated: pd.DataFrame, n_states: int) -> list[dict[str, Any]]:
    if annotated.empty:
        return []
    ordered = annotated.sort_values(["session", "bar_index"], kind="stable").copy()
    ordered["next_state"] = ordered.groupby("session", sort=False)["hmm_state"].shift(-1)
    valid = ordered.dropna(subset=["next_state"]).copy()
    if valid.empty:
        return []
    rows: list[dict[str, Any]] = []
    counts = valid.groupby(["hmm_state", "next_state"], as_index=False).size().rename(columns={"size": "transitions"})
    totals = counts.groupby("hmm_state")["transitions"].sum().rename("state_total").reset_index()
    counts = counts.merge(totals, on="hmm_state", how="left")
    for from_state in range(n_states):
        state_counts = counts[counts["hmm_state"] == from_state]
        for to_state in range(n_states):
            match = state_counts[state_counts["next_state"] == float(to_state)]
            transitions = int(match["transitions"].iloc[0]) if not match.empty else 0
            state_total = int(state_counts["state_total"].iloc[0]) if not state_counts.empty else 0
            rows.append(
                {
                    "from_state": int(from_state),
                    "to_state": int(to_state),
                    "transitions": transitions,
                    "transition_probability": transitions / state_total if state_total else 0.0,
                }
            )
    return rows


def _hour_concentration(annotated: pd.DataFrame) -> tuple[float, int | None]:
    if annotated.empty:
        return np.nan, None
    hours = pd.to_datetime(annotated["timestamp"]).dt.hour
    top_pct = float(hours.value_counts(normalize=True).iloc[0])
    top_hour = int(hours.value_counts().index[0])
    return top_pct, top_hour


def _split_slices(frame: pd.DataFrame, fold: LabFold) -> dict[str, pd.DataFrame]:
    sessions = {
        "train": fold.train_sessions,
        "validation": fold.validation_sessions,
        "test": fold.test_sessions,
    }
    return {split: frame[frame["session"].isin(split_sessions)].copy() for split, split_sessions in sessions.items()}


def _combo_id(feature_set: str, n_states: int, seed: int, fold: int) -> str:
    return f"{feature_set}__k{int(n_states)}__seed{int(seed)}__fold{int(fold)}"


def run_feature_set_grid(
    features: pd.DataFrame,
    feature_set: dict[str, Any],
    folds: list[LabFold],
    config: dict[str, Any],
    target_symbol: str,
    feature_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    hmm_frame = prepare_hmm_frame(features, feature_set["columns"])
    k_values = _configured_k_values(config)
    seeds = _configured_seeds(config)
    min_state_frequency = float(_lab_cfg(config).get("min_state_frequency", 0.01))
    model_root = model_output_root(config, target_symbol, feature_config)

    metric_rows: list[dict[str, Any]] = []
    posterior_frames: list[pd.DataFrame] = []
    occupancy_frames: list[pd.DataFrame] = []
    transition_rows: list[dict[str, Any]] = []
    save_posteriors = bool(_lab_cfg(config).get("save_posteriors", True))

    for n_states in k_values:
        for seed in seeds:
            for fold in folds:
                combo_id = _combo_id(feature_set["name"], n_states, seed, fold.fold)
                fold_frame = hmm_frame[
                    hmm_frame["session"].isin(fold.train_sessions + fold.validation_sessions + fold.test_sessions)
                ].copy()
                split_frames = _split_slices(fold_frame, fold)
                train_frame = split_frames["train"]
                if train_frame.empty:
                    metric_rows.append(
                        {
                            "target_symbol": target_symbol,
                            "feature_set_version": _feature_set_version(config, feature_config),
                            "feature_set": feature_set["name"],
                            "n_features": len(feature_set["columns"]),
                            "n_states": int(n_states),
                            "seed": int(seed),
                            "fold": int(fold.fold),
                            "split": "train",
                            "status": "empty_train",
                        }
                    )
                    continue

                try:
                    model, scaler = _fit_model(train_frame, feature_set["columns"], n_states, seed, config)
                except Exception as exc:  # pragma: no cover - defensive path for ill-conditioned real grids
                    metric_rows.append(
                        {
                            "target_symbol": target_symbol,
                            "feature_set_version": _feature_set_version(config, feature_config),
                            "feature_set": feature_set["name"],
                            "n_features": len(feature_set["columns"]),
                            "n_states": int(n_states),
                            "seed": int(seed),
                            "fold": int(fold.fold),
                            "split": "train",
                            "status": "fit_failed",
                            "error": str(exc),
                        }
                    )
                    continue

                model_dir = model_root / feature_set["name"] / f"k{int(n_states)}_seed{int(seed)}" / f"fold_{int(fold.fold)}"
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {
                        "model": model,
                        "scaler": scaler,
                        "feature_columns": feature_set["columns"],
                        "target_symbol": target_symbol,
                        "feature_set": feature_set["name"],
                        "n_states": int(n_states),
                        "seed": int(seed),
                        "fold": int(fold.fold),
                        "train_sessions": fold.train_sessions,
                    },
                    model_dir / "hmm.joblib",
                )

                for split, split_frame in split_frames.items():
                    annotated, total_loglik, avg_loglik = _annotate_split(model, scaler, split_frame, feature_set["columns"], n_states)
                    occupancy = _state_occupancy(annotated, n_states)
                    if not occupancy.empty:
                        annotated_occupancy = occupancy.copy()
                        annotated_occupancy.insert(0, "split", split)
                        annotated_occupancy.insert(0, "fold", int(fold.fold))
                        annotated_occupancy.insert(0, "seed", int(seed))
                        annotated_occupancy.insert(0, "n_states", int(n_states))
                        annotated_occupancy.insert(0, "feature_set", feature_set["name"])
                        occupancy_frames.append(annotated_occupancy)

                    for row in _transition_rows(annotated, n_states):
                        transition_rows.append(
                            {
                                "feature_set": feature_set["name"],
                                "n_states": int(n_states),
                                "seed": int(seed),
                                "fold": int(fold.fold),
                                "split": split,
                                **row,
                            }
                        )

                    top_hour_pct, top_hour = _hour_concentration(annotated)
                    state_frequencies = occupancy["state_frequency"] if not occupancy.empty else pd.Series(dtype=float)
                    metric_rows.append(
                        {
                            "target_symbol": target_symbol,
                            "feature_set_version": _feature_set_version(config, feature_config),
                            "feature_set": feature_set["name"],
                            "feature_description": feature_set.get("description", ""),
                            "n_features": len(feature_set["columns"]),
                            "feature_columns": ",".join(feature_set["columns"]),
                            "n_states": int(n_states),
                            "seed": int(seed),
                            "fold": int(fold.fold),
                            "train_months": ",".join(fold.train_months),
                            "validation_months": ",".join(fold.validation_months),
                            "test_months": ",".join(fold.test_months),
                            "split": split,
                            "status": "ok",
                            "rows": int(len(split_frame)),
                            "sessions": int(split_frame["session"].nunique()),
                            "total_loglik": total_loglik,
                            "avg_loglik": avg_loglik,
                            "converged": bool(model.monitor_.converged),
                            "iterations": int(model.monitor_.iter),
                            "mean_hmm_entropy": float(annotated["hmm_entropy"].mean()) if not annotated.empty else np.nan,
                            "mean_hmm_max_prob": float(annotated["hmm_max_prob"].mean()) if not annotated.empty else np.nan,
                            "min_state_frequency": float(state_frequencies.min()) if not state_frequencies.empty else np.nan,
                            "max_state_frequency": float(state_frequencies.max()) if not state_frequencies.empty else np.nan,
                            "empty_state_count": int((state_frequencies < min_state_frequency).sum()) if not state_frequencies.empty else np.nan,
                            "mean_state_duration": float(occupancy["mean_duration"].mean()) if not occupancy.empty else np.nan,
                            "top_hour_pct": top_hour_pct,
                            "top_hour": top_hour,
                            "model_path": str(model_dir / "hmm.joblib"),
                        }
                    )

                    if save_posteriors and not annotated.empty:
                        posterior = annotated.copy()
                        posterior.insert(0, "split", split)
                        posterior.insert(0, "fold", int(fold.fold))
                        posterior.insert(0, "seed", int(seed))
                        posterior.insert(0, "n_states", int(n_states))
                        posterior.insert(0, "feature_set", feature_set["name"])
                        posterior_frames.append(posterior)

    return {
        "metrics": pd.DataFrame(metric_rows),
        "posteriors": pd.concat(posterior_frames, ignore_index=True) if posterior_frames else pd.DataFrame(),
        "state_occupancy": pd.concat(occupancy_frames, ignore_index=True) if occupancy_frames else pd.DataFrame(),
        "transitions": pd.DataFrame(transition_rows),
    }


def summarize_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ok = metrics[metrics["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    if "n_features" not in ok:
        ok["n_features"] = np.nan
    grouped = (
        ok.groupby(["feature_set", "n_features", "n_states", "seed", "split"], as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            rows=("rows", "sum"),
            weighted_loglik=("total_loglik", "sum"),
            avg_fold_loglik=("avg_loglik", "mean"),
            converged_folds=("converged", "sum"),
            avg_entropy=("mean_hmm_entropy", "mean"),
            avg_max_prob=("mean_hmm_max_prob", "mean"),
            avg_min_state_frequency=("min_state_frequency", "mean"),
            max_empty_state_count=("empty_state_count", "max"),
            avg_top_hour_pct=("top_hour_pct", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["weighted_avg_loglik"] = grouped["weighted_loglik"] / grouped["rows"]
    grouped["weighted_avg_loglik_per_feature"] = grouped["weighted_avg_loglik"] / grouped["n_features"]

    pivot = grouped.pivot_table(
        index=["feature_set", "n_features", "n_states", "seed"],
        columns="split",
        values=[
            "folds",
            "rows",
            "weighted_avg_loglik",
            "weighted_avg_loglik_per_feature",
            "avg_fold_loglik",
            "avg_entropy",
            "avg_min_state_frequency",
            "max_empty_state_count",
            "avg_top_hour_pct",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{split}_{metric}" for metric, split in pivot.columns]
    summary = pivot.reset_index()
    sort_columns = [
        column
        for column in [
            "validation_weighted_avg_loglik_per_feature",
            "validation_avg_min_state_frequency",
            "validation_avg_entropy",
        ]
        if column in summary
    ]
    ascending = [False, False, True][: len(sort_columns)]
    if sort_columns:
        summary = summary.sort_values(sort_columns, ascending=ascending, kind="stable").reset_index(drop=True)
        summary["validation_rank"] = np.arange(1, len(summary) + 1)
    return summary


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


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    feature_validations: pd.DataFrame,
    folds: list[LabFold],
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    lab_cfg = _lab_cfg(config)
    ready_count = int(feature_validations["status"].eq("ready").sum()) if not feature_validations.empty else 0
    failed = metrics[~metrics["status"].eq("ok")] if not metrics.empty and "status" in metrics else pd.DataFrame()
    top_cols = [
        "validation_rank",
        "feature_set",
        "n_features",
        "n_states",
        "seed",
        "validation_folds",
        "validation_weighted_avg_loglik_per_feature",
        "validation_weighted_avg_loglik",
        "validation_avg_min_state_frequency",
        "validation_max_empty_state_count",
        "validation_avg_entropy",
        "validation_avg_top_hour_pct",
        "test_weighted_avg_loglik",
    ]
    available_top_cols = [column for column in top_cols if column in summary.columns]
    fold_text = (
        f"{len(folds)} (`{folds[0].train_months[0]}` to `{folds[-1].test_months[-1]}`)"
        if folds
        else "0"
    )
    return f"""# HMM Feature Lab Cross-Asset - {target_symbol}

## Scope

- Feature version: `{lab_cfg.get("feature_set_version", "cross_asset_v1")}`
- Feature sets configured: `{len(feature_validations)}`
- Feature sets ready: `{ready_count}`
- K values: `{_configured_k_values(config)}`
- Seeds: `{_configured_seeds(config)}`
- Covariance: `{lab_cfg.get("covariance_type", "diag")}`
- HMM n_iter: `{lab_cfg.get("n_iter", 80)}`
- Walk-forward folds run: {fold_text}
- Ranking rule: validation diagnostics only; test is reported as holdout sanity.
- Log-likelihood is normalized by feature count for cross-set ordering because raw likelihood is dimension-dependent.

## Feature Set Validation

{_markdown_table(feature_validations)}

## Top Validation Diagnostics

{_markdown_table(summary.loc[:, available_top_cols] if not summary.empty else summary, max_rows=30)}

## Failed Fits

{_markdown_table(failed[["feature_set", "n_states", "seed", "fold", "split", "status", "error"]] if not failed.empty and "error" in failed else failed, max_rows=20)}

## Outputs

- Metrics: `{outputs["metrics"]}`
- Summary: `{outputs["summary"]}`
- Posteriors: `{outputs["posteriors"]}`
- State occupancy: `{outputs["state_occupancy"]}`
- Transitions: `{outputs["transitions"]}`
- Models: `{model_output_root(config, target_symbol)}`

## Notes

- The scaler is fit on train only per fold.
- HMM fit uses only train rows and session lengths.
- Validation/test inference uses causal forward-filtered probabilities and resets at each session.
- This block does not accept a trading edge; it only prepares candidates for state interpretability and stability work.
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    feature_config_path = Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml"))
    feature_config = load_yaml(feature_config_path)
    target = _target_symbol(config, target_symbol)

    features_path = features_input_path(config, target, feature_config)
    features = pd.read_parquet(features_path)
    ready_sets, feature_validations = resolve_feature_sets(config, feature_config, features)
    if not ready_sets:
        raise ValueError("No ready HMM feature sets; inspect missing_columns in feature validation")

    first_ready_frame = prepare_hmm_frame(features, ready_sets[0]["columns"])
    folds = build_lab_folds(first_ready_frame, config)
    if not folds:
        raise ValueError("No walk-forward folds could be built for HMM lab")

    output_frames = {"metrics": [], "posteriors": [], "state_occupancy": [], "transitions": []}
    for feature_set in ready_sets:
        outputs = run_feature_set_grid(features, feature_set, folds, config, target, feature_config)
        for key, frame in outputs.items():
            output_frames[key].append(frame)

    metrics = pd.concat(output_frames["metrics"], ignore_index=True) if output_frames["metrics"] else pd.DataFrame()
    posteriors = pd.concat(output_frames["posteriors"], ignore_index=True) if output_frames["posteriors"] else pd.DataFrame()
    state_occupancy = pd.concat(output_frames["state_occupancy"], ignore_index=True) if output_frames["state_occupancy"] else pd.DataFrame()
    transitions = pd.concat(output_frames["transitions"], ignore_index=True) if output_frames["transitions"] else pd.DataFrame()
    summary = summarize_grid(metrics)

    results_dir = results_output_dir(config, target)
    report_path = report_output_path(config, target)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "metrics": results_dir / "hmm_feature_lab_cross_asset.parquet",
        "summary": results_dir / "hmm_feature_lab_cross_asset_summary.parquet",
        "posteriors": results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet",
        "state_occupancy": results_dir / "hmm_feature_lab_cross_asset_state_occupancy.parquet",
        "transitions": results_dir / "hmm_feature_lab_cross_asset_transitions.parquet",
        "feature_sets": results_dir / "hmm_feature_lab_cross_asset_feature_sets.parquet",
    }

    metrics.to_parquet(output_paths["metrics"], index=False)
    summary.to_parquet(output_paths["summary"], index=False)
    posteriors.to_parquet(output_paths["posteriors"], index=False)
    state_occupancy.to_parquet(output_paths["state_occupancy"], index=False)
    transitions.to_parquet(output_paths["transitions"], index=False)
    feature_validations.to_parquet(output_paths["feature_sets"], index=False)
    report_path.write_text(render_report(config, target, feature_validations, folds, metrics, summary, output_paths), encoding="utf-8")
    return output_paths["metrics"], report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-asset HMM feature lab.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    metrics_path, report_path = run(args.config, args.target)
    print(f"HMM feature lab metrics written to: {metrics_path}")
    print(f"HMM feature lab report written to: {report_path}")


if __name__ == "__main__":
    main()

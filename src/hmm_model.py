from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.hmm_filter import add_hmm_probability_columns, filtered_probabilities


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _lengths_by_session(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("session", sort=False).size().astype(int).tolist()


def _split_sessions(frame: pd.DataFrame, train_fraction: float) -> tuple[list[str], list[str]]:
    sessions = frame["session"].drop_duplicates().tolist()
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions to create train/test HMM diagnostics split")

    n_train = int(np.floor(len(sessions) * train_fraction))
    n_train = min(max(n_train, 1), len(sessions) - 1)
    return sessions[:n_train], sessions[n_train:]


def prepare_hmm_frame(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required = {"timestamp", "session", "bar_index", "ret_1", "rv_12", *feature_columns}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Features data is missing required HMM columns: {missing}")

    hmm_frame = features.sort_values(["session", "bar_index"]).reset_index(drop=False).rename(columns={"index": "source_index"})
    valid_mask = hmm_frame[feature_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return hmm_frame.loc[valid_mask].reset_index(drop=True)


def fit_hmm_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    n_states: int,
    covariance_type: str,
    random_state: int,
    n_iter: int,
) -> tuple[GaussianHMM, StandardScaler]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_frame[feature_columns].to_numpy())
    lengths = _lengths_by_session(train_frame)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-4,
    )
    model.fit(train_x, lengths=lengths)
    return model, scaler


def filter_hmm_frame(model: GaussianHMM, scaler: StandardScaler, hmm_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    transformed = scaler.transform(hmm_frame[feature_columns].to_numpy())
    probabilities = filtered_probabilities(model, transformed, hmm_frame["session"])
    return add_hmm_probability_columns(hmm_frame, probabilities)


def _state_durations(states: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    runs: list[dict[str, int]] = []
    for session, group in pd.DataFrame({"state": states, "session": sessions}).groupby("session", sort=False):
        run_state = None
        run_length = 0
        for state in group["state"]:
            if run_state is None or state != run_state:
                if run_state is not None:
                    runs.append({"state": int(run_state), "duration": int(run_length)})
                run_state = state
                run_length = 1
            else:
                run_length += 1
        if run_state is not None:
            runs.append({"state": int(run_state), "duration": int(run_length)})

    if not runs:
        return pd.DataFrame(columns=["state", "mean_duration", "runs"])
    return (
        pd.DataFrame(runs)
        .groupby("state", as_index=False)
        .agg(mean_duration=("duration", "mean"), runs=("duration", "size"))
    )


def summarize_regimes(filtered: pd.DataFrame, n_states: int) -> pd.DataFrame:
    total = len(filtered)
    occupancy = filtered["hmm_state"].value_counts().reindex(range(n_states), fill_value=0).rename_axis("state").reset_index(name="count")
    occupancy["occupancy"] = occupancy["count"] / total if total else 0.0

    metrics = (
        filtered.groupby("hmm_state")
        .agg(mean_ret_1=("ret_1", "mean"), mean_rv_12=("rv_12", "mean"), mean_entropy=("hmm_entropy", "mean"))
        .rename_axis("state")
        .reset_index()
    )
    durations = _state_durations(filtered["hmm_state"], filtered["session"])

    summary = occupancy.merge(metrics, on="state", how="left").merge(durations, on="state", how="left")
    return summary.fillna({"mean_ret_1": 0.0, "mean_rv_12": 0.0, "mean_entropy": 0.0, "mean_duration": 0.0, "runs": 0})


def evaluate_candidates(
    hmm_frame: pd.DataFrame,
    feature_columns: list[str],
    train_sessions: list[str],
    test_sessions: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    hmm_cfg = config["hmm"]
    rows = []
    train_frame = hmm_frame[hmm_frame["session"].isin(train_sessions)].copy()
    test_frame = hmm_frame[hmm_frame["session"].isin(test_sessions)].copy()

    for n_states in hmm_cfg.get("candidate_states", [2, 3, 4, 5, 6]):
        for seed in hmm_cfg.get("stability_seeds", [hmm_cfg.get("random_state", 42)]):
            model, scaler = fit_hmm_model(
                train_frame,
                feature_columns,
                n_states=int(n_states),
                covariance_type=hmm_cfg.get("covariance_type", "diag"),
                random_state=int(seed),
                n_iter=int(hmm_cfg.get("diagnostic_n_iter", hmm_cfg.get("n_iter", 100))),
            )
            train_x = scaler.transform(train_frame[feature_columns].to_numpy())
            test_x = scaler.transform(test_frame[feature_columns].to_numpy())
            rows.append(
                {
                    "n_states": int(n_states),
                    "seed": int(seed),
                    "train_avg_loglik": float(model.score(train_x, lengths=_lengths_by_session(train_frame)) / len(train_x)),
                    "test_avg_loglik": float(model.score(test_x, lengths=_lengths_by_session(test_frame)) / len(test_x)),
                    "converged": bool(model.monitor_.converged),
                    "iterations": int(model.monitor_.iter),
                }
            )
    return pd.DataFrame(rows)


def render_regime_report(
    regime_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    transition_matrix: np.ndarray,
    config: dict[str, Any],
    train_sessions: list[str],
    test_sessions: list[str],
    filtered_rows: int,
) -> str:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda value: f"{value:.6f}")
        headers = display.columns.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    transition = pd.DataFrame(
        transition_matrix,
        columns=[f"to_{idx}" for idx in range(transition_matrix.shape[1])],
    )
    transition.insert(0, "from_state", range(transition_matrix.shape[0]))

    hmm_cfg = config["hmm"]
    return f"""# Regime Diagnostics

## Scope

- Input features: `{config["data"]["features_file"]}`
- Output features: `{config["data"]["hmm_features_file"]}`
- Filtered rows: {filtered_rows}
- HMM columns: `{", ".join(hmm_cfg["feature_columns"])}`
- Main model: K={hmm_cfg["n_states"]}, covariance `{hmm_cfg["covariance_type"]}`, seed {hmm_cfg["random_state"]}
- Train sessions: {len(train_sessions)} (`{train_sessions[0]}` to `{train_sessions[-1]}`)
- Test sessions: {len(test_sessions)} (`{test_sessions[0]}` to `{test_sessions[-1]}`)

## State Diagnostics

{markdown_table(regime_summary)}

## Transition Matrix

{markdown_table(transition)}

## Stability

{markdown_table(candidate_summary)}

## Notes

- Scaling is fit only on train sessions.
- HMM training uses session `lengths`.
- Probabilities are causal forward-filtered and reset at each session.
- The report uses a chronological train/test split for diagnostics; walk-forward validation is implemented later.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    hmm_cfg = config["hmm"]
    feature_columns = list(hmm_cfg["feature_columns"])

    features_path = Path(config["data"]["features_file"])
    output_path = Path(config["data"]["hmm_features_file"])
    report_path = Path(config["paths"]["reports"]) / "regime_diagnostics.md"
    model_path = Path(hmm_cfg.get("model_file", "models/hmm_k4.joblib"))

    features = pd.read_parquet(features_path)
    hmm_frame = prepare_hmm_frame(features, feature_columns)
    train_sessions, test_sessions = _split_sessions(hmm_frame, float(hmm_cfg.get("train_fraction", 0.7)))
    train_frame = hmm_frame[hmm_frame["session"].isin(train_sessions)].copy()

    model, scaler = fit_hmm_model(
        train_frame,
        feature_columns,
        n_states=int(hmm_cfg["n_states"]),
        covariance_type=hmm_cfg.get("covariance_type", "diag"),
        random_state=int(hmm_cfg.get("random_state", 42)),
        n_iter=int(hmm_cfg.get("n_iter", 200)),
    )
    filtered = filter_hmm_frame(model, scaler, hmm_frame, feature_columns)

    hmm_output = features.copy()
    n_states = int(hmm_cfg["n_states"])
    hmm_columns = [f"hmm_p{idx}" for idx in range(n_states)] + ["hmm_state", "hmm_entropy", "hmm_max_prob"]
    for col in hmm_columns:
        hmm_output[col] = np.nan
    hmm_output.loc[filtered["source_index"].to_numpy(), hmm_columns] = filtered[hmm_columns].to_numpy()

    regime_summary = summarize_regimes(filtered, n_states)
    candidate_summary = evaluate_candidates(hmm_frame, feature_columns, train_sessions, test_sessions, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    hmm_output.to_parquet(output_path, index=False)
    report_path.write_text(
        render_regime_report(regime_summary, candidate_summary, model.transmat_, config, train_sessions, test_sessions, len(filtered)),
        encoding="utf-8",
    )
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": feature_columns, "train_sessions": train_sessions}, model_path)
    return output_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HMM regimes and generate causal filtered probabilities.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    output_path, report_path = run(args.config)
    print(f"HMM features written to: {output_path}")
    print(f"Regime diagnostics written to: {report_path}")


if __name__ == "__main__":
    main()

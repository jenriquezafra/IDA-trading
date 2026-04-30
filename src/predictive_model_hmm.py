from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.predictive_model import (
    get_base_feature_columns,
    load_config,
    prepare_model_frame,
    train_predictive_pipeline,
)


def merge_labels_with_hmm(labels: pd.DataFrame, hmm_features: pd.DataFrame, n_states: int) -> pd.DataFrame:
    hmm_columns = ["timestamp", "session", "bar_index", *[f"hmm_p{i}" for i in range(n_states)], "hmm_state", "hmm_entropy", "hmm_max_prob"]
    missing = sorted(set(hmm_columns) - set(hmm_features.columns))
    if missing:
        raise ValueError(f"HMM features are missing required columns: {missing}")

    merged = labels.merge(
        hmm_features.loc[:, hmm_columns],
        on=["timestamp", "session", "bar_index"],
        how="left",
        validate="one_to_one",
    )
    for state in range(n_states):
        merged[f"hmm_state_{state}"] = (merged["hmm_state"] == state).astype(int)
    return merged


def get_hmm_feature_columns(config: dict[str, Any]) -> list[str]:
    return get_base_feature_columns(config) + list(config["model"]["hmm_feature_columns"])


def compare_with_base(hmm_metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    base_metadata_path = Path(config["paths"]["models"]) / "predictive_base" / "fold_0" / "metadata.joblib"
    if not base_metadata_path.exists():
        return pd.DataFrame()

    base_metadata = joblib.load(base_metadata_path)
    base_metrics = base_metadata["metrics"].copy()
    base_test = base_metrics[base_metrics["split"] == "test_calibrated"].iloc[0]
    hmm_test = hmm_metrics[hmm_metrics["split"] == "test_calibrated"].iloc[0]

    rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
        base_value = float(base_test[metric])
        hmm_value = float(hmm_test[metric])
        delta = hmm_value - base_value
        if metric == "log_loss":
            improved = delta < 0
        else:
            improved = delta > 0
        rows.append(
            {
                "metric": metric,
                "base": base_value,
                "with_hmm": hmm_value,
                "delta": delta,
                "improved": bool(improved),
            }
        )
    return pd.DataFrame(rows)


def add_net_return_comparison(predictions_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    base_predictions_path = Path(config["model"]["base_predictions_file"])
    if not base_predictions_path.exists() or not predictions_path.exists():
        return pd.DataFrame()

    threshold = 0.0
    cost = float(config["labeling"]["round_trip_cost_bps"]) / 10_000.0

    rows = []
    for name, path in [("base", base_predictions_path), ("with_hmm", predictions_path)]:
        predictions = pd.read_parquet(path)
        test = predictions[predictions["split"] == "test"].copy()
        position = np.sign(test["score"]).astype(int)
        position = position.where(test["score"].abs() > threshold, 0)
        net_ret = position * test["fwd_ret"] - position.abs() * cost
        rows.append({"model": name, "test_rule_net_return": float(net_ret.sum()), "test_rule_trades": int((position != 0).sum())})
    return pd.DataFrame(rows)


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    n_states = int(config["hmm"]["n_states"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    hmm_features = pd.read_parquet(config["data"]["hmm_features_file"])
    merged = merge_labels_with_hmm(labels, hmm_features, n_states)
    feature_columns = get_hmm_feature_columns(config)
    frame = prepare_model_frame(merged, feature_columns)

    predictions_path = Path(config["model"]["hmm_predictions_file"])
    report_path = Path(config["model"]["hmm_report_file"])
    artifact_dir = Path(config["paths"]["models"]) / "predictive_hmm" / "fold_0"

    _, metrics, _ = train_predictive_pipeline(
        frame,
        feature_columns,
        config,
        predictions_path,
        report_path,
        artifact_dir,
        title="Predictive Model With HMM Report",
        notes=[
            "Base engineered features plus HMM probabilities, entropy, max probability, and one-hot state are included.",
            "The scaler is fit only on train sessions.",
            "Calibration is fit only on validation sessions.",
            "Test metrics are reported once after calibration.",
        ],
    )

    comparison = compare_with_base(metrics, config)
    net_comparison = add_net_return_comparison(predictions_path, config)
    report_text = report_path.read_text(encoding="utf-8")
    if not comparison.empty:
        report_text += "\n## OOS Metric Comparison vs Base\n\n"
        report_text += _markdown_table(comparison)
        report_text += "\n"
    if not net_comparison.empty:
        report_text += "\n## Simple Test Score Rule Net Return\n\n"
        report_text += _markdown_table(net_comparison)
        report_text += "\n"
    report_path.write_text(report_text, encoding="utf-8")

    return predictions_path, report_path


def _markdown_table(frame: pd.DataFrame) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive Logistic Regression model with HMM features.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    predictions_path, report_path = run(args.config)
    print(f"Predictive HMM probabilities written to: {predictions_path}")
    print(f"Predictive HMM report written to: {report_path}")


if __name__ == "__main__":
    main()

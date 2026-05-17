from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from src.predictive_model import (
    CLASS_ORDER,
    evaluate_probabilities,
    get_base_feature_columns,
    load_config,
    prepare_model_frame,
    split_sessions,
)
from src.signal import apply_selected_signal, build_signal_frame, evaluate_signal, select_thresholds_on_validation


TARGET_TO_XGB = {-1: 0, 0: 1, 1: 2}
XGB_TO_TARGET = np.array([-1, 0, 1])


def encode_target(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    encoded = series.map(TARGET_TO_XGB)
    if encoded.isna().any():
        bad = sorted(series.loc[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Unsupported target labels for XGBoost: {bad}")
    return encoded.to_numpy(dtype=int)


def fit_xgboost_model(train: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> XGBClassifier:
    xgb_cfg = config["model"].get("xgboost", {})
    model = XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 250)),
        max_depth=int(xgb_cfg.get("max_depth", 3)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 25.0)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        reg_alpha=float(xgb_cfg.get("reg_alpha", 0.1)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 5.0)),
        objective=xgb_cfg.get("objective", "multi:softprob"),
        eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
        tree_method=xgb_cfg.get("tree_method", "hist"),
        n_jobs=int(xgb_cfg.get("n_jobs", 4)),
        random_state=int(config["model"].get("random_state", 42)),
        num_class=len(CLASS_ORDER),
    )
    model.fit(train[feature_columns].to_numpy(), encode_target(train["target"]))
    return model


def calibrate_xgboost_model(
    model: XGBClassifier,
    validation: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> CalibratedClassifierCV:
    method = config["model"].get("calibration", {}).get("method", "sigmoid")
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method, ensemble=False)
    calibrator.fit(validation[feature_columns].to_numpy(), encode_target(validation["target"]))
    return calibrator


def _align_xgboost_probabilities(classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(CLASS_ORDER)))
    for source_idx, encoded_class in enumerate(classes.astype(int)):
        target_label = XGB_TO_TARGET[encoded_class]
        target_idx = np.where(CLASS_ORDER == target_label)[0][0]
        aligned[:, target_idx] = probabilities[:, source_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / len(CLASS_ORDER)), where=row_sums > 0)
    return aligned


def predict_xgboost_probabilities(estimator, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    raw_probabilities = estimator.predict_proba(frame[feature_columns].to_numpy())
    probabilities = _align_xgboost_probabilities(estimator.classes_, raw_probabilities)

    output = frame.loc[:, ["timestamp", "session", "bar_index", "target", "fwd_ret", "neutral_zone"]].copy()
    output["p_down"] = probabilities[:, 0]
    output["p_neutral"] = probabilities[:, 1]
    output["p_up"] = probabilities[:, 2]
    output["predicted_class"] = CLASS_ORDER[probabilities.argmax(axis=1)]
    output["score"] = output["p_up"] - output["p_down"]
    return output


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def compare_against_logistic(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    metadata_path = Path(config["paths"]["models"]) / "predictive_base" / "fold_0" / "metadata.joblib"
    if not metadata_path.exists():
        return pd.DataFrame()

    base_metrics = joblib.load(metadata_path)["metrics"]
    base_test = base_metrics[base_metrics["split"] == "test_calibrated"].iloc[0]
    xgb_test = metrics[metrics["split"] == "test_calibrated"].iloc[0]

    rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
        base_value = float(base_test[metric])
        xgb_value = float(xgb_test[metric])
        delta = xgb_value - base_value
        rows.append(
            {
                "metric": metric,
                "logistic_regression": base_value,
                "xgboost": xgb_value,
                "delta": delta,
                "improved": bool(delta < 0 if metric == "log_loss" else delta > 0),
            }
        )
    return pd.DataFrame(rows)


def _signal_metrics(predictions: pd.DataFrame, selected: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    frame = build_signal_frame(predictions)
    signals = apply_selected_signal(frame, selected, config)
    rows = []
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    for split, group in signals.groupby("split", sort=False):
        metrics = evaluate_signal(group, group["signal"], cost_bps)
        sessions = max(int(group["session"].nunique()), 1)
        rows.append({"split": split, **metrics, "turnover_trades_per_day": metrics["trades"] / sessions})
    return pd.DataFrame(rows)


def render_report(
    metrics: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    selected: dict[str, float],
    grid: pd.DataFrame,
    comparison: pd.DataFrame,
    split_map: dict[str, list[str]],
    feature_columns: list[str],
    config: dict[str, Any],
) -> str:
    xgb_cfg = config["model"].get("xgboost", {})
    return f"""# Predictive XGBoost Report

## Scope

- Input labels: `{config["data"]["labels_file"]}`
- Model: XGBoost multiclass challenger
- Calibration: `{config["model"].get("calibration", {}).get("method", "sigmoid")}` on validation split
- Train sessions: {len(split_map["train"])} (`{split_map["train"][0]}` to `{split_map["train"][-1]}`)
- Validation sessions: {len(split_map["validation"])} (`{split_map["validation"][0]}` to `{split_map["validation"][-1]}`)
- Test sessions: {len(split_map["test"])} (`{split_map["test"][0]}` to `{split_map["test"][-1]}`)

## Regularization

- max_depth: `{xgb_cfg.get("max_depth", 3)}`
- n_estimators: `{xgb_cfg.get("n_estimators", 250)}`
- learning_rate: `{xgb_cfg.get("learning_rate", 0.03)}`
- min_child_weight: `{xgb_cfg.get("min_child_weight", 25.0)}`
- subsample: `{xgb_cfg.get("subsample", 0.8)}`
- colsample_bytree: `{xgb_cfg.get("colsample_bytree", 0.8)}`
- reg_alpha: `{xgb_cfg.get("reg_alpha", 0.1)}`
- reg_lambda: `{xgb_cfg.get("reg_lambda", 5.0)}`

## Probability Metrics

{_markdown_table(metrics)}

## Logistic Regression Comparison

{_markdown_table(comparison)}

## Selected Signal Thresholds

- theta_prob: {selected["theta_prob"]:.4f}
- theta_score: {selected["theta_score"]:.4f}
- max_neutral: {selected["max_neutral"]:.4f}
- max_hmm_entropy: {selected["max_hmm_entropy"]:.4f}

## Signal Metrics

{_markdown_table(signal_metrics)}

## Validation Grid Top 10

{_markdown_table(grid.head(10))}

## Feature Columns

{chr(10).join(f"- `{col}`" for col in feature_columns)}

## Decision Guardrails

- Treat XGBoost as useful only if it improves validation and test, not train only.
- Probability calibration and threshold selection are fit only on validation rows.
- The configured model limits depth and uses shrinkage, subsampling, column sampling, L1, and L2 regularization.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    labels = pd.read_parquet(config["data"]["labels_file"])
    feature_columns = get_base_feature_columns(config)
    frame = prepare_model_frame(labels, feature_columns)
    model_cfg = config["model"]
    split_map = split_sessions(
        frame,
        train_fraction=float(model_cfg.get("train_fraction", 0.6)),
        validation_fraction=float(model_cfg.get("validation_fraction", 0.2)),
    )

    train = frame[frame["session"].isin(split_map["train"])].copy()
    validation = frame[frame["session"].isin(split_map["validation"])].copy()
    test = frame[frame["session"].isin(split_map["test"])].copy()

    model = fit_xgboost_model(train, feature_columns, config)
    calibrator = calibrate_xgboost_model(model, validation, feature_columns, config)
    predictions = pd.concat(
        [
            predict_xgboost_probabilities(model, train, feature_columns).assign(split="train", calibrated=False),
            predict_xgboost_probabilities(calibrator, validation, feature_columns).assign(split="validation", calibrated=True),
            predict_xgboost_probabilities(calibrator, test, feature_columns).assign(split="test", calibrated=True),
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        [
            evaluate_probabilities(predictions[predictions["split"] == "train"], "train_uncalibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "validation"], "validation_calibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "test"], "test_calibrated"),
        ]
    )
    signal_frame = build_signal_frame(predictions)
    selected, grid = select_thresholds_on_validation(signal_frame, config)
    signal_metrics = _signal_metrics(signal_frame, selected, config)
    comparison = compare_against_logistic(metrics, config)

    predictions_path = Path(model_cfg["xgboost_predictions_file"])
    report_path = Path(model_cfg["xgboost_report_file"])
    artifact_dir = Path(config["paths"]["models"]) / "predictive_xgboost" / "fold_0"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(predictions_path, index=False)
    report_path.write_text(
        render_report(metrics, signal_metrics, selected, grid, comparison, split_map, feature_columns, config),
        encoding="utf-8",
    )
    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
    joblib.dump({"feature_columns": feature_columns, "split_map": split_map, "metrics": metrics}, artifact_dir / "metadata.joblib")
    return predictions_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost challenger without HMM features.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    predictions_path, report_path = run(args.config)
    print(f"XGBoost probabilities written to: {predictions_path}")
    print(f"XGBoost report written to: {report_path}")


if __name__ == "__main__":
    main()

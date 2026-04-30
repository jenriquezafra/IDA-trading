from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


CLASS_ORDER = np.array([-1, 0, 1])


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_base_feature_columns(config: dict[str, Any]) -> list[str]:
    return list(config["model"]["base_feature_columns"])


def prepare_model_frame(labels: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required = {"timestamp", "session", "bar_index", "target", *feature_columns}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels data is missing required model columns: {missing}")

    frame = labels.sort_values(["session", "bar_index"]).reset_index(drop=True).copy()
    finite = frame[feature_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return frame.loc[finite].reset_index(drop=True)


def split_sessions(frame: pd.DataFrame, train_fraction: float, validation_fraction: float) -> dict[str, list[str]]:
    sessions = frame["session"].drop_duplicates().tolist()
    if len(sessions) < 3:
        raise ValueError("Need at least three sessions for train/validation/test split")

    n_sessions = len(sessions)
    n_train = int(np.floor(n_sessions * train_fraction))
    n_validation = int(np.floor(n_sessions * validation_fraction))

    n_train = max(1, min(n_train, n_sessions - 2))
    n_validation = max(1, min(n_validation, n_sessions - n_train - 1))

    return {
        "train": sessions[:n_train],
        "validation": sessions[n_train : n_train + n_validation],
        "test": sessions[n_train + n_validation :],
    }


def fit_base_model(train: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> tuple[StandardScaler, LogisticRegression]:
    model_cfg = config["model"]["logistic_regression"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns].to_numpy())
    y_train = train["target"].to_numpy()

    model = LogisticRegression(
        solver=model_cfg.get("solver", "saga"),
        l1_ratio=float(model_cfg.get("l1_ratio", 0.2)),
        C=float(model_cfg.get("C", 1.0)),
        max_iter=int(model_cfg.get("max_iter", 5000)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        random_state=int(config["model"].get("random_state", 42)),
    )
    model.fit(x_train, y_train)
    return scaler, model


def calibrate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    validation: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> CalibratedClassifierCV:
    x_validation = scaler.transform(validation[feature_columns].to_numpy())
    y_validation = validation["target"].to_numpy()
    method = config["model"].get("calibration", {}).get("method", "sigmoid")
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method, ensemble=False)
    calibrator.fit(x_validation, y_validation)
    return calibrator


def _align_probabilities(classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(CLASS_ORDER)))
    for source_idx, klass in enumerate(classes):
        target_idx = np.where(CLASS_ORDER == klass)[0][0]
        aligned[:, target_idx] = probabilities[:, source_idx]
    return aligned


def predict_probabilities(estimator, scaler: StandardScaler, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = scaler.transform(frame[feature_columns].to_numpy())
    raw_probabilities = estimator.predict_proba(x)
    probabilities = _align_probabilities(estimator.classes_, raw_probabilities)

    output = frame.loc[:, ["timestamp", "session", "bar_index", "target", "fwd_ret", "neutral_zone"]].copy()
    output["p_down"] = probabilities[:, 0]
    output["p_neutral"] = probabilities[:, 1]
    output["p_up"] = probabilities[:, 2]
    output["predicted_class"] = CLASS_ORDER[probabilities.argmax(axis=1)]
    output["score"] = output["p_up"] - output["p_down"]
    return output


def evaluate_probabilities(predictions: pd.DataFrame, split_name: str) -> dict[str, float | int | str]:
    y_true = predictions["target"].to_numpy()
    proba = predictions[["p_down", "p_neutral", "p_up"]].to_numpy()
    y_pred = predictions["predicted_class"].to_numpy()

    return {
        "split": split_name,
        "rows": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_ORDER)),
        "avg_p_down": float(predictions["p_down"].mean()),
        "avg_p_neutral": float(predictions["p_neutral"].mean()),
        "avg_p_up": float(predictions["p_up"].mean()),
    }


def render_report(
    metrics: pd.DataFrame,
    split_map: dict[str, list[str]],
    config: dict[str, Any],
    feature_columns: list[str],
    title: str = "Predictive Base Model Report",
    notes: list[str] | None = None,
    comparison: pd.DataFrame | None = None,
) -> str:
    display = metrics.copy()
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

    notes = notes or [
        "HMM features are excluded.",
        "The scaler is fit only on train sessions.",
        "Calibration is fit only on validation sessions.",
        "Test metrics are reported once after calibration.",
    ]

    comparison_text = ""
    if comparison is not None and not comparison.empty:
        comparison_display = comparison.copy()
        for col in comparison_display.columns:
            if pd.api.types.is_float_dtype(comparison_display[col]):
                comparison_display[col] = comparison_display[col].map(lambda value: f"{value:.6f}")
        comparison_headers = comparison_display.columns.tolist()
        comparison_lines = [
            "| " + " | ".join(comparison_headers) + " |",
            "| " + " | ".join(["---"] * len(comparison_headers)) + " |",
        ]
        for _, row in comparison_display.iterrows():
            comparison_lines.append("| " + " | ".join(str(row[col]) for col in comparison_headers) + " |")
        comparison_text = "\n## Comparison\n\n" + "\n".join(comparison_lines) + "\n"

    return f"""# {title}

## Scope

- Input labels: `{config["data"]["labels_file"]}`
- Model: multinomial Logistic Regression with elastic-net regularization
- Calibration: `{config["model"].get("calibration", {}).get("method", "sigmoid")}` on validation split
- Train sessions: {len(split_map["train"])} (`{split_map["train"][0]}` to `{split_map["train"][-1]}`)
- Validation sessions: {len(split_map["validation"])} (`{split_map["validation"][0]}` to `{split_map["validation"][-1]}`)
- Test sessions: {len(split_map["test"])} (`{split_map["test"][0]}` to `{split_map["test"][-1]}`)

## Metrics

{chr(10).join(lines)}
{comparison_text}

## Feature Columns

{chr(10).join(f"- `{col}`" for col in feature_columns)}

## Notes

{chr(10).join(f"- {note}" for note in notes)}
"""


def train_predictive_pipeline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    predictions_path: Path,
    report_path: Path,
    artifact_dir: Path,
    title: str = "Predictive Base Model Report",
    notes: list[str] | None = None,
    comparison: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    model_cfg = config["model"]
    split_map = split_sessions(
        frame,
        train_fraction=float(model_cfg.get("train_fraction", 0.6)),
        validation_fraction=float(model_cfg.get("validation_fraction", 0.2)),
    )

    train = frame[frame["session"].isin(split_map["train"])].copy()
    validation = frame[frame["session"].isin(split_map["validation"])].copy()
    test = frame[frame["session"].isin(split_map["test"])].copy()

    scaler, model = fit_base_model(train, feature_columns, config)
    calibrator = calibrate_model(model, scaler, validation, feature_columns, config)

    predictions = pd.concat(
        [
            predict_probabilities(model, scaler, train, feature_columns).assign(split="train", calibrated=False),
            predict_probabilities(calibrator, scaler, validation, feature_columns).assign(split="validation", calibrated=True),
            predict_probabilities(calibrator, scaler, test, feature_columns).assign(split="test", calibrated=True),
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

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(predictions_path, index=False)
    report_path.write_text(render_report(metrics, split_map, config, feature_columns, title=title, notes=notes, comparison=comparison), encoding="utf-8")
    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
    joblib.dump({"feature_columns": feature_columns, "split_map": split_map, "metrics": metrics}, artifact_dir / "metadata.joblib")
    return predictions, metrics, split_map


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    labels = pd.read_parquet(config["data"]["labels_file"])
    feature_columns = get_base_feature_columns(config)
    frame = prepare_model_frame(labels, feature_columns)

    model_cfg = config["model"]
    predictions_path = Path(model_cfg["base_predictions_file"])
    report_path = Path(model_cfg["base_report_file"])
    artifact_dir = Path(config["paths"]["models"]) / "predictive_base" / "fold_0"

    train_predictive_pipeline(
        frame,
        feature_columns,
        config,
        predictions_path,
        report_path,
        artifact_dir,
        title="Predictive Base Model Report",
    )
    return predictions_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline predictive Logistic Regression model without HMM features.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    predictions_path, report_path = run(args.config)
    print(f"Predictive probabilities written to: {predictions_path}")
    print(f"Predictive report written to: {report_path}")


if __name__ == "__main__":
    main()

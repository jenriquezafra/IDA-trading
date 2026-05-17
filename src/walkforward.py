from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.hmm_model import filter_hmm_frame, fit_hmm_model, prepare_hmm_frame
from src.predictive_model import (
    evaluate_probabilities,
    fit_base_model,
    calibrate_model,
    get_base_feature_columns,
    predict_probabilities,
    prepare_model_frame,
)
from src.predictive_xgboost import calibrate_xgboost_model, fit_xgboost_model, predict_xgboost_probabilities
from src.signal import apply_selected_signal, build_signal_frame, select_thresholds_on_validation


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_months: list[str]
    validation_months: list[str]
    test_months: list[str]
    train_sessions: list[str]
    validation_sessions: list[str]
    test_sessions: list[str]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _session_months(frame: pd.DataFrame) -> pd.DataFrame:
    sessions = frame[["session"]].drop_duplicates().copy()
    sessions["session_date"] = pd.to_datetime(sessions["session"])
    sessions["month"] = sessions["session_date"].dt.to_period("M").astype(str)
    return sessions


def build_monthly_folds(frame: pd.DataFrame, config: dict[str, Any]) -> list[WalkForwardFold]:
    wf_cfg = config["walkforward"]
    train_months = int(wf_cfg["train_months"])
    validation_months = int(wf_cfg["validation_months"])
    test_months = int(wf_cfg["test_months"])
    step_months = int(wf_cfg["step_months"])

    session_months = _session_months(frame)
    months = sorted(session_months["month"].unique().tolist())
    window = train_months + validation_months + test_months
    folds: list[WalkForwardFold] = []

    fold_id = 0
    for start in range(0, max(len(months) - window + 1, 0), step_months):
        train = months[start : start + train_months]
        validation = months[start + train_months : start + train_months + validation_months]
        test = months[start + train_months + validation_months : start + window]
        if len(train) != train_months or len(validation) != validation_months or len(test) != test_months:
            continue
        folds.append(
            WalkForwardFold(
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
    return folds


def apply_purge_and_embargo(labels: pd.DataFrame, fold: WalkForwardFold, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    wf_cfg = config["walkforward"]
    purge_bars = int(wf_cfg.get("purge_bars", config.get("labeling", {}).get("horizon_bars", 0)))
    embargo_bars = int(wf_cfg.get("embargo_bars", 0))

    train = labels[labels["session"].isin(fold.train_sessions)].copy()
    validation = labels[labels["session"].isin(fold.validation_sessions)].copy()
    test = labels[labels["session"].isin(fold.test_sessions)].copy()

    if purge_bars > 0 and not train.empty:
        session_size = train.groupby("session")["bar_index"].transform("size")
        session_pos = train.groupby("session").cumcount()
        train = train.loc[session_pos < session_size - purge_bars].copy()
    if embargo_bars > 0 and not validation.empty:
        session_pos = validation.groupby("session").cumcount()
        validation = validation.loc[session_pos >= embargo_bars].copy()

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _add_hmm_one_hot(frame: pd.DataFrame, n_states: int) -> pd.DataFrame:
    output = frame.copy()
    for state in range(n_states):
        output[f"hmm_state_{state}"] = (output["hmm_state"] == state).astype(int)
    return output


def _fold_hmm_features(features: pd.DataFrame, fold: WalkForwardFold, config: dict[str, Any]) -> pd.DataFrame:
    hmm_cfg = config["hmm"]
    feature_columns = list(hmm_cfg["feature_columns"])
    hmm_frame = prepare_hmm_frame(features, feature_columns)
    train_frame = hmm_frame[hmm_frame["session"].isin(fold.train_sessions)].copy()
    fold_frame = hmm_frame[hmm_frame["session"].isin(fold.train_sessions + fold.validation_sessions + fold.test_sessions)].copy()

    model, scaler = fit_hmm_model(
        train_frame,
        feature_columns,
        n_states=int(hmm_cfg["n_states"]),
        covariance_type=hmm_cfg.get("covariance_type", "diag"),
        random_state=int(hmm_cfg.get("random_state", 42)),
        n_iter=int(hmm_cfg.get("n_iter", 200)),
    )
    filtered = filter_hmm_frame(model, scaler, fold_frame, feature_columns)
    return filtered, model, scaler


def _merge_labels_with_fold_hmm(labels: pd.DataFrame, filtered_hmm: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    n_states = int(config["hmm"]["n_states"])
    hmm_columns = ["timestamp", "session", "bar_index", *[f"hmm_p{idx}" for idx in range(n_states)], "hmm_state", "hmm_entropy", "hmm_max_prob"]
    merged = labels.merge(filtered_hmm[hmm_columns], on=["timestamp", "session", "bar_index"], how="left", validate="one_to_one")
    return _add_hmm_one_hot(merged, n_states)


def _net_return_from_predictions(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, float | int]:
    cost = float(config["labeling"]["round_trip_cost_bps"]) / 10_000.0
    position = predictions["signal"].astype(int)
    net_ret = position * predictions["fwd_ret"] - position.abs() * cost
    active = position != 0
    return {
        "trades": int(active.sum()),
        "net_return": float(net_ret.sum()),
        "avg_trade_net": float(net_ret[active].mean()) if active.any() else 0.0,
        "hit_ratio": float((net_ret[active] > 0).mean()) if active.any() else np.nan,
    }


def _run_xgboost_challenger(
    split_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    fold_dir: Path,
) -> dict[str, Any]:
    feature_columns = get_base_feature_columns(config)
    model_frame = pd.concat(
        [
            split_frames["train"].assign(split="train"),
            split_frames["validation"].assign(split="validation"),
            split_frames["test"].assign(split="test"),
        ],
        ignore_index=True,
    )
    model_frame = prepare_model_frame(model_frame, feature_columns)
    train = model_frame[model_frame["split"] == "train"].copy()
    validation = model_frame[model_frame["split"] == "validation"].copy()
    test = model_frame[model_frame["split"] == "test"].copy()

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
    signal_frame = build_signal_frame(predictions)
    selected, grid = select_thresholds_on_validation(signal_frame, config)
    signals = apply_selected_signal(signal_frame, selected, config)
    metrics = pd.DataFrame(
        [
            evaluate_probabilities(predictions[predictions["split"] == "train"], "train_uncalibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "validation"], "validation_calibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "test"], "test_calibrated"),
        ]
    )
    test_signal_metrics = _net_return_from_predictions(signals[signals["split"] == "test"], config)

    predictions.to_parquet(fold_dir / "xgboost_predictions.parquet", index=False)
    signals.to_parquet(fold_dir / "xgboost_signals.parquet", index=False)
    grid.to_parquet(fold_dir / "xgboost_threshold_grid.parquet", index=False)
    metrics.to_parquet(fold_dir / "xgboost_metrics.parquet", index=False)
    joblib.dump(model, fold_dir / "xgboost_model.joblib")
    joblib.dump(calibrator, fold_dir / "xgboost_calibrator.joblib")

    test_metrics = metrics[metrics["split"] == "test_calibrated"].iloc[0].to_dict()
    return {
        "xgboost_theta_prob": selected["theta_prob"],
        "xgboost_theta_score": selected["theta_score"],
        "xgboost_max_neutral": selected["max_neutral"],
        "xgboost_max_hmm_entropy": selected["max_hmm_entropy"],
        "xgboost_test_accuracy": float(test_metrics["accuracy"]),
        "xgboost_test_log_loss": float(test_metrics["log_loss"]),
        **{f"xgboost_test_signal_{key}": value for key, value in test_signal_metrics.items()},
    }


def run_fold(features: pd.DataFrame, labels: pd.DataFrame, fold: WalkForwardFold, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    filtered_hmm, hmm_model, hmm_scaler = _fold_hmm_features(features, fold, config)
    fold_labels = labels[labels["session"].isin(fold.train_sessions + fold.validation_sessions + fold.test_sessions)].copy()
    fold_frame = _merge_labels_with_fold_hmm(fold_labels, filtered_hmm, config)

    split_frames = apply_purge_and_embargo(fold_frame, fold, config)
    model_frame = pd.concat(
        [
            split_frames["train"].assign(split="train"),
            split_frames["validation"].assign(split="validation"),
            split_frames["test"].assign(split="test"),
        ],
        ignore_index=True,
    )

    feature_columns = get_base_feature_columns(config) + list(config["model"]["hmm_feature_columns"])
    model_frame = prepare_model_frame(model_frame, feature_columns)
    train = model_frame[model_frame["split"] == "train"].copy()
    validation = model_frame[model_frame["split"] == "validation"].copy()
    test = model_frame[model_frame["split"] == "test"].copy()

    model_scaler, model = fit_base_model(train, feature_columns, config)
    calibrator = calibrate_model(model, model_scaler, validation, feature_columns, config)

    predictions = pd.concat(
        [
            predict_probabilities(model, model_scaler, train, feature_columns).assign(split="train", calibrated=False),
            predict_probabilities(calibrator, model_scaler, validation, feature_columns).assign(split="validation", calibrated=True),
            predict_probabilities(calibrator, model_scaler, test, feature_columns).assign(split="test", calibrated=True),
        ],
        ignore_index=True,
    )
    predictions = predictions.merge(
        model_frame[["timestamp", "session", "bar_index", "hmm_state", "hmm_entropy", "hmm_max_prob"]],
        on=["timestamp", "session", "bar_index"],
        how="left",
    )

    selected, grid = select_thresholds_on_validation(predictions, config)
    signals = apply_selected_signal(predictions, selected, config)

    metrics = pd.DataFrame(
        [
            evaluate_probabilities(predictions[predictions["split"] == "train"], "train_uncalibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "validation"], "validation_calibrated"),
            evaluate_probabilities(predictions[predictions["split"] == "test"], "test_calibrated"),
        ]
    )
    test_signal_metrics = _net_return_from_predictions(signals[signals["split"] == "test"], config)

    fold_dir = output_dir / f"fold_{fold.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    xgboost_metrics = _run_xgboost_challenger(split_frames, config, fold_dir)
    predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
    signals.to_parquet(fold_dir / "signals.parquet", index=False)
    grid.to_parquet(fold_dir / "threshold_grid.parquet", index=False)
    metrics.to_parquet(fold_dir / "metrics.parquet", index=False)
    joblib.dump(hmm_model, fold_dir / "hmm_model.joblib")
    joblib.dump(hmm_scaler, fold_dir / "hmm_scaler.joblib")
    joblib.dump(model, fold_dir / "model.joblib")
    joblib.dump(model_scaler, fold_dir / "model_scaler.joblib")
    joblib.dump(calibrator, fold_dir / "calibrator.joblib")

    test_metrics = metrics[metrics["split"] == "test_calibrated"].iloc[0].to_dict()
    xgboost_metrics["xgboost_delta_test_log_loss_vs_hmm_lr"] = xgboost_metrics["xgboost_test_log_loss"] - float(test_metrics["log_loss"])
    xgboost_metrics["xgboost_delta_signal_net_return_vs_hmm_lr"] = (
        xgboost_metrics["xgboost_test_signal_net_return"] - test_signal_metrics["net_return"]
    )
    return {
        "fold": fold.fold,
        "train_months": ",".join(fold.train_months),
        "validation_months": ",".join(fold.validation_months),
        "test_months": ",".join(fold.test_months),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "theta_prob": selected["theta_prob"],
        "theta_score": selected["theta_score"],
        "max_neutral": selected["max_neutral"],
        "max_hmm_entropy": selected["max_hmm_entropy"],
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_log_loss": float(test_metrics["log_loss"]),
        **{f"test_signal_{key}": value for key, value in test_signal_metrics.items()},
        **xgboost_metrics,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_summary(summary: pd.DataFrame, folds: list[WalkForwardFold], config: dict[str, Any], labels: pd.DataFrame) -> str:
    wf = config["walkforward"]
    if summary.empty:
        months = sorted(_session_months(labels)["month"].unique().tolist())
        return f"""# Walk-Forward Summary

No folds were generated.

Configured schema:

- fit: {wf["train_months"]} months
- validation: {wf["validation_months"]} month(s)
- test: {wf["test_months"]} month(s)
- step: {wf["step_months"]} month(s)

Available months: {", ".join(months) if months else "none"}

The current dataset is too short for the configured 5/1/1 month walk-forward scheme. The implementation is ready; rerun after loading a longer intraday history.
"""

    return f"""# Walk-Forward Summary

## Scope

- Folds: {len(folds)}
- Schema: fit {wf["train_months"]} months, validation {wf["validation_months"]} month(s), test {wf["test_months"]} month(s), step {wf["step_months"]} month(s)
- Purge bars: {wf.get("purge_bars", 0)}
- Embargo bars: {wf.get("embargo_bars", 0)}

## Fold Results

{_markdown_table(summary)}

## Notes

- HMM scaler/model are fit only on train rows in each fold.
- Predictive scaler/model are fit only on train rows in each fold.
- Calibration and threshold selection use validation only.
- Test is evaluated once per fold.
- XGBoost challenger metrics use base features only and are reported with the `xgboost_` prefix.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    folds = build_monthly_folds(labels, config)

    output_dir = Path(config["walkforward"]["output_dir"])
    summary_path = Path(config["walkforward"]["summary_file"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [run_fold(features, labels, fold, config, output_dir) for fold in folds]
    summary = pd.DataFrame(rows)
    summary_file = output_dir / "fold_summary.parquet"
    summary.to_parquet(summary_file, index=False)
    summary_path.write_text(render_summary(summary, folds, config, labels), encoding="utf-8")
    return summary_file, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monthly walk-forward HMM + predictive + signal pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    summary_file, summary_path = run(args.config)
    print(f"Walk-forward fold summary written to: {summary_file}")
    print(f"Walk-forward report written to: {summary_path}")


if __name__ == "__main__":
    main()

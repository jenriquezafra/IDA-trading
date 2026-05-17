from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation import calibration_metrics
from src.robustness import data_sufficiency
from src.signal import apply_selected_signal, build_signal_frame, evaluate_signal, select_thresholds_on_validation


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_optional_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(config))


def _disable_hmm_signal_filters(config: dict[str, Any]) -> dict[str, Any]:
    output = _copy_config(config)
    output.setdefault("signal", {})
    output["signal"]["allowed_hmm_states"] = []
    output["signal"]["max_hmm_entropy"] = 999.0
    output["signal"]["max_hmm_entropy_grid"] = [999.0]
    return output


def prediction_quality(predictions: pd.DataFrame, variant: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["variant", "split", "rows", "log_loss", "brier_multiclass", "expected_calibration_error", "avg_confidence", "accuracy"])
    rows = []
    for split, group in predictions.groupby("split", sort=False):
        rows.append({"variant": variant, "split": split, **calibration_metrics(group)})
    return pd.DataFrame(rows)


def signal_quality(signals: pd.DataFrame, variant: str, config: dict[str, Any]) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["variant", "split", "rows", "trades", "net_return", "avg_trade_net", "hit_ratio", "daily_sharpe"])
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    rows = []
    for split, group in signals.groupby("split", sort=False):
        rows.append({"variant": variant, "split": split, **evaluate_signal(group, group["signal"], cost_bps)})
    return pd.DataFrame(rows)


def run_signal_variant(
    variant: str,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    hmm_features: pd.DataFrame | None = None,
    use_hmm_filters: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if predictions.empty:
        return pd.DataFrame(), {}
    variant_config = config if use_hmm_filters else _disable_hmm_signal_filters(config)
    frame = build_signal_frame(predictions, hmm_features if use_hmm_filters else None)
    selected, _ = select_thresholds_on_validation(frame, variant_config)
    signals = apply_selected_signal(frame, selected, variant_config)
    signals.insert(0, "variant", variant)
    return signals, selected


def compare_test_variants(signal_metrics: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if signal_metrics.empty:
        return pd.DataFrame(columns=["variant", "test_net_return", "test_trades", "test_avg_trade_net", "test_hit_ratio", "test_log_loss", "delta_net_return_vs_base"])
    test_signal = signal_metrics[signal_metrics["split"] == "test"].copy()
    test_quality = quality[quality["split"] == "test"].copy()
    merged = test_signal.merge(
        test_quality[["variant", "log_loss", "accuracy", "expected_calibration_error"]],
        on="variant",
        how="left",
    )
    output = merged.rename(
        columns={
            "net_return": "test_net_return",
            "trades": "test_trades",
            "avg_trade_net": "test_avg_trade_net",
            "hit_ratio": "test_hit_ratio",
            "log_loss": "test_log_loss",
            "accuracy": "test_accuracy",
            "expected_calibration_error": "test_ece",
        }
    )
    base = output.loc[output["variant"] == "base_no_hmm", "test_net_return"]
    base_net = float(base.iloc[0]) if not base.empty else np.nan
    output["delta_net_return_vs_base"] = output["test_net_return"].astype(float) - base_net
    columns = [
        "variant",
        "test_net_return",
        "delta_net_return_vs_base",
        "test_trades",
        "test_avg_trade_net",
        "test_hit_ratio",
        "test_log_loss",
        "test_accuracy",
        "test_ece",
    ]
    return output[columns].sort_values(["test_net_return", "test_log_loss"], ascending=[False, True]).reset_index(drop=True)


def experiment_plan(sufficiency: dict[str, Any]) -> pd.DataFrame:
    rerun_status = "ready_for_full_rerun" if sufficiency["has_walkforward_evidence"] else "pending_long_intraday_history"
    return pd.DataFrame(
        [
            {"variant": "base_no_hmm", "status": "available_current_artifact", "notes": "Uses predictive_base_predictions."},
            {"variant": "hmm_all_features_no_filter", "status": "available_current_artifact", "notes": "Uses current HMM-feature model without entropy/regime signal filters."},
            {"variant": "hmm_all_features_with_filter", "status": "available_current_artifact", "notes": "Uses current HMM-feature model with configured signal filters."},
            {"variant": "hard_hmm_state_only", "status": rerun_status, "notes": "Requires retraining model with hard state one-hot and without HMM probabilities."},
            {"variant": "hmm_probabilities_only", "status": rerun_status, "notes": "Requires retraining model with HMM probabilities and without hard state one-hot."},
            {"variant": "hmm_filters_only", "status": rerun_status, "notes": "Requires base model plus HMM filters without HMM features in the model matrix."},
            {"variant": "separate_models_by_regime", "status": rerun_status, "notes": "Requires enough per-regime data inside each train fold."},
            {"variant": "xgboost_no_hmm", "status": "blocked_until_block_17", "notes": "XGBoost challenger is the next roadmap block."},
            {"variant": "xgboost_with_hmm", "status": "blocked_until_block_17", "notes": "XGBoost challenger is the next roadmap block."},
        ]
    )


def decision_summary(comparison: pd.DataFrame, sufficiency: dict[str, Any]) -> str:
    if comparison.empty:
        return "No comparable current artifacts were available."
    base = comparison[comparison["variant"] == "base_no_hmm"]
    hmm = comparison[comparison["variant"] == "hmm_all_features_with_filter"]
    if base.empty or hmm.empty:
        return "Base and HMM variants were not both available; HMM contribution cannot be compared yet."
    delta = float(hmm["delta_net_return_vs_base"].iloc[0])
    evidence = "walk-forward evidence" if sufficiency["has_walkforward_evidence"] else "short-sample exploratory evidence"
    if delta > 0:
        return f"HMM with filters improves current test net return vs base by {delta:.6f}, but this is only {evidence}."
    if delta < 0:
        return f"HMM with filters underperforms current base test net return by {abs(delta):.6f}; this is only {evidence}."
    return f"HMM with filters matches base current test net return; this is only {evidence}."


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_report(
    sufficiency: dict[str, Any],
    comparison: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    quality: pd.DataFrame,
    selected: pd.DataFrame,
    plan: pd.DataFrame,
) -> str:
    evidence_note = (
        "Full walk-forward ablation can be interpreted."
        if sufficiency["has_walkforward_evidence"]
        else "This is an ablation framework run on the current short dataset; it is not acceptance evidence."
    )
    return f"""# Ablation Report

## Evidence Status

- Available months: {sufficiency["available_months"]} (`{sufficiency["available_month_list"] or "none"}`)
- Required months for one configured fold: {sufficiency["required_months"]}
- Generated walk-forward folds: {sufficiency["generated_folds"]}
- Interpretation: {evidence_note}

## Current Test Comparison

{_markdown_table(comparison)}

## Decision Summary

{decision_summary(comparison, sufficiency)}

## Selected Signal Thresholds

{_markdown_table(selected)}

## Signal Metrics

{_markdown_table(signal_metrics)}

## Prediction Quality

{_markdown_table(quality)}

## Experiment Plan

{_markdown_table(plan)}

## Future Rerun Checklist

- Retrain hard-state-only and probability-only HMM variants after loading long intraday history.
- Run separate per-regime models only when each train fold has enough observations by regime.
- Add XGBoost variants in block 17, then rerun this ablation report including those artifacts.
- Treat HMM as useful only if it improves the base model OOS after costs and across folds.
"""


def build_ablation(config: dict[str, Any]) -> dict[str, Any]:
    labels = _read_optional_parquet(config["data"]["labels_file"])
    base_predictions = _read_optional_parquet(config["model"]["base_predictions_file"])
    hmm_predictions = _read_optional_parquet(config["model"]["hmm_predictions_file"])
    hmm_features = _read_optional_parquet(config["data"]["hmm_features_file"])

    sufficiency = data_sufficiency(labels, config)
    variants = [
        ("base_no_hmm", base_predictions, None, False),
        ("hmm_all_features_no_filter", hmm_predictions, None, False),
        ("hmm_all_features_with_filter", hmm_predictions, hmm_features, True),
    ]

    signal_frames = []
    selected_rows = []
    quality_frames = []
    metric_frames = []
    for name, predictions, variant_hmm_features, use_filters in variants:
        if predictions.empty:
            continue
        signals, selected = run_signal_variant(name, predictions, config, variant_hmm_features, use_filters)
        signal_frames.append(signals)
        selected_rows.append({"variant": name, **selected})
        quality_frames.append(prediction_quality(predictions, name))
        metric_frames.append(signal_quality(signals, name, config))

    quality = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    signal_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    comparison = compare_test_variants(signal_metrics, quality)
    return {
        "sufficiency": sufficiency,
        "signals": pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(),
        "selected": pd.DataFrame(selected_rows),
        "quality": quality,
        "signal_metrics": signal_metrics,
        "comparison": comparison,
        "plan": experiment_plan(sufficiency),
    }


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    outputs = build_ablation(config)

    ablation_cfg = config.get("ablation", {})
    output_dir = Path(ablation_cfg.get("output_dir", "reports/ablation"))
    report_path = Path(ablation_cfg.get("report_file", "reports/ablation.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for name in ["signals", "selected", "quality", "signal_metrics", "comparison", "plan"]:
        outputs[name].to_parquet(output_dir / f"{name}.parquet", index=False)
    pd.DataFrame([outputs["sufficiency"]]).to_parquet(output_dir / "data_sufficiency.parquet", index=False)
    report_path.write_text(
        render_report(
            outputs["sufficiency"],
            outputs["comparison"],
            outputs["signal_metrics"],
            outputs["quality"],
            outputs["selected"],
            outputs["plan"],
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation framework over current strategy artifacts.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path = run(args.config)
    print(f"Ablation report written to: {report_path}")


if __name__ == "__main__":
    main()

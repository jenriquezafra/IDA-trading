from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation import pnl_by_hour, pnl_by_regime
from src.labels import build_labels
from src.signal import apply_signal_rules, build_signal_frame, evaluate_signal
from src.walkforward import build_monthly_folds


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_optional_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def _session_months(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "session" not in frame:
        return []
    sessions = pd.Series(frame["session"].dropna().unique())
    return sorted(pd.to_datetime(sessions).dt.to_period("M").astype(str).unique().tolist())


def data_sufficiency(labels: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    months = _session_months(labels)
    wf_cfg = config["walkforward"]
    required_months = int(wf_cfg["train_months"]) + int(wf_cfg["validation_months"]) + int(wf_cfg["test_months"])
    folds = build_monthly_folds(labels, config) if not labels.empty else []
    return {
        "available_months": len(months),
        "required_months": required_months,
        "available_month_list": ",".join(months),
        "generated_folds": len(folds),
        "has_walkforward_evidence": len(folds) > 0,
    }


def horizon_sensitivity(features: pd.DataFrame, config: dict[str, Any], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        horizon_config = yaml.safe_load(yaml.safe_dump(config))
        horizon_config["labeling"]["horizon_bars"] = int(horizon)
        labels = build_labels(features, horizon_config, drop_invalid=True) if not features.empty else pd.DataFrame()
        if labels.empty:
            rows.append(
                {
                    "horizon_bars": int(horizon),
                    "rows": 0,
                    "sessions": 0,
                    "target_down_pct": np.nan,
                    "target_neutral_pct": np.nan,
                    "target_up_pct": np.nan,
                    "avg_abs_fwd_ret": np.nan,
                    "median_neutral_zone": np.nan,
                    "status": "no_rows",
                }
            )
            continue
        target_share = labels["target"].value_counts(normalize=True)
        rows.append(
            {
                "horizon_bars": int(horizon),
                "rows": int(len(labels)),
                "sessions": int(labels["session"].nunique()),
                "target_down_pct": float(target_share.get(-1, 0.0)),
                "target_neutral_pct": float(target_share.get(0, 0.0)),
                "target_up_pct": float(target_share.get(1, 0.0)),
                "avg_abs_fwd_ret": float(labels["fwd_ret"].abs().mean()),
                "median_neutral_zone": float(labels["neutral_zone"].median()),
                "status": "exploratory_current_data",
            }
        )
    return pd.DataFrame(rows)


def cost_stress(signals: pd.DataFrame, cost_bps_grid: list[float]) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=["split", "cost_bps", "rows", "trades", "net_return", "avg_trade_net", "hit_ratio", "daily_sharpe"])
    rows = []
    for split, group in signals.groupby("split", sort=False):
        for cost_bps in cost_bps_grid:
            metrics = evaluate_signal(group, group["signal"], float(cost_bps))
            rows.append({"split": split, "cost_bps": float(cost_bps), **metrics})
    return pd.DataFrame(rows)


def threshold_sensitivity(frame: pd.DataFrame, config: dict[str, Any], max_rows: int | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    robust_cfg = config.get("robustness", {})
    signal_cfg = config["signal"]
    grids = {
        "theta_prob": robust_cfg.get("theta_prob_grid", signal_cfg.get("theta_prob_grid", [signal_cfg.get("theta_prob", 0.55)])),
        "theta_score": robust_cfg.get("theta_score_grid", signal_cfg.get("theta_score_grid", [signal_cfg.get("theta_score", 0.10)])),
        "max_neutral": robust_cfg.get("max_neutral_grid", signal_cfg.get("max_neutral_grid", [signal_cfg.get("max_neutral", 0.55)])),
        "max_hmm_entropy": robust_cfg.get(
            "max_hmm_entropy_grid", signal_cfg.get("max_hmm_entropy_grid", [signal_cfg.get("max_hmm_entropy", 0.90)])
        ),
    }
    allowed_states = [int(state) for state in signal_cfg.get("allowed_hmm_states", [])]
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    rows = []
    for values in product(*grids.values()):
        params = {key: float(value) for key, value in zip(grids.keys(), values)}
        signal = apply_signal_rules(frame, allowed_hmm_states=allowed_states, **params)
        metrics = evaluate_signal(frame, signal, cost_bps)
        rows.append({**params, **metrics})
    output = pd.DataFrame(rows).sort_values(["net_return", "daily_sharpe", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return output.head(max_rows).copy() if max_rows else output


def experiment_plan(config: dict[str, Any], sufficiency: dict[str, Any]) -> pd.DataFrame:
    robust_cfg = config.get("robustness", {})
    enough_data = bool(sufficiency["has_walkforward_evidence"])
    rerun_status = "ready_for_full_rerun" if enough_data else "pending_long_intraday_history"
    rows = []

    for horizon in robust_cfg.get("horizons", [1, 2, 3]):
        rows.append({"family": "horizon", "parameter": "horizon_bars", "value": str(horizon), "status": "profiled_current_data"})
    for state_count in robust_cfg.get("hmm_states", [2, 3, 4, 5, 6]):
        rows.append({"family": "hmm_k", "parameter": "n_states", "value": str(state_count), "status": rerun_status})
    for cost_bps in robust_cfg.get("cost_bps", [1.0, 2.0, 5.0]):
        rows.append({"family": "cost", "parameter": "round_trip_cost_bps", "value": str(cost_bps), "status": "stress_replayed_current_signals"})
    for seed in robust_cfg.get("seeds", config.get("hmm", {}).get("stability_seeds", [42])):
        rows.append({"family": "seed", "parameter": "random_state", "value": str(seed), "status": rerun_status})
    for train_months in robust_cfg.get("train_months", [config["walkforward"]["train_months"]]):
        rows.append({"family": "training_window", "parameter": "train_months", "value": str(train_months), "status": rerun_status})
    for period in robust_cfg.get("periods", ["all"]):
        rows.append({"family": "period", "parameter": "period", "value": period, "status": rerun_status})

    rows.append({"family": "regime", "parameter": "hmm_state", "value": "all_states", "status": "profiled_current_trades"})
    rows.append({"family": "hour", "parameter": "entry_hour", "value": "all_hours", "status": "profiled_current_trades"})
    return pd.DataFrame(rows)


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
    horizon: pd.DataFrame,
    costs: pd.DataFrame,
    thresholds: pd.DataFrame,
    regime: pd.DataFrame,
    hour: pd.DataFrame,
    plan: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    evidence_note = (
        "Full walk-forward robustness can be interpreted."
        if sufficiency["has_walkforward_evidence"]
        else "This is a robustness framework run on the current short dataset; it is not acceptance evidence."
    )
    return f"""# Robustness Report

## Evidence Status

- Available months: {sufficiency["available_months"]} (`{sufficiency["available_month_list"] or "none"}`)
- Required months for one configured fold: {sufficiency["required_months"]}
- Generated walk-forward folds: {sufficiency["generated_folds"]}
- Interpretation: {evidence_note}

## Horizon Sensitivity

{_markdown_table(horizon)}

## Cost Stress

{_markdown_table(costs)}

## Threshold Sensitivity Top 20

{_markdown_table(thresholds, max_rows=20)}

## PnL By Regime

{_markdown_table(regime)}

## PnL By Hour

{_markdown_table(hour)}

## Experiment Plan

{_markdown_table(plan)}

## Future Rerun Checklist

- Load enough intraday history for the configured walk-forward schema before treating robustness as evidence.
- Rerun HMM K, seed, training-window, and period experiments end to end after loading long history.
- Keep threshold and cost stress separate from threshold selection; do not optimize on test.
- Recreate this report with `python -m src.robustness --config configs/base.yaml` after replacing the dataset.
"""


def build_robustness(config: dict[str, Any]) -> dict[str, Any]:
    features = _read_optional_parquet(config["data"]["features_file"])
    labels = _read_optional_parquet(config["data"]["labels_file"])
    predictions = _read_optional_parquet(config["signal"].get("predictions_file", config["model"]["hmm_predictions_file"]))
    hmm_features = _read_optional_parquet(config["data"]["hmm_features_file"])
    signals = _read_optional_parquet(config["data"]["signals_file"])
    trades = _read_optional_parquet(config["backtest"]["trades_file"])

    robust_cfg = config.get("robustness", {})
    sufficiency = data_sufficiency(labels, config)
    signal_frame = build_signal_frame(predictions, hmm_features) if not predictions.empty else pd.DataFrame()
    return {
        "sufficiency": sufficiency,
        "horizon": horizon_sensitivity(features, config, [int(value) for value in robust_cfg.get("horizons", [1, 2, 3])]),
        "costs": cost_stress(signals, [float(value) for value in robust_cfg.get("cost_bps", [1.0, 2.0, 5.0])]),
        "thresholds": threshold_sensitivity(signal_frame[signal_frame["split"] == "test"].copy() if "split" in signal_frame else signal_frame, config),
        "regime": pnl_by_regime(trades, hmm_features),
        "hour": pnl_by_hour(trades),
        "plan": experiment_plan(config, sufficiency),
    }


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    outputs = build_robustness(config)

    robust_cfg = config.get("robustness", {})
    output_dir = Path(robust_cfg.get("output_dir", "reports/robustness"))
    report_path = Path(robust_cfg.get("report_file", "reports/robustness.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    for name in ["horizon", "costs", "thresholds", "regime", "hour", "plan"]:
        outputs[name].to_parquet(output_dir / f"{name}.parquet", index=False)
    pd.DataFrame([outputs["sufficiency"]]).to_parquet(output_dir / "data_sufficiency.parquet", index=False)
    report_path.write_text(
        render_report(
            outputs["sufficiency"],
            outputs["horizon"],
            outputs["costs"],
            outputs["thresholds"],
            outputs["regime"],
            outputs["hour"],
            outputs["plan"],
            config,
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness framework over current strategy artifacts.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path = run(args.config)
    print(f"Robustness report written to: {report_path}")


if __name__ == "__main__":
    main()

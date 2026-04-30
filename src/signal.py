from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_predictions_with_hmm(predictions: pd.DataFrame, hmm_features: pd.DataFrame) -> pd.DataFrame:
    hmm_cols = ["timestamp", "session", "bar_index", "hmm_state", "hmm_entropy", "hmm_max_prob"]
    missing = sorted(set(hmm_cols) - set(hmm_features.columns))
    if missing:
        raise ValueError(f"HMM features missing columns required for signals: {missing}")
    return predictions.merge(hmm_features[hmm_cols], on=["timestamp", "session", "bar_index"], how="left", validate="one_to_one")


def build_signal_frame(predictions: pd.DataFrame, hmm_features: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"p_up", "p_down", "p_neutral", "score", "fwd_ret", "split"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing signal columns: {missing}")

    frame = predictions.copy()
    if hmm_features is not None and ("hmm_entropy" not in frame.columns or "hmm_state" not in frame.columns):
        frame = merge_predictions_with_hmm(frame, hmm_features)
    if "hmm_entropy" not in frame.columns:
        frame["hmm_entropy"] = 0.0
    if "hmm_state" not in frame.columns:
        frame["hmm_state"] = -1
    if "hmm_max_prob" not in frame.columns:
        frame["hmm_max_prob"] = np.nan
    frame["score"] = frame["p_up"] - frame["p_down"]
    return frame


def apply_signal_rules(
    frame: pd.DataFrame,
    theta_prob: float,
    theta_score: float,
    max_neutral: float,
    max_hmm_entropy: float,
    allowed_hmm_states: list[int] | None = None,
) -> pd.Series:
    allowed_hmm_states = allowed_hmm_states or []
    entropy_ok = frame["hmm_entropy"].fillna(1.0) <= max_hmm_entropy
    neutral_ok = frame["p_neutral"] <= max_neutral
    regime_ok = pd.Series(True, index=frame.index)
    if allowed_hmm_states:
        regime_ok = frame["hmm_state"].isin(allowed_hmm_states)

    tradable = entropy_ok & neutral_ok & regime_ok
    long_mask = tradable & (frame["p_up"] >= theta_prob) & (frame["score"] >= theta_score)
    short_mask = tradable & (frame["p_down"] >= theta_prob) & (frame["score"] <= -theta_score)

    signal = pd.Series(0, index=frame.index, dtype="int64")
    signal.loc[long_mask] = 1
    signal.loc[short_mask] = -1
    return signal


def evaluate_signal(frame: pd.DataFrame, signal: pd.Series, round_trip_cost_bps: float) -> dict[str, float | int]:
    position = signal.astype(int)
    active = position != 0
    gross_ret = position * frame["fwd_ret"]
    cost_ret = position.abs() * (round_trip_cost_bps / 10_000.0)
    net_ret = gross_ret - cost_ret
    daily = net_ret.groupby(frame["session"]).sum()
    sharpe = np.nan
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))

    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "gross_return": float(gross_ret.sum()),
        "total_cost": float(cost_ret.sum()),
        "net_return": float(net_ret.sum()),
        "avg_trade_net": float(net_ret[active].mean()) if active.any() else 0.0,
        "hit_ratio": float((net_ret[active] > 0).mean()) if active.any() else np.nan,
        "daily_sharpe": sharpe,
    }


def _threshold_grid(config: dict[str, Any]) -> list[dict[str, float]]:
    signal_cfg = config["signal"]
    return [
        {
            "theta_prob": float(theta_prob),
            "theta_score": float(theta_score),
            "max_neutral": float(max_neutral),
            "max_hmm_entropy": float(max_hmm_entropy),
        }
        for theta_prob, theta_score, max_neutral, max_hmm_entropy in product(
            signal_cfg.get("theta_prob_grid", [signal_cfg.get("theta_prob", 0.55)]),
            signal_cfg.get("theta_score_grid", [signal_cfg.get("theta_score", 0.10)]),
            signal_cfg.get("max_neutral_grid", [signal_cfg.get("max_neutral", 0.55)]),
            signal_cfg.get("max_hmm_entropy_grid", [signal_cfg.get("max_hmm_entropy", 0.90)]),
        )
    ]


def select_thresholds_on_validation(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame]:
    validation = frame[frame["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("Validation split is empty; cannot select signal thresholds")

    allowed_states = [int(state) for state in config["signal"].get("allowed_hmm_states", [])]
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    rows = []
    for params in _threshold_grid(config):
        signal = apply_signal_rules(validation, allowed_hmm_states=allowed_states, **params)
        metrics = evaluate_signal(validation, signal, cost_bps)
        rows.append({**params, **metrics})

    grid = pd.DataFrame(rows)
    ranked = grid.sort_values(["net_return", "daily_sharpe", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    best = ranked.iloc[0][["theta_prob", "theta_score", "max_neutral", "max_hmm_entropy"]].astype(float).to_dict()
    return best, ranked


def apply_selected_signal(frame: pd.DataFrame, params: dict[str, float], config: dict[str, Any]) -> pd.DataFrame:
    allowed_states = [int(state) for state in config["signal"].get("allowed_hmm_states", [])]
    output = frame.copy()
    output["signal"] = apply_signal_rules(output, allowed_hmm_states=allowed_states, **params)
    output["signal_name"] = output["signal"].map({-1: "short", 0: "flat", 1: "long"})
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    output["signal_gross_ret"] = output["signal"] * output["fwd_ret"]
    output["signal_cost_ret"] = output["signal"].abs() * (cost_bps / 10_000.0)
    output["signal_net_ret"] = output["signal_gross_ret"] - output["signal_cost_ret"]
    return output


def render_report(selected: dict[str, float], grid: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]) -> str:
    def table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
        display = frame.head(max_rows).copy() if max_rows else frame.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        headers = display.columns.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    split_rows = []
    for split, group in signals.groupby("split", sort=False):
        split_rows.append({"split": split, **evaluate_signal(group, group["signal"], cost_bps)})
    split_metrics = pd.DataFrame(split_rows)
    counts = signals.groupby(["split", "signal_name"]).size().unstack(fill_value=0).reset_index()

    return f"""# Signal Report

## Selected Thresholds

- theta_prob: {selected["theta_prob"]:.4f}
- theta_score: {selected["theta_score"]:.4f}
- max_neutral: {selected["max_neutral"]:.4f}
- max_hmm_entropy: {selected["max_hmm_entropy"]:.4f}
- allowed_hmm_states: {config["signal"].get("allowed_hmm_states", [])}

## Split Metrics

{table(split_metrics)}

## Signal Counts

{table(counts)}

## Validation Grid Top 10

{table(grid, max_rows=10)}

## Notes

- Thresholds are selected only on validation rows.
- Test rows are evaluated with the selected thresholds without re-optimization.
- Signals use `p_up`, `p_down`, `p_neutral`, `score`, HMM entropy, and optional regime filtering.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    predictions_path = Path(config["signal"].get("predictions_file", config["model"]["hmm_predictions_file"]))
    hmm_path = Path(config["data"]["hmm_features_file"])
    output_path = Path(config["signal"].get("output_file", config["data"]["signals_file"]))
    report_path = Path(config["signal"].get("report_file", "reports/signal_report.md"))

    predictions = pd.read_parquet(predictions_path)
    hmm_features = pd.read_parquet(hmm_path)
    frame = build_signal_frame(predictions, hmm_features)
    selected, grid = select_thresholds_on_validation(frame, config)
    signals = apply_selected_signal(frame, selected, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_path, index=False)
    report_path.write_text(render_report(selected, grid, signals, config), encoding="utf-8")
    return output_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert model probabilities into long/short/flat signals.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    output_path, report_path = run(args.config)
    print(f"Signals written to: {output_path}")
    print(f"Signal report written to: {report_path}")


if __name__ == "__main__":
    main()

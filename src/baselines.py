from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


BASELINE_ORDER = ("always_flat", "random", "intraday_buy_hold", "momentum", "reversion")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _threshold_sign(values: pd.Series, thresholds: pd.Series) -> pd.Series:
    signal = pd.Series(0, index=values.index, dtype="int64")
    signal.loc[values > thresholds] = 1
    signal.loc[values < -thresholds] = -1
    return signal


def generate_baseline_positions(labels: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    baseline_cfg = config.get("baselines", {})
    threshold_col = baseline_cfg.get("signal_threshold_column", "neutral_zone")
    momentum_col = baseline_cfg.get("momentum_column", "ret_3")
    reversal_col = baseline_cfg.get("reversal_column", "ret_3")

    required = {"fwd_ret", threshold_col, momentum_col, reversal_col}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels data is missing required columns for baselines: {missing}")

    rng = np.random.default_rng(int(baseline_cfg.get("random_seed", 42)))
    random_probs = [
        float(baseline_cfg.get("random_prob_short", 1.0 / 3.0)),
        float(baseline_cfg.get("random_prob_flat", 1.0 / 3.0)),
        float(baseline_cfg.get("random_prob_long", 1.0 / 3.0)),
    ]
    random_probs = np.array(random_probs, dtype=float)
    random_probs = random_probs / random_probs.sum()

    momentum = _threshold_sign(labels[momentum_col], labels[threshold_col])
    reversion = -_threshold_sign(labels[reversal_col], labels[threshold_col])

    return {
        "always_flat": pd.Series(0, index=labels.index, dtype="int64"),
        "random": pd.Series(rng.choice([-1, 0, 1], size=len(labels), p=random_probs), index=labels.index, dtype="int64"),
        "intraday_buy_hold": pd.Series(1, index=labels.index, dtype="int64"),
        "momentum": momentum,
        "reversion": reversion,
    }


def evaluate_strategy(labels: pd.DataFrame, name: str, position: pd.Series, round_trip_cost_bps: float) -> pd.DataFrame:
    trades = labels.loc[:, ["timestamp", "session", "bar_index", "entry_px", "exit_px", "fwd_ret", "target"]].copy()
    trades["strategy"] = name
    trades["position"] = position.astype(int).to_numpy()
    trades["gross_ret"] = trades["position"] * trades["fwd_ret"]
    trades["cost_ret"] = (round_trip_cost_bps / 10_000.0) * trades["position"].abs()
    trades["net_ret"] = trades["gross_ret"] - trades["cost_ret"]
    return trades


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for strategy, group in trades.groupby("strategy", sort=False):
        active = group[group["position"] != 0]
        daily = group.groupby("session", sort=True)["net_ret"].sum()
        downside = daily[daily < 0]
        sharpe = np.nan
        if len(daily) > 1 and daily.std(ddof=1) > 0:
            sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))

        gross_profit = float(active.loc[active["net_ret"] > 0, "net_ret"].sum())
        gross_loss = float(-active.loc[active["net_ret"] < 0, "net_ret"].sum())
        profit_factor = np.inf if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else np.nan)

        rows.append(
            {
                "strategy": strategy,
                "rows": int(len(group)),
                "trades": int(len(active)),
                "exposure": float((group["position"] != 0).mean()),
                "net_return": float(group["net_ret"].sum()),
                "gross_return": float(group["gross_ret"].sum()),
                "total_cost": float(group["cost_ret"].sum()),
                "avg_trade_net": float(active["net_ret"].mean()) if len(active) else 0.0,
                "median_trade_net": float(active["net_ret"].median()) if len(active) else 0.0,
                "hit_ratio": float((active["net_ret"] > 0).mean()) if len(active) else np.nan,
                "profit_factor": profit_factor,
                "daily_sharpe": sharpe,
                "max_daily_loss": float(daily.min()) if len(daily) else 0.0,
                "downside_days": int(len(downside)),
            }
        )
    return pd.DataFrame(rows)


def build_baseline_results(labels: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_trip_cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    positions = generate_baseline_positions(labels, config)
    trades = pd.concat(
        [evaluate_strategy(labels, name, positions[name], round_trip_cost_bps) for name in BASELINE_ORDER],
        ignore_index=True,
    )
    summary = summarize_trades(trades)
    return trades, summary


def render_report(summary: pd.DataFrame, config: dict[str, Any], labels: pd.DataFrame) -> str:
    start = labels["timestamp"].min()
    end = labels["timestamp"].max()
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])

    display = summary.copy()
    for col in [
        "exposure",
        "net_return",
        "gross_return",
        "total_cost",
        "avg_trade_net",
        "median_trade_net",
        "hit_ratio",
        "profit_factor",
        "daily_sharpe",
        "max_daily_loss",
    ]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")

    headers = display.columns.tolist()
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    table = "\n".join(table_lines)
    return f"""# Baseline Report

## Scope

- Input labels: `{config["data"]["labels_file"]}`
- Rows: {len(labels)}
- Sessions: {labels["session"].nunique()}
- Period: `{start}` to `{end}`
- Horizon: {int(config["labeling"]["horizon_bars"])} bars
- Execution: signal at close `t`, entry at `open_(t+1)`, exit at `open_(t+h+1)`
- Round-trip cost: {cost_bps:.2f} bps

## Results

{table}

## Notes

- `always_flat` is the zero-risk benchmark.
- `intraday_buy_hold` is long on every eligible label row over the same next-open horizon.
- `momentum` follows `{config.get("baselines", {}).get("momentum_column", "ret_3")}` when it exceeds the neutral-zone threshold.
- `reversion` takes the opposite side of the same threshold rule.
- `random` uses a fixed seed for reproducibility.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    labels_path = Path(config["data"]["labels_file"])
    report_path = Path(config["paths"]["reports"]) / "baseline_report.md"
    trades_path = Path(config["paths"]["reports"]) / "baseline_trades.parquet"

    labels = pd.read_parquet(labels_path)
    trades, summary = build_baseline_results(labels, config)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(trades_path, index=False)
    report_path.write_text(render_report(summary, config, labels), encoding="utf-8")
    return report_path, trades_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple next-open baseline strategies.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path, trades_path = run(args.config)
    print(f"Baseline report written to: {report_path}")
    print(f"Baseline trades written to: {trades_path}")


if __name__ == "__main__":
    main()

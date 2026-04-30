from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

from src.predictive_model import CLASS_ORDER


PROBABILITY_COLUMNS = ["p_down", "p_neutral", "p_up"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_optional_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)


def daily_sharpe(daily: pd.DataFrame, periods_per_year: int = 252) -> float:
    if daily.empty or "net_ret" not in daily:
        return np.nan
    returns = daily["net_ret"].astype(float).dropna()
    if len(returns) < 2:
        return np.nan
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.DataFrame) -> float:
    if equity.empty or "equity" not in equity:
        return 0.0
    curve = equity["equity"].astype(float).ffill().fillna(0.0)
    drawdown = curve.cummax() - curve
    return float(drawdown.max()) if not drawdown.empty else 0.0


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty or "net_ret" not in trades:
        return np.nan
    pnl = trades["net_ret"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / abs(gross_loss))


def summarize_trades(trades: pd.DataFrame, daily: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trades": 0,
            "net_return": 0.0,
            "gross_return": 0.0,
            "total_cost": 0.0,
            "daily_sharpe_net": daily_sharpe(daily),
            "max_drawdown": max_drawdown(equity),
            "profit_factor": np.nan,
            "hit_ratio": np.nan,
            "avg_trade_net": 0.0,
            "median_trade_net": 0.0,
            "turnover_trades_per_day": 0.0,
        }

    net = trades["net_ret"].astype(float)
    sessions = int(daily["session"].nunique()) if not daily.empty and "session" in daily else int(trades["session"].nunique())
    return {
        "trades": int(len(trades)),
        "net_return": float(net.sum()),
        "gross_return": float(trades["gross_ret"].astype(float).sum()) if "gross_ret" in trades else np.nan,
        "total_cost": float(trades["total_cost_ret"].astype(float).sum()) if "total_cost_ret" in trades else np.nan,
        "daily_sharpe_net": daily_sharpe(daily),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(trades),
        "hit_ratio": float((net > 0).mean()),
        "avg_trade_net": float(net.mean()),
        "median_trade_net": float(net.median()),
        "turnover_trades_per_day": float(len(trades) / sessions) if sessions else np.nan,
    }


def exposure_from_trades(trades: pd.DataFrame, cleaned: pd.DataFrame) -> float:
    trade_required = {"session", "entry_bar_index", "exit_bar_index"}
    if trades.empty or cleaned.empty or trade_required - set(trades.columns):
        return 0.0
    if "session" not in cleaned or "bar_index" not in cleaned:
        return np.nan
    total_bars = len(cleaned[["session", "bar_index"]].drop_duplicates())
    if total_bars == 0:
        return np.nan
    held_bars = (trades["exit_bar_index"].astype(int) - trades["entry_bar_index"].astype(int) + 1).clip(lower=0).sum()
    return float(held_bars / total_bars)


def pnl_by_side(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "side" not in trades:
        return pd.DataFrame(columns=["side", "trades", "net_return", "avg_trade_net", "hit_ratio"])
    return (
        trades.groupby("side", as_index=False)
        .agg(
            trades=("net_ret", "size"),
            net_return=("net_ret", "sum"),
            avg_trade_net=("net_ret", "mean"),
            hit_ratio=("net_ret", lambda values: float((values > 0).mean())),
        )
        .sort_values("side")
        .reset_index(drop=True)
    )


def pnl_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "entry_timestamp" not in trades:
        return pd.DataFrame(columns=["entry_hour", "trades", "net_return", "avg_trade_net", "hit_ratio"])
    frame = trades.copy()
    frame["entry_hour"] = pd.to_datetime(frame["entry_timestamp"]).dt.hour
    return (
        frame.groupby("entry_hour", as_index=False)
        .agg(
            trades=("net_ret", "size"),
            net_return=("net_ret", "sum"),
            avg_trade_net=("net_ret", "mean"),
            hit_ratio=("net_ret", lambda values: float((values > 0).mean())),
        )
        .sort_values("entry_hour")
        .reset_index(drop=True)
    )


def pnl_by_regime(trades: pd.DataFrame, hmm_features: pd.DataFrame) -> pd.DataFrame:
    columns = ["hmm_state", "trades", "net_return", "avg_trade_net", "hit_ratio"]
    if trades.empty or hmm_features.empty:
        return pd.DataFrame(columns=columns)
    required = {"signal_timestamp", "session", "net_ret"}
    hmm_required = {"timestamp", "session", "bar_index", "hmm_state"}
    if required - set(trades.columns) or hmm_required - set(hmm_features.columns):
        return pd.DataFrame(columns=columns)

    regimes = hmm_features[["timestamp", "session", "hmm_state"]].rename(columns={"timestamp": "signal_timestamp"})
    merged = trades.merge(regimes, on=["signal_timestamp", "session"], how="left", validate="many_to_one")
    if "hmm_state" not in merged or merged["hmm_state"].isna().all():
        return pd.DataFrame(columns=columns)
    return (
        merged.dropna(subset=["hmm_state"])
        .assign(hmm_state=lambda frame: frame["hmm_state"].astype(int))
        .groupby("hmm_state", as_index=False)
        .agg(
            trades=("net_ret", "size"),
            net_return=("net_ret", "sum"),
            avg_trade_net=("net_ret", "mean"),
            hit_ratio=("net_ret", lambda values: float((values > 0).mean())),
        )
        .sort_values("hmm_state")
        .reset_index(drop=True)
    )


def pnl_by_fold(fold_summary: pd.DataFrame) -> pd.DataFrame:
    if fold_summary.empty:
        return pd.DataFrame(columns=["fold", "test_months", "trades", "net_return", "avg_trade_net", "hit_ratio"])
    rename = {
        "test_signal_trades": "trades",
        "test_signal_net_return": "net_return",
        "test_signal_avg_trade_net": "avg_trade_net",
        "test_signal_hit_ratio": "hit_ratio",
    }
    available = ["fold", "test_months", *[col for col in rename if col in fold_summary.columns]]
    output = fold_summary[available].rename(columns=rename).copy()
    for col in ["trades", "net_return", "avg_trade_net", "hit_ratio"]:
        if col not in output:
            output[col] = np.nan
    return output[["fold", "test_months", "trades", "net_return", "avg_trade_net", "hit_ratio"]]


def calibration_metrics(predictions: pd.DataFrame, n_bins: int = 10) -> dict[str, float | int]:
    if predictions.empty or not set(PROBABILITY_COLUMNS + ["target"]).issubset(predictions.columns):
        return {
            "rows": 0,
            "log_loss": np.nan,
            "brier_multiclass": np.nan,
            "expected_calibration_error": np.nan,
            "avg_confidence": np.nan,
            "accuracy": np.nan,
        }

    frame = predictions.dropna(subset=PROBABILITY_COLUMNS + ["target"]).copy()
    if frame.empty:
        return {
            "rows": 0,
            "log_loss": np.nan,
            "brier_multiclass": np.nan,
            "expected_calibration_error": np.nan,
            "avg_confidence": np.nan,
            "accuracy": np.nan,
        }

    proba = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    y_true = frame["target"].to_numpy()
    pred_idx = proba.argmax(axis=1)
    y_pred = CLASS_ORDER[pred_idx]
    confidence = proba.max(axis=1)
    correct = y_pred == y_true

    one_hot = np.zeros_like(proba)
    for idx, klass in enumerate(CLASS_ORDER):
        one_hot[:, idx] = y_true == klass
    brier = np.mean(np.sum((proba - one_hot) ** 2, axis=1))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())

    return {
        "rows": int(len(frame)),
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_ORDER)),
        "brier_multiclass": float(brier),
        "expected_calibration_error": float(ece),
        "avg_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(_format_value)
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    return f"{float(value):.6f}"


def _summary_table(summary: dict[str, float | int]) -> str:
    rows = []
    for metric, value in summary.items():
        rows.append({"metric": metric, "value": _format_value(value) if isinstance(value, float) else value})
    return _markdown_table(pd.DataFrame(rows))


def render_report(
    summary: dict[str, float | int],
    side: pd.DataFrame,
    regime: pd.DataFrame,
    hour: pd.DataFrame,
    folds: pd.DataFrame,
    calibration: dict[str, float | int],
    config: dict[str, Any],
) -> str:
    fold_note = ""
    if folds.empty:
        wf = config.get("walkforward", {})
        fold_note = (
            "\n\nNo real walk-forward folds are available with the current dataset. "
            f"The configured schema needs {wf.get('train_months', '?')} train month(s), "
            f"{wf.get('validation_months', '?')} validation month(s), and {wf.get('test_months', '?')} test month(s)."
        )

    return f"""# Walk-Forward Evaluation Summary

## Scope

- Trades: `{config["backtest"]["trades_file"]}`
- Daily PnL: `{config["backtest"]["daily_pnl_file"]}`
- Equity curve: `{config["backtest"]["equity_file"]}`
- Predictions: `{config["signal"].get("predictions_file", config["model"]["hmm_predictions_file"])}`
- Cost scenario: `{config["backtest"].get("cost_scenario", "base")}`

## Net Metrics

{_summary_table(summary)}

## Calibration

{_summary_table(calibration)}

## PnL Long Vs Short

{_markdown_table(side)}

## PnL By Regime

{_markdown_table(regime)}

## PnL By Hour

{_markdown_table(hour)}

## PnL By Fold

{_markdown_table(folds)}
{fold_note}

## Notes

- Daily Sharpe is computed from net daily PnL and annualized with 252 sessions.
- Max drawdown is computed from the additive net equity curve.
- Turnover is reported as completed trades per day.
- Exposure is estimated as held bars divided by cleaned regular-session bars.
"""


def build_evaluation(config: dict[str, Any]) -> tuple[dict[str, float | int], dict[str, pd.DataFrame], dict[str, float | int]]:
    trades = _read_optional_parquet(config["backtest"]["trades_file"])
    equity = _read_optional_parquet(config["backtest"]["equity_file"])
    daily = _read_optional_parquet(config["backtest"]["daily_pnl_file"])
    cleaned = _read_optional_parquet(config["data"]["cleaned_file"])
    hmm_features = _read_optional_parquet(config["data"]["hmm_features_file"])
    fold_summary = _read_optional_parquet(Path(config["walkforward"]["output_dir"]) / "fold_summary.parquet")
    predictions = _read_optional_parquet(config["signal"].get("predictions_file", config["model"]["hmm_predictions_file"]))

    summary = summarize_trades(trades, daily, equity)
    summary["exposure"] = exposure_from_trades(trades, cleaned)
    tables = {
        "side": pnl_by_side(trades),
        "regime": pnl_by_regime(trades, hmm_features),
        "hour": pnl_by_hour(trades),
        "folds": pnl_by_fold(fold_summary),
    }
    calibration = calibration_metrics(predictions[predictions["split"] == "test"] if "split" in predictions else predictions)
    return summary, tables, calibration


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    summary, tables, calibration = build_evaluation(config)

    eval_cfg = config.get("evaluation", {})
    report_path = Path(eval_cfg.get("report_file", "reports/walkforward_summary.md"))
    metrics_path = Path(eval_cfg.get("metrics_file", "reports/evaluation_metrics.parquet"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([summary | {f"calibration_{key}": value for key, value in calibration.items()}])
    metrics.to_parquet(metrics_path, index=False)
    report_path.write_text(
        render_report(summary, tables["side"], tables["regime"], tables["hour"], tables["folds"], calibration, config),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate net intraday strategy results.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path = run(args.config)
    print(f"Evaluation report written to: {report_path}")


if __name__ == "__main__":
    main()

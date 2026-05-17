from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.signal import apply_selected_signal, build_signal_frame, select_thresholds_on_validation


SUMMARY_COLUMNS = [
    "strategy",
    "source",
    "cost_bps",
    "trades",
    "net_return",
    "gross_return",
    "total_cost",
    "daily_sharpe_net",
    "profit_factor_net",
    "avg_trade_net",
    "max_drawdown",
    "folds_positive",
    "folds_negative",
    "beats_always_flat",
    "beats_random",
    "beats_momentum",
    "beats_reversion",
    "beats_model_without_hmm",
    "status",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _daily_sharpe(daily_net: pd.Series) -> float:
    values = daily_net.astype(float).dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * values.mean() / std)


def _max_drawdown(net_returns: pd.Series) -> float:
    if net_returns.empty:
        return 0.0
    equity = net_returns.astype(float).cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _profit_factor(net_returns: pd.Series) -> float:
    active = net_returns.astype(float).dropna()
    gross_profit = active[active > 0].sum()
    gross_loss = -active[active < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def summarize_labeled_positions(
    frame: pd.DataFrame,
    strategy: str,
    source: str,
    cost_bps: float,
    position_col: str = "position",
    gross_col: str = "gross_ret",
    cost_col: str = "cost_ret",
    net_col: str = "net_ret",
) -> dict[str, Any]:
    if frame.empty:
        return _empty_row(strategy, source, cost_bps)

    ordered = frame.sort_values(["session", "timestamp", "bar_index"], kind="stable") if "bar_index" in frame else frame.sort_values(["session"])
    active = ordered[ordered[position_col].astype(float) != 0] if position_col in ordered else ordered
    daily = ordered.groupby("session", sort=True)[net_col].sum() if "session" in ordered else pd.Series(dtype=float)
    monthly = ordered.groupby(pd.to_datetime(ordered["session"]).dt.to_period("M").astype(str))[net_col].sum() if "session" in ordered else pd.Series(dtype=float)

    return {
        "strategy": strategy,
        "source": source,
        "cost_bps": float(cost_bps),
        "trades": int(len(active)),
        "net_return": float(ordered[net_col].sum()),
        "gross_return": float(ordered[gross_col].sum()) if gross_col in ordered else np.nan,
        "total_cost": float(ordered[cost_col].sum()) if cost_col in ordered else np.nan,
        "daily_sharpe_net": _daily_sharpe(daily),
        "profit_factor_net": _profit_factor(active[net_col]) if not active.empty else np.nan,
        "avg_trade_net": float(active[net_col].mean()) if not active.empty else 0.0,
        "max_drawdown": _max_drawdown(ordered[net_col]),
        "folds_positive": int((monthly > 0).sum()) if not monthly.empty else 0,
        "folds_negative": int((monthly < 0).sum()) if not monthly.empty else 0,
    }


def summarize_backtest_trades(trades: pd.DataFrame, daily: pd.DataFrame, equity: pd.DataFrame, cost_bps: float) -> dict[str, Any]:
    if trades.empty:
        return _empty_row("hmm_lr_static_backtest", "reports/backtest_trades.parquet", cost_bps)

    monthly = trades.groupby(pd.to_datetime(trades["session"]).dt.to_period("M").astype(str))["net_ret"].sum()
    return {
        "strategy": "hmm_lr_static_backtest",
        "source": "reports/backtest_trades.parquet",
        "cost_bps": float(cost_bps),
        "trades": int(len(trades)),
        "net_return": float(trades["net_ret"].sum()),
        "gross_return": float(trades["gross_ret"].sum()),
        "total_cost": float(trades["total_cost_ret"].sum()),
        "daily_sharpe_net": _daily_sharpe(daily["net_ret"]) if not daily.empty and "net_ret" in daily else np.nan,
        "profit_factor_net": _profit_factor(trades["net_ret"]),
        "avg_trade_net": float(trades["net_ret"].mean()),
        "max_drawdown": float((equity["equity"].cummax() - equity["equity"]).max()) if not equity.empty and "equity" in equity else 0.0,
        "folds_positive": int((monthly > 0).sum()),
        "folds_negative": int((monthly < 0).sum()),
    }


def summarize_walkforward_signals(output_dir: str | Path, cost_bps: float) -> dict[str, Any]:
    output_path = Path(output_dir)
    frames: list[pd.DataFrame] = []
    fold_returns: list[float] = []
    for signals_path in sorted(output_path.glob("fold_*/signals.parquet"), key=lambda path: int(path.parent.name.split("_")[-1])):
        signals = pd.read_parquet(signals_path)
        test = signals[signals["split"] == "test"].copy()
        if test.empty:
            continue
        fold_returns.append(float(test["signal_net_ret"].sum()))
        frames.append(test.assign(fold=int(signals_path.parent.name.split("_")[-1])))

    if not frames:
        return _empty_row("hmm_lr_walkforward_oos", "reports/walkforward/fold_*/signals.parquet", cost_bps)

    frame = pd.concat(frames, ignore_index=True)
    active = frame[frame["signal"] != 0]
    daily = frame.groupby("session", sort=True)["signal_net_ret"].sum()
    fold_series = pd.Series(fold_returns, dtype=float)
    return {
        "strategy": "hmm_lr_walkforward_oos",
        "source": "reports/walkforward/fold_*/signals.parquet",
        "cost_bps": float(cost_bps),
        "trades": int(len(active)),
        "net_return": float(frame["signal_net_ret"].sum()),
        "gross_return": float(frame["signal_gross_ret"].sum()),
        "total_cost": float(frame["signal_cost_ret"].sum()),
        "daily_sharpe_net": _daily_sharpe(daily),
        "profit_factor_net": _profit_factor(active["signal_net_ret"]) if not active.empty else np.nan,
        "avg_trade_net": float(active["signal_net_ret"].mean()) if not active.empty else 0.0,
        "max_drawdown": _max_drawdown(frame["signal_net_ret"]),
        "folds_positive": int((fold_series > 0).sum()),
        "folds_negative": int((fold_series < 0).sum()),
    }


def summarize_prediction_signals(
    predictions: pd.DataFrame,
    strategy: str,
    source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    frame = build_signal_frame(predictions)
    selected, _ = select_thresholds_on_validation(frame, config)
    signals = apply_selected_signal(frame, selected, config)
    if "split" in signals:
        signals = signals[signals["split"] == "test"].copy()
    return summarize_labeled_positions(
        signals,
        strategy=strategy,
        source=source,
        cost_bps=cost_bps,
        position_col="signal",
        gross_col="signal_gross_ret",
        cost_col="signal_cost_ret",
        net_col="signal_net_ret",
    )


def _empty_row(strategy: str, source: str, cost_bps: float) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "source": source,
        "cost_bps": float(cost_bps),
        "trades": 0,
        "net_return": 0.0,
        "gross_return": 0.0,
        "total_cost": 0.0,
        "daily_sharpe_net": np.nan,
        "profit_factor_net": np.nan,
        "avg_trade_net": 0.0,
        "max_drawdown": 0.0,
        "folds_positive": 0,
        "folds_negative": 0,
    }


def add_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    net_by_strategy = output.set_index("strategy")["net_return"].to_dict()
    always_flat = float(net_by_strategy.get("always_flat", 0.0))
    random = float(net_by_strategy.get("random", np.nan))
    momentum = float(net_by_strategy.get("momentum", np.nan))
    reversion = float(net_by_strategy.get("reversion", np.nan))
    model_without_hmm = float(net_by_strategy.get("base_no_hmm_static_signal", always_flat))

    output["beats_always_flat"] = output["net_return"] > always_flat
    output["beats_random"] = output["net_return"] > random
    output["beats_momentum"] = output["net_return"] > momentum
    output["beats_reversion"] = output["net_return"] > reversion
    output["beats_model_without_hmm"] = output["net_return"] > model_without_hmm
    output["status"] = output.apply(_status_for_row, axis=1)
    return output.loc[:, SUMMARY_COLUMNS]


def _status_for_row(row: pd.Series) -> str:
    if row["strategy"] == "always_flat":
        return "benchmark"
    if int(row["trades"]) == 0:
        return "rejected_no_oos_trades"
    if row["net_return"] <= 0 or row["avg_trade_net"] <= 0:
        return "rejected_economic"
    if not bool(row["beats_always_flat"]):
        return "rejected_vs_flat"
    return "candidate_requires_more_evidence"


def _format_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_report(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    current = summary[summary["strategy"].isin(["hmm_lr_static_backtest", "hmm_lr_walkforward_oos"])]
    rejected = bool((current["status"] == "rejected_economic").all()) if not current.empty else True
    conclusion = (
        "Baseline actual rechazado economicamente salvo nueva evidencia HMM-first."
        if rejected
        else "Baseline actual requiere revision; hay al menos una rama candidata."
    )

    compact_cols = [
        "strategy",
        "cost_bps",
        "trades",
        "net_return",
        "daily_sharpe_net",
        "profit_factor_net",
        "avg_trade_net",
        "max_drawdown",
        "folds_positive",
        "folds_negative",
        "beats_always_flat",
        "beats_random",
        "beats_momentum",
        "beats_reversion",
        "beats_model_without_hmm",
        "status",
    ]
    return f"""# Baseline Status

## Scope

- Config: `configs/base.yaml`
- Coste base: {cost_bps:.2f} bps round-trip
- Objetivo: congelar el baseline negativo actual como referencia reproducible antes de explorar nuevas hipotesis HMM-first.

## Unified Metrics

{_markdown_table(summary[compact_cols])}

## Sources

{_markdown_table(summary[["strategy", "source"]])}

## Closed Branches

- `hmm_lr_static_backtest`: rechazado economicamente con la configuracion actual porque el PnL neto y el avg trade net son negativos.
- `hmm_lr_walkforward_oos`: rechazado economicamente con la configuracion actual porque el PnL neto OOS agregado es negativo.
- `Logistic Regression con HMM como feature plana`: no queda aceptada como edge; solo queda como referencia negativa.
- `XGBoost estatico`: mejora metricas predictivas estaticas, pero no ha generado senales utiles con el grid actual.

## Conclusion

{conclusion}

Este reporte no rechaza definitivamente la hipotesis HMM-first. Rechaza el baseline actual y obliga a que cualquier nueva rama demuestre mejora neta frente a esta tabla, siempre con costes, walk-forward y sin optimizar en test.
"""


def build_baseline_status(config: dict[str, Any]) -> pd.DataFrame:
    reports_dir = Path(config["paths"]["reports"])
    cost_bps = float(config["labeling"]["round_trip_cost_bps"])
    rows: list[dict[str, Any]] = []

    baseline_trades_path = reports_dir / "baseline_trades.parquet"
    if baseline_trades_path.exists():
        baseline_trades = pd.read_parquet(baseline_trades_path)
        for strategy, group in baseline_trades.groupby("strategy", sort=False):
            rows.append(
                summarize_labeled_positions(
                    group,
                    strategy=strategy,
                    source="reports/baseline_trades.parquet",
                    cost_bps=cost_bps,
                    position_col="position",
                    gross_col="gross_ret",
                    cost_col="cost_ret",
                    net_col="net_ret",
                )
            )

    backtest_path = reports_dir / "backtest_trades.parquet"
    daily_path = reports_dir / "daily_pnl.parquet"
    equity_path = reports_dir / "equity_curve.parquet"
    if backtest_path.exists():
        rows.append(
            summarize_backtest_trades(
                pd.read_parquet(backtest_path),
                pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame(),
                pd.read_parquet(equity_path) if equity_path.exists() else pd.DataFrame(),
                cost_bps=cost_bps,
            )
        )

    base_predictions_path = Path(config["model"]["base_predictions_file"])
    if base_predictions_path.exists():
        rows.append(
            summarize_prediction_signals(
                pd.read_parquet(base_predictions_path),
                strategy="base_no_hmm_static_signal",
                source="data/features/predictive_base_predictions.parquet",
                config=config,
            )
        )

    xgboost_predictions_path = Path(config["model"].get("xgboost_predictions_file", ""))
    if xgboost_predictions_path.exists():
        rows.append(
            summarize_prediction_signals(
                pd.read_parquet(xgboost_predictions_path),
                strategy="xgboost_static_signal",
                source="data/features/predictive_xgboost_predictions.parquet",
                config=config,
            )
        )

    wf_dir = Path(config["walkforward"]["output_dir"])
    rows.append(summarize_walkforward_signals(wf_dir, cost_bps=cost_bps))

    summary = pd.DataFrame(rows)
    return add_comparisons(summary)


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    summary = build_baseline_status(config)
    report_path = Path(config["paths"]["reports"]) / "baseline_status.md"
    metrics_path = Path(config["paths"]["reports"]) / "baseline_status.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(metrics_path, index=False)
    report_path.write_text(render_report(summary, config), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the current negative baseline status.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path = run(args.config)
    print(f"Baseline status written to: {report_path}")


if __name__ == "__main__":
    main()

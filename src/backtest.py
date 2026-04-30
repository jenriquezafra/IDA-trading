from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.costs import per_side_cost
from src.risk import DailyRiskState, can_open_trade, check_stop, desired_position, register_trade_close, register_trade_open, should_force_flat


@dataclass
class OpenTrade:
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    session: str
    entry_idx: int
    entry_bar_index: int
    entry_px: float
    position: float
    side: str


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_backtest_frame(cleaned: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    signal_cols = ["timestamp", "session", "bar_index", "signal", "split", "score", "p_up", "p_down", "p_neutral"]
    missing = sorted(set(signal_cols) - set(signals.columns))
    if missing:
        raise ValueError(f"Signals missing required backtest columns: {missing}")

    frame = cleaned.sort_values(["session", "bar_index"]).reset_index(drop=True).copy()
    frame = frame.merge(signals[signal_cols], on=["timestamp", "session", "bar_index"], how="left", validate="one_to_one")
    frame["signal"] = frame["signal"].fillna(0).astype(int)
    frame["split"] = frame["split"].fillna("unlabeled")
    return frame


def _same_session_next_index(frame: pd.DataFrame, idx: int) -> int | None:
    if idx + 1 >= len(frame):
        return None
    if frame.at[idx + 1, "session"] != frame.at[idx, "session"]:
        return None
    return idx + 1


def _find_temporal_exit_index(frame: pd.DataFrame, entry_idx: int, hold_bars: int) -> int:
    session = frame.at[entry_idx, "session"]
    target_idx = min(entry_idx + hold_bars, len(frame) - 1)
    while target_idx > entry_idx and frame.at[target_idx, "session"] != session:
        target_idx -= 1
    return target_idx


def _trade_record(
    trade: OpenTrade,
    exit_row: pd.Series,
    exit_px: float,
    exit_reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    entry_cost = per_side_cost(config).total_return * abs(trade.position)
    exit_cost = per_side_cost(config).total_return * abs(trade.position)
    gross_ret = trade.position * ((exit_px / trade.entry_px) - 1.0)
    net_ret = gross_ret - entry_cost - exit_cost
    return {
        "signal_timestamp": trade.signal_timestamp,
        "entry_timestamp": trade.entry_timestamp,
        "exit_timestamp": exit_row["timestamp"],
        "session": trade.session,
        "entry_bar_index": trade.entry_bar_index,
        "exit_bar_index": int(exit_row["bar_index"]),
        "side": trade.side,
        "position": trade.position,
        "entry_px": trade.entry_px,
        "exit_px": float(exit_px),
        "exit_reason": exit_reason,
        "gross_ret": gross_ret,
        "entry_cost_ret": entry_cost,
        "exit_cost_ret": exit_cost,
        "total_cost_ret": entry_cost + exit_cost,
        "net_ret": net_ret,
    }


def run_event_backtest(cleaned: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = prepare_backtest_frame(cleaned, signals)
    risk_state = DailyRiskState()
    trades: list[dict[str, Any]] = []
    open_trade: OpenTrade | None = None
    pending_signal_idx: int | None = None
    hold_bars = int(config.get("risk", {}).get("time_stop_bars", config.get("labeling", {}).get("horizon_bars", 2)))

    equity_rows = []
    equity = 0.0

    for idx, row in frame.iterrows():
        if pending_signal_idx is not None and idx == pending_signal_idx + 1 and open_trade is None:
            signal_row = frame.loc[pending_signal_idx]
            if signal_row["session"] == row["session"] and int(signal_row["signal"]) != 0 and can_open_trade(signal_row, risk_state, config):
                position = desired_position(signal_row, config)
                if position != 0:
                    open_trade = OpenTrade(
                        signal_timestamp=signal_row["timestamp"],
                        entry_timestamp=row["timestamp"],
                        session=str(row["session"]),
                        entry_idx=int(idx),
                        entry_bar_index=int(row["bar_index"]),
                        entry_px=float(row["open"]),
                        position=float(position),
                        side="long" if position > 0 else "short",
                    )
                    register_trade_open(signal_row, risk_state, config)
            pending_signal_idx = None

        if open_trade is not None and row["session"] == open_trade.session and int(row["bar_index"]) >= open_trade.entry_bar_index:
            exit_reason = None
            exit_px = None
            stopped, stop_px = check_stop(row, open_trade.entry_px, open_trade.position, config)
            temporal_exit_idx = _find_temporal_exit_index(frame, open_trade.entry_idx, hold_bars)
            if stopped:
                exit_reason = "stop"
                exit_px = float(stop_px)
            elif should_force_flat(row, config):
                exit_reason = "force_flat"
                exit_px = float(row["open"])
            elif idx >= temporal_exit_idx:
                exit_reason = "time"
                exit_px = float(row["open"])

            if exit_reason is not None and exit_px is not None:
                record = _trade_record(open_trade, row, exit_px, exit_reason, config)
                trades.append(record)
                equity += record["net_ret"]
                register_trade_close(record["net_ret"], risk_state, config)
                open_trade = None

        next_idx = _same_session_next_index(frame, idx)
        if open_trade is None and next_idx is not None and int(row["signal"]) != 0:
            pending_signal_idx = idx

        equity_rows.append({"timestamp": row["timestamp"], "session": row["session"], "equity": equity})

    if open_trade is not None:
        session_rows = frame[frame["session"] == open_trade.session]
        exit_row = session_rows.iloc[-1]
        record = _trade_record(open_trade, exit_row, float(exit_row["open"]), "session_close", config)
        trades.append(record)
        equity += record["net_ret"]
        register_trade_close(record["net_ret"], risk_state, config)
        equity_rows.append({"timestamp": exit_row["timestamp"], "session": exit_row["session"], "equity": equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if trades_df.empty:
        daily_df = pd.DataFrame(columns=["session", "gross_ret", "net_ret", "trades"])
    else:
        daily_df = (
            trades_df.groupby("session", as_index=False)
            .agg(gross_ret=("gross_ret", "sum"), net_ret=("net_ret", "sum"), trades=("net_ret", "size"))
        )
    return trades_df, equity_df, daily_df


def render_report(trades: pd.DataFrame, daily: pd.DataFrame, config: dict[str, Any]) -> str:
    if trades.empty:
        summary = {
            "trades": 0,
            "gross_return": 0.0,
            "net_return": 0.0,
            "total_cost": 0.0,
            "hit_ratio": np.nan,
            "avg_trade_net": 0.0,
            "max_daily_loss": 0.0,
        }
    else:
        summary = {
            "trades": int(len(trades)),
            "gross_return": float(trades["gross_ret"].sum()),
            "net_return": float(trades["net_ret"].sum()),
            "total_cost": float(trades["total_cost_ret"].sum()),
            "hit_ratio": float((trades["net_ret"] > 0).mean()),
            "avg_trade_net": float(trades["net_ret"].mean()),
            "max_daily_loss": float(daily["net_ret"].min()) if not daily.empty else 0.0,
        }

    lines = ["| metric | value |", "| --- | ---: |"]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")

    exit_counts = trades["exit_reason"].value_counts().to_dict() if not trades.empty else {}
    exit_lines = ["| exit_reason | trades |", "| --- | ---: |"]
    for reason, count in exit_counts.items():
        exit_lines.append(f"| {reason} | {count} |")

    return f"""# Backtest Report

## Scope

- Signals: `{config["data"]["signals_file"]}`
- Cost scenario: `{config["backtest"].get("cost_scenario", "base")}`
- Entry: next open only
- Time stop bars: {config["risk"].get("time_stop_bars", config["labeling"].get("horizon_bars", 2))}
- Stop loss: {float(config["risk"].get("stop_loss_bps", 0.0)):.2f} bps

## Summary

{chr(10).join(lines)}

## Exit Reasons

{chr(10).join(exit_lines)}

## Notes

- PnL metrics are net of entry and exit costs unless explicitly labelled gross.
- Entries are scheduled from signal bar `t` and executed at `open_(t+1)`.
- Trades are forced flat before the configured intraday cutoff and no overnight positions are allowed.
"""


def run(config_path: str | Path) -> tuple[Path, Path, Path, Path]:
    config = load_config(config_path)
    cleaned = pd.read_parquet(config["data"]["cleaned_file"])
    signals = pd.read_parquet(config["data"]["signals_file"])
    trades, equity, daily = run_event_backtest(cleaned, signals, config)

    trades_path = Path(config["backtest"]["trades_file"])
    equity_path = Path(config["backtest"]["equity_file"])
    daily_path = Path(config["backtest"]["daily_pnl_file"])
    report_path = Path(config["backtest"]["report_file"])
    for path in [trades_path, equity_path, daily_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    trades.to_parquet(trades_path, index=False)
    equity.to_parquet(equity_path, index=False)
    daily.to_parquet(daily_path, index=False)
    report_path.write_text(render_report(trades, daily, config), encoding="utf-8")
    return trades_path, equity_path, daily_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-driven intraday backtest.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    trades_path, equity_path, daily_path, report_path = run(args.config)
    print(f"Trades written to: {trades_path}")
    print(f"Equity curve written to: {equity_path}")
    print(f"Daily PnL written to: {daily_path}")
    print(f"Backtest report written to: {report_path}")


if __name__ == "__main__":
    main()

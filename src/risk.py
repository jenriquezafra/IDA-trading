from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DailyRiskState:
    session: str | None = None
    trades: int = 0
    realized_pnl: float = 0.0
    cooldown_until_bar: int = -1
    kill_switch: bool = False


def base_position(signal: int) -> int:
    if signal > 0:
        return 1
    if signal < 0:
        return -1
    return 0


def position_size(row: pd.Series, config: dict[str, Any]) -> float:
    risk_cfg = config.get("risk", {})
    size = float(risk_cfg.get("fixed_size", 1.0))
    if risk_cfg.get("volatility_scaling", False):
        sigma_h = row.get("sigma_h", np.nan)
        target_sigma = float(risk_cfg.get("target_sigma_h", 0.003))
        if pd.notna(sigma_h) and sigma_h > 0:
            size *= target_sigma / float(sigma_h)
    return min(size, float(risk_cfg.get("max_leverage", 1.0)))


def desired_position(row: pd.Series, config: dict[str, Any]) -> float:
    return base_position(int(row.get("signal", 0))) * position_size(row, config)


def reset_daily_state_if_needed(state: DailyRiskState, session: str) -> None:
    if state.session != session:
        state.session = session
        state.trades = 0
        state.realized_pnl = 0.0
        state.cooldown_until_bar = -1
        state.kill_switch = False


def can_open_trade(row: pd.Series, state: DailyRiskState, config: dict[str, Any]) -> bool:
    reset_daily_state_if_needed(state, str(row["session"]))
    risk_cfg = config.get("risk", {})
    backtest_cfg = config.get("backtest", {})

    if state.kill_switch:
        return False
    if state.trades >= int(risk_cfg.get("max_trades_per_day", 999999)):
        return False
    if state.realized_pnl <= -float(risk_cfg.get("max_daily_loss", np.inf)):
        return False
    if int(row["bar_index"]) < state.cooldown_until_bar:
        return False

    timestamp = row.get("timestamp")
    if pd.notna(timestamp):
        clock = timestamp.strftime("%H:%M")
        if clock >= backtest_cfg.get("no_new_trades_after", "15:45"):
            return False
    return True


def register_trade_open(row: pd.Series, state: DailyRiskState, config: dict[str, Any]) -> None:
    reset_daily_state_if_needed(state, str(row["session"]))
    state.trades += 1
    cooldown = int(config.get("risk", {}).get("cooldown_bars", 0))
    state.cooldown_until_bar = int(row["bar_index"]) + cooldown


def register_trade_close(net_pnl: float, state: DailyRiskState, config: dict[str, Any]) -> None:
    state.realized_pnl += net_pnl
    kill_loss = float(config.get("risk", {}).get("kill_switch_daily_loss", np.inf))
    if state.realized_pnl <= -kill_loss:
        state.kill_switch = True


def should_force_flat(row: pd.Series, config: dict[str, Any]) -> bool:
    timestamp = row.get("timestamp")
    if pd.isna(timestamp):
        return False
    return timestamp.strftime("%H:%M") >= config.get("backtest", {}).get("force_flat_before", "15:55")


def stop_price(entry_px: float, position: float, config: dict[str, Any]) -> float:
    stop_loss = float(config.get("risk", {}).get("stop_loss_bps", 0.0)) / 10_000.0
    if position > 0:
        return entry_px * (1.0 - stop_loss)
    if position < 0:
        return entry_px * (1.0 + stop_loss)
    return entry_px


def check_stop(row: pd.Series, entry_px: float, position: float, config: dict[str, Any]) -> tuple[bool, float | None]:
    stop_px = stop_price(entry_px, position, config)
    if position > 0 and row["low"] <= stop_px:
        return True, stop_px
    if position < 0 and row["high"] >= stop_px:
        return True, stop_px
    return False, None

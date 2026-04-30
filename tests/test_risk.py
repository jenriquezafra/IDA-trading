from __future__ import annotations

import pandas as pd
import pytest

from src.risk import DailyRiskState, can_open_trade, check_stop, desired_position, register_trade_close, register_trade_open, should_force_flat


def _config() -> dict:
    return {
        "risk": {
            "fixed_size": 1.0,
            "volatility_scaling": True,
            "target_sigma_h": 0.003,
            "max_leverage": 0.5,
            "stop_loss_bps": 10.0,
            "max_daily_loss": 0.02,
            "max_trades_per_day": 1,
            "cooldown_bars": 2,
            "kill_switch_daily_loss": 0.03,
        },
        "backtest": {"no_new_trades_after": "15:45", "force_flat_before": "15:55"},
    }


def _row(bar_index: int = 10, clock: str = "10:30", signal: int = 1) -> pd.Series:
    return pd.Series(
        {
            "timestamp": pd.Timestamp(f"2024-01-02 {clock}", tz="America/New_York"),
            "session": "2024-01-02",
            "bar_index": bar_index,
            "signal": signal,
            "sigma_h": 0.006,
            "high": 101.0,
            "low": 99.0,
        }
    )


def test_desired_position_applies_fixed_size_vol_scaling_and_max_leverage() -> None:
    assert desired_position(_row(signal=1), _config()) == pytest.approx(0.5)
    assert desired_position(_row(signal=-1), _config()) == pytest.approx(-0.5)
    assert desired_position(_row(signal=0), _config()) == 0


def test_can_open_trade_enforces_daily_limits_cooldown_and_time() -> None:
    state = DailyRiskState()
    config = _config()
    row = _row(bar_index=10)

    assert can_open_trade(row, state, config)
    register_trade_open(row, state, config)
    assert not can_open_trade(_row(bar_index=11), state, config)
    assert not can_open_trade(_row(bar_index=12), state, config)
    assert not can_open_trade(_row(bar_index=20, clock="15:45"), DailyRiskState(), config)


def test_kill_switch_and_force_flat() -> None:
    state = DailyRiskState(session="2024-01-02")
    config = _config()

    register_trade_close(-0.04, state, config)

    assert state.kill_switch
    assert not can_open_trade(_row(), state, config)
    assert should_force_flat(_row(clock="15:55"), config)


def test_check_stop_for_long_and_short() -> None:
    config = _config()

    assert check_stop(_row(), entry_px=100.0, position=1.0, config=config) == (True, 99.9)
    assert check_stop(_row(), entry_px=100.0, position=-1.0, config=config) == (True, 100.1)

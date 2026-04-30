from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import prepare_backtest_frame, run_event_backtest


def _config(stop_loss_bps: float = 1000.0, time_stop_bars: int = 2) -> dict:
    return {
        "labeling": {"horizon_bars": 2},
        "backtest": {
            "cost_scenario": "base",
            "no_new_trades_after": "15:45",
            "force_flat_before": "15:55",
        },
        "costs": {
            "base": {
                "commission_bps_per_side": 0.0,
                "spread_bps_per_side": 0.25,
                "slippage_bps_per_side": 0.25,
                "impact_bps_per_1pct_participation": 0.0,
            }
        },
        "risk": {
            "fixed_size": 1.0,
            "volatility_scaling": False,
            "max_leverage": 1.0,
            "stop_loss_bps": stop_loss_bps,
            "time_stop_bars": time_stop_bars,
            "max_daily_loss": 1.0,
            "max_trades_per_day": 10,
            "cooldown_bars": 0,
            "kill_switch_daily_loss": 1.0,
        },
    }


def _cleaned() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": 1000,
            "session": "2024-01-02",
            "bar_index": list(range(6)),
        }
    )


def _signals(signal_bar: int = 1, signal: int = 1) -> pd.DataFrame:
    rows = []
    for idx, row in _cleaned().iterrows():
        rows.append(
            {
                "timestamp": row["timestamp"],
                "session": row["session"],
                "bar_index": row["bar_index"],
                "signal": signal if idx == signal_bar else 0,
                "split": "test",
                "score": 1.0,
                "p_up": 0.8,
                "p_down": 0.1,
                "p_neutral": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_backtest_frame_merges_signals_and_fills_flat() -> None:
    frame = prepare_backtest_frame(_cleaned(), _signals())

    assert frame["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert frame.loc[0, "split"] == "test"


def test_backtest_enters_next_open_and_exits_by_time_with_costs() -> None:
    trades, equity, daily = run_event_backtest(_cleaned(), _signals(signal_bar=1, signal=1), _config())

    trade = trades.iloc[0]
    assert trade["signal_timestamp"] == _cleaned().loc[1, "timestamp"]
    assert trade["entry_timestamp"] == _cleaned().loc[2, "timestamp"]
    assert trade["entry_px"] == 102
    assert trade["exit_bar_index"] == 4
    assert trade["exit_px"] == 104
    assert trade["exit_reason"] == "time"
    assert trade["gross_ret"] == pytest.approx((104 / 102) - 1)
    assert trade["total_cost_ret"] == pytest.approx(0.0001)
    assert trade["net_ret"] == pytest.approx(((104 / 102) - 1) - 0.0001)
    assert equity["equity"].iloc[-1] == pytest.approx(trade["net_ret"])
    assert daily["net_ret"].iloc[0] == pytest.approx(trade["net_ret"])


def test_backtest_stop_loss_exits_before_time() -> None:
    cleaned = _cleaned()
    cleaned.loc[2, "low"] = 100.0

    trades, _, _ = run_event_backtest(cleaned, _signals(signal_bar=1, signal=1), _config(stop_loss_bps=50.0))

    trade = trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_px"] == pytest.approx(102 * (1 - 0.005))


def test_backtest_does_not_open_overnight() -> None:
    cleaned = _cleaned()
    next_session = _cleaned()
    next_session["timestamp"] = pd.date_range("2024-01-03 09:30", periods=6, freq="5min", tz="America/New_York")
    next_session["session"] = "2024-01-03"
    cleaned = pd.concat([cleaned, next_session], ignore_index=True)
    signals = _signals(signal_bar=5, signal=1)
    signals = pd.concat([signals, _signals(signal_bar=99, signal=0).assign(timestamp=next_session["timestamp"], session="2024-01-03")], ignore_index=True)

    trades, _, _ = run_event_backtest(cleaned, signals, _config())

    assert trades.empty

from __future__ import annotations

import pandas as pd
import pytest

from src.resample_aligned_panel import panel_symbols, resample_factor, resample_panel


def _panel() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="5min", tz="America/New_York")
    frame = pd.DataFrame({"timestamp": timestamps, "session": ["2024-01-02"] * 6, "bar_index": list(range(6))})
    for symbol, offset in [("SPY", 0.0), ("QQQ", 10.0)]:
        frame[f"{symbol}__open"] = [100.0 + offset + idx for idx in range(6)]
        frame[f"{symbol}__high"] = [101.0 + offset + idx for idx in range(6)]
        frame[f"{symbol}__low"] = [99.0 + offset + idx for idx in range(6)]
        frame[f"{symbol}__close"] = [100.5 + offset + idx for idx in range(6)]
        frame[f"{symbol}__volume"] = [1000 + idx for idx in range(6)]
        frame[f"is_available_{symbol}"] = True
    frame["SPY__target_open_next"] = frame.groupby("session")["SPY__open"].shift(-1)
    frame["SPY__next_open_timestamp"] = frame.groupby("session")["timestamp"].shift(-1)
    frame["SPY__target_crosses_session_close"] = False
    frame["SPY__can_open_trade"] = True
    frame["SPY__force_flat_bar"] = False
    frame["SPY__trade_could_remain_open_past_close"] = False
    return frame


def test_resample_factor_requires_integer_multiple() -> None:
    assert resample_factor("5min", "15min") == 3
    with pytest.raises(ValueError):
        resample_factor("5min", "7min")


def test_panel_symbols_reads_ohlcv_prefixes() -> None:
    assert panel_symbols(_panel()) == ["SPY", "QQQ"]


def test_resample_panel_aggregates_ohlcv_and_rebuilds_target_helpers() -> None:
    resampled, report = resample_panel(_panel(), "SPY", "5min", "15min")

    assert report.factor == 3
    assert len(resampled) == 2
    first = resampled.iloc[0]
    assert first["bar_index"] == 0
    assert first["timestamp"] == pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    assert first["SPY__open"] == 100.0
    assert first["SPY__high"] == 103.0
    assert first["SPY__low"] == 99.0
    assert first["SPY__close"] == 102.5
    assert first["SPY__volume"] == 3003
    assert first["SPY__target_open_next"] == 103.0
    assert first["SPY__can_open_trade"]

    last = resampled.iloc[-1]
    assert pd.isna(last["SPY__target_open_next"])
    assert last["SPY__target_crosses_session_close"]
    assert not last["SPY__can_open_trade"]

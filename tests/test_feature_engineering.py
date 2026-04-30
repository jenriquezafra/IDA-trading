from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import build_features


def _config() -> dict:
    return {
        "project": {"frequency": "5min"},
        "features": {
            "return_windows": [1, 2, 3, 6, 12],
            "realized_vol_windows": [3, 6, 12, 24],
            "atr_windows": [6, 12],
            "sma_windows": [6, 12, 24],
            "trend_windows": [6, 12, 24],
            "open_window_bars": 6,
            "close_window_bars": 6,
            "midday_start": "12:00",
            "midday_end": "14:00",
        },
    }


def _session_frame(session: str, close_start: float, volume: int) -> pd.DataFrame:
    timestamps = pd.date_range(f"{session} 09:30", periods=78, freq="5min", tz="America/New_York")
    close = close_start + np.arange(78, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
            "session": session,
            "bar_index": np.arange(78),
            "bars_in_session": 78,
        }
    )


def test_build_features_creates_expected_columns() -> None:
    df = _session_frame("2024-01-02", 100.0, 1000)

    features = build_features(df, _config())

    expected_columns = {
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_6",
        "ret_12",
        "rv_3",
        "rv_6",
        "rv_12",
        "rv_24",
        "range",
        "atr_6",
        "atr_12",
        "sma_6",
        "sma_12",
        "sma_24",
        "trend_6",
        "trend_12",
        "trend_24",
        "vwap",
        "dist_vwap",
        "intraday_drawdown",
        "rel_volume",
        "sin_time",
        "cos_time",
        "minutes_to_close",
        "open_window",
        "close_window",
        "midday",
    }
    assert expected_columns.issubset(features.columns)


def test_returns_and_rolling_features_do_not_cross_sessions() -> None:
    df = pd.concat(
        [
            _session_frame("2024-01-02", 100.0, 1000),
            _session_frame("2024-01-03", 200.0, 2000),
        ],
        ignore_index=True,
    )

    features = build_features(df, _config())
    second_session = features[features["session"] == "2024-01-03"].reset_index(drop=True)

    assert np.isnan(second_session.loc[0, "ret_1"])
    assert np.isnan(second_session.loc[0, "sma_6"])
    assert np.isnan(second_session.loc[0, "rv_3"])
    assert second_session.loc[5, "sma_6"] == np.mean([200, 201, 202, 203, 204, 205])


def test_relative_volume_uses_prior_sessions_only() -> None:
    df = pd.concat(
        [
            _session_frame("2024-01-02", 100.0, 1000),
            _session_frame("2024-01-03", 200.0, 2000),
            _session_frame("2024-01-04", 300.0, 6000),
        ],
        ignore_index=True,
    )

    features = build_features(df, _config())
    first_rows = features[features["bar_index"] == 0].reset_index(drop=True)

    assert np.isnan(first_rows.loc[0, "rel_volume"])
    assert first_rows.loc[1, "rel_volume"] == 2.0
    assert first_rows.loc[2, "rel_volume"] == 4.0


def test_time_features_match_session_boundaries() -> None:
    df = _session_frame("2024-01-02", 100.0, 1000)

    features = build_features(df, _config())

    assert features.loc[0, "minutes_to_close"] == 385.0
    assert features.loc[77, "minutes_to_close"] == 0.0
    assert features.loc[:5, "open_window"].all()
    assert not features.loc[6, "open_window"]
    assert features.loc[72:, "close_window"].all()
    assert not features.loc[71, "close_window"]

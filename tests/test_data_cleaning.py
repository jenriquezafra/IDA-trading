from __future__ import annotations

import pandas as pd

from src.data_cleaning import clean_ohlcv


def _config(drop_incomplete_sessions: bool = True) -> dict:
    return {
        "project": {"timezone": "America/New_York"},
        "data": {"timestamp_col": "timestamp"},
        "session": {
            "market_open": "09:30",
            "market_close": "16:00",
            "timestamp_label": "start",
            "regular_session_only": True,
            "drop_incomplete_sessions": drop_incomplete_sessions,
            "expected_bars_per_session": 78,
        },
        "quality": {
            "drop_duplicate_timestamps": True,
            "drop_critical_nan_rows": True,
            "drop_invalid_price_rows": True,
            "drop_negative_volume_rows": True,
            "drop_extreme_range_rows": True,
            "max_bar_range_bps": 1000.0,
        },
    }


def _calendar_config() -> dict:
    config = _config(drop_incomplete_sessions=True)
    config.update(
        {
            "project": {"timezone": "America/New_York", "frequency": "5min"},
            "calendar": {"enabled": True, "name": "NYSE", "drop_non_trading_days": True, "drop_half_days": True},
            "labeling": {"horizon_bars": 2},
            "backtest": {"no_new_trades_after": "15:45", "force_flat_before": "15:55"},
        }
    )
    return config


def _session_frame(session: str, bars: int = 78) -> pd.DataFrame:
    timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )


def test_clean_ohlcv_adds_session_and_bar_index_for_regular_session() -> None:
    df = _session_frame("2024-01-02")

    cleaned, report = clean_ohlcv(df, _config(), input_path="raw.parquet", output_path="clean.parquet")

    assert len(cleaned) == 78
    assert cleaned["timestamp"].dt.tz is not None
    assert cleaned["session"].nunique() == 1
    assert cleaned["bar_index"].tolist() == list(range(78))
    assert report.out_of_session_rows == 0


def test_clean_ohlcv_removes_duplicates_bad_rows_and_out_of_session_rows() -> None:
    regular = _session_frame("2024-01-02")
    dirty_rows = pd.DataFrame(
        {
                "timestamp": [
                    "2024-01-02 09:30",  # duplicate
                    "2024-01-02 08:00",  # out of session
                    "2024-01-02 10:02",  # invalid price
                    "2024-01-02 10:07",  # negative volume
                    "2024-01-02 10:12",  # nan
                    "2024-01-02 10:17",  # extreme range
                ],
            "open": [100.0, 100.0, 100.0, 100.0, None, 100.0],
            "high": [101.0, 101.0, 99.0, 101.0, 101.0, 130.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.5, 100.5, 100.5, 100.5, 100.5, 100.5],
            "volume": [1000, 1000, 1000, -1, 1000, 1000],
        }
    )
    df = pd.concat([regular, dirty_rows], ignore_index=True)

    cleaned, report = clean_ohlcv(df, _config(), input_path="raw.parquet", output_path="clean.parquet")

    assert len(cleaned) == 78
    assert report.duplicate_timestamps == 1
    assert report.invalid_price_rows == 1
    assert report.negative_volume_rows == 1
    assert report.critical_nan_rows == 1
    assert report.extreme_range_rows == 1
    assert report.out_of_session_rows == 1


def test_clean_ohlcv_drops_incomplete_sessions_when_configured() -> None:
    complete = _session_frame("2024-01-02")
    incomplete = _session_frame("2024-01-03", bars=77)
    df = pd.concat([complete, incomplete], ignore_index=True)

    cleaned, report = clean_ohlcv(df, _config(drop_incomplete_sessions=True), input_path="raw.parquet", output_path="clean.parquet")

    assert len(cleaned) == 78
    assert cleaned["session"].unique().tolist() == ["2024-01-02"]
    assert report.incomplete_sessions == ["2024-01-03"]
    assert report.dropped_incomplete_rows == 77


def test_clean_ohlcv_uses_nyse_calendar_for_holidays_and_half_days() -> None:
    regular = _session_frame("2024-11-27")
    holiday = _session_frame("2024-11-28")
    half_day = _session_frame("2024-11-29")
    df = pd.concat([regular, holiday, half_day], ignore_index=True)

    cleaned, report = clean_ohlcv(df, _calendar_config(), input_path="raw.parquet", output_path="clean.parquet")

    assert cleaned["session"].unique().tolist() == ["2024-11-27"]
    assert report.non_trading_session_rows == 78
    assert report.non_trading_sessions == ["2024-11-28"]
    assert report.half_day_sessions == ["2024-11-29"]
    assert report.dropped_half_day_rows == 42


def test_clean_ohlcv_adds_execution_safety_columns() -> None:
    df = _session_frame("2024-01-02")

    cleaned, report = clean_ohlcv(df, _calendar_config(), input_path="raw.parquet", output_path="clean.parquet")

    assert cleaned["target_crosses_session_close"].sum() == 3
    assert (~cleaned["can_open_trade"]).sum() == 3
    assert cleaned["force_flat_bar"].sum() == 1
    assert report.target_crosses_session_close_rows == 3
    assert report.cannot_open_trade_rows == 3
    assert report.force_flat_rows == 1

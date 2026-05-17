from __future__ import annotations

import pandas as pd

from src import data_download


def _ts_ms(value: str) -> int:
    return int(pd.Timestamp(value).value // 1_000_000)


def test_resample_ohlcv_minutes_builds_left_labeled_5min_bars() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=5, freq="1min", tz="America/New_York"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    result = data_download._resample_ohlcv_minutes(frame, "5m")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 99.0
    assert row["close"] == 104.5
    assert row["volume"] == 150


def test_download_polygon_ohlcv_normalizes_and_saves_source(monkeypatch, tmp_path) -> None:
    calls = []
    payload = {
        "status": "OK",
        "results": [
            {"t": _ts_ms("2024-01-02 14:30:00Z"), "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 10},
            {"t": _ts_ms("2024-01-02 14:31:00Z"), "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 20},
            {"t": _ts_ms("2024-01-02 14:32:00Z"), "o": 101.5, "h": 103.0, "l": 101.0, "c": 102.5, "v": 30},
            {"t": _ts_ms("2024-01-02 14:33:00Z"), "o": 102.5, "h": 104.0, "l": 102.0, "c": 103.5, "v": 40},
            {"t": _ts_ms("2024-01-02 14:34:00Z"), "o": 103.5, "h": 105.0, "l": 103.0, "c": 104.5, "v": 50},
        ],
    }

    def fake_fetch(url, params):
        calls.append((url, params))
        return payload

    monkeypatch.setattr(data_download, "_fetch_polygon_json", fake_fetch)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    raw_source_file = tmp_path / "spy_1min.parquet"
    config = {
        "project": {"timezone": "America/New_York"},
        "data": {
            "symbol": "SPY",
            "download_interval": "5m",
            "start_date": "2024-01-02",
            "end_date": "2024-01-02",
            "polygon": {
                "api_key_env": "POLYGON_API_KEY",
                "source_interval": "1m",
                "raw_source_file": str(raw_source_file),
                "adjusted": True,
            },
        },
    }

    result = data_download.download_polygon_ohlcv(config)

    assert len(calls) == 1
    assert "/SPY/range/1/minute/2024-01-02/2024-01-02" in calls[0][0]
    assert calls[0][1]["apiKey"] == "test-key"
    assert raw_source_file.exists()
    assert len(pd.read_parquet(raw_source_file)) == 5
    assert len(result) == 1
    assert result.iloc[0]["timestamp"] == pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    assert result.iloc[0]["volume"] == 150

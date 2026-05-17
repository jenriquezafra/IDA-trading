from __future__ import annotations

import pandas as pd
import pytest

from src.data.cboe_risk_context import build_risk_context, parse_daily_options_stats, parse_volatility_index_csv


def test_parse_volatility_index_csv_handles_ohlc_and_single_value_formats() -> None:
    vix = parse_volatility_index_csv("VIX", "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,13,14,12,13.5\n")
    vvix = parse_volatility_index_csv("VVIX", "DATE,VVIX\n01/02/2024,88.2\n")

    assert vix.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert vix.loc[0, "vix_close"] == 13.5
    assert vvix.loc[0, "vvix_close"] == 88.2


def test_parse_daily_options_stats_extracts_ratios_and_category_volume() -> None:
    payload = {
        "ratios": [
            {"name": "TOTAL PUT/CALL RATIO", "value": "0.91"},
            {"name": "SPX + SPXW PUT/CALL RATIO", "value": "1.23"},
        ],
        "INDEX OPTIONS": [
            {"name": "VOLUME", "call": 10, "put": 20, "total": 30},
            {"name": "OPEN INTEREST", "call": 100, "put": 200, "total": 300},
        ],
    }

    row = parse_daily_options_stats("2024-01-02", payload).iloc[0]

    assert row["total_put_call_ratio"] == pytest.approx(0.91)
    assert row["spx_spxw_put_call_ratio"] == pytest.approx(1.23)
    assert row["index_options_volume_put"] == 20
    assert row["index_options_open_interest_total"] == 300


def test_build_risk_context_lags_daily_data_to_next_session() -> None:
    volatility = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
            "vix_close": [13.0, 14.0, 15.0, 16.0, 18.0],
            "vix9d_close": [12.0, 13.0, 14.0, 15.0, 19.0],
            "vix3m_close": [16.0, 16.0, 17.0, 18.0, 20.0],
        }
    )
    put_call = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
            "total_put_call_ratio": [0.8, 0.9, 1.0, 1.1, 1.2],
        }
    )

    context = build_risk_context(volatility, put_call, zscore_window=5)

    assert context.loc[0, "source_date"] == pd.Timestamp("2024-01-02")
    assert context.loc[0, "available_session"] == pd.Timestamp("2024-01-03")
    assert context.loc[0, "prev_vix_close"] == 13.0
    assert "prev_vix9d_vix_ratio" in context.columns
    assert "prev_total_put_call_ratio_z5" in context.columns

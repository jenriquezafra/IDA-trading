from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.earnings_events_h3 import (
    add_gap_features,
    add_exclusion_features,
    add_opening_range_features,
    add_sector_peer_features,
    add_volume_features,
    build_earnings_events,
    normalize_benzinga_earnings,
    split_events_by_timing,
)


def _config() -> dict:
    return {
        "data": {
            "timestamp_timezone": "America/New_York",
            "calendar": "XNYS",
            "consensus_revision_policy": {"rule": "hard_reject_without_pre_event_snapshot"},
        }
    }


def _config_with_surprise_min_history(min_history: int) -> dict:
    cfg = _config()
    cfg["features"] = {
        "fundamental": {
            "surprise_zscore": {
                "min_history": min_history,
                "eps_consensus_abs_floor": 0.01,
                "revenue_consensus_abs_floor": 1.0,
            }
        }
    }
    return cfg


def _config_with_gap_lookback(lookback: int) -> dict:
    cfg = _config()
    cfg["features"] = {"opening_reaction": {"gap_atr": {"atr_lookback_sessions": lookback, "atr_min_history_sessions": lookback}}}
    return cfg


def _config_with_rel_volume_lookback(lookback: int) -> dict:
    cfg = _config()
    cfg["timeframe"] = "5min"
    cfg["features"] = {"opening_reaction": {"rel_volume_30m": {"lookback_sessions": lookback, "min_history_sessions": lookback}}}
    return cfg


def _opening_volume_panel(session_totals: dict[str, float], symbol: str = "AAPL", bars: int = 6) -> pd.DataFrame:
    rows = []
    for session, total_volume in session_totals.items():
        timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="5min", tz="America/New_York")
        for bar_index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "session": session,
                    "bar_index": bar_index,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": total_volume / bars,
                }
            )
    return pd.DataFrame(rows)


def _opening_price_panel(symbol: str, session: str, open_price: float, close_price: float, bars: int = 6) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="5min", tz="America/New_York")
    for bar_index, timestamp in enumerate(timestamps):
        close = open_price + (close_price - open_price) * bar_index / max(bars - 1, 1)
        rows.append(
            {
                "timestamp": timestamp,
                "session": session,
                "bar_index": bar_index,
                "symbol": symbol,
                "open": open_price if bar_index == 0 else close,
                "high": close + 0.25,
                "low": close - 0.25,
                "close": close,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


def _event_rows_for(symbol: str, date: str, event_id: str = "event") -> list[dict]:
    return [
        {
            "benzinga_id": event_id,
            "ticker": symbol,
            "date": date,
            "time": "07:00:00",
            "last_updated": f"{date}T11:00:00Z",
            "actual_eps": 2.0,
            "estimated_eps": 1.9,
            "actual_revenue": 100.0,
            "estimated_revenue": 99.0,
        }
    ]


def test_normalize_benzinga_earnings_classifies_report_session_and_entry() -> None:
    rows = [
        {
            "benzinga_id": "aapl-q1",
            "ticker": "AAPL",
            "date": "2024-01-25",
            "time": "07:00:00",
            "last_updated": "2024-01-25T11:55:00Z",
            "actual_eps": 2.0,
            "estimated_eps": 1.9,
            "actual_revenue": 100.0,
            "estimated_revenue": 99.0,
        },
        {
            "benzinga_id": "msft-q1",
            "ticker": "MSFT",
            "date": "2024-01-25",
            "time": "16:05:00",
            "last_updated": "2024-01-25T20:55:00Z",
            "actual_eps": 3.0,
            "estimated_eps": 2.9,
            "actual_revenue": 200.0,
            "estimated_revenue": 199.0,
        },
    ]

    frame = normalize_benzinga_earnings(rows, _config())

    pre = frame.loc[frame["symbol"].eq("AAPL")].iloc[0]
    after = frame.loc[frame["symbol"].eq("MSFT")].iloc[0]
    assert pre["report_timing"] == "pre_market"
    assert pre["event_session"] == "2024-01-25"
    assert pre["entry_timestamp"] == pd.Timestamp("2024-01-25 15:00:00Z")
    assert bool(pre["is_full_regular_session"])
    assert bool(pre["is_tradeable_v1"])
    assert after["report_timing"] == "after_market"
    assert after["event_session"] == "2024-01-26"
    assert after["entry_timestamp"] == pd.Timestamp("2024-01-26 15:00:00Z")
    assert bool(after["is_full_regular_session"])
    assert bool(after["is_tradeable_v1"])


def test_normalize_benzinga_earnings_hard_rejects_post_event_consensus_update() -> None:
    frame = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "nvda-q1",
                "ticker": "NVDA",
                "date": "2024-02-21",
                "time": "16:05:00",
                "last_updated": "2024-02-22T00:00:00Z",
                "actual_eps": 5.16,
                "estimated_eps": 4.64,
                "actual_revenue": 22.1,
                "estimated_revenue": 20.6,
            }
        ],
        _config(),
    )

    row = frame.iloc[0]
    assert not bool(row["consensus_snapshot_is_pre_event"])
    assert not bool(row["is_tradeable_v1"])
    assert "post_event_revised_consensus_without_snapshot" in row["exclusion_flags"]


def test_normalize_benzinga_earnings_computes_surprise_zscores_without_future_events() -> None:
    rows = [
        {
            "benzinga_id": "aapl-q1",
            "ticker": "AAPL",
            "date": "2024-01-02",
            "time": "07:00:00",
            "last_updated": "2024-01-02T11:00:00Z",
            "actual_eps": 1.1,
            "estimated_eps": 1.0,
            "actual_revenue": 105.0,
            "estimated_revenue": 100.0,
        },
        {
            "benzinga_id": "msft-q1",
            "ticker": "MSFT",
            "date": "2024-01-03",
            "time": "07:00:00",
            "last_updated": "2024-01-03T11:00:00Z",
            "actual_eps": 1.4,
            "estimated_eps": 1.0,
            "actual_revenue": 130.0,
            "estimated_revenue": 100.0,
        },
        {
            "benzinga_id": "nvda-q1",
            "ticker": "NVDA",
            "date": "2024-01-04",
            "time": "07:00:00",
            "last_updated": "2024-01-04T11:00:00Z",
            "actual_eps": 1.7,
            "estimated_eps": 1.0,
            "actual_revenue": 155.0,
            "estimated_revenue": 100.0,
        },
        {
            "benzinga_id": "jpm-q1",
            "ticker": "JPM",
            "date": "2024-01-05",
            "time": "07:00:00",
            "last_updated": "2024-01-05T11:00:00Z",
            "actual_eps": 2.0,
            "estimated_eps": 1.0,
            "actual_revenue": 180.0,
        },
    ]

    frame = normalize_benzinga_earnings(rows, _config_with_surprise_min_history(2))

    third = frame.loc[frame["symbol"].eq("NVDA")].iloc[0]
    missing = frame.loc[frame["symbol"].eq("JPM")].iloc[0]
    assert third["eps_surprise_pct"] == pytest.approx(0.7)
    assert third["revenue_surprise_pct"] == pytest.approx(0.55)
    assert third["eps_surprise_z"] == pytest.approx(3.0)
    assert third["revenue_surprise_z"] == pytest.approx(3.0)
    assert pd.isna(frame.loc[frame["symbol"].eq("AAPL"), "eps_surprise_z"].iloc[0])
    assert pd.isna(frame.loc[frame["symbol"].eq("MSFT"), "eps_surprise_z"].iloc[0])
    assert bool(missing["missing_revenue_consensus"])
    assert "missing_pre_event_revenue_consensus_snapshot" in missing["exclusion_flags"]


def test_add_gap_features_uses_prior_atr_and_event_session_open() -> None:
    events = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-gap",
                "ticker": "AAPL",
                "date": "2024-01-05",
                "time": "07:00:00",
                "last_updated": "2024-01-05T11:00:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            }
        ],
        _config_with_gap_lookback(2),
    )
    panel = pd.DataFrame(
        [
            {"symbol": "AAPL", "session": "2024-01-02", "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"symbol": "AAPL", "session": "2024-01-03", "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"symbol": "AAPL", "session": "2024-01-04", "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"symbol": "AAPL", "session": "2024-01-05", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5},
        ]
    )

    enriched = add_gap_features(events, panel, _config_with_gap_lookback(2))
    row = enriched.iloc[0]

    assert row["gap_open"] == pytest.approx(102.0)
    assert row["gap_prev_close"] == pytest.approx(100.0)
    assert row["gap_return"] == pytest.approx(0.02)
    assert row["recent_atr_return"] == pytest.approx(0.04)
    assert row["gap_atr"] == pytest.approx(0.5)
    assert not bool(row["missing_gap_open"])
    assert not bool(row["missing_recent_atr"])
    assert bool(row["is_tradeable_v1"])


def test_add_gap_features_flags_missing_price_history() -> None:
    events = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-gap",
                "ticker": "AAPL",
                "date": "2024-01-05",
                "time": "07:00:00",
                "last_updated": "2024-01-05T11:00:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            }
        ],
        _config_with_gap_lookback(2),
    )
    panel = pd.DataFrame(
        [
            {"symbol": "MSFT", "session": "2024-01-05", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5},
        ]
    )

    enriched = add_gap_features(events, panel, _config_with_gap_lookback(2))
    row = enriched.iloc[0]

    assert bool(row["missing_gap_open"])
    assert bool(row["missing_gap_prev_close"])
    assert bool(row["missing_recent_atr"])
    assert not bool(row["is_tradeable_v1"])
    assert "missing_gap_open" in row["exclusion_flags"]


def test_add_volume_features_uses_prior_opening_volume_median() -> None:
    events = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-volume",
                "ticker": "AAPL",
                "date": "2024-01-05",
                "time": "07:00:00",
                "last_updated": "2024-01-05T11:00:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            }
        ],
        _config_with_rel_volume_lookback(2),
    )
    panel = _opening_volume_panel(
        {
            "2024-01-02": 600.0,
            "2024-01-03": 1200.0,
            "2024-01-04": 1800.0,
            "2024-01-05": 3000.0,
        }
    )

    enriched = add_volume_features(events, panel, _config_with_rel_volume_lookback(2))
    row = enriched.iloc[0]

    assert row["volume_30m"] == pytest.approx(3000.0)
    assert row["expected_volume_30m"] == pytest.approx(1500.0)
    assert row["opening_30m_bar_count"] == pytest.approx(6.0)
    assert row["rel_volume_30m"] == pytest.approx(2.0)
    assert not bool(row["missing_volume_30m"])
    assert not bool(row["missing_expected_volume_30m"])
    assert not bool(row["insufficient_opening_30m_bars"])
    assert bool(row["is_tradeable_v1"])


def test_add_volume_features_flags_missing_or_incomplete_opening_volume() -> None:
    events = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-volume",
                "ticker": "AAPL",
                "date": "2024-01-05",
                "time": "07:00:00",
                "last_updated": "2024-01-05T11:00:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            }
        ],
        _config_with_rel_volume_lookback(2),
    )
    panel = _opening_volume_panel({"2024-01-05": 3000.0}, bars=5)

    enriched = add_volume_features(events, panel, _config_with_rel_volume_lookback(2))
    row = enriched.iloc[0]

    assert bool(row["missing_expected_volume_30m"])
    assert bool(row["insufficient_opening_30m_bars"])
    assert not bool(row["is_tradeable_v1"])
    assert "missing_expected_volume_30m" in row["exclusion_flags"]
    assert "insufficient_opening_30m_bars" in row["exclusion_flags"]


def test_add_opening_range_features_computes_vwap_range_and_close() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-range"), _config())
    rows = []
    for bar_index, timestamp in enumerate(pd.date_range("2024-01-05 09:30", periods=6, freq="5min", tz="America/New_York")):
        price = 100.0 + bar_index
        rows.append(
            {
                "timestamp": timestamp,
                "session": "2024-01-05",
                "bar_index": bar_index,
                "symbol": "AAPL",
                "open": price - 0.25,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 10.0,
                "bar_vwap": price + 0.5,
            }
        )
    panel = pd.DataFrame(rows)

    enriched = add_opening_range_features(events, panel, _config())
    row = enriched.iloc[0]

    assert row["vwap_30m"] == pytest.approx(103.0)
    assert row["range_high_30m"] == pytest.approx(106.0)
    assert row["range_low_30m"] == pytest.approx(99.0)
    assert row["close_30m"] == pytest.approx(105.0)
    assert not bool(row["missing_vwap_30m"])
    assert not bool(row["missing_range_30m"])
    assert not bool(row["missing_close_30m"])
    assert bool(row["is_tradeable_v1"])


def test_add_opening_range_features_flags_missing_vwap_and_incomplete_window() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-range"), _config())
    rows = []
    for bar_index, timestamp in enumerate(pd.date_range("2024-01-05 09:30", periods=5, freq="5min", tz="America/New_York")):
        rows.append(
            {
                "timestamp": timestamp,
                "session": "2024-01-05",
                "bar_index": bar_index,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        )
    panel = pd.DataFrame(rows)

    enriched = add_opening_range_features(events, panel, _config())
    row = enriched.iloc[0]

    assert bool(row["missing_vwap_30m"])
    assert not bool(row["missing_range_30m"])
    assert not bool(row["missing_close_30m"])
    assert bool(row["insufficient_opening_30m_bars"])
    assert not bool(row["is_tradeable_v1"])
    assert "missing_vwap_30m" in row["exclusion_flags"]
    assert "insufficient_opening_30m_bars" in row["exclusion_flags"]


def test_add_sector_peer_features_uses_sector_etf_return() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-sector"), _config())
    panel = _opening_price_panel("XLK", "2024-01-05", 100.0, 102.0)

    enriched = add_sector_peer_features(events, panel, _config())
    row = enriched.iloc[0]

    assert row["sector_id"] == "technology"
    assert row["sector_proxy"] == "XLK"
    assert row["peer_proxy_symbol"] == "XLK"
    assert row["sector_return_30m"] == pytest.approx(0.02)
    assert row["peer_proxy_return_30m"] == pytest.approx(0.02)
    assert not bool(row["peer_proxy_fallback_used"])
    assert bool(row["is_tradeable_v1"])


def test_add_sector_peer_features_falls_back_to_spy_when_sector_proxy_missing() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-sector"), _config())
    panel = _opening_price_panel("SPY", "2024-01-05", 100.0, 101.0)

    enriched = add_sector_peer_features(events, panel, _config())
    row = enriched.iloc[0]

    assert row["sector_id"] == "technology"
    assert row["sector_proxy"] == "XLK"
    assert row["peer_proxy_symbol"] == "SPY"
    assert row["sector_return_30m"] == pytest.approx(0.01)
    assert row["peer_proxy_return_30m"] == pytest.approx(0.01)
    assert bool(row["peer_proxy_fallback_used"])
    assert bool(row["is_tradeable_v1"])


def test_add_sector_peer_features_flags_missing_proxy_return() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-sector"), _config())
    panel = _opening_price_panel("XLF", "2024-01-05", 100.0, 101.0)

    enriched = add_sector_peer_features(events, panel, _config())
    row = enriched.iloc[0]

    assert bool(row["missing_sector_proxy_return_30m"])
    assert bool(row["missing_peer_proxy_return_30m"])
    assert not bool(row["is_tradeable_v1"])
    assert "missing_sector_proxy_return_30m" in row["exclusion_flags"]


def test_add_exclusion_features_flags_external_event_exclusions() -> None:
    events = normalize_benzinga_earnings(_event_rows_for("AAPL", "2024-01-05", "aapl-exclusions"), _config())
    macro_calendar = pd.DataFrame([{"session": "2024-01-05", "macro_dominant_session": True}])
    halt_events = pd.DataFrame([{"symbol": "AAPL", "session": "2024-01-05", "halt_flag": True}])
    quote_quality = pd.DataFrame([{"symbol": "AAPL", "session": "2024-01-05", "spread_bps_30m": 15.0}])
    binary_news = pd.DataFrame([{"symbol": "AAPL", "session": "2024-01-05", "binary_news_flag": True}])

    enriched = add_exclusion_features(
        events,
        _config(),
        macro_calendar=macro_calendar,
        halt_events=halt_events,
        quote_quality=quote_quality,
        binary_news=binary_news,
    )
    row = enriched.iloc[0]

    assert bool(row["macro_day_flag"])
    assert bool(row["halt_flag"])
    assert bool(row["suspected_halt_or_bad_session"])
    assert row["spread_bps_30m"] == pytest.approx(15.0)
    assert bool(row["high_spread_flag"])
    assert bool(row["binary_news_flag"])
    assert not bool(row["is_tradeable_v1"])
    for flag in ["macro_dominant_session", "trading_halt", "high_spread", "binary_news_event"]:
        assert flag in row["exclusion_flags"]


def test_add_exclusion_features_flags_simultaneous_sector_peer_earnings() -> None:
    events = normalize_benzinga_earnings(
        _event_rows_for("AAPL", "2024-01-05", "aapl-same-sector")
        + _event_rows_for("MSFT", "2024-01-05", "msft-same-sector"),
        _config(),
    )

    enriched = add_exclusion_features(events, _config())

    assert enriched["simultaneous_peer_earnings_flag"].tolist() == [True, True]
    assert not enriched["is_tradeable_v1"].any()
    assert enriched["exclusion_flags"].str.contains("simultaneous_core_peer_earnings").all()


def test_normalize_benzinga_earnings_excludes_non_full_regular_entry_session() -> None:
    frame = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-half-day",
                "ticker": "AAPL",
                "date": "2024-07-02",
                "time": "16:05:00",
                "last_updated": "2024-07-02T20:00:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            }
        ],
        _config(),
    )

    row = frame.iloc[0]
    assert row["event_session"] == "2024-07-03"
    assert row["entry_timestamp"] == pd.Timestamp("2024-07-03 14:00:00Z")
    assert row["exit_timestamp"] == pd.Timestamp("2024-07-03 17:00:00Z")
    assert not bool(row["is_full_regular_session"])
    assert not bool(row["is_tradeable_v1"])
    assert "non_full_regular_session" in row["exclusion_flags"]


def test_split_events_by_timing_returns_tradeable_buckets_and_exclusions() -> None:
    frame = normalize_benzinga_earnings(
        [
            {
                "benzinga_id": "aapl-q1",
                "ticker": "AAPL",
                "date": "2024-01-25",
                "time": "07:00:00",
                "last_updated": "2024-01-25T11:55:00Z",
                "actual_eps": 2.0,
                "estimated_eps": 1.9,
                "actual_revenue": 100.0,
                "estimated_revenue": 99.0,
            },
            {
                "benzinga_id": "msft-q1",
                "ticker": "MSFT",
                "date": "2024-01-25",
                "time": "16:05:00",
                "last_updated": "2024-01-25T20:55:00Z",
                "actual_eps": 3.0,
                "estimated_eps": 2.9,
                "actual_revenue": 200.0,
                "estimated_revenue": 199.0,
            },
            {
                "benzinga_id": "jpm-q1",
                "ticker": "JPM",
                "date": "2024-01-25",
                "time": "11:00:00",
                "last_updated": "2024-01-25T15:55:00Z",
                "actual_eps": 4.0,
                "estimated_eps": 3.9,
                "actual_revenue": 300.0,
                "estimated_revenue": 299.0,
            },
        ],
        _config(),
    )

    partitions = split_events_by_timing(frame)

    assert partitions["pre_market"]["symbol"].tolist() == ["AAPL"]
    assert partitions["after_market"]["symbol"].tolist() == ["MSFT"]
    assert partitions["excluded_timing"]["symbol"].tolist() == ["JPM"]
    assert partitions["excluded_timing"]["report_timing"].tolist() == ["during_session"]


def test_build_earnings_events_writes_parquet(tmp_path) -> None:
    raw_path = tmp_path / "earnings.json"
    output_path = tmp_path / "earnings_events.parquet"
    raw_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "benzinga_id": "xom-q1",
                        "ticker": "XOM",
                        "date": "2024-04-26",
                        "time": "06:30:00",
                        "last_updated": "2024-04-26T10:00:00Z",
                        "actual_eps": 2.1,
                        "estimated_eps": 2.0,
                        "actual_revenue": 90.0,
                        "estimated_revenue": 89.0,
                    },
                    {
                        "benzinga_id": "cvx-q1",
                        "ticker": "CVX",
                        "date": "2024-04-26",
                        "time": "16:05:00",
                        "last_updated": "2024-04-26T20:00:00Z",
                        "actual_eps": 3.1,
                        "estimated_eps": 3.0,
                        "actual_revenue": 80.0,
                        "estimated_revenue": 79.0,
                    },
                    {
                        "benzinga_id": "cop-q1",
                        "ticker": "COP",
                        "date": "2024-04-26",
                        "time": "12:00:00",
                        "last_updated": "2024-04-26T15:00:00Z",
                        "actual_eps": 2.6,
                        "estimated_eps": 2.5,
                        "actual_revenue": 70.0,
                        "estimated_revenue": 69.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_earnings_events(raw_path, output_path=output_path, config_path="configs/strategy/equity_earnings_continuation_h3_v1.yaml")

    assert result == output_path
    frame = pd.read_parquet(output_path)
    assert len(frame) == 3
    assert set(["event_id", "symbol", "report_timing", "exclusion_flags", "is_tradeable_v1"]).issubset(frame.columns)
    assert pd.read_parquet(tmp_path / "earnings_events_pre_market.parquet")["symbol"].tolist() == ["XOM"]
    assert pd.read_parquet(tmp_path / "earnings_events_after_market.parquet")["symbol"].tolist() == ["CVX"]
    assert pd.read_parquet(tmp_path / "earnings_events_excluded_timing.parquet")["symbol"].tolist() == ["COP"]

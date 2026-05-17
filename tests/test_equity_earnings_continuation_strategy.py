from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.strategy.equity_earnings_continuation import (
    PRIMARY_LABEL,
    SENSITIVITY_LABEL,
    run_strategy,
)


def _events() -> pd.DataFrame:
    rows = []
    for event_id, session, rel_volume, revenue_z in [
        ("train", "2026-01-05", 1.0, 0.5),
        ("validation_pass", "2026-02-05", 1.5, 0.5),
        ("validation_fail", "2026-02-05", 0.8, 0.5),
        ("test", "2026-03-05", 2.0, 0.5),
    ]:
        rows.append(
            {
                "event_id": f"evt_{event_id}",
                "symbol": "AAPL",
                "event_session": session,
                "report_timing": "pre_market",
                "sector_id": "technology",
                "sector_proxy": "XLK",
                "peer_proxy_symbol": "XLK",
                "eps_surprise": 0.20,
                "revenue_surprise": 100.0,
                "eps_surprise_z": 1.0,
                "revenue_surprise_z": revenue_z,
                "gap_atr": 1.0,
                "rel_volume_30m": rel_volume,
                "close_30m": 101.5,
                "vwap_30m": 101.0,
                "sector_return_30m": 0.001,
                "exclusion_flags": "",
                "is_tradeable_v1": True,
            }
        )
    return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-02-05",
        "2026-02-06",
        "2026-03-05",
        "2026-03-06",
    ]
    symbol_values = {
        "AAPL": (100.0, 101.0, 105.0),
        "XLK": (50.0, 50.5, 51.0),
        "SPY": (400.0, 401.0, 402.0),
    }
    rows = []
    for session in sessions:
        timestamps = pd.to_datetime(
            [
                f"{session} 09:30",
                f"{session} 10:00",
                f"{session} 10:30",
                f"{session} 15:55",
            ]
        ).tz_localize("America/New_York")
        for symbol, (first_open, entry_open, close_px) in symbol_values.items():
            opens = [first_open, entry_open, entry_open + 0.5, close_px - 0.2]
            closes = [entry_open - 0.2, entry_open + 0.2, entry_open + 0.8, close_px]
            for bar_index, timestamp in enumerate(timestamps):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "session": session,
                        "bar_index": bar_index,
                        "symbol": symbol,
                        "open": opens[bar_index],
                        "high": max(opens[bar_index], closes[bar_index]),
                        "low": min(opens[bar_index], closes[bar_index]),
                        "close": closes[bar_index],
                        "volume": 1_000_000,
                    }
                )
    return pd.DataFrame(rows)


def _write_config(tmp_path: Path, events_path: Path, panel_path: Path) -> Path:
    config_path = tmp_path / "h3_strategy_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "test_h3_strategy",
                "hypothesis_id": "H3",
                "timeframe": "5min",
                "data": {
                    "earnings_events_path": events_path.as_posix(),
                    "intraday_panel_path": panel_path.as_posix(),
                    "timestamp_timezone": "America/New_York",
                    "timezone_session_policy": {
                        "entry_time": "10:00",
                        "latest_allowed_entry_time": "10:05",
                        "regular_close": "16:00",
                    },
                },
                "events": {
                    "entry": {
                        "entry_time": "10:00",
                        "latest_allowed_entry_time": "10:05",
                    },
                    "exit": {
                        "primary_exit": "same_session_close",
                        "primary_exit_time": "16:00",
                        "secondary_exits": ["t_plus_1_open", "t_plus_1_close"],
                    },
                },
                "signal": {
                    "filters": [
                        {"field": "eps_surprise_z", "operator": ">", "value": 0.0},
                        {"field": "revenue_surprise_z", "operator": ">=", "value": 0.0},
                        {"field": "gap_atr", "operator": "between", "min_value": 0.25, "max_value": 2.50},
                        {"field": "rel_volume_30m", "operator": ">=", "fit_on": "train_fold", "quantile": 0.60},
                        {"field": "close_30m", "operator": ">=", "field_value": "vwap_30m"},
                        {"field": "sector_return_30m", "operator": ">=", "value": -0.001},
                    ]
                },
                "position": {
                    "max_positions_per_session": 10,
                    "max_symbol_weight_per_session": 0.20,
                },
                "costs": {"round_trip_bps": {"base": 0.0, "conservative": 10.0, "stress": 20.0}},
                "split_policy": {
                    "train_months": 1,
                    "validation_months": 1,
                    "test_months": 1,
                    "step_months": 1,
                    "embargo_sessions": 0,
                },
                "outputs": {
                    "output_dir": (tmp_path / "results").as_posix(),
                    "strategy_output_dir": (tmp_path / "results" / "h3_v1").as_posix(),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _run(tmp_path: Path):
    events_path = tmp_path / "events.parquet"
    panel_path = tmp_path / "panel.parquet"
    _events().to_parquet(events_path, index=False)
    _panel().to_parquet(panel_path, index=False)
    return run_strategy(config_path=_write_config(tmp_path, events_path, panel_path))


def test_run_strategy_writes_phase4_artifacts_and_sensitivity_rows(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    trades = pd.read_parquet(outputs.trades_path)
    selected_events = pd.read_parquet(outputs.events_path)
    thresholds = pd.read_parquet(outputs.thresholds_path)

    assert outputs.coverage_path.exists()
    assert outputs.events_path.exists()
    assert outputs.trades_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.distribution_path.exists()
    assert outputs.thresholds_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.report_path.exists()
    assert "evt_validation_fail" not in set(selected_events["event_id"])
    assert thresholds.loc[thresholds["threshold"].eq("rel_volume_30m"), "value"].iloc[0] == 1.0
    assert set(trades["label"]) == {PRIMARY_LABEL, SENSITIVITY_LABEL}
    assert set(trades.loc[trades["label"].eq(PRIMARY_LABEL), "horizon"]) == {"same_session_close"}
    assert {"t_plus_1_open", "t_plus_1_close"}.issubset(
        set(trades.loc[trades["label"].eq(SENSITIVITY_LABEL), "horizon"])
    )


def test_strategy_entry_exit_timestamps_and_costs_are_explicit(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    trades = pd.read_parquet(outputs.trades_path)
    row = trades.loc[
        trades["label"].eq(PRIMARY_LABEL)
        & trades["split"].eq("validation")
        & trades["cost_bps_round_trip"].eq(10.0)
    ].iloc[0]

    entry_clock = pd.Timestamp(row["entry_timestamp"]).tz_convert("America/New_York").strftime("%H:%M")
    exit_clock = pd.Timestamp(row["exit_timestamp"]).tz_convert("America/New_York").strftime("%H:%M")
    assert entry_clock == "10:00"
    assert exit_clock == "15:55"
    assert np.isclose(row["cost_return_gross"], 0.001)
    assert np.isclose(row["net_return"], row["gross_return"] - 0.001)


def test_strategy_residual_returns_use_sector_and_index_proxies(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    trades = pd.read_parquet(outputs.trades_path)
    row = trades.loc[
        trades["label"].eq(PRIMARY_LABEL)
        & trades["split"].eq("validation")
        & trades["cost_bps_round_trip"].eq(0.0)
    ].iloc[0]

    expected_stock = np.log(105.0 / 101.0)
    expected_sector = np.log(51.0 / 50.5)
    expected_index = np.log(402.0 / 401.0)
    assert np.isclose(row["gross_return"], expected_stock)
    assert np.isclose(row["sector_residual_return"], expected_stock - expected_sector)
    assert np.isclose(row["index_residual_return"], expected_stock - expected_index)

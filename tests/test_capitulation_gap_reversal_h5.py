from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.strategy.capitulation_gap_reversal_h5 import (
    PRIMARY_LABEL,
    SHORT_CONTINUATION_LABEL,
    WAIT_1D_LABEL,
    run_strategy,
)


def _write_symbol(cleaned_dir: Path, symbol: str) -> Path:
    rows = []
    spec_by_session = {
        "2026-01-05": {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
        "2026-01-06": {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
        "2026-02-02": {"open": 70.0, "high": 82.0, "low": 60.0, "close": 78.0, "volume": 20_000_000},
        "2026-02-03": {"open": 76.0, "high": 109.0, "low": 75.0, "close": 108.0, "volume": 3_000_000},
        "2026-02-04": {"open": 80.0, "high": 85.0, "low": 78.0, "close": 82.0, "volume": 2_000_000},
        "2026-03-02": {"open": 65.0, "high": 90.0, "low": 70.0, "close": 84.0, "volume": 25_000_000},
        "2026-03-03": {"open": 84.0, "high": 114.0, "low": 83.0, "close": 112.0, "volume": 4_000_000},
        "2026-03-04": {"open": 86.0, "high": 88.0, "low": 84.0, "close": 87.0, "volume": 2_000_000},
    }
    for session, spec in spec_by_session.items():
        timestamps = pd.date_range(f"{session} 09:30", periods=4, freq="5min", tz="America/New_York")
        opens = np.linspace(spec["open"], spec["close"], 4)
        closes = np.linspace(spec["open"], spec["close"], 4)
        highs = [max(open_px, close_px) for open_px, close_px in zip(opens, closes, strict=True)]
        lows = [min(open_px, close_px) for open_px, close_px in zip(opens, closes, strict=True)]
        highs[1] = spec["high"]
        lows[1] = spec["low"]
        for bar_index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "session": session,
                    "bar_index": bar_index,
                    "open": float(opens[bar_index]),
                    "high": float(highs[bar_index]),
                    "low": float(lows[bar_index]),
                    "close": float(closes[bar_index]),
                    "volume": float(spec["volume"] / 4),
                }
            )
    path = cleaned_dir / "5min" / symbol / f"{symbol}_5min_clean.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_config(tmp_path: Path, cleaned_dir: Path) -> Path:
    config_path = tmp_path / "h5_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "test_h5",
                "hypothesis_id": "H5",
                "timeframe": "5min",
                "data": {
                    "provider": "massive",
                    "symbols": ["AAA"],
                    "cleaned_dir": cleaned_dir.as_posix(),
                    "file_template": "{cleaned_dir}/{timeframe}/{symbol}/{symbol}_{timeframe}_clean.parquet",
                    "timestamp_timezone": "America/New_York",
                },
                "signal": {
                    "gap_down_threshold": -0.20,
                    "daily_drop_threshold": -0.30,
                    "three_day_drop_threshold": -0.45,
                    "min_rel_volume": 10.0,
                    "rel_volume_window": 2,
                    "close_location_min": 0.60,
                    "min_price": 5.0,
                    "min_adv_dollar": 1_000_000.0,
                    "adv_window": 2,
                    "min_history_sessions": 1,
                    "require_next_open_above_event_low": True,
                },
                "exit": {
                    "rules": [
                        {"exit_id": "target_2r_or_5d", "profit_target_r": 2.0, "max_hold_sessions": 5},
                        {"exit_id": "fixed_3d", "profit_target_r": None, "max_hold_sessions": 3},
                    ]
                },
                "position": {
                    "stop_buffer_bps": 0.0,
                    "max_initial_risk_pct": 0.25,
                    "max_positions_per_session": 5,
                },
                "costs": {"round_trip_bps": [0.0, 25.0]},
                "controls": {
                    "all_capitulation_liquid": True,
                    "no_close_location": True,
                    "wait_1d": True,
                    "short_continuation": True,
                },
                "split_policy": {
                    "train_months": 1,
                    "validation_months": 1,
                    "test_months": 1,
                    "step_months": 1,
                    "embargo_sessions": 0,
                },
                "outputs": {"output_dir": (tmp_path / "results").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _run(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    _write_symbol(cleaned_dir, "AAA")
    return run_strategy(_write_config(tmp_path, cleaned_dir))


def test_h5_run_writes_standard_artifacts_and_controls(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    events = pd.read_parquet(outputs.events_path)
    trades = pd.read_parquet(outputs.trades_path)

    assert outputs.coverage_path.exists()
    assert outputs.daily_path.exists()
    assert outputs.events_path.exists()
    assert outputs.trades_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.distribution_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.report_path.exists()
    assert PRIMARY_LABEL in set(events["label"])
    assert WAIT_1D_LABEL in set(trades["label"])
    assert SHORT_CONTINUATION_LABEL in set(trades["label"])


def test_h5_enters_next_open_and_reports_r_units(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    trades = pd.read_parquet(outputs.trades_path)
    row = trades.loc[
        trades["label"].eq(PRIMARY_LABEL)
        & trades["split"].eq("validation")
        & trades["exit_id"].eq("target_2r_or_5d")
        & trades["cost_bps_round_trip"].eq(0.0)
    ].iloc[0]

    entry_clock = pd.Timestamp(row["entry_timestamp"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
    assert entry_clock == "2026-02-03 09:30"
    assert row["exit_reason"] == "target"
    assert np.isclose(row["entry_price"], 76.0)
    assert np.isclose(row["stop_price"], 60.0)
    assert np.isclose(row["gross_r"], 2.0)


def test_h5_costs_are_converted_to_stop_risk_units(tmp_path: Path) -> None:
    outputs = _run(tmp_path)
    trades = pd.read_parquet(outputs.trades_path)
    row = trades.loc[
        trades["label"].eq(PRIMARY_LABEL)
        & trades["split"].eq("validation")
        & trades["exit_id"].eq("target_2r_or_5d")
        & trades["cost_bps_round_trip"].eq(25.0)
    ].iloc[0]

    expected_cost_r = row["cost_return_gross"] / row["risk_pct"]
    assert np.isclose(row["net_r"], row["gross_r"] - expected_cost_r)

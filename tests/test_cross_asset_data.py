from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cross_asset_data import (
    align_symbols,
    build_aligned_panel,
    build_symbol_config,
    missing_detail_frame,
    resolve_symbols,
    symbol_paths,
    write_coverage_report,
)


def _config(tmp_path: Path) -> dict:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
universe_id: test_universe
context_universe_core:
  indices:
    - SPY
    - QQQ
  sectors:
    - XLK
benchmark_universe:
  - SPY
""".strip(),
        encoding="utf-8",
    )
    return {
        "project": {"name": "ida-trading", "timezone": "America/New_York", "frequency": "5min"},
        "lab": {
            "target_symbol": "SPY",
            "timeframe": "5min",
            "provider": "polygon",
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
            "universe_config": str(universe_path),
            "universe_id": "test_universe",
            "context_key": "context_universe_core",
            "include_target_in_context": True,
        },
        "polygon": {"api_key_env": "POLYGON_API_KEY", "source_interval": "5m", "adjusted": True, "default_years": 5},
        "paths": {
            "raw_dir": str(tmp_path / "data/raw/polygon"),
            "cleaned_dir": str(tmp_path / "data/cleaned"),
            "aligned_dir": str(tmp_path / "data/aligned"),
            "metadata_dir": str(tmp_path / "data/metadata"),
            "data_coverage_dir": str(tmp_path / "reports/data_coverage"),
            "alignment_dir": str(tmp_path / "reports/alignment"),
        },
        "session": {
            "market_open": "09:30",
            "market_close": "16:00",
            "timestamp_label": "start",
            "regular_session_only": True,
            "drop_incomplete_sessions": True,
            "expected_bars_per_session": 78,
        },
        "calendar": {"enabled": True, "name": "NYSE", "drop_non_trading_days": True, "drop_half_days": True},
        "quality": {},
        "labeling": {"horizon_bars": 2},
        "backtest": {"no_new_trades_after": "15:45", "force_flat_before": "15:55"},
        "alignment": {"missing_policy": "drop_core_missing"},
    }


def _cleaned_frame(symbol_offset: float = 0.0, periods: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=periods, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session": ["2024-01-02"] * periods,
            "bar_index": list(range(periods)),
            "open": [100.0 + symbol_offset + i for i in range(periods)],
            "high": [101.0 + symbol_offset + i for i in range(periods)],
            "low": [99.0 + symbol_offset + i for i in range(periods)],
            "close": [100.5 + symbol_offset + i for i in range(periods)],
            "volume": [1000 + i for i in range(periods)],
            "next_open_timestamp": list(timestamps[1:]) + [pd.NaT],
            "target_crosses_session_close": [False] * max(0, periods - 1) + [True],
            "can_open_trade": [True] * max(0, periods - 1) + [False],
            "force_flat_bar": [False] * periods,
            "trade_could_remain_open_past_close": [False] * max(0, periods - 1) + [True],
        }
    )


def test_resolve_symbols_flattens_universe_and_includes_target_once(tmp_path) -> None:
    config = _config(tmp_path)

    assert resolve_symbols(config, target_symbol="SPY") == ["SPY", "QQQ", "XLK"]


def test_symbol_config_uses_target_aware_paths(tmp_path) -> None:
    config = _config(tmp_path)

    paths = symbol_paths(config, "QQQ", target_symbol="SPY")
    symbol_config = build_symbol_config(config, "QQQ", target_symbol="SPY")

    assert paths.raw_file.as_posix().endswith("data/raw/polygon/5min/QQQ/QQQ_5min.parquet")
    assert paths.cleaned_file.as_posix().endswith("data/cleaned/5min/QQQ/QQQ_5min_clean.parquet")
    assert symbol_config["data"]["symbol"] == "QQQ"
    assert symbol_config["data"]["input_file"] == str(paths.raw_file)


def test_build_aligned_panel_drops_missing_core_timestamps() -> None:
    spy = _cleaned_frame(0.0, periods=3)
    qqq = _cleaned_frame(10.0, periods=2)

    panel, missing = build_aligned_panel({"SPY": spy, "QQQ": qqq}, target_symbol="SPY")

    assert len(panel) == 2
    assert missing == {"SPY": 0, "QQQ": 1}
    assert "SPY__open" in panel.columns
    assert "QQQ__open" in panel.columns
    assert "SPY__can_open_trade" in panel.columns
    assert "SPY__target_open_next" in panel.columns
    assert panel.loc[0, "SPY__target_open_next"] == 101.0
    assert panel["is_available_QQQ"].all()

    detail = missing_detail_frame({"SPY": spy, "QQQ": qqq}, "SPY", ["SPY", "QQQ"])
    assert detail.loc[0, "symbol"] == "QQQ"
    assert detail.loc[0, "missing_bars"] == 1


def test_write_coverage_and_align_symbols(tmp_path) -> None:
    config = _config(tmp_path)
    for symbol, offset in [("SPY", 0.0), ("QQQ", 10.0)]:
        path = symbol_paths(config, symbol, target_symbol="SPY").cleaned_file
        path.parent.mkdir(parents=True, exist_ok=True)
        _cleaned_frame(offset, periods=3).to_parquet(path, index=False)
        raw_path = symbol_paths(config, symbol, target_symbol="SPY").raw_file
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        _cleaned_frame(offset, periods=3).loc[:, ["timestamp"]].to_parquet(raw_path, index=False)

    coverage_path = write_coverage_report(config, [], "SPY")
    report = align_symbols(config, ["SPY", "QQQ"], "SPY")

    assert coverage_path.exists()
    assert Path(report.output_path).exists()
    assert report.missing_detail_path is not None
    assert Path(report.missing_detail_path).exists()
    assert report.aligned_rows == 3
    assert report.dropped_target_rows == 0

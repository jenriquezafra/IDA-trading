from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cross_asset_leakage_audit import (
    build_feature_timestamp_catalog,
    check_feature_catalog,
    check_panel_structure,
    check_symbol_columns,
    run,
)


def _config(tmp_path: Path) -> dict:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
context_universe_core:
  - SPY
  - QQQ
""".strip(),
        encoding="utf-8",
    )
    return {
        "project": {"timezone": "America/New_York", "frequency": "5min"},
        "lab": {
            "target_symbol": "SPY",
            "timeframe": "5min",
            "universe_config": str(universe_path),
            "universe_id": "test_universe",
            "context_key": "context_universe_core",
        },
        "paths": {
            "aligned_dir": str(tmp_path / "data/aligned"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "session": {"timestamp_label": "start"},
        "alignment": {"missing_policy": "drop_core_missing"},
    }


def _panel() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session": ["2024-01-02"] * 3,
            "bar_index": [0, 1, 2],
            "SPY__open": [100.0, 101.0, 102.0],
            "SPY__high": [101.0, 102.0, 103.0],
            "SPY__low": [99.0, 100.0, 101.0],
            "SPY__close": [100.5, 101.5, 102.5],
            "SPY__volume": [1000, 1001, 1002],
            "SPY__target_open_next": [101.0, 102.0, pd.NA],
            "SPY__next_open_timestamp": [timestamps[1], timestamps[2], pd.NaT],
            "SPY__target_crosses_session_close": [False, False, True],
            "SPY__can_open_trade": [True, True, False],
            "SPY__force_flat_bar": [False, False, False],
            "SPY__trade_could_remain_open_past_close": [False, False, True],
            "QQQ__open": [200.0, 201.0, 202.0],
            "QQQ__high": [201.0, 202.0, 203.0],
            "QQQ__low": [199.0, 200.0, 201.0],
            "QQQ__close": [200.5, 201.5, 202.5],
            "QQQ__volume": [2000, 2001, 2002],
            "is_available_SPY": [True, True, True],
            "is_available_QQQ": [True, True, True],
        }
    )


def test_catalog_blocks_target_execution_columns() -> None:
    catalog = build_feature_timestamp_catalog(_panel(), _config(Path("/tmp")), "SPY")

    usable = catalog.set_index("column")["usable_as_feature"].to_dict()
    assert usable["SPY__close"] is True
    assert usable["QQQ__close"] is True
    assert usable["is_available_QQQ"] is True
    assert usable["SPY__target_open_next"] is False


def test_negative_future_like_column_fails_feature_catalog() -> None:
    panel = _panel()
    panel["QQQ__future_close"] = [201.5, 202.5, 203.5]
    catalog = build_feature_timestamp_catalog(panel, _config(Path("/tmp")), "SPY")

    checks = check_feature_catalog(catalog)

    assert any(check.status == "FAIL" and check.check_id == "no_unknown_feature_inputs" for check in checks)


def test_negative_non_target_execution_field_fails() -> None:
    panel = _panel()
    panel["QQQ__target_open_next"] = [201.0, 202.0, pd.NA]

    checks = check_symbol_columns(panel, ["SPY", "QQQ"], "SPY")

    assert any(check.status == "FAIL" and check.check_id == "no_non_target_execution_fields" for check in checks)


def test_negative_target_next_open_mismatch_fails() -> None:
    panel = _panel()
    panel.loc[0, "SPY__target_open_next"] = 999.0

    checks = check_panel_structure(panel, "SPY")

    assert any(check.status == "FAIL" and check.check_id == "target_next_open_alignment" for check in checks)


def test_run_writes_reports(tmp_path) -> None:
    config = _config(tmp_path)
    panel_path = tmp_path / "data/aligned/SPY/5min/test_universe/panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, index=False)
    config_path = tmp_path / "hmm_lab.yaml"
    config_path.write_text(
        """
project:
  timezone: America/New_York
  frequency: 5min
lab:
  target_symbol: SPY
  timeframe: 5min
  universe_config: universe.yaml
  universe_id: test_universe
  context_key: context_universe_core
paths:
  aligned_dir: data/aligned
  reports_dir: reports
session:
  timestamp_label: start
alignment:
  missing_policy: drop_core_missing
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "universe.yaml").write_text(
        """
context_universe_core:
  - SPY
  - QQQ
""".strip(),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        report_path = run(config_path)
    finally:
        os.chdir(cwd)

    assert report_path.exists()
    assert (tmp_path / "reports/SPY/leakage_audit_cross_asset.parquet").exists()
    assert (tmp_path / "reports/SPY/feature_timestamp_catalog.parquet").exists()

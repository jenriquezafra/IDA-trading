from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.execution.paper_data_refresh import _merge_ohlcv, run_paper_data_refresh


def test_merge_ohlcv_keeps_latest_duplicate_timestamp() -> None:
    existing = pd.DataFrame(
        [
            {"timestamp": "2026-05-01 09:30:00-04:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 10},
            {"timestamp": "2026-05-01 09:45:00-04:00", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 20},
        ]
    )
    incoming = pd.DataFrame(
        [
            {"timestamp": "2026-05-01 09:45:00-04:00", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 30},
            {"timestamp": "2026-05-01 10:00:00-04:00", "open": 4, "high": 4, "low": 4, "close": 4, "volume": 40},
        ]
    )

    merged = _merge_ohlcv(existing, incoming)

    assert len(merged) == 3
    assert merged.iloc[1]["open"] == 3


def test_paper_data_refresh_dry_run_writes_manifest(tmp_path: Path) -> None:
    lab_config_path = tmp_path / "lab.yaml"
    feature_config_path = tmp_path / "features.yaml"
    cboe_config_path = tmp_path / "cboe.yaml"
    config_path = tmp_path / "refresh.yaml"
    universe_path = tmp_path / "universe.yaml"

    universe_path.write_text("context_universe_core: [SPY]\n", encoding="utf-8")
    lab_config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test", "timezone": "America/New_York", "frequency": "15min"},
                "lab": {
                    "target_symbol": "QQQ",
                    "timeframe": "15min",
                    "provider": "polygon",
                    "start_date": "2026-05-01",
                    "end_date": "2026-05-01",
                    "universe_config": universe_path.as_posix(),
                    "universe_id": "test_universe",
                    "context_key": "context_universe_core",
                    "include_target_in_context": True,
                    "optional_symbols": [],
                },
                "polygon": {"api_key_env": "POLYGON_API_KEY", "source_interval": "5m", "adjusted": True, "default_years": 1},
                "paths": {
                    "raw_dir": (tmp_path / "raw").as_posix(),
                    "cleaned_dir": (tmp_path / "cleaned").as_posix(),
                    "aligned_dir": (tmp_path / "aligned").as_posix(),
                    "data_coverage_dir": (tmp_path / "coverage").as_posix(),
                    "alignment_dir": (tmp_path / "alignment").as_posix(),
                    "metadata_dir": (tmp_path / "metadata").as_posix(),
                    "reports_dir": (tmp_path / "reports").as_posix(),
                },
                "session": {"drop_incomplete_sessions": True, "expected_bars_per_session": 26},
                "calendar": {"enabled": False},
                "quality": {},
                "labeling": {"horizon_bars": 2},
                "backtest": {"no_new_trades_after": "15:45", "force_flat_before": "15:55"},
                "alignment": {"missing_policy": "drop_core_missing"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    feature_config_path.write_text("feature_set_version: test_features\n", encoding="utf-8")
    cboe_config_path.write_text("paths:\n  context: cboe.parquet\n", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "refresh": {
                    "target_symbol": "QQQ",
                    "lab_config_path": lab_config_path.as_posix(),
                    "feature_config_path": feature_config_path.as_posix(),
                    "cboe_config_path": cboe_config_path.as_posix(),
                    "lookback_days": 1,
                    "download": True,
                    "clean": True,
                    "align": True,
                    "build_features": True,
                    "refresh_cboe": True,
                    "keep_incomplete_sessions_for_paper": True,
                },
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    paths, manifest = run_paper_data_refresh(config_path=config_path, dry_run=True)

    assert paths.manifest_path.exists()
    assert paths.report_path.exists()
    assert manifest["run"]["status"] == "dry_run"
    assert manifest["steps"] == {"download": False, "clean": False, "align": False, "build_features": False, "refresh_cboe": False}
    assert manifest["keep_incomplete_sessions_for_paper"] is True

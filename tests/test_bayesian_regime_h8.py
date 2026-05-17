from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.bayesian_regime_h8 import prepare_h8_frame, run


def _features(months: int = 5, sessions_per_month: int = 5, bars: int = 8) -> pd.DataFrame:
    rows = []
    price = 100.0
    for month in range(months):
        start = pd.Timestamp("2024-01-02") + pd.DateOffset(months=month)
        sessions = pd.date_range(start, periods=sessions_per_month, freq="B")
        for session_idx, session_ts in enumerate(sessions):
            session = session_ts.strftime("%Y-%m-%d")
            timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="15min", tz="America/New_York")
            month_direction = 1.0 if month % 2 == 0 else -1.0
            for bar_index, timestamp in enumerate(timestamps):
                drift = month_direction * (0.001 + 0.0001 * bar_index)
                price *= float(np.exp(drift))
                rows.append(
                    {
                        "timestamp": timestamp,
                        "session": session,
                        "bar_index": bar_index,
                        "target_ret_3": drift * 3.0,
                        "target_ret_4": drift * 4.0,
                        "target_rv_12_rel_by_bar": 0.8 + 0.05 * ((bar_index + session_idx + month) % 5),
                        "target_signed_efficiency_12": month_direction * (0.5 + 0.02 * bar_index),
                        "target_open_next": price * np.exp(drift),
                    }
                )
    return pd.DataFrame(rows)


def _config(tmp_path: Path, features_path: Path) -> dict:
    return {
        "lab": {"target_symbol": "SPY"},
        "bayesian_regime_h8": {
            "features_file": str(features_path),
            "results_dir": str(tmp_path / "results"),
            "report_file": str(tmp_path / "reports/h8.md"),
            "models_dir": str(tmp_path / "models"),
            "momentum_column": "target_ret_4",
            "volatility_column": "target_rv_12_rel_by_bar",
            "efficiency_column": "target_signed_efficiency_12",
            "variants": ["manual_tv3", "manual_h8a", "trained_h8b"],
            "walk_forward": {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1},
            "max_folds": 1,
            "trained_n_states": 4,
            "trained_n_iter": 10,
            "trained_random_state": 7,
            "probability_thresholds": [0.55],
            "max_entropy_values": [None],
            "horizons": [1],
            "cost_bps": [1.0],
        },
    }


def test_prepare_h8_frame_drops_invalid_source_rows() -> None:
    features = _features(months=1)
    features.loc[0, "target_ret_4"] = np.nan

    prepared = prepare_h8_frame(features, {"bayesian_regime_h8": {}})

    assert len(prepared) == len(features) - 1
    assert {"raw_mom_z", "raw_vol_z", "raw_eff_z"}.issubset(prepared.columns)
    assert prepared["source_index"].min() == 1


def test_h8_runner_writes_report_and_diagnostics(tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    _features().to_parquet(features_path, index=False)
    config = _config(tmp_path, features_path)
    config_path = tmp_path / "h8.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report_path, diagnostics_path = run(config_path)

    assert report_path.exists()
    assert diagnostics_path.exists()
    diagnostics = pd.read_parquet(diagnostics_path)
    assert {"manual_tv3", "manual_h8a", "trained_h8b"}.issubset(set(diagnostics["variant"]))
    assert (tmp_path / "results/h8_posteriors.parquet").exists()

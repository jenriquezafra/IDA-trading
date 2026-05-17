from __future__ import annotations

import numpy as np
import pandas as pd

from src.leakage_audit import check_execution, check_labels, recompute_relative_volume


def test_recompute_relative_volume_uses_only_prior_sessions() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"],
            "bar_index": [0, 1, 0, 1, 0, 1],
            "volume": [10.0, 20.0, 20.0, 40.0, 60.0, 80.0],
        }
    )

    rel = recompute_relative_volume(frame)

    assert np.isnan(rel.iloc[0])
    assert np.isnan(rel.iloc[1])
    assert rel.iloc[2] == 2.0
    assert rel.iloc[3] == 2.0
    assert rel.iloc[4] == 4.0
    assert rel.iloc[5] == 80.0 / 30.0


def test_check_labels_passes_next_open_and_no_close_crossing() -> None:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="5min", tz="America/New_York")
    features = pd.DataFrame(
        {
            "timestamp": timestamps,
            "session": ["2024-01-02"] * 6,
            "bar_index": list(range(6)),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    labels = pd.DataFrame(
        {
            "timestamp": timestamps[:3],
            "session": ["2024-01-02"] * 3,
            "bar_index": [0, 1, 2],
            "bars_in_session": [6, 6, 6],
            "open": [100.0, 101.0, 102.0],
            "rv_12": [0.001, 0.001, 0.001],
            "entry_px": [101.0, 102.0, 103.0],
            "exit_px": [103.0, 104.0, 105.0],
            "fwd_ret": [np.log(103 / 101), np.log(104 / 102), np.log(105 / 103)],
            "sigma_h": [0.001 * np.sqrt(2)] * 3,
            "neutral_zone": [max(0.00015, 0.25 * 0.001 * np.sqrt(2))] * 3,
            "target_crosses_session_close": [False, False, False],
        }
    )

    config = {"labeling": {"horizon_bars": 2, "round_trip_cost_bps": 1.0, "buffer_bps": 0.5, "lambda_vol": 0.25}}
    checks = check_labels(labels, features, config)

    assert {check.status for check in checks} == {"PASS"}


def test_check_execution_detects_next_open_alignment() -> None:
    cleaned = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 3,
            "bar_index": [0, 1, 2],
        }
    )
    trades = pd.DataFrame(
        {
            "signal_timestamp": [cleaned.loc[0, "timestamp"]],
            "entry_timestamp": [cleaned.loc[1, "timestamp"]],
            "exit_timestamp": [cleaned.loc[2, "timestamp"]],
            "session": ["2024-01-02"],
            "entry_bar_index": [1],
            "position": [1.0],
            "total_cost_ret": [0.0001],
        }
    )

    checks = check_execution(cleaned, trades, {"labeling": {"round_trip_cost_bps": 1.0}})

    assert {check.status for check in checks} == {"PASS"}

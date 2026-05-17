from __future__ import annotations

import pandas as pd
import pytest

from src.spy_setup_feasibility import evaluate_direction, iter_segments, stability_table


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=12, freq="5min"),
            "session": ["2024-01-02"] * 6 + ["2024-01-03"] * 6,
            "hour": [9, 9, 10, 10, 11, 11, 9, 9, 10, 10, 11, 11],
            "fwd_ret": [0.002, -0.001, 0.003, 0.001, -0.002, 0.004, 0.001, 0.002, -0.001, 0.003, -0.002, 0.004],
            "target_open_next": [100.0] * 12,
            "target_first_60m": [True, True, False, False, False, False, True, True, False, False, False, False],
            "target_lunch": [False] * 12,
            "target_last_60m": [False] * 12,
            "target_overnight_ret": [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "target_failed_breakout_high_12": [False, True, False, False, False, True, False, False, True, False, False, True],
        }
    )


def test_evaluate_direction_applies_cost_to_each_trade() -> None:
    frame = _frame().iloc[:3].copy()
    scenario = {"cost_scenario": "bps_2", "cost_kind": "bps", "cost_bps": 2.0}

    metrics = evaluate_direction(frame, "long", scenario)

    assert metrics["trades"] == 3
    assert metrics["gross_return"] == pytest.approx(0.004)
    assert metrics["cost_return"] == pytest.approx(0.0006)
    assert metrics["net_return"] == pytest.approx(0.0034)


def test_iter_segments_uses_reference_terciles_and_setup_flags() -> None:
    frame = _frame()
    segments = list(iter_segments(frame, frame, {"spy_setup_feasibility": {"min_segment_rows": 1}}))
    keys = {(name, value) for name, value, _ in segments}

    assert ("all", "all") in keys
    assert ("hour", "9") in keys
    assert ("day_part", "first_60m") in keys
    assert ("gap_tercile", "low") in keys
    assert ("setup_flag", "target_failed_breakout_high_12") in keys


def test_stability_table_marks_positive_validation_and_test() -> None:
    grid = pd.DataFrame(
        [
            {
                "fold": 0,
                "split": "validation",
                "horizon_bars": 12,
                "segment_name": "all",
                "segment_value": "all",
                "direction": "long",
                "cost_scenario": "ibkr_tiered_10000",
                "rows": 100,
                "net_return": 0.03,
                "avg_trade_net": 0.0003,
                "daily_sharpe": 1.0,
                "profit_factor": 1.2,
                "top_session_abs_net_share": 0.2,
            },
            {
                "fold": 0,
                "split": "test",
                "horizon_bars": 12,
                "segment_name": "all",
                "segment_value": "all",
                "direction": "long",
                "cost_scenario": "ibkr_tiered_10000",
                "rows": 100,
                "net_return": 0.02,
                "avg_trade_net": 0.0002,
                "daily_sharpe": 0.8,
                "profit_factor": 1.1,
                "top_session_abs_net_share": 0.3,
            },
        ]
    )

    stability = stability_table(grid, {"spy_setup_feasibility": {"primary_cost_scenario": "ibkr_tiered_10000"}})

    assert bool(stability.loc[0, "stable_positive"])
    assert stability.loc[0, "avg_trade_decay"] == pytest.approx(-0.0001)

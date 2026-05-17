from __future__ import annotations

import pandas as pd

from src.setup_signal_portfolio_lifecycle import evaluate_lifecycle_frame, lifecycle_position_from_entries


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["AAA", "AAA", "AAA", "AAA"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 10:30",
                    "2024-01-02 10:35",
                    "2024-01-02 10:40",
                    "2024-01-02 10:45",
                ]
            ),
            "session": ["2024-01-02"] * 4,
            "bar_index": [10, 11, 12, 13],
            "entry_px": [100.0, 101.0, 100.0, 100.5],
            "exit_px": [101.0, 100.0, 100.5, 101.0],
            "fwd_ret": [0.0100, -0.0100, 0.0050, 0.0050],
        }
    )


def test_lifecycle_position_exits_after_signal_loss() -> None:
    frame = _frame()
    signal = pd.Series([True, False, False, False], index=frame.index)

    position = lifecycle_position_from_entries(
        frame,
        signal,
        direction=1.0,
        max_hold_bars=24,
        min_hold_bars=1,
        exit_on_signal_loss=True,
    )

    assert position.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_lifecycle_position_exits_after_stop_loss_close_to_close() -> None:
    frame = _frame()
    frame["fwd_ret"] = [-0.0060, 0.0020, 0.0030, 0.0010]
    signal = pd.Series([True, True, True, True], index=frame.index)

    position = lifecycle_position_from_entries(
        frame,
        signal,
        direction=1.0,
        max_hold_bars=24,
        min_hold_bars=1,
        stop_loss_bps=50.0,
        cooldown_bars=1,
    )

    assert position.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_lifecycle_costs_are_charged_on_turnover_not_active_bars() -> None:
    frame = _frame().iloc[:3].copy()
    position = pd.Series([1.0, 1.0, 0.0], index=frame.index)
    scenario = {"cost_scenario": "bps_2", "cost_kind": "bps", "round_trip_bps": 2.0}

    metrics = evaluate_lifecycle_frame(frame, position, scenario)

    assert metrics["active_bars"] == 2
    assert metrics["entries"] == 1
    assert abs(metrics["turnover"] - 2.0) < 1e-12
    assert abs(metrics["total_cost"] - 0.0002) < 1e-12


def test_portfolio_turnover_is_grouped_by_target_session() -> None:
    frame = pd.DataFrame(
        {
            "target": ["AAA", "AAA", "BBB", "BBB"],
            "timestamp": pd.to_datetime(["2024-01-02 10:30", "2024-01-02 10:35", "2024-01-02 10:30", "2024-01-02 10:35"]),
            "session": ["2024-01-02"] * 4,
            "bar_index": [10, 11, 10, 11],
            "entry_px": [100.0, 101.0, 50.0, 51.0],
            "exit_px": [101.0, 102.0, 51.0, 52.0],
            "fwd_ret": [0.001, 0.001, 0.001, 0.001],
        }
    )
    position = pd.Series([0.5, 0.5, 0.5, 0.5], index=frame.index)
    scenario = {"cost_scenario": "bps_2", "cost_kind": "bps", "round_trip_bps": 2.0}

    metrics = evaluate_lifecycle_frame(frame, position, scenario)

    assert metrics["entries"] == 2
    assert abs(metrics["turnover"] - 2.0) < 1e-12

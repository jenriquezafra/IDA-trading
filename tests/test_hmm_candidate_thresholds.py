from __future__ import annotations

import pandas as pd
import pytest

from src.hmm_candidate_thresholds import (
    decide_candidates,
    evaluate_threshold_frame,
    select_validation_thresholds,
    threshold_position,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "bar_index": [0, 1, 0, 1],
            "fwd_ret": [0.003, -0.002, 0.004, -0.001],
            "ret_3": [0.003, -0.002, 0.0005, -0.003],
            "neutral_zone": [0.001, 0.001, 0.001, 0.001],
        }
    )


def test_threshold_position_respects_multiplier() -> None:
    loose = threshold_position(_frame(), "momentum_ret_3", 1.0)
    tight = threshold_position(_frame(), "momentum_ret_3", 2.0)

    assert loose.tolist() == [1, -1, 0, -1]
    assert tight.tolist() == [1, 0, 0, -1]


def test_evaluate_threshold_frame_adds_drawdown() -> None:
    metrics = evaluate_threshold_frame(_frame(), "momentum_ret_3", threshold_multiplier=1.0, cost_bps=1.0)

    assert metrics["trades"] == 3
    assert metrics["net_return"] == pytest.approx(0.003 + 0.002 + 0.001 - 0.0003)
    assert metrics["max_drawdown_abs"] >= 0.0


def test_select_validation_thresholds_prefers_candidate_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "split": "validation",
                "threshold_multiplier": 1.0,
                "cost_bps": 1.0,
                "candidate_status": "weak_profit_factor",
                "total_net_return": 0.10,
                "return_to_drawdown": 2.0,
                "total_trades": 100,
            },
            {
                "candidate_id": "a",
                "split": "validation",
                "threshold_multiplier": 1.5,
                "cost_bps": 1.0,
                "candidate_status": "candidate",
                "total_net_return": 0.05,
                "return_to_drawdown": 1.0,
                "total_trades": 80,
            },
        ]
    )

    selected = select_validation_thresholds(summary)

    assert selected.loc[0, "validation_threshold_multiplier"] == 1.5


def test_decide_candidates_requires_two_bps_candidate() -> None:
    selected_tests = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "source_rank": 1,
                "feature_set": "x",
                "cost_bps": 1.0,
                "candidate_status": "candidate",
                "total_net_return": 0.2,
                "max_drawdown_abs": 0.05,
            },
            {
                "candidate_id": "a",
                "source_rank": 1,
                "feature_set": "x",
                "cost_bps": 2.0,
                "candidate_status": "weak_profit_factor",
                "total_net_return": 0.1,
                "max_drawdown_abs": 0.08,
            },
            {
                "candidate_id": "b",
                "source_rank": 2,
                "feature_set": "x",
                "cost_bps": 1.0,
                "candidate_status": "candidate",
                "total_net_return": 0.15,
                "max_drawdown_abs": 0.04,
            },
            {
                "candidate_id": "b",
                "source_rank": 2,
                "feature_set": "x",
                "cost_bps": 2.0,
                "candidate_status": "candidate",
                "total_net_return": 0.08,
                "max_drawdown_abs": 0.06,
            },
        ]
    )

    decisions = decide_candidates(selected_tests)

    assert decisions.loc[0, "candidate_id"] == "b"
    assert decisions.loc[0, "accepted"]
    assert decisions.loc[1, "cost_fragile"]

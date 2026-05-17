from __future__ import annotations

import pandas as pd
import pytest

from src.hmm_state_economics_cross_asset import (
    build_forward_returns,
    classify_economic_row,
    evaluate_action_metrics,
    select_validation_candidates,
)


def test_build_forward_returns_respects_session_boundary() -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=4, freq="5min").tolist()
            + pd.date_range("2024-01-03 09:30", periods=4, freq="5min").tolist(),
            "session": ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "bar_index": [0, 1, 2, 3] * 2,
            "target_open_next": [101, 102, 103, None, 201, 202, 203, None],
            "target_ret_3": [0.1] * 8,
        }
    )

    returns = build_forward_returns(features, [1, 2])

    assert set(returns["horizon_bars"]) == {1, 2}
    assert len(returns[returns["horizon_bars"] == 1]) == 4
    assert len(returns[returns["horizon_bars"] == 2]) == 2
    assert returns["fwd_ret"].notna().all()


def test_evaluate_action_metrics_applies_costs_and_momentum() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "hour": [10, 10, 11, 11],
            "fwd_ret": [0.002, -0.001, 0.003, -0.002],
            "target_ret_3": [0.001, -0.001, 0.001, -0.001],
        }
    )

    long_metrics = evaluate_action_metrics(frame, "long", 1.0)
    momentum_metrics = evaluate_action_metrics(frame, "momentum", 1.0)

    assert long_metrics["gross_return"] == pytest.approx(0.002)
    assert long_metrics["total_cost"] == pytest.approx(0.0004)
    assert momentum_metrics["net_return"] > long_metrics["net_return"]


def test_classify_economic_row_requires_same_hour_edge() -> None:
    row = pd.Series(
        {
            "bucket": "state",
            "action": "long",
            "stability_status": "stable_profile_candidate",
            "trades": 100,
            "net_return": 0.05,
            "avg_trade_net": 0.0005,
            "profit_factor": 1.2,
            "daily_sharpe": 1.2,
            "avg_trade_net_vs_same_hour_ex_state": 0.0001,
            "top_hour_pct": 0.2,
            "top_session_pct": 0.03,
            "max_daily_abs_net_share": 0.1,
        }
    )

    assert classify_economic_row(row, {"hmm_state_economics_cross_asset": {}}) == "economic_candidate"
    weak = row.copy()
    weak["avg_trade_net_vs_same_hour_ex_state"] = -0.0001
    assert classify_economic_row(weak, {"hmm_state_economics_cross_asset": {}}) == "rejected_no_same_hour_edge"


def test_select_validation_candidates_ignores_test_candidates() -> None:
    diagnostics = pd.DataFrame(
        [
            {"bucket": "state", "split": "test", "economic_status": "economic_candidate", "avg_trade_net": 1.0, "profit_factor": 2.0, "daily_sharpe": 2.0},
            {"bucket": "state", "split": "validation", "economic_status": "rejected_negative_net", "avg_trade_net": -1.0, "profit_factor": 0.5, "daily_sharpe": -1.0},
        ]
    )

    selected = select_validation_candidates(diagnostics, {"hmm_state_economics_cross_asset": {"candidate_split": "validation"}})

    assert selected.empty

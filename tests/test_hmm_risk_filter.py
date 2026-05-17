from __future__ import annotations

import pandas as pd
import pytest

from src.hmm_risk_filter import base_position, classify_filter_row, evaluate_position, filter_multiplier, same_hour_multiplier


def test_base_position_builds_expected_simple_strategies() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_3": [0.002, -0.002, 0.0],
            "target_dist_vwap_atr": [1.0, -1.0, 0.0],
            "supervised_score": [0.2, -0.2, 0.0],
        }
    )

    assert base_position(frame, "momentum_simple", 0.001).tolist() == [1.0, -1.0, 0.0]
    assert base_position(frame, "reversion_simple", 0.001).tolist() == [-1.0, 1.0, -0.0]
    assert base_position(frame, "vwap_location", 0.5).tolist() == [-1.0, 1.0, -0.0]
    assert base_position(frame, "supervised_simple", 0.1).tolist() == [1.0, -1.0, 0.0]


def test_filter_multiplier_excludes_and_reduces_stress() -> None:
    frame = pd.DataFrame({"proposed_label": ["risk_on_trend", "risk_off_stress", "high_volatility_expansion", "chop_neutral", "defensive_rotation"]})

    assert filter_multiplier(frame, "only_risk_on").tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert filter_multiplier(frame, "exclude_chop").tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]
    assert filter_multiplier(frame, "exclude_stress").tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]
    assert filter_multiplier(frame, "reduce_stress").tolist() == [1.0, 0.5, 0.5, 1.0, 1.0]


def test_same_hour_multiplier_uses_frozen_hours() -> None:
    frame = pd.DataFrame({"hour": [10, 11, 12]})

    assert same_hour_multiplier(frame, (10, 12)).tolist() == [1.0, 0.0, 1.0]


def test_evaluate_position_applies_costs_and_drawdown() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "fwd_ret": [0.002, -0.001, 0.003],
        }
    )
    position = pd.Series([1.0, 0.0, 0.5])

    metrics = evaluate_position(frame, position, 2.0)

    assert metrics["trades"] == 2
    assert metrics["gross_return"] == pytest.approx(0.0035)
    assert metrics["total_cost"] == pytest.approx(0.0003)
    assert metrics["net_return"] == pytest.approx(0.0032)


def test_classify_filter_row_requires_quality_improvement() -> None:
    row = pd.Series(
        {
            "bucket": "hmm_filter",
            "strategy": "momentum_simple",
            "trades": 100,
            "net_return": 0.05,
            "profit_factor": 1.1,
            "daily_sharpe_delta_vs_base": 0.1,
            "drawdown_reduction_vs_base": 0.0,
            "daily_sharpe_delta_vs_same_hour": 0.1,
            "drawdown_reduction_vs_same_hour": 0.0,
        }
    )

    assert classify_filter_row(row, {"hmm_risk_filter": {}}) == "risk_filter_candidate"
    weak = row.copy()
    weak["daily_sharpe_delta_vs_base"] = -0.1
    weak["drawdown_reduction_vs_base"] = -0.1
    assert classify_filter_row(weak, {"hmm_risk_filter": {}}) == "rejected_no_quality_improvement"

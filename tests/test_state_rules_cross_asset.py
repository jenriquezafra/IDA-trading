from __future__ import annotations

import pandas as pd
import pytest

from src.state_rules_cross_asset import classify_rule_row, evaluate_rule_metrics, position_for_rule, select_validation_rules


def test_position_for_rule_maps_simple_rule_types() -> None:
    frame = pd.DataFrame({"target_ret_3": [0.002, -0.002, 0.0]})

    assert position_for_rule(frame, "long_momentum", 0.001, "long").tolist() == [1.0, 0.0, 0.0]
    assert position_for_rule(frame, "short_momentum", 0.001, "short").tolist() == [0.0, -1.0, 0.0]
    assert position_for_rule(frame, "mean_reversion", 0.001, "long").tolist() == [-1.0, 1.0, 0.0]
    assert position_for_rule(frame, "reduce_risk", 0.001, "momentum").tolist() == [0.5, -0.5, 0.0]


def test_evaluate_rule_metrics_applies_fractional_costs() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "hour": [10, 10, 11, 11],
            "target_ret_3": [0.002, -0.002, 0.002, -0.002],
            "fwd_ret": [0.003, -0.001, 0.002, -0.002],
        }
    )

    metrics = evaluate_rule_metrics(frame, "reduce_risk", 0.001, 2.0, "momentum")

    assert metrics["trades"] == 4
    assert metrics["turnover"] == pytest.approx(2.0)
    assert metrics["gross_return"] == pytest.approx(0.004)
    assert metrics["total_cost"] == pytest.approx(0.0004)
    assert metrics["net_return"] == pytest.approx(0.0036)


def test_classify_rule_row_requires_no_hmm_and_same_hour_edge() -> None:
    row = pd.Series(
        {
            "bucket": "hmm_state_rule",
            "rule_type": "long_momentum",
            "trades": 100,
            "net_return": 0.05,
            "avg_trade_net": 0.0005,
            "profit_factor": 1.2,
            "daily_sharpe": 1.2,
            "avg_trade_net_vs_no_hmm": 0.0001,
            "avg_trade_net_vs_same_hour": 0.0001,
            "top_hour_pct": 0.2,
            "top_session_pct": 0.03,
            "max_daily_abs_net_share": 0.1,
        }
    )

    assert classify_rule_row(row, {"state_rules_cross_asset": {}}) == "rule_candidate"
    weak = row.copy()
    weak["avg_trade_net_vs_no_hmm"] = -0.0001
    assert classify_rule_row(weak, {"state_rules_cross_asset": {}}) == "rejected_no_no_hmm_edge"


def test_select_validation_rules_prefers_candidate_thresholds_and_limits_state() -> None:
    base = {
        "candidate_state_id": "s1",
        "bucket": "hmm_state_rule",
        "horizon_bars": 12,
        "cost_bps": 1.0,
        "rule_type": "long_momentum",
        "net_return": 0.01,
        "profit_factor": 1.2,
        "daily_sharpe": 1.2,
        "avg_trade_net_vs_no_hmm": 0.0001,
        "avg_trade_net_vs_same_hour": 0.0001,
    }
    grid = pd.DataFrame(
        [
            {**base, "rule_id": "bad", "rule_status": "rejected_negative_net", "avg_trade_net": 0.001, "hmm_prob_threshold": 0.8, "signal_threshold": 0.0},
            {**base, "rule_id": "good", "rule_status": "rule_candidate", "avg_trade_net": 0.0005, "hmm_prob_threshold": 0.5, "signal_threshold": 0.0},
            {**base, "rule_id": "good", "bucket": "no_hmm_equivalent", "rule_status": "control", "avg_trade_net": 0.0002, "hmm_prob_threshold": 0.5, "signal_threshold": 0.0},
        ]
    )

    selected = select_validation_rules(grid, {"state_rules_cross_asset": {"max_rules_per_state": 1}})

    assert set(selected["rule_id"]) == {"good"}

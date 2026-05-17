from __future__ import annotations

import pandas as pd
import pytest

from src.operable_candidate_search import classify_validation_row, decision_table, position_for_signal, select_specs, thresholds_for_signal


def test_position_for_signal_builds_directional_modes() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.002, -0.002, 0.0],
            "target_dist_vwap_atr": [1.0, -1.0, 0.0],
            "risk_on_score": [2.0, 0.5, 0.0],
            "risk_off_score": [0.0, 2.0, 0.5],
        }
    )

    assert position_for_signal(frame, "momentum_ret_6", 0.001).tolist() == [1.0, -1.0, 0.0]
    assert position_for_signal(frame, "vwap_reversion", 0.5).tolist() == [-1.0, 1.0, 0.0]
    assert position_for_signal(frame, "risk_on_long", 1.0).tolist() == [1.0, 0.0, 0.0]
    assert position_for_signal(frame, "risk_off_short", 1.0).tolist() == [0.0, -1.0, 0.0]


def test_thresholds_for_signal_use_absolute_values_for_signed_modes() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [-0.02, 0.01, 0.03, 0.0],
            "risk_on_score": [-1.0, 0.0, 1.0, 2.0],
        }
    )

    assert thresholds_for_signal(frame, "momentum_ret_6", [0.5]) == pytest.approx([0.015])
    assert thresholds_for_signal(frame, "risk_on_long", [0.5]) == pytest.approx([1.5])


def test_classify_validation_row_requires_operable_edge_after_controls() -> None:
    row = pd.Series(
        {
            "bucket": "hmm_filter",
            "trades": 40,
            "turnover": 2.0,
            "net_return": 0.04,
            "avg_trade_net": 0.001,
            "profit_factor": 1.2,
            "daily_sharpe": 1.2,
            "max_drawdown": 0.05,
            "top_day_abs_net_share": 0.20,
            "net_delta_vs_base": 0.01,
            "drawdown_reduction_vs_base": 0.0,
            "net_delta_vs_same_hour": 0.01,
            "drawdown_reduction_vs_same_hour": 0.0,
        }
    )

    assert classify_validation_row(row, {"operable_candidate_search": {}}) == "operable_validation_candidate"
    weak = row.copy()
    weak["net_delta_vs_same_hour"] = -0.01
    weak["drawdown_reduction_vs_same_hour"] = -0.01
    assert classify_validation_row(weak, {"operable_candidate_search": {}}) == "rejected_no_same_hour_edge"


def test_select_specs_prefers_operable_primary_cost_candidates() -> None:
    common = {
        "feature_set": "cross_asset_full_core",
        "n_states": 3,
        "seed": 42,
        "fold": 0,
        "strategy": "momentum_ret_6",
        "filter_name": "exclude_stress",
        "horizon_bars": 6,
        "selected_hours": "10,11",
        "bucket": "hmm_filter",
        "cost_scenario": "ibkr_tiered_10000",
        "profit_factor": 1.2,
        "drawdown_reduction_vs_base": 0.01,
        "turnover": 1.0,
        "avg_trade_net": 0.001,
    }
    validation = pd.DataFrame(
        [
            {
                **common,
                "candidate_id": "accepted",
                "threshold": 0.002,
                "candidate_status": "operable_validation_candidate",
                "daily_sharpe": 1.0,
                "net_return": 0.02,
            },
            {
                **common,
                "candidate_id": "rejected",
                "threshold": 0.003,
                "candidate_status": "rejected_high_turnover",
                "daily_sharpe": 5.0,
                "net_return": 0.20,
            },
        ]
    )

    selected = select_specs(validation, {"operable_candidate_search": {"max_selected": 1}})

    assert selected["candidate_id"].tolist() == ["accepted"]


def test_decision_table_does_not_accept_low_trade_count_test_result() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "feature_set": "target_only_frozen",
                "n_states": 3,
                "seed": 42,
                "fold": 0,
                "strategy": "supervised_score",
                "filter_name": "only_risk_on",
                "horizon_bars": 24,
                "threshold": 0.05,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "hmm_filter",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "operable_validation_candidate",
            }
        ]
    )
    test_rows = []
    for cost_scenario in ["ibkr_tiered_10000", "bps_2"]:
        test_rows.append(
            {
                "candidate_id": "c1",
                "bucket": "hmm_filter",
                "cost_scenario": cost_scenario,
                "trades": 10,
                "net_return": 0.05,
                "avg_trade_net": 0.001,
                "profit_factor": 1.5,
                "daily_sharpe": 2.0,
                "max_drawdown": 0.02,
                "top_day_abs_net_share": 0.2,
                "turnover": 0.5,
            }
        )
    test = pd.DataFrame(test_rows)

    decisions = decision_table(validation, test, specs, {"operable_candidate_search": {"min_trades": 30}})

    assert decisions.loc[0, "decision"] == "research_candidate"

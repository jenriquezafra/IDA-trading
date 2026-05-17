from __future__ import annotations

import pandas as pd
import pytest

from src.operable_alpha_refinement import alpha_position, alpha_thresholds, confirmation_gates, decision_table


def test_alpha_position_applies_ret12_and_vwap_confirmations() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.003, 0.003, -0.003, -0.003],
            "target_ret_12": [0.002, -0.002, -0.002, 0.002],
            "target_dist_vwap_atr": [1.0, 1.0, -1.0, -1.0],
        }
    )

    assert alpha_position(frame, "m6_base", 0.001).tolist() == [1.0, 1.0, -1.0, -1.0]
    assert alpha_position(frame, "m6_ret12_confirm", 0.001).tolist() == [1.0, 0.0, -1.0, 0.0]
    assert alpha_position(frame, "m6_ret12_vwap", 0.001).tolist() == [1.0, 0.0, -1.0, 0.0]


def test_confirmation_gates_and_directional_risk_are_frozen_from_frame() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.003, -0.003, 0.003, -0.003],
            "target_ret_12": [0.002, -0.002, 0.002, -0.002],
            "risk_on_score": [2.0, 0.0, 0.5, 0.0],
            "risk_off_score": [0.0, 2.0, 0.0, 0.5],
        }
    )
    gates = confirmation_gates(frame, {"operable_alpha_refinement": {"risk_score_quantile": 0.50}})

    assert gates["risk_on_min"] == pytest.approx(0.25)
    assert alpha_position(frame, "m6_ret12_directional_risk", 0.001, gates).tolist() == [1.0, -1.0, 1.0, -1.0]

    strict = {**gates, "risk_on_min": 1.0, "risk_off_min": 1.0}
    assert alpha_position(frame, "m6_ret12_directional_risk", 0.001, strict).tolist() == [1.0, -1.0, 0.0, 0.0]


def test_alpha_thresholds_use_abs_ret6_quantiles() -> None:
    frame = pd.DataFrame({"target_ret_6": [-0.03, -0.01, 0.02, 0.0]})

    assert alpha_thresholds(frame, [0.5]) == pytest.approx([0.02])


def test_decision_table_accepts_candidate_that_passes_primary_and_conservative() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "feature_set": "fs",
                "n_states": 3,
                "seed": 42,
                "fold": 0,
                "alpha_variant": "m6_base",
                "filter_name": "only_risk_on",
                "horizon_bars": 24,
                "threshold": 0.001,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "hmm_filter",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "alpha_validation_candidate",
            }
        ]
    )
    rows = []
    for cost_scenario in ["ibkr_tiered_10000", "bps_2", "bps_5"]:
        rows.append(
            {
                "candidate_id": "c1",
                "bucket": "hmm_filter",
                "cost_scenario": cost_scenario,
                "trades": 50,
                "net_return": 0.04,
                "avg_trade_net": 0.001,
                "profit_factor": 1.2,
                "daily_sharpe": 1.2,
                "max_drawdown": 0.03,
                "top_day_abs_net_share": 0.2,
                "turnover": 1.0,
                "net_delta_vs_base": 0.01,
                "net_delta_vs_same_hour": 0.01,
            }
        )
    test = pd.DataFrame(rows)

    decisions = decision_table(validation, test, specs, {"operable_alpha_refinement": {}})

    assert decisions.loc[0, "decision"] == "accepted_candidate"

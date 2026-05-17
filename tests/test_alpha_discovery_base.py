from __future__ import annotations

import pandas as pd
import pytest

from src.alpha_discovery_base import alpha_position, decision_table, frozen_gates, thresholds_for_variant


def test_alpha_position_supports_directional_and_inverse_variants() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.003, -0.003, 0.0],
            "target_dist_vwap_atr": [1.2, -1.2, 0.0],
            "risk_off_score": [0.2, 2.0, 0.0],
        }
    )

    assert alpha_position(frame, "m6_base", 0.001).tolist() == [1.0, -1.0, 0.0]
    assert alpha_position(frame, "m6_long_only", 0.001).tolist() == [1.0, 0.0, 0.0]
    assert alpha_position(frame, "m6_short_only", 0.001).tolist() == [0.0, -1.0, 0.0]
    assert alpha_position(frame, "vwap_reversion", 0.5).tolist() == [-1.0, 1.0, 0.0]
    assert alpha_position(frame, "risk_off_short", 1.0).tolist() == [0.0, -1.0, 0.0]


def test_alpha_position_applies_frozen_confirmation_gates() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.003, 0.003, -0.003, -0.003],
            "target_ret_12": [0.002, -0.002, -0.002, 0.002],
            "target_dist_vwap_atr": [1.0, 1.0, -1.0, -1.0],
            "risk_on_score": [2.0, 0.0, 0.0, 0.0],
            "risk_off_score": [0.0, 0.0, 2.0, 0.0],
        }
    )

    assert alpha_position(frame, "m6_ret12_confirm", 0.001).tolist() == [1.0, 0.0, -1.0, 0.0]
    gates = {"risk_on_min": 1.0, "risk_off_min": 1.0}
    assert alpha_position(frame, "m6_directional_risk", 0.001, gates).tolist() == [1.0, 0.0, -1.0, 0.0]


def test_thresholds_and_frozen_gates_are_validation_only_calculations() -> None:
    frame = pd.DataFrame(
        {
            "target_ret_6": [-0.03, -0.01, 0.02, 0.0],
            "target_signed_efficiency_12": [-0.5, 0.1, 0.3, 0.0],
            "chop_score": [0.1, 0.2, 0.3, 0.4],
        }
    )

    assert thresholds_for_variant(frame, "m6_base", [0.5]) == pytest.approx([0.02])
    gates = frozen_gates(frame, {"alpha_discovery_base": {"efficiency_abs_quantile": 0.5, "chop_max_quantile": 0.5}})
    assert gates["abs_efficiency_min"] == pytest.approx(0.2)
    assert gates["chop_score_max"] == pytest.approx(0.25)


def test_decision_table_accepts_base_alpha_when_gates_pass() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "fold": 0,
                "alpha_variant": "m6_base",
                "base_variant": "m6_base",
                "horizon_bars": 24,
                "threshold": 0.001,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "alpha_base_validation_candidate",
            }
        ]
    )
    rows = []
    for cost_scenario in ["ibkr_tiered_10000", "bps_2", "bps_5"]:
        rows.append(
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": cost_scenario,
                "trades": 60,
                "net_return": 0.05,
                "avg_trade_net": 0.001,
                "profit_factor": 1.2,
                "daily_sharpe": 1.3,
                "max_drawdown": 0.03,
                "top_day_abs_net_share": 0.2,
                "turnover": 1.0,
                "net_delta_vs_base": 0.0,
                "net_delta_vs_same_hour": 0.0,
            }
        )
    decisions = decision_table(validation, pd.DataFrame(rows), specs, {"alpha_discovery_base": {}})

    assert decisions.loc[0, "decision"] == "accepted_candidate"

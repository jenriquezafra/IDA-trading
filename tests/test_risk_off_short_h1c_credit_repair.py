from __future__ import annotations

import pandas as pd

from src.strategy.risk_off_short_h1c_credit_repair import (
    CreditRepairFilter,
    apply_credit_repair_filter,
    decide_h1c,
    fit_credit_repair_filter,
    select_h1c_validation_variant,
)


def test_credit_spread_lte_zero_filter_is_economic_rule() -> None:
    frame = pd.DataFrame(
        {
            "spread_credit_12": [-0.01, 0.0, 0.01],
            "relret_HYG_LQD_12": [-0.01, 0.0, 0.01],
            "risk_on_score": [1.0, 1.0, 1.0],
            "defensive_rotation_score": [0.0, 0.0, 0.0],
        }
    )
    fitted = CreditRepairFilter(policy="credit_spread_lte_0", thresholds={"spread_credit_12_max": 0.0})

    mask = apply_credit_repair_filter(frame, fitted)

    assert mask.tolist() == [True, True, False]


def test_fit_credit_q50_plus_iqr_sets_looser_threshold() -> None:
    train = pd.DataFrame(
        {
            "spread_credit_12": [-2.0, -1.0, 1.0, 2.0],
            "risk_on_score": [0.0, 1.0, 2.0, 3.0],
            "defensive_rotation_score": [0.0, 1.0, 2.0, 3.0],
        }
    )

    fitted = fit_credit_repair_filter(train, "credit_q50_plus_025iqr")

    assert fitted.thresholds["spread_credit_12_max"] == 0.625


def test_select_h1c_variant_prefers_interpretable_policy_over_anchor() -> None:
    sweep = pd.DataFrame(
        [
            {
                "variant_id": "anchor_high_return",
                "validation_status": "freeze_review",
                "credit_policy_interpretable": False,
                "credit_policy_rank": 6,
                "failed_gate_count": 0,
                "validation_net_return": 0.10,
                "validation_control_edge": 0.08,
                "validation_positive_folds": 5,
                "validation_trades": 100,
            },
            {
                "variant_id": "interpretable_lower_return",
                "validation_status": "freeze_review",
                "credit_policy_interpretable": True,
                "credit_policy_rank": 0,
                "failed_gate_count": 0,
                "validation_net_return": 0.06,
                "validation_control_edge": 0.05,
                "validation_positive_folds": 4,
                "validation_trades": 90,
            },
        ]
    )

    selected = select_h1c_validation_variant(sweep)

    assert selected["variant_id"] == "interpretable_lower_return"


def test_decide_h1c_marks_credit_repaired_with_interpretable_pass() -> None:
    sweep = pd.DataFrame(
        [
            {
                "validation_status": "freeze_review",
                "credit_policy_interpretable": True,
                "credit_policy": "credit_spread_lte_0",
            }
        ]
    )
    selected_decision = {"status": "freeze_review"}
    cost = pd.DataFrame(
        [
            {"split": "validation", "cost_bps": 5.0, "net_return": 0.01},
            {"split": "test", "cost_bps": 5.0, "net_return": 0.01},
        ]
    )

    decision = decide_h1c(sweep, selected_decision, cost)

    assert decision["status"] == "credit_repaired"

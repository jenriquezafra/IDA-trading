from __future__ import annotations

import pandas as pd

from src.strategy.risk_off_short_promotion_sweep import apply_hour_policy, select_validation_variant, variant_id


def test_variant_id_is_stable() -> None:
    assert variant_id(0.7, 0.8, "all") == "riskq70__vixq80__all"


def test_apply_hour_policy_can_exclude_late_hour() -> None:
    signal = pd.Series([True, True, True])
    frame = pd.DataFrame({"hour": [12, 13, 14]})

    filtered = apply_hour_policy(signal, frame, "midday_12_13")

    assert filtered.tolist() == [True, True, False]


def test_select_validation_variant_prefers_gate_pass_over_higher_return() -> None:
    sweep = pd.DataFrame(
        [
            {
                "variant_id": "high_return_failed",
                "validation_status": "continue_research",
                "failed_gate_count": 1,
                "validation_net_return": 0.10,
                "validation_control_edge": 0.08,
                "validation_positive_folds": 5,
                "validation_trades": 100,
            },
            {
                "variant_id": "lower_return_passed",
                "validation_status": "freeze_review",
                "failed_gate_count": 0,
                "validation_net_return": 0.04,
                "validation_control_edge": 0.03,
                "validation_positive_folds": 4,
                "validation_trades": 50,
            },
        ]
    )

    selected = select_validation_variant(sweep)

    assert selected["variant_id"] == "lower_return_passed"

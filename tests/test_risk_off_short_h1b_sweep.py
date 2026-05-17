from __future__ import annotations

import pandas as pd

from src.strategy.risk_off_short_h1b_sweep import (
    H1BFilter,
    apply_h1b_filter,
    h1b_variant_id,
    select_h1b_validation_variant,
)


def test_h1b_variant_id_is_stable() -> None:
    assert h1b_variant_id(0.7, 0.45, "credit_weak_q50") == "riskq70__vixq45__credit_weak_q50"


def test_apply_h1b_filter_can_cap_extreme_vix() -> None:
    frame = pd.DataFrame(
        {
            "prev_vix_z20": [0.1, 1.0, 2.5],
            "risk_on_score": [0.0, 0.0, 0.0],
            "spread_credit_12": [0.0, 0.0, 0.0],
            "defensive_rotation_score": [0.0, 0.0, 0.0],
            "intraday_stress_score": [0.0, 0.0, 0.0],
            "prev_vix9d_vix_ratio": [1.0, 1.0, 1.0],
            "positive_index_count_12": [0, 0, 0],
            "target_dist_vwap_atr": [-1.0, -1.0, -1.0],
        }
    )
    fitted = H1BFilter(policy="vix_cap_q90", thresholds={"vix_cap_q90": 1.5})

    filtered = apply_h1b_filter(frame, fitted)

    assert filtered.tolist() == [True, True, False]


def test_select_h1b_validation_variant_prioritizes_concentration_repair() -> None:
    sweep = pd.DataFrame(
        [
            {
                "variant_id": "high_return_concentrated",
                "validation_status": "continue_research",
                "validation_concentration_repaired": False,
                "validation_avg_trade_gate_pass": True,
                "validation_stress_gate_pass": True,
                "failed_gate_count": 1,
                "validation_max_top5_abs_share": 0.88,
                "validation_net_return": 0.10,
                "validation_control_edge": 0.09,
                "validation_positive_folds": 4,
                "validation_trades": 100,
            },
            {
                "variant_id": "lower_return_repaired",
                "validation_status": "continue_research",
                "validation_concentration_repaired": True,
                "validation_avg_trade_gate_pass": False,
                "validation_stress_gate_pass": True,
                "failed_gate_count": 1,
                "validation_max_top5_abs_share": 0.60,
                "validation_net_return": 0.04,
                "validation_control_edge": 0.03,
                "validation_positive_folds": 4,
                "validation_trades": 130,
            },
        ]
    )

    selected = select_h1b_validation_variant(sweep)

    assert selected["variant_id"] == "lower_return_repaired"

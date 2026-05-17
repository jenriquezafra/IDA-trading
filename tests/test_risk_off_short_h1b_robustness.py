from __future__ import annotations

import pandas as pd

from src.strategy.risk_off_short_h1b_robustness import (
    credit_filter_policy,
    decide_robustness,
    local_quantile_grid,
    selected_quantiles_from_spec,
)


def test_selected_quantiles_from_spec_reads_h1b_rules() -> None:
    raw = {
        "alpha": {
            "rules": [
                {"column": "risk_off_score", "quantile": 0.55},
                {"column": "prev_vix_z20", "quantile": 0.45},
                {"column": "spread_credit_12", "quantile": 0.50},
            ]
        }
    }

    quantiles = selected_quantiles_from_spec(raw)

    assert quantiles == {"risk_off_quantile": 0.55, "vix_quantile": 0.45, "credit_quantile": 0.50}


def test_local_quantile_grid_builds_three_point_neighborhood() -> None:
    assert local_quantile_grid(0.55) == (0.5, 0.55, 0.6)
    assert credit_filter_policy(0.45) == "credit_weak_q45"


def test_decide_robustness_returns_paper_candidate_with_extra_cost_warning() -> None:
    local_sweep = pd.DataFrame(
        [
            {
                "is_anchor": idx == 0,
                "status": "freeze_review",
                "risk_off_quantile": [0.50, 0.55, 0.60, 0.50, 0.55][idx],
                "vix_quantile": [0.40, 0.45, 0.50, 0.45, 0.50][idx],
                "credit_quantile": [0.45, 0.50, 0.55, 0.55, 0.45][idx],
            }
            for idx in range(5)
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"split": "validation", "cost_bps": 5.0, "net_return": 0.02},
            {"split": "test", "cost_bps": 5.0, "net_return": 0.02},
            {"split": "validation", "cost_bps": 7.5, "net_return": -0.001},
            {"split": "test", "cost_bps": 7.5, "net_return": -0.001},
        ]
    )
    fold_stability = pd.DataFrame(
        {
            "trades": [10, 11],
            "top5_abs_share": [0.5, 0.6],
        }
    )

    decision = decide_robustness(local_sweep, cost_sensitivity, fold_stability)

    assert decision["status"] == "paper_candidate"
    assert "extra_stress_7_5bps_not_positive_all_splits" in decision["warnings"]

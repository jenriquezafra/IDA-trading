from __future__ import annotations

import pandas as pd

from src.strategy.risk_off_short_h1c_robustness import decide_robustness, selected_params_from_spec


def test_selected_params_from_spec_reads_h1c_rules() -> None:
    raw = {
        "alpha": {
            "rules": [
                {"column": "risk_off_score", "quantile": 0.50},
                {"column": "prev_vix_z20", "quantile": 0.45},
                {"column": "spread_credit_12", "filter_policy": "credit_spread_lte_0", "value": 0.0},
            ]
        }
    }

    params = selected_params_from_spec(raw)

    assert params == {"risk_off_quantile": 0.50, "vix_quantile": 0.45, "credit_policy": "credit_spread_lte_0"}


def test_decide_robustness_can_mark_h1c_paper_candidate_with_10bps_warning() -> None:
    local_sweep = pd.DataFrame(
        [
            {
                "is_anchor": idx == 0,
                "status": "freeze_review",
                "risk_off_quantile": [0.45, 0.50, 0.55, 0.50][idx],
                "vix_quantile": [0.40, 0.45, 0.50, 0.50][idx],
            }
            for idx in range(4)
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"split": "validation", "cost_bps": 5.0, "net_return": 0.02},
            {"split": "test", "cost_bps": 5.0, "net_return": 0.02},
            {"split": "validation", "cost_bps": 7.5, "net_return": 0.001},
            {"split": "test", "cost_bps": 7.5, "net_return": 0.001},
            {"split": "validation", "cost_bps": 10.0, "net_return": -0.02},
            {"split": "test", "cost_bps": 10.0, "net_return": -0.02},
        ]
    )
    fold_stability = pd.DataFrame({"trades": [20, 25], "top5_abs_share": [0.5, 0.6]})

    decision = decide_robustness(local_sweep, cost_sensitivity, fold_stability)

    assert decision["status"] == "paper_candidate"
    assert "extra_stress_10bps_not_positive_all_splits" in decision["warnings"]

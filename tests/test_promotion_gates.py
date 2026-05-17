from __future__ import annotations

import pandas as pd

from src.research.promotion import evaluate_promotion_gates


def test_generic_promotion_gates_can_pass_for_non_h1_candidate() -> None:
    rows = []
    for split in ("validation", "test"):
        for fold in range(2):
            rows.append(
                {
                    "label": "hypothesis_b_candidate",
                    "fold": fold,
                    "split": split,
                    "cost_bps": 2.0,
                    "trades": 10,
                    "net_return": 0.02,
                    "daily_sharpe": 1.0,
                    "max_drawdown": 0.01,
                }
            )
            rows.append(
                {
                    "label": "simple_control",
                    "fold": fold,
                    "split": split,
                    "cost_bps": 2.0,
                    "trades": 20,
                    "net_return": 0.001,
                    "daily_sharpe": 0.1,
                    "max_drawdown": 0.02,
                }
            )
    primary = pd.DataFrame(rows)
    stress = primary.copy()
    stress["cost_bps"] = 5.0
    stress.loc[stress["label"].eq("hypothesis_b_candidate"), "net_return"] = 0.01
    stress.loc[stress["label"].eq("simple_control"), "net_return"] = -0.01
    concentration = pd.DataFrame(
        {
            "split": ["validation", "validation", "test", "test"],
            "fold": [0, 1, 0, 1],
            "sessions_with_trades": [10, 10, 10, 10],
            "top5_abs_share": [0.4, 0.4, 0.4, 0.4],
        }
    )

    gates, decision = evaluate_promotion_gates(
        pd.concat([primary, stress], ignore_index=True),
        concentration,
        {
            "min_validation_trades": 10,
            "min_test_trades": 10,
            "min_validation_positive_folds": 2,
            "min_test_positive_folds": 2,
            "min_validation_net_return": 0.0,
            "min_test_net_return": 0.0,
            "min_avg_trade_net_bps": 5.0,
            "stress_cost_bps": 5.0,
            "min_validation_stress_net_return": 0.0,
            "min_test_stress_net_return": 0.0,
            "min_sessions_per_fold": 8,
            "max_top5_abs_share": 0.7,
            "require_beats_best_control": True,
        },
        candidate_label="hypothesis_b_candidate",
    )

    assert decision["status"] == "freeze_review"
    assert set(gates["status"]) == {"pass"}

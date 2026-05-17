from __future__ import annotations

import pandas as pd
import pytest

from src.hmm_state_economics import classify_candidate, evaluate_state_action


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "fwd_ret": [0.002, -0.001, 0.003, -0.002],
            "ret_3": [0.003, -0.003, 0.004, -0.004],
            "neutral_zone": [0.001, 0.001, 0.001, 0.001],
        }
    )


def test_evaluate_state_action_applies_round_trip_costs() -> None:
    metrics = evaluate_state_action(_frame(), "long", cost_bps=1.0)

    assert metrics["trades"] == 4
    assert metrics["gross_return"] == pytest.approx(0.002)
    assert metrics["total_cost"] == pytest.approx(0.0004)
    assert metrics["net_return"] == pytest.approx(0.0016)
    assert metrics["avg_trade_net"] == pytest.approx(0.0004)


def test_evaluate_state_action_supports_momentum_and_reversion() -> None:
    momentum = evaluate_state_action(_frame(), "momentum_ret_3", cost_bps=1.0)
    reversion = evaluate_state_action(_frame(), "reversion_ret_3", cost_bps=1.0)

    assert momentum["net_return"] > 0
    assert reversion["net_return"] < 0


def test_classify_candidate_requires_economic_strength_and_stability() -> None:
    candidate = pd.Series(
        {
            "total_trades": 100,
            "total_net_return": 0.05,
            "avg_trade_net": 0.0005,
            "median_profit_factor": 1.2,
            "median_daily_sharpe": 1.1,
            "positive_folds": 10,
            "negative_folds": 3,
        }
    )
    weak = candidate.copy()
    weak["median_profit_factor"] = 1.0

    assert classify_candidate(candidate) == "candidate"
    assert classify_candidate(weak) == "weak_profit_factor"

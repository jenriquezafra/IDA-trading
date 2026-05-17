from __future__ import annotations

import pandas as pd
import pytest

from src.candidate_cost_sensitivity_cross_asset import (
    ibkr_order_cost_usd,
    label_candidates,
    max_drawdown,
    max_drawdown_duration,
    threshold_variants,
)


def test_ibkr_tiered_minimum_commission_dominates_small_order() -> None:
    cfg = {
        "tiered_commission_per_share_usd": 0.0035,
        "tiered_min_commission_per_order_usd": 0.35,
        "max_commission_pct_trade_value": 0.01,
        "tiered_clearing_per_share_per_side_usd": 0.00020,
        "sec_fee_rate_on_sell": 0.0000206,
        "finra_taf_per_share_on_sell_usd": 0.000195,
        "finra_taf_cap_usd": 9.79,
    }

    buy_cost = ibkr_order_cost_usd(2_500.0, 500.0, "tiered", cfg, is_sell=False)
    sell_cost = ibkr_order_cost_usd(2_500.0, 500.0, "tiered", cfg, is_sell=True)

    assert buy_cost == pytest.approx(0.351)
    assert sell_cost == pytest.approx(0.403475)


def test_drawdown_and_duration_are_computed_from_equity_curve() -> None:
    returns = pd.Series([0.02, -0.01, -0.03, 0.005, 0.02, -0.001])

    assert max_drawdown(returns) == pytest.approx(0.04)
    assert max_drawdown_duration(returns) == 5


def test_threshold_variants_include_coarse_grid_and_local_perturbations() -> None:
    config = {
        "hmm_risk_filter": {"return_thresholds": [0.0, 0.0001, 0.0002]},
        "candidate_cost_sensitivity_cross_asset": {
            "threshold_multipliers": [0.5, 1.0, 1.5],
            "threshold_additive": {"momentum_simple": [-0.0001, 0.0, 0.0001]},
        },
    }

    variants = threshold_variants(config, "momentum_simple", 0.0002)

    assert {0.0, 0.0001, 0.0002, 0.0003}.issubset(set(variants["threshold"]))


def test_label_candidates_marks_cost_fragile_when_two_bps_fails() -> None:
    config = {
        "candidate_cost_sensitivity_cross_asset": {
            "acceptance": {
                "primary_cost_scenario": "bps_1",
                "conservative_cost_scenario": "bps_2",
                "ibkr_required_scenario": "ibkr_tiered_10000",
                "min_daily_sharpe": 1.0,
                "min_profit_factor": 1.10,
                "max_drawdown": 0.20,
                "max_top_day_abs_net_share": 0.40,
                "min_threshold_pass_rate": 0.50,
                "min_horizon_pass_rate": 0.50,
            }
        }
    }
    base = {
        "filter_id": "f1",
        "candidate_id": "c1",
        "family_id": "fam",
        "split": "test",
        "feature_set": "fs",
        "n_states": 3,
        "seed": 7,
        "fold": 0,
        "strategy": "momentum_simple",
        "filter_name": "only_risk_on",
        "horizon_bars": 6,
        "threshold": 0.0,
        "avg_trade_net": 0.0001,
        "max_drawdown": 0.05,
        "top_day_abs_net_share": 0.20,
    }
    cost_frame = pd.DataFrame(
        [
            {**base, "cost_scenario": "bps_1", "net_return": 0.05, "daily_sharpe": 1.2, "profit_factor": 1.2},
            {**base, "cost_scenario": "bps_2", "net_return": 0.01, "daily_sharpe": 0.5, "profit_factor": 1.05},
            {**base, "cost_scenario": "ibkr_tiered_10000", "net_return": 0.02, "daily_sharpe": 1.1, "profit_factor": 1.2, "effective_cost_bps": 1.8},
        ]
    )
    threshold_frame = cost_frame.copy()
    horizon_frame = cost_frame.copy()

    decisions = label_candidates(cost_frame, threshold_frame, horizon_frame, config)

    assert decisions.loc[0, "decision_label"] == "cost-fragile"

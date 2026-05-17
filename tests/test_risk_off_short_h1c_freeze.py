from __future__ import annotations

import pandas as pd

from src.research.splits import ResearchFold
from src.strategy import StrategySpec
from src.strategy.freeze_risk_off_short_h1c import DEFAULT_STRATEGY_SPEC_PATH, build_h1c_fold_thresholds


def test_h1c_strategy_spec_yaml_validates() -> None:
    spec = StrategySpec.from_yaml(DEFAULT_STRATEGY_SPEC_PATH)

    assert spec.strategy_id == "qqq_15min_risk_off_short_h1c_v1"
    assert spec.target_symbol == "QQQ"
    assert spec.position.side == "short_only"
    assert spec.exit_rule.horizon_bars == 6


def test_build_h1c_fold_thresholds_freezes_fixed_credit_rule() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-03", "2024-02-02", "2024-03-02"],
            "target_ret_6": [-0.01, -0.02, -0.03, -0.04],
            "target_ret_12": [-0.01, -0.02, -0.03, -0.04],
            "risk_off_score": [1.0, 3.0, 5.0, 7.0],
            "prev_vix_z20": [0.2, 0.8, 1.2, 1.6],
            "spread_credit_12": [-0.5, 0.5, -2.0, -3.0],
            "risk_on_score": [1.0, 1.0, 1.0, 1.0],
            "defensive_rotation_score": [0.0, 0.0, 0.0, 0.0],
            "hour": [10, 11, 12, 13],
        }
    )
    fold = ResearchFold(
        fold=0,
        train_months=("2024-01",),
        validation_months=("2024-02",),
        test_months=("2024-03",),
        train_sessions=("2024-01-02", "2024-01-03"),
        validation_sessions=("2024-02-02",),
        test_sessions=("2024-03-02",),
    )

    thresholds = build_h1c_fold_thresholds(
        frame,
        (fold,),
        risk_off_quantile=0.50,
        vix_quantile=0.50,
        credit_policy="credit_spread_lte_0",
    )

    row = thresholds.iloc[0]
    assert row["risk_off_min"] == 2.0
    assert row["vix_z20_min"] == 0.5
    assert row["spread_credit_12_max_threshold"] == 0.0
    assert row["credit_policy"] == "credit_spread_lte_0"

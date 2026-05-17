from __future__ import annotations

import pandas as pd
import pytest

from src.alpha import alpha_position, fit_confirmation_gates, load_alpha_research_plan, thresholds_for_spec
from src.alpha.specs import AlphaSpec


def test_load_alpha_research_plan_from_yaml() -> None:
    plan = load_alpha_research_plan("configs/alpha/alpha_research_v1.yaml")

    assert plan.research_id == "alpha_research_v1"
    assert plan.target_symbol == "QQQ"
    assert plan.timeframe == "15min"
    assert plan.feature_set_id == "cross_asset_liquid_15min"
    assert "target_ret_6" in plan.required_feature_columns
    assert plan.feature_path().as_posix() == "data/features/QQQ/15min/core_cross_asset_v1/cross_asset_liquid_15min/features.parquet"


def test_alpha_position_from_declarative_spec() -> None:
    spec = AlphaSpec.from_mapping(
        {
            "alpha_id": "m6_ret12_confirm",
            "family": "intraday_momentum",
            "signal_column": "target_ret_6",
            "mode": "signed",
            "horizons": [2],
            "threshold_quantiles": [0.5],
            "confirmations": [{"type": "same_sign", "column": "target_ret_12"}],
        }
    )
    frame = pd.DataFrame(
        {
            "target_ret_6": [0.003, 0.003, -0.003, -0.003],
            "target_ret_12": [0.002, -0.002, -0.002, 0.002],
        }
    )

    assert alpha_position(frame, spec, 0.001).tolist() == [1.0, 0.0, -1.0, 0.0]


def test_thresholds_and_quantile_gates_are_fit_from_validation_frame() -> None:
    spec = AlphaSpec.from_mapping(
        {
            "alpha_id": "m6_low_stress",
            "family": "intraday_momentum",
            "signal_column": "target_ret_6",
            "mode": "signed",
            "horizons": [2],
            "threshold_quantiles": [0.5],
            "confirmations": [{"type": "max_quantile", "column": "intraday_stress_score", "quantile": 0.5}],
        }
    )
    frame = pd.DataFrame(
        {
            "target_ret_6": [-0.03, -0.01, 0.02, 0.0],
            "intraday_stress_score": [0.1, 0.2, 0.3, 0.4],
        }
    )

    gates = fit_confirmation_gates(frame, [spec])

    assert thresholds_for_spec(frame, spec) == pytest.approx((0.02,))
    assert gates["m6_low_stress:max_quantile:intraday_stress_score:q0.5"] == pytest.approx(0.25)
    assert alpha_position(frame, spec, 0.005, gates).tolist() == [-1.0, -1.0, 0.0, 0.0]

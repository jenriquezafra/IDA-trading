from __future__ import annotations

import pandas as pd

from src.setup_signal_shape_stability import parse_rule_shape, rule_shape_key, shape_stability_table


def test_rule_shape_uses_quantile_not_absolute_threshold() -> None:
    params = (
        '{"close_location_min": 0.55, "direction": "long", '
        '"rel_volume_min": 0.8039603907012034, "rel_volume_q": 0.5, "vwap_min": 0.0, '
        '"filter_set": "margin_cumvol_breadth", "breakout_margin_q": 0.67, '
        '"breakout_margin_min": 0.123, "rel_volume_accel_q": 0.5, "rel_volume_accel_min": 1.1, '
        '"positive_index_min": 2, "max_upper_wick": 0.35, '
        '"bias_attempts_min": 1, "vwap_floor": -0.25, "vwap_ceiling": 0.75, '
        '"dist_open_min": 0.0, "min_minutes_from_open": 30}'
    )

    assert parse_rule_shape(params) == {
        "close_location_min": 0.55,
        "rel_volume_q": 0.5,
        "vwap_min": 0.0,
        "filter_set": "margin_cumvol_breadth",
        "breakout_margin_q": 0.67,
        "rel_volume_accel_q": 0.5,
        "positive_index_min": 2,
        "max_upper_wick": 0.35,
        "bias_attempts_min": 1,
        "vwap_floor": -0.25,
        "vwap_ceiling": 0.75,
        "dist_open_min": 0.0,
        "min_minutes_from_open": 30,
    }
    assert "rel_volume_min" not in rule_shape_key(params)
    assert "breakout_margin_min" not in rule_shape_key(params)
    assert "rel_volume_accel_min" not in rule_shape_key(params)
    assert "filter_set=margin_cumvol_breadth" in rule_shape_key(params)
    assert "vwap_ceiling=0.75" in rule_shape_key(params)


def test_shape_stability_requires_every_fold_to_pass_validation_and_stress() -> None:
    ranked = pd.DataFrame(
        [
            {
                "candidate_id": "a0",
                "fold": 0,
                "rule_shape": "shape=a",
                "anti_status": "anti_concentration_candidate",
                "trades": 50,
                "net_return": 0.02,
                "avg_trade_net": 0.0004,
                "leave_one_month_min_net": 0.002,
                "top_month_abs_net_share_rebuilt": 0.30,
            },
            {
                "candidate_id": "a1",
                "fold": 1,
                "rule_shape": "shape=a",
                "anti_status": "rejected_top_month_dependency",
                "trades": 55,
                "net_return": 0.01,
                "avg_trade_net": 0.0002,
                "leave_one_month_min_net": -0.001,
                "top_month_abs_net_share_rebuilt": 0.40,
            },
        ]
    )
    test = pd.DataFrame(
        [
            {
                "candidate_id": "a0",
                "fold": 0,
                "rule_shape": "shape=a",
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "trades": 50,
                "net_return": 0.02,
                "avg_trade_net": 0.0004,
            },
            {
                "candidate_id": "a1",
                "fold": 1,
                "rule_shape": "shape=a",
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "trades": 55,
                "net_return": 0.01,
                "avg_trade_net": 0.0002,
            },
            {
                "candidate_id": "a0",
                "fold": 0,
                "rule_shape": "shape=a",
                "bucket": "setup_signal",
                "cost_scenario": "bps_5",
                "trades": 50,
                "net_return": 0.002,
                "avg_trade_net": 0.00004,
            },
            {
                "candidate_id": "a1",
                "fold": 1,
                "rule_shape": "shape=a",
                "bucket": "setup_signal",
                "cost_scenario": "bps_5",
                "trades": 55,
                "net_return": -0.001,
                "avg_trade_net": -0.00002,
            },
        ]
    )
    config = {
        "setup_signal_search": {
            "primary_cost_scenario": "ibkr_tiered_10000",
            "stress_cost_scenario": "bps_5",
        },
        "setup_signal_anti_concentration": {
            "min_trades": 40,
        },
    }

    stability = shape_stability_table(ranked, test, config)
    row = stability.iloc[0]

    assert row["validation_anti_candidate_folds"] == 1
    assert not row["stable_validation_shape"]
    assert row["test_primary_positive_folds"] == 2
    assert row["test_stress_nonnegative_folds"] == 1
    assert not row["stable_test_shape"]

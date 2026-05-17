from __future__ import annotations

import pandas as pd
import pytest

from src.setup_signal_search import decision_table, evaluate_candidate, generate_family_specs, signal_mask


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=8, freq="5min"),
            "session": ["2024-01-02"] * 8,
            "month": ["2024-01"] * 8,
            "fwd_ret": [0.003, 0.002, -0.001, 0.004, -0.002, -0.003, 0.001, 0.002],
            "target_open_next": [100.0] * 8,
            "target_above_or_6_high": [True, True, False, True, False, False, True, False],
            "target_below_or_6_low": [False, False, True, False, True, True, False, False],
            "target_breaks_roll_high_12": [True, True, False, True, False, False, True, False],
            "target_breaks_roll_low_12": [False, False, True, False, True, True, False, False],
            "target_failed_breakout_high_12": [False, False, True, False, True, False, False, False],
            "target_failed_breakout_low_12": [False, False, False, False, True, True, False, False],
            "target_rel_volume_by_bar": [1.5, 1.2, 1.1, 1.4, 1.6, 1.7, 0.8, 0.9],
            "target_dist_vwap_atr": [0.4, 0.2, -0.4, 0.5, -0.6, -0.8, 0.1, 0.0],
            "target_close_location_bar": [0.8, 0.7, 0.2, 0.9, 0.2, 0.1, 0.6, 0.5],
            "target_upper_wick_ratio": [0.1, 0.2, 0.7, 0.1, 0.8, 0.2, 0.1, 0.2],
            "target_lower_wick_ratio": [0.2, 0.1, 0.1, 0.2, 0.6, 0.7, 0.2, 0.2],
            "target_range_ratio_6_24": [0.6, 0.7, 0.6, 0.5, 0.5, 0.4, 0.8, 0.9],
            "target_breakout_margin_or_6_high_atr": [0.5, 0.1, -0.2, 0.6, -0.4, -0.5, 0.1, 0.0],
            "target_breakdown_margin_or_6_low_atr": [-0.5, -0.1, 0.4, -0.6, 0.7, 0.8, -0.1, 0.0],
            "target_above_or_6_high_persist_2": [1, 2, 0, 1, 0, 0, 1, 0],
            "target_below_or_6_low_persist_2": [0, 0, 1, 0, 1, 2, 0, 0],
            "target_breakout_attempt_count_or_6_high": [1, 2, 2, 3, 3, 3, 4, 4],
            "target_breakout_attempt_count_or_6_low": [0, 0, 1, 1, 2, 3, 3, 3],
            "target_rel_cum_volume_by_bar": [1.4, 1.3, 1.1, 1.5, 1.6, 1.7, 0.9, 0.8],
            "target_rel_volume_accel_2": [1.2, 0.9, 1.0, 1.3, 1.4, 1.5, 0.8, 0.7],
            "target_rv_12_rel_by_bar": [1.1, 0.8, 1.0, 1.2, 1.3, 1.4, 0.9, 0.7],
            "target_minutes_to_close": [360, 355, 350, 345, 340, 335, 330, 325],
            "positive_index_count_open": [4, 3, 1, 4, 0, 1, 3, 4],
            "positive_sector_count_open": [6, 5, 2, 6, 1, 1, 5, 6],
            "index_above_vwap_count": [4, 3, 1, 4, 0, 1, 3, 4],
            "sector_above_vwap_count": [6, 5, 2, 6, 1, 1, 5, 6],
            "relopen_QQQ_SPY": [0.001, -0.001, -0.002, 0.002, -0.003, -0.004, 0.0, 0.0],
            "relopen_IWM_SPY": [0.001, 0.0, -0.002, 0.002, -0.003, -0.004, 0.0, 0.0],
            "risk_on_open_confirmation": [1.5, 1.0, 0.2, 1.6, 0.1, 0.1, 0.8, 0.7],
            "target_overnight_ret": [-0.02, -0.01, 0.02, 0.01, -0.03, 0.03, 0.0, 0.01],
            "target_gap_fill_progress": [0.6, 0.4, 0.7, 0.2, 0.8, 0.5, 0.1, 0.1],
            "target_dist_open": [0.001, 0.0005, -0.001, 0.002, -0.002, -0.003, 0.0001, -0.0001],
            "target_minutes_from_open": [0, 5, 10, 15, 20, 25, 30, 35],
            "risk_off_score": [0.1, 0.2, 0.8, 0.1, 0.9, 1.0, 0.0, 0.1],
            "positive_index_count_6": [4, 3, 1, 4, 0, 1, 3, 4],
            "positive_sector_count_6": [6, 5, 2, 6, 1, 1, 5, 6],
        }
    )


def test_signal_mask_keeps_long_and_short_setups_separate() -> None:
    frame = _frame()
    long_params = {
        "direction": "long",
        "rel_volume_min": 1.0,
        "vwap_min": 0.25,
        "close_location_min": 0.70,
    }
    short_params = {
        "direction": "short",
        "rel_volume_min": 1.0,
        "vwap_min": 0.25,
        "close_location_min": 0.70,
    }

    assert signal_mask(frame, "opening_range_breakout", long_params).tolist() == [True, False, False, True, False, False, False, False]
    assert signal_mask(frame, "opening_range_breakout", short_params).tolist() == [False, False, True, False, True, True, False, False]


def test_signal_mask_accepts_configured_setup_columns() -> None:
    frame = _frame()
    frame["custom_above_or_high"] = [False, True, False, False, False, False, False, False]
    frame["custom_breaks_high"] = False
    params = {
        "direction": "long",
        "rel_volume_min": 1.0,
        "vwap_min": 0.25,
        "close_location_min": 0.70,
    }

    result = signal_mask(
        frame,
        "opening_range_breakout",
        params,
        {"opening_high": "custom_above_or_high", "breaks_roll_high": "custom_breaks_high"},
    )

    assert result.tolist() == [False, False, False, False, False, False, False, False]


def test_signal_mask_applies_optional_opening_breakout_confirmation_filters() -> None:
    frame = _frame()
    params = {
        "direction": "long",
        "rel_volume_min": 1.0,
        "vwap_min": 0.25,
        "close_location_min": 0.70,
        "breakout_margin_min": 0.55,
        "rel_cum_volume_min": 1.4,
        "rel_volume_accel_min": 1.2,
        "positive_index_open_min": 4,
        "sector_above_vwap_min": 6,
        "relopen_qqq_spy_min": 0.0,
        "max_upper_wick": 0.2,
        "min_minutes_to_close": 300,
    }

    assert signal_mask(frame, "opening_range_breakout", params).tolist() == [False, False, False, True, False, False, False, False]


def test_signal_mask_supports_opening_bias_followthrough_after_breakout_attempt() -> None:
    frame = _frame()
    params = {
        "direction": "long",
        "bias_attempts_min": 2,
        "rel_volume_min": 1.0,
        "vwap_floor": -0.25,
        "vwap_ceiling": 0.6,
        "dist_open_min": 0.0,
        "close_location_min": 0.55,
        "positive_index_min": 3,
        "min_minutes_from_open": 5,
    }

    assert signal_mask(frame, "opening_bias_followthrough", params).tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_generate_family_specs_freezes_thresholds_from_validation_frame() -> None:
    specs = generate_family_specs(_frame(), {"setup_signal_search": {"families": ["failed_breakout"]}}, fold=0, horizon=12)

    assert not specs.empty
    assert set(specs["family"]) == {"failed_breakout"}
    assert {"long", "short"}.issubset(set(specs["direction"]))
    assert specs["candidate_id"].is_unique


def test_generate_family_specs_can_emit_opening_breakout_confirmation_specs() -> None:
    specs = generate_family_specs(
        _frame(),
        {
            "setup_signal_search": {
                "families": ["opening_range_breakout"],
                "rel_volume_quantiles": [0.5],
                "close_location_thresholds": [0.7],
                "vwap_abs_mins": [0.25],
                "breakout_margin_quantiles": [0.5],
                "opening_breakout_confirmations": {
                    "enabled": True,
                    "directions": ["long"],
                    "rel_volume_quantiles": [0.5],
                    "close_location_thresholds": [0.7],
                    "vwap_abs_mins": [0.25],
                    "filter_sets": [
                        {
                            "name": "confirm",
                            "breakout_margin_q": 0.5,
                            "rel_cum_volume_q": 0.5,
                            "positive_index_open_min": 3,
                        }
                    ],
                },
            }
        },
        fold=0,
        horizon=12,
    )

    confirmed = specs[specs["params_json"].str.contains("filter_set")]

    assert not confirmed.empty
    assert confirmed["params_json"].str.contains("breakout_margin_min").any()
    assert confirmed["params_json"].str.contains("positive_index_open_min").any()


def test_generate_family_specs_can_emit_opening_bias_followthrough_specs() -> None:
    specs = generate_family_specs(
        _frame(),
        {
            "setup_signal_search": {
                "families": ["opening_bias_followthrough"],
                "rel_volume_quantiles": [0.5],
                "opening_bias_followthrough": {
                    "enabled": True,
                    "directions": ["long"],
                    "rel_volume_quantiles": [0.5],
                    "close_location_thresholds": [0.55],
                    "vwap_floors": [-0.25],
                    "vwap_ceilings": [0.75],
                    "filter_sets": [
                        {
                            "name": "liquid_bias",
                            "bias_attempts_min": 1,
                            "positive_index_min": 3,
                            "min_minutes_from_open": 5,
                        }
                    ],
                },
            }
        },
        fold=0,
        horizon=12,
    )

    assert not specs.empty
    assert set(specs["family"]) == {"opening_bias_followthrough"}
    assert specs["params_json"].str.contains("bias_attempts_min").any()
    assert specs["params_json"].str.contains("vwap_ceiling").any()


def test_evaluate_candidate_compares_signal_to_base_segment() -> None:
    spec = pd.Series(
        {
            "candidate_id": "c1",
            "fold": 0,
            "family": "opening_range_breakout",
            "direction": "long",
            "horizon_bars": 12,
            "params_json": '{"close_location_min": 0.7, "direction": "long", "rel_volume_min": 1.0, "vwap_min": 0.25}',
        }
    )
    rows = evaluate_candidate(_frame(), spec, "validation", {"cost_scenario": "bps_2", "cost_kind": "bps", "cost_bps": 2.0})
    signal = rows[rows["bucket"].eq("setup_signal")].iloc[0]

    assert signal["trades"] == 2
    assert signal["net_return"] == pytest.approx(0.0066)
    assert "net_delta_vs_base_segment" in rows.columns


def test_decision_table_marks_cost_fragile_when_stress_fails() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "fold": 0,
                "family": "opening_range_breakout",
                "direction": "long",
                "horizon_bars": 12,
                "params_json": "{}",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "setup_validation_candidate",
            }
        ]
    )
    test = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "trades": 50,
                "net_return": 0.03,
                "avg_trade_net": 0.0006,
                "profit_factor": 1.2,
                "max_drawdown": 0.05,
                "top_day_abs_net_share": 0.1,
                "top_month_abs_net_share": 0.2,
                "daily_sharpe": 1.0,
                "net_delta_vs_base_segment": 0.01,
            },
            {
                "candidate_id": "c1",
                "bucket": "setup_signal",
                "cost_scenario": "bps_2",
                "net_return": 0.02,
                "avg_trade_net": 0.0004,
            },
            {
                "candidate_id": "c1",
                "bucket": "setup_signal",
                "cost_scenario": "bps_5",
                "net_return": -0.01,
                "avg_trade_net": -0.0002,
            },
        ]
    )

    decisions = decision_table(validation, test, specs, {"setup_signal_search": {}})

    assert decisions.loc[0, "decision"] == "cost_fragile"

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.excess_reversion_search import (
    evaluate_candidate,
    excess_reversion_position,
    train_thresholds,
    validation_grid,
)


def _config() -> dict:
    return {
        "candidate_cost_sensitivity_cross_asset": {"cost_bps": [1.0], "ibkr": {"enabled": False}},
        "excess_reversion_search": {
            "candidate_split": "validation",
            "horizons": [2],
            "variants": ["extension_only"],
            "sides": ["short"],
            "vwap_quantiles": [0.5],
            "open_quantiles": [0.5],
            "range_quantiles": [0.5],
            "volume_accel_quantiles": [0.5],
            "close_location_short_max": [0.55],
            "close_location_long_min": [0.45],
            "cost_scenarios": ["bps_1"],
            "primary_cost_scenario": "bps_1",
            "min_trades": 1,
            "require_random_improvement": False,
            "require_inverted_improvement": False,
            "require_extension_improvement": False,
            "min_minutes_from_open": 45,
            "min_minutes_to_close": 30,
            "max_trades_per_day": 2,
        },
    }


def _rows(split: str, values: list[float], session: str) -> pd.DataFrame:
    timestamps = pd.date_range(f"{session} 10:30", periods=len(values), freq="15min", tz="America/New_York")
    return pd.DataFrame(
        {
            "fold": 0,
            "split": split,
            "horizon_bars": 2,
            "timestamp": timestamps,
            "session": session,
            "bar_index": range(4, 4 + len(values)),
            "hour": timestamps.hour,
            "target_ret_3": 0.0,
            "target_open_next": 100.0,
            "entry_px": 100.0,
            "exit_px": 99.0,
            "fwd_ret": -0.001,
            "proposed_label": "no_hmm",
            "target_dist_vwap_atr": values,
            "target_dist_open": [value / 100.0 for value in values],
            "target_range_ratio_2_8": [1.0 + value / 10.0 for value in values],
            "target_rel_volume_accel_2": [0.4 + value / 100.0 for value in values],
            "target_close_location_bar": 0.5,
            "target_minutes_from_open": 60.0,
            "target_minutes_to_close": 240.0,
            "target_rv_4_rel_by_bar": 1.0,
            "positive_index_count_2": 2,
            "positive_sector_count_2": 4,
            "sector_above_vwap_count": 4,
            "index_above_vwap_count": 2,
            "spread_credit_12": 0.0,
            "spread_equity_bonds_12": 0.0,
            "cross_asset_vol_expansion_score": 1.0,
            "intraday_stress_score": 0.0,
            "risk_on_score": 0.0,
            "risk_off_score": 0.0,
        }
    )


def test_validation_grid_uses_train_thresholds_not_validation_values() -> None:
    dataset = pd.concat(
        [
            _rows("train", [1.0, 2.0], "2024-01-02"),
            _rows("validation", [100.0, 200.0], "2024-07-02"),
        ],
        ignore_index=True,
    )

    grid = validation_grid(dataset, _config())

    signal = grid[grid["bucket"].eq("alpha_signal")].iloc[0]
    thresholds = json.loads(signal["thresholds_json"])
    assert thresholds["vwap_abs_min"] == pytest.approx(1.5)
    assert thresholds["open_abs_min"] == pytest.approx(0.015)
    assert thresholds["range_ratio_min"] == pytest.approx(1.15)


def test_excess_reversion_position_generates_fades_and_caps_by_session() -> None:
    frame = _rows("validation", [2.0, -2.0, 3.0, -3.0], "2024-07-02")
    spec = {
        "variant": "extension_only",
        "side": "both",
        "use_range_filter": True,
        "max_trades_per_day": 2,
    }
    thresholds = {
        "vwap_abs_min": 1.0,
        "open_abs_min": 0.01,
        "range_ratio_min": 0.5,
        "min_minutes_from_open": 45.0,
        "min_minutes_to_close": 30.0,
        "close_location_short_max": 0.55,
        "close_location_long_min": 0.45,
    }

    position = excess_reversion_position(frame, spec, thresholds, _config())

    assert position.tolist() == [-1.0, 1.0, 0.0, 0.0]


def test_evaluate_candidate_emits_required_negative_controls() -> None:
    frame = _rows("validation", [2.0, 2.5, 3.0, 3.5], "2024-07-02")
    spec = {
        "candidate_id": "c1",
        "fold": 0,
        "variant": "extension_only",
        "side": "short",
        "horizon_bars": 2,
        "vwap_quantile": 0.5,
        "open_quantile": 0.5,
        "range_quantile": 0.5,
        "volume_accel_quantile": 0.5,
        "close_location_short_max": 0.55,
        "close_location_long_min": 0.45,
        "risk_filter_name": "none",
        "thresholds_json": json.dumps(
            {
                "vwap_abs_min": 1.0,
                "open_abs_min": 0.01,
                "range_ratio_min": 1.0,
                "volume_accel_max": 1.0,
                "min_minutes_from_open": 45.0,
                "min_minutes_to_close": 30.0,
                "close_location_short_max": 0.55,
                "close_location_long_min": 0.45,
            }
        ),
    }
    scenario = {"cost_scenario": "bps_1", "cost_kind": "bps", "cost_bps": 1.0}

    rows = evaluate_candidate(frame, "validation", spec, scenario, _config())

    assert set(rows["bucket"]) == {
        "alpha_signal",
        "extension_only_control",
        "target_only_control",
        "same_hour_random_control",
        "inverted_signal",
        "always_flat",
    }
    trades = rows.set_index("bucket")["trades"]
    assert trades["same_hour_random_control"] == trades["alpha_signal"]

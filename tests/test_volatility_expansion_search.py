from __future__ import annotations

import json

import pandas as pd
import pytest

from src.volatility_expansion_search import decision_table, evaluate_candidate, expansion_position, validation_grid


def _config() -> dict:
    return {
        "candidate_cost_sensitivity_cross_asset": {"cost_bps": [1.0], "ibkr": {"enabled": False}},
        "volatility_expansion_search": {
            "candidate_split": "validation",
            "horizons": [2],
            "variants": ["compression_breakout"],
            "sides": ["long"],
            "compression_quantiles": [0.5],
            "rv_compression_quantiles": [0.5],
            "breakout_margin_quantiles": [0.5],
            "volume_quantiles": [0.5],
            "vol_filters": ["none"],
            "cost_scenarios": ["bps_1"],
            "primary_cost_scenario": "bps_1",
            "min_trades": 1,
            "require_random_improvement": False,
            "require_inverted_improvement": False,
            "require_breakout_improvement": False,
            "min_minutes_from_open": 45,
            "min_minutes_to_close": 30,
            "max_trades_per_day": 2,
            "close_location_long_min": 0.65,
            "close_location_short_max": 0.35,
        },
    }


def _rows(
    split: str,
    prior_range: list[float],
    session: str,
    *,
    breakout: list[bool] | None = None,
    fwd_ret: list[float] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range(f"{session} 10:30", periods=len(prior_range), freq="15min", tz="America/New_York")
    breakout = breakout if breakout is not None else [True] * len(prior_range)
    fwd_ret = fwd_ret if fwd_ret is not None else [0.001] * len(prior_range)
    return pd.DataFrame(
        {
            "fold": 0,
            "split": split,
            "horizon_bars": 2,
            "timestamp": timestamps,
            "session": session,
            "bar_index": range(4, 4 + len(prior_range)),
            "hour": timestamps.hour,
            "target_open_next": 100.0,
            "entry_px": 100.0,
            "exit_px": 101.0,
            "fwd_ret": fwd_ret,
            "proposed_label": "no_hmm",
            "target_ret_3": 0.001,
            "target_range_ratio_2_8": 1.0,
            "target_rv_4_rel_by_bar": 1.0,
            "prior_target_range_ratio_2_8": prior_range,
            "prior_target_rv_4_rel_by_bar": prior_range,
            "prior_target_rel_volume_by_bar": 0.8,
            "target_breaks_roll_high_4": breakout,
            "target_breaks_roll_low_4": [False] * len(prior_range),
            "target_breakout_margin_roll_high_4_atr": [0.10, 0.20, 0.30, 0.40][: len(prior_range)],
            "target_breakdown_margin_roll_low_4_atr": 0.0,
            "target_close_location_bar": 0.80,
            "target_rel_volume_by_bar": 1.2,
            "target_rel_volume_accel_2": 1.1,
            "target_minutes_from_open": 60.0,
            "target_minutes_to_close": 240.0,
            "positive_index_count_2": 2,
            "positive_sector_count_2": 4,
            "index_above_vwap_count": 2,
            "sector_above_vwap_count": 4,
            "spread_credit_12": 0.0,
            "risk_on_score": 0.0,
            "risk_off_score": 0.0,
            "intraday_stress_score": 0.0,
            "cross_asset_vol_expansion_score": 0.0,
        }
    )


def test_validation_grid_uses_train_thresholds_not_validation_values() -> None:
    dataset = pd.concat(
        [
            _rows("train", [0.2, 0.4], "2024-01-02"),
            _rows("validation", [99.0, 100.0], "2024-07-02"),
        ],
        ignore_index=True,
    )

    grid = validation_grid(dataset, _config())

    assert "compression_quantile" in grid.columns
    assert "breakout_margin_quantile" in grid.columns
    assert "compression_quantile_x" not in grid.columns
    assert "breakout_margin_quantile_x" not in grid.columns
    signal = grid[grid["bucket"].eq("alpha_signal")].iloc[0]
    thresholds = json.loads(signal["thresholds_json"])
    assert thresholds["range_compression_max"] == pytest.approx(0.3)
    assert thresholds["rv_compression_max"] == pytest.approx(0.3)
    assert thresholds["breakout_margin_min"] == pytest.approx(0.15)


def test_expansion_position_uses_prior_compression_and_current_breakout() -> None:
    frame = _rows("validation", [float("nan"), 0.20, 0.80], "2024-07-02", breakout=[True, True, True])
    spec = {
        "variant": "compression_breakout",
        "side": "long",
        "max_trades_per_day": 2,
        "vol_filter_name": "none",
    }
    thresholds = {
        "range_compression_max": 0.30,
        "rv_compression_max": 0.30,
        "breakout_margin_min": 0.05,
        "min_minutes_from_open": 45.0,
        "min_minutes_to_close": 30.0,
        "close_location_long_min": 0.65,
        "close_location_short_max": 0.35,
    }

    position = expansion_position(frame, spec, thresholds, _config())

    assert position.tolist() == [0.0, 1.0, 0.0]


def test_expansion_position_applies_hour_filter() -> None:
    frame = _rows("validation", [0.20, 0.20, 0.20], "2024-07-02", breakout=[True, True, True])
    frame["hour"] = [11, 12, 13]
    spec = {
        "variant": "compression_breakout",
        "side": "long",
        "max_trades_per_day": 2,
        "vol_filter_name": "none",
        "hour_filter_name": "hour_12",
    }
    thresholds = {
        "range_compression_max": 0.30,
        "rv_compression_max": 0.30,
        "breakout_margin_min": 0.05,
        "min_minutes_from_open": 45.0,
        "min_minutes_to_close": 30.0,
        "close_location_long_min": 0.65,
        "close_location_short_max": 0.35,
    }

    position = expansion_position(frame, spec, thresholds, _config())

    assert position.tolist() == [0.0, 1.0, 0.0]


def test_evaluate_candidate_emits_expansion_controls() -> None:
    frame = _rows("validation", [0.20, 0.20, 0.20], "2024-07-02")
    spec = {
        "candidate_id": "c1",
        "fold": 0,
        "variant": "compression_breakout",
        "side": "long",
        "horizon_bars": 2,
        "compression_quantile": 0.5,
        "rv_compression_quantile": 0.5,
        "breakout_margin_quantile": 0.5,
        "volume_quantile": 0.5,
        "vol_filter_name": "none",
        "max_trades_per_day": 2,
        "thresholds_json": json.dumps(
            {
                "range_compression_max": 0.30,
                "rv_compression_max": 0.30,
                "breakout_margin_min": 0.05,
                "min_minutes_from_open": 45.0,
                "min_minutes_to_close": 30.0,
                "close_location_long_min": 0.65,
                "close_location_short_max": 0.35,
            }
        ),
    }
    scenario = {"cost_scenario": "bps_1", "cost_kind": "bps", "cost_bps": 1.0}

    rows = evaluate_candidate(frame, "validation", spec, scenario, _config())

    assert set(rows["bucket"]) == {
        "alpha_signal",
        "breakout_only_control",
        "compression_only_control",
        "same_hour_random_control",
        "inverted_signal",
        "always_flat",
    }
    trades = rows.set_index("bucket")["trades"]
    assert trades["same_hour_random_control"] == trades["alpha_signal"]


def test_decision_table_never_accepts_specs_that_failed_validation() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "fold": 0,
                "variant": "compression_breakout",
                "side": "long",
                "horizon_bars": 2,
                "vol_filter_name": "none",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "rejected_negative_edge",
            }
        ]
    )
    test_rows = []
    for cost_scenario in ["ibkr_tiered_10000", "bps_2", "bps_5"]:
        test_rows.append(
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": cost_scenario,
                "trades": 100,
                "net_return": 0.05,
                "avg_trade_net": 0.0005,
                "profit_factor": 1.5,
                "daily_sharpe": 1.2,
                "max_drawdown": 0.02,
                "top_day_abs_net_share": 0.1,
                "turnover": 1.0,
                "net_delta_vs_random": 0.01,
                "net_delta_vs_inverted": 0.02,
            }
        )

    decisions = decision_table(validation, pd.DataFrame(test_rows), specs, {"volatility_expansion_search": {}})

    assert decisions.loc[0, "decision"] == "rejected_validation_failed"


def test_decision_table_requires_stress_cost_for_acceptance() -> None:
    specs = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "fold": 0,
                "variant": "compression_breakout",
                "side": "long",
                "horizon_bars": 4,
                "vol_filter_name": "none",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "candidate_status": "volatility_expansion_validation_candidate",
            }
        ]
    )
    test_rows = []
    for cost_scenario, net_return in [
        ("ibkr_tiered_10000", 0.05),
        ("bps_2", 0.04),
        ("bps_5", -0.01),
    ]:
        test_rows.append(
            {
                "candidate_id": "c1",
                "bucket": "alpha_signal",
                "cost_scenario": cost_scenario,
                "trades": 100,
                "net_return": net_return,
                "avg_trade_net": net_return / 100,
                "profit_factor": 1.5,
                "daily_sharpe": 1.2,
                "max_drawdown": 0.02,
                "top_day_abs_net_share": 0.1,
                "turnover": 1.0,
                "net_delta_vs_random": 0.01,
                "net_delta_vs_inverted": 0.02,
            }
        )

    decisions = decision_table(validation, pd.DataFrame(test_rows), specs, {"volatility_expansion_search": {}})

    assert decisions.loc[0, "decision"] == "cost_fragile"

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.cross_asset_divergence_search import decision_table, divergence_position, evaluate_candidate, train_thresholds, validation_grid


def _config() -> dict:
    return {
        "candidate_cost_sensitivity_cross_asset": {"cost_bps": [1.0], "ibkr": {"enabled": False}},
        "cross_asset_divergence_search": {
            "candidate_split": "validation",
            "horizons": [2],
            "variants": ["relative_leadership_fade"],
            "sides": ["short"],
            "move_quantiles": [0.5],
            "recent_quantiles": [0.5],
            "relative_quantiles": [0.5],
            "vol_filters": ["none"],
            "cost_scenarios": ["bps_1"],
            "primary_cost_scenario": "bps_1",
            "min_trades": 1,
            "require_random_improvement": False,
            "require_inverted_improvement": False,
            "require_target_only_improvement": False,
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
            "target_open_next": 100.0,
            "entry_px": 100.0,
            "exit_px": 99.0,
            "fwd_ret": -0.001,
            "proposed_label": "no_hmm",
            "target_dist_open": values,
            "target_ret_3": values,
            "target_minutes_from_open": 60.0,
            "target_minutes_to_close": 240.0,
            "positive_index_count_2": 2,
            "positive_sector_count_2": 4,
            "sector_above_vwap_count": 4,
            "index_above_vwap_count": 2,
            "sector_rel_strength_count_2": 3,
            "relret_QQQ_SPY_6": values,
            "relret_IWM_SPY_6": [0.0] * len(values),
            "relret_DIA_SPY_6": [0.0] * len(values),
            "relret_XLK_SPY_6": values,
            "relret_HYG_LQD_6": 0.0,
            "spread_credit_12": 0.0,
            "spread_equity_bonds_12": 0.0,
            "spread_growth_defensive_12": 0.0,
            "spread_cyclicals_defensive_12": 0.0,
            "risk_on_score": 0.0,
            "risk_off_score": 0.0,
            "intraday_stress_score": 0.0,
            "cross_asset_vol_expansion_score": 1.0,
            "target_rv_4_rel_by_bar": 1.0,
        }
    )


def test_validation_grid_uses_train_thresholds_not_validation_values() -> None:
    dataset = pd.concat(
        [
            _rows("train", [0.01, 0.02], "2024-01-02"),
            _rows("validation", [0.50, 0.60], "2024-07-02"),
        ],
        ignore_index=True,
    )

    grid = validation_grid(dataset, _config(), "SPY")

    signal = grid[grid["bucket"].eq("alpha_signal")].iloc[0]
    thresholds = json.loads(signal["thresholds_json"])
    assert thresholds["target_move_abs_min"] == pytest.approx(0.015)
    assert thresholds["target_recent_abs_min"] == pytest.approx(0.015)
    assert thresholds["qqq_spy_high"] == pytest.approx(0.015)


def test_relative_leadership_fade_generates_spy_short_when_growth_leads() -> None:
    frame = _rows("validation", [0.02, -0.02, 0.03], "2024-07-02")
    spec = {"variant": "relative_leadership_fade", "side": "short", "max_trades_per_day": 2, "vol_filter_name": "none"}
    thresholds = {
        "target_move_abs_min": 0.01,
        "target_recent_abs_min": 0.01,
        "min_minutes_from_open": 45.0,
        "min_minutes_to_close": 30.0,
        "qqq_spy_high": 0.015,
        "xlk_spy_high": 0.015,
        "iwm_spy_high": 0.01,
        "dia_spy_high": 0.01,
        "credit_high": 0.01,
    }

    position = divergence_position(frame, spec, thresholds, _config(), "SPY")

    assert position.tolist() == [-1.0, 0.0, -1.0]


def test_evaluate_candidate_emits_negative_controls() -> None:
    frame = _rows("validation", [0.02, 0.025, 0.03], "2024-07-02")
    spec = {
        "candidate_id": "c1",
        "fold": 0,
        "variant": "relative_leadership_fade",
        "side": "short",
        "horizon_bars": 2,
        "move_quantile": 0.5,
        "recent_quantile": 0.5,
        "relative_quantile": 0.5,
        "vol_filter_name": "none",
        "thresholds_json": json.dumps(
            {
                "target_move_abs_min": 0.01,
                "target_recent_abs_min": 0.01,
                "min_minutes_from_open": 45.0,
                "min_minutes_to_close": 30.0,
                "qqq_spy_high": 0.015,
                "xlk_spy_high": 0.015,
                "iwm_spy_high": 0.01,
                "dia_spy_high": 0.01,
                "credit_high": 0.01,
            }
        ),
    }
    scenario = {"cost_scenario": "bps_1", "cost_kind": "bps", "cost_bps": 1.0}

    rows = evaluate_candidate(frame, "validation", spec, scenario, _config(), "SPY")

    assert set(rows["bucket"]) == {
        "alpha_signal",
        "target_only_control",
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
                "variant": "breadth_nonconfirm",
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

    decisions = decision_table(validation, pd.DataFrame(test_rows), specs, {"cross_asset_divergence_search": {}})

    assert decisions.loc[0, "decision"] == "rejected_validation_failed"

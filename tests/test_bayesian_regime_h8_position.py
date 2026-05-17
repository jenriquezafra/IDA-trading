from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.bayesian_regime_h8_position import (
    _position_from_probabilities,
    _lifecycle_position_for_params,
    lifecycle_position_from_entry_exit,
    lifecycle_position_from_signal,
    prepare_supervised_frame,
    run,
    select_lifecycle_candidate,
)


def _features(months: int = 5, sessions_per_month: int = 5, bars: int = 8) -> pd.DataFrame:
    rows = []
    price = 100.0
    for month in range(months):
        start = pd.Timestamp("2024-01-02") + pd.DateOffset(months=month)
        sessions = pd.date_range(start, periods=sessions_per_month, freq="B")
        month_direction = 1.0 if month % 2 == 0 else -1.0
        for session_idx, session_ts in enumerate(sessions):
            session = session_ts.strftime("%Y-%m-%d")
            timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="15min", tz="America/New_York")
            for bar_index, timestamp in enumerate(timestamps):
                drift = month_direction * (0.001 + 0.0001 * bar_index)
                price *= float(np.exp(drift))
                ret_1 = drift
                rows.append(
                    {
                        "timestamp": timestamp,
                        "session": session,
                        "bar_index": bar_index,
                        "target_ret_1": ret_1,
                        "target_ret_2": ret_1 * 2.0,
                        "target_ret_3": ret_1 * 3.0,
                        "target_ret_4": ret_1 * 4.0,
                        "target_rv_12_rel_by_bar": 0.8 + 0.05 * ((bar_index + session_idx + month) % 5),
                        "target_signed_efficiency_12": month_direction * (0.5 + 0.02 * bar_index),
                        "target_dist_vwap_atr": month_direction * 0.1,
                        "target_pos_session_range": 0.75 if month_direction > 0 else 0.25,
                        "risk_on_score": month_direction * 0.01,
                        "risk_off_score": -month_direction * 0.01,
                        "target_open_next": price * np.exp(drift),
                    }
                )
    return pd.DataFrame(rows)


def _config(tmp_path: Path, features_path: Path) -> dict:
    return {
        "lab": {"target_symbol": "QQQ"},
        "bayesian_regime_h8": {
            "features_file": str(features_path),
            "results_dir": str(tmp_path / "h8-results"),
            "report_file": str(tmp_path / "h8-reports/h8.md"),
            "models_dir": str(tmp_path / "h8-models"),
            "momentum_column": "target_ret_4",
            "volatility_column": "target_rv_12_rel_by_bar",
            "efficiency_column": "target_signed_efficiency_12",
            "variants": ["manual_h8a"],
            "walk_forward": {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1},
            "max_folds": 1,
            "probability_thresholds": [0.55],
            "max_entropy_values": [None],
            "horizons": [1],
            "cost_bps": [1.0],
        },
        "h8_position_model": {
            "results_dir": str(tmp_path / "h8c-results"),
            "report_file": str(tmp_path / "h8c-reports/h8c.md"),
            "models_dir": str(tmp_path / "h8c-models"),
            "selected_h8_variant": "manual_h8a",
            "horizon_bars": 1,
            "label_cost_bps": 1.0,
            "label_cost_scenario": "ibkr_tiered_10000",
            "primary_cost_bps": 1.0,
            "include_regime_dynamics_features": True,
            "use_expected_net_gate": True,
            "expected_net_threshold_bps": [0.0],
            "min_expected_net_gaps_bps": [0.0],
            "cost_bps": [1.0],
            "probability_thresholds": [0.5],
            "min_probability_gaps": [0.0],
            "min_validation_trades": 1,
            "max_iter": 500,
            "model_feature_columns": [
                "target_ret_1",
                "target_ret_2",
                "target_ret_4",
                "target_dist_vwap_atr",
                "target_pos_session_range",
                "risk_on_score",
                "risk_off_score",
            ],
        },
        "candidate_cost_sensitivity_cross_asset": {
            "cost_bps": [1.0, 5.0],
            "ibkr": {
                "enabled": True,
                "plans": ["tiered"],
                "notionals_usd": [10000],
                "spread_slippage_bps_round_trip": 1.5,
                "tiered_commission_per_share_usd": 0.0035,
                "tiered_min_commission_per_order_usd": 0.35,
                "tiered_clearing_per_share_per_side_usd": 0.0002,
                "max_commission_pct_trade_value": 0.01,
                "sec_fee_rate_on_sell": 0.0000206,
                "finra_taf_per_share_on_sell_usd": 0.000195,
                "finra_taf_cap_usd": 9.79,
            },
        },
    }


def _posteriors(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.reset_index(names="source_index").loc[:, ["source_index", "timestamp", "session", "bar_index"]].copy()
    frame.insert(0, "variant", "manual_h8a")
    frame.insert(1, "fold", 0)
    frame.insert(2, "split", "train")
    frame["p_bull_trend"] = np.where(features["target_ret_1"] > 0.0, 0.75, 0.10)
    frame["p_bear_stress"] = np.where(features["target_ret_1"] < 0.0, 0.75, 0.10)
    frame["p_chop_noise"] = 0.10
    frame["p_shock_reversal"] = 0.05
    frame["max_prob"] = frame[["p_bull_trend", "p_bear_stress", "p_chop_noise", "p_shock_reversal"]].max(axis=1)
    frame["entropy"] = 0.4
    frame["mom_z"] = features["target_ret_4"]
    frame["vol_z"] = np.log(features["target_rv_12_rel_by_bar"])
    frame["eff_z"] = features["target_signed_efficiency_12"]
    return frame


def test_prepare_supervised_frame_builds_profit_labels() -> None:
    features = _features(months=1)
    config = {"h8_position_model": {"horizon_bars": 1, "label_cost_bps": 1.0, "model_feature_columns": ["target_ret_1", "risk_on_score"]}}

    frame, columns = prepare_supervised_frame(features, _posteriors(features), config)

    assert "p_bull_trend" in columns
    assert "target_ret_1" in columns
    assert "posterior_margin" in columns
    assert "dominant_state_age_bars" in columns
    assert {"label_long_profit", "label_short_profit", "p_bull_trend"}.issubset(frame.columns)
    assert frame["horizon_bars"].eq(1).all()


def test_prepare_supervised_frame_can_use_ibkr_label_cost() -> None:
    features = _features(months=1)
    config = _config(Path("/tmp"), Path("/tmp/features.parquet"))

    frame, _columns = prepare_supervised_frame(features, _posteriors(features), config)

    assert frame["label_cost_scenario"].eq("ibkr_tiered_10000").all()
    assert frame["label_cost_return"].gt(0.0).all()
    assert frame["label_cost_effective_bps"].mean() > 1.0


def test_expected_net_gate_requires_predicted_edge() -> None:
    frame = pd.DataFrame(
        {
            "p_long_profit": [0.70, 0.70, 0.40],
            "p_short_profit": [0.20, 0.20, 0.80],
            "expected_long_net": [0.0003, 0.00005, -0.0001],
            "expected_short_net": [-0.0001, 0.0000, 0.00025],
        }
    )

    position = _position_from_probabilities(
        frame,
        threshold=0.50,
        min_probability_gap=0.10,
        min_expected_net_bps=2.0,
        min_expected_net_gap_bps=1.0,
    )

    assert position.tolist() == [1.0, 0.0, -1.0]


def test_lifecycle_position_flips_and_exits_without_overlap() -> None:
    signal = pd.Series([1.0, 0.0, -1.0, -1.0, 0.0, 0.0])
    sessions = pd.Series(["2024-01-02"] * len(signal))

    position = lifecycle_position_from_signal(signal, sessions, max_hold_bars=2, exit_on_signal_loss=False)
    exit_on_loss = lifecycle_position_from_signal(signal, sessions, max_hold_bars=2, exit_on_signal_loss=True)

    assert position.tolist() == [1.0, 1.0, -1.0, -1.0, -1.0, 0.0]
    assert exit_on_loss.tolist() == [1.0, 0.0, -1.0, -1.0, 0.0, 0.0]


def test_entry_exit_lifecycle_uses_hysteresis_and_cooldown() -> None:
    entry = pd.Series([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0])
    hold = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0])
    sessions = pd.Series(["2024-01-02"] * len(entry))

    position = lifecycle_position_from_entry_exit(
        entry,
        hold,
        sessions,
        max_hold_bars=0,
        min_hold_bars=1,
        cooldown_bars=1,
        exit_on_signal_loss=True,
    )

    assert position.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0]


def test_h8e_setup_filter_limits_entries_to_matching_setup_direction() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 4,
            "p_long_profit": [0.80, 0.80, 0.80, 0.20],
            "p_short_profit": [0.10, 0.10, 0.10, 0.80],
            "expected_long_net": [0.0003, 0.0003, 0.0003, -0.0001],
            "expected_short_net": [-0.0001, -0.0001, -0.0001, 0.0004],
            "p_bull_trend": [0.80, 0.80, 0.80, 0.10],
            "p_bear_stress": [0.10, 0.10, 0.10, 0.80],
            "entropy": [0.3, 0.3, 0.3, 0.3],
            "target_above_or_6_high": [True, True, False, False],
            "target_breaks_roll_high_12": [False, False, False, False],
            "target_rel_volume_by_bar": [1.2, 0.7, 1.3, 1.3],
            "target_dist_vwap_atr": [0.4, 0.4, 0.4, 0.4],
            "target_close_location_bar": [0.8, 0.8, 0.8, 0.8],
        }
    )
    params = {
        "entry_exit_mode": True,
        "setup_entry_mode": True,
        "setup_family": "opening_range_breakout",
        "setup_direction": "long",
        "setup_params_json": json.dumps(
            {"direction": "long", "rel_volume_min": 1.0, "vwap_min": 0.25, "close_location_min": 0.70}
        ),
        "setup_column_map_json": "{}",
        "entry_gate_mode": "regime_confirmed",
        "entry_threshold": 0.50,
        "entry_min_probability_gap": 0.10,
        "entry_expected_net_threshold_bps": 0.0,
        "entry_min_expected_net_gap_bps": 0.0,
        "entry_regime_threshold": 0.65,
        "entry_max_entropy": 0.75,
        "exit_gate_mode": "regime_confirmed",
        "exit_threshold": 0.45,
        "exit_min_probability_gap": 0.0,
        "exit_expected_net_threshold_bps": -1.0,
        "exit_min_expected_net_gap_bps": 0.0,
        "exit_regime_threshold": 0.45,
        "exit_max_entropy": 0.75,
        "max_hold_bars": 1,
        "min_hold_bars": 0,
        "cooldown_bars": 0,
        "exit_on_signal_loss": True,
    }

    position = _lifecycle_position_for_params(frame, params)

    assert position.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_lifecycle_selection_rejects_zero_turnover_candidates() -> None:
    aggregate = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "cost_scenario": ["bps_5", "bps_5"],
            "turnover": [0.0, 4.0],
            "daily_sharpe_mean": [np.nan, 2.0],
            "net_return": [0.0, 0.10],
            "net_per_turnover_pooled": [np.nan, 0.025],
        }
    )
    config = {
        "h8_position_lifecycle": {
            "selection_cost_scenario": "bps_5",
            "min_validation_turnover": 10.0,
        }
    }

    selected = select_lifecycle_candidate(aggregate, config)

    assert selected.empty


def test_h8c_runner_writes_selected_metrics(tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    _features().to_parquet(features_path, index=False)
    config = _config(tmp_path, features_path)
    config_path = tmp_path / "h8c.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report_path, selected_metrics_path = run(config_path)

    assert report_path.exists()
    assert selected_metrics_path.exists()
    selected_metrics = pd.read_parquet(selected_metrics_path)
    cost_sensitivity = pd.read_parquet(tmp_path / "h8c-results/h8c_cost_sensitivity_aggregate.parquet")
    lifecycle_decision = pd.read_parquet(tmp_path / "h8c-results/h8c_lifecycle_decision.parquet")
    assert {"validation", "test"}.issubset(set(selected_metrics["split"]))
    assert {"bps_1", "bps_5", "ibkr_tiered_10000"}.issubset(set(cost_sensitivity["cost_scenario"]))
    assert (cost_sensitivity["effective_cost_bps"] > 0.0).all()
    assert lifecycle_decision["decision"].notna().all()
    assert (tmp_path / "h8c-results/h8c_position_predictions.parquet").exists()
    assert (tmp_path / "h8c-results/h8c_lifecycle_sensitivity.parquet").exists()

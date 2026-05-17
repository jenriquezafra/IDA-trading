from __future__ import annotations

import pandas as pd

from src.hmm_state_interpretability_cross_asset import (
    assign_economic_label,
    build_feature_profiles,
    build_ticker_dependency,
    feature_tickers,
    select_interpretability_combo,
)


def test_select_interpretability_combo_uses_validation_rank() -> None:
    summary = pd.DataFrame(
        [
            {"feature_set": "a", "n_states": 3, "seed": 42, "validation_rank": 2},
            {"feature_set": "b", "n_states": 6, "seed": 7, "validation_rank": 1},
        ]
    )
    config = {"hmm_state_interpretability": {"selected_rank": 1}}

    selected = select_interpretability_combo(summary, config)

    assert selected["feature_set"] == "b"
    assert int(selected["n_states"]) == 6


def test_build_feature_profiles_computes_split_relative_zscore() -> None:
    frame = pd.DataFrame(
        [
            {"fold": 0, "split": "validation", "hmm_state": 0, "timestamp": pd.Timestamp("2024-01-02 09:30"), "session": "2024-01-02", "feature": 1.0},
            {"fold": 0, "split": "validation", "hmm_state": 0, "timestamp": pd.Timestamp("2024-01-02 09:35"), "session": "2024-01-02", "feature": 1.0},
            {"fold": 0, "split": "validation", "hmm_state": 1, "timestamp": pd.Timestamp("2024-01-02 09:40"), "session": "2024-01-02", "feature": 3.0},
            {"fold": 0, "split": "validation", "hmm_state": 1, "timestamp": pd.Timestamp("2024-01-02 09:45"), "session": "2024-01-02", "feature": 3.0},
        ]
    )

    profiles = build_feature_profiles(frame, ["feature"])

    state0 = profiles[(profiles["hmm_state"] == 0) & (profiles["feature"] == "feature")].iloc[0]
    state1 = profiles[(profiles["hmm_state"] == 1) & (profiles["feature"] == "feature")].iloc[0]
    assert state0["state_z"] < 0
    assert state1["state_z"] > 0


def test_assign_economic_label_prefers_risk_off_profile() -> None:
    profile = pd.DataFrame(
        [
            {"feature": "risk_off_score", "state_z": 1.2, "abs_state_z": 1.2},
            {"feature": "intraday_stress_score", "state_z": 1.0, "abs_state_z": 1.0},
            {"feature": "target_ret_12", "state_z": -1.1, "abs_state_z": 1.1},
            {"feature": "risk_on_score", "state_z": -0.8, "abs_state_z": 0.8},
        ]
    )

    label = assign_economic_label(profile, {"min_abs_z_clear": 0.5, "min_score_partial": 0.25})

    assert label["proposed_label"] == "risk_off_stress"
    assert label["best_score"] > 0


def test_feature_tickers_maps_target_and_aggregate_features() -> None:
    known = ["SPY", "TLT", "IEF", "GLD", "HYG", "LQD"]

    assert feature_tickers("target_ret_12", known, "SPY") == ("SPY",)
    assert set(feature_tickers("spread_credit_12", known, "SPY")) == {"HYG", "LQD"}


def test_build_ticker_dependency_reports_top_symbol() -> None:
    profiles = pd.DataFrame(
        [
            {"fold": 0, "hmm_state": 0, "feature": "relret_TLT_SPY_12", "state_z": 2.0, "abs_state_z": 2.0},
            {"fold": 0, "hmm_state": 0, "feature": "risk_off_score", "state_z": 0.5, "abs_state_z": 0.5},
        ]
    )
    feature_config = {"groups": {"bonds": ["TLT"], "indices": ["SPY"], "haven": ["TLT"]}}

    dependency, leave_one = build_ticker_dependency(profiles, feature_config, "SPY")

    assert dependency.loc[0, "top_ticker"] in {"SPY", "TLT"}
    assert not leave_one.empty

from __future__ import annotations

import pandas as pd

from src.hmm_stability_cross_asset import align_seed_states, build_stability_grid, classify_stability_row


def test_align_seed_states_matches_by_profile_not_state_number() -> None:
    profiles = pd.DataFrame(
        [
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "split": "validation", "hmm_state": 0, "feature": "x", "state_z": 1.0},
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "split": "validation", "hmm_state": 0, "feature": "y", "state_z": 0.0},
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "split": "validation", "hmm_state": 1, "feature": "x", "state_z": 0.0},
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "split": "validation", "hmm_state": 1, "feature": "y", "state_z": 1.0},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "split": "validation", "hmm_state": 0, "feature": "x", "state_z": 0.0},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "split": "validation", "hmm_state": 0, "feature": "y", "state_z": 1.0},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "split": "validation", "hmm_state": 1, "feature": "x", "state_z": 1.0},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "split": "validation", "hmm_state": 1, "feature": "y", "state_z": 0.0},
        ]
    )
    names = pd.DataFrame(
        [
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "hmm_state": 0, "proposed_label": "a"},
            {"feature_set": "macro", "n_states": 2, "seed": 1, "fold": 0, "hmm_state": 1, "proposed_label": "b"},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "hmm_state": 0, "proposed_label": "b"},
            {"feature_set": "macro", "n_states": 2, "seed": 2, "fold": 0, "hmm_state": 1, "proposed_label": "a"},
        ]
    )

    alignment = align_seed_states(profiles, names, {"hmm_state_stability": {"reference_seed": "min"}})
    row = alignment[(alignment["seed"] == 2) & (alignment["hmm_state"] == 0)].iloc[0]

    assert row["anchor_hmm_state"] == 1
    assert row["profile_cosine"] == 1.0
    assert row["label_match"]


def test_classify_stability_row_rejects_single_seed_and_accepts_stable() -> None:
    base = pd.Series(
        {
            "proposed_label": "risk_on_trend",
            "seeds_present": 2,
            "folds_present": 2,
            "avg_state_frequency": 0.08,
            "max_top_hour_pct": 0.2,
            "non_target_top_ticker_share_max": 0.0,
            "min_seed_alignment_cosine": 0.8,
        }
    )

    assert classify_stability_row(base, {}) == "stable_profile_candidate"
    weak = base.copy()
    weak["seeds_present"] = 1
    assert classify_stability_row(weak, {}) == "rejected_single_seed"


def test_build_stability_grid_counts_label_presence() -> None:
    states = pd.DataFrame(
        [
            {
                "feature_set": "macro",
                "n_states": 3,
                "seed": 1,
                "fold": 0,
                "hmm_state": 0,
                "local_state_id": "a",
                "proposed_label": "risk_on_trend",
                "state_frequency": 0.1,
                "profile_strength": 1.0,
                "train_validation_label_match": True,
                "top_hour_pct": 0.2,
                "top_session_pct": 0.02,
            },
            {
                "feature_set": "macro",
                "n_states": 3,
                "seed": 2,
                "fold": 1,
                "hmm_state": 1,
                "local_state_id": "b",
                "proposed_label": "risk_on_trend",
                "state_frequency": 0.12,
                "profile_strength": 1.1,
                "train_validation_label_match": True,
                "top_hour_pct": 0.22,
                "top_session_pct": 0.03,
            },
        ]
    )
    alignment = pd.DataFrame(
        [
            {"feature_set": "macro", "n_states": 3, "fold": 0, "seed": 2, "reference_seed": 1, "proposed_label": "risk_on_trend", "profile_cosine": 0.9, "label_match": True},
            {"feature_set": "macro", "n_states": 3, "fold": 1, "seed": 2, "reference_seed": 1, "proposed_label": "risk_on_trend", "profile_cosine": 0.8, "label_match": True},
        ]
    )
    dependency = pd.DataFrame(
        [
            {"local_state_id": "a", "top_ticker": "SPY", "top_ticker_abs_z_share": 0.4, "ticker_count": 4},
            {"local_state_id": "b", "top_ticker": "SPY", "top_ticker_abs_z_share": 0.5, "ticker_count": 4},
        ]
    )
    leave_one = pd.DataFrame(
        [
            {"local_state_id": "a", "ticker_removed": "SPY", "removed_abs_z_share": 0.4, "profile_cosine_after_removal": 0.9},
            {"local_state_id": "b", "ticker_removed": "SPY", "removed_abs_z_share": 0.5, "profile_cosine_after_removal": 0.8},
        ]
    )
    period = pd.DataFrame(
        [
            {"feature_set": "macro", "n_states": 3, "seed": 1, "fold": 0, "split": "validation", "hmm_state": 0, "period_dimension": "year", "period_bucket": "2024", "state_frequency": 0.1},
            {"feature_set": "macro", "n_states": 3, "seed": 2, "fold": 1, "split": "validation", "hmm_state": 1, "period_dimension": "vol_regime", "period_bucket": "high_vol", "state_frequency": 0.1},
            {"feature_set": "macro", "n_states": 3, "seed": 2, "fold": 1, "split": "validation", "hmm_state": 1, "period_dimension": "trend_regime", "period_bucket": "up_past_12", "state_frequency": 0.1},
        ]
    )

    grid = build_stability_grid(states, alignment, dependency, leave_one, period, {"hmm_state_stability": {}}, "SPY")

    assert grid.loc[0, "proposed_label"] == "risk_on_trend"
    assert grid.loc[0, "seeds_present"] == 2
    assert grid.loc[0, "folds_present"] == 2
    assert grid.loc[0, "status"] == "stable_profile_candidate"

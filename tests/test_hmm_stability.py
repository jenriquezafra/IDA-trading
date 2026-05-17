from __future__ import annotations

import pandas as pd

from src.hmm_stability import aggregate_stability_ranking, candidate_holdout_summary, summarize_feature_sets


def test_aggregate_stability_ranking_keeps_k_and_seed_separate() -> None:
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "n_states": 3,
                "seed": 42,
                "split": "validation",
                "action": "long",
                "horizon_bars": 6,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "fold": 0,
                "trades": 100,
                "net_return": 0.02,
                "avg_trade_net": 0.0002,
                "profit_factor": 1.2,
                "daily_sharpe": 1.2,
                "state_frequency": 0.25,
                "persistence": 0.8,
            },
            {
                "feature_set": "a",
                "n_states": 4,
                "seed": 7,
                "split": "validation",
                "action": "long",
                "horizon_bars": 6,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "fold": 0,
                "trades": 100,
                "net_return": -0.01,
                "avg_trade_net": -0.0001,
                "profit_factor": 0.8,
                "daily_sharpe": -1.0,
                "state_frequency": 0.25,
                "persistence": 0.8,
            },
        ]
    )

    ranking = aggregate_stability_ranking(metrics)

    assert set(ranking["n_states"]) == {3, 4}
    assert ranking.loc[ranking["n_states"] == 3, "candidate_status"].iloc[0] == "candidate"
    assert ranking.loc[ranking["n_states"] == 4, "candidate_status"].iloc[0] == "negative_economic"


def test_candidate_holdout_summary_scores_same_validation_candidate_on_test() -> None:
    ranking = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "n_states": 3,
                "seed": 42,
                "horizon_bars": 6,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "action": "long",
                "candidate_status": "candidate",
            }
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "n_states": 3,
                "seed": 42,
                "split": "validation",
                "action": "long",
                "horizon_bars": 6,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "fold": 0,
                "trades": 100,
                "net_return": 0.02,
                "avg_trade_net": 0.0002,
                "profit_factor": 1.2,
                "daily_sharpe": 1.2,
                "state_frequency": 0.25,
                "persistence": 0.8,
            },
            {
                "feature_set": "a",
                "n_states": 3,
                "seed": 42,
                "split": "test",
                "action": "long",
                "horizon_bars": 6,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "fold": 0,
                "trades": 100,
                "net_return": -0.01,
                "avg_trade_net": -0.0001,
                "profit_factor": 0.8,
                "daily_sharpe": -1.0,
                "state_frequency": 0.25,
                "persistence": 0.8,
            },
        ]
    )

    holdout = candidate_holdout_summary(metrics, ranking)

    assert holdout["split"].tolist() == ["validation", "test"]
    assert holdout.loc[holdout["split"] == "test", "candidate_status"].iloc[0] == "negative_economic"


def test_summarize_feature_sets_counts_combo_survival() -> None:
    combo_summary = pd.DataFrame(
        [
            {"feature_set": "a", "n_states": 3, "seed": 42, "cost_bps": 1.0, "validation_candidates": 1, "test_candidates": 1},
            {"feature_set": "a", "n_states": 4, "seed": 42, "cost_bps": 1.0, "validation_candidates": 1, "test_candidates": 0},
            {"feature_set": "b", "n_states": 3, "seed": 42, "cost_bps": 1.0, "validation_candidates": 0, "test_candidates": 0},
        ]
    )

    summary = summarize_feature_sets(combo_summary)

    row = summary.loc[summary["feature_set"] == "a"].iloc[0]
    assert row["combos_with_validation_candidate"] == 2
    assert row["combos_with_test_candidate"] == 1

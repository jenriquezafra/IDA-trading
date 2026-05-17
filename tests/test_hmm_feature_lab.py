from __future__ import annotations

import pandas as pd

from src.hmm_feature_lab import aggregate_feature_set_ranking, candidate_holdout_summary, summarize_feature_sets


def test_aggregate_feature_set_ranking_keeps_sets_separate() -> None:
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "split": "validation",
                "action": "long",
                "horizon_bars": 2,
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
                "feature_set": "b",
                "split": "validation",
                "action": "long",
                "horizon_bars": 2,
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

    ranking = aggregate_feature_set_ranking(metrics)

    assert ranking["feature_set"].tolist() == ["a", "b"]
    assert ranking.loc[ranking["feature_set"] == "a", "candidate_status"].iloc[0] == "candidate"
    assert ranking.loc[ranking["feature_set"] == "b", "candidate_status"].iloc[0] == "negative_economic"


def test_summarize_feature_sets_includes_validation_status() -> None:
    validations = pd.DataFrame(
        [{"feature_set": "a", "n_features": 2, "columns": "x,y", "missing_columns": "", "status": "ready"}]
    )
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "split": "validation",
                "action": "flat",
                "state_frequency": 0.25,
                "persistence": 0.7,
                "mean_duration": 3.0,
            }
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "horizon_bars": 2,
                "cost_bps": 1.0,
                "hmm_state": 0,
                "action": "long",
                "total_trades": 100,
                "total_net_return": 0.01,
                "avg_trade_net": 0.0001,
                "median_profit_factor": 1.2,
                "median_daily_sharpe": 1.1,
                "candidate_status": "candidate",
            }
        ]
    )

    summary = summarize_feature_sets(metrics, ranking, validations)

    assert summary.loc[0, "feature_set"] == "a"
    assert summary.loc[0, "candidate_count"] == 1
    assert summary.loc[0, "best_action"] == "long"


def test_candidate_holdout_summary_pairs_validation_candidates_with_test() -> None:
    ranking = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "horizon_bars": 2,
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
                "split": "validation",
                "action": "long",
                "horizon_bars": 2,
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
                "split": "test",
                "action": "long",
                "horizon_bars": 2,
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
    assert holdout.loc[holdout["split"] == "validation", "candidate_status"].iloc[0] == "candidate"
    assert holdout.loc[holdout["split"] == "test", "candidate_status"].iloc[0] == "negative_economic"

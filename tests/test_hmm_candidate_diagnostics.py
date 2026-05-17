from __future__ import annotations

import pandas as pd

from src.hmm_candidate_diagnostics import select_surviving_candidates, summarize_fold_metrics, summarize_time_concentration


def test_select_surviving_candidates_filters_configured_feature_set_and_cost() -> None:
    holdout = pd.DataFrame(
        [
            {
                "feature_set": "rich_extreme_reversion",
                "split": "test",
                "candidate_status": "candidate",
                "cost_bps": 1.0,
                "n_states": 4,
                "seed": 42,
                "horizon_bars": 6,
                "hmm_state": 3,
                "action": "momentum_ret_3",
            },
            {
                "feature_set": "minimal_vwap_location",
                "split": "test",
                "candidate_status": "candidate",
                "cost_bps": 1.0,
                "n_states": 4,
                "seed": 42,
                "horizon_bars": 6,
                "hmm_state": 3,
                "action": "momentum_ret_3",
            },
            {
                "feature_set": "rich_extreme_reversion",
                "split": "test",
                "candidate_status": "candidate",
                "cost_bps": 2.0,
                "n_states": 4,
                "seed": 42,
                "horizon_bars": 6,
                "hmm_state": 3,
                "action": "momentum_ret_3",
            },
        ]
    )
    config = {
        "hmm_candidate_diagnostics": {
            "feature_sets_to_inspect": ["rich_extreme_reversion"],
            "split": "test",
            "candidate_status": "candidate",
            "cost_bps": [1.0],
        }
    }

    selected = select_surviving_candidates(holdout, config)

    assert len(selected) == 1
    assert selected.loc[0, "candidate_id"] == "rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1"


def test_summarize_fold_metrics_classifies_grouped_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "split": "test",
                "bucket": "candidate_state",
                "fold": 0,
                "trades": 100,
                "net_return": 0.02,
                "avg_trade_net": 0.0002,
                "profit_factor": 1.2,
                "daily_sharpe": 1.2,
            },
            {
                "candidate_id": "a",
                "split": "test",
                "bucket": "candidate_state",
                "fold": 1,
                "trades": 100,
                "net_return": 0.01,
                "avg_trade_net": 0.0001,
                "profit_factor": 1.3,
                "daily_sharpe": 1.3,
            },
        ]
    )

    summary = summarize_fold_metrics(frame, ["candidate_id", "split", "bucket"])

    assert summary.loc[0, "total_trades"] == 200
    assert summary.loc[0, "candidate_status"] == "candidate"


def test_summarize_time_concentration_reports_top_hour() -> None:
    hour_summary = pd.DataFrame(
        [
            {"candidate_id": "a", "split": "test", "hour": 10, "state_pct": 0.2, "hour_lift": 0.8},
            {"candidate_id": "a", "split": "test", "hour": 11, "state_pct": 0.6, "hour_lift": 2.0},
            {"candidate_id": "a", "split": "test", "hour": 12, "state_pct": 0.2, "hour_lift": 1.0},
        ]
    )

    concentration = summarize_time_concentration(hour_summary)

    assert concentration.loc[0, "top_hour"] == 11
    assert concentration.loc[0, "top_hour_state_pct"] == 0.6

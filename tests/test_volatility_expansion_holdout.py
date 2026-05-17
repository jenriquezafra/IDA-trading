from __future__ import annotations

import pandas as pd
import pytest

from src.volatility_expansion_holdout import holdout_frame_for_spec, label_holdout


def test_holdout_frame_starts_after_fold_test_boundary() -> None:
    features = pd.DataFrame(
        {
            "session": pd.date_range("2024-01-01", periods=5, freq="MS").strftime("%Y-%m-%d"),
        }
    )
    dataset = pd.DataFrame(
        {
            "session": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "bar_index": [1, 1, 1, 1, 1],
            "horizon_bars": [4, 4, 4, 4, 4],
        }
    )
    config = {
        "hmm_lab": {
            "walk_forward": {"train_months": 1, "validation_months": 1, "test_months": 1, "step_months": 1},
            "max_folds": 1,
        }
    }
    spec = {"fold": 0, "horizon_bars": 4}

    holdout = holdout_frame_for_spec(dataset, features, spec, config)

    assert holdout["session"].tolist() == ["2024-04-01", "2024-05-01"]
    assert holdout["split"].eq("holdout").all()


def test_label_holdout_passes_clean_candidate() -> None:
    specs = pd.DataFrame([{"candidate_id": "c1"}])
    cost_curve = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "split": "holdout",
                "cost_scenario": "ibkr_tiered_10000",
                "holdout_start_session": "2024-11-01",
                "holdout_end_session": "2026-05-01",
                "holdout_sessions": 370,
                "net_return": 0.04,
                "gross_return": 0.07,
                "avg_trade_net": 0.0005,
                "trades": 80,
            },
            {
                "candidate_id": "c1",
                "split": "holdout",
                "cost_scenario": "bps_5",
                "net_return": 0.02,
                "gross_return": 0.07,
                "avg_trade_net": 0.00025,
                "trades": 80,
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "net_return": 0.01},
            {"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "net_return": -0.001},
            {"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "net_return": 0.02},
        ]
    )
    leave_one = pd.DataFrame(
        [{"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "net_without_period": 0.01}]
    )
    bootstrap = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "prob_total_net_positive": 0.90},
            {"candidate_id": "c1", "split": "holdout", "cost_scenario": "bps_5", "prob_total_net_positive": 0.70},
        ]
    )
    random_summary = pd.DataFrame(
        [{"candidate_id": "c1", "split": "holdout", "cost_scenario": "ibkr_tiered_10000", "prob_random_beats_alpha": 0.05}]
    )

    decisions = label_holdout(specs, cost_curve, monthly, leave_one, bootstrap, random_summary, {"volatility_expansion_holdout": {}})

    assert decisions.loc[0, "holdout_status"] == "holdout_pass"
    assert decisions.loc[0, "breakeven_cost_bps"] == pytest.approx(8.75)


def test_label_holdout_fails_negative_stress() -> None:
    specs = pd.DataFrame([{"candidate_id": "c1"}])
    cost_curve = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "split": "holdout",
                "cost_scenario": "ibkr_tiered_10000",
                "net_return": 0.04,
                "gross_return": 0.07,
                "avg_trade_net": 0.0005,
                "trades": 80,
            },
            {
                "candidate_id": "c1",
                "split": "holdout",
                "cost_scenario": "bps_5",
                "net_return": -0.01,
                "gross_return": 0.07,
                "avg_trade_net": -0.0001,
                "trades": 80,
            },
        ]
    )
    monthly = pd.DataFrame()
    leave_one = pd.DataFrame()
    bootstrap = pd.DataFrame()
    random_summary = pd.DataFrame()

    decisions = label_holdout(specs, cost_curve, monthly, leave_one, bootstrap, random_summary, {"volatility_expansion_holdout": {}})

    assert decisions.loc[0, "holdout_status"] == "holdout_failed"
    assert "stress_net_positive" in decisions.loc[0, "failed_checks"]

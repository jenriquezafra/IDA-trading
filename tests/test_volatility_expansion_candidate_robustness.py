from __future__ import annotations

import pandas as pd
import pytest

from src.volatility_expansion_candidate_robustness import (
    bootstrap_trade_distribution,
    label_robustness,
    leave_one_period_summary,
    select_frozen_specs,
    summarize_periods,
)


def test_select_frozen_specs_prefers_accepted_candidates() -> None:
    specs = pd.DataFrame(
        [
            {"candidate_id": "c1", "thresholds_json": "{}"},
            {"candidate_id": "c2", "thresholds_json": "{}"},
        ]
    )
    decisions = pd.DataFrame(
        [
            {"candidate_id": "c1", "decision": "research_candidate", "test_net_primary": 0.05},
            {"candidate_id": "c2", "decision": "accepted_candidate", "test_net_primary": 0.01},
        ]
    )

    selected = select_frozen_specs(specs, decisions, {"volatility_expansion_candidate_robustness": {}})

    assert selected["candidate_id"].tolist() == ["c2"]
    assert selected.loc[0, "decision"] == "accepted_candidate"


def test_summarize_periods_groups_active_trade_returns() -> None:
    trades = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c1"],
            "split": ["test", "test", "test"],
            "cost_scenario": ["ibkr_tiered_10000"] * 3,
            "month": ["2024-07", "2024-07", "2024-08"],
            "gross_return": [0.01, -0.002, 0.003],
            "cost_return": [0.001, 0.001, 0.001],
            "net_return": [0.009, -0.003, 0.002],
        }
    )

    summary = summarize_periods(trades, "month")

    july = summary[summary["month"].eq("2024-07")].iloc[0]
    assert july["trades"] == 2
    assert july["net_return"] == pytest.approx(0.006)
    assert july["win_rate"] == pytest.approx(0.5)


def test_leave_one_period_summary_removes_each_period() -> None:
    trades = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c1"],
            "split": ["test", "test", "test"],
            "cost_scenario": ["bps_5"] * 3,
            "month": ["2024-07", "2024-08", "2024-08"],
            "net_return": [0.01, -0.002, 0.003],
        }
    )

    leave_one = leave_one_period_summary(trades, "month")

    removed_july = leave_one[leave_one["removed_period"].eq("2024-07")].iloc[0]
    assert removed_july["net_without_period"] == pytest.approx(0.001)


def test_bootstrap_trade_distribution_reports_positive_probability() -> None:
    trades = pd.DataFrame(
        {
            "candidate_id": ["c1"] * 4,
            "split": ["test"] * 4,
            "cost_scenario": ["bps_5"] * 4,
            "net_return": [0.01, 0.02, -0.001, 0.004],
        }
    )

    bootstrap = bootstrap_trade_distribution(trades, samples=200, seed=7)

    row = bootstrap.iloc[0]
    assert row["trades"] == 4
    assert row["prob_total_net_positive"] > 0.90
    assert row["observed_total_net"] == pytest.approx(0.033)


def test_label_robustness_marks_provisional_when_only_random_control_fails() -> None:
    specs = pd.DataFrame([{"candidate_id": "c1"}])
    cost_curve = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "ibkr_tiered_10000",
                "net_return": 0.02,
                "gross_return": 0.03,
                "avg_trade_net": 0.001,
                "trades": 20,
            },
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "bps_5",
                "net_return": 0.01,
                "gross_return": 0.03,
                "avg_trade_net": 0.0005,
                "trades": 20,
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "test", "cost_scenario": "ibkr_tiered_10000", "net_return": 0.01},
            {"candidate_id": "c1", "split": "test", "cost_scenario": "ibkr_tiered_10000", "net_return": 0.01},
        ]
    )
    leave_one = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "test", "cost_scenario": "ibkr_tiered_10000", "net_without_period": 0.01}
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "test", "cost_scenario": "ibkr_tiered_10000", "prob_total_net_positive": 0.90},
            {"candidate_id": "c1", "split": "test", "cost_scenario": "bps_5", "prob_total_net_positive": 0.70},
        ]
    )
    random_summary = pd.DataFrame(
        [
            {"candidate_id": "c1", "split": "test", "cost_scenario": "ibkr_tiered_10000", "prob_random_beats_alpha": 0.25}
        ]
    )

    decisions = label_robustness(specs, cost_curve, monthly, leave_one, bootstrap, random_summary, {"volatility_expansion_candidate_robustness": {}})

    assert decisions.loc[0, "robustness_status"] == "robustness_provisional"
    assert "random_control" in decisions.loc[0, "failed_checks"]

from __future__ import annotations

import pandas as pd
import pytest

from src.setup_signal_diagnostics import diagnostic_findings, select_focus_specs, summarize_returns


def _config() -> dict:
    return {
        "setup_signal_search": {
            "primary_cost_scenario": "ibkr_tiered_10000",
            "stress_cost_scenario": "bps_5",
        },
        "setup_signal_diagnostics": {
            "family": "breakdown_short_risk_off",
            "direction": "short",
            "horizon_bars": 4,
            "max_specs_per_fold": 2,
        },
    }


def test_select_focus_specs_keeps_month_concentration_rejections_for_diagnosis() -> None:
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "f0",
                "fold": 0,
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "family": "breakdown_short_risk_off",
                "direction": "short",
                "horizon_bars": 4,
                "params_json": "{}",
                "column_map_json": "{}",
                "candidate_status": "setup_validation_candidate",
                "trades": 50,
                "net_return": 0.02,
                "avg_trade_net": 0.0004,
                "profit_factor": 1.2,
                "daily_sharpe": 0.8,
                "top_day_abs_net_share": 0.1,
                "top_month_abs_net_share": 0.3,
            },
            {
                "candidate_id": "f1",
                "fold": 1,
                "bucket": "setup_signal",
                "cost_scenario": "ibkr_tiered_10000",
                "family": "breakdown_short_risk_off",
                "direction": "short",
                "horizon_bars": 4,
                "params_json": "{}",
                "column_map_json": "{}",
                "candidate_status": "rejected_month_concentration",
                "trades": 80,
                "net_return": 0.05,
                "avg_trade_net": 0.0006,
                "profit_factor": 1.8,
                "daily_sharpe": 1.4,
                "top_day_abs_net_share": 0.2,
                "top_month_abs_net_share": 0.7,
            },
        ]
    )
    decisions = pd.DataFrame({"candidate_id": ["f0"]})

    specs = select_focus_specs(validation, decisions, _config())

    assert specs["candidate_id"].tolist() == ["f0", "f1"]
    assert specs.loc[specs["candidate_id"].eq("f1"), "validation_positive"].iloc[0]


def test_summarize_returns_groups_focus_bars() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test", "test"],
            "fold": [0, 0, 1],
            "candidate_id": ["a", "a", "b"],
            "net_return": [0.02, -0.01, 0.03],
            "gross_return": [0.03, 0.0, 0.04],
            "cost_return": [0.01, 0.01, 0.01],
            "fwd_ret": [-0.03, 0.0, -0.04],
        }
    )

    summary = summarize_returns(frame, ["split", "fold"])
    fold0 = summary[summary["fold"].eq(0)].iloc[0]

    assert fold0["trades"] == 2
    assert fold0["net_return"] == pytest.approx(0.01)
    assert fold0["hit_rate"] == pytest.approx(0.5)


def test_diagnostic_findings_flags_decay_and_hmm_block() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "candidate_id": "f1",
                "bucket": "setup_signal",
                "fold": 1,
                "split": "validation",
                "cost_scenario": "ibkr_tiered_10000",
                "net_return": 0.05,
                "avg_trade_net": 0.0006,
                "candidate_status": "rejected_month_concentration",
                "top_month_abs_net_share": 0.7,
            },
            {
                "candidate_id": "f1",
                "bucket": "setup_signal",
                "fold": 1,
                "split": "test",
                "cost_scenario": "ibkr_tiered_10000",
                "net_return": -0.01,
                "avg_trade_net": -0.0002,
                "candidate_status": "rejected_month_concentration",
                "top_month_abs_net_share": 0.4,
            },
            {
                "candidate_id": "f0",
                "bucket": "setup_signal",
                "fold": 0,
                "split": "test",
                "cost_scenario": "bps_5",
                "net_return": 0.02,
                "avg_trade_net": 0.0003,
                "candidate_status": "setup_validation_candidate",
                "top_month_abs_net_share": 0.6,
            },
            {
                "candidate_id": "f1",
                "bucket": "setup_signal",
                "fold": 1,
                "split": "test",
                "cost_scenario": "bps_5",
                "net_return": -0.02,
                "avg_trade_net": -0.0004,
                "candidate_status": "rejected_month_concentration",
                "top_month_abs_net_share": 0.4,
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"split": "test", "fold": 0, "month": "2024-01", "net_return": 0.03},
            {"split": "test", "fold": 0, "month": "2024-02", "net_return": 0.01},
        ]
    )
    feature_shift = pd.DataFrame()

    findings = diagnostic_findings(evaluation, monthly, feature_shift, _config())

    assert "fold1_validation_candidates_exist" in findings["finding"].tolist()
    assert "fold1_validation_to_test_decay" in findings["finding"].tolist()
    assert "hmm_still_blocked" in findings["finding"].tolist()


def test_diagnostic_findings_does_not_block_when_selected_candidate_survives_costs() -> None:
    base = {
        "candidate_id": "accepted",
        "bucket": "setup_signal",
        "fold": 1,
        "split": "test",
        "family": "opening_range_breakout",
        "direction": "long",
        "horizon_bars": 2,
        "was_selected": True,
        "trades": 44,
        "profit_factor": 1.49,
        "max_drawdown": 0.02,
        "top_day_abs_net_share": 0.21,
        "top_month_abs_net_share": 0.30,
    }
    evaluation = pd.DataFrame(
        [
            {
                **base,
                "cost_scenario": "ibkr_tiered_10000",
                "net_return": 0.018,
                "avg_trade_net": 0.0004,
            },
            {
                **base,
                "cost_scenario": "bps_2",
                "net_return": 0.017,
                "avg_trade_net": 0.0003,
            },
            {
                **base,
                "cost_scenario": "bps_5",
                "net_return": 0.004,
                "avg_trade_net": 0.0001,
            },
        ]
    )

    findings = diagnostic_findings(evaluation, pd.DataFrame(), pd.DataFrame(), _config())

    assert "focused_candidate_accepted_but_family_unstable" in findings["finding"].tolist()
    assert "hmm_still_blocked" not in findings["finding"].tolist()

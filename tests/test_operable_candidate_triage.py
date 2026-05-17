from __future__ import annotations

import pandas as pd

from src.operable_candidate_triage import build_triage_frame, failure_counts, gate_failures, triage_label


def test_gate_failures_marks_sparse_and_concentrated_candidate() -> None:
    row = pd.Series(
        {
            "primary_trades": 10,
            "primary_net_return": 0.02,
            "primary_avg_trade_net": 0.001,
            "primary_profit_factor": 1.5,
            "primary_daily_sharpe": 1.4,
            "primary_max_drawdown": 0.02,
            "primary_top_day_abs_net_share": 0.5,
            "primary_turnover": 0.2,
            "primary_net_delta_vs_base": 0.01,
            "primary_net_delta_vs_same_hour": 0.01,
            "primary_drawdown_reduction_vs_base": 0.01,
            "primary_drawdown_reduction_vs_same_hour": 0.01,
            "conservative_net_return": 0.01,
            "stress_net_return": -0.01,
            "validation_net_return": 0.03,
        }
    )

    failures = gate_failures(row, {"operable_candidate_triage": {"min_trades": 30, "max_top_day_abs_net_share": 0.35}})

    assert "insufficient_test_trades" in failures
    assert "concentrated_test_pnl" in failures
    assert "fails_5bps_stress_cost" in failures


def test_triage_label_prefers_no_incrementality_before_low_sharpe() -> None:
    row = pd.Series(
        {
            "decision": "research_candidate",
            "primary_net_return": 0.02,
            "primary_profit_factor": 1.2,
            "failure_reasons": "weak_test_sharpe,no_base_incrementality",
        }
    )

    assert triage_label(row, {"operable_candidate_triage": {}}) == "no_hmm_incrementality"


def test_build_triage_frame_merges_test_controls_and_counts_failures() -> None:
    decisions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "feature_set": "fs",
                "n_states": 3,
                "seed": 42,
                "fold": 0,
                "strategy": "momentum_ret_6",
                "filter_name": "only_risk_on",
                "horizon_bars": 24,
                "threshold": 0.1,
                "validation_status": "operable_validation_candidate",
                "decision": "research_candidate",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "cost_scenario": "ibkr_tiered_10000",
                "bucket": "hmm_filter",
                "net_return": 0.04,
                "trades": 40,
            }
        ]
    )
    test_rows = []
    for cost_scenario, net_return in [("ibkr_tiered_10000", 0.03), ("bps_2", 0.02), ("bps_5", -0.01)]:
        test_rows.append(
            {
                "candidate_id": "c1",
                "cost_scenario": cost_scenario,
                "bucket": "hmm_filter",
                "rows": 100,
                "trades": 20,
                "exposure": 0.2,
                "turnover": 1.0,
                "gross_return": net_return,
                "total_cost": 0.0,
                "effective_cost_bps": 2.0,
                "net_return": net_return,
                "avg_trade_net": 0.001,
                "profit_factor": 1.2,
                "daily_sharpe": 0.8,
                "max_drawdown": 0.02,
                "drawdown_duration_bars": 5,
                "drawdown_duration_days": 1,
                "worst_day_net": -0.01,
                "worst_month_net": -0.01,
                "top_day_abs_net_share": 0.2,
                "top_month_abs_net_share": 0.2,
                "top_hour_abs_net_share": 0.2,
                "top_state_abs_net_share": 0.2,
                "net_delta_vs_base": 0.01,
                "net_delta_vs_same_hour": 0.01,
                "daily_sharpe_delta_vs_base": 0.1,
                "daily_sharpe_delta_vs_same_hour": 0.1,
                "drawdown_reduction_vs_base": 0.01,
                "drawdown_reduction_vs_same_hour": 0.01,
                "turnover_reduction_vs_base": 1.0,
            }
        )
    test = pd.DataFrame(test_rows)

    triage = build_triage_frame(decisions, validation, test, {"operable_candidate_triage": {"min_trades": 30}})
    counts = failure_counts(triage)

    assert triage.loc[0, "primary_net_return"] == 0.03
    assert triage.loc[0, "triage_label"] == "too_sparse_or_concentrated"
    assert "insufficient_test_trades" in counts["failure_reason"].tolist()

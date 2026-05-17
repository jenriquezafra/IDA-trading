from __future__ import annotations

import pandas as pd
import pytest

from src.volatility_expansion_diagnostics import (
    candidate_failure_attribution,
    control_delta_summary,
    select_focus_specs,
    side_labels,
    summarize_trades,
)


def test_side_labels_classifies_direction() -> None:
    labels = side_labels(pd.Series([1.0, -1.0, 0.0, float("nan")]))

    assert labels.tolist() == ["long", "short", "flat", "flat"]


def test_summarize_trades_includes_cost_and_day_concentration() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c1"],
            "split": ["test", "test", "test"],
            "cost_scenario": ["ibkr_tiered_10000"] * 3,
            "bucket": ["alpha_signal"] * 3,
            "session": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "net_return": [0.01, -0.002, 0.004],
            "gross_return": [0.012, 0.0, 0.006],
            "cost_return": [0.002, 0.002, 0.002],
        }
    )

    summary = summarize_trades(frame, ["candidate_id", "split", "cost_scenario", "bucket"])

    row = summary.iloc[0]
    assert row["trades"] == 3
    assert row["net_return"] == pytest.approx(0.012)
    assert row["avg_trade_net"] == pytest.approx(0.004)
    assert row["effective_cost_bps"] == pytest.approx(20.0)
    assert row["top_day_abs_net_share"] == pytest.approx(0.008 / (0.008 + 0.004))


def test_control_delta_summary_compares_alpha_to_controls() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "ibkr_tiered_10000",
                "bucket": "alpha_signal",
                "trades": 2,
                "net_return": 0.03,
                "avg_trade_net": 0.015,
                "win_rate": 1.0,
                "effective_cost_bps": 2.0,
            },
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "ibkr_tiered_10000",
                "bucket": "same_hour_random_control",
                "trades": 2,
                "net_return": 0.02,
                "avg_trade_net": 0.010,
                "win_rate": 0.5,
                "effective_cost_bps": 2.0,
            },
        ]
    )

    deltas = control_delta_summary(summary)

    assert deltas.loc[0, "net_delta_vs_random"] == pytest.approx(0.01)
    assert deltas.loc[0, "avg_trade_delta_vs_random"] == pytest.approx(0.005)


def test_candidate_failure_attribution_flags_binding_reasons() -> None:
    decisions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "decision": "cost_fragile",
                "fold": 1,
                "variant": "compression_breakout",
                "side": "long",
                "horizon_bars": 4,
                "test_net_primary": 0.006,
                "test_avg_trade_net_primary": 0.0002,
                "test_trades_primary": 33,
                "test_net_stress": -0.004,
                "test_net_delta_vs_random_primary": -0.001,
                "test_net_delta_vs_breakout_primary": 0.02,
                "test_top_day_abs_net_share_primary": 0.10,
            }
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "ibkr_tiered_10000",
                "effective_cost_bps": 2.0,
                "net_delta_vs_breakout": 0.02,
            },
            {
                "candidate_id": "c1",
                "split": "test",
                "cost_scenario": "bps_5",
                "effective_cost_bps": 5.0,
                "net_delta_vs_breakout": 0.02,
            },
        ]
    )
    config = {
        "volatility_expansion_search": {
            "primary_cost_scenario": "ibkr_tiered_10000",
            "stress_cost_scenario": "bps_5",
            "min_trades": 35,
        }
    }

    attribution = candidate_failure_attribution(decisions, deltas, config)

    reasons = attribution.loc[0, "diagnostic_reasons"]
    assert "insufficient_test_trades" in reasons
    assert "thin_avg_trade_vs_5bps" in reasons
    assert "stress_cost_fragility" in reasons
    assert "random_control_stronger" in reasons


def test_select_focus_specs_prioritizes_accepted_then_configured_order() -> None:
    specs = pd.DataFrame({"candidate_id": ["c1", "c2", "c3", "c4"]})
    decisions = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "validation_status": "volatility_expansion_validation_candidate",
                "decision": "rejected",
                "test_net_primary": 0.10,
                "test_sharpe_primary": 1.0,
            },
            {
                "candidate_id": "c2",
                "validation_status": "volatility_expansion_validation_candidate",
                "decision": "research_candidate",
                "test_net_primary": 0.01,
                "test_sharpe_primary": 0.5,
            },
            {
                "candidate_id": "c3",
                "validation_status": "volatility_expansion_validation_candidate",
                "decision": "cost_fragile",
                "test_net_primary": 0.005,
                "test_sharpe_primary": 0.4,
            },
            {
                "candidate_id": "c4",
                "validation_status": "volatility_expansion_validation_candidate",
                "decision": "accepted_candidate",
                "test_net_primary": 0.004,
                "test_sharpe_primary": 0.3,
            },
        ]
    )

    focus = select_focus_specs(specs, decisions, {"volatility_expansion_diagnostics": {"max_candidates": 2}})

    assert focus["candidate_id"].tolist() == ["c4", "c3"]

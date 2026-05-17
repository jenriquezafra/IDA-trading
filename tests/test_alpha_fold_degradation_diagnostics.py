from __future__ import annotations

import pandas as pd
import pytest

from src.alpha_fold_degradation_diagnostics import (
    diagnostic_findings,
    feature_shift_summary,
    matched_variant_summary,
    side_labels,
    summarize_returns,
)


def test_side_labels_classifies_direction() -> None:
    position = pd.Series([1.0, -1.0, 0.0, float("nan")])

    assert side_labels(position).tolist() == ["long", "short", "flat", "flat"]


def test_summarize_returns_groups_trade_metrics() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test", "test"],
            "fold": [0, 0, 0],
            "side": ["short", "short", "long"],
            "candidate_id": ["a", "b", "a"],
            "net_return": [0.02, -0.01, 0.03],
            "gross_return": [0.03, 0.0, 0.04],
            "cost_return": [0.01, 0.01, 0.01],
            "fwd_ret": [-0.03, 0.0, 0.04],
        }
    )

    summary = summarize_returns(frame, ["split", "fold", "side"])
    short = summary[summary["side"].eq("short")].iloc[0]

    assert short["candidates"] == 2
    assert short["trades"] == 2
    assert short["net_return"] == pytest.approx(0.01)
    assert short["avg_trade_net"] == pytest.approx(0.005)
    assert short["win_rate"] == pytest.approx(0.5)


def test_feature_shift_summary_returns_long_format() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test", "test"],
            "fold": [0, 0],
            "side": ["short", "short"],
            "target_ret_6": [-0.01, -0.03],
            "chop_score": [0.2, 0.4],
        }
    )

    summary = feature_shift_summary(frame, ["target_ret_6", "chop_score"])

    ret6 = summary[summary["feature"].eq("target_ret_6")].iloc[0]
    assert ret6["mean"] == pytest.approx(-0.02)
    assert ret6["rows"] == 2


def test_matched_variant_summary_compares_best_fold_rows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "fold": 0,
                "alpha_variant": "m6_base",
                "horizon_bars": 12,
                "decision": "accepted_candidate",
                "test_net_primary": 0.10,
                "test_sharpe_primary": 1.4,
                "test_profit_factor_primary": 1.5,
                "test_avg_trade_net_primary": 0.001,
                "test_net_stress": 0.05,
                "test_trades_primary": 100,
            },
            {
                "fold": 1,
                "alpha_variant": "m6_base",
                "horizon_bars": 12,
                "decision": "research_candidate",
                "test_net_primary": 0.02,
                "test_sharpe_primary": 0.3,
                "test_profit_factor_primary": 1.1,
                "test_avg_trade_net_primary": 0.0002,
                "test_net_stress": -0.01,
                "test_trades_primary": 120,
            },
        ]
    )

    matched = matched_variant_summary(decisions)

    assert matched.loc[0, "fold1_minus_fold0_net"] == pytest.approx(-0.08)
    assert matched.loc[0, "fold1_avg_trade_ratio"] == pytest.approx(0.2)


def test_diagnostic_findings_flags_side_degradation() -> None:
    side_summary = pd.DataFrame(
        [
            {"split": "test", "fold": 0, "side": "short", "net_return": 0.30, "avg_trade_net": 0.003},
            {"split": "test", "fold": 0, "side": "long", "net_return": 0.01, "avg_trade_net": 0.0001},
            {"split": "test", "fold": 1, "side": "short", "net_return": 0.05, "avg_trade_net": 0.0004},
            {"split": "test", "fold": 1, "side": "long", "net_return": -0.20, "avg_trade_net": -0.001},
        ]
    )
    monthly_summary = pd.DataFrame(
        [
            {"split": "test", "fold": 0, "side": "short", "month": "2024-04", "net_return": 0.25},
            {"split": "test", "fold": 0, "side": "short", "month": "2024-03", "net_return": 0.05},
        ]
    )
    feature_shift = pd.DataFrame(
        [
            {"split": "test", "fold": 0, "side": "short", "feature": "chop_score", "mean": 0.2},
            {"split": "test", "fold": 1, "side": "short", "feature": "chop_score", "mean": 0.5},
        ]
    )

    findings = diagnostic_findings(side_summary, monthly_summary, feature_shift)

    assert "fold1_long_side_flipped_negative" in findings["finding"].tolist()
    assert "fold1_short_edge_compressed" in findings["finding"].tolist()
    assert "fold0_has_month_concentration" in findings["finding"].tolist()

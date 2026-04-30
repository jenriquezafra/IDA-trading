from __future__ import annotations

import pandas as pd
import pytest

from src.signal import apply_selected_signal, apply_signal_rules, evaluate_signal, select_thresholds_on_validation


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:30", periods=6, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "bar_index": [12, 13, 14, 12, 13, 14],
            "split": ["validation", "validation", "validation", "test", "test", "test"],
            "p_up": [0.60, 0.10, 0.40, 0.60, 0.10, 0.40],
            "p_down": [0.10, 0.62, 0.20, 0.10, 0.62, 0.20],
            "p_neutral": [0.30, 0.28, 0.40, 0.30, 0.28, 0.40],
            "score": [0.50, -0.52, 0.20, 0.50, -0.52, 0.20],
            "hmm_entropy": [0.2, 0.2, 0.95, 0.2, 0.2, 0.95],
            "hmm_state": [1, 1, 1, 1, 1, 1],
            "fwd_ret": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )


def _config() -> dict:
    return {
        "labeling": {"round_trip_cost_bps": 1.0},
        "signal": {
            "theta_prob_grid": [0.55],
            "theta_score_grid": [0.10],
            "max_neutral_grid": [0.55],
            "max_hmm_entropy_grid": [0.90],
            "allowed_hmm_states": [],
        },
    }


def test_apply_signal_rules_long_short_flat() -> None:
    signal = apply_signal_rules(_frame(), theta_prob=0.55, theta_score=0.10, max_neutral=0.55, max_hmm_entropy=0.90)

    assert signal.tolist() == [1, -1, 0, 1, -1, 0]


def test_apply_signal_rules_filters_by_entropy_and_regime() -> None:
    frame = _frame()

    entropy_filtered = apply_signal_rules(frame, 0.55, 0.10, 0.55, max_hmm_entropy=0.10)
    regime_filtered = apply_signal_rules(frame, 0.55, 0.10, 0.55, 0.90, allowed_hmm_states=[2])

    assert entropy_filtered.tolist() == [0, 0, 0, 0, 0, 0]
    assert regime_filtered.tolist() == [0, 0, 0, 0, 0, 0]


def test_evaluate_signal_applies_round_trip_cost() -> None:
    frame = _frame().iloc[:2].copy()
    signal = pd.Series([1, -1])

    metrics = evaluate_signal(frame, signal, round_trip_cost_bps=1.0)

    assert metrics["trades"] == 2
    assert metrics["gross_return"] == 0.0
    assert metrics["total_cost"] == 0.0002
    assert metrics["net_return"] == pytest.approx(-0.0002)


def test_select_thresholds_uses_validation_only() -> None:
    selected, grid = select_thresholds_on_validation(_frame(), _config())

    assert selected == {"theta_prob": 0.55, "theta_score": 0.10, "max_neutral": 0.55, "max_hmm_entropy": 0.90}
    assert len(grid) == 1
    assert grid.loc[0, "rows"] == 3


def test_apply_selected_signal_adds_returns() -> None:
    signals = apply_selected_signal(
        _frame(),
        {"theta_prob": 0.55, "theta_score": 0.10, "max_neutral": 0.55, "max_hmm_entropy": 0.90},
        _config(),
    )

    assert {"signal", "signal_name", "signal_gross_ret", "signal_cost_ret", "signal_net_ret"}.issubset(signals.columns)
    assert signals["signal_name"].tolist() == ["long", "short", "flat", "long", "short", "flat"]

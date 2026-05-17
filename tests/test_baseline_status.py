from __future__ import annotations

import pandas as pd
import pytest

from src.baseline_status import add_comparisons, summarize_labeled_positions


def test_summarize_labeled_positions_computes_net_metrics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 3,
            "bar_index": [0, 1, 2],
            "position": [1, 0, -1],
            "gross_ret": [0.003, 0.0, -0.001],
            "cost_ret": [0.0001, 0.0, 0.0001],
            "net_ret": [0.0029, 0.0, -0.0011],
        }
    )

    row = summarize_labeled_positions(frame, "demo", "memory", cost_bps=1.0)

    assert row["trades"] == 2
    assert row["net_return"] == pytest.approx(0.0018)
    assert row["avg_trade_net"] == pytest.approx(0.0009)
    assert row["profit_factor_net"] == pytest.approx(0.0029 / 0.0011)


def test_add_comparisons_rejects_negative_strategy() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "always_flat", "source": "x", "cost_bps": 1.0, "trades": 0, "net_return": 0.0, "gross_return": 0.0, "total_cost": 0.0, "daily_sharpe_net": None, "profit_factor_net": None, "avg_trade_net": 0.0, "max_drawdown": 0.0, "folds_positive": 0, "folds_negative": 0},
            {"strategy": "random", "source": "x", "cost_bps": 1.0, "trades": 10, "net_return": -1.0, "gross_return": 0.0, "total_cost": 1.0, "daily_sharpe_net": -1.0, "profit_factor_net": 0.5, "avg_trade_net": -0.1, "max_drawdown": 1.0, "folds_positive": 0, "folds_negative": 1},
            {"strategy": "demo", "source": "x", "cost_bps": 1.0, "trades": 2, "net_return": -0.1, "gross_return": 0.0, "total_cost": 0.1, "daily_sharpe_net": -1.0, "profit_factor_net": 0.8, "avg_trade_net": -0.05, "max_drawdown": 0.1, "folds_positive": 0, "folds_negative": 1},
        ]
    )

    output = add_comparisons(summary)

    demo = output[output["strategy"] == "demo"].iloc[0]
    assert not bool(demo["beats_always_flat"])
    assert bool(demo["beats_random"])
    assert demo["status"] == "rejected_economic"

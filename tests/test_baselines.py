from __future__ import annotations

import pandas as pd

from src.baselines import build_baseline_results, evaluate_strategy, generate_baseline_positions


def _config() -> dict:
    return {
        "labeling": {"round_trip_cost_bps": 1.0, "horizon_bars": 2},
        "paths": {"reports": "reports"},
        "data": {"labels_file": "data/features/labels.parquet"},
        "baselines": {
            "random_seed": 7,
            "momentum_column": "ret_3",
            "reversal_column": "ret_3",
            "signal_threshold_column": "neutral_zone",
        },
    }


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:30", periods=5, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 5,
            "bar_index": [12, 13, 14, 15, 16],
            "entry_px": [100, 101, 102, 103, 104],
            "exit_px": [101, 100, 104, 102, 105],
            "fwd_ret": [0.01, -0.01, 0.02, -0.02, 0.0],
            "target": [1, -1, 1, -1, 0],
            "ret_3": [0.003, -0.003, 0.0, 0.004, -0.004],
            "neutral_zone": [0.002] * 5,
        }
    )


def test_generate_baseline_positions() -> None:
    positions = generate_baseline_positions(_labels(), _config())

    assert positions["always_flat"].tolist() == [0, 0, 0, 0, 0]
    assert positions["intraday_buy_hold"].tolist() == [1, 1, 1, 1, 1]
    assert positions["momentum"].tolist() == [1, -1, 0, 1, -1]
    assert positions["reversion"].tolist() == [-1, 1, 0, -1, 1]


def test_evaluate_strategy_applies_cost_only_when_active() -> None:
    labels = _labels()
    position = pd.Series([1, 0, -1, 1, 0])

    trades = evaluate_strategy(labels, "test", position, round_trip_cost_bps=1.0)

    assert trades["gross_ret"].tolist() == [0.01, 0.0, -0.02, -0.02, 0.0]
    assert trades["cost_ret"].tolist() == [0.0001, 0.0, 0.0001, 0.0001, 0.0]
    assert trades["net_ret"].tolist() == [0.0099, 0.0, -0.0201, -0.0201, 0.0]


def test_build_baseline_results_has_all_strategies_and_summary() -> None:
    trades, summary = build_baseline_results(_labels(), _config())

    assert set(summary["strategy"]) == {"always_flat", "random", "intraday_buy_hold", "momentum", "reversion"}
    assert set(trades["strategy"]) == set(summary["strategy"])
    assert summary.loc[summary["strategy"] == "always_flat", "trades"].item() == 0
    assert summary.loc[summary["strategy"] == "intraday_buy_hold", "trades"].item() == 5

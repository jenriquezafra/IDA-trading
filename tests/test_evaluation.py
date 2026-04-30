from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calibration_metrics,
    daily_sharpe,
    exposure_from_trades,
    max_drawdown,
    pnl_by_fold,
    pnl_by_hour,
    pnl_by_regime,
    pnl_by_side,
    profit_factor,
    summarize_trades,
)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_timestamp": pd.to_datetime(
                ["2024-01-02 09:35", "2024-01-02 10:35", "2024-01-03 11:35"]
            ).tz_localize("America/New_York"),
            "entry_timestamp": pd.to_datetime(
                ["2024-01-02 09:40", "2024-01-02 10:40", "2024-01-03 11:40"]
            ).tz_localize("America/New_York"),
            "exit_timestamp": pd.to_datetime(
                ["2024-01-02 09:50", "2024-01-02 10:50", "2024-01-03 11:50"]
            ).tz_localize("America/New_York"),
            "session": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "entry_bar_index": [2, 14, 26],
            "exit_bar_index": [4, 16, 28],
            "side": ["long", "short", "long"],
            "position": [1.0, -1.0, 1.0],
            "gross_ret": [0.011, -0.004, 0.006],
            "total_cost_ret": [0.001, 0.001, 0.001],
            "net_ret": [0.010, -0.005, 0.005],
        }
    )


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-03"],
            "gross_ret": [0.007, 0.006],
            "net_ret": [0.005, 0.005],
            "trades": [2, 1],
        }
    )


def _equity() -> pd.DataFrame:
    return pd.DataFrame({"equity": [0.0, 0.010, 0.005, 0.015]})


def test_core_trade_metrics_are_net_of_costs() -> None:
    trades = _trades()
    summary = summarize_trades(trades, _daily(), _equity())

    assert summary["trades"] == 3
    assert summary["net_return"] == pytest.approx(0.010)
    assert summary["gross_return"] == pytest.approx(0.013)
    assert summary["total_cost"] == pytest.approx(0.003)
    assert summary["profit_factor"] == pytest.approx(3.0)
    assert summary["hit_ratio"] == pytest.approx(2 / 3)
    assert summary["avg_trade_net"] == pytest.approx(0.010 / 3)
    assert summary["median_trade_net"] == pytest.approx(0.005)
    assert summary["turnover_trades_per_day"] == pytest.approx(1.5)


def test_sharpe_drawdown_and_profit_factor_handle_edge_cases() -> None:
    assert np.isnan(daily_sharpe(pd.DataFrame({"net_ret": [0.001]})))
    assert max_drawdown(_equity()) == pytest.approx(0.005)
    assert profit_factor(pd.DataFrame({"net_ret": [0.001, 0.002]})) == np.inf


def test_grouped_pnl_tables() -> None:
    trades = _trades()
    by_side = pnl_by_side(trades)
    by_hour = pnl_by_hour(trades)

    assert by_side.loc[by_side["side"] == "long", "net_return"].iloc[0] == pytest.approx(0.015)
    assert by_side.loc[by_side["side"] == "short", "net_return"].iloc[0] == pytest.approx(-0.005)
    assert by_hour["entry_hour"].tolist() == [9, 10, 11]


def test_pnl_by_regime_joins_signal_time_to_hmm_state() -> None:
    hmm = pd.DataFrame(
        {
            "timestamp": _trades()["signal_timestamp"],
            "session": _trades()["session"],
            "bar_index": [1, 13, 25],
            "hmm_state": [0, 1, 0],
        }
    )

    by_regime = pnl_by_regime(_trades(), hmm)

    assert by_regime.loc[by_regime["hmm_state"] == 0, "trades"].iloc[0] == 2
    assert by_regime.loc[by_regime["hmm_state"] == 0, "net_return"].iloc[0] == pytest.approx(0.015)
    assert by_regime.loc[by_regime["hmm_state"] == 1, "net_return"].iloc[0] == pytest.approx(-0.005)


def test_exposure_uses_held_bars_over_cleaned_session_bars() -> None:
    cleaned = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 10 + ["2024-01-03"] * 10,
            "bar_index": list(range(10)) + list(range(10)),
        }
    )

    assert exposure_from_trades(_trades(), cleaned) == pytest.approx(9 / 20)


def test_pnl_by_fold_normalizes_walkforward_columns() -> None:
    folds = pd.DataFrame(
        {
            "fold": [0],
            "test_months": ["2024-07"],
            "test_signal_trades": [12],
            "test_signal_net_return": [0.02],
            "test_signal_avg_trade_net": [0.0017],
            "test_signal_hit_ratio": [0.58],
        }
    )

    output = pnl_by_fold(folds)

    assert output.loc[0, "trades"] == 12
    assert output.loc[0, "net_return"] == pytest.approx(0.02)


def test_calibration_metrics_for_multiclass_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "target": [-1, 0, 1],
            "p_down": [0.8, 0.1, 0.1],
            "p_neutral": [0.1, 0.7, 0.2],
            "p_up": [0.1, 0.2, 0.7],
        }
    )

    metrics = calibration_metrics(predictions, n_bins=5)

    assert metrics["rows"] == 3
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier_multiclass"] < 0.2

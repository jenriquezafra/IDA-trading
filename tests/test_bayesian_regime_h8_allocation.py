from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.bayesian_regime_h8_allocation import (
    cost_return,
    executable_position,
    run,
    target_position,
    turnover_series,
)


def _features(months: int = 5, sessions_per_month: int = 5, bars: int = 8) -> pd.DataFrame:
    rows = []
    price = 100.0
    for month in range(months):
        start = pd.Timestamp("2024-01-02") + pd.DateOffset(months=month)
        sessions = pd.date_range(start, periods=sessions_per_month, freq="B")
        month_direction = 1.0 if month % 2 == 0 else -1.0
        for session_idx, session_ts in enumerate(sessions):
            session = session_ts.strftime("%Y-%m-%d")
            timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="15min", tz="America/New_York")
            for bar_index, timestamp in enumerate(timestamps):
                drift = month_direction * (0.001 + 0.0001 * bar_index)
                price *= float(np.exp(drift))
                rows.append(
                    {
                        "timestamp": timestamp,
                        "session": session,
                        "bar_index": bar_index,
                        "target_ret_3": drift * 3.0,
                        "target_ret_4": drift * 4.0,
                        "target_rv_12_rel_by_bar": 0.8 + 0.05 * ((bar_index + session_idx + month) % 5),
                        "target_signed_efficiency_12": month_direction * (0.5 + 0.02 * bar_index),
                        "target_open_next": price * np.exp(drift),
                    }
                )
    return pd.DataFrame(rows)


def _config(tmp_path: Path, features_path: Path) -> dict:
    return {
        "lab": {"target_symbol": "QQQ"},
        "bayesian_regime_h8": {
            "features_file": str(features_path),
            "results_dir": str(tmp_path / "h8-results"),
            "report_file": str(tmp_path / "h8-report/h8.md"),
            "models_dir": str(tmp_path / "h8-models"),
            "momentum_column": "target_ret_4",
            "volatility_column": "target_rv_12_rel_by_bar",
            "efficiency_column": "target_signed_efficiency_12",
            "variants": ["manual_h8a"],
            "walk_forward": {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1},
            "max_folds": 1,
        },
        "h8_probability_allocation": {
            "results_dir": str(tmp_path / "h8d-results"),
            "report_file": str(tmp_path / "h8d-report/h8d.md"),
            "holding_horizon_bars": [1],
            "allocation_methods": ["edge", "confidence_edge"],
            "min_abs_positions": [0.0, 0.2],
            "max_entropy_values": [None],
            "smoothing_alphas": [1.0],
            "rebalance_thresholds": [0.0],
            "primary_cost_scenario": "bps_1",
            "conservative_cost_scenario": "bps_1",
            "stress_cost_scenario": "bps_2",
            "min_validation_turnover": 0.1,
            "min_daily_sharpe": -10.0,
            "min_net_per_turnover_bps": -10.0,
        },
        "candidate_cost_sensitivity_cross_asset": {
            "cost_bps": [1.0, 2.0],
            "ibkr": {"enabled": False},
        },
    }


def test_target_position_uses_bull_minus_bear_probability() -> None:
    frame = pd.DataFrame(
        {
            "p_bull_trend": [0.80, 0.20, 0.40],
            "p_bear_stress": [0.10, 0.70, 0.35],
            "p_chop_compression": [0.05, 0.05, 0.20],
            "p_volatile_noise": [0.05, 0.05, 0.05],
            "max_prob": [0.80, 0.70, 0.40],
            "entropy": [0.2, 0.3, 0.9],
        }
    )

    position = target_position(frame, method="edge", min_abs_position=0.0, max_entropy=None)

    assert position.tolist() == pytest.approx([0.70, -0.50, 0.05])
    filtered = target_position(frame, method="edge", min_abs_position=0.10, max_entropy=0.75)
    assert filtered.tolist() == pytest.approx([0.70, -0.50, 0.0])


def test_turnover_cost_interprets_bps_as_round_trip() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02"],
            "entry_px": [100.0, 101.0],
            "exit_px": [101.0, 102.0],
        }
    )
    position = pd.Series([1.0, 1.0])
    entry_turnover, exit_turnover = turnover_series(position, frame["session"])

    cost = cost_return(frame, position, entry_turnover, exit_turnover, {"cost_kind": "bps", "round_trip_bps": 2.0})

    assert entry_turnover.tolist() == pytest.approx([1.0, 0.0])
    assert exit_turnover.tolist() == pytest.approx([0.0, 1.0])
    assert float(cost.sum()) == pytest.approx(0.0002)


def test_executable_position_can_suppress_small_rebalances() -> None:
    target = pd.Series([0.40, 0.42, 0.10, 0.0])
    sessions = pd.Series(["2024-01-02"] * 4)

    position = executable_position(target, sessions, smoothing_alpha=1.0, rebalance_threshold=0.05)

    assert position.tolist() == pytest.approx([0.40, 0.40, 0.10, 0.0])


def test_h8d_runner_writes_promotion_artifacts(tmp_path) -> None:
    features_path = tmp_path / "features.parquet"
    _features().to_parquet(features_path, index=False)
    config = _config(tmp_path, features_path)
    config_path = tmp_path / "h8d.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report_path, sensitivity_path = run(config_path)

    assert report_path.exists()
    assert sensitivity_path.exists()
    sensitivity = pd.read_parquet(sensitivity_path)
    decision = pd.read_parquet(tmp_path / "h8d-results/h8d_promotion_decision.parquet")
    assert {"validation", "test"}.issubset(set(sensitivity["split"]))
    assert set(sensitivity["cost_scenario"]).issubset({"bps_1", "bps_2"})
    assert decision["decision"].notna().all()
    assert (tmp_path / "h8d-results/h8d_probability_allocation_aggregate.parquet").exists()

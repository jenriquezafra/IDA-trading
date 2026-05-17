from __future__ import annotations

import pandas as pd
import pytest

from src.robustness import cost_stress, data_sufficiency, experiment_plan, horizon_sensitivity, threshold_sensitivity
from src.signal import build_signal_frame


def _config() -> dict:
    return {
        "labeling": {
            "horizon_bars": 2,
            "round_trip_cost_bps": 1.0,
            "buffer_bps": 0.5,
            "lambda_vol": 0.25,
        },
        "walkforward": {
            "train_months": 5,
            "validation_months": 1,
            "test_months": 1,
            "step_months": 1,
            "purge_bars": 2,
            "embargo_bars": 0,
        },
        "signal": {
            "theta_prob_grid": [0.55],
            "theta_score_grid": [0.10],
            "max_neutral_grid": [0.55],
            "max_hmm_entropy_grid": [0.90],
            "allowed_hmm_states": [],
        },
        "robustness": {
            "horizons": [1, 2],
            "hmm_states": [2, 4],
            "cost_bps": [1.0, 5.0],
            "theta_prob_grid": [0.55],
            "theta_score_grid": [0.10],
            "max_neutral_grid": [0.55],
            "max_hmm_entropy_grid": [0.90],
            "train_months": [3, 5],
            "seeds": [42, 7],
            "periods": ["all"],
        },
    }


def _labels(months: int = 3) -> pd.DataFrame:
    rows = []
    start = pd.Period("2024-01", freq="M")
    for month_idx in range(months):
        month = start + month_idx
        for session_idx in range(2):
            session = f"{month}-0{session_idx + 1}"
            for bar_index in range(4):
                rows.append(
                    {
                        "timestamp": pd.Timestamp(f"{session} 10:{30 + bar_index:02d}", tz="America/New_York"),
                        "session": session,
                        "bar_index": bar_index,
                        "target": 0,
                    }
                )
    return pd.DataFrame(rows)


def _features() -> pd.DataFrame:
    rows = []
    for session in ["2024-01-02", "2024-01-03"]:
        timestamps = pd.date_range(f"{session} 09:30", periods=6, freq="5min", tz="America/New_York")
        for bar_index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": 100.0 + bar_index,
                    "session": session,
                    "bar_index": bar_index,
                    "bars_in_session": 6,
                    "rv_12": 0.001,
                    "target_crosses_session_close": False,
                }
            )
    return pd.DataFrame(rows)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:00", periods=4, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "bar_index": [6, 7, 6, 7],
            "split": ["test", "test", "test", "test"],
            "signal": [1, -1, 0, 1],
            "fwd_ret": [0.002, 0.001, -0.001, -0.002],
        }
    )


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:00", periods=3, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 3,
            "bar_index": [6, 7, 8],
            "split": ["test"] * 3,
            "p_up": [0.70, 0.10, 0.40],
            "p_down": [0.10, 0.70, 0.20],
            "p_neutral": [0.20, 0.20, 0.40],
            "score": [0.60, -0.60, 0.20],
            "hmm_entropy": [0.2, 0.2, 0.2],
            "hmm_state": [0, 0, 0],
            "fwd_ret": [0.002, 0.001, -0.001],
        }
    )


def test_data_sufficiency_marks_short_dataset_as_not_evidence() -> None:
    sufficiency = data_sufficiency(_labels(months=3), _config())

    assert sufficiency["available_months"] == 3
    assert sufficiency["required_months"] == 7
    assert sufficiency["generated_folds"] == 0
    assert sufficiency["has_walkforward_evidence"] is False


def test_horizon_sensitivity_profiles_label_distribution() -> None:
    output = horizon_sensitivity(_features(), _config(), [1, 2])

    assert output["horizon_bars"].tolist() == [1, 2]
    assert output["rows"].min() > 0
    assert set(output["status"]) == {"exploratory_current_data"}


def test_cost_stress_replays_existing_signals_with_cost_grid() -> None:
    output = cost_stress(_signals(), [1.0, 5.0])

    assert output["cost_bps"].tolist() == [1.0, 5.0]
    assert output.loc[output["cost_bps"] == 5.0, "net_return"].iloc[0] < output.loc[output["cost_bps"] == 1.0, "net_return"].iloc[0]


def test_threshold_sensitivity_applies_grid_to_test_frame() -> None:
    frame = build_signal_frame(_prediction_frame())

    output = threshold_sensitivity(frame, _config())

    assert len(output) == 1
    assert output.loc[0, "trades"] == 2
    assert output.loc[0, "net_return"] == pytest.approx(0.0008)


def test_experiment_plan_keeps_rerun_items_pending_without_folds() -> None:
    sufficiency = data_sufficiency(_labels(months=3), _config())
    plan = experiment_plan(_config(), sufficiency)

    assert "pending_long_intraday_history" in set(plan["status"])
    assert "stress_replayed_current_signals" in set(plan["status"])
    assert "profiled_current_data" in set(plan["status"])

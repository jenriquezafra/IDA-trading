from __future__ import annotations

import numpy as np
import pandas as pd

from src.predictive_model import (
    CLASS_ORDER,
    fit_base_model,
    get_base_feature_columns,
    predict_probabilities,
    prepare_model_frame,
    split_sessions,
)


FEATURE_COLUMNS = ["ret_1", "ret_2", "rv_3", "range", "open_window", "midday"]


def _config() -> dict:
    return {
        "model": {
            "base_feature_columns": FEATURE_COLUMNS,
            "random_state": 1,
            "logistic_regression": {
                "penalty": "elasticnet",
                "solver": "saga",
                "l1_ratio": 0.1,
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": None,
            },
        }
    }


def _frame() -> pd.DataFrame:
    rows = []
    for session_idx, session in enumerate(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]):
        for bar_index in range(6):
            value = session_idx * 0.1 + bar_index * 0.01
            target = [-1, 0, 1][(session_idx + bar_index) % 3]
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{session} 10:{30 + bar_index:02d}", tz="America/New_York"),
                    "session": session,
                    "bar_index": bar_index,
                    "target": target,
                    "fwd_ret": value,
                    "neutral_zone": 0.001,
                    "ret_1": value,
                    "ret_2": value * 2,
                    "rv_3": 0.01 + value,
                    "range": 0.02 + value,
                    "open_window": bar_index < 2,
                    "midday": False,
                    "hmm_p0": 0.5,
                    "entry_px": 100.0,
                    "exit_px": 101.0,
                }
            )
    return pd.DataFrame(rows)


def test_get_base_feature_columns_excludes_hmm_and_future_columns() -> None:
    columns = get_base_feature_columns(_config())

    assert columns == FEATURE_COLUMNS
    assert "hmm_p0" not in columns
    assert "entry_px" not in columns
    assert "fwd_ret" not in columns


def test_prepare_model_frame_drops_missing_feature_rows() -> None:
    frame = _frame()
    frame.loc[0, "ret_1"] = np.nan

    prepared = prepare_model_frame(frame, FEATURE_COLUMNS)

    assert len(prepared) == len(frame) - 1
    assert prepared[FEATURE_COLUMNS].notna().all().all()


def test_split_sessions_is_chronological() -> None:
    splits = split_sessions(_frame(), train_fraction=0.6, validation_fraction=0.2)

    assert splits["train"] == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert splits["validation"] == ["2024-01-05"]
    assert splits["test"] == ["2024-01-08"]


def test_fit_base_model_scaler_uses_train_only_and_predicts_probabilities() -> None:
    frame = prepare_model_frame(_frame(), FEATURE_COLUMNS)
    train = frame[frame["session"].isin(["2024-01-02", "2024-01-03", "2024-01-04"])].copy()

    scaler, model = fit_base_model(train, FEATURE_COLUMNS, _config())
    predictions = predict_probabilities(model, scaler, train, FEATURE_COLUMNS)

    assert np.allclose(scaler.mean_, train[FEATURE_COLUMNS].astype(float).mean().to_numpy())
    assert predictions[["p_down", "p_neutral", "p_up"]].shape == (len(train), 3)
    assert np.allclose(predictions[["p_down", "p_neutral", "p_up"]].sum(axis=1), 1.0)
    assert set(predictions["predicted_class"]).issubset(set(CLASS_ORDER))

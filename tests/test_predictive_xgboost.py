from __future__ import annotations

import numpy as np
import pandas as pd

from src.predictive_xgboost import (
    TARGET_TO_XGB,
    _align_xgboost_probabilities,
    encode_target,
    fit_xgboost_model,
    predict_xgboost_probabilities,
)


FEATURE_COLUMNS = ["ret_1", "rv_3", "range"]


def _config() -> dict:
    return {
        "model": {
            "random_state": 1,
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "n_jobs": 1,
            },
        }
    }


def _frame() -> pd.DataFrame:
    rows = []
    for idx, target in enumerate([-1, 0, 1] * 8):
        rows.append(
            {
                "timestamp": pd.Timestamp("2024-01-02 09:30", tz="America/New_York") + pd.Timedelta(minutes=5 * idx),
                "session": "2024-01-02",
                "bar_index": idx,
                "target": target,
                "fwd_ret": 0.001 * target,
                "neutral_zone": 0.001,
                "ret_1": idx * 0.01,
                "rv_3": 0.1 + idx * 0.01,
                "range": 0.01 + idx * 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_encode_target_maps_project_labels_to_xgboost_labels() -> None:
    encoded = encode_target(pd.Series([-1, 0, 1]))

    assert encoded.tolist() == [TARGET_TO_XGB[-1], TARGET_TO_XGB[0], TARGET_TO_XGB[1]]


def test_align_xgboost_probabilities_returns_project_class_order() -> None:
    raw = np.array([[0.2, 0.5, 0.3]])
    aligned = _align_xgboost_probabilities(np.array([0, 1, 2]), raw)

    assert aligned.tolist() == [[0.2, 0.5, 0.3]]


def test_fit_xgboost_model_predicts_project_probabilities() -> None:
    frame = _frame()
    model = fit_xgboost_model(frame, FEATURE_COLUMNS, _config())
    predictions = predict_xgboost_probabilities(model, frame, FEATURE_COLUMNS)

    assert predictions[["p_down", "p_neutral", "p_up"]].shape == (len(frame), 3)
    assert np.allclose(predictions[["p_down", "p_neutral", "p_up"]].sum(axis=1), 1.0)
    assert set(predictions["predicted_class"]).issubset({-1, 0, 1})

from __future__ import annotations

import joblib
import pandas as pd

from src.predictive_model_hmm import compare_with_base, get_hmm_feature_columns, merge_labels_with_hmm


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:30", periods=4, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02"] * 4,
            "bar_index": [12, 13, 14, 15],
            "target": [-1, 0, 1, 0],
        }
    )


def _hmm() -> pd.DataFrame:
    frame = _labels().copy()
    frame["hmm_p0"] = [0.7, 0.1, 0.1, 0.1]
    frame["hmm_p1"] = [0.1, 0.7, 0.1, 0.1]
    frame["hmm_p2"] = [0.1, 0.1, 0.7, 0.1]
    frame["hmm_p3"] = [0.1, 0.1, 0.1, 0.7]
    frame["hmm_state"] = [0, 1, 2, 3]
    frame["hmm_entropy"] = [0.3, 0.4, 0.5, 0.6]
    frame["hmm_max_prob"] = [0.7, 0.7, 0.7, 0.7]
    return frame


def test_merge_labels_with_hmm_adds_probabilities_and_one_hot_states() -> None:
    merged = merge_labels_with_hmm(_labels(), _hmm(), n_states=4)

    assert {"hmm_p0", "hmm_p1", "hmm_p2", "hmm_p3", "hmm_entropy", "hmm_max_prob"}.issubset(merged.columns)
    assert merged[["hmm_state_0", "hmm_state_1", "hmm_state_2", "hmm_state_3"]].sum(axis=1).tolist() == [1, 1, 1, 1]
    assert merged["hmm_state"].tolist() == [0, 1, 2, 3]


def test_get_hmm_feature_columns_appends_hmm_features() -> None:
    config = {
        "model": {
            "base_feature_columns": ["ret_1", "rv_12"],
            "hmm_feature_columns": ["hmm_p0", "hmm_entropy", "hmm_state_0"],
        }
    }

    assert get_hmm_feature_columns(config) == ["ret_1", "rv_12", "hmm_p0", "hmm_entropy", "hmm_state_0"]


def test_compare_with_base_uses_test_metrics(tmp_path) -> None:
    metadata_path = tmp_path / "models" / "predictive_base" / "fold_0"
    metadata_path.mkdir(parents=True)
    base_metrics = pd.DataFrame(
        [
            {"split": "test_calibrated", "accuracy": 0.4, "balanced_accuracy": 0.3, "macro_f1": 0.2, "log_loss": 1.1}
        ]
    )
    joblib.dump({"metrics": base_metrics}, metadata_path / "metadata.joblib")
    hmm_metrics = pd.DataFrame(
        [
            {"split": "test_calibrated", "accuracy": 0.5, "balanced_accuracy": 0.4, "macro_f1": 0.25, "log_loss": 1.0}
        ]
    )
    config = {"paths": {"models": str(tmp_path / "models")}}

    comparison = compare_with_base(hmm_metrics, config)

    assert comparison["improved"].tolist() == [True, True, True, True]

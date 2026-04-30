from __future__ import annotations

import numpy as np
import pandas as pd

from src.hmm_filter import add_hmm_probability_columns, filtered_probabilities
from src.hmm_model import _lengths_by_session, _split_sessions, fit_hmm_model, prepare_hmm_frame


class DummyHMM:
    n_components = 2
    startprob_ = np.array([0.8, 0.2])
    transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])

    def _compute_log_likelihood(self, observations: np.ndarray) -> np.ndarray:
        if len(observations) == 0:
            return np.empty((0, 2))
        # Observation is already a per-state log-likelihood pair in this test.
        return observations


def _hmm_feature_frame() -> pd.DataFrame:
    rows = []
    for session_idx, session in enumerate(["2024-01-02", "2024-01-03", "2024-01-04"]):
        for bar_index in range(4):
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{session} 10:{30 + 5 * bar_index:02d}", tz="America/New_York"),
                    "session": session,
                    "bar_index": bar_index,
                    "ret_1": 0.01 * (bar_index + 1),
                    "ret_3": 0.01 * (bar_index + 1),
                    "rv_6": 0.02,
                    "rv_12": 0.03,
                    "range": 0.01,
                    "rel_volume": 1.0 + session_idx,
                    "trend_12": 0.001,
                    "intraday_drawdown": -0.001,
                }
            )
    return pd.DataFrame(rows)


def test_filtered_probabilities_are_normalized_and_reset_by_session() -> None:
    observations = np.array(
        [
            [0.0, -2.0],
            [-2.0, 0.0],
            [0.0, -2.0],
            [-2.0, 0.0],
        ]
    )
    sessions = pd.Series(["a", "a", "b", "b"])

    probabilities = filtered_probabilities(DummyHMM(), observations, sessions)

    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], probabilities[2])


def test_add_hmm_probability_columns() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    probabilities = np.array([[0.25, 0.75], [0.9, 0.1]])

    annotated = add_hmm_probability_columns(df, probabilities)

    assert annotated["hmm_p0"].tolist() == [0.25, 0.9]
    assert annotated["hmm_p1"].tolist() == [0.75, 0.1]
    assert annotated["hmm_state"].tolist() == [1, 0]
    assert annotated["hmm_max_prob"].tolist() == [0.75, 0.9]
    assert annotated["hmm_entropy"].between(0, 1).all()


def test_prepare_hmm_frame_drops_rows_with_missing_hmm_features() -> None:
    frame = _hmm_feature_frame()
    frame.loc[0, "ret_1"] = np.nan

    prepared = prepare_hmm_frame(frame, ["ret_1", "ret_3", "rv_6", "rv_12", "range", "rel_volume", "trend_12", "intraday_drawdown"])

    assert len(prepared) == len(frame) - 1
    assert prepared["source_index"].min() == 1


def test_lengths_and_split_are_session_based() -> None:
    frame = _hmm_feature_frame()

    train_sessions, test_sessions = _split_sessions(frame, 0.67)

    assert train_sessions == ["2024-01-02", "2024-01-03"]
    assert test_sessions == ["2024-01-04"]
    assert _lengths_by_session(frame[frame["session"].isin(train_sessions)]) == [4, 4]


def test_fit_hmm_scaler_uses_train_only() -> None:
    frame = _hmm_feature_frame()
    feature_columns = ["ret_1", "ret_3", "rv_6", "rv_12", "range", "rel_volume", "trend_12", "intraday_drawdown"]
    train = frame[frame["session"].isin(["2024-01-02", "2024-01-03"])].copy()

    _, scaler = fit_hmm_model(train, feature_columns, n_states=2, covariance_type="diag", random_state=1, n_iter=5)

    assert np.allclose(scaler.mean_, train[feature_columns].mean().to_numpy())

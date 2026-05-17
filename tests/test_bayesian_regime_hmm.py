from __future__ import annotations

import numpy as np
import pandas as pd

from src.bayesian_regime_hmm import (
    add_regime_probability_columns,
    build_h8_observations,
    filtered_regime_probabilities,
    h8_regime_specs,
    h8_transition_matrix,
    tradingview_regime_specs,
    tradingview_transition_matrix,
)


def test_tradingview_transition_matrix_matches_script_weights() -> None:
    matrix = tradingview_transition_matrix(p_stay_bull=0.80, p_stay_bear=0.80, p_stay_chop=0.60)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.allclose(matrix[0], [0.80, 0.04, 0.16])
    assert np.allclose(matrix[1], [0.04, 0.80, 0.16])
    assert np.allclose(matrix[2], [0.20, 0.20, 0.60])


def test_filtered_regime_probabilities_are_causal_and_reset_by_session() -> None:
    observations = pd.DataFrame(
        {
            "mom_z": [1.0, -1.0, 1.0],
            "vol_z": [-0.5, 1.0, -0.5],
        }
    )
    sessions = pd.Series(["a", "a", "b"])

    probabilities = filtered_regime_probabilities(
        observations,
        tradingview_regime_specs(),
        tradingview_transition_matrix(),
        sessions=sessions,
    )

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] == probabilities[0].max()
    assert probabilities[1, 1] == probabilities[1].max()
    assert np.allclose(probabilities[0], probabilities[2])


def test_invalid_observation_rows_emit_nan_and_reset_prior() -> None:
    observations = pd.DataFrame(
        {
            "mom_z": [1.0, np.nan, 1.0],
            "vol_z": [-0.5, 1.0, -0.5],
        }
    )
    sessions = pd.Series(["a", "a", "a"])

    probabilities = filtered_regime_probabilities(
        observations,
        tradingview_regime_specs(),
        tradingview_transition_matrix(),
        sessions=sessions,
    )

    assert np.isnan(probabilities[1]).all()
    assert np.allclose(probabilities[0], probabilities[2])


def test_add_regime_probability_columns_uses_state_names() -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    specs = tradingview_regime_specs()
    probabilities = np.array([[0.80, 0.10, 0.10], [0.20, 0.70, 0.10]])

    annotated = add_regime_probability_columns(frame, probabilities, specs)

    assert "bayes_hmm_p_bull_trend" in annotated.columns
    assert annotated["bayes_hmm_state"].tolist() == ["bull_trend", "bear_stress"]
    assert annotated["bayes_hmm_state_id"].tolist() == [0, 1]
    assert annotated["bayes_hmm_entropy"].between(0, 1).all()


def test_h8_defaults_are_four_state_and_observation_builder_emits_expected_columns() -> None:
    idx = np.arange(80, dtype=float)
    close = 100.0 + (idx * 0.08) + np.sin(idx / 3.0)
    frame = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.4 + (idx % 5) * 0.03,
            "low": close - 0.4 - (idx % 7) * 0.02,
        }
    )

    observations = build_h8_observations(frame, length=12)

    assert [state.name for state in h8_regime_specs()] == [
        "bull_trend",
        "bear_stress",
        "chop_compression",
        "volatile_noise",
    ]
    assert h8_transition_matrix().shape == (4, 4)
    assert {"mom_z", "vol_z", "eff_z"}.issubset(observations.columns)
    assert observations.tail(10).notna().all().all()

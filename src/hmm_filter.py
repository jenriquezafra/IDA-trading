from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def filtered_probabilities(model, observations: np.ndarray, sessions: pd.Series) -> np.ndarray:
    """Run causal forward filtering, resetting the prior at each session."""
    if len(observations) != len(sessions):
        raise ValueError("observations and sessions must have the same length")
    if len(observations) == 0:
        return np.empty((0, model.n_components))

    log_likelihood = model._compute_log_likelihood(observations)
    log_start = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_trans = np.log(np.clip(model.transmat_, 1e-300, 1.0))

    probabilities = np.empty_like(log_likelihood)
    previous_log_alpha: np.ndarray | None = None
    previous_session = None

    for idx, session in enumerate(sessions.to_numpy()):
        if previous_log_alpha is None or session != previous_session:
            log_alpha = log_start + log_likelihood[idx]
        else:
            prediction = logsumexp(previous_log_alpha[:, None] + log_trans, axis=0)
            log_alpha = prediction + log_likelihood[idx]

        log_alpha = log_alpha - logsumexp(log_alpha)
        probabilities[idx] = np.exp(log_alpha)
        previous_log_alpha = log_alpha
        previous_session = session

    return probabilities


def add_hmm_probability_columns(df: pd.DataFrame, probabilities: np.ndarray, prefix: str = "hmm_p") -> pd.DataFrame:
    if len(df) != len(probabilities):
        raise ValueError("df and probabilities must have the same number of rows")

    annotated = df.copy()
    n_states = probabilities.shape[1]
    for state in range(n_states):
        annotated[f"{prefix}{state}"] = probabilities[:, state]

    annotated["hmm_state"] = probabilities.argmax(axis=1)
    annotated["hmm_max_prob"] = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))).sum(axis=1)
    annotated["hmm_entropy"] = entropy / np.log(n_states)
    return annotated

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    means: dict[str, float]
    sigmas: dict[str, float]


def tradingview_regime_specs() -> list[RegimeSpec]:
    """Three-state baseline equivalent to the TradingView script emissions."""
    return [
        RegimeSpec("bull_trend", {"mom_z": 1.0, "vol_z": -0.5}, {"mom_z": 1.0, "vol_z": 1.0}),
        RegimeSpec("bear_stress", {"mom_z": -1.0, "vol_z": 1.0}, {"mom_z": 1.0, "vol_z": 1.0}),
        RegimeSpec("chop_noise", {"mom_z": 0.0, "vol_z": 1.5}, {"mom_z": 0.5, "vol_z": 1.0}),
    ]


def h8_regime_specs() -> list[RegimeSpec]:
    """Improved H8 starting point: split quiet chop from volatile noise."""
    return [
        RegimeSpec(
            "bull_trend",
            {"mom_z": 1.0, "vol_z": -0.25, "eff_z": 0.75},
            {"mom_z": 1.0, "vol_z": 1.0, "eff_z": 0.85},
        ),
        RegimeSpec(
            "bear_stress",
            {"mom_z": -1.0, "vol_z": 1.0, "eff_z": -0.75},
            {"mom_z": 1.0, "vol_z": 1.0, "eff_z": 0.85},
        ),
        RegimeSpec(
            "chop_compression",
            {"mom_z": 0.0, "vol_z": -0.75, "eff_z": 0.0},
            {"mom_z": 0.65, "vol_z": 1.0, "eff_z": 0.7},
        ),
        RegimeSpec(
            "volatile_noise",
            {"mom_z": 0.0, "vol_z": 1.25, "eff_z": 0.0},
            {"mom_z": 0.75, "vol_z": 1.0, "eff_z": 0.7},
        ),
    ]


def tradingview_transition_matrix(
    p_stay_bull: float = 0.80,
    p_stay_bear: float = 0.80,
    p_stay_chop: float = 0.60,
) -> np.ndarray:
    names = ["bull_trend", "bear_stress", "chop_noise"]
    stay = {
        "bull_trend": p_stay_bull,
        "bear_stress": p_stay_bear,
        "chop_noise": p_stay_chop,
    }
    exits = {
        "bull_trend": {"bear_stress": 0.2, "chop_noise": 0.8},
        "bear_stress": {"bull_trend": 0.2, "chop_noise": 0.8},
        "chop_noise": {"bull_trend": 0.5, "bear_stress": 0.5},
    }
    return transition_matrix_from_stay_probabilities(names, stay, exits)


def h8_transition_matrix() -> np.ndarray:
    names = [state.name for state in h8_regime_specs()]
    stay = {
        "bull_trend": 0.82,
        "bear_stress": 0.82,
        "chop_compression": 0.70,
        "volatile_noise": 0.65,
    }
    exits = {
        "bull_trend": {"bear_stress": 0.15, "chop_compression": 0.55, "volatile_noise": 0.30},
        "bear_stress": {"bull_trend": 0.15, "chop_compression": 0.25, "volatile_noise": 0.60},
        "chop_compression": {"bull_trend": 0.35, "bear_stress": 0.25, "volatile_noise": 0.40},
        "volatile_noise": {"bull_trend": 0.25, "bear_stress": 0.35, "chop_compression": 0.40},
    }
    return transition_matrix_from_stay_probabilities(names, stay, exits)


def transition_matrix_from_stay_probabilities(
    state_names: list[str],
    stay_probabilities: dict[str, float],
    exit_weights: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    if not state_names:
        raise ValueError("state_names must not be empty")
    duplicate_names = sorted({name for name in state_names if state_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate state names: {duplicate_names}")

    n_states = len(state_names)
    name_to_idx = {name: idx for idx, name in enumerate(state_names)}
    matrix = np.zeros((n_states, n_states), dtype=float)

    for state in state_names:
        stay = float(stay_probabilities.get(state, 0.0))
        if stay < 0.0 or stay >= 1.0:
            raise ValueError(f"Stay probability for {state} must be in [0, 1)")

        row_idx = name_to_idx[state]
        matrix[row_idx, row_idx] = stay
        remaining = 1.0 - stay
        targets = [target for target in state_names if target != state]
        weights = (exit_weights or {}).get(state)
        if weights is None:
            weight_values = {target: 1.0 for target in targets}
        else:
            unknown = sorted(set(weights) - set(targets))
            if unknown:
                raise ValueError(f"Unknown transition targets for {state}: {unknown}")
            weight_values = {target: float(weights.get(target, 0.0)) for target in targets}

        total_weight = sum(weight_values.values())
        if total_weight <= 0.0:
            raise ValueError(f"Exit weights for {state} must sum to a positive value")
        for target in targets:
            matrix[row_idx, name_to_idx[target]] = remaining * weight_values[target] / total_weight

    return matrix


def build_tradingview_observations(
    frame: pd.DataFrame,
    length: int = 20,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    _require_columns(frame, [close_col, high_col, low_col])
    close = frame[close_col].astype(float)
    high = frame[high_col].astype(float)
    low = frame[low_col].astype(float)

    mom_raw = close.pct_change() * 100.0
    mom_smooth = mom_raw.ewm(span=int(length), adjust=False, min_periods=int(length)).mean()
    obs_mom = _rolling_zscore(mom_smooth, length)

    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / int(length), adjust=False, min_periods=int(length)).mean()
    obs_vol = _rolling_zscore(atr, length)

    return pd.DataFrame({"mom_z": obs_mom, "vol_z": obs_vol}, index=frame.index)


def build_h8_observations(
    frame: pd.DataFrame,
    length: int = 20,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    observations = build_tradingview_observations(frame, length=length, close_col=close_col, high_col=high_col, low_col=low_col)
    close = frame[close_col].astype(float)
    log_ret = np.log(close / close.shift(1))
    signed_efficiency = log_ret.rolling(window=int(length), min_periods=int(length)).sum() / log_ret.abs().rolling(
        window=int(length), min_periods=int(length)
    ).sum()
    observations["eff_z"] = _rolling_zscore(signed_efficiency, length)
    return observations


def filtered_regime_probabilities(
    observations: pd.DataFrame,
    state_specs: list[RegimeSpec],
    transition_matrix: np.ndarray,
    sessions: pd.Series | None = None,
    start_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    feature_columns = _feature_columns(state_specs)
    _require_columns(observations, feature_columns)
    transition = _validate_transition_matrix(transition_matrix, len(state_specs))
    start = _validate_start_probabilities(start_probabilities, len(state_specs))

    values = observations[feature_columns].replace([np.inf, -np.inf], np.nan).astype(float).to_numpy()
    valid = np.isfinite(values).all(axis=1)
    means = np.array([[state.means[column] for column in feature_columns] for state in state_specs], dtype=float)
    sigmas = np.array([[state.sigmas[column] for column in feature_columns] for state in state_specs], dtype=float)
    if np.any(sigmas <= 0.0):
        raise ValueError("All state sigmas must be positive")

    log_start = np.log(np.clip(start, 1e-300, 1.0))
    log_transition = np.log(np.clip(transition, 1e-300, 1.0))
    session_values = sessions.to_numpy() if sessions is not None else np.zeros(len(observations), dtype=int)
    if len(session_values) != len(observations):
        raise ValueError("sessions and observations must have the same length")

    probabilities = np.full((len(observations), len(state_specs)), np.nan, dtype=float)
    previous_log_alpha: np.ndarray | None = None
    previous_session: Any = None

    for idx, obs in enumerate(values):
        session = session_values[idx]
        if not valid[idx]:
            previous_log_alpha = None
            previous_session = session
            continue

        log_likelihood = _gaussian_log_likelihood(obs, means, sigmas)
        if previous_log_alpha is None or session != previous_session:
            log_alpha = log_start + log_likelihood
        else:
            prediction = logsumexp(previous_log_alpha[:, None] + log_transition, axis=0)
            log_alpha = prediction + log_likelihood

        log_alpha = log_alpha - logsumexp(log_alpha)
        probabilities[idx] = np.exp(log_alpha)
        previous_log_alpha = log_alpha
        previous_session = session

    return probabilities


def add_regime_probability_columns(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    state_specs: list[RegimeSpec],
    prefix: str = "bayes_hmm",
) -> pd.DataFrame:
    if len(frame) != len(probabilities):
        raise ValueError("frame and probabilities must have the same number of rows")
    if probabilities.shape[1] != len(state_specs):
        raise ValueError("probability column count must match state_specs")

    annotated = frame.copy()
    state_names = [state.name for state in state_specs]
    for idx, name in enumerate(state_names):
        annotated[f"{prefix}_p_{name}"] = probabilities[:, idx]

    valid = np.isfinite(probabilities).all(axis=1)
    state_ids = np.full(len(annotated), -1, dtype=int)
    state_ids[valid] = np.argmax(probabilities[valid], axis=1)
    annotated[f"{prefix}_state_id"] = state_ids
    annotated[f"{prefix}_state"] = pd.Series(state_ids, index=annotated.index).map(
        {idx: name for idx, name in enumerate(state_names)}
    )
    max_probabilities = np.full(len(annotated), np.nan, dtype=float)
    max_probabilities[valid] = probabilities[valid].max(axis=1)
    annotated[f"{prefix}_max_prob"] = max_probabilities
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))).sum(axis=1)
    annotated[f"{prefix}_entropy"] = entropy / np.log(len(state_specs))
    annotated.loc[~valid, [f"{prefix}_max_prob", f"{prefix}_entropy"]] = np.nan
    return annotated


def _feature_columns(state_specs: list[RegimeSpec]) -> list[str]:
    if not state_specs:
        raise ValueError("state_specs must not be empty")
    first = list(state_specs[0].means)
    for state in state_specs:
        if list(state.means) != first:
            raise ValueError("All state specs must use the same mean feature order")
        if list(state.sigmas) != first:
            raise ValueError("State means and sigmas must use the same feature order")
    return first


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=int(window), min_periods=int(window)).mean()
    rolling_std = series.rolling(window=int(window), min_periods=int(window)).std()
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _gaussian_log_likelihood(obs: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    variance = sigmas * sigmas
    return (-0.5 * np.log(2.0 * np.pi * variance) - ((obs - means) ** 2) / (2.0 * variance)).sum(axis=1)


def _validate_transition_matrix(matrix: np.ndarray, n_states: int) -> np.ndarray:
    transition = np.asarray(matrix, dtype=float)
    if transition.shape != (n_states, n_states):
        raise ValueError(f"transition_matrix must have shape {(n_states, n_states)}")
    if np.any(transition < 0.0):
        raise ValueError("transition_matrix must not contain negative values")
    row_sums = transition.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("transition_matrix rows must sum to a positive value")
    return transition / row_sums[:, None]


def _validate_start_probabilities(start_probabilities: np.ndarray | None, n_states: int) -> np.ndarray:
    if start_probabilities is None:
        return np.full(n_states, 1.0 / n_states)
    start = np.asarray(start_probabilities, dtype=float)
    if start.shape != (n_states,):
        raise ValueError(f"start_probabilities must have shape {(n_states,)}")
    if np.any(start < 0.0) or start.sum() <= 0.0:
        raise ValueError("start_probabilities must be non-negative and sum to a positive value")
    return start / start.sum()

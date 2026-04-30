from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import build_labels


def _config() -> dict:
    return {
        "labeling": {
            "horizon_bars": 2,
            "buffer_bps": 0.5,
            "lambda_vol": 0.25,
            "round_trip_cost_bps": 1.0,
        }
    }


def _features_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=78, freq="5min", tz="America/New_York")
    open_px = 100.0 + np.arange(78, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_px,
            "high": open_px + 0.2,
            "low": open_px - 0.2,
            "close": open_px + 0.05,
            "volume": 1000,
            "session": "2024-01-02",
            "bar_index": np.arange(78),
            "bars_in_session": 78,
            "rv_12": 0.001,
            "target_crosses_session_close": np.arange(78) + 3 >= 78,
        }
    )


def test_build_labels_uses_next_open_entry_and_horizon_exit() -> None:
    labels = build_labels(_features_frame(), _config(), drop_invalid=False)

    assert labels.loc[0, "entry_px"] == labels.loc[1, "open"]
    assert labels.loc[0, "exit_px"] == labels.loc[3, "open"]
    assert labels.loc[0, "entry_timestamp"] == labels.loc[1, "timestamp"]
    assert labels.loc[0, "exit_timestamp"] == labels.loc[3, "timestamp"]
    assert labels.loc[0, "fwd_ret"] == np.log(labels.loc[3, "open"] / labels.loc[1, "open"])


def test_build_labels_neutral_zone_uses_cost_plus_buffer_floor() -> None:
    labels = build_labels(_features_frame(), _config(), drop_invalid=False)

    cost_plus_buffer = (1.0 + 0.5) / 10_000.0
    sigma_component = 0.25 * 0.001 * np.sqrt(2)
    assert labels.loc[20, "sigma_h"] == 0.001 * np.sqrt(2)
    assert labels.loc[20, "neutral_zone"] == max(cost_plus_buffer, sigma_component)


def test_build_labels_drops_rows_that_cross_session_close_or_lack_vol() -> None:
    features = _features_frame()
    features.loc[0, "rv_12"] = np.nan

    labels = build_labels(features, _config(), drop_invalid=True)

    assert labels["bar_index"].min() == 1
    assert labels["bar_index"].max() == 74
    assert not labels["target_crosses_session_close"].any()
    assert labels[["entry_px", "exit_px", "fwd_ret", "sigma_h", "neutral_zone"]].notna().all().all()


def test_build_labels_creates_ternary_target() -> None:
    features = _features_frame()
    features.loc[:, "rv_12"] = 0.0
    features.loc[4, "open"] = 90.0

    labels = build_labels(features, _config(), drop_invalid=False)

    assert set(labels["target"].dropna().unique()).issubset({-1, 0, 1})
    assert labels.loc[1, "target"] == -1
    assert labels.loc[3, "target"] == 1

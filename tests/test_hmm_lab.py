from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hmm_lab import FULL_CORE_TOKEN, build_lab_folds, prepare_hmm_frame, resolve_feature_sets, summarize_grid


def _daily_frame(months: int = 5) -> pd.DataFrame:
    sessions = pd.date_range("2024-01-02", periods=months * 21, freq="B").strftime("%Y-%m-%d")
    rows = []
    for session in sessions:
        for bar_index in range(2):
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{session} 09:{30 + 5 * bar_index}:00"),
                    "session": session,
                    "bar_index": bar_index,
                    "feature_a": float(bar_index),
                    "feature_b": float(bar_index + 1),
                }
            )
    return pd.DataFrame(rows)


def test_resolve_feature_sets_expands_full_core_token_and_reports_missing() -> None:
    config = {
        "hmm_lab": {
            "feature_sets": [
                {"name": "full", "columns": FULL_CORE_TOKEN},
                {"name": "bad", "columns": ["feature_a", "missing_feature"]},
            ]
        }
    }
    feature_config = {"hmm_feature_columns": ["feature_a", "feature_b"]}
    features = _daily_frame()

    ready, validation = resolve_feature_sets(config, feature_config, features)

    assert [item["name"] for item in ready] == ["full"]
    assert ready[0]["columns"] == ["feature_a", "feature_b"]
    assert validation.loc[validation["feature_set"] == "bad", "status"].iloc[0] == "missing_columns"
    assert validation.loc[validation["feature_set"] == "bad", "missing_columns"].iloc[0] == "missing_feature"


def test_prepare_hmm_frame_drops_nan_and_infinite_rows() -> None:
    features = _daily_frame(months=1)
    features.loc[0, "feature_a"] = np.nan
    features.loc[1, "feature_b"] = np.inf

    prepared = prepare_hmm_frame(features, ["feature_a", "feature_b"])

    assert len(prepared) == len(features) - 2
    assert "source_index" in prepared.columns
    assert prepared[["feature_a", "feature_b"]].notna().all().all()


def test_prepare_hmm_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required HMM columns"):
        prepare_hmm_frame(_daily_frame(), ["feature_a", "missing_feature"])


def test_build_lab_folds_uses_configured_month_windows() -> None:
    config = {
        "hmm_lab": {
            "max_folds": 2,
            "walk_forward": {
                "train_months": 2,
                "validation_months": 1,
                "test_months": 1,
                "step_months": 1,
            },
        }
    }

    folds = build_lab_folds(_daily_frame(months=6), config)

    assert len(folds) == 2
    assert folds[0].train_months == ["2024-01", "2024-02"]
    assert folds[0].validation_months == ["2024-03"]
    assert folds[0].test_months == ["2024-04"]
    assert folds[1].train_months == ["2024-02", "2024-03"]


def test_summarize_grid_ranks_by_validation_not_test() -> None:
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "a",
                "n_features": 2,
                "n_states": 3,
                "seed": 42,
                "split": "validation",
                "fold": 0,
                "status": "ok",
                "rows": 100,
                "total_loglik": -100.0,
                "avg_loglik": -1.0,
                "converged": True,
                "mean_hmm_entropy": 0.2,
                "mean_hmm_max_prob": 0.8,
                "min_state_frequency": 0.2,
                "empty_state_count": 0,
                "top_hour_pct": 0.3,
            },
            {
                "feature_set": "b",
                "n_features": 2,
                "n_states": 3,
                "seed": 42,
                "split": "validation",
                "fold": 0,
                "status": "ok",
                "rows": 100,
                "total_loglik": -120.0,
                "avg_loglik": -1.2,
                "converged": True,
                "mean_hmm_entropy": 0.2,
                "mean_hmm_max_prob": 0.8,
                "min_state_frequency": 0.2,
                "empty_state_count": 0,
                "top_hour_pct": 0.3,
            },
            {
                "feature_set": "b",
                "n_features": 2,
                "n_states": 3,
                "seed": 42,
                "split": "test",
                "fold": 0,
                "status": "ok",
                "rows": 100,
                "total_loglik": -50.0,
                "avg_loglik": -0.5,
                "converged": True,
                "mean_hmm_entropy": 0.2,
                "mean_hmm_max_prob": 0.8,
                "min_state_frequency": 0.2,
                "empty_state_count": 0,
                "top_hour_pct": 0.3,
            },
        ]
    )

    summary = summarize_grid(metrics)

    assert summary.loc[0, "feature_set"] == "a"
    assert summary.loc[0, "validation_rank"] == 1

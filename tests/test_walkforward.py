from __future__ import annotations

import pandas as pd

from src.walkforward import apply_purge_and_embargo, build_monthly_folds, render_summary


def _config() -> dict:
    return {
        "walkforward": {
            "train_months": 5,
            "validation_months": 1,
            "test_months": 1,
            "step_months": 1,
            "purge_bars": 2,
            "embargo_bars": 1,
            "output_dir": "reports/walkforward",
            "summary_file": "reports/walkforward_folds_summary.md",
        },
        "labeling": {"horizon_bars": 2},
    }


def _labels(months: int = 8) -> pd.DataFrame:
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


def test_build_monthly_folds_uses_configured_5_1_1_schema() -> None:
    folds = build_monthly_folds(_labels(months=8), _config())

    assert len(folds) == 2
    assert folds[0].train_months == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    assert folds[0].validation_months == ["2024-06"]
    assert folds[0].test_months == ["2024-07"]
    assert folds[1].train_months == ["2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    assert folds[1].validation_months == ["2024-07"]
    assert folds[1].test_months == ["2024-08"]


def test_apply_purge_and_embargo_removes_boundary_rows() -> None:
    labels = _labels(months=7)
    fold = build_monthly_folds(labels, _config())[0]

    splits = apply_purge_and_embargo(labels, fold, _config())

    assert splits["train"].groupby("session").size().max() == 2
    assert splits["validation"].groupby("session").size().max() == 3
    assert splits["test"].groupby("session").size().max() == 4


def test_render_summary_reports_insufficient_data() -> None:
    labels = _labels(months=3)
    folds = build_monthly_folds(labels, _config())

    report = render_summary(pd.DataFrame(), folds, _config(), labels)

    assert "No folds were generated" in report
    assert "too short" in report

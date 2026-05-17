from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alpha.risk_off_eda import build_condition_summary, build_control_pnl, load_eda_frame, run_eda


def _features() -> pd.DataFrame:
    rows = []
    for session in ["2024-01-03", "2024-01-04"]:
        for bar in range(8):
            direction = -1 if bar >= 2 else 1
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{session} 09:30:00") + pd.Timedelta(minutes=15 * bar),
                    "session": session,
                    "bar_index": bar,
                    "target_open_next": 100.0 - 0.2 * bar if session == "2024-01-03" else 101.0 - 0.1 * bar,
                    "target_can_open_trade": bar < 7,
                    "target_ret_6": direction * 0.01,
                    "target_ret_12": direction * 0.02,
                    "target_dist_vwap_atr": direction * 0.5,
                    "risk_off_score": float(bar),
                    "risk_on_score": float(8 - bar),
                    "spread_credit_12": -float(bar),
                    "defensive_rotation_score": float(bar),
                    "intraday_stress_score": float(bar) / 10.0,
                }
            )
    return pd.DataFrame(rows)


def _risk_context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "available_session": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "prev_vix_z20": [1.0, 2.0],
            "prev_vix9d_vix_ratio": [1.1, 1.2],
            "prev_total_put_call_ratio_z20": [0.5, 0.8],
            "prev_index_put_call_ratio_z20": [0.2, 0.3],
        }
    )


def test_load_eda_frame_joins_context_by_available_session(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    context_path = tmp_path / "risk_context.parquet"
    _features().to_parquet(features_path, index=False)
    _risk_context().to_parquet(context_path, index=False)

    frame = load_eda_frame(features_path, context_path, horizons=(2,))

    assert "fwd_ret_2" in frame.columns
    assert frame.loc[0, "prev_vix_z20"] == 1.0
    assert frame.loc[8, "prev_vix_z20"] == 2.0


def test_condition_summary_contains_h1_core(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    context_path = tmp_path / "risk_context.parquet"
    _features().to_parquet(features_path, index=False)
    _risk_context().to_parquet(context_path, index=False)
    frame = load_eda_frame(features_path, context_path, horizons=(2,))

    summary = build_condition_summary(frame, horizons=(2,))

    assert "h1_core" in set(summary["label"])
    assert summary.loc[summary["label"].eq("h1_core"), "valid_returns"].iloc[0] > 0


def test_control_pnl_contains_candidate_and_controls(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    context_path = tmp_path / "risk_context.parquet"
    _features().to_parquet(features_path, index=False)
    _risk_context().to_parquet(context_path, index=False)
    frame = load_eda_frame(features_path, context_path, horizons=(2,))

    pnl = build_control_pnl(frame, horizons=(2,), cost_bps_values=(2.0,))

    labels = set(pnl["label"])
    assert "target_breakdown__risk_off__vix_pressure" in labels
    assert "same_hour_short_control" in labels
    assert "random_same_count_control" in labels


def test_run_eda_writes_report_and_artifacts(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    context_path = tmp_path / "risk_context.parquet"
    _features().to_parquet(features_path, index=False)
    _risk_context().to_parquet(context_path, index=False)

    outputs = run_eda(features_path=features_path, risk_context_path=context_path, output_dir=tmp_path / "eda", horizons=(2,))

    assert outputs.report_path.exists()
    assert outputs.condition_summary_path.exists()
    assert outputs.bucket_summary_path.exists()
    assert outputs.yearly_summary_path.exists()
    assert outputs.control_pnl_path.exists()

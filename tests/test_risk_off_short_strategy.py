from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.strategy.risk_off_short import CANDIDATE_LABEL, fit_thresholds, run_strategy, simulate_trades


def _features() -> pd.DataFrame:
    rows = []
    for month in range(1, 5):
        session = f"2024-{month:02d}-02"
        for bar in range(8):
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"{session} 09:30:00") + pd.Timedelta(minutes=15 * bar),
                    "session": session,
                    "bar_index": bar,
                    "target_open_next": 100.0 - month - 0.25 * bar,
                    "target_next_open_timestamp": pd.Timestamp(f"{session} 09:45:00") + pd.Timedelta(minutes=15 * bar),
                    "target_can_open_trade": bar < 7,
                    "target_ret_6": -0.01 if bar >= 2 else 0.01,
                    "target_ret_12": -0.02 if bar >= 2 else 0.02,
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
            "source_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "available_session": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-02", "2024-04-02"]),
            "prev_vix_z20": [1.0, 1.2, 1.4, 1.6],
            "prev_vix9d_vix_ratio": [1.1, 1.1, 1.2, 1.2],
            "prev_total_put_call_ratio_z20": [0.2, 0.3, 0.4, 0.5],
            "prev_index_put_call_ratio_z20": [0.2, 0.3, 0.4, 0.5],
        }
    )


def test_simulate_trades_creates_non_overlapping_short_trades() -> None:
    frame = _features().head(8).copy()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["prev_vix_z20"] = 1.0
    thresholds = fit_thresholds(frame)
    signal = pd.Series([False, False, True, True, True, True, False, False], index=frame.index)

    trades = simulate_trades(frame, signal, label=CANDIDATE_LABEL, fold=0, split="test", horizon=2, cost_bps=2.0, thresholds=thresholds)

    assert len(trades) == 2
    assert trades["net_return"].iloc[0] > 0
    assert set(trades["label"]) == {CANDIDATE_LABEL}


def test_run_strategy_writes_standard_outputs(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    context_path = tmp_path / "risk_context.parquet"
    _features().to_parquet(features_path, index=False)
    _risk_context().to_parquet(context_path, index=False)

    outputs = run_strategy(
        features_path=features_path,
        risk_context_path=context_path,
        output_dir=tmp_path / "strategy",
        horizons=(2,),
        cost_bps_values=(2.0,),
        split_policy={"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1},
    )

    assert outputs.trades_path.exists()
    assert outputs.daily_path.exists()
    assert outputs.monthly_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.report_path.exists()
    summary = pd.read_parquet(outputs.summary_path)
    assert CANDIDATE_LABEL in set(summary["label"])

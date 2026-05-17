from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.promotion import evaluate_promotion_gates
from src.strategy.risk_off_short import CANDIDATE_LABEL
from src.strategy.risk_off_short_triage import control_rollup, enrich_trade_times, run_triage, session_concentration


def _summary() -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        for fold, net in enumerate([0.01, -0.004]):
            rows.append(
                {
                    "strategy_id": "risk_off_short_v1",
                    "label": CANDIDATE_LABEL,
                    "fold": fold,
                    "split": split,
                    "horizon_bars": 6,
                    "cost_bps": 2.0,
                    "trades": 2,
                    "gross_return": net + 0.0004,
                    "total_cost": 0.0004,
                    "net_return": net,
                    "avg_trade_net": net / 2,
                    "profit_factor": 1.2,
                    "daily_sharpe": 0.5,
                    "max_drawdown": 0.01,
                    "win_rate": 0.5,
                    "worst_day": "2024-01-03",
                    "worst_day_net": -0.004,
                    "best_day": "2024-01-02",
                    "best_day_net": 0.01,
                    "sessions": 2,
                }
            )
            rows.append(
                {
                    "strategy_id": "risk_off_short_v1",
                    "label": "target_breakdown",
                    "fold": fold,
                    "split": split,
                    "horizon_bars": 6,
                    "cost_bps": 2.0,
                    "trades": 3,
                    "gross_return": -0.002,
                    "total_cost": 0.0006,
                    "net_return": -0.002,
                    "avg_trade_net": -0.002 / 3,
                    "profit_factor": 0.8,
                    "daily_sharpe": -0.2,
                    "max_drawdown": 0.02,
                    "win_rate": 0.4,
                    "worst_day": "2024-01-03",
                    "worst_day_net": -0.004,
                    "best_day": "2024-01-02",
                    "best_day_net": 0.002,
                    "sessions": 2,
                }
            )
    return pd.DataFrame(rows)


def _trades() -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        for fold in (0, 1):
            for bar, net in ((12, 0.006), (18, -0.002)):
                session = f"2024-01-0{fold + 2}"
                rows.append(
                    {
                        "strategy_id": "risk_off_short_v1",
                        "label": CANDIDATE_LABEL,
                        "fold": fold,
                        "split": split,
                        "horizon_bars": 6,
                        "session": session,
                        "signal_timestamp": pd.Timestamp(f"{session} 09:30:00", tz="America/New_York")
                        + pd.Timedelta(minutes=15 * bar),
                        "entry_timestamp": pd.Timestamp(f"{session} 09:45:00", tz="America/New_York")
                        + pd.Timedelta(minutes=15 * bar),
                        "exit_timestamp": pd.Timestamp(f"{session} 11:15:00", tz="America/New_York")
                        + pd.Timedelta(minutes=15 * bar),
                        "bar_index": bar,
                        "entry_px": 100.0,
                        "exit_px": 99.5,
                        "side": "short",
                        "gross_return": net + 0.0002,
                        "risk_off_min": 0.5,
                        "vix_z20_min": 0.6,
                        "risk_off_score": 0.7,
                        "prev_vix_z20": 0.8,
                        "target_ret_6": -0.01,
                        "target_ret_12": -0.01,
                        "cost_bps": 2.0,
                        "cost_return": 0.0002,
                        "net_return": net,
                    }
                )
    return pd.DataFrame(rows)


def test_enrich_trade_times_adds_intraday_bucket_and_weekday() -> None:
    enriched = enrich_trade_times(_trades())

    assert set(enriched["intraday_bucket"].astype(str)) == {"midday_1230_1345", "afternoon_1400_close"}
    assert set(enriched["weekday"]) == {"Tuesday", "Wednesday"}


def test_control_rollup_aggregates_folds_and_controls() -> None:
    rollup = control_rollup(_summary())
    candidate = rollup[(rollup["split"] == "validation") & (rollup["label"] == CANDIDATE_LABEL)].iloc[0]

    assert candidate["folds"] == 2
    assert candidate["trades"] == 4
    assert candidate["positive_folds"] == 1
    assert round(candidate["net_return"], 6) == 0.006


def test_session_concentration_reports_abs_share() -> None:
    concentration = session_concentration(enrich_trade_times(_trades()))

    assert set(concentration["split"]) == {"validation", "test"}
    assert concentration["top1_abs_share"].between(0.0, 1.0).all()
    assert concentration["sessions_with_trades"].min() == 1


def test_evaluate_promotion_gates_blocks_concentrated_candidate() -> None:
    summary = _summary().copy()
    summary["cost_bps"] = 2.0
    stress = summary.copy()
    stress["cost_bps"] = 5.0
    stress["net_return"] = stress["net_return"] - 0.001
    selected_summary = pd.concat([summary, stress], ignore_index=True)
    concentration = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "fold": [0, 0],
            "sessions_with_trades": [2, 2],
            "net_return": [0.006, 0.006],
            "top1_abs_share": [0.6, 0.6],
            "top5_abs_share": [1.0, 1.0],
        }
    )

    gates, decision = evaluate_promotion_gates(
        selected_summary,
        concentration,
        {
            "min_validation_trades": 2,
            "min_test_trades": 2,
            "min_validation_positive_folds": 1,
            "min_test_positive_folds": 1,
            "min_validation_net_return": 0.0,
            "min_test_net_return": 0.0,
            "min_avg_trade_net_bps": 0.0,
            "stress_cost_bps": 5.0,
            "min_validation_stress_net_return": 0.0,
            "min_test_stress_net_return": 0.0,
            "min_sessions_per_fold": 3,
            "max_top5_abs_share": 0.7,
            "require_beats_best_control": True,
        },
        candidate_label=CANDIDATE_LABEL,
    )

    assert decision["status"] == "continue_research"
    assert "validation_top5_abs_share" in set(gates.loc[gates["status"] == "fail", "gate_id"])


def test_run_triage_writes_outputs_without_sensitivity(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "strategy"
    strategy_dir.mkdir()
    _summary().to_parquet(strategy_dir / "summary.parquet", index=False)
    _trades().to_parquet(strategy_dir / "trades.parquet", index=False)

    outputs = run_triage(strategy_dir=strategy_dir, output_dir=tmp_path / "triage", run_sensitivity=False)

    assert outputs.report_path.exists()
    assert outputs.controls_rollup_path.exists()
    assert outputs.session_concentration_path.exists()
    assert outputs.selected_threshold_confirmation_path.exists()
    assert outputs.promotion_gates_path.exists()
    assert outputs.promotion_decision_path.exists()

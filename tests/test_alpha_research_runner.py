from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.alpha.research import add_forward_returns, dry_run_summary, run_alpha_research
from src.alpha.specs import load_alpha_research_plan
from src.research.splits import build_monthly_folds


def _sample_features() -> pd.DataFrame:
    rows = []
    for month in range(1, 5):
        session = f"2026-{month:02d}-02"
        for bar in range(5):
            px = 100.0 + month + bar
            timestamp = pd.Timestamp(f"{session} 09:30:00") + pd.Timedelta(minutes=15 * bar)
            rows.append(
                {
                    "timestamp": timestamp,
                    "session": session,
                    "bar_index": bar,
                    "target_open_next": px + 0.1,
                    "target_can_open_trade": bar < 4,
                    "target_ret_1": 0.001 * (1 if bar % 2 == 0 else -1),
                    "target_ret_6": 0.002 * (1 if bar % 2 == 0 else -1),
                    "target_ret_12": 0.003 * (1 if bar % 2 == 0 else -1),
                    "target_dist_vwap_atr": 0.5 * (1 if bar % 2 == 0 else -1),
                    "intraday_stress_score": 0.1 * bar,
                    "risk_on_score": float(bar),
                    "risk_off_score": float(4 - bar),
                }
            )
    return pd.DataFrame(rows)


def _write_plan(tmp_path: Path) -> Path:
    features_path = tmp_path / "features.parquet"
    _sample_features().to_parquet(features_path, index=False)
    config = {
        "research": {
            "research_id": "unit_alpha",
            "target_symbol": "QQQ",
            "timeframe": "15min",
            "split_policy_id": "unit_2_1_1",
            "output_dir_template": str(tmp_path / "out"),
            "split_policy": {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1},
        },
        "dataset": {
            "feature_set_id": "unit_features",
            "feature_path_template": str(features_path),
        },
        "costs": {"primary": "bps_1", "conservative": "bps_2", "stress": "bps_5"},
        "promotion_gates": {"min_trades": 1, "min_profit_factor": 0.0, "min_daily_sharpe": -100.0},
        "alphas": [
            {
                "alpha_id": "m6_base",
                "family": "intraday_momentum",
                "signal_column": "target_ret_6",
                "mode": "signed",
                "horizons": [1],
                "threshold_quantiles": [0.5],
            }
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_forward_returns_use_next_open_exit_inside_session() -> None:
    frame = _sample_features().head(5)
    enriched = add_forward_returns(frame, {1, 2})

    assert "fwd_ret_1" in enriched.columns
    assert enriched.loc[0, "fwd_ret_1"] > 0
    assert pd.isna(enriched.loc[4, "fwd_ret_1"])


def test_monthly_folds_follow_policy() -> None:
    folds = build_monthly_folds(_sample_features(), {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1})

    assert len(folds) == 1
    assert folds[0].validation_months == ("2026-03",)
    assert folds[0].test_months == ("2026-04",)


def test_monthly_folds_apply_optional_session_embargo() -> None:
    folds = build_monthly_folds(
        _sample_features(),
        {"train_months": 2, "validation_months": 1, "test_months": 1, "step_months": 1, "embargo_sessions": 1},
    )

    assert folds[0].train_sessions == ("2026-01-02",)
    assert folds[0].validation_sessions == ("2026-03-02",)


def test_alpha_research_runner_writes_standard_artifacts(tmp_path: Path) -> None:
    plan = load_alpha_research_plan(_write_plan(tmp_path))
    summary = dry_run_summary(plan)
    artifacts = run_alpha_research(plan)

    assert summary["folds"] == 1
    assert artifacts.validation_path.exists()
    assert artifacts.test_path.exists()
    assert artifacts.decisions_path.exists()
    assert artifacts.manifest_path.exists()
    assert artifacts.report_path.exists()
    assert pd.read_parquet(artifacts.validation_path)["candidate_id"].nunique() == 1

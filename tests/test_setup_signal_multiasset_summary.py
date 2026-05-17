from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.setup_signal_multiasset_summary import family_summary, run


def test_family_summary_marks_stable_multiasset_family() -> None:
    best = pd.DataFrame(
        [
            {
                "target": "SPY",
                "family": "gap_fill",
                "direction": "short",
                "horizon_bars": 12,
                "decision": "accepted_candidate",
                "test_net_primary": 0.04,
                "test_avg_trade_net_primary": 0.0002,
                "test_net_stress": 0.01,
                "primary_positive": True,
                "stress_nonnegative": True,
                "research_or_better": True,
            },
            {
                "target": "QQQ",
                "family": "gap_fill",
                "direction": "short",
                "horizon_bars": 12,
                "decision": "research_candidate",
                "test_net_primary": 0.02,
                "test_avg_trade_net_primary": 0.0001,
                "test_net_stress": 0.0,
                "primary_positive": True,
                "stress_nonnegative": True,
                "research_or_better": True,
            },
        ]
    )
    config = {
        "setup_signal_multiasset": {
            "targets": ["SPY", "QQQ"],
            "min_targets": 2,
            "min_research_targets": 2,
            "min_primary_positive_targets": 2,
            "min_stress_nonnegative_targets": 2,
        }
    }

    summary = family_summary(best, config)

    assert bool(summary.iloc[0]["stable_family"]) is True
    assert summary.iloc[0]["research_or_better_targets"] == 2


def test_multiasset_summary_run_reads_target_decisions(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    for target, decision, stress in [("SPY", "accepted_candidate", 0.01), ("QQQ", "cost_fragile", 0.00)]:
        target_dir = results_dir / target
        target_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "candidate_id": f"{target}-1",
                    "fold": 0,
                    "family": "gap_fill",
                    "direction": "short",
                    "horizon_bars": 12,
                    "params_json": "{}",
                    "validation_status": "setup_validation_candidate",
                    "decision": decision,
                    "test_net_primary": 0.02,
                    "test_avg_trade_net_primary": 0.0001,
                    "test_net_stress": stress,
                }
            ]
        ).to_parquet(target_dir / "setup_signal_decisions.parquet", index=False)
    config = {
        "paths": {"results_dir": results_dir.as_posix(), "reports_dir": reports_dir.as_posix()},
        "setup_signal_multiasset": {
            "targets": ["SPY", "QQQ"],
            "results_dir": (results_dir / "_h9").as_posix(),
            "reports_dir": (reports_dir / "_h9").as_posix(),
            "min_targets": 2,
            "min_research_targets": 2,
            "min_primary_positive_targets": 2,
            "min_stress_nonnegative_targets": 2,
        },
    }
    config_path = tmp_path / "h9.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report_path, summary_path = run(config_path)

    summary = pd.read_parquet(summary_path)
    assert report_path.exists()
    assert summary.iloc[0]["family"] == "gap_fill"
    assert bool(summary.iloc[0]["stable_family"]) is True

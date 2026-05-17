from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research_app.registry import index_workspace, list_artifacts, list_candidates, list_reports, list_runs


def test_index_workspace_indexes_legacy_artifacts_idempotently(tmp_path: Path) -> None:
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    out_dir = results / "15min_divergence" / "SPY"
    report_dir = reports / "15min_divergence" / "SPY"
    out_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "decision": "keep",
                "validation_status": "research_candidate",
                "test_net_primary": 1.2,
            }
        ]
    ).to_parquet(out_dir / "cross_asset_divergence_decisions.parquet", index=False)
    (report_dir / "cross_asset_divergence_search.md").write_text("# Report\n", encoding="utf-8")
    db = tmp_path / "registry.sqlite"

    first = index_workspace(results_dir=results, reports_dir=reports, db_path=db, reset=True)
    second = index_workspace(results_dir=results, reports_dir=reports, db_path=db, reset=False)

    runs = list_runs(db)
    artifacts = list_artifacts(db)
    reports_frame = list_reports(db)
    candidates = list_candidates(db)

    assert first.runs == second.runs == 1
    assert len(runs) == 1
    assert len(artifacts) == 1
    assert len(reports_frame) == 1
    assert len(candidates) == 1
    assert candidates.iloc[0]["candidate_id"] == "C1"
    assert candidates.iloc[0]["target_symbol"] == "SPY"
    assert candidates.iloc[0]["timeframe"] == "15min"


def test_index_workspace_ignores_non_summary_candidate_files(tmp_path: Path) -> None:
    results = tmp_path / "results" / "SPY"
    reports = tmp_path / "reports"
    results.mkdir(parents=True)
    reports.mkdir()
    pd.DataFrame([{"candidate_id": "C1", "net_return": 1.0}]).to_parquet(results / "alpha_fold_bar_returns.parquet", index=False)
    db = tmp_path / "registry.sqlite"

    index_workspace(results_dir=tmp_path / "results", reports_dir=reports, db_path=db, reset=True)

    candidates = list_candidates(db)
    assert candidates.empty

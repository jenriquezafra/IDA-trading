from __future__ import annotations

from pathlib import Path

import pytest

from src.research_app.decisions import create_decision_log, list_decision_logs


def test_create_decision_log_requires_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence"):
        create_decision_log(db_path=tmp_path / "registry.sqlite", decision_type="reject", decision="reject", evidence=[])


def test_create_and_list_decision_log(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite"
    created = create_decision_log(
        db_path=db,
        decision_type="reject",
        decision="reject cost-fragile candidate",
        evidence=[{"path": "results/SPY/candidate_decisions.parquet", "candidate_id": "C1"}],
        candidate_id="C1",
        rationale="Fails conservative cost scenario.",
    )

    logs = list_decision_logs(db, candidate_id="C1")

    assert len(logs) == 1
    assert logs.iloc[0]["decision_id"] == created.decision_id
    assert logs.iloc[0]["candidate_id"] == "C1"

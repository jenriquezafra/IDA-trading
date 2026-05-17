from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_app.registry import connect, utc_now


@dataclass(frozen=True)
class DecisionLog:
    decision_id: str
    decision_type: str
    decision: str
    evidence: tuple[dict[str, Any], ...]
    run_id: str | None = None
    candidate_id: str | None = None
    rationale: str | None = None
    next_action: str | None = None
    human_owner: str | None = None


def create_decision_log(
    *,
    db_path: str | Path,
    decision_type: str,
    decision: str,
    evidence: list[dict[str, Any]],
    run_id: str | None = None,
    candidate_id: str | None = None,
    rationale: str | None = None,
    next_action: str | None = None,
    human_owner: str | None = None,
) -> DecisionLog:
    if not evidence:
        raise ValueError("decision logs require at least one evidence reference")
    decision_id = "DEC_" + uuid.uuid4().hex[:12]
    conn = connect(db_path)
    conn.execute(
        """
        insert into decision_logs(
          decision_id, created_at_utc, human_owner, decision_type, run_id,
          candidate_id, decision, rationale, evidence_json, next_action
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            utc_now(),
            human_owner,
            decision_type,
            run_id,
            candidate_id,
            decision,
            rationale,
            json.dumps(evidence, sort_keys=True),
            next_action,
        ),
    )
    conn.commit()
    conn.close()
    return DecisionLog(
        decision_id=decision_id,
        decision_type=decision_type,
        decision=decision,
        evidence=tuple(evidence),
        run_id=run_id,
        candidate_id=candidate_id,
        rationale=rationale,
        next_action=next_action,
        human_owner=human_owner,
    )


def list_decision_logs(db_path: str | Path, candidate_id: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    query = "select * from decision_logs"
    params: tuple[Any, ...] = ()
    if candidate_id:
        query += " where candidate_id = ?"
        params = (candidate_id,)
    query += " order by created_at_utc desc"
    frame = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return frame

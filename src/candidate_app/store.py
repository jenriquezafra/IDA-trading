from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from src.candidate_app.models import AuditLogEntry, CandidateStrategy, PaperLedgerEntry, utc_now, validate_status
from src.candidate_app.seed import SEED_CANDIDATES, SEED_PAPER_LEDGER_ENTRIES


DEFAULT_DB_PATH = Path("results/candidate_app/candidates.sqlite")
JSON_COLUMNS = (
    "asset_universe",
    "backtest_summary",
    "metrics",
    "paper_trading_metrics",
    "notes",
    "equity_curve",
    "drawdown_curve",
    "trades",
)


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists candidate_strategies(
          id text primary key,
          name text not null,
          strategy_type text not null,
          asset_universe_json text not null,
          status text not null,
          created_at text not null,
          promoted_at text,
          description text,
          backtest_summary_json text not null,
          metrics_json text not null,
          paper_trading_metrics_json text not null,
          notes_json text not null,
          equity_curve_json text not null,
          drawdown_curve_json text not null,
          trades_json text not null,
          updated_at text not null
        );

        create table if not exists candidate_audit_log(
          event_id text primary key,
          candidate_id text not null,
          changed_at text not null,
          actor text not null,
          from_status text,
          to_status text not null,
          reason text,
          note text
        );

        create table if not exists candidate_paper_ledger(
          entry_id text primary key,
          candidate_id text not null,
          event_at text not null,
          event_type text not null,
          strategy_run_id text,
          symbol text,
          side text,
          quantity real,
          price real,
          gross_pnl real,
          fees real,
          slippage_bps real,
          net_pnl real,
          exposure real,
          currency text not null,
          notes text,
          metadata_json text not null,
          created_at text not null
        );

        create table if not exists strategy_runtime_controls(
          candidate_id text not null,
          mode text not null,
          enabled integer not null,
          capital_mode text not null,
          capital_value real not null,
          capital_basis text not null,
          updated_at text not null,
          updated_by text not null,
          notes text,
          primary key(candidate_id, mode)
        );
        """
    )
    conn.commit()


def json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or f"candidate-{uuid.uuid4().hex[:8]}"


def row_to_candidate(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for column in JSON_COLUMNS:
        record[column] = json_load(record.pop(f"{column}_json"), [] if column in {"asset_universe", "notes", "equity_curve", "drawdown_curve", "trades"} else {})
    return record


def row_to_audit(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def row_to_ledger(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = json_load(record.pop("metadata_json"), {})
    return record


def audit_event(
    *,
    candidate_id: str,
    from_status: str | None,
    to_status: str,
    actor: str = "system",
    reason: str = "",
    note: str = "",
) -> AuditLogEntry:
    return AuditLogEntry(
        event_id=f"audit_{uuid.uuid4().hex[:16]}",
        candidate_id=candidate_id,
        changed_at=utc_now(),
        actor=actor or "system",
        from_status=from_status,
        to_status=validate_status(to_status),
        reason=reason or "",
        note=note or "",
    )


def insert_audit(conn: sqlite3.Connection, entry: AuditLogEntry) -> None:
    conn.execute(
        """
        insert into candidate_audit_log(
          event_id, candidate_id, changed_at, actor, from_status, to_status, reason, note
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.event_id,
            entry.candidate_id,
            entry.changed_at,
            entry.actor,
            entry.from_status,
            entry.to_status,
            entry.reason,
            entry.note,
        ),
    )


def candidate_exists(conn: sqlite3.Connection, candidate_id: str) -> bool:
    row = conn.execute("select 1 from candidate_strategies where id = ?", (candidate_id,)).fetchone()
    return row is not None


def save_candidate(
    conn: sqlite3.Connection,
    candidate: CandidateStrategy,
    *,
    actor: str = "system",
    reason: str = "created",
    audit: bool = True,
) -> dict[str, Any]:
    if candidate_exists(conn, candidate.id):
        raise ValueError(f"candidate already exists: {candidate.id}")

    record = candidate.to_record()
    now = utc_now()
    conn.execute(
        """
        insert into candidate_strategies(
          id, name, strategy_type, asset_universe_json, status, created_at, promoted_at,
          description, backtest_summary_json, metrics_json, paper_trading_metrics_json,
          notes_json, equity_curve_json, drawdown_curve_json, trades_json, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["id"],
            record["name"],
            record["strategy_type"],
            json_dump(record["asset_universe"]),
            record["status"],
            record["created_at"],
            record["promoted_at"],
            record["description"],
            json_dump(record["backtest_summary"]),
            json_dump(record["metrics"]),
            json_dump(record["paper_trading_metrics"]),
            json_dump(record["notes"]),
            json_dump(record["equity_curve"]),
            json_dump(record["drawdown_curve"]),
            json_dump(record["trades"]),
            now,
        ),
    )
    if audit:
        insert_audit(
            conn,
            audit_event(
                candidate_id=candidate.id,
                from_status=None,
                to_status=candidate.status,
                actor=actor,
                reason=reason,
            ),
        )
    conn.commit()
    return get_candidate(conn, candidate.id)


def get_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    row = conn.execute("select * from candidate_strategies where id = ?", (candidate_id,)).fetchone()
    if row is None:
        raise KeyError(candidate_id)
    return row_to_candidate(row)


def list_candidates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("select * from candidate_strategies order by created_at desc, name asc").fetchall()
    return [row_to_candidate(row) for row in rows]


def list_audit(conn: sqlite3.Connection, candidate_id: str | None = None) -> list[dict[str, Any]]:
    if candidate_id:
        rows = conn.execute(
            "select * from candidate_audit_log where candidate_id = ? order by changed_at desc, rowid desc",
            (candidate_id,),
        ).fetchall()
    else:
        rows = conn.execute("select * from candidate_audit_log order by changed_at desc, rowid desc").fetchall()
    return [row_to_audit(row) for row in rows]


def save_paper_ledger_entry(conn: sqlite3.Connection, entry: PaperLedgerEntry) -> dict[str, Any]:
    if not candidate_exists(conn, entry.candidate_id):
        raise ValueError(f"candidate does not exist: {entry.candidate_id}")
    record = entry.to_record()
    conn.execute(
        """
        insert into candidate_paper_ledger(
          entry_id, candidate_id, event_at, event_type, strategy_run_id, symbol, side,
          quantity, price, gross_pnl, fees, slippage_bps, net_pnl, exposure, currency,
          notes, metadata_json, created_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["entry_id"],
            record["candidate_id"],
            record["event_at"],
            record["event_type"],
            record["strategy_run_id"],
            record["symbol"],
            record["side"],
            record["quantity"],
            record["price"],
            record["gross_pnl"],
            record["fees"],
            record["slippage_bps"],
            record["net_pnl"],
            record["exposure"],
            record["currency"],
            record["notes"],
            json_dump(record["metadata"]),
            record["created_at"],
        ),
    )
    conn.commit()
    return get_paper_ledger_entry(conn, entry.entry_id)


def get_paper_ledger_entry(conn: sqlite3.Connection, entry_id: str) -> dict[str, Any]:
    row = conn.execute("select * from candidate_paper_ledger where entry_id = ?", (entry_id,)).fetchone()
    if row is None:
        raise KeyError(entry_id)
    return row_to_ledger(row)


def list_paper_ledger_entries(conn: sqlite3.Connection, candidate_id: str | None = None) -> list[dict[str, Any]]:
    if candidate_id:
        rows = conn.execute(
            """
            select * from candidate_paper_ledger
            where candidate_id = ?
            order by event_at asc, rowid asc
            """,
            (candidate_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "select * from candidate_paper_ledger order by event_at asc, rowid asc"
        ).fetchall()
    return [row_to_ledger(row) for row in rows]


def update_candidate_status(
    conn: sqlite3.Connection,
    candidate_id: str,
    status: str,
    *,
    actor: str = "dashboard",
    reason: str = "",
    note: str = "",
) -> dict[str, Any]:
    next_status = validate_status(status)
    current = get_candidate(conn, candidate_id)
    previous_status = current["status"]
    promoted_at = current.get("promoted_at")
    if next_status == "paper_trading" and not promoted_at:
        promoted_at = utc_now()
    conn.execute(
        """
        update candidate_strategies
        set status = ?, promoted_at = ?, updated_at = ?
        where id = ?
        """,
        (next_status, promoted_at, utc_now(), candidate_id),
    )
    insert_audit(
        conn,
        audit_event(
            candidate_id=candidate_id,
            from_status=previous_status,
            to_status=next_status,
            actor=actor,
            reason=reason,
            note=note,
        ),
    )
    conn.commit()
    return get_candidate(conn, candidate_id)


def count_candidates(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select count(*) from candidate_strategies").fetchone()[0])


def count_paper_ledger_entries(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select count(*) from candidate_paper_ledger").fetchone()[0])


def ensure_seed_data(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = connect(db_path)
    try:
        for payload in SEED_CANDIDATES:
            if not candidate_exists(conn, payload["id"]):
                try:
                    save_candidate(conn, CandidateStrategy(**payload), actor="seed", reason="seed data")
                except sqlite3.IntegrityError:
                    pass
        existing_ledger_ids = {
            row[0]
            for row in conn.execute("select entry_id from candidate_paper_ledger").fetchall()
        }
        for payload in SEED_PAPER_LEDGER_ENTRIES:
            if payload["entry_id"] not in existing_ledger_ids and candidate_exists(conn, payload["candidate_id"]):
                try:
                    save_paper_ledger_entry(conn, PaperLedgerEntry(**payload))
                except sqlite3.IntegrityError:
                    pass
    finally:
        conn.close()

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.research_app.manifest import file_fingerprint, get_git_branch, get_git_commit, get_git_dirty, load_manifest


DEFAULT_DB_PATH = Path("results/ida_registry.sqlite")
TARGET_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}$")
TIMEFRAME_RE = re.compile(r"^(\d+min)")


@dataclass(frozen=True)
class IndexSummary:
    db_path: Path
    runs: int
    artifacts: int
    reports: int
    candidates: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        create table if not exists runs(
          run_id text primary key,
          created_at_utc text,
          run_type text,
          status text,
          git_commit text,
          git_branch text,
          git_dirty integer,
          config_hash text,
          dataset_id text,
          dataset_hash text,
          instrument text,
          timeframe text,
          target_id text,
          feature_set_id text,
          split_policy_id text,
          cost_profile_id text,
          experiment_stage text,
          manifest_path text,
          source_kind text,
          artifact_count integer default 0,
          report_count integer default 0,
          updated_at_utc text,
          warning text
        );

        create table if not exists artifacts(
          artifact_id text primary key,
          run_id text not null,
          candidate_id text,
          artifact_type text,
          logical_name text,
          path text unique,
          hash text,
          schema_id text,
          rows integer,
          columns_json text,
          size_bytes integer,
          created_at_utc text,
          updated_at_utc text
        );

        create table if not exists reports(
          report_id text primary key,
          run_id text not null,
          candidate_id text,
          report_type text,
          path text unique,
          hash text,
          created_at_utc text,
          updated_at_utc text
        );

        create table if not exists candidates(
          candidate_key text primary key,
          candidate_id text not null,
          run_id text not null,
          candidate_family_id text,
          status text,
          decision text,
          validation_status text,
          source_path text,
          source_file text,
          target_symbol text,
          timeframe text,
          metrics_json text,
          created_at_utc text,
          updated_at_utc text
        );

        create table if not exists decision_logs(
          decision_id text primary key,
          created_at_utc text not null,
          human_owner text,
          decision_type text not null,
          run_id text,
          candidate_id text,
          decision text not null,
          rationale text,
          evidence_json text not null,
          next_action text
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        drop table if exists runs;
        drop table if exists artifacts;
        drop table if exists reports;
        drop table if exists candidates;
        drop table if exists decision_logs;
        """
    )
    init_db(conn)


def path_id(prefix: str, path: str | Path) -> str:
    import hashlib

    normalized = str(Path(path).as_posix())
    return f"{prefix}_{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]}"


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "UNKNOWN"


def infer_context(path: str | Path, root: str | Path) -> dict[str, str]:
    p = Path(path)
    root_path = Path(root)
    try:
        rel = p.relative_to(root_path)
    except ValueError:
        rel = p

    parts = rel.parts
    target = "UNKNOWN"
    experiment = "root"
    timeframe = "UNKNOWN"

    if not parts:
        return {"run_id": "LEGACY_UNKNOWN", "target": target, "timeframe": timeframe, "experiment": experiment}

    first = parts[0]
    if TARGET_RE.match(first) and (len(parts) == 1 or "." not in first):
        target = first
        experiment = "core"
        timeframe = "5min"
    elif len(parts) >= 2 and TARGET_RE.match(parts[1]):
        experiment = first
        target = parts[1]
        match = TIMEFRAME_RE.match(first)
        timeframe = match.group(1) if match else "UNKNOWN"
    else:
        match = TIMEFRAME_RE.match(first)
        if match:
            timeframe = match.group(1)
            experiment = first

    if timeframe == "UNKNOWN" and target != "UNKNOWN":
        timeframe = "5min"

    run_id = f"LEGACY_{safe_token(target)}_{safe_token(timeframe)}_{safe_token(experiment)}"
    return {"run_id": run_id, "target": target, "timeframe": timeframe, "experiment": experiment}


def upsert_run(conn: sqlite3.Connection, *, run_id: str, target: str, timeframe: str, run_type: str, source_kind: str, manifest_path: str | None = None, warning: str | None = None) -> None:
    now = utc_now()
    git_dirty = get_git_dirty()
    dirty_value = None if git_dirty is None else int(git_dirty)
    conn.execute(
        """
        insert into runs(
          run_id, created_at_utc, run_type, status, git_commit, git_branch, git_dirty,
          instrument, timeframe, target_id, manifest_path, source_kind, updated_at_utc, warning
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(run_id) do update set
          run_type=excluded.run_type,
          status=excluded.status,
          git_commit=excluded.git_commit,
          git_branch=excluded.git_branch,
          git_dirty=excluded.git_dirty,
          instrument=excluded.instrument,
          timeframe=excluded.timeframe,
          target_id=excluded.target_id,
          manifest_path=coalesce(excluded.manifest_path, runs.manifest_path),
          source_kind=excluded.source_kind,
          updated_at_utc=excluded.updated_at_utc,
          warning=excluded.warning
        """,
        (
            run_id,
            now,
            run_type,
            "indexed",
            get_git_commit(),
            get_git_branch(),
            dirty_value,
            target,
            timeframe,
            target,
            manifest_path,
            source_kind,
            now,
            warning,
        ),
    )


def parquet_metadata(path: Path) -> tuple[int | None, list[str]]:
    try:
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(path)
        return metadata.metadata.num_rows, list(metadata.schema.names)
    except Exception:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return None, []
        return len(frame), list(frame.columns)


def index_artifact(conn: sqlite3.Connection, path: Path, results_dir: Path) -> None:
    context = infer_context(path, results_dir)
    upsert_run(
        conn,
        run_id=context["run_id"],
        target=context["target"],
        timeframe=context["timeframe"],
        run_type=context["experiment"],
        source_kind="legacy_results",
        warning="legacy artifact without manifest",
    )
    rows, columns = parquet_metadata(path) if path.suffix == ".parquet" else (None, [])
    now = utc_now()
    conn.execute(
        """
        insert into artifacts(
          artifact_id, run_id, artifact_type, logical_name, path, hash, rows,
          columns_json, size_bytes, created_at_utc, updated_at_utc
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(path) do update set
          run_id=excluded.run_id,
          artifact_type=excluded.artifact_type,
          logical_name=excluded.logical_name,
          hash=excluded.hash,
          rows=excluded.rows,
          columns_json=excluded.columns_json,
          size_bytes=excluded.size_bytes,
          updated_at_utc=excluded.updated_at_utc
        """,
        (
            path_id("ART", path),
            context["run_id"],
            path.suffix.lstrip(".") or "file",
            path.stem,
            path.as_posix(),
            file_fingerprint(path),
            rows,
            json.dumps(columns),
            path.stat().st_size,
            now,
            now,
        ),
    )


def index_report(conn: sqlite3.Connection, path: Path, reports_dir: Path) -> None:
    context = infer_context(path, reports_dir)
    upsert_run(
        conn,
        run_id=context["run_id"],
        target=context["target"],
        timeframe=context["timeframe"],
        run_type=context["experiment"],
        source_kind="legacy_reports",
        warning="legacy report without manifest",
    )
    now = utc_now()
    conn.execute(
        """
        insert into reports(report_id, run_id, report_type, path, hash, created_at_utc, updated_at_utc)
        values (?, ?, ?, ?, ?, ?, ?)
        on conflict(path) do update set
          run_id=excluded.run_id,
          report_type=excluded.report_type,
          hash=excluded.hash,
          updated_at_utc=excluded.updated_at_utc
        """,
        (
            path_id("REP", path),
            context["run_id"],
            path.suffix.lstrip(".") or "file",
            path.as_posix(),
            file_fingerprint(path),
            now,
            now,
        ),
    )


def candidate_source(path: Path) -> bool:
    name = path.name.lower()
    if "bar_returns" in name or "posteriors" in name or "threshold_grid" in name:
        return False
    tokens = ("candidate_registry", "decisions", "triage", "selected_specs", "selected_validation", "focus_specs")
    return any(token in name for token in tokens)


def jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def index_candidates_from_parquet(conn: sqlite3.Connection, path: Path, results_dir: Path) -> int:
    if not candidate_source(path):
        return 0
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return 0
    if "candidate_id" not in frame.columns or frame.empty:
        return 0

    context = infer_context(path, results_dir)
    selected_cols = [
        "candidate_id",
        "family_id",
        "candidate_family_id",
        "family",
        "variant",
        "side",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "threshold",
        "candidate_status",
        "validation_status",
        "decision",
        "decision_label",
        "closed_status",
        "close_reason",
        "test_net_primary",
        "test_sharpe_primary",
        "test_profit_factor_primary",
        "test_avg_trade_net_primary",
        "test_trades_primary",
        "test_turnover_primary",
        "test_net_conservative",
        "test_net_stress",
        "net_return",
        "daily_sharpe",
        "profit_factor",
        "avg_trade_net",
        "trades",
        "max_drawdown",
        "cost_scenario",
        "configured_cost_bps",
        "effective_cost_bps",
        "primary_net_return",
        "primary_avg_trade_net",
        "primary_daily_sharpe",
        "primary_profit_factor",
        "primary_max_drawdown",
    ]
    now = utc_now()
    count = 0
    subset = frame.drop_duplicates(subset=["candidate_id"], keep="first")
    for _, row in subset.iterrows():
        candidate_id = str(row["candidate_id"])
        metrics = {col: jsonable(row[col]) for col in selected_cols if col in frame.columns}
        status = metrics.get("candidate_status") or metrics.get("closed_status")
        decision = metrics.get("decision") or metrics.get("decision_label")
        validation_status = metrics.get("validation_status")
        family_id = metrics.get("candidate_family_id") or metrics.get("family_id") or metrics.get("family")
        key = "|".join([context["run_id"], candidate_id, path.as_posix()])
        conn.execute(
            """
            insert into candidates(
              candidate_key, candidate_id, run_id, candidate_family_id, status, decision,
              validation_status, source_path, source_file, target_symbol, timeframe,
              metrics_json, created_at_utc, updated_at_utc
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(candidate_key) do update set
              candidate_family_id=excluded.candidate_family_id,
              status=excluded.status,
              decision=excluded.decision,
              validation_status=excluded.validation_status,
              target_symbol=excluded.target_symbol,
              timeframe=excluded.timeframe,
              metrics_json=excluded.metrics_json,
              updated_at_utc=excluded.updated_at_utc
            """,
            (
                key,
                candidate_id,
                context["run_id"],
                None if family_id is None else str(family_id),
                None if status is None else str(status),
                None if decision is None else str(decision),
                None if validation_status is None else str(validation_status),
                path.as_posix(),
                path.name,
                context["target"],
                context["timeframe"],
                json.dumps(metrics, default=str, sort_keys=True),
                now,
                now,
            ),
        )
        count += 1
    return count


def index_manifest(conn: sqlite3.Connection, path: Path) -> None:
    data = load_manifest(path)
    run = data.get("run", {})
    data_section = data.get("data", {})
    upsert_run(
        conn,
        run_id=str(run.get("run_id")),
        target=str(data_section.get("instrument") or data_section.get("target_symbol") or "UNKNOWN"),
        timeframe=str(data_section.get("timeframe") or "UNKNOWN"),
        run_type=str(run.get("run_type") or "manifest"),
        source_kind="manifest",
        manifest_path=path.as_posix(),
        warning=None,
    )


def update_counts(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        update runs set artifact_count = (
          select count(*) from artifacts where artifacts.run_id = runs.run_id
        )
        """
    )
    conn.execute(
        """
        update runs set report_count = (
          select count(*) from reports where reports.run_id = runs.run_id
        )
        """
    )


def index_workspace(
    *,
    results_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    db_path: str | Path = DEFAULT_DB_PATH,
    reset: bool = False,
) -> IndexSummary:
    results_path = Path(results_dir)
    reports_path = Path(reports_dir)
    conn = connect(db_path)
    if reset:
        reset_db(conn)

    for manifest in sorted(results_path.rglob("manifest.y*ml")) if results_path.exists() else []:
        try:
            index_manifest(conn, manifest)
        except Exception:
            continue

    artifact_count = 0
    candidate_count = 0
    if results_path.exists():
        for path in sorted(results_path.rglob("*.parquet")):
            index_artifact(conn, path, results_path)
            artifact_count += 1
            candidate_count += index_candidates_from_parquet(conn, path, results_path)

    report_count = 0
    if reports_path.exists():
        for path in sorted(reports_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".png", ".jpg", ".jpeg", ".parquet", ".yaml", ".yml"}:
                index_report(conn, path, reports_path)
                report_count += 1

    update_counts(conn)
    conn.commit()
    runs = conn.execute("select count(*) from runs").fetchone()[0]
    candidates = conn.execute("select count(*) from candidates").fetchone()[0]
    conn.close()
    return IndexSummary(Path(db_path), int(runs), artifact_count, report_count, int(candidates))


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def list_runs(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = connect(db_path)
    frame = pd.read_sql_query(
        """
        select run_id, instrument, timeframe, run_type, status, source_kind, artifact_count,
               report_count, git_commit, git_dirty, warning, updated_at_utc
        from runs
        order by instrument, timeframe, run_type, run_id
        """,
        conn,
    )
    conn.close()
    return frame


def list_artifacts(db_path: str | Path = DEFAULT_DB_PATH, run_id: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    query = "select * from artifacts"
    params: tuple[Any, ...] = ()
    if run_id:
        query += " where run_id = ?"
        params = (run_id,)
    query += " order by run_id, logical_name"
    frame = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return frame


def list_reports(db_path: str | Path = DEFAULT_DB_PATH, run_id: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    query = "select * from reports"
    params: tuple[Any, ...] = ()
    if run_id:
        query += " where run_id = ?"
        params = (run_id,)
    query += " order by run_id, path"
    frame = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return frame


def list_candidates(db_path: str | Path = DEFAULT_DB_PATH, run_id: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    query = "select * from candidates"
    params: tuple[Any, ...] = ()
    if run_id:
        query += " where run_id = ?"
        params = (run_id,)
    query += " order by target_symbol, timeframe, candidate_id, source_file"
    raw = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if raw.empty:
        return raw

    metric_rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        metrics = json.loads(row.get("metrics_json") or "{}")
        out = {key: row[key] for key in raw.columns if key != "metrics_json"}
        for key, value in metrics.items():
            if key not in out:
                out[key] = value
        metric_rows.append(out)
    return pd.DataFrame(metric_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index IDA Trading research artifacts into a local SQLite registry.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)
    summary = index_workspace(results_dir=args.results, reports_dir=args.reports, db_path=args.db, reset=args.reset)
    print(
        f"indexed runs={summary.runs} artifacts={summary.artifacts} "
        f"reports={summary.reports} candidates={summary.candidates} db={summary.db_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

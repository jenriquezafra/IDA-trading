from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.candidate_app.models import CandidateStrategy, PaperLedgerEntry, metric_value, normalize_metrics, utc_now
from src.candidate_app.store import (
    DEFAULT_DB_PATH,
    connect,
    ensure_seed_data,
    get_candidate as store_get_candidate,
    list_audit as store_list_audit,
    list_candidates as store_list_candidates,
    list_paper_ledger_entries as store_list_paper_ledger_entries,
    save_paper_ledger_entry,
    save_candidate,
    slugify,
    update_candidate_status,
)


def prepare_store(db_path: str | Path = DEFAULT_DB_PATH, *, seed: bool = True) -> None:
    if seed:
        ensure_seed_data(db_path)
        return
    conn = connect(db_path)
    conn.close()


def candidate_from_payload(payload: dict[str, Any]) -> CandidateStrategy:
    candidate_id = payload.get("id") or slugify(str(payload.get("name", "")))
    metrics = normalize_metrics(payload.get("metrics") or {})
    promoted_at = payload.get("promoted_at")
    status = payload.get("status", "candidate")
    if status == "paper_trading" and not promoted_at:
        promoted_at = utc_now()
    return CandidateStrategy(
        id=candidate_id,
        name=payload["name"],
        strategy_type=payload["strategy_type"],
        asset_universe=payload.get("asset_universe") or [],
        status=status,
        created_at=payload.get("created_at") or utc_now(),
        promoted_at=promoted_at,
        description=payload.get("description") or "",
        backtest_summary=payload.get("backtest_summary") or {},
        metrics=metrics,
        paper_trading_metrics=payload.get("paper_trading_metrics") or {},
        notes=payload.get("notes") or [],
        equity_curve=payload.get("equity_curve") or [],
        drawdown_curve=payload.get("drawdown_curve") or [],
        trades=payload.get("trades") or [],
    )


def create_candidate(db_path: str | Path, payload: dict[str, Any], *, actor: str = "dashboard") -> dict[str, Any]:
    conn = connect(db_path)
    try:
        return save_candidate(conn, candidate_from_payload(payload), actor=actor)
    finally:
        conn.close()


def get_candidate(db_path: str | Path, candidate_id: str) -> dict[str, Any]:
    conn = connect(db_path)
    try:
        candidate = store_get_candidate(conn, candidate_id)
        candidate["audit_log"] = store_list_audit(conn, candidate_id)
        candidate["paper_ledger_summary"] = summarize_paper_ledger(candidate, store_list_paper_ledger_entries(conn, candidate_id))
        return candidate
    finally:
        conn.close()


def change_candidate_status(
    db_path: str | Path,
    candidate_id: str,
    status: str,
    *,
    actor: str = "dashboard",
    reason: str = "",
    note: str = "",
) -> dict[str, Any]:
    conn = connect(db_path)
    try:
        candidate = update_candidate_status(conn, candidate_id, status, actor=actor, reason=reason, note=note)
        candidate["audit_log"] = store_list_audit(conn, candidate_id)
        return candidate
    finally:
        conn.close()


def list_candidate_records(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    status: str | None = None,
    strategy_type: str | None = None,
    asset: str | None = None,
    sharpe_min: float | None = None,
    sharpe_max: float | None = None,
    max_drawdown_min: float | None = None,
    max_drawdown_max: float | None = None,
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        candidates = store_list_candidates(conn)
    finally:
        conn.close()

    if status:
        candidates = [candidate for candidate in candidates if candidate["status"] == status]
    if strategy_type:
        candidates = [candidate for candidate in candidates if candidate["strategy_type"] == strategy_type]
    if asset:
        asset_upper = asset.upper()
        candidates = [candidate for candidate in candidates if asset_upper in candidate["asset_universe"]]
    if sharpe_min is not None:
        candidates = [candidate for candidate in candidates if float(metric_value(candidate, "sharpe", float("-inf")) or float("-inf")) >= sharpe_min]
    if sharpe_max is not None:
        candidates = [candidate for candidate in candidates if float(metric_value(candidate, "sharpe", float("inf")) or float("inf")) <= sharpe_max]
    if max_drawdown_min is not None:
        candidates = [
            candidate
            for candidate in candidates
            if float(metric_value(candidate, "max_drawdown", float("-inf")) or float("-inf")) >= max_drawdown_min
        ]
    if max_drawdown_max is not None:
        candidates = [
            candidate
            for candidate in candidates
            if float(metric_value(candidate, "max_drawdown", float("inf")) or float("inf")) <= max_drawdown_max
        ]
    return candidates


def paper_trading_candidates(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    return list_candidate_records(db_path, status="paper_trading")


def compare_candidates(db_path: str | Path, candidate_ids: list[str]) -> list[dict[str, Any]]:
    records = []
    for candidate_id in candidate_ids:
        candidate = get_candidate(db_path, candidate_id)
        metrics = candidate.get("metrics") or {}
        records.append(
            {
                "id": candidate["id"],
                "name": candidate["name"],
                "status": candidate["status"],
                "strategy_type": candidate["strategy_type"],
                "asset_universe": candidate["asset_universe"],
                "cagr": metrics.get("cagr"),
                "annualized_return": metrics.get("annualized_return"),
                "sharpe": metrics.get("sharpe"),
                "sortino": metrics.get("sortino"),
                "max_drawdown": metrics.get("max_drawdown"),
                "volatility": metrics.get("volatility"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "trade_count": metrics.get("trade_count"),
                "turnover": metrics.get("turnover"),
                "estimated_costs_bps": metrics.get("estimated_costs_bps"),
                "estimated_slippage_bps": metrics.get("estimated_slippage_bps"),
                "last_evaluated_at": metrics.get("last_evaluated_at"),
            }
        )
    return records


def metadata(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    records = list_candidate_records(db_path)
    statuses = sorted({record["status"] for record in records})
    strategy_types = sorted({record["strategy_type"] for record in records})
    assets = sorted({asset for record in records for asset in record["asset_universe"]})
    return {
        "statuses": statuses,
        "strategy_types": strategy_types,
        "assets": assets,
        "counts": {
            "total": len(records),
            "paper_trading": sum(1 for record in records if record["status"] == "paper_trading"),
            "candidate": sum(1 for record in records if record["status"] == "candidate"),
            "rejected": sum(1 for record in records if record["status"] == "rejected"),
            "archived": sum(1 for record in records if record["status"] == "archived"),
        },
    }


def ledger_entry_from_payload(payload: dict[str, Any]) -> PaperLedgerEntry:
    entry_id = payload.get("entry_id") or f"ledger_{uuid.uuid4().hex[:16]}"
    return PaperLedgerEntry(
        entry_id=entry_id,
        candidate_id=payload["candidate_id"],
        event_at=payload.get("event_at") or utc_now(),
        event_type=payload["event_type"],
        strategy_run_id=payload.get("strategy_run_id"),
        symbol=payload.get("symbol"),
        side=payload.get("side"),
        quantity=payload.get("quantity"),
        price=payload.get("price"),
        gross_pnl=payload.get("gross_pnl"),
        fees=payload.get("fees"),
        slippage_bps=payload.get("slippage_bps"),
        net_pnl=payload.get("net_pnl"),
        exposure=payload.get("exposure"),
        currency=payload.get("currency") or "USD",
        notes=payload.get("notes") or "",
        metadata=payload.get("metadata") or {},
        created_at=payload.get("created_at") or utc_now(),
    )


def create_paper_ledger_entry(db_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    conn = connect(db_path)
    try:
        return save_paper_ledger_entry(conn, ledger_entry_from_payload(payload))
    finally:
        conn.close()


def number_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cumulative_pnl_curve(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cumulative = 0.0
    curve: list[dict[str, Any]] = []
    for entry in entries:
        net_pnl = number_or_none(entry.get("net_pnl"))
        if net_pnl is None:
            continue
        cumulative += net_pnl
        curve.append(
            {
                "event_at": entry["event_at"],
                "entry_id": entry["entry_id"],
                "net_pnl": net_pnl,
                "cumulative_net_pnl": round(cumulative, 6),
            }
        )
    return curve


def pnl_drawdown(curve: list[dict[str, Any]]) -> float:
    high_water = 0.0
    max_drawdown = 0.0
    for point in curve:
        value = float(point["cumulative_net_pnl"])
        high_water = max(high_water, value)
        max_drawdown = min(max_drawdown, value - high_water)
    return round(max_drawdown, 6)


def summarize_paper_ledger(candidate: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_values = [value for value in (number_or_none(entry.get("net_pnl")) for entry in entries) if value is not None]
    gross_values = [value for value in (number_or_none(entry.get("gross_pnl")) for entry in entries) if value is not None]
    fees_values = [value for value in (number_or_none(entry.get("fees")) for entry in entries) if value is not None]
    slippage_values = [value for value in (number_or_none(entry.get("slippage_bps")) for entry in entries) if value is not None]
    exposure_values = [value for value in (number_or_none(entry.get("exposure")) for entry in entries) if value is not None]
    curve = cumulative_pnl_curve(entries)
    wins = sum(1 for value in pnl_values if value > 0)
    losses = sum(1 for value in pnl_values if value < 0)
    pnl_event_count = len(pnl_values)
    net_pnl = sum(pnl_values)
    return {
        "candidate_id": candidate["id"],
        "candidate_name": candidate["name"],
        "status": candidate["status"],
        "strategy_type": candidate["strategy_type"],
        "asset_universe": candidate["asset_universe"],
        "event_count": len(entries),
        "pnl_event_count": pnl_event_count,
        "net_pnl": round(net_pnl, 6),
        "gross_pnl": round(sum(gross_values), 6),
        "fees": round(sum(fees_values), 6),
        "win_rate": round(wins / pnl_event_count, 6) if pnl_event_count else None,
        "wins": wins,
        "losses": losses,
        "avg_net_pnl": round(net_pnl / pnl_event_count, 6) if pnl_event_count else None,
        "max_pnl_drawdown": pnl_drawdown(curve),
        "avg_slippage_bps": round(sum(slippage_values) / len(slippage_values), 6) if slippage_values else None,
        "latest_exposure": exposure_values[-1] if exposure_values else None,
        "last_event_at": entries[-1]["event_at"] if entries else None,
        "pnl_curve": curve,
    }


def list_paper_ledger(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    candidate_id: str | None = None,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        entries = store_list_paper_ledger_entries(conn, candidate_id)
        if not active_only:
            return entries
        active_ids = {candidate["id"] for candidate in store_list_candidates(conn) if candidate["status"] == "paper_trading"}
        return [entry for entry in entries if entry["candidate_id"] in active_ids]
    finally:
        conn.close()


def paper_ledger_summaries(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        candidates = store_list_candidates(conn)
        if active_only:
            candidates = [candidate for candidate in candidates if candidate["status"] == "paper_trading"]
        entries_by_candidate: dict[str, list[dict[str, Any]]] = {candidate["id"]: [] for candidate in candidates}
        for entry in store_list_paper_ledger_entries(conn):
            if entry["candidate_id"] in entries_by_candidate:
                entries_by_candidate[entry["candidate_id"]].append(entry)
        return [summarize_paper_ledger(candidate, entries_by_candidate[candidate["id"]]) for candidate in candidates]
    finally:
        conn.close()

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import subprocess
from typing import Any

import pandas as pd
import yaml

from src.candidate_app.models import json_safe, utc_now
from src.candidate_app.store import DEFAULT_DB_PATH, connect, list_paper_ledger_entries
from src.execution.operational_events import DEFAULT_OPERATIONAL_EVENTS_PATH, read_recent_operational_events


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get("TRADING_STRATS_RUNTIME_DIR", PROJECT_ROOT / "ops" / "runtime")).expanduser().resolve()
CONTROL_CENTER_UNITS = (
    "trading-strats-ibgateway.service",
    "trading-strats-vnc.service",
    "trading-strats-paper.service",
    "trading-strats-paper-c2.service",
    "trading-strats-control-center.service",
)
WATCHDOG_STATUS_PATH = Path("results/paper/ibkr_watchdog/status.yaml")
OPERATIONAL_EVENTS_PATH = DEFAULT_OPERATIONAL_EVENTS_PATH


@dataclass(frozen=True)
class PaperCandidateSource:
    candidate_id: str
    name: str
    strategy_id: str
    mode: str
    symbol: str
    account: str | None
    kill_switch_path: Path
    daemon_status_path: Path
    state_path: Path
    state_events_path: Path
    pnl_events_path: Path
    auto_runner_dir: Path
    config_path: Path | None = None
    connection_config_path: Path | None = None
    demo: bool = False


ACTIVE_PAPER_SOURCES: tuple[PaperCandidateSource, ...] = (
    PaperCandidateSource(
        candidate_id="qqq-risk-off-credit-spread",
        name="QQQ Risk-Off Credit Spread",
        strategy_id="qqq_15min_risk_off_short_h1c_v1",
        mode="paper",
        symbol="QQQ",
        account="DU9782002",
        kill_switch_path=RUNTIME_ROOT / "kill_switches/h1c_auto_paused",
        daemon_status_path=Path("results/paper/h1c_auto_runner/daemon_status.yaml"),
        state_path=Path("results/paper/h1c_state/state.yaml"),
        state_events_path=Path("results/paper/h1c_state/events.parquet"),
        pnl_events_path=Path("results/paper/h1c_state/pnl_events.parquet"),
        auto_runner_dir=Path("results/paper/h1c_auto_runner"),
        config_path=RUNTIME_ROOT / "h1c_auto_runner.paper.yaml",
        connection_config_path=Path("configs/execution/ibkr_paper_readonly.yaml"),
    ),
    PaperCandidateSource(
        candidate_id="c2-googl-opening-bias-followthrough",
        name="GOOGL Opening Bias Followthrough",
        strategy_id="c2_h9_googl_5min_opening_bias_followthrough_v1",
        mode="paper",
        symbol="GOOGL",
        account="DU9782002",
        kill_switch_path=RUNTIME_ROOT / "kill_switches/c2_auto_paused",
        daemon_status_path=Path("results/paper/c2_auto_runner/daemon_status.yaml"),
        state_path=Path("results/paper/c2_state/state.yaml"),
        state_events_path=Path("results/paper/c2_state/events.parquet"),
        pnl_events_path=Path("results/paper/c2_state/pnl_events.parquet"),
        auto_runner_dir=Path("results/paper/c2_auto_runner"),
        config_path=RUNTIME_ROOT / "c2_auto_runner.paper.yaml",
        connection_config_path=Path("configs/execution/ibkr_paper_readonly_c2.yaml"),
    ),
)

LIVE_SOURCES: tuple[PaperCandidateSource, ...] = ()


def resolve_workspace_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    root_path = Path(root).resolve()
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    allowed_roots = (root_path, RUNTIME_ROOT)
    if not any(_is_relative_to(resolved, allowed_root) for allowed_root in allowed_roots):
        raise ValueError(f"path is outside workspace/runtime roots: {path}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def workspace_relpath(path: str | Path, root: str | Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    for allowed_root in (Path(root).resolve(), RUNTIME_ROOT):
        try:
            return resolved.relative_to(allowed_root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"_read_error": str(exc)}
    return raw if isinstance(raw, dict) else {}


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def frame_records(frame: pd.DataFrame, *, limit: int | None = None, ascending: bool = False) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    output = frame.copy()
    sort_column = "created_at_utc" if "created_at_utc" in output.columns else None
    if sort_column:
        output = output.sort_values(sort_column, ascending=ascending, kind="stable")
    if limit is not None:
        output = output.head(limit)
    return json_safe(output.to_dict(orient="records"))


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except Exception:
        return None


def source_for(candidate_id: str) -> PaperCandidateSource:
    for source in (*ACTIVE_PAPER_SOURCES, *LIVE_SOURCES):
        if source.candidate_id == candidate_id:
            return source
    raise KeyError(candidate_id)


def sources_for_mode(mode: str) -> tuple[PaperCandidateSource, ...]:
    if mode == "paper":
        return ACTIVE_PAPER_SOURCES
    if mode == "live":
        return LIVE_SOURCES
    return ()


def read_runtime_control(
    source: PaperCandidateSource,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    kill_switch_exists: bool | None = None,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    conn = connect(db_path)
    try:
        row = conn.execute(
            """
            select * from strategy_runtime_controls
            where candidate_id = ? and mode = ?
            """,
            (source.candidate_id, source.mode),
        ).fetchone()
    finally:
        conn.close()
    if row is not None:
        record = dict(row)
        record["enabled"] = bool(record["enabled"])
        return json_safe(record)

    enabled = True if kill_switch_exists is None else not kill_switch_exists
    config = read_yaml(resolve_workspace_path(source.config_path, root)) if source.config_path else {}
    auto = dict(config.get("auto", {}) or {})
    sizing_mode = str(auto.get("sizing_mode") or "buying_power_fraction")
    if auto.get("max_order_notional_usd") not in {None, ""}:
        capital_mode = "absolute_usd"
        capital_value = float(auto.get("max_order_notional_usd") or 0.0)
        capital_basis = "max_order_notional_usd"
    elif sizing_mode in {"buying_power_fraction", "available_funds_fraction"}:
        capital_mode = "net_fraction"
        capital_value = float(auto.get("capital_fraction", 1.0) or 1.0)
        capital_basis = sizing_mode
    else:
        capital_mode = "net_fraction"
        capital_value = 1.0
        capital_basis = "ticket_quantity"
    return {
        "candidate_id": source.candidate_id,
        "mode": source.mode,
        "enabled": enabled,
        "capital_mode": capital_mode,
        "capital_value": capital_value,
        "capital_basis": capital_basis,
        "updated_at": None,
        "updated_by": "config",
        "notes": "",
    }


def save_runtime_control(
    source: PaperCandidateSource,
    *,
    enabled: bool,
    capital_mode: str,
    capital_value: float,
    capital_basis: str = "buying_power_fraction",
    actor: str = "dashboard",
    notes: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    if capital_mode not in {"net_fraction", "absolute_usd"}:
        raise ValueError(f"unsupported capital_mode: {capital_mode}")
    if capital_mode == "net_fraction" and not 0 < capital_value <= 1:
        raise ValueError("net_fraction capital_value must be in (0, 1]")
    if capital_mode == "absolute_usd" and capital_value <= 0:
        raise ValueError("absolute_usd capital_value must be positive")
    if capital_basis not in {"buying_power_fraction", "available_funds_fraction", "max_order_notional_usd"}:
        raise ValueError(f"unsupported capital_basis: {capital_basis}")

    conn = connect(db_path)
    try:
        conn.execute(
            """
            insert into strategy_runtime_controls(
              candidate_id, mode, enabled, capital_mode, capital_value,
              capital_basis, updated_at, updated_by, notes
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(candidate_id, mode) do update set
              enabled=excluded.enabled,
              capital_mode=excluded.capital_mode,
              capital_value=excluded.capital_value,
              capital_basis=excluded.capital_basis,
              updated_at=excluded.updated_at,
              updated_by=excluded.updated_by,
              notes=excluded.notes
            """,
            (
                source.candidate_id,
                source.mode,
                int(enabled),
                capital_mode,
                float(capital_value),
                capital_basis,
                utc_now(),
                actor,
                notes,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return read_runtime_control(source, db_path=db_path)


def list_manual_ledger(source: PaperCandidateSource, *, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        return list_paper_ledger_entries(conn, source.candidate_id)
    finally:
        conn.close()


def curve_from_manual_ledger(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cumulative = 0.0
    curve: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda row: str(row.get("event_at") or "")):
        raw = entry.get("net_pnl")
        if raw is None:
            continue
        value = float(raw)
        cumulative += value
        curve.append(
            {
                "timestamp": entry.get("event_at"),
                "event_type": entry.get("event_type"),
                "realized_pnl": value,
                "cumulative_realized_pnl": round(cumulative, 6),
                "quantity": entry.get("quantity"),
                "entry_price": entry.get("price"),
                "exit_price": None,
            }
        )
    return curve


def normalize_ledger_events(
    *,
    artifact_events: list[dict[str, Any]],
    manual_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in artifact_events:
        rows.append(
            {
                "event_at": event.get("created_at_utc") or event.get("event_at"),
                "source": "runner_pnl_events",
                "event_type": event.get("event_type"),
                "strategy_run_id": event.get("strategy_run_id"),
                "symbol": event.get("symbol"),
                "side": event.get("side"),
                "quantity": event.get("quantity"),
                "price": event.get("exit_price") or event.get("entry_price") or event.get("price"),
                "net_pnl": event.get("realized_pnl"),
                "gross_pnl": event.get("gross_pnl"),
                "fees": event.get("fees"),
                "slippage_bps": event.get("slippage_bps"),
                "exposure": event.get("exposure"),
                "notes": event.get("notes") or "",
            }
        )
    for entry in manual_entries:
        rows.append(
            {
                "event_at": entry.get("event_at"),
                "source": "manual_ledger",
                "event_type": entry.get("event_type"),
                "strategy_run_id": entry.get("strategy_run_id"),
                "symbol": entry.get("symbol"),
                "side": entry.get("side"),
                "quantity": entry.get("quantity"),
                "price": entry.get("price"),
                "net_pnl": entry.get("net_pnl"),
                "gross_pnl": entry.get("gross_pnl"),
                "fees": entry.get("fees"),
                "slippage_bps": entry.get("slippage_bps"),
                "exposure": entry.get("exposure"),
                "notes": entry.get("notes") or "",
            }
        )
    return json_safe(sorted(rows, key=lambda row: str(row.get("event_at") or ""), reverse=True))


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_side(*, position_side: Any = None, position_unit: Any = None, action: Any = None) -> str | None:
    side = str(position_side or "").strip().lower()
    if side in {"long", "short"}:
        return side
    unit = optional_float(position_unit)
    if unit is not None:
        if unit > 0:
            return "long"
        if unit < 0:
            return "short"
    action_text = str(action or "").strip().upper()
    if action_text == "BUY":
        return "long"
    if action_text == "SELL":
        return "short"
    return None


def current_position_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    open_position = dict(state.get("open_position") or {})
    pending_ticket = dict(state.get("pending_ticket") or {})
    position_side = infer_side(position_side=state.get("position_side"), position_unit=state.get("position_unit"), action=pending_ticket.get("action"))
    quantity = optional_float(state.get("quantity")) or 0.0
    position_unit = optional_float(state.get("position_unit")) or 0.0
    signed_quantity = quantity * (1.0 if position_unit > 0 else -1.0 if position_unit < 0 else 0.0)
    entry_price = optional_float(open_position.get("entry_price")) or optional_float(open_position.get("theoretical_entry_price"))
    current: dict[str, Any] = {
        "available": bool(state.get("available")),
        "status": state.get("status"),
        "symbol": state.get("symbol"),
        "account": state.get("account"),
        "side": position_side,
        "quantity": quantity,
        "signed_quantity": signed_quantity,
        "position_unit": position_unit,
        "desired_position_unit": optional_float(state.get("desired_position_unit")) or 0.0,
        "entry_price": entry_price,
        "opened_at_utc": open_position.get("opened_at_utc"),
        "signal_timestamp": open_position.get("signal_timestamp") or pending_ticket.get("signal_timestamp") or state.get("last_signal_timestamp"),
        "theoretical_entry_timestamp": open_position.get("theoretical_entry_timestamp") or pending_ticket.get("theoretical_entry_timestamp"),
        "theoretical_entry_price": optional_float(open_position.get("theoretical_entry_price") or pending_ticket.get("theoretical_entry_price")),
        "theoretical_exit_timestamp": open_position.get("theoretical_exit_timestamp") or pending_ticket.get("theoretical_exit_timestamp"),
        "theoretical_exit_price": optional_float(open_position.get("theoretical_exit_price") or pending_ticket.get("theoretical_exit_price")),
        "entry_slippage_bps": optional_float(open_position.get("entry_slippage_bps")),
        "exit_rule": open_position.get("exit_rule") or pending_ticket.get("exit_rule"),
        "horizon_bars": open_position.get("horizon_bars") or pending_ticket.get("horizon_bars"),
        "min_hold_bars": open_position.get("min_hold_bars") or pending_ticket.get("min_hold_bars"),
        "stop_loss_bps": optional_float(open_position.get("stop_loss_bps") or pending_ticket.get("stop_loss_bps")),
        "take_profit_bps": optional_float(open_position.get("take_profit_bps") or pending_ticket.get("take_profit_bps")),
        "pending_action": pending_ticket.get("action"),
        "pending_quantity": optional_float(pending_ticket.get("quantity")),
        "pending_status": pending_ticket.get("status"),
        "updated_at_utc": state.get("updated_at_utc"),
    }
    if entry_price is not None and quantity:
        current["notional"] = round(abs(entry_price * quantity), 6)
    else:
        current["notional"] = None
    return json_safe(current)


def position_timeline_from_state_events(events: list[dict[str, Any]], *, fallback_side: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: str(row.get("created_at_utc") or "")):
        event_type = str(event.get("event_type") or "")
        quantity = optional_float(event.get("quantity"))
        ticket_quantity = optional_float(event.get("ticket_quantity"))
        position_unit = optional_float(event.get("position_unit")) or 0.0
        if quantity is None:
            quantity = ticket_quantity if event_type in {"entry_fill_marked_open", "open_position_confirmed", "pending_exit_confirmed"} else 0.0
        side = infer_side(position_side=fallback_side, position_unit=position_unit, action=event.get("ticket_action"))
        signed_quantity = quantity * (1.0 if position_unit > 0 else -1.0 if position_unit < 0 else 0.0)
        is_relevant = event_type in {
            "pending_entry_created",
            "pending_exit_created",
            "entry_fill_marked_open",
            "exit_fill_marked_flat",
            "open_position_confirmed",
            "pending_exit_confirmed",
        }
        if not is_relevant:
            continue
        rows.append(
            {
                "event_at": event.get("created_at_utc"),
                "event_type": event_type,
                "status": event.get("new_status"),
                "previous_status": event.get("previous_status"),
                "action": event.get("ticket_action"),
                "side": side,
                "quantity": quantity,
                "signed_quantity": signed_quantity,
                "position_unit": position_unit,
                "source": "state_events",
                "state_updated": event.get("state_updated"),
            }
        )
    return json_safe(sorted(rows, key=lambda row: str(row.get("event_at") or ""), reverse=True))


def operation_history(
    *,
    state_events: list[dict[str, Any]],
    pnl_events: list[dict[str, Any]],
    fallback_side: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in state_events:
        event_type = str(event.get("event_type") or "")
        quantity = optional_float(event.get("quantity"))
        ticket_quantity = optional_float(event.get("ticket_quantity"))
        action = event.get("ticket_action")
        if quantity is None:
            quantity = ticket_quantity
        if event_type in {"flat_no_signal", "pending_entry_kept", "pending_exit_kept", "accounting_no_change"} and action in {None, "", "NONE"}:
            continue
        position_unit = optional_float(event.get("position_unit")) or 0.0
        rows.append(
            {
                "event_at": event.get("created_at_utc"),
                "source": "state_events",
                "event_type": event_type,
                "action": action,
                "status": event.get("new_status"),
                "previous_status": event.get("previous_status"),
                "side": infer_side(position_side=fallback_side, position_unit=position_unit, action=action),
                "quantity": quantity,
                "price": None,
                "position_after": quantity if event.get("new_status") in {"open", "pending_exit"} else 0.0,
                "signed_position_after": (quantity or 0.0) * (1.0 if position_unit > 0 else -1.0 if position_unit < 0 else 0.0),
                "realized_pnl": None,
                "slippage_bps": None,
            }
        )
    for event in pnl_events:
        event_type = str(event.get("event_type") or "")
        entry_price = optional_float(event.get("entry_price"))
        exit_price = optional_float(event.get("exit_price"))
        price = exit_price if event_type == "exit" else entry_price
        slippage = optional_float(event.get("exit_slippage_bps")) if event_type == "exit" else optional_float(event.get("entry_slippage_bps"))
        rows.append(
            {
                "event_at": event.get("created_at_utc"),
                "source": "pnl_events",
                "event_type": event_type,
                "action": "EXIT" if event_type == "exit" else "ENTRY" if event_type == "entry" else event_type.upper(),
                "status": "flat" if event_type == "exit" else "open" if event_type == "entry" else None,
                "previous_status": None,
                "side": fallback_side,
                "quantity": optional_float(event.get("quantity")),
                "price": price,
                "position_after": 0.0 if event_type == "exit" else optional_float(event.get("quantity")),
                "signed_position_after": None,
                "realized_pnl": optional_float(event.get("realized_pnl")),
                "slippage_bps": slippage,
            }
        )
    return json_safe(sorted(rows, key=lambda row: str(row.get("event_at") or ""), reverse=True))


def apply_capital_policy_to_config(
    source: PaperCandidateSource,
    control: dict[str, Any],
    *,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if not source.config_path:
        return {"applied": False, "reason": "source has no config_path"}
    config_path = resolve_workspace_path(source.config_path, root)
    config = read_yaml(config_path)
    if "_read_error" in config:
        return {"applied": False, "reason": config["_read_error"]}
    auto = dict(config.get("auto", {}) or {})
    if control["capital_mode"] == "net_fraction":
        auto["sizing_mode"] = control.get("capital_basis") if control.get("capital_basis") in {"buying_power_fraction", "available_funds_fraction"} else "buying_power_fraction"
        auto["capital_fraction"] = float(control["capital_value"])
        auto["max_order_notional_usd"] = None
    else:
        auto["max_order_notional_usd"] = float(control["capital_value"])
    config["auto"] = auto
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"applied": True, "config_path": workspace_relpath(config_path, root)}


def manifest_summary(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    run = dict(manifest.get("run", {}) or {})
    config = dict(manifest.get("config", {}) or {})
    ticket = dict(manifest.get("signal", {}).get("ticket", {}) or {})
    raw_ticket = dict(manifest.get("signal", {}).get("raw_ticket", {}) or {})
    active_ticket = ticket or raw_ticket
    pre_recon = dict(manifest.get("pre_trade_reconciliation", {}) or {})
    post_recon = dict(manifest.get("post_execution_reconciliation", {}) or {})
    plan = dict(manifest.get("order_plan", {}).get("summary", {}) or {})
    execution = dict(manifest.get("execution", {}).get("summary", {}) or {})
    state = dict(manifest.get("state", {}).get("state", {}) or {})
    return json_safe(
        {
            "created_at_utc": run.get("created_at_utc"),
            "status": run.get("status"),
            "strategy_id": config.get("strategy_id") or active_ticket.get("strategy_id"),
            "decision": manifest.get("decision"),
            "reason": manifest.get("reason"),
            "market_open": manifest.get("market", {}).get("open"),
            "signal_timestamp": active_ticket.get("signal_timestamp"),
            "signal_action": active_ticket.get("action"),
            "ticket_quantity": active_ticket.get("quantity"),
            "ticket_status": active_ticket.get("status"),
            "pre_trade_reconciliation": pre_recon.get("decision"),
            "pre_trade_severity": pre_recon.get("severity"),
            "post_execution_reconciliation": post_recon.get("decision"),
            "post_execution_severity": post_recon.get("severity"),
            "funds_ok": manifest.get("funds", {}).get("ok"),
            "entry_safety_ok": manifest.get("entry_safety", {}).get("ok"),
            "order_plan_decision": plan.get("decision"),
            "planned_orders": plan.get("planned_orders", 0),
            "submitted_orders": execution.get("submitted_orders", 0),
            "state_status": state.get("status"),
            "latency_seconds": manifest.get("latency", {}).get("total_seconds"),
            "run_dir": workspace_relpath(manifest_path.parent, root),
            "manifest_path": workspace_relpath(manifest_path, root),
        }
    )


def list_auto_runs(source: PaperCandidateSource, root: str | Path = PROJECT_ROOT, *, limit: int = 80) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    runner_dir = resolve_workspace_path(source.auto_runner_dir, root_path)
    if not runner_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for manifest_path in runner_dir.glob("*/manifest.yaml"):
        row = manifest_summary(manifest_path, root_path)
        if row.get("strategy_id") == source.strategy_id:
            rows.append(row)
    rows = sorted(rows, key=lambda row: str(row.get("created_at_utc") or ""), reverse=True)
    return rows[:limit]


def demo_ko_daemon() -> dict[str, Any]:
    return {
        "available": True,
        "status": "running",
        "error_streak": 0,
        "scheduler": {
            "market_open": False,
            "reason": "waiting for next US cash session",
            "next_open_utc": "2026-05-18T13:30:00Z",
        },
        "mtime_utc": "2026-05-17T10:00:00Z",
        "path": "demo://ko/daemon",
    }


def demo_ko_state() -> dict[str, Any]:
    return {
        "available": True,
        "strategy_id": "ko_daily_defensive_reversion_demo_v1",
        "status": "long",
        "symbol": "KO",
        "quantity": 90,
        "desired_position_unit": 1.0,
        "position_unit": 1.0,
        "avg_entry_price": 62.4,
        "path": "demo://ko/state",
    }


def demo_ko_runs() -> list[dict[str, Any]]:
    rows = [
        ("2026-05-15T20:05:00Z", "hold", "price above stop, signal remains constructive", "hold", 90, 0, 0, "ok"),
        ("2026-05-08T20:05:00Z", "accepted", "pullback into defensive support band", "buy", 90, 1, 1, "ok"),
        ("2026-04-24T20:05:00Z", "accepted", "mean reversion target reached", "sell", 80, 1, 1, "ok"),
        ("2026-04-03T20:05:00Z", "hold", "dividend defensive basket still in regime", "hold", 80, 0, 0, "ok"),
        ("2026-03-06T20:05:00Z", "accepted", "oversold staple rotation entry", "buy", 80, 1, 1, "ok"),
    ]
    return [
        {
            "created_at_utc": created_at,
            "status": "ok",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "decision": decision,
            "reason": reason,
            "market_open": False,
            "signal_timestamp": created_at,
            "signal_action": action,
            "ticket_quantity": quantity,
            "ticket_status": "demo",
            "pre_trade_reconciliation": recon,
            "pre_trade_severity": "info",
            "post_execution_reconciliation": recon,
            "post_execution_severity": "info",
            "funds_ok": True,
            "entry_safety_ok": True,
            "order_plan_decision": "submit" if submitted_orders else "noop",
            "planned_orders": planned_orders,
            "submitted_orders": submitted_orders,
            "state_status": "long",
            "latency_seconds": 1.2,
            "run_dir": "demo://ko/runs",
            "manifest_path": "demo://ko/manifest",
        }
        for created_at, decision, reason, action, quantity, planned_orders, submitted_orders, recon in rows
    ]


def demo_ko_state_events() -> list[dict[str, Any]]:
    return [
        {
            "created_at_utc": "2026-05-15T20:05:00Z",
            "event_type": "hold",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "account": "DEMO-PAPER",
            "symbol": "KO",
            "previous_status": "long",
            "new_status": "long",
            "ticket_action": "hold",
            "ticket_quantity": 90,
            "desired_position_unit": 1.0,
            "position_unit": 1.0,
            "state_updated": False,
        },
        {
            "created_at_utc": "2026-05-08T20:05:00Z",
            "event_type": "fill",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "account": "DEMO-PAPER",
            "symbol": "KO",
            "previous_status": "flat",
            "new_status": "long",
            "ticket_action": "buy",
            "ticket_quantity": 90,
            "desired_position_unit": 1.0,
            "position_unit": 1.0,
            "state_updated": True,
        },
        {
            "created_at_utc": "2026-04-24T20:05:00Z",
            "event_type": "fill",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "account": "DEMO-PAPER",
            "symbol": "KO",
            "previous_status": "long",
            "new_status": "flat",
            "ticket_action": "sell",
            "ticket_quantity": 80,
            "desired_position_unit": 0.0,
            "position_unit": 0.0,
            "state_updated": True,
        },
        {
            "created_at_utc": "2026-04-03T20:05:00Z",
            "event_type": "hold",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "account": "DEMO-PAPER",
            "symbol": "KO",
            "previous_status": "long",
            "new_status": "long",
            "ticket_action": "hold",
            "ticket_quantity": 80,
            "desired_position_unit": 1.0,
            "position_unit": 1.0,
            "state_updated": False,
        },
        {
            "created_at_utc": "2026-03-06T20:05:00Z",
            "event_type": "fill",
            "strategy_id": "ko_daily_defensive_reversion_demo_v1",
            "account": "DEMO-PAPER",
            "symbol": "KO",
            "previous_status": "flat",
            "new_status": "long",
            "ticket_action": "buy",
            "ticket_quantity": 80,
            "desired_position_unit": 1.0,
            "position_unit": 1.0,
            "state_updated": True,
        },
    ]


def demo_ko_market_series() -> list[dict[str, Any]]:
    prices = [
        ("2026-03-02", 60.20),
        ("2026-03-06", 59.40),
        ("2026-03-13", 60.10),
        ("2026-03-20", 61.35),
        ("2026-03-27", 62.05),
        ("2026-04-03", 62.50),
        ("2026-04-10", 63.15),
        ("2026-04-17", 64.05),
        ("2026-04-24", 65.10),
        ("2026-05-01", 63.70),
        ("2026-05-08", 62.40),
        ("2026-05-15", 64.20),
    ]
    actions = {
        "2026-03-06": {"action": "buy", "quantity": 80, "label": "BUY 80"},
        "2026-04-03": {"action": "hold", "quantity": 80, "label": "HOLD"},
        "2026-04-24": {"action": "sell", "quantity": 80, "label": "SELL 80"},
        "2026-05-08": {"action": "buy", "quantity": 90, "label": "BUY 90"},
        "2026-05-15": {"action": "hold", "quantity": 90, "label": "HOLD"},
    }
    return [
        {
            "timestamp": f"{date}T20:00:00Z",
            "date": date,
            "close": close,
            "marker": actions.get(date),
        }
        for date, close in prices
    ]


def market_snapshot(source: PaperCandidateSource, ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    if source.demo and source.symbol == "KO":
        return {
            "symbol": "KO",
            "source": "demo_price_series",
            "series": demo_ko_market_series(),
        }
    return {
        "symbol": source.symbol,
        "source": "not_configured",
        "series": [],
    }


def pnl_curve_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cumulative = 0.0
    curve: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: str(row.get("created_at_utc") or "")):
        raw = event.get("realized_pnl")
        if raw is None:
            continue
        value = float(raw)
        cumulative += value
        curve.append(
            {
                "timestamp": event.get("created_at_utc"),
                "event_type": event.get("event_type"),
                "realized_pnl": value,
                "cumulative_realized_pnl": round(cumulative, 6),
                "quantity": event.get("quantity"),
                "entry_price": event.get("entry_price"),
                "exit_price": event.get("exit_price"),
            }
        )
    return curve


def pnl_metrics(curve: list[dict[str, Any]]) -> dict[str, Any]:
    if not curve:
        return {
            "realized_pnl": 0.0,
            "event_count": 0,
            "winning_events": 0,
            "losing_events": 0,
            "win_rate": None,
            "max_drawdown": 0.0,
        }
    values = [float(row["realized_pnl"]) for row in curve]
    high = 0.0
    max_drawdown = 0.0
    for row in curve:
        value = float(row["cumulative_realized_pnl"])
        high = max(high, value)
        max_drawdown = min(max_drawdown, value - high)
    wins = sum(1 for value in values if value > 0)
    losses = sum(1 for value in values if value < 0)
    return {
        "realized_pnl": round(float(curve[-1]["cumulative_realized_pnl"]), 6),
        "event_count": len(values),
        "winning_events": wins,
        "losing_events": losses,
        "win_rate": round(wins / len(values), 6) if values else None,
        "max_drawdown": round(max_drawdown, 6),
    }


def build_alerts(
    *,
    source: PaperCandidateSource,
    kill_switch_exists: bool,
    daemon: dict[str, Any],
    state: dict[str, Any],
    latest_run: dict[str, Any] | None,
    pnl_log_available: bool,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if kill_switch_exists:
        alerts.append(
            {
                "severity": "critical",
                "code": "KILL_SWITCH_ACTIVE",
                "title": "Automation paused",
                "message": f"Kill switch exists at {source.kill_switch_path.as_posix()}.",
            }
        )
    if not daemon.get("available"):
        alerts.append(
            {
                "severity": "warning",
                "code": "DAEMON_STATUS_MISSING",
                "title": "Daemon status unavailable",
                "message": "No daemon status file was found for this candidate.",
            }
        )
    elif int(daemon.get("error_streak") or 0) > 0:
        alerts.append(
            {
                "severity": "critical",
                "code": "DAEMON_ERRORS",
                "title": "Daemon error streak",
                "message": f"Daemon error streak is {daemon.get('error_streak')}.",
            }
        )
    if latest_run:
        decision = str(latest_run.get("decision") or "")
        if decision.startswith("blocked"):
            alerts.append(
                {
                    "severity": "critical",
                    "code": "RUN_BLOCKED",
                    "title": "Latest run blocked",
                    "message": latest_run.get("reason") or decision,
                }
            )
        if latest_run.get("pre_trade_severity") == "block" or latest_run.get("post_execution_severity") == "block":
            alerts.append(
                {
                    "severity": "critical",
                    "code": "RECONCILIATION_BLOCK",
                    "title": "Reconciliation block",
                    "message": "The latest run reported a reconciliation block.",
                }
            )
    if state.get("available") and state.get("status") not in {None, "flat"}:
        alerts.append(
            {
                "severity": "warning",
                "code": "OPEN_STATE",
                "title": "State is not flat",
                "message": f"Current state is {state.get('status')} with quantity {state.get('quantity')}.",
            }
        )
    if not pnl_log_available:
        alerts.append(
            {
                "severity": "info",
                "code": "PNL_LOG_MISSING",
                "title": "No paper PnL log yet",
                "message": "No realized PnL parquet exists for this candidate yet.",
            }
        )
    if not alerts:
        alerts.append(
            {
                "severity": "ok",
                "code": "NO_ACTIVE_ALERTS",
                "title": "No active alerts",
                "message": "No operational alerts are active for this candidate.",
            }
        )
    return alerts


def overall_state(alerts: list[dict[str, Any]], daemon: dict[str, Any]) -> str:
    severities = {alert["severity"] for alert in alerts}
    if "critical" in severities:
        return "blocked"
    if "warning" in severities:
        return "attention"
    scheduler = dict(daemon.get("scheduler", {}) or {})
    if scheduler.get("market_open") is False:
        return "waiting"
    return "running"


def tcp_check(host: str, port: int, timeout_seconds: float = 1.0) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds):
            latency_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
            return {"ok": True, "latency_ms": round(latency_ms, 2), "error": None}
    except Exception as exc:
        latency_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        return {"ok": False, "latency_ms": round(latency_ms, 2), "error": str(exc)}


def run_command(args: list[str], timeout_seconds: float = 2.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "returncode": 127}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "stdout": exc.stdout or "", "stderr": "command timed out", "returncode": None}
    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def systemd_unit_snapshot(unit: str) -> dict[str, Any]:
    active = run_command(["systemctl", "is-active", unit])
    enabled = run_command(["systemctl", "is-enabled", unit])
    return {
        "unit": unit,
        "active_state": active["stdout"] or "unknown",
        "enabled_state": enabled["stdout"] or "unknown",
        "ok": active["stdout"] == "active",
        "error": active["stderr"] or None,
    }


def uptime_seconds() -> float | None:
    path = Path("/proc/uptime")
    if not path.exists():
        return None
    try:
        return round(float(path.read_text(encoding="utf-8").split()[0]), 3)
    except Exception:
        return None


def ibkr_gateway_snapshot(
    *,
    host: str = "127.0.0.1",
    port: int = 4002,
    expected_account: str = "DU9782002",
) -> dict[str, Any]:
    port_result = tcp_check(host, port, timeout_seconds=1.0)
    snapshot: dict[str, Any] = {
        "host": host,
        "port": port,
        "port_open": bool(port_result.get("ok")),
        "latency_ms": port_result.get("latency_ms"),
        "expected_account": expected_account,
        "managed_accounts": [],
        "account_ok": False,
        "server_time": None,
        "ok": False,
        "error": port_result.get("error"),
    }
    if not port_result.get("ok"):
        return snapshot
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from ib_insync import IB

        ib = IB()
        try:
            ib.connect(host, port, clientId=91, timeout=3)
            accounts = list(ib.managedAccounts())
            server_time = ib.reqCurrentTime()
        finally:
            if ib.isConnected():
                ib.disconnect()
            asyncio.set_event_loop(None)
            loop.close()
        snapshot.update(
            {
                "managed_accounts": accounts,
                "account_ok": expected_account in accounts,
                "server_time": str(server_time),
                "ok": expected_account in accounts,
                "error": None if expected_account in accounts else f"expected account {expected_account} not in managed accounts",
            }
        )
    except Exception as exc:  # noqa: BLE001
        snapshot.update({"ok": False, "error": repr(exc)})
        asyncio.set_event_loop(None)
        loop.close()
    return snapshot


def daemon_file_health(source: PaperCandidateSource, root_path: Path) -> dict[str, Any]:
    if source.demo:
        return {"available": True, "ok": True, "mtime_utc": "2026-05-17T10:00:00Z", "age_seconds": None}
    path = resolve_workspace_path(source.daemon_status_path, root_path)
    if not path.exists():
        return {"available": False, "ok": False, "path": workspace_relpath(path, root_path), "error": "missing daemon status file"}
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_seconds = max(0.0, (datetime.now(timezone.utc) - mtime).total_seconds())
    payload = read_yaml(path)
    error_streak = int(payload.get("error_streak") or 0) if "_read_error" not in payload else 1
    return {
        "available": "_read_error" not in payload,
        "ok": "_read_error" not in payload and error_streak == 0 and age_seconds < 1800,
        "path": workspace_relpath(path, root_path),
        "mtime_utc": mtime.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "age_seconds": round(age_seconds, 3),
        "error_streak": error_streak,
        "scheduler_reason": dict(payload.get("scheduler", {}) or {}).get("reason"),
        "last_decision": dict(payload.get("runner_summary", {}) or {}).get("decision"),
        "error": payload.get("_read_error") or payload.get("error"),
    }


def watchdog_file_health(root_path: Path, *, max_age_seconds: int = 180) -> dict[str, Any]:
    path = resolve_workspace_path(WATCHDOG_STATUS_PATH, root_path)
    if not path.exists():
        return {
            "available": False,
            "ok": False,
            "path": workspace_relpath(path, root_path),
            "error": "missing watchdog status file",
        }
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_seconds = max(0.0, (datetime.now(timezone.utc) - mtime).total_seconds())
    payload = read_yaml(path)
    checks = list(payload.get("checks", []) or []) if "_read_error" not in payload else []
    failed = [str(check.get("name") or check.get("config_path") or "target") for check in checks if not check.get("ok")]
    ok = "_read_error" not in payload and bool(payload.get("ok")) and age_seconds <= max_age_seconds
    return {
        "available": "_read_error" not in payload,
        "ok": ok,
        "path": workspace_relpath(path, root_path),
        "mtime_utc": mtime.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "age_seconds": round(age_seconds, 3),
        "status": payload.get("status"),
        "action": payload.get("action"),
        "message": payload.get("_read_error") or payload.get("message"),
        "failed_targets": failed,
        "check_count": len(checks),
        "error": payload.get("_read_error") or (", ".join(failed) if failed else None),
    }


def operational_events_health(root_path: Path, *, limit: int = 12) -> dict[str, Any]:
    path = resolve_workspace_path(OPERATIONAL_EVENTS_PATH, root_path)
    if not path.exists():
        return {
            "available": True,
            "ok": True,
            "path": workspace_relpath(path, root_path),
            "recent": [],
            "latest": None,
            "count_returned": 0,
        }
    try:
        recent = read_recent_operational_events(path, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "ok": False,
            "path": workspace_relpath(path, root_path),
            "recent": [],
            "latest": None,
            "count_returned": 0,
            "error": repr(exc),
        }
    return {
        "available": True,
        "ok": True,
        "path": workspace_relpath(path, root_path),
        "recent": json_safe(recent),
        "latest": json_safe(recent[0]) if recent else None,
        "count_returned": len(recent),
    }


def operational_snapshot(*, root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    primary = ACTIVE_PAPER_SOURCES[0]
    systemd = [systemd_unit_snapshot(unit) for unit in CONTROL_CENTER_UNITS]
    daemon = daemon_file_health(primary, root_path)
    watchdog = watchdog_file_health(root_path)
    events = operational_events_health(root_path)
    ibkr = ibkr_gateway_snapshot(expected_account=primary.account or "")
    runtime_config = read_yaml(resolve_workspace_path(primary.config_path, root_path)) if primary.config_path else {}
    optional_units = {"trading-strats-control-center.service"}
    service_ok = all(unit.get("ok") for unit in systemd if unit["unit"] not in optional_units)
    ok = bool(service_ok and ibkr.get("ok") and daemon.get("ok") and watchdog.get("ok") and "_read_error" not in runtime_config)
    return {
        "status": "ok" if ok else "degraded",
        "ok": ok,
        "host": {
            "hostname": socket.gethostname(),
            "uptime_seconds": uptime_seconds(),
            "runtime_root": RUNTIME_ROOT.as_posix(),
            "project_root": root_path.as_posix(),
        },
        "systemd": systemd,
        "ibkr": ibkr,
        "daemon": daemon,
        "watchdog": watchdog,
        "events": events,
        "runtime": {
            "config_path": workspace_relpath(primary.config_path, root_path) if primary.config_path else None,
            "kill_switch_path": workspace_relpath(primary.kill_switch_path, root_path),
            "config_loaded": "_read_error" not in runtime_config,
            "config_error": runtime_config.get("_read_error"),
            "enabled": dict(runtime_config.get("auto", {}) or {}).get("enabled"),
            "capital_fraction": dict(runtime_config.get("auto", {}) or {}).get("capital_fraction"),
            "max_order_notional_usd": dict(runtime_config.get("auto", {}) or {}).get("max_order_notional_usd"),
            "execute_orders": dict(runtime_config.get("auto", {}) or {}).get("execute_orders"),
            "transmit_orders": dict(runtime_config.get("auto", {}) or {}).get("transmit_orders"),
        },
        "updated_at_utc": utc_now(),
    }


def connection_snapshot(*, root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    checks: list[dict[str, Any]] = []
    for source in (*ACTIVE_PAPER_SOURCES, *LIVE_SOURCES):
        if not source.connection_config_path:
            continue
        config_path = resolve_workspace_path(source.connection_config_path, root_path)
        config = read_yaml(config_path)
        connection = dict(config.get("connection", {}) or {})
        host = str(connection.get("host") or "127.0.0.1")
        port = int(connection.get("port") or 0)
        timeout = min(float(connection.get("timeout_seconds") or 2.0), 2.0)
        result = tcp_check(host, port, timeout_seconds=timeout) if port else {"ok": False, "latency_ms": None, "error": "missing port"}
        checks.append(
            {
                "id": f"{source.candidate_id}:ibkr",
                "candidate_id": source.candidate_id,
                "mode": source.mode,
                "name": "IBKR paper gateway",
                "host": host,
                "port": port,
                "trading_mode": connection.get("trading_mode"),
                "expected_account": connection.get("expected_account"),
                "config_path": workspace_relpath(config_path, root_path),
                **result,
            }
        )

    daemon_checks = []
    for source in (*ACTIVE_PAPER_SOURCES, *LIVE_SOURCES):
        if source.demo:
            daemon_checks.append(
                {
                    "id": f"{source.candidate_id}:daemon_file",
                    "candidate_id": source.candidate_id,
                    "mode": source.mode,
                    "name": "Demo daemon status",
                    "ok": True,
                    "path": "demo://ko/daemon",
                    "mtime_utc": "2026-05-17T10:00:00Z",
                    "error": None,
                }
            )
            continue
        daemon_path = resolve_workspace_path(source.daemon_status_path, root_path)
        available = daemon_path.exists()
        mtime = (
            datetime.fromtimestamp(daemon_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if available
            else None
        )
        daemon_checks.append(
            {
                "id": f"{source.candidate_id}:daemon_file",
                "candidate_id": source.candidate_id,
                "mode": source.mode,
                "name": "Daemon status file",
                "ok": available,
                "path": source.daemon_status_path.as_posix(),
                "mtime_utc": mtime,
                "error": None if available else "missing daemon status file",
            }
        )

    all_checks = checks + daemon_checks
    ok_count = sum(1 for check in all_checks if check.get("ok"))
    return {
        "status": "ok" if ok_count == len(all_checks) and all_checks else "degraded",
        "ok_count": ok_count,
        "check_count": len(all_checks),
        "checks": all_checks,
        "operations": operational_snapshot(root=root_path),
        "updated_at_utc": utc_now(),
    }


def candidate_control_snapshot(
    candidate_id: str,
    *,
    root: str | Path = PROJECT_ROOT,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    source = source_for(candidate_id)
    root_path = Path(root).resolve()
    kill_switch_path = resolve_workspace_path(source.kill_switch_path, root_path)
    daemon_path = resolve_workspace_path(source.daemon_status_path, root_path)
    state_path = resolve_workspace_path(source.state_path, root_path)
    state_events_path = resolve_workspace_path(source.state_events_path, root_path)
    pnl_events_path = resolve_workspace_path(source.pnl_events_path, root_path)

    daemon = read_yaml(daemon_path)
    if daemon:
        daemon["available"] = "_read_error" not in daemon
        daemon["path"] = workspace_relpath(daemon_path, root_path)
        if daemon_path.exists():
            daemon["mtime_utc"] = (
                datetime.fromtimestamp(daemon_path.stat().st_mtime, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
    else:
        daemon = {"available": False, "path": workspace_relpath(daemon_path, root_path)}
    if source.demo and not daemon.get("available"):
        daemon = demo_ko_daemon()

    state = read_yaml(state_path)
    if state:
        state["available"] = "_read_error" not in state
        state["path"] = workspace_relpath(state_path, root_path)
    else:
        state = {"available": False, "path": workspace_relpath(state_path, root_path)}
    if source.demo and not state.get("available"):
        state = demo_ko_state()

    runs = list_auto_runs(source, root_path)
    if source.demo and not runs:
        runs = demo_ko_runs()
    state_event_frame = read_parquet(state_events_path)
    state_event_columns = [
        column
        for column in [
            "created_at_utc",
            "event_type",
            "strategy_id",
            "account",
            "symbol",
            "previous_status",
            "new_status",
            "signal_timestamp",
            "ticket_action",
            "ticket_quantity",
            "quantity",
            "desired_position_unit",
            "position_unit",
            "reconciliation_decision",
            "state_updated",
        ]
        if column in state_event_frame.columns
    ]
    if state_event_columns:
        state_event_frame = state_event_frame.loc[:, state_event_columns]
    state_events = frame_records(state_event_frame, limit=80)
    if source.demo and not state_events:
        state_events = demo_ko_state_events()
    pnl_event_frame = read_parquet(pnl_events_path)
    pnl_events = frame_records(pnl_event_frame, limit=200, ascending=True)
    curve = pnl_curve_from_events(pnl_events)
    manual_ledger = list_manual_ledger(source, db_path=db_path)
    manual_curve = curve_from_manual_ledger(manual_ledger)
    ledger_rows = normalize_ledger_events(artifact_events=pnl_events, manual_entries=[])
    manual_ledger_rows = normalize_ledger_events(artifact_events=[], manual_entries=manual_ledger)
    position = current_position_snapshot(state)
    fallback_side = str(position.get("side") or "").lower() or None
    position_timeline = position_timeline_from_state_events(state_events, fallback_side=fallback_side)
    operations = operation_history(state_events=state_events, pnl_events=pnl_events, fallback_side=fallback_side)
    latest_run = runs[0] if runs else None
    kill_switch_exists = kill_switch_path.exists()
    runtime_control = read_runtime_control(source, db_path=db_path, kill_switch_exists=kill_switch_exists, root=root_path)
    alerts = build_alerts(
        source=source,
        kill_switch_exists=kill_switch_exists,
        daemon=daemon,
        state=state,
        latest_run=latest_run,
        pnl_log_available=pnl_events_path.exists(),
    )

    return {
        "candidate_id": source.candidate_id,
        "name": source.name,
        "strategy_id": source.strategy_id,
        "mode": source.mode,
        "symbol": source.symbol,
        "account": source.account,
        "overall_state": overall_state(alerts, daemon),
        "control": {
            "kill_switch_exists": kill_switch_exists,
            "kill_switch_path": workspace_relpath(source.kill_switch_path, root_path),
            "pause_enabled": not kill_switch_exists,
            "resume_enabled": kill_switch_exists,
            "runtime": runtime_control,
            "effective_enabled": bool(runtime_control.get("enabled")) and not kill_switch_exists,
        },
        "daemon": json_safe(daemon),
        "state": json_safe(state),
        "position": {
            "current": position,
            "timeline": position_timeline,
        },
        "operations": operations,
        "latest_run": latest_run,
        "recent_runs": runs,
        "state_events": state_events,
        "pnl": {
            **pnl_metrics(curve),
            "source_available": pnl_events_path.exists(),
            "source_type": "runner_pnl_events",
            "source_path": source.pnl_events_path.as_posix(),
            "curve": curve,
            "events": pnl_events,
            "manual_ledger_affects_pnl": False,
            "excluded_manual_ledger_count": len(manual_ledger),
        },
        "ledger": {
            "events": ledger_rows,
            "manual_count": 0,
            "artifact_count": len(pnl_events),
        },
        "manual_ledger": {
            "events": manual_ledger_rows,
            "count": len(manual_ledger),
            "metrics": pnl_metrics(manual_curve),
            "affects_operational_pnl": False,
        },
        "market": market_snapshot(source, ledger_rows),
        "alerts": alerts,
        "updated_at_utc": utc_now(),
    }


def control_center_snapshot(
    *,
    root: str | Path = PROJECT_ROOT,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    candidates = [candidate_control_snapshot(source.candidate_id, root=root, db_path=db_path) for source in ACTIVE_PAPER_SOURCES]
    live_candidates = [candidate_control_snapshot(source.candidate_id, root=root, db_path=db_path) for source in LIVE_SOURCES]
    all_candidates = candidates + live_candidates
    critical = sum(1 for candidate in all_candidates for alert in candidate["alerts"] if alert["severity"] == "critical")
    warnings = sum(1 for candidate in all_candidates for alert in candidate["alerts"] if alert["severity"] == "warning")
    paused = sum(1 for candidate in all_candidates if candidate["control"]["kill_switch_exists"])
    connection = connection_snapshot(root=root)
    return {
        "title": "Trading Strats Control Center",
        "active_candidates": all_candidates,
        "sections": {
            "paper": candidates,
            "live": live_candidates,
        },
        "summary": {
            "active_count": len(all_candidates),
            "paper_count": len(candidates),
            "live_count": len(live_candidates),
            "paused_count": paused,
            "critical_alerts": critical,
            "warning_alerts": warnings,
        },
        "connection": connection,
        "operations": connection.get("operations", {}),
        "updated_at_utc": utc_now(),
    }


def apply_control_action(
    candidate_id: str,
    *,
    action: str,
    reason: str = "",
    actor: str = "dashboard",
    root: str | Path = PROJECT_ROOT,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    source = source_for(candidate_id)
    root_path = Path(root).resolve()
    kill_switch_path = resolve_workspace_path(source.kill_switch_path, root_path)
    action_normalized = action.strip().lower()
    if action_normalized == "pause":
        current = read_runtime_control(source, db_path=db_path, kill_switch_exists=True, root=root_path)
        kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        kill_switch_path.write_text(
            yaml.safe_dump(
                {
                    "created_at_utc": utc_now(),
                    "candidate_id": candidate_id,
                    "strategy_id": source.strategy_id,
                    "actor": actor,
                    "reason": reason or "manual pause from candidate control center",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        save_runtime_control(
            source,
            enabled=False,
            capital_mode=current.get("capital_mode", "net_fraction"),
            capital_value=float(current.get("capital_value", 1.0)),
            capital_basis=current.get("capital_basis", "buying_power_fraction"),
            actor=actor,
            notes=reason,
            db_path=db_path,
        )
    elif action_normalized == "resume":
        if kill_switch_path.exists():
            kill_switch_path.unlink()
        current = read_runtime_control(source, db_path=db_path, kill_switch_exists=False, root=root_path)
        save_runtime_control(
            source,
            enabled=True,
            capital_mode=current.get("capital_mode", "net_fraction"),
            capital_value=float(current.get("capital_value", 1.0)),
            capital_basis=current.get("capital_basis", "buying_power_fraction"),
            actor=actor,
            notes=reason,
            db_path=db_path,
        )
    else:
        raise ValueError(f"unsupported control action: {action}")
    return candidate_control_snapshot(candidate_id, root=root_path, db_path=db_path)


def update_strategy_runtime_control(
    candidate_id: str,
    *,
    enabled: bool,
    capital_mode: str,
    capital_value: float,
    capital_basis: str = "buying_power_fraction",
    actor: str = "dashboard",
    notes: str = "",
    apply_to_config: bool = False,
    root: str | Path = PROJECT_ROOT,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    source = source_for(candidate_id)
    root_path = Path(root).resolve()
    current_kill_switch = resolve_workspace_path(source.kill_switch_path, root_path).exists()
    control = save_runtime_control(
        source,
        enabled=enabled,
        capital_mode=capital_mode,
        capital_value=capital_value,
        capital_basis=capital_basis,
        actor=actor,
        notes=notes,
        db_path=db_path,
    )
    if enabled and current_kill_switch:
        apply_control_action(candidate_id, action="resume", actor=actor, reason=notes, root=root_path, db_path=db_path)
    if not enabled and not current_kill_switch:
        apply_control_action(candidate_id, action="pause", actor=actor, reason=notes, root=root_path, db_path=db_path)
    apply_result = {"applied": False, "reason": "not requested"}
    if apply_to_config:
        apply_result = apply_capital_policy_to_config(source, control, root=root_path)
    snapshot = candidate_control_snapshot(candidate_id, root=root_path, db_path=db_path)
    snapshot["capital_apply_result"] = apply_result
    return snapshot

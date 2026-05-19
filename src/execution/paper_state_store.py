from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_state_h1c.yaml")
VALID_STATUSES = {"flat", "pending_entry", "open", "pending_exit"}


@dataclass(frozen=True)
class PaperStateConfig:
    strategy_id: str
    account: str
    symbol: str
    position_side: str
    state_path: Path
    event_log_path: Path
    output_dir: Path
    allow_send_orders_tickets: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PaperStateConfig":
        state = dict(raw.get("state", {}))
        config = cls(
            strategy_id=str(state.get("strategy_id", "")).strip(),
            account=str(state.get("account", "")).strip(),
            symbol=str(state.get("symbol", "")).strip().upper(),
            position_side=str(state.get("position_side", "short")).strip().lower(),
            state_path=Path(state.get("state_path", "results/paper/h1c_state/state.yaml")),
            event_log_path=Path(state.get("event_log_path", "results/paper/h1c_state/events.parquet")),
            output_dir=Path(state.get("output_dir", "results/paper/h1c_state/runs")),
            allow_send_orders_tickets=bool(state.get("allow_send_orders_tickets", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("state.strategy_id is required")
        if not self.account:
            raise ValueError("state.account is required")
        if not self.symbol:
            raise ValueError("state.symbol is required")
        if self.position_side not in {"long", "short"}:
            raise ValueError("state.position_side must be long or short")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ["state_path", "event_log_path", "output_dir"]:
            data[key] = data[key].as_posix()
        return data


@dataclass(frozen=True)
class PaperStatePaths:
    output_dir: Path
    state_path: Path
    event_log_path: Path
    event_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_state_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperStateConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return PaperStateConfig.from_mapping(raw)


def initial_state(config: PaperStateConfig) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": 1,
        "strategy_id": config.strategy_id,
        "account": config.account,
        "symbol": config.symbol,
        "position_side": config.position_side,
        "status": "flat",
        "position_unit": 0.0,
        "quantity": 0.0,
        "desired_position_unit": 0.0,
        "pending_ticket": None,
        "open_position": None,
        "last_signal_timestamp": None,
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def load_state(config: PaperStateConfig) -> dict[str, Any]:
    if not config.state_path.exists():
        return initial_state(config)
    with config.state_path.open("r", encoding="utf-8") as handle:
        state = yaml.safe_load(handle) or {}
    if not isinstance(state, dict):
        raise ValueError(f"expected YAML mapping: {config.state_path}")
    validate_state(state, config)
    return state


def write_state(state: dict[str, Any], path: str | Path) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(yaml.safe_dump(state, sort_keys=False), encoding="utf-8")


def load_ticket(path: str | Path) -> dict[str, Any]:
    ticket_path = Path(path)
    with ticket_path.open("r", encoding="utf-8") as handle:
        ticket = yaml.safe_load(handle) or {}
    if not isinstance(ticket, dict):
        raise ValueError(f"expected YAML mapping: {ticket_path}")
    return ticket


def validate_state(state: dict[str, Any], config: PaperStateConfig) -> None:
    if state.get("strategy_id") != config.strategy_id:
        raise ValueError("state strategy_id does not match config")
    if state.get("account") != config.account:
        raise ValueError("state account does not match config")
    if state.get("symbol") != config.symbol:
        raise ValueError("state symbol does not match config")
    if state.get("status") not in VALID_STATUSES:
        raise ValueError(f"unsupported state status: {state.get('status')}")
    if state.get("position_side") not in {None, config.position_side}:
        raise ValueError("state position_side does not match config")


def validate_ticket(ticket: dict[str, Any], config: PaperStateConfig) -> None:
    if ticket.get("strategy_id") != config.strategy_id:
        raise ValueError("ticket strategy_id does not match config")
    if ticket.get("account") != config.account:
        raise ValueError("ticket account does not match config")
    if str(ticket.get("symbol", "")).upper() != config.symbol:
        raise ValueError("ticket symbol does not match config")
    if bool(ticket.get("send_orders", False)) and not config.allow_send_orders_tickets:
        raise ValueError("state store only accepts signal-only tickets with send_orders=false")
    action = str(ticket.get("action", "NONE")).upper()
    if action not in {"NONE", "SELL", "BUY"}:
        raise ValueError(f"unsupported ticket action: {action}")


def _ticket_key(ticket: dict[str, Any]) -> str:
    return f"{ticket.get('strategy_id')}|{ticket.get('symbol')}|{ticket.get('signal_timestamp')}|{ticket.get('action')}|{ticket.get('quantity')}"


def _position_unit(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def _entry_action(side: str) -> str:
    return "BUY" if side == "long" else "SELL"


def _exit_action(side: str) -> str:
    return "SELL" if side == "long" else "BUY"


def apply_ticket_to_state(state: dict[str, Any], ticket: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(state)
    now = utc_now()
    action = str(ticket.get("action", "NONE")).upper()
    desired = float(ticket.get("desired_position_unit", 0.0) or 0.0)
    quantity = float(ticket.get("quantity", 0.0) or 0.0)
    status = str(updated.get("status", "flat"))
    side = str(updated.get("position_side") or ("long" if desired > 0 else "short")).lower()
    side_unit = _position_unit(side)
    entry_action = _entry_action(side)
    exit_action = _exit_action(side)
    previous_ticket = updated.get("pending_ticket") or {}
    duplicate_pending = bool(status in {"pending_entry", "pending_exit"} and previous_ticket.get("ticket_key") == _ticket_key(ticket))

    event_type = "no_change"
    if duplicate_pending:
        event_type = "duplicate_pending_ticket_ignored"
    elif status == "flat" and action == entry_action and desired == side_unit and quantity > 0.0:
        event_type = "pending_entry_created"
        updated["status"] = "pending_entry"
        updated["position_side"] = side
        updated["position_unit"] = 0.0
        updated["quantity"] = 0.0
        updated["desired_position_unit"] = desired
        updated["pending_ticket"] = {
            "ticket_key": _ticket_key(ticket),
            "action": action,
            "quantity": quantity,
            "order_type": ticket.get("order_type"),
            "time_in_force": ticket.get("time_in_force"),
            "signal_timestamp": ticket.get("signal_timestamp"),
            "session": ticket.get("session"),
            "bar_index": ticket.get("bar_index"),
            "theoretical_entry_timestamp": ticket.get("theoretical_entry_timestamp"),
            "theoretical_entry_price": ticket.get("theoretical_entry_price"),
            "theoretical_exit_timestamp": ticket.get("theoretical_exit_timestamp"),
            "theoretical_exit_price": ticket.get("theoretical_exit_price"),
            "exit_rule": ticket.get("exit_rule"),
            "horizon_bars": ticket.get("horizon_bars"),
            "min_hold_bars": ticket.get("min_hold_bars"),
            "stop_loss_bps": ticket.get("stop_loss_bps"),
            "take_profit_bps": ticket.get("take_profit_bps"),
            "status": "awaiting_fill_or_simulated_fill",
        }
    elif status == "open" and action == exit_action and desired == 0.0 and quantity > 0.0:
        event_type = "pending_exit_created"
        updated["status"] = "pending_exit"
        updated["position_side"] = side
        updated["desired_position_unit"] = 0.0
        updated["pending_ticket"] = {
            "ticket_key": _ticket_key(ticket),
            "action": action,
            "quantity": quantity,
            "order_type": ticket.get("order_type"),
            "time_in_force": ticket.get("time_in_force"),
            "signal_timestamp": ticket.get("signal_timestamp"),
            "session": ticket.get("session"),
            "bar_index": ticket.get("bar_index"),
            "theoretical_entry_timestamp": ticket.get("theoretical_entry_timestamp"),
            "theoretical_entry_price": ticket.get("theoretical_entry_price"),
            "theoretical_exit_timestamp": ticket.get("theoretical_exit_timestamp"),
            "theoretical_exit_price": ticket.get("theoretical_exit_price"),
            "exit_rule": ticket.get("exit_rule"),
            "horizon_bars": ticket.get("horizon_bars"),
            "min_hold_bars": ticket.get("min_hold_bars"),
            "stop_loss_bps": ticket.get("stop_loss_bps"),
            "take_profit_bps": ticket.get("take_profit_bps"),
            "status": "awaiting_exit_fill",
        }
    elif status == "flat" and action == "NONE":
        event_type = "flat_no_signal"
        updated["desired_position_unit"] = 0.0
        updated["pending_ticket"] = None
    elif status == "pending_entry" and action == "NONE":
        event_type = "pending_entry_kept"
    elif status == "pending_exit" and action == "NONE":
        event_type = "pending_exit_kept"
    elif status in {"open", "pending_exit"}:
        event_type = "state_unchanged_open_position"
    else:
        event_type = "state_unchanged_unhandled_ticket"

    updated["last_signal_timestamp"] = ticket.get("signal_timestamp")
    updated["updated_at_utc"] = now
    event = {
        "created_at_utc": now,
        "event_type": event_type,
        "strategy_id": updated["strategy_id"],
        "account": updated["account"],
        "symbol": updated["symbol"],
        "previous_status": status,
        "new_status": updated["status"],
        "signal_timestamp": ticket.get("signal_timestamp"),
        "ticket_action": action,
        "ticket_quantity": quantity,
        "desired_position_unit": desired,
        "position_unit": float(updated.get("position_unit", 0.0) or 0.0),
        "state_updated": event_type not in {"duplicate_pending_ticket_ignored"},
        "ticket_json": json.dumps(ticket, sort_keys=True, default=str),
    }
    return updated, event


def append_event(event: dict[str, Any], event_log_path: str | Path) -> None:
    path = Path(event_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([event])
    if path.exists():
        existing = pd.read_parquet(path)
        output = pd.concat([existing, new_row], ignore_index=True)
    else:
        output = new_row
    output.to_parquet(path, index=False)


def _write_report(path: Path, state: dict[str, Any], event: dict[str, Any]) -> None:
    lines = [
        "# H1c paper state update",
        "",
        f"- Event: `{event['event_type']}`",
        f"- Previous status: `{event['previous_status']}`",
        f"- New status: `{event['new_status']}`",
        f"- Signal timestamp: `{event['signal_timestamp']}`",
        f"- Ticket action: `{event['ticket_action']}`",
        f"- Ticket quantity: `{event['ticket_quantity']}`",
        f"- Position unit: `{state['position_unit']}`",
        f"- Desired position unit: `{state['desired_position_unit']}`",
        "",
        "This state store does not submit orders and does not mark fills by itself.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_ticket(
    *,
    ticket_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    state_path: str | Path | None = None,
    event_log_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PaperStatePaths, dict[str, Any]]:
    config = load_state_config(config_path)
    if state_path is not None:
        config = PaperStateConfig(**{**asdict(config), "state_path": Path(state_path)})
    if event_log_path is not None:
        config = PaperStateConfig(**{**asdict(config), "event_log_path": Path(event_log_path)})
    if output_dir is not None:
        config = PaperStateConfig(**{**asdict(config), "output_dir": Path(output_dir)})

    ticket = load_ticket(ticket_path)
    validate_ticket(ticket, config)
    state = load_state(config)
    updated, event = apply_ticket_to_state(state, ticket)
    validate_state(updated, config)

    run_dir = config.output_dir / utc_now().replace(":", "").replace("-", "")
    paths = PaperStatePaths(
        output_dir=run_dir,
        state_path=config.state_path,
        event_log_path=config.event_log_path,
        event_path=run_dir / "event.yaml",
        report_path=run_dir / "report.md",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_state(updated, config.state_path)
    append_event(event, config.event_log_path)
    paths.event_path.write_text(yaml.safe_dump(event, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, updated, event)
    return paths, {"state": updated, "event": event}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply an H1c paper signal ticket to the local paper state store")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--event-log-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, summary = apply_ticket(
        ticket_path=args.ticket,
        config_path=args.config,
        state_path=args.state_path,
        event_log_path=args.event_log_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": summary}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

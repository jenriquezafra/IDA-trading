from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.execution.flatten_plan import ExecutionPolicy


DEFAULT_CONFIG_PATH = Path("configs/execution/h1c_order_plan.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_order_plan")


@dataclass(frozen=True)
class H1COrderPlanConfig:
    strategy_id: str
    account: str
    symbol: str
    sec_type: str
    currency: str
    routing_exchange: str
    primary_exchange: str
    require_reconciliation_ok: bool
    allowed_reconciliation_decisions: tuple[str, ...]
    exit_allowed_reconciliation_decisions: tuple[str, ...]
    max_quantity: float
    execution_policy: ExecutionPolicy
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "H1COrderPlanConfig":
        plan = dict(raw.get("plan", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            strategy_id=str(plan.get("strategy_id", "")).strip(),
            account=str(plan.get("account", "")).strip(),
            symbol=str(plan.get("symbol", "QQQ")).strip().upper(),
            sec_type=str(plan.get("sec_type", "STK")).strip().upper(),
            currency=str(plan.get("currency", "USD")).strip().upper(),
            routing_exchange=str(plan.get("routing_exchange", "SMART")).strip().upper(),
            primary_exchange=str(plan.get("primary_exchange", "NASDAQ")).strip().upper(),
            require_reconciliation_ok=bool(plan.get("require_reconciliation_ok", True)),
            allowed_reconciliation_decisions=tuple(str(value).strip() for value in plan.get("allowed_reconciliation_decisions", ["OK_FLAT"])),
            exit_allowed_reconciliation_decisions=tuple(str(value).strip() for value in plan.get("exit_allowed_reconciliation_decisions", ["OK_OPEN"])),
            max_quantity=float(plan.get("max_quantity", 1.0)),
            execution_policy=ExecutionPolicy.from_mapping(raw.get("execution_policy", {})),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("plan.strategy_id is required")
        if not self.account:
            raise ValueError("plan.account is required")
        if self.sec_type != "STK":
            raise ValueError("H1c order planner currently supports STK only")
        if self.max_quantity <= 0:
            raise ValueError("max_quantity must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_reconciliation_decisions"] = list(self.allowed_reconciliation_decisions)
        data["exit_allowed_reconciliation_decisions"] = list(self.exit_allowed_reconciliation_decisions)
        data["execution_policy"] = self.execution_policy.to_dict()
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class H1COrderPlanPaths:
    output_dir: Path
    manifest_path: Path
    orders_path: Path
    orders_csv_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_order_plan_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H1COrderPlanConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return H1COrderPlanConfig.from_mapping(raw)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def _reference_price(ticket: dict[str, Any], action: str) -> float:
    keys = ("theoretical_exit_price", "theoretical_entry_price") if action == "BUY" else ("theoretical_entry_price", "theoretical_exit_price")
    for key in keys:
        value = ticket.get(key)
        if value is None or pd.isna(value):
            continue
        return float(value)
    return 0.0


def build_h1c_order_plan(ticket: dict[str, Any], reconciliation_manifest: dict[str, Any] | None, config: H1COrderPlanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if ticket.get("strategy_id") != config.strategy_id:
        raise ValueError("ticket strategy_id does not match plan config")
    if ticket.get("account") != config.account:
        raise ValueError("ticket account does not match plan config")
    if str(ticket.get("symbol", "")).upper() != config.symbol:
        raise ValueError("ticket symbol does not match plan config")

    action = str(ticket.get("action", "NONE")).upper()
    quantity = float(ticket.get("quantity", 0.0) or 0.0)
    reconciliation = dict((reconciliation_manifest or {}).get("reconciliation", {}) or {})
    reconciliation_decision = str(reconciliation.get("decision", "NOT_PROVIDED"))
    allowed_reconciliation_decisions = config.exit_allowed_reconciliation_decisions if action == "BUY" else config.allowed_reconciliation_decisions
    reconciliation_ok = reconciliation_decision in set(allowed_reconciliation_decisions)
    reference_price = _reference_price(ticket, action)
    intent = "h1c_short_exit" if action == "BUY" else "h1c_short_entry"

    orders: list[dict[str, Any]] = []
    block_reason = ""
    decision = "no_order_no_signal"
    if action == "NONE" or quantity <= 0.0:
        decision = "no_order_no_signal"
    elif action not in {"SELL", "BUY"}:
        decision = "blocked_unsupported_action"
        block_reason = f"unsupported action={action}"
    elif quantity > config.max_quantity:
        decision = "blocked_quantity_limit"
        block_reason = f"quantity {quantity} exceeds max_quantity {config.max_quantity}"
    elif config.require_reconciliation_ok and not reconciliation_ok:
        decision = "blocked_reconciliation"
        block_reason = f"reconciliation decision {reconciliation_decision} is not in allowed set"
    else:
        decision = "ready_for_review"
        orders.append(
            {
                "account": config.account,
                "symbol": config.symbol,
                "sec_type": config.sec_type,
                "currency": config.currency,
                "action": action,
                "quantity": quantity,
                "order_type": config.execution_policy.order_type,
                "tif": config.execution_policy.tif,
                "outside_rth": config.execution_policy.outside_rth,
                "routing_exchange": config.routing_exchange,
                "primary_exchange": config.primary_exchange,
                "transmit": False,
                "dry_run": True,
                "status": "planned",
                "ticket_signal_timestamp": ticket.get("signal_timestamp"),
                "theoretical_entry_price": ticket.get("theoretical_entry_price"),
                "theoretical_exit_price": ticket.get("theoretical_exit_price"),
                "theoretical_exit_timestamp": ticket.get("theoretical_exit_timestamp"),
                "exit_rule": ticket.get("exit_rule"),
                "horizon_bars": ticket.get("horizon_bars"),
                "intent": intent,
                "approx_notional_at_ticket_price": quantity * reference_price,
            }
        )
    columns = [
        "account",
        "symbol",
        "sec_type",
        "currency",
        "action",
        "quantity",
        "order_type",
        "tif",
        "outside_rth",
        "routing_exchange",
        "primary_exchange",
        "transmit",
        "dry_run",
        "status",
        "ticket_signal_timestamp",
        "theoretical_entry_price",
        "theoretical_exit_price",
        "theoretical_exit_timestamp",
        "exit_rule",
        "horizon_bars",
        "intent",
        "approx_notional_at_ticket_price",
    ]
    orders_df = pd.DataFrame(orders, columns=columns)
    summary = {
        "decision": decision,
        "block_reason": block_reason,
        "strategy_id": config.strategy_id,
        "expected_account": config.account,
        "symbol": config.symbol,
        "ticket_action": action,
        "ticket_quantity": quantity,
        "planned_orders": int(len(orders_df)),
        "dry_run": True,
        "transmit": False,
        "reconciliation_decision": reconciliation_decision,
        "reconciliation_required": config.require_reconciliation_ok,
        "allowed_reconciliation_decisions": list(allowed_reconciliation_decisions),
        "intent": intent,
        "execution_policy": config.execution_policy.to_dict(),
    }
    return orders_df, summary


def write_h1c_order_plan(
    *,
    ticket_path: str | Path,
    reconciliation_manifest_path: str | Path | None,
    orders: pd.DataFrame,
    summary: dict[str, Any],
    config: H1COrderPlanConfig,
    output_dir: str | Path | None = None,
) -> H1COrderPlanPaths:
    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = H1COrderPlanPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        orders_path=root / "orders.parquet",
        orders_csv_path=root / "orders.csv",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(paths.orders_path, index=False)
    orders.to_csv(paths.orders_csv_path, index=False)
    manifest = {
        "schema_version": 1,
        "run": {"run_type": "h1c_paper_order_plan", "created_at_utc": created, "status": "complete"},
        "config": config.to_dict(),
        "source": {
            "ticket_path": Path(ticket_path).as_posix(),
            "reconciliation_manifest_path": "" if reconciliation_manifest_path is None else Path(reconciliation_manifest_path).as_posix(),
        },
        "summary": summary,
        "outputs": {"orders": paths.orders_path.as_posix(), "orders_csv": paths.orders_csv_path.as_posix(), "report": paths.report_path.as_posix()},
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    lines = [
        "# H1c paper order plan",
        "",
        f"- Created UTC: `{created}`",
        f"- Decision: `{summary['decision']}`",
        f"- Block reason: `{summary['block_reason']}`",
        f"- Ticket action: `{summary['ticket_action']}`",
        f"- Ticket quantity: `{summary['ticket_quantity']}`",
        f"- Reconciliation decision: `{summary['reconciliation_decision']}`",
        f"- Planned orders: `{summary['planned_orders']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Transmit: `{summary['transmit']}`",
        "",
        "This plan is offline. It does not submit orders and all planned rows use `transmit=false`.",
    ]
    paths.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def create_h1c_order_plan(
    *,
    ticket_path: str | Path,
    reconciliation_manifest_path: str | Path | None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
) -> tuple[H1COrderPlanPaths, dict[str, Any]]:
    config = load_order_plan_config(config_path)
    ticket = load_yaml_mapping(ticket_path)
    reconciliation_manifest = load_yaml_mapping(reconciliation_manifest_path) if reconciliation_manifest_path is not None else None
    orders, summary = build_h1c_order_plan(ticket, reconciliation_manifest, config)
    paths = write_h1c_order_plan(
        ticket_path=ticket_path,
        reconciliation_manifest_path=reconciliation_manifest_path,
        orders=orders,
        summary=summary,
        config=config,
        output_dir=output_dir,
    )
    return paths, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a reviewable H1c paper order plan from a signal ticket")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--reconciliation-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, summary = create_h1c_order_plan(
        ticket_path=args.ticket,
        reconciliation_manifest_path=args.reconciliation_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

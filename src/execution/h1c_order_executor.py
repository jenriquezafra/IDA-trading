from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.execution.flatten_executor import (
    IBKRFlattenExecutorClient,
    IBKRFlattenExecutorConfig,
    validate_execution_unlock,
)
from src.execution.paper_state_store import utc_now


DEFAULT_CONFIG_PATH = Path("configs/execution/h1c_order_executor.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_order_execution")


@dataclass(frozen=True)
class H1CExecutionPaths:
    output_dir: Path
    manifest_path: Path
    preflight_path: Path
    submitted_orders_path: Path
    report_path: Path


def load_executor_config(path: str | Path = DEFAULT_CONFIG_PATH) -> IBKRFlattenExecutorConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return IBKRFlattenExecutorConfig.from_mapping(raw)


def plan_fingerprint(plan_dir: str | Path) -> str:
    root = Path(plan_dir)
    digest = hashlib.sha256()
    for filename in ("manifest.yaml", "orders.parquet"):
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"plan file not found: {path}")
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_plan(plan_dir: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(plan_dir)
    with (root / "manifest.yaml").open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"expected YAML mapping: {root / 'manifest.yaml'}")
    return manifest, pd.read_parquet(root / "orders.parquet")


def validate_h1c_plan_for_execution(*, plan_dir: str | Path, config: IBKRFlattenExecutorConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest, orders = load_plan(plan_dir)
    summary = dict(manifest.get("summary", {}) or {})
    run = dict(manifest.get("run", {}) or {})
    fingerprint = plan_fingerprint(plan_dir)
    errors: list[str] = []
    if run.get("run_type") != "h1c_paper_order_plan":
        errors.append("plan manifest run_type must be h1c_paper_order_plan")
    if summary.get("decision") != "ready_for_review":
        errors.append("plan decision must be ready_for_review")
    if summary.get("expected_account") != config.expected_account:
        errors.append("plan expected_account does not match executor config")
    if bool(summary.get("dry_run")) is not True:
        errors.append("plan dry_run must be true")
    if bool(summary.get("transmit")) is not False:
        errors.append("plan transmit must be false")
    if len(orders) != 1:
        errors.append("H1c executor requires exactly one planned order")
    if len(orders) > config.max_orders:
        errors.append(f"planned order count {len(orders)} exceeds max_orders {config.max_orders}")
    required = {"account", "symbol", "sec_type", "currency", "action", "quantity", "order_type", "tif", "outside_rth", "routing_exchange", "primary_exchange", "transmit", "dry_run", "status"}
    missing = sorted(required.difference(orders.columns))
    if missing:
        errors.append(f"orders missing columns: {missing}")
    total_notional = 0.0
    if not missing and not orders.empty:
        accounts = set(orders["account"].astype(str))
        if accounts != {config.expected_account}:
            errors.append(f"order accounts {sorted(accounts)} do not match expected account {config.expected_account}")
        actions = set(orders["action"].astype(str).str.upper())
        if not actions <= {"SELL", "BUY"}:
            errors.append(f"H1c executor supports SELL/BUY only, got {sorted(actions)}")
        if set(orders["sec_type"].astype(str).str.upper()).difference(config.allowed_sec_types):
            errors.append("unsupported sec_type in plan")
        if set(orders["order_type"].astype(str).str.upper()).difference(config.allowed_order_types):
            errors.append("unsupported order_type in plan")
        if set(orders["tif"].astype(str).str.upper()).difference(config.allowed_tifs):
            errors.append("unsupported tif in plan")
        if not (orders["dry_run"].astype(bool) == True).all():  # noqa: E712
            errors.append("all orders must have dry_run=true")
        if not (orders["transmit"].astype(bool) == False).all():  # noqa: E712
            errors.append("all orders must have transmit=false")
        if set(orders["status"].astype(str)) != {"planned"}:
            errors.append("all orders must have status=planned")
        if (orders["quantity"].astype(float) <= 0).any():
            errors.append("all quantities must be positive")
        if "approx_notional_at_ticket_price" in orders.columns:
            total_notional = float(pd.to_numeric(orders["approx_notional_at_ticket_price"], errors="coerce").fillna(0.0).sum())
            if total_notional > config.max_total_notional_at_avg_cost:
                errors.append(f"planned notional {total_notional:.2f} exceeds limit {config.max_total_notional_at_avg_cost:.2f}")
    preflight = {
        "plan_dir": Path(plan_dir).as_posix(),
        "plan_fingerprint": fingerprint,
        "expected_account": config.expected_account,
        "planned_orders": int(len(orders)),
        "approx_total_notional_at_ticket_price": total_notional,
        "offline_valid": len(errors) == 0,
        "errors": errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    submit_orders = orders.copy()
    submit_orders["approx_notional_at_avg_cost"] = total_notional
    return submit_orders, preflight


def write_execution_run(
    *,
    config: IBKRFlattenExecutorConfig,
    preflight: dict[str, Any],
    unlock: dict[str, Any],
    live_preflight: dict[str, Any] | None,
    submitted_orders: pd.DataFrame,
    execute: bool,
    output_dir: str | Path | None = None,
) -> H1CExecutionPaths:
    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = H1CExecutionPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        preflight_path=root / "preflight.yaml",
        submitted_orders_path=root / "submitted_orders.parquet",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    submitted_orders.to_parquet(paths.submitted_orders_path, index=False)
    status = "submitted" if execute and unlock.get("unlocked") and not submitted_orders.empty else ("blocked" if execute else "dry_run")
    manifest = {
        "schema_version": 1,
        "run": {"run_type": "h1c_paper_order_execution", "created_at_utc": created, "status": status},
        "config": config.to_dict(),
        "preflight": preflight,
        "unlock": unlock,
        "live_preflight": live_preflight or {},
        "outputs": {"preflight": paths.preflight_path.as_posix(), "submitted_orders": paths.submitted_orders_path.as_posix(), "report": paths.report_path.as_posix()},
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    paths.preflight_path.write_text(yaml.safe_dump({**preflight, "unlock": unlock, "live_preflight": live_preflight or {}}, sort_keys=False), encoding="utf-8")
    lines = [
        "# H1c paper order execution",
        "",
        f"- Created UTC: `{created}`",
        f"- Status: `{status}`",
        f"- Execute requested: `{execute}`",
        f"- Unlock: `{unlock.get('unlocked')}`",
        f"- Plan fingerprint: `{preflight['plan_fingerprint']}`",
        f"- Planned orders: `{preflight['planned_orders']}`",
        f"- Submitted orders: `{len(submitted_orders)}`",
        "",
        "No orders are submitted unless every unlock condition is satisfied.",
    ]
    paths.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def run_h1c_order_execution(
    *,
    plan_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    execute: bool = False,
    transmit_orders: bool = False,
    confirm_account: str = "",
    confirm_fingerprint: str = "",
    connect_preflight: bool = False,
    output_dir: str | Path | None = None,
    ib_factory: Callable[[], Any] | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[H1CExecutionPaths, dict[str, Any]]:
    config = load_executor_config(config_path)
    orders, preflight = validate_h1c_plan_for_execution(plan_dir=plan_dir, config=config)
    unlock = validate_execution_unlock(
        config=config,
        preflight=preflight,
        execute=execute,
        transmit_orders=transmit_orders,
        confirm_account=confirm_account,
        confirm_fingerprint=confirm_fingerprint,
        environ=environ,
    )
    live: dict[str, Any] | None = None
    submitted = pd.DataFrame()
    live_block_error = ""
    if connect_preflight or execute:
        client = IBKRFlattenExecutorClient(config, ib_factory=ib_factory)
        try:
            client.connect(readonly=not execute)
            live = client.live_preflight(orders)
            if live["errors"]:
                live_block_error = "; ".join(live["errors"])
            if execute and not live_block_error:
                submitted = client.submit_orders(orders)
        finally:
            client.disconnect()
    paths = write_execution_run(config=config, preflight=preflight, unlock=unlock, live_preflight=live, submitted_orders=submitted, execute=execute, output_dir=output_dir)
    if execute and live_block_error:
        raise ValueError(f"{live_block_error}; report written to {paths.report_path}")
    return paths, {"preflight": preflight, "unlock": unlock, "live_preflight": live or {}, "submitted_orders": int(len(submitted))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate or execute a reviewed H1c paper order plan")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--connect-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--transmit-orders", action="store_true")
    parser.add_argument("--confirm-account", default="")
    parser.add_argument("--confirm-fingerprint", default="")
    args = parser.parse_args(argv)
    paths, summary = run_h1c_order_execution(
        plan_dir=args.plan_dir,
        config_path=args.config,
        execute=args.execute,
        transmit_orders=args.transmit_orders,
        confirm_account=args.confirm_account,
        confirm_fingerprint=args.confirm_fingerprint,
        connect_preflight=args.connect_preflight,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

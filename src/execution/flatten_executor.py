from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.execution.ibkr_read_only import PAPER_GATEWAY_PORT, PAPER_TWS_PORT


DEFAULT_CONFIG_PATH = Path("configs/execution/ibkr_paper_executor.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/flatten_execution")


@dataclass(frozen=True)
class IBKRFlattenExecutorConfig:
    host: str
    port: int
    client_id: int
    timeout_seconds: float
    trading_mode: str
    expected_account: str
    execution_enabled: bool
    allow_orders: bool
    require_paper_account: bool
    require_paper_port: bool
    require_no_open_trades: bool
    require_market_open: bool
    allow_opg_outside_rth_submission: bool
    require_account_confirmation: bool
    require_fingerprint_confirmation: bool
    require_env_confirmation: bool
    env_confirmation_var: str
    max_orders: int
    max_total_notional_at_avg_cost: float
    allowed_sec_types: tuple[str, ...]
    allowed_order_types: tuple[str, ...]
    allowed_tifs: tuple[str, ...]
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "IBKRFlattenExecutorConfig":
        connection = dict(raw.get("connection", {}))
        safety = dict(raw.get("safety", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            host=str(connection.get("host", "127.0.0.1")).strip(),
            port=int(connection.get("port", PAPER_GATEWAY_PORT)),
            client_id=int(connection.get("client_id", 72)),
            timeout_seconds=float(connection.get("timeout_seconds", 10.0)),
            trading_mode=str(connection.get("trading_mode", "paper")).strip().lower(),
            expected_account=str(connection.get("expected_account", "") or "").strip(),
            execution_enabled=bool(safety.get("execution_enabled", False)),
            allow_orders=bool(safety.get("allow_orders", False)),
            require_paper_account=bool(safety.get("require_paper_account", True)),
            require_paper_port=bool(safety.get("require_paper_port", True)),
            require_no_open_trades=bool(safety.get("require_no_open_trades", True)),
            require_market_open=bool(safety.get("require_market_open", True)),
            allow_opg_outside_rth_submission=bool(safety.get("allow_opg_outside_rth_submission", False)),
            require_account_confirmation=bool(safety.get("require_account_confirmation", True)),
            require_fingerprint_confirmation=bool(safety.get("require_fingerprint_confirmation", True)),
            require_env_confirmation=bool(safety.get("require_env_confirmation", True)),
            env_confirmation_var=str(safety.get("env_confirmation_var", "IBKR_PAPER_EXECUTION_CONFIRM")).strip(),
            max_orders=int(safety.get("max_orders", 50)),
            max_total_notional_at_avg_cost=float(safety.get("max_total_notional_at_avg_cost", 100_000.0)),
            allowed_sec_types=tuple(str(value).upper() for value in safety.get("allowed_sec_types", ["STK"])),
            allowed_order_types=tuple(str(value).upper() for value in safety.get("allowed_order_types", ["MKT"])),
            allowed_tifs=tuple(str(value).upper() for value in safety.get("allowed_tifs", ["DAY", "OPG"])),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.trading_mode != "paper":
            raise ValueError("IBKR flatten executor requires trading_mode=paper")
        if self.require_paper_port and self.port not in {PAPER_GATEWAY_PORT, PAPER_TWS_PORT}:
            raise ValueError(f"paper mode should use IB Gateway {PAPER_GATEWAY_PORT} or TWS {PAPER_TWS_PORT}; got port={self.port}")
        if not self.expected_account:
            raise ValueError("expected_account is required for flatten execution")
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_orders <= 0:
            raise ValueError("max_orders must be positive")
        if self.max_total_notional_at_avg_cost <= 0:
            raise ValueError("max_total_notional_at_avg_cost must be positive")
        if not self.env_confirmation_var:
            raise ValueError("env_confirmation_var is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_sec_types"] = list(self.allowed_sec_types)
        data["allowed_order_types"] = list(self.allowed_order_types)
        data["allowed_tifs"] = list(self.allowed_tifs)
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class FlattenExecutionPaths:
    output_dir: Path
    manifest_path: Path
    preflight_path: Path
    submitted_orders_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    manifest_path = root / "manifest.yaml"
    orders_path = root / "orders.parquet"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"expected YAML mapping: {manifest_path}")
    orders = pd.read_parquet(orders_path)
    return manifest, orders


def validate_plan_for_execution(
    *,
    plan_dir: str | Path,
    config: IBKRFlattenExecutorConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest, orders = load_plan(plan_dir)
    summary = dict(manifest.get("summary", {}) or {})
    run = dict(manifest.get("run", {}) or {})
    source = dict(manifest.get("source", {}) or {})
    fingerprint = plan_fingerprint(plan_dir)

    errors: list[str] = []
    if run.get("run_type") != "ibkr_flatten_plan":
        errors.append("plan manifest run_type must be ibkr_flatten_plan")
    if summary.get("decision") != "ready_for_review":
        errors.append("plan decision must be ready_for_review")
    if summary.get("expected_account") != config.expected_account:
        errors.append("plan expected_account does not match executor config")
    if bool(summary.get("dry_run")) is not True:
        errors.append("plan summary dry_run must be true")
    if bool(summary.get("transmit")) is not False:
        errors.append("plan summary transmit must be false")
    if int(summary.get("open_trades", -1)) != 0:
        errors.append("source snapshot must have zero open trades")
    if int(summary.get("unsupported_positions", -1)) != 0:
        errors.append("plan must have zero unsupported positions")
    if len(orders) > config.max_orders:
        errors.append(f"planned order count {len(orders)} exceeds max_orders {config.max_orders}")

    required_columns = {
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
        "transmit",
        "dry_run",
        "approx_notional_at_avg_cost",
        "status",
    }
    missing = sorted(required_columns.difference(orders.columns))
    if missing:
        errors.append(f"orders are missing columns: {missing}")

    if not missing and not orders.empty:
        accounts = set(orders["account"].astype(str))
        if accounts != {config.expected_account}:
            errors.append(f"order accounts {sorted(accounts)} do not match expected account {config.expected_account}")
        sec_types = set(orders["sec_type"].astype(str).str.upper())
        invalid_sec_types = sorted(sec_types.difference(config.allowed_sec_types))
        if invalid_sec_types:
            errors.append(f"unsupported order sec_types: {invalid_sec_types}")
        order_types = set(orders["order_type"].astype(str).str.upper())
        invalid_order_types = sorted(order_types.difference(config.allowed_order_types))
        if invalid_order_types:
            errors.append(f"unsupported order types: {invalid_order_types}")
        tifs = set(orders["tif"].astype(str).str.upper())
        invalid_tifs = sorted(tifs.difference(config.allowed_tifs))
        if invalid_tifs:
            errors.append(f"unsupported TIF values: {invalid_tifs}")
        if (orders["tif"].astype(str).str.upper() == "OPG").any() and (orders["outside_rth"].astype(bool) == True).any():  # noqa: E712
            errors.append("OPG orders must have outside_rth=false")
        statuses = set(orders["status"].astype(str))
        if statuses != {"planned"}:
            errors.append(f"all order statuses must be planned; got {sorted(statuses)}")
        if not (orders["dry_run"].astype(bool) == True).all():  # noqa: E712
            errors.append("all plan orders must have dry_run=true")
        if not (orders["transmit"].astype(bool) == False).all():  # noqa: E712
            errors.append("all plan orders must have transmit=false")
        if (orders["quantity"].astype(float) <= 0).any():
            errors.append("all quantities must be positive")
        total_notional = float(orders["approx_notional_at_avg_cost"].astype(float).sum())
        if total_notional > config.max_total_notional_at_avg_cost:
            errors.append(f"planned notional {total_notional:.2f} exceeds max_total_notional_at_avg_cost {config.max_total_notional_at_avg_cost:.2f}")
    else:
        total_notional = 0.0

    preflight = {
        "plan_dir": Path(plan_dir).as_posix(),
        "plan_fingerprint": fingerprint,
        "source_snapshot_dir": source.get("snapshot_dir", ""),
        "expected_account": config.expected_account,
        "planned_orders": int(len(orders)),
        "approx_total_notional_at_avg_cost": total_notional,
        "offline_valid": len(errors) == 0,
        "errors": errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return orders.copy(), preflight


def validate_execution_unlock(
    *,
    config: IBKRFlattenExecutorConfig,
    preflight: dict[str, Any],
    execute: bool,
    transmit_orders: bool,
    confirm_account: str,
    confirm_fingerprint: str,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    checks = {
        "execute_requested": bool(execute),
        "transmit_orders_requested": bool(transmit_orders),
        "config_execution_enabled": config.execution_enabled,
        "config_allow_orders": config.allow_orders,
        "account_confirmed": (not config.require_account_confirmation) or confirm_account == config.expected_account,
        "fingerprint_confirmed": (not config.require_fingerprint_confirmation) or confirm_fingerprint == preflight["plan_fingerprint"],
        "env_confirmed": (not config.require_env_confirmation) or env.get(config.env_confirmation_var) == config.expected_account,
    }
    if not execute:
        return {**checks, "unlocked": False, "reason": "dry_run_only"}

    errors: list[str] = []
    if not transmit_orders:
        errors.append("execution requires --transmit-orders")
    if not config.execution_enabled:
        errors.append("config safety.execution_enabled must be true")
    if not config.allow_orders:
        errors.append("config safety.allow_orders must be true")
    if not checks["account_confirmed"]:
        errors.append(f"execution requires --confirm-account {config.expected_account}")
    if not checks["fingerprint_confirmed"]:
        errors.append(f"execution requires --confirm-fingerprint {preflight['plan_fingerprint']}")
    if not checks["env_confirmed"]:
        errors.append(f"execution requires {config.env_confirmation_var}={config.expected_account}")
    if errors:
        raise ValueError("; ".join(errors))
    return {**checks, "unlocked": True, "reason": "all_execution_confirmations_present"}


def _require_ib_insync() -> tuple[Any, Any, Any]:
    try:
        from ib_insync import IB, MarketOrder, Stock
    except ImportError as exc:
        raise RuntimeError("ib_insync is required for IBKR execution connectivity. Install requirements.txt first.") from exc
    return IB, Stock, MarketOrder


def is_nyse_rth(server_time: datetime) -> bool:
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError("pandas_market_calendars is required for market-hours validation") from exc
    ts = pd.Timestamp(server_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_utc = ts.tz_convert("UTC")
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(start_date=ts_utc.date().isoformat(), end_date=ts_utc.date().isoformat())
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]
    return bool(market_open <= ts_utc <= market_close)


class IBKRFlattenExecutorClient:
    def __init__(self, config: IBKRFlattenExecutorConfig, ib_factory: Callable[[], Any] | None = None) -> None:
        self.config = config
        if ib_factory is None:
            IB, _, _ = _require_ib_insync()
            self._ib_factory = IB
        else:
            self._ib_factory = ib_factory
        self.ib: Any | None = None

    def connect(self, *, readonly: bool) -> None:
        if self.ib is None:
            self.ib = self._ib_factory()
        self.ib.connect(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            timeout=self.config.timeout_seconds,
            readonly=readonly,
        )

    def disconnect(self) -> None:
        if self.ib is not None and getattr(self.ib, "isConnected", lambda: False)():
            self.ib.disconnect()

    def live_preflight(self, orders: pd.DataFrame | None = None) -> dict[str, Any]:
        if self.ib is None or not getattr(self.ib, "isConnected", lambda: False)():
            raise RuntimeError("IBKR executor client is not connected")
        accounts = list(self.ib.managedAccounts())
        open_trades = list(self.ib.openTrades())
        server_time = self.ib.reqCurrentTime()
        market_open = is_nyse_rth(server_time)
        all_orders_opg = bool(orders is not None and not orders.empty and (orders["tif"].astype(str).str.upper() == "OPG").all())
        opg_exception = bool(self.config.allow_opg_outside_rth_submission and all_orders_opg)
        errors: list[str] = []
        if self.config.require_paper_account and self.config.expected_account not in accounts:
            errors.append(f"expected paper account {self.config.expected_account} not found")
        if self.config.require_no_open_trades and open_trades:
            errors.append(f"IBKR has {len(open_trades)} open trades")
        if self.config.require_market_open and not market_open and not opg_exception:
            errors.append("NYSE regular trading hours are not open")
        return {
            "connected": True,
            "host": self.config.host,
            "port": self.config.port,
            "client_id": self.config.client_id,
            "managed_accounts": accounts,
            "open_trades": len(open_trades),
            "server_time": str(server_time),
            "nyse_rth_open": market_open,
            "all_orders_opg": all_orders_opg,
            "opg_outside_rth_submission_allowed": opg_exception,
            "live_valid": len(errors) == 0,
            "errors": errors,
        }

    def submit_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        if self.ib is None or not getattr(self.ib, "isConnected", lambda: False)():
            raise RuntimeError("IBKR executor client is not connected")
        _, Stock, MarketOrder = _require_ib_insync()
        rows: list[dict[str, Any]] = []
        for _, row in orders.iterrows():
            contract = Stock(
                str(row["symbol"]),
                str(row["routing_exchange"] or "SMART"),
                str(row["currency"]),
                primaryExchange=str(row.get("primary_exchange", "") or ""),
            )
            order = MarketOrder(
                str(row["action"]),
                float(row["quantity"]),
                account=self.config.expected_account,
                tif=str(row.get("tif", "DAY") or "DAY"),
                outsideRth=bool(row.get("outside_rth", False)),
            )
            trade = self.ib.placeOrder(contract, order)
            status = getattr(trade, "orderStatus", None)
            placed_order = getattr(trade, "order", order)
            rows.append(
                {
                    "symbol": row["symbol"],
                    "action": row["action"],
                    "quantity": float(row["quantity"]),
                    "order_type": row["order_type"],
                    "tif": row.get("tif", ""),
                    "outside_rth": bool(row.get("outside_rth", False)),
                    "account": self.config.expected_account,
                    "order_id": getattr(placed_order, "orderId", None),
                    "perm_id": getattr(placed_order, "permId", None),
                    "status": getattr(status, "status", "Submitted"),
                    "filled": getattr(status, "filled", None),
                    "remaining": getattr(status, "remaining", None),
                    "avg_fill_price": getattr(status, "avgFillPrice", None),
                }
            )
        return pd.DataFrame(rows)


def write_execution_run(
    *,
    config: IBKRFlattenExecutorConfig,
    preflight: dict[str, Any],
    unlock: dict[str, Any],
    live_preflight: dict[str, Any] | None,
    submitted_orders: pd.DataFrame,
    execute: bool,
    output_dir: str | Path | None = None,
) -> FlattenExecutionPaths:
    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = FlattenExecutionPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        preflight_path=root / "preflight.yaml",
        submitted_orders_path=root / "submitted_orders.parquet",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    submitted_orders.to_parquet(paths.submitted_orders_path, index=False)

    if execute and unlock.get("unlocked") and submitted_orders.shape[0] > 0:
        status = "submitted"
    elif execute:
        status = "blocked"
    else:
        status = "dry_run"
    payload = {
        "schema_version": 1,
        "run": {
            "run_type": "ibkr_flatten_execution",
            "created_at_utc": created,
            "status": status,
        },
        "config": config.to_dict(),
        "preflight": preflight,
        "unlock": unlock,
        "live_preflight": live_preflight or {},
        "outputs": {
            "preflight": paths.preflight_path.as_posix(),
            "submitted_orders": paths.submitted_orders_path.as_posix(),
            "report": paths.report_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    paths.preflight_path.write_text(yaml.safe_dump({**preflight, "unlock": unlock, "live_preflight": live_preflight or {}}, sort_keys=False), encoding="utf-8")

    report = [
        "# IBKR flatten execution",
        "",
        f"- Created UTC: `{created}`",
        f"- Status: `{status}`",
        f"- Execute requested: `{execute}`",
        f"- Unlock: `{unlock.get('unlocked')}`",
        f"- Plan fingerprint: `{preflight['plan_fingerprint']}`",
        f"- Planned orders: `{preflight['planned_orders']}`",
        f"- Submitted orders: `{len(submitted_orders)}`",
        f"- Approx notional at avg cost: `{preflight['approx_total_notional_at_avg_cost']:.2f}`",
        "",
    ]
    if live_preflight is not None:
        report.extend(
            [
                "## Live preflight",
                "",
                f"- Open trades: `{live_preflight['open_trades']}`",
                f"- NYSE RTH open: `{live_preflight['nyse_rth_open']}`",
                f"- All orders OPG: `{live_preflight.get('all_orders_opg', False)}`",
                f"- OPG outside RTH allowed: `{live_preflight.get('opg_outside_rth_submission_allowed', False)}`",
                f"- Live valid: `{live_preflight['live_valid']}`",
                f"- Errors: `{', '.join(live_preflight['errors']) if live_preflight['errors'] else ''}`",
                "",
            ]
        )
    if submitted_orders.empty:
        report.append("No orders were submitted by this run.")
    else:
        report.append("Orders were submitted to IBKR paper by this run.")
    paths.report_path.write_text("\n".join(report), encoding="utf-8")
    return paths


def run_flatten_execution(
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
) -> tuple[FlattenExecutionPaths, dict[str, Any]]:
    config = load_executor_config(config_path)
    orders, preflight = validate_plan_for_execution(plan_dir=plan_dir, config=config)
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
    should_connect = connect_preflight or execute
    if should_connect:
        client = IBKRFlattenExecutorClient(config, ib_factory=ib_factory)
        live_block_error = ""
        try:
            client.connect(readonly=not execute)
            live = client.live_preflight(orders)
            if live["errors"]:
                live_block_error = "; ".join(live["errors"])
            if execute and not live_block_error:
                submitted = client.submit_orders(orders)
        finally:
            client.disconnect()
    else:
        live_block_error = ""

    paths = write_execution_run(
        config=config,
        preflight=preflight,
        unlock=unlock,
        live_preflight=live,
        submitted_orders=submitted,
        execute=execute,
        output_dir=output_dir,
    )
    if execute and live_block_error:
        raise ValueError(f"{live_block_error}; report written to {paths.report_path}")
    return paths, {"preflight": preflight, "unlock": unlock, "live_preflight": live or {}, "submitted_orders": int(len(submitted))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate or execute an IBKR paper flatten plan")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--connect-preflight", action="store_true", help="connect to IBKR read-only and validate account/open trades/market")
    parser.add_argument("--execute", action="store_true", help="submit orders only when every execution unlock is present")
    parser.add_argument("--transmit-orders", action="store_true", help="required with --execute")
    parser.add_argument("--confirm-account", default="")
    parser.add_argument("--confirm-fingerprint", default="")
    args = parser.parse_args(argv)

    paths, summary = run_flatten_execution(
        plan_dir=args.plan_dir,
        config_path=args.config,
        execute=args.execute,
        transmit_orders=args.transmit_orders,
        confirm_account=args.confirm_account,
        confirm_fingerprint=args.confirm_fingerprint,
        connect_preflight=args.connect_preflight,
        output_dir=args.output_dir,
    )
    print(json.dumps({"summary": summary, "paths": {key: str(value) for key, value in asdict(paths).items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

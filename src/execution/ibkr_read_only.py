from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = Path("configs/execution/ibkr_paper_readonly.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/ibkr_read_only")
PAPER_GATEWAY_PORT = 4002
PAPER_TWS_PORT = 7497


@dataclass(frozen=True)
class IBKRReadOnlyConfig:
    host: str
    port: int
    client_id: int
    timeout_seconds: float
    trading_mode: str
    expected_account: str
    read_only: bool
    allow_orders: bool
    require_paper_account: bool
    require_paper_port: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "IBKRReadOnlyConfig":
        connection = dict(raw.get("connection", {}))
        safety = dict(raw.get("safety", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            host=str(connection.get("host", "127.0.0.1")).strip(),
            port=int(connection.get("port", PAPER_GATEWAY_PORT)),
            client_id=int(connection.get("client_id", 71)),
            timeout_seconds=float(connection.get("timeout_seconds", 10.0)),
            trading_mode=str(connection.get("trading_mode", "paper")).strip().lower(),
            expected_account=str(connection.get("expected_account", "") or "").strip(),
            read_only=bool(safety.get("read_only", True)),
            allow_orders=bool(safety.get("allow_orders", False)),
            require_paper_account=bool(safety.get("require_paper_account", True)),
            require_paper_port=bool(safety.get("require_paper_port", True)),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.trading_mode != "paper":
            raise ValueError("IBKR read-only config must use trading_mode=paper")
        if not self.read_only:
            raise ValueError("IBKR read-only config requires safety.read_only=true")
        if self.allow_orders:
            raise ValueError("IBKR read-only config requires safety.allow_orders=false")
        if self.require_paper_port and self.port not in {PAPER_GATEWAY_PORT, PAPER_TWS_PORT}:
            raise ValueError(f"paper mode should use IB Gateway {PAPER_GATEWAY_PORT} or TWS {PAPER_TWS_PORT}; got port={self.port}")
        if self.client_id < 0:
            raise ValueError("client_id must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class IBKRReadOnlySnapshotPaths:
    output_dir: Path
    manifest_path: Path
    account_summary_path: Path
    positions_path: Path
    open_trades_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_ibkr_read_only_config(path: str | Path = DEFAULT_CONFIG_PATH) -> IBKRReadOnlyConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return IBKRReadOnlyConfig.from_mapping(raw)


def _require_ib_insync_factory() -> Any:
    try:
        from ib_insync import IB
    except ImportError as exc:
        raise RuntimeError("ib_insync is required for IBKR connectivity. Install requirements.txt first.") from exc
    return IB


def _contract_dict(contract: Any) -> dict[str, Any]:
    if contract is None:
        return {}
    return {
        "con_id": getattr(contract, "conId", None),
        "symbol": getattr(contract, "symbol", ""),
        "sec_type": getattr(contract, "secType", ""),
        "exchange": getattr(contract, "exchange", ""),
        "primary_exchange": getattr(contract, "primaryExchange", ""),
        "currency": getattr(contract, "currency", ""),
        "local_symbol": getattr(contract, "localSymbol", ""),
        "trading_class": getattr(contract, "tradingClass", ""),
    }


def account_summary_frame(values: list[Any]) -> pd.DataFrame:
    rows = [
        {
            "account": str(getattr(value, "account", "")),
            "tag": str(getattr(value, "tag", "")),
            "value": str(getattr(value, "value", "")),
            "currency": str(getattr(value, "currency", "")),
            "model_code": str(getattr(value, "modelCode", "")),
        }
        for value in values
    ]
    return pd.DataFrame(rows, columns=["account", "tag", "value", "currency", "model_code"])


def positions_frame(values: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in values:
        contract = _contract_dict(getattr(value, "contract", None))
        rows.append(
            {
                "account": str(getattr(value, "account", "")),
                **contract,
                "position": float(getattr(value, "position", 0.0)),
                "avg_cost": float(getattr(value, "avgCost", 0.0)),
            }
        )
    columns = [
        "account",
        "con_id",
        "symbol",
        "sec_type",
        "exchange",
        "primary_exchange",
        "currency",
        "local_symbol",
        "trading_class",
        "position",
        "avg_cost",
    ]
    return pd.DataFrame(rows, columns=columns)


def open_trades_frame(values: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in values:
        contract = _contract_dict(getattr(trade, "contract", None))
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        rows.append(
            {
                **contract,
                "order_id": getattr(order, "orderId", None),
                "client_id": getattr(order, "clientId", None),
                "action": getattr(order, "action", ""),
                "order_type": getattr(order, "orderType", ""),
                "total_quantity": getattr(order, "totalQuantity", None),
                "lmt_price": getattr(order, "lmtPrice", None),
                "aux_price": getattr(order, "auxPrice", None),
                "status": getattr(status, "status", ""),
                "filled": getattr(status, "filled", None),
                "remaining": getattr(status, "remaining", None),
                "avg_fill_price": getattr(status, "avgFillPrice", None),
            }
        )
    columns = [
        "con_id",
        "symbol",
        "sec_type",
        "exchange",
        "primary_exchange",
        "currency",
        "local_symbol",
        "trading_class",
        "order_id",
        "client_id",
        "action",
        "order_type",
        "total_quantity",
        "lmt_price",
        "aux_price",
        "status",
        "filled",
        "remaining",
        "avg_fill_price",
    ]
    return pd.DataFrame(rows, columns=columns)


class IBKRReadOnlyClient:
    def __init__(self, config: IBKRReadOnlyConfig, ib_factory: Callable[[], Any] | None = None) -> None:
        config.validate()
        self.config = config
        self._ib_factory = ib_factory or _require_ib_insync_factory()
        self.ib: Any | None = None

    def connect(self) -> None:
        if self.ib is None:
            self.ib = self._ib_factory()
        self.ib.connect(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            timeout=self.config.timeout_seconds,
            readonly=True,
        )

    def disconnect(self) -> None:
        if self.ib is not None and getattr(self.ib, "isConnected", lambda: False)():
            self.ib.disconnect()

    def health_check(self) -> dict[str, Any]:
        if self.ib is None or not getattr(self.ib, "isConnected", lambda: False)():
            raise RuntimeError("IBKR client is not connected")
        accounts = list(self.ib.managedAccounts())
        if self.config.require_paper_account and self.config.expected_account and self.config.expected_account not in accounts:
            raise RuntimeError(f"expected paper account {self.config.expected_account} not found in managed accounts")
        current_time = self.ib.reqCurrentTime()
        return {
            "connected": True,
            "host": self.config.host,
            "port": self.config.port,
            "client_id": self.config.client_id,
            "trading_mode": self.config.trading_mode,
            "read_only": self.config.read_only,
            "allow_orders": self.config.allow_orders,
            "managed_accounts": accounts,
            "server_time": str(current_time),
        }

    def snapshot(self) -> dict[str, Any]:
        health = self.health_check()
        account_summary = account_summary_frame(list(self.ib.accountSummary()))
        positions = positions_frame(list(self.ib.positions()))
        open_trades = open_trades_frame(list(self.ib.openTrades()))
        return {
            "created_at_utc": utc_now(),
            "health": health,
            "account_summary": account_summary,
            "positions": positions,
            "open_trades": open_trades,
        }


def write_snapshot(snapshot: dict[str, Any], config: IBKRReadOnlyConfig, output_dir: str | Path | None = None) -> IBKRReadOnlySnapshotPaths:
    created = str(snapshot["created_at_utc"]).replace(":", "").replace("-", "")
    root = Path(output_dir) if output_dir is not None else config.output_dir
    run_dir = root / created
    paths = IBKRReadOnlySnapshotPaths(
        output_dir=run_dir,
        manifest_path=run_dir / "manifest.yaml",
        account_summary_path=run_dir / "account_summary.parquet",
        positions_path=run_dir / "positions.parquet",
        open_trades_path=run_dir / "open_trades.parquet",
        report_path=run_dir / "report.md",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot["account_summary"].to_parquet(paths.account_summary_path, index=False)
    snapshot["positions"].to_parquet(paths.positions_path, index=False)
    snapshot["open_trades"].to_parquet(paths.open_trades_path, index=False)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "ibkr_read_only_snapshot",
            "created_at_utc": snapshot["created_at_utc"],
            "status": "complete",
        },
        "connection": config.to_dict(),
        "health": snapshot["health"],
        "outputs": {
            "account_summary": paths.account_summary_path.as_posix(),
            "positions": paths.positions_path.as_posix(),
            "open_trades": paths.open_trades_path.as_posix(),
            "report": paths.report_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    report = [
        "# IBKR read-only snapshot",
        "",
        f"- Created UTC: `{snapshot['created_at_utc']}`",
        f"- Host/port: `{config.host}:{config.port}`",
        f"- Client ID: `{config.client_id}`",
        f"- Managed accounts: `{', '.join(snapshot['health']['managed_accounts'])}`",
        f"- Account summary rows: `{len(snapshot['account_summary'])}`",
        f"- Positions: `{len(snapshot['positions'])}`",
        f"- Open trades: `{len(snapshot['open_trades'])}`",
        "",
        "This snapshot is read-only. No orders are submitted by this command.",
        "",
    ]
    paths.report_path.write_text("\n".join(report), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IBKR Gateway paper read-only connection check and snapshot")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--validate-only", action="store_true", help="validate config without connecting")
    parser.add_argument("--snapshot", action="store_true", help="write account/positions/open-trades snapshot")
    args = parser.parse_args(argv)

    config = load_ibkr_read_only_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "valid", "config": config.to_dict()}, indent=2, sort_keys=True))
        return

    client = IBKRReadOnlyClient(config)
    try:
        client.connect()
        if args.snapshot:
            snapshot = client.snapshot()
            paths = write_snapshot(snapshot, config)
            print(json.dumps({key: str(value) for key, value in asdict(paths).items()}, indent=2, sort_keys=True))
        else:
            print(json.dumps(client.health_check(), indent=2, sort_keys=True))
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

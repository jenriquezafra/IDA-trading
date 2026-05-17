from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.execution.ibkr_read_only import DEFAULT_CONFIG_PATH, load_ibkr_read_only_config


DEFAULT_SNAPSHOT_ROOT = Path("results/paper/ibkr_read_only")
DEFAULT_OUTPUT_DIR = Path("results/paper/flatten_plan")
SUPPORTED_SEC_TYPES = {"STK"}
SUPPORTED_ORDER_TYPES = {"MKT"}
SUPPORTED_TIFS = {"DAY", "OPG"}


@dataclass(frozen=True)
class ExecutionPolicy:
    order_type: str = "MKT"
    tif: str = "DAY"
    outside_rth: bool = False
    routing_exchange: str = "SMART"
    intent: str = "flatten_now"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None = None) -> "ExecutionPolicy":
        data = dict(raw or {})
        policy = cls(
            order_type=str(data.get("order_type", "MKT")).strip().upper(),
            tif=str(data.get("tif", "DAY")).strip().upper(),
            outside_rth=bool(data.get("outside_rth", False)),
            routing_exchange=str(data.get("routing_exchange", "SMART")).strip().upper(),
            intent=str(data.get("intent", "flatten_now")).strip(),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.order_type not in SUPPORTED_ORDER_TYPES:
            raise ValueError(f"unsupported order_type={self.order_type}; supported={sorted(SUPPORTED_ORDER_TYPES)}")
        if self.tif not in SUPPORTED_TIFS:
            raise ValueError(f"unsupported tif={self.tif}; supported={sorted(SUPPORTED_TIFS)}")
        if self.order_type == "MKT" and self.tif not in {"DAY", "OPG"}:
            raise ValueError("MKT flatten orders support tif DAY or OPG")
        if self.outside_rth and self.tif == "OPG":
            raise ValueError("OPG opening orders must use outside_rth=false")
        if not self.routing_exchange:
            raise ValueError("routing_exchange is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlattenPlanPaths:
    output_dir: Path
    manifest_path: Path
    orders_path: Path
    orders_csv_path: Path
    skipped_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_latest_snapshot(root: str | Path = DEFAULT_SNAPSHOT_ROOT) -> Path:
    snapshot_root = Path(root)
    if not snapshot_root.exists():
        raise FileNotFoundError(f"snapshot root does not exist: {snapshot_root}")
    candidates = [path for path in snapshot_root.iterdir() if path.is_dir() and (path / "manifest.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"no IBKR read-only snapshots found under {snapshot_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def load_snapshot_manifest(snapshot_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(snapshot_dir) / "manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"expected YAML mapping: {manifest_path}")
    return manifest


def _read_parquet(snapshot_dir: Path, filename: str) -> pd.DataFrame:
    path = snapshot_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"snapshot file not found: {path}")
    return pd.read_parquet(path)


def _nonzero_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    required = {"account", "symbol", "sec_type", "currency", "position", "avg_cost"}
    missing = sorted(required.difference(positions.columns))
    if missing:
        raise ValueError(f"positions snapshot is missing columns: {missing}")
    out = positions.copy()
    out["position"] = out["position"].astype(float)
    return out[out["position"] != 0].copy()


def _planned_route(row: pd.Series) -> str:
    sec_type = str(row.get("sec_type", "")).upper()
    if sec_type == "STK":
        return "SMART"
    return str(row.get("exchange", "") or "")


def build_flatten_plan(
    positions: pd.DataFrame,
    open_trades: pd.DataFrame,
    *,
    expected_account: str = "",
    execution_policy: ExecutionPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    policy = execution_policy or ExecutionPolicy()
    nonzero = _nonzero_positions(positions)
    if expected_account and not nonzero.empty:
        accounts = set(nonzero["account"].astype(str))
        if accounts != {expected_account}:
            raise ValueError(f"snapshot accounts {sorted(accounts)} do not match expected account {expected_account}")

    unsupported_mask = ~nonzero["sec_type"].astype(str).str.upper().isin(SUPPORTED_SEC_TYPES) if not nonzero.empty else pd.Series(dtype=bool)
    unsupported = nonzero[unsupported_mask].copy() if not nonzero.empty else nonzero.copy()
    supported = nonzero[~unsupported_mask].copy() if not nonzero.empty else nonzero.copy()

    open_trade_count = int(len(open_trades))
    blocked_by_open_trades = open_trade_count > 0
    orders: list[dict[str, Any]] = []
    for _, row in supported.iterrows():
        position = float(row["position"])
        quantity = abs(position)
        action = "SELL" if position > 0 else "BUY"
        avg_cost = float(row.get("avg_cost", 0.0) or 0.0)
        orders.append(
            {
                "account": str(row.get("account", "")),
                "con_id": row.get("con_id", None),
                "symbol": str(row.get("symbol", "")),
                "sec_type": str(row.get("sec_type", "")),
                "currency": str(row.get("currency", "")),
                "local_symbol": str(row.get("local_symbol", "")),
                "trading_class": str(row.get("trading_class", "")),
                "current_position": position,
                "action": action,
                "quantity": quantity,
                "order_type": policy.order_type,
                "tif": policy.tif,
                "outside_rth": policy.outside_rth,
                "routing_exchange": policy.routing_exchange if str(row.get("sec_type", "")).upper() == "STK" else _planned_route(row),
                "primary_exchange": str(row.get("primary_exchange", "")),
                "transmit": False,
                "dry_run": True,
                "approx_notional_at_avg_cost": quantity * avg_cost,
                "status": "blocked_open_trades" if blocked_by_open_trades else "planned",
            }
        )

    order_columns = [
        "account",
        "con_id",
        "symbol",
        "sec_type",
        "currency",
        "local_symbol",
        "trading_class",
        "current_position",
        "action",
        "quantity",
        "order_type",
        "tif",
        "outside_rth",
        "routing_exchange",
        "primary_exchange",
        "transmit",
        "dry_run",
        "approx_notional_at_avg_cost",
        "status",
    ]
    orders_df = pd.DataFrame(orders, columns=order_columns)
    skipped_columns = list(nonzero.columns) + ["skip_reason"]
    skipped_rows: list[dict[str, Any]] = []
    for _, row in unsupported.iterrows():
        item = row.to_dict()
        item["skip_reason"] = "unsupported_sec_type"
        skipped_rows.append(item)
    skipped_df = pd.DataFrame(skipped_rows, columns=skipped_columns)

    has_unsupported = not skipped_df.empty
    if blocked_by_open_trades:
        decision = "blocked_open_trades"
    elif has_unsupported:
        decision = "blocked_unsupported_instruments"
    elif orders_df.empty:
        decision = "already_flat"
    else:
        decision = "ready_for_review"

    summary = {
        "decision": decision,
        "positions_seen": int(len(positions)),
        "nonzero_positions": int(len(nonzero)),
        "supported_positions": int(len(supported)),
        "unsupported_positions": int(len(unsupported)),
        "open_trades": open_trade_count,
        "planned_orders": int(len(orders_df)),
        "dry_run": True,
        "transmit": False,
        "supported_sec_types": sorted(SUPPORTED_SEC_TYPES),
        "execution_policy": policy.to_dict(),
    }
    return orders_df, skipped_df, summary


def load_execution_policy(path: str | Path | None) -> ExecutionPolicy:
    if path is None:
        return ExecutionPolicy()
    policy_path = Path(path)
    with policy_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {policy_path}")
    policy_raw = raw.get("execution_policy", raw)
    if not isinstance(policy_raw, dict):
        raise ValueError(f"expected execution_policy mapping: {policy_path}")
    return ExecutionPolicy.from_mapping(policy_raw)


def _snapshot_created_at(manifest: dict[str, Any]) -> str:
    run = manifest.get("run", {})
    if isinstance(run, dict):
        return str(run.get("created_at_utc", ""))
    return ""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    data = frame[columns].copy()
    rows = [[str(value) for value in row] for row in data.to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_flatten_plan(
    *,
    snapshot_dir: str | Path,
    orders: pd.DataFrame,
    skipped: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> FlattenPlanPaths:
    created = utc_now()
    run_name = created.replace(":", "").replace("-", "")
    root = Path(output_dir) / run_name
    paths = FlattenPlanPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        orders_path=root / "orders.parquet",
        orders_csv_path=root / "orders.csv",
        skipped_path=root / "skipped_positions.parquet",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(paths.orders_path, index=False)
    orders.to_csv(paths.orders_csv_path, index=False)
    skipped.to_parquet(paths.skipped_path, index=False)

    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "ibkr_flatten_plan",
            "created_at_utc": created,
            "status": "complete",
        },
        "source": {
            "snapshot_dir": Path(snapshot_dir).as_posix(),
        },
        "summary": summary,
        "outputs": {
            "orders": paths.orders_path.as_posix(),
            "orders_csv": paths.orders_csv_path.as_posix(),
            "skipped_positions": paths.skipped_path.as_posix(),
            "report": paths.report_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = [
        "# IBKR flatten plan",
        "",
        f"- Created UTC: `{created}`",
        f"- Source snapshot: `{Path(snapshot_dir).as_posix()}`",
        f"- Decision: `{summary['decision']}`",
        f"- Nonzero positions: `{summary['nonzero_positions']}`",
        f"- Planned orders: `{summary['planned_orders']}`",
        f"- Open trades in snapshot: `{summary['open_trades']}`",
        f"- Unsupported positions: `{summary['unsupported_positions']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Transmit: `{summary['transmit']}`",
        f"- Order type: `{summary['execution_policy']['order_type']}`",
        f"- TIF: `{summary['execution_policy']['tif']}`",
        f"- Outside RTH: `{summary['execution_policy']['outside_rth']}`",
        f"- Intent: `{summary['execution_policy']['intent']}`",
        "",
        "This plan is offline and does not submit orders. Every planned ticket has `transmit=false`.",
        "",
    ]
    if not orders.empty:
        display_columns = ["symbol", "sec_type", "current_position", "action", "quantity", "order_type", "tif", "outside_rth", "routing_exchange", "status"]
        report.extend(["## Planned tickets", "", _markdown_table(orders, display_columns), ""])
    if not skipped.empty:
        report.extend(["## Skipped positions", "", _markdown_table(skipped, list(skipped.columns)), ""])
    paths.report_path.write_text("\n".join(report), encoding="utf-8")
    return paths


def create_flatten_plan(
    *,
    snapshot_dir: str | Path | None = None,
    snapshot_root: str | Path = DEFAULT_SNAPSHOT_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    execution_policy_path: str | Path | None = None,
    execution_policy: ExecutionPolicy | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[FlattenPlanPaths, dict[str, Any]]:
    config = load_ibkr_read_only_config(config_path)
    selected_policy = execution_policy or load_execution_policy(execution_policy_path)
    selected_snapshot = Path(snapshot_dir) if snapshot_dir is not None else find_latest_snapshot(snapshot_root)
    manifest = load_snapshot_manifest(selected_snapshot)
    managed_accounts = set(manifest.get("health", {}).get("managed_accounts", []))
    if config.expected_account and config.expected_account not in managed_accounts:
        raise ValueError(f"snapshot does not include expected account {config.expected_account}")
    positions = _read_parquet(selected_snapshot, "positions.parquet")
    open_trades = _read_parquet(selected_snapshot, "open_trades.parquet")
    orders, skipped, summary = build_flatten_plan(positions, open_trades, expected_account=config.expected_account, execution_policy=selected_policy)
    summary = {
        **summary,
        "snapshot_created_at_utc": _snapshot_created_at(manifest),
        "expected_account": config.expected_account,
    }
    paths = write_flatten_plan(snapshot_dir=selected_snapshot, orders=orders, skipped=skipped, summary=summary, output_dir=output_dir)
    return paths, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an offline IBKR paper flatten plan from a read-only snapshot")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--snapshot-dir", default=None, help="specific read-only snapshot directory; latest is used by default")
    parser.add_argument("--snapshot-root", default=str(DEFAULT_SNAPSHOT_ROOT), help="root containing read-only snapshots")
    parser.add_argument("--execution-policy", default=None, help="YAML file with execution_policy for planned tickets")
    parser.add_argument("--order-type", default=None, help="override execution policy order_type, e.g. MKT")
    parser.add_argument("--tif", default=None, help="override execution policy tif, e.g. DAY or OPG")
    parser.add_argument("--outside-rth", action="store_true", help="override execution policy outside_rth=true")
    parser.add_argument("--intent", default=None, help="override execution policy intent label")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    execution_policy_path = args.execution_policy
    if any(value is not None for value in [args.order_type, args.tif, args.intent]) or args.outside_rth:
        base_policy = load_execution_policy(execution_policy_path)
        execution_policy = ExecutionPolicy.from_mapping(
            {
                **base_policy.to_dict(),
                **({"order_type": args.order_type} if args.order_type is not None else {}),
                **({"tif": args.tif} if args.tif is not None else {}),
                **({"outside_rth": True} if args.outside_rth else {}),
                **({"intent": args.intent} if args.intent is not None else {}),
            }
        )
    else:
        execution_policy = None

    paths, summary = create_flatten_plan(
        snapshot_dir=args.snapshot_dir,
        snapshot_root=args.snapshot_root,
        config_path=args.config,
        execution_policy_path=execution_policy_path,
        execution_policy=execution_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps({"summary": summary, "paths": {key: str(value) for key, value in asdict(paths).items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

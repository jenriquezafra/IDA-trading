from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.execution.ibkr_read_only import (
    IBKRReadOnlyClient,
    IBKRReadOnlySnapshotPaths,
    load_ibkr_read_only_config,
    open_trades_frame,
    write_snapshot,
)
from src.execution.paper_state_store import load_state, load_state_config


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_reconcile_h1c.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_reconciliation")

POSITION_COLUMNS = [
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
OPEN_TRADE_COLUMNS = [
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
EXECUTION_COLUMNS = [
    "account",
    "symbol",
    "sec_type",
    "currency",
    "exec_id",
    "time",
    "order_id",
    "perm_id",
    "side",
    "shares",
    "price",
    "commission",
    "realized_pnl",
]


@dataclass(frozen=True)
class H1CReconciliationConfig:
    strategy_id: str
    account: str
    symbol: str
    position_side: str
    state_config_path: Path
    ibkr_config_path: Path
    position_tolerance: float
    allow_unrelated_open_orders: bool
    request_all_open_orders: bool
    include_executions: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "H1CReconciliationConfig":
        reconciliation = dict(raw.get("reconciliation", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            strategy_id=str(reconciliation.get("strategy_id", "")).strip(),
            account=str(reconciliation.get("account", "")).strip(),
            symbol=str(reconciliation.get("symbol", "QQQ")).strip().upper(),
            position_side=str(reconciliation.get("position_side", "short")).strip().lower(),
            state_config_path=Path(reconciliation.get("state_config_path", "configs/execution/paper_state_h1c.yaml")),
            ibkr_config_path=Path(reconciliation.get("ibkr_config_path", "configs/execution/ibkr_paper_readonly.yaml")),
            position_tolerance=float(reconciliation.get("position_tolerance", 1e-6)),
            allow_unrelated_open_orders=bool(reconciliation.get("allow_unrelated_open_orders", False)),
            request_all_open_orders=bool(reconciliation.get("request_all_open_orders", True)),
            include_executions=bool(reconciliation.get("include_executions", True)),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("reconciliation.strategy_id is required")
        if not self.account:
            raise ValueError("reconciliation.account is required")
        if not self.symbol:
            raise ValueError("reconciliation.symbol is required")
        if self.position_side not in {"long", "short"}:
            raise ValueError("reconciliation.position_side must be long or short")
        if self.position_tolerance < 0.0:
            raise ValueError("position_tolerance must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state_config_path"] = self.state_config_path.as_posix()
        data["ibkr_config_path"] = self.ibkr_config_path.as_posix()
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class H1CReconciliationPaths:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    state_snapshot_path: Path
    positions_path: Path
    open_trades_path: Path
    executions_path: Path
    ibkr_snapshot_dir: Path | None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_reconciliation_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H1CReconciliationConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return H1CReconciliationConfig.from_mapping(raw)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy() if frame is not None else _empty_frame(columns)
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns].copy()


def _nonzero_positions(positions: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    positions = _ensure_columns(positions, POSITION_COLUMNS)
    if positions.empty:
        return positions
    numeric = pd.to_numeric(positions["position"], errors="coerce").fillna(0.0)
    return positions[numeric.abs() > tolerance].copy()


def _target_positions(positions: pd.DataFrame, config: H1CReconciliationConfig) -> pd.DataFrame:
    nonzero = _nonzero_positions(positions, config.position_tolerance)
    if nonzero.empty:
        return nonzero
    return nonzero[
        nonzero["account"].astype(str).eq(config.account)
        & nonzero["symbol"].astype(str).str.upper().eq(config.symbol)
        & nonzero["sec_type"].astype(str).str.upper().eq("STK")
    ].copy()


def _account_open_trades(open_trades: pd.DataFrame, account: str) -> pd.DataFrame:
    trades = _ensure_columns(open_trades, OPEN_TRADE_COLUMNS)
    if trades.empty:
        return trades
    active_statuses = {"PENDINGSUBMIT", "PRESUBMITTED", "SUBMITTED", "APIPENDING", "PENDINGCANCEL"}
    status = trades["status"].astype(str).str.upper()
    remaining = pd.to_numeric(trades["remaining"], errors="coerce")
    active = status.isin(active_statuses) | remaining.fillna(0.0).gt(0.0)
    out = trades[active].copy()
    if "account" in out.columns:
        out = out[out["account"].astype(str).isin(["", account])].copy()
    return out


def _target_open_trades(open_trades: pd.DataFrame, config: H1CReconciliationConfig) -> pd.DataFrame:
    trades = _account_open_trades(open_trades, config.account)
    if trades.empty:
        return trades
    return trades[
        trades["symbol"].astype(str).str.upper().eq(config.symbol)
        & trades["sec_type"].astype(str).str.upper().eq("STK")
    ].copy()


def _position_matches_side(position_qty: float, config: H1CReconciliationConfig) -> bool:
    if config.position_side == "long":
        return position_qty > config.position_tolerance
    return position_qty < -config.position_tolerance


def _side_position_abs_diff(position_qty: float, expected_quantity: float) -> float:
    return abs(abs(position_qty) - abs(expected_quantity))


def _entry_action(config: H1CReconciliationConfig) -> str:
    return "BUY" if config.position_side == "long" else "SELL"


def _exit_action(config: H1CReconciliationConfig) -> str:
    return "SELL" if config.position_side == "long" else "BUY"


def _execution_contract(fill: Any) -> dict[str, Any]:
    contract = getattr(fill, "contract", None)
    return {
        "symbol": getattr(contract, "symbol", ""),
        "sec_type": getattr(contract, "secType", ""),
        "currency": getattr(contract, "currency", ""),
    }


def executions_frame(values: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fill in values:
        execution = getattr(fill, "execution", fill)
        commission = getattr(fill, "commissionReport", None)
        rows.append(
            {
                "account": str(getattr(execution, "acctNumber", "")),
                **_execution_contract(fill),
                "exec_id": getattr(execution, "execId", ""),
                "time": str(getattr(execution, "time", "")),
                "order_id": getattr(execution, "orderId", None),
                "perm_id": getattr(execution, "permId", None),
                "side": getattr(execution, "side", ""),
                "shares": getattr(execution, "shares", None),
                "price": getattr(execution, "price", None),
                "commission": getattr(commission, "commission", None),
                "realized_pnl": getattr(commission, "realizedPNL", None),
            }
        )
    return pd.DataFrame(rows, columns=EXECUTION_COLUMNS)


def _collect_ibkr_snapshot(
    config: H1CReconciliationConfig,
    *,
    ib_factory: Callable[[], Any] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], IBKRReadOnlySnapshotPaths]:
    ibkr_config = load_ibkr_read_only_config(config.ibkr_config_path)
    client = IBKRReadOnlyClient(ibkr_config, ib_factory=ib_factory)
    try:
        client.connect()
        if config.request_all_open_orders and client.ib is not None and hasattr(client.ib, "reqAllOpenOrders"):
            all_open_orders = list(client.ib.reqAllOpenOrders())
        else:
            all_open_orders = []
        snapshot = client.snapshot()
        if all_open_orders:
            snapshot["open_trades"] = open_trades_frame(all_open_orders)
        if config.include_executions and client.ib is not None and hasattr(client.ib, "reqExecutions"):
            snapshot["executions"] = executions_frame(list(client.ib.reqExecutions()))
        else:
            snapshot["executions"] = _empty_frame(EXECUTION_COLUMNS)
        paths = write_snapshot(snapshot, ibkr_config, output_dir=output_dir)
        return snapshot, paths
    finally:
        client.disconnect()


def _load_snapshot_dir(snapshot_dir: str | Path) -> tuple[dict[str, Any], Path]:
    root = Path(snapshot_dir)
    positions_path = root / "positions.parquet"
    open_trades_path = root / "open_trades.parquet"
    manifest_path = root / "manifest.yaml"
    if not positions_path.exists():
        raise FileNotFoundError(f"snapshot missing positions.parquet: {root}")
    if not open_trades_path.exists():
        raise FileNotFoundError(f"snapshot missing open_trades.parquet: {root}")
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
    executions_path = root / "executions.parquet"
    return {
        "created_at_utc": manifest.get("run", {}).get("created_at_utc", ""),
        "health": manifest.get("health", {}),
        "positions": pd.read_parquet(positions_path),
        "open_trades": pd.read_parquet(open_trades_path),
        "executions": pd.read_parquet(executions_path) if executions_path.exists() else _empty_frame(EXECUTION_COLUMNS),
    }, root


def reconcile_state_snapshot(
    *,
    state: dict[str, Any],
    positions: pd.DataFrame,
    open_trades: pd.DataFrame,
    config: H1CReconciliationConfig,
) -> dict[str, Any]:
    if state.get("strategy_id") != config.strategy_id:
        raise ValueError("state strategy_id does not match reconciliation config")
    if state.get("account") != config.account:
        raise ValueError("state account does not match reconciliation config")
    if str(state.get("symbol", "")).upper() != config.symbol:
        raise ValueError("state symbol does not match reconciliation config")

    account_positions = _nonzero_positions(positions, config.position_tolerance)
    if not account_positions.empty:
        account_positions = account_positions[account_positions["account"].astype(str).eq(config.account)].copy()
    target_positions = _target_positions(positions, config)
    account_trades = _account_open_trades(open_trades, config.account)
    target_trades = _target_open_trades(open_trades, config)
    unrelated_trades = account_trades[~account_trades.index.isin(target_trades.index)].copy()

    target_position_qty = float(pd.to_numeric(target_positions["position"], errors="coerce").fillna(0.0).sum()) if not target_positions.empty else 0.0
    target_order_count = int(len(target_trades))
    unrelated_order_count = int(len(unrelated_trades))
    status = str(state.get("status", "flat"))
    expected_quantity = float(state.get("quantity", 0.0) or 0.0)
    side_label = config.position_side
    entry_action = _entry_action(config)
    exit_action = _exit_action(config)

    decision = "UNKNOWN_IBKR_STATE"
    severity = "error"
    reason = "state/open-order/position combination is not explicitly handled"
    state_transition_hint = "manual_review"

    if unrelated_order_count > 0 and not config.allow_unrelated_open_orders:
        decision = "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS"
        severity = "block"
        reason = "IBKR account has open orders outside the target strategy symbol"
        state_transition_hint = "pause_strategy_until_account_clean"
    elif status == "flat":
        if abs(target_position_qty) > config.position_tolerance:
            decision = "DRIFT_POSITION_MISMATCH"
            severity = "block"
            reason = "state is flat but IBKR has a nonzero target position"
            state_transition_hint = "manual_reconcile_position"
        elif target_order_count > 0:
            decision = "OPEN_ORDER_WITHOUT_PENDING_TICKET"
            severity = "block"
            reason = "state is flat but IBKR has active target orders"
            state_transition_hint = "cancel_or_adopt_order"
        else:
            decision = "OK_FLAT"
            severity = "ok"
            reason = "state is flat and IBKR has no target position or target open order"
            state_transition_hint = "no_change"
    elif status == "pending_entry":
        if target_order_count > 0:
            decision = "OK_PENDING_ENTRY"
            severity = "ok"
            reason = "state has pending entry and IBKR has an active target order"
            state_transition_hint = "wait_for_fill"
        elif _position_matches_side(target_position_qty, config):
            decision = "FILL_DETECTED_PENDING_ENTRY"
            severity = "action_required"
            reason = f"pending entry appears filled because IBKR has a {side_label} target position"
            state_transition_hint = "mark_open_after_fill_accounting"
        else:
            decision = "PENDING_TICKET_WITHOUT_OPEN_ORDER"
            severity = "block"
            reason = "state has pending entry but IBKR has no active target order or target position"
            state_transition_hint = "manual_review_or_reset_pending_ticket"
    elif status == "open":
        if _position_matches_side(target_position_qty, config):
            if expected_quantity > 0.0 and _side_position_abs_diff(target_position_qty, expected_quantity) > config.position_tolerance:
                decision = "DRIFT_POSITION_MISMATCH"
                severity = "block"
                reason = "state is open but IBKR target quantity differs from state quantity"
                state_transition_hint = "manual_reconcile_quantity"
            else:
                decision = "OK_OPEN"
                severity = "ok"
                reason = f"state is open and IBKR has the expected {side_label} target position"
                state_transition_hint = "monitor_exit"
        else:
            decision = "DRIFT_POSITION_MISMATCH"
            severity = "block"
            reason = f"state is open but IBKR has no {side_label} target position"
            state_transition_hint = "manual_reconcile_position"
    elif status == "pending_exit":
        exit_orders = target_trades[target_trades["action"].astype(str).str.upper().eq(exit_action)]
        if _position_matches_side(target_position_qty, config) and not exit_orders.empty:
            decision = "OK_PENDING_EXIT"
            severity = "ok"
            reason = f"state has pending exit and IBKR has active target {exit_action.lower()} order"
            state_transition_hint = "wait_for_exit_fill"
        elif abs(target_position_qty) <= config.position_tolerance and exit_orders.empty:
            decision = "FILL_DETECTED_PENDING_EXIT"
            severity = "action_required"
            reason = f"pending exit appears filled because IBKR has no target position or target {exit_action.lower()} order"
            state_transition_hint = "mark_flat_after_exit_accounting"
        elif _position_matches_side(target_position_qty, config) and exit_orders.empty:
            decision = "PENDING_EXIT_WITHOUT_OPEN_ORDER"
            severity = "block"
            reason = f"state has pending exit but IBKR still has the {side_label} target position and no active {exit_action.lower()} order"
            state_transition_hint = "manual_review_or_resubmit_exit"
        else:
            decision = "UNKNOWN_IBKR_STATE"
            severity = "block"
            reason = "pending exit does not match IBKR target position/order state"
            state_transition_hint = "manual_review"

    return {
        "decision": decision,
        "severity": severity,
        "reason": reason,
        "state_transition_hint": state_transition_hint,
        "state_status": status,
        "target_position_qty": target_position_qty,
        "target_open_orders": target_order_count,
        "entry_action": entry_action,
        "exit_action": exit_action,
        "position_side": config.position_side,
        "account_nonzero_positions": int(len(account_positions)),
        "account_open_orders": int(len(account_trades)),
        "unrelated_open_orders": unrelated_order_count,
        "expected_state_quantity": expected_quantity,
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    result = manifest["reconciliation"]
    lines = [
        "# H1c paper reconciliation",
        "",
        f"- Created UTC: `{manifest['run']['created_at_utc']}`",
        f"- Decision: `{result['decision']}`",
        f"- Severity: `{result['severity']}`",
        f"- Reason: `{result['reason']}`",
        f"- State status: `{result['state_status']}`",
        f"- Target position qty: `{result['target_position_qty']}`",
        f"- Target open orders: `{result['target_open_orders']}`",
        f"- Account open orders: `{result['account_open_orders']}`",
        f"- Unrelated open orders: `{result['unrelated_open_orders']}`",
        f"- Transition hint: `{result['state_transition_hint']}`",
        "",
        "## Outputs",
        "",
        f"- State snapshot: `{manifest['outputs']['state_snapshot']}`",
        f"- Positions: `{manifest['outputs']['positions']}`",
        f"- Open trades: `{manifest['outputs']['open_trades']}`",
        f"- Executions: `{manifest['outputs']['executions']}`",
        f"- IBKR snapshot dir: `{manifest['outputs'].get('ibkr_snapshot_dir') or ''}`",
        "",
        "This reconciliation is read-only. It does not submit, cancel, or modify orders.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_h1c_reconciliation(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    snapshot_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ib_factory: Callable[[], Any] | None = None,
) -> tuple[H1CReconciliationPaths, dict[str, Any]]:
    config = load_reconciliation_config(config_path)
    state_config = load_state_config(config.state_config_path)
    state = load_state(state_config)

    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    root.mkdir(parents=True, exist_ok=True)

    if snapshot_dir is None:
        snapshot, ibkr_snapshot_paths = _collect_ibkr_snapshot(config, ib_factory=ib_factory, output_dir=root / "ibkr_snapshot")
        ibkr_snapshot_dir: Path | None = ibkr_snapshot_paths.output_dir
    else:
        snapshot, loaded_snapshot_dir = _load_snapshot_dir(snapshot_dir)
        ibkr_snapshot_dir = loaded_snapshot_dir

    positions = _ensure_columns(snapshot.get("positions", _empty_frame(POSITION_COLUMNS)), POSITION_COLUMNS)
    open_trades = _ensure_columns(snapshot.get("open_trades", _empty_frame(OPEN_TRADE_COLUMNS)), OPEN_TRADE_COLUMNS)
    executions = _ensure_columns(snapshot.get("executions", _empty_frame(EXECUTION_COLUMNS)), EXECUTION_COLUMNS)
    result = reconcile_state_snapshot(state=state, positions=positions, open_trades=open_trades, config=config)

    paths = H1CReconciliationPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
        state_snapshot_path=root / "state_snapshot.yaml",
        positions_path=root / "positions.parquet",
        open_trades_path=root / "open_trades.parquet",
        executions_path=root / "executions.parquet",
        ibkr_snapshot_dir=ibkr_snapshot_dir,
    )
    paths.state_snapshot_path.write_text(yaml.safe_dump(state, sort_keys=False), encoding="utf-8")
    positions.to_parquet(paths.positions_path, index=False)
    open_trades.to_parquet(paths.open_trades_path, index=False)
    executions.to_parquet(paths.executions_path, index=False)

    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "h1c_paper_reconciliation",
            "created_at_utc": created,
            "status": "complete",
        },
        "config": config.to_dict(),
        "state": {
            "status": state.get("status"),
            "quantity": state.get("quantity"),
            "desired_position_unit": state.get("desired_position_unit"),
            "last_signal_timestamp": state.get("last_signal_timestamp"),
        },
        "ibkr": {
            "snapshot_created_at_utc": snapshot.get("created_at_utc", ""),
            "health": snapshot.get("health", {}),
        },
        "reconciliation": result,
        "outputs": {
            "manifest": paths.manifest_path.as_posix(),
            "report": paths.report_path.as_posix(),
            "state_snapshot": paths.state_snapshot_path.as_posix(),
            "positions": paths.positions_path.as_posix(),
            "open_trades": paths.open_trades_path.as_posix(),
            "executions": paths.executions_path.as_posix(),
            "ibkr_snapshot_dir": None if paths.ibkr_snapshot_dir is None else paths.ibkr_snapshot_dir.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, manifest)
    return paths, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only reconciliation for the H1c paper strategy state vs IBKR")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--snapshot-dir", default=None, help="optional existing IBKR read-only snapshot directory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, manifest = run_h1c_reconciliation(config_path=args.config, snapshot_dir=args.snapshot_dir, output_dir=args.output_dir)
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": manifest}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

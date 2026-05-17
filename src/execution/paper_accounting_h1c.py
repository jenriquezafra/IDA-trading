from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.execution.paper_state_store import append_event, load_state, load_state_config, validate_state, write_state


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_accounting_h1c.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_accounting")


@dataclass(frozen=True)
class H1CAccountingConfig:
    strategy_id: str
    account: str
    symbol: str
    state_config_path: Path
    pnl_log_path: Path
    allow_open_from_position_without_execution: bool
    allow_flatten_without_pending_exit: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "H1CAccountingConfig":
        accounting = dict(raw.get("accounting", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            strategy_id=str(accounting.get("strategy_id", "")).strip(),
            account=str(accounting.get("account", "")).strip(),
            symbol=str(accounting.get("symbol", "QQQ")).strip().upper(),
            state_config_path=Path(accounting.get("state_config_path", "configs/execution/paper_state_h1c.yaml")),
            pnl_log_path=Path(accounting.get("pnl_log_path", "results/paper/h1c_state/pnl_events.parquet")),
            allow_open_from_position_without_execution=bool(accounting.get("allow_open_from_position_without_execution", True)),
            allow_flatten_without_pending_exit=bool(accounting.get("allow_flatten_without_pending_exit", False)),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("accounting.strategy_id is required")
        if not self.account:
            raise ValueError("accounting.account is required")
        if not self.symbol:
            raise ValueError("accounting.symbol is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state_config_path"] = self.state_config_path.as_posix()
        data["pnl_log_path"] = self.pnl_log_path.as_posix()
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class H1CAccountingPaths:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    state_path: Path
    event_path: Path
    pnl_event_path: Path
    pnl_log_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_accounting_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H1CAccountingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return H1CAccountingConfig.from_mapping(raw)


def load_reconciliation_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return manifest


def _read_output_frame(manifest: dict[str, Any], key: str) -> pd.DataFrame:
    output_path = manifest.get("outputs", {}).get(key, "")
    if not output_path:
        return pd.DataFrame()
    path = Path(output_path)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _target_executions(executions: pd.DataFrame, account: str, symbol: str) -> pd.DataFrame:
    if executions.empty:
        return executions.copy()
    out = executions.copy()
    return out[
        out.get("account", pd.Series("", index=out.index)).astype(str).isin(["", account])
        & out.get("symbol", pd.Series("", index=out.index)).astype(str).str.upper().eq(symbol)
    ].copy()


def _entry_fill_price(executions: pd.DataFrame, positions: pd.DataFrame, account: str, symbol: str) -> float | None:
    target_execs = _target_executions(executions, account, symbol)
    if not target_execs.empty and "price" in target_execs.columns and "shares" in target_execs.columns:
        sell_execs = target_execs[target_execs.get("side", "").astype(str).str.upper().isin(["SLD", "SELL"])]
        if not sell_execs.empty:
            shares = pd.to_numeric(sell_execs["shares"], errors="coerce").abs()
            prices = pd.to_numeric(sell_execs["price"], errors="coerce")
            denom = float(shares.sum())
            if denom > 0:
                return float((prices * shares).sum() / denom)
    if not positions.empty and "avg_cost" in positions.columns:
        target_positions = positions[positions.get("symbol", pd.Series("", index=positions.index)).astype(str).str.upper().eq(symbol)]
        if not target_positions.empty:
            avg_cost = pd.to_numeric(target_positions["avg_cost"], errors="coerce").dropna()
            if not avg_cost.empty:
                return float(avg_cost.iloc[0])
    return None


def _exit_fill_price(executions: pd.DataFrame, account: str, symbol: str) -> float | None:
    target_execs = _target_executions(executions, account, symbol)
    if target_execs.empty or "price" not in target_execs.columns or "shares" not in target_execs.columns:
        return None
    buy_execs = target_execs[target_execs.get("side", "").astype(str).str.upper().isin(["BOT", "BUY"])]
    if buy_execs.empty:
        return None
    shares = pd.to_numeric(buy_execs["shares"], errors="coerce").abs()
    prices = pd.to_numeric(buy_execs["price"], errors="coerce")
    denom = float(shares.sum())
    if denom <= 0:
        return None
    return float((prices * shares).sum() / denom)


def _exit_realized_pnl(executions: pd.DataFrame, account: str, symbol: str) -> float | None:
    target_execs = _target_executions(executions, account, symbol)
    if target_execs.empty or "realized_pnl" not in target_execs.columns:
        return None
    buy_execs = target_execs[target_execs.get("side", "").astype(str).str.upper().isin(["BOT", "BUY"])]
    values = pd.to_numeric(buy_execs["realized_pnl"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _append_pnl_event(event: dict[str, Any], path: str | Path) -> None:
    pnl_path = Path(path)
    pnl_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([event])
    if pnl_path.exists():
        existing = pd.read_parquet(pnl_path)
        output = pd.concat([existing, row], ignore_index=True)
    else:
        output = row
    output.to_parquet(pnl_path, index=False)


def apply_accounting_to_state(
    *,
    state: dict[str, Any],
    reconciliation_manifest: dict[str, Any],
    positions: pd.DataFrame,
    executions: pd.DataFrame,
    config: H1CAccountingConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    now = utc_now()
    updated = dict(state)
    reconciliation = dict(reconciliation_manifest.get("reconciliation", {}) or {})
    decision = str(reconciliation.get("decision", ""))
    status = str(updated.get("status", "flat"))
    event_type = "accounting_no_change"
    pnl_event: dict[str, Any] = {
        "created_at_utc": now,
        "event_type": "none",
        "strategy_id": config.strategy_id,
        "account": config.account,
        "symbol": config.symbol,
        "quantity": 0.0,
        "entry_price": None,
        "exit_price": None,
        "theoretical_entry_price": None,
        "theoretical_exit_price": None,
        "entry_slippage_bps": None,
        "exit_slippage_bps": None,
        "realized_pnl": None,
    }

    if status == "pending_entry" and decision == "FILL_DETECTED_PENDING_ENTRY":
        target_qty = abs(float(reconciliation.get("target_position_qty", 0.0) or 0.0))
        if target_qty <= 0:
            raise ValueError("FILL_DETECTED_PENDING_ENTRY requires nonzero target_position_qty")
        entry_price = _entry_fill_price(executions, positions, config.account, config.symbol)
        pending = updated.get("pending_ticket") or {}
        theoretical_entry = pending.get("theoretical_entry_price")
        theoretical_entry_float = float(theoretical_entry) if theoretical_entry is not None else None
        entry_slippage_bps = None
        if entry_price is not None and theoretical_entry_float and theoretical_entry_float > 0:
            entry_slippage_bps = (entry_price / theoretical_entry_float - 1.0) * 10_000.0
        updated["status"] = "open"
        updated["position_unit"] = -1.0
        updated["quantity"] = target_qty
        updated["desired_position_unit"] = -1.0
        updated["pending_ticket"] = None
        updated["open_position"] = {
            "opened_at_utc": now,
            "quantity": target_qty,
            "side": "SHORT",
            "signal_timestamp": pending.get("signal_timestamp"),
            "theoretical_entry_timestamp": pending.get("theoretical_entry_timestamp"),
            "theoretical_exit_timestamp": pending.get("theoretical_exit_timestamp"),
            "theoretical_exit_price": _optional_float(pending.get("theoretical_exit_price")),
            "exit_rule": pending.get("exit_rule"),
            "horizon_bars": pending.get("horizon_bars"),
            "entry_price": entry_price,
            "theoretical_entry_price": theoretical_entry_float,
            "entry_slippage_bps": entry_slippage_bps,
            "source_reconciliation_decision": decision,
        }
        event_type = "entry_fill_marked_open"
        pnl_event = {
            **pnl_event,
            "event_type": "entry",
            "quantity": target_qty,
            "entry_price": entry_price,
            "theoretical_entry_price": theoretical_entry_float,
            "entry_slippage_bps": entry_slippage_bps,
        }
    elif status == "pending_exit" and decision == "FILL_DETECTED_PENDING_EXIT":
        open_position = dict(updated.get("open_position") or {})
        pending = dict(updated.get("pending_ticket") or {})
        target_qty = abs(float(updated.get("quantity", 0.0) or open_position.get("quantity", 0.0) or reconciliation.get("expected_state_quantity", 0.0) or 0.0))
        exit_price = _exit_fill_price(executions, config.account, config.symbol)
        entry_price = _optional_float(open_position.get("entry_price")) or _optional_float(open_position.get("theoretical_entry_price"))
        theoretical_exit = pending.get("theoretical_exit_price", open_position.get("theoretical_exit_price"))
        theoretical_exit_float = _optional_float(theoretical_exit)
        realized_pnl = _exit_realized_pnl(executions, config.account, config.symbol)
        if realized_pnl is None and entry_price is not None and exit_price is not None and target_qty > 0:
            realized_pnl = (entry_price - exit_price) * target_qty
        exit_slippage_bps = None
        if exit_price is not None and theoretical_exit_float and theoretical_exit_float > 0:
            exit_slippage_bps = (exit_price / theoretical_exit_float - 1.0) * 10_000.0
        updated["status"] = "flat"
        updated["position_unit"] = 0.0
        updated["quantity"] = 0.0
        updated["desired_position_unit"] = 0.0
        updated["pending_ticket"] = None
        updated["open_position"] = None
        event_type = "exit_fill_marked_flat"
        pnl_event = {
            **pnl_event,
            "event_type": "exit",
            "quantity": target_qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "theoretical_entry_price": _optional_float(open_position.get("theoretical_entry_price")),
            "theoretical_exit_price": theoretical_exit_float,
            "exit_slippage_bps": exit_slippage_bps,
            "realized_pnl": realized_pnl,
        }
    elif status == "pending_entry" and decision == "PENDING_TICKET_WITHOUT_OPEN_ORDER":
        event_type = "pending_entry_missing_order_block"
    elif status == "open" and decision == "OK_OPEN":
        event_type = "open_position_confirmed"
    elif status == "pending_exit" and decision == "OK_PENDING_EXIT":
        event_type = "pending_exit_confirmed"

    updated["updated_at_utc"] = now
    event = {
        "created_at_utc": now,
        "event_type": event_type,
        "strategy_id": config.strategy_id,
        "account": config.account,
        "symbol": config.symbol,
        "previous_status": status,
        "new_status": updated.get("status"),
        "reconciliation_decision": decision,
        "position_unit": float(updated.get("position_unit", 0.0) or 0.0),
        "quantity": float(updated.get("quantity", 0.0) or 0.0),
        "state_updated": event_type in {"entry_fill_marked_open", "exit_fill_marked_flat"},
    }
    return updated, event, pnl_event


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    accounting = manifest["accounting"]
    lines = [
        "# H1c paper accounting",
        "",
        f"- Created UTC: `{manifest['run']['created_at_utc']}`",
        f"- Event: `{accounting['event']['event_type']}`",
        f"- Previous status: `{accounting['event']['previous_status']}`",
        f"- New status: `{accounting['event']['new_status']}`",
        f"- Reconciliation decision: `{accounting['event']['reconciliation_decision']}`",
        f"- PnL event: `{accounting['pnl_event']['event_type']}`",
        "",
        "This accounting run updates local paper state only. It does not submit orders.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_h1c_accounting(
    *,
    reconciliation_manifest_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
) -> tuple[H1CAccountingPaths, dict[str, Any]]:
    config = load_accounting_config(config_path)
    state_config = load_state_config(config.state_config_path)
    state = load_state(state_config)
    manifest_in = load_reconciliation_manifest(reconciliation_manifest_path)
    positions = _read_output_frame(manifest_in, "positions")
    executions = _read_output_frame(manifest_in, "executions")
    updated, event, pnl_event = apply_accounting_to_state(
        state=state,
        reconciliation_manifest=manifest_in,
        positions=positions,
        executions=executions,
        config=config,
    )
    validate_state(updated, state_config)

    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = H1CAccountingPaths(
        output_dir=root,
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
        state_path=state_config.state_path,
        event_path=root / "event.yaml",
        pnl_event_path=root / "pnl_event.yaml",
        pnl_log_path=config.pnl_log_path,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_state(updated, state_config.state_path)
    append_event(event, state_config.event_log_path)
    _append_pnl_event(pnl_event, config.pnl_log_path)
    paths.event_path.write_text(yaml.safe_dump(event, sort_keys=False), encoding="utf-8")
    paths.pnl_event_path.write_text(yaml.safe_dump(pnl_event, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run": {"run_type": "h1c_paper_accounting", "created_at_utc": created, "status": "complete"},
        "config": config.to_dict(),
        "source": {"reconciliation_manifest": Path(reconciliation_manifest_path).as_posix()},
        "accounting": {"event": event, "pnl_event": pnl_event, "state": updated},
        "outputs": {
            "manifest": paths.manifest_path.as_posix(),
            "report": paths.report_path.as_posix(),
            "event": paths.event_path.as_posix(),
            "pnl_event": paths.pnl_event_path.as_posix(),
            "pnl_log": paths.pnl_log_path.as_posix(),
            "state": paths.state_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, manifest)
    return paths, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply H1c paper fill/accounting updates from reconciliation output")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--reconciliation-manifest", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, manifest = run_h1c_accounting(
        reconciliation_manifest_path=args.reconciliation_manifest,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": manifest}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

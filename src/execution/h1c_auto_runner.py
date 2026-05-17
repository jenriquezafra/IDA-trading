from __future__ import annotations

import argparse
import fcntl
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.execution.h1c_order_executor import run_h1c_order_execution
from src.execution.h1c_order_plan import create_h1c_order_plan
from src.execution.flatten_executor import is_nyse_rth
from src.execution.paper_accounting_h1c import run_h1c_accounting
from src.execution.paper_data_refresh import run_paper_data_refresh
from src.execution.paper_h1c_signal_runner import run_h1c_signal_runner
from src.execution.paper_reconcile_h1c import run_h1c_reconciliation
from src.execution.paper_state_store import apply_ticket, utc_now


DEFAULT_CONFIG_PATH = Path("configs/execution/h1c_auto_runner.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_auto_runner")


@dataclass(frozen=True)
class H1CAutoConfig:
    strategy_id: str
    account: str
    symbol: str
    paper_only: bool
    require_market_open: bool
    execute_orders: bool
    transmit_orders: bool
    apply_state_after_submission: bool
    run_accounting_after_reconciliation: bool
    default_skip_cboe: bool
    default_skip_download: bool
    min_available_funds_usd: float
    min_buying_power_usd: float
    max_order_notional_usd: float | None
    sizing_mode: str
    capital_fraction: float
    reserve_cash_usd: float
    min_quantity: float
    require_account_summary: bool
    enabled: bool
    kill_switch_path: Path
    max_daily_entry_orders: int | None
    max_daily_realized_loss_usd: float | None
    max_entry_slippage_bps: float | None
    max_exit_slippage_bps: float | None
    data_refresh_config_path: Path
    signal_runner_config_path: Path
    state_config_path: Path
    reconciliation_config_path: Path
    accounting_config_path: Path
    order_plan_config_path: Path
    order_executor_config_path: Path
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "H1CAutoConfig":
        auto = dict(raw.get("auto", {}))
        components = dict(raw.get("components", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            strategy_id=str(auto.get("strategy_id", "")).strip(),
            account=str(auto.get("account", "")).strip(),
            symbol=str(auto.get("symbol", "QQQ")).strip().upper(),
            paper_only=bool(auto.get("paper_only", True)),
            require_market_open=bool(auto.get("require_market_open", True)),
            execute_orders=bool(auto.get("execute_orders", True)),
            transmit_orders=bool(auto.get("transmit_orders", True)),
            apply_state_after_submission=bool(auto.get("apply_state_after_submission", True)),
            run_accounting_after_reconciliation=bool(auto.get("run_accounting_after_reconciliation", True)),
            default_skip_cboe=bool(auto.get("default_skip_cboe", True)),
            default_skip_download=bool(auto.get("default_skip_download", False)),
            min_available_funds_usd=float(auto.get("min_available_funds_usd", 1000.0)),
            min_buying_power_usd=float(auto.get("min_buying_power_usd", 1000.0)),
            max_order_notional_usd=None if auto.get("max_order_notional_usd") in {None, ""} else float(auto.get("max_order_notional_usd", 1000.0)),
            sizing_mode=str(auto.get("sizing_mode", "ticket_quantity")).strip(),
            capital_fraction=float(auto.get("capital_fraction", 1.0)),
            reserve_cash_usd=float(auto.get("reserve_cash_usd", 0.0)),
            min_quantity=float(auto.get("min_quantity", 1.0)),
            require_account_summary=bool(auto.get("require_account_summary", True)),
            enabled=bool(auto.get("enabled", True)),
            kill_switch_path=Path(auto.get("kill_switch_path", "ops/kill_switches/h1c_auto_paused")),
            max_daily_entry_orders=None if auto.get("max_daily_entry_orders") in {None, ""} else int(auto.get("max_daily_entry_orders", 4)),
            max_daily_realized_loss_usd=None
            if auto.get("max_daily_realized_loss_usd") in {None, ""}
            else float(auto.get("max_daily_realized_loss_usd", 1000.0)),
            max_entry_slippage_bps=None if auto.get("max_entry_slippage_bps") in {None, ""} else float(auto.get("max_entry_slippage_bps", 25.0)),
            max_exit_slippage_bps=None if auto.get("max_exit_slippage_bps") in {None, ""} else float(auto.get("max_exit_slippage_bps", 25.0)),
            data_refresh_config_path=Path(components.get("data_refresh_config_path", "configs/execution/paper_data_refresh.yaml")),
            signal_runner_config_path=Path(components.get("signal_runner_config_path", "configs/execution/paper_runner_h1c_signal_only.yaml")),
            state_config_path=Path(components.get("state_config_path", "configs/execution/paper_state_h1c.yaml")),
            reconciliation_config_path=Path(components.get("reconciliation_config_path", "configs/execution/paper_reconcile_h1c.yaml")),
            accounting_config_path=Path(components.get("accounting_config_path", "configs/execution/paper_accounting_h1c.yaml")),
            order_plan_config_path=Path(components.get("order_plan_config_path", "configs/execution/h1c_order_plan.yaml")),
            order_executor_config_path=Path(components.get("order_executor_config_path", "configs/execution/h1c_order_executor_auto_paper.yaml")),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("auto.strategy_id is required")
        if not self.account:
            raise ValueError("auto.account is required")
        if not self.symbol:
            raise ValueError("auto.symbol is required")
        if not self.paper_only:
            raise ValueError("H1c auto runner is paper-only")
        if self.max_order_notional_usd is not None and self.max_order_notional_usd <= 0:
            raise ValueError("max_order_notional_usd must be positive")
        if self.sizing_mode not in {"ticket_quantity", "buying_power_fraction", "available_funds_fraction"}:
            raise ValueError("unsupported sizing_mode")
        if not 0 < self.capital_fraction <= 1:
            raise ValueError("capital_fraction must be in (0, 1]")
        if self.reserve_cash_usd < 0:
            raise ValueError("reserve_cash_usd must be non-negative")
        if self.min_quantity <= 0:
            raise ValueError("min_quantity must be positive")
        if self.min_available_funds_usd < 0 or self.min_buying_power_usd < 0:
            raise ValueError("cash/buying-power thresholds must be non-negative")
        if self.max_daily_entry_orders is not None and self.max_daily_entry_orders <= 0:
            raise ValueError("max_daily_entry_orders must be positive when set")
        if self.max_daily_realized_loss_usd is not None and self.max_daily_realized_loss_usd <= 0:
            raise ValueError("max_daily_realized_loss_usd must be positive when set")
        if self.max_entry_slippage_bps is not None and self.max_entry_slippage_bps <= 0:
            raise ValueError("max_entry_slippage_bps must be positive when set")
        if self.max_exit_slippage_bps is not None and self.max_exit_slippage_bps <= 0:
            raise ValueError("max_exit_slippage_bps must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in [
            "data_refresh_config_path",
            "signal_runner_config_path",
            "state_config_path",
            "reconciliation_config_path",
            "accounting_config_path",
            "order_plan_config_path",
            "order_executor_config_path",
            "kill_switch_path",
            "output_dir",
        ]:
            data[key] = data[key].as_posix()
        return data


@dataclass(frozen=True)
class H1CAutoPaths:
    output_dir: Path
    manifest_path: Path
    report_path: Path


def load_auto_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H1CAutoConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return H1CAutoConfig.from_mapping(raw)


def _paths_to_dict(paths: Any | None) -> dict[str, str]:
    if paths is None:
        return {}
    return {key: str(value) for key, value in asdict(paths).items()}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _account_summary_from_reconciliation(reconciliation_manifest: dict[str, Any]) -> pd.DataFrame:
    snapshot_dir = reconciliation_manifest.get("outputs", {}).get("ibkr_snapshot_dir")
    if not snapshot_dir:
        return pd.DataFrame()
    path = Path(snapshot_dir) / "account_summary.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def account_value(account_summary: pd.DataFrame, account: str, tag: str, currency: str = "USD") -> float | None:
    if account_summary.empty:
        return None
    frame = account_summary.copy()
    mask = frame.get("account", pd.Series("", index=frame.index)).astype(str).eq(account)
    mask &= frame.get("tag", pd.Series("", index=frame.index)).astype(str).eq(tag)
    rows = frame[mask]
    if rows.empty:
        return None
    if "currency" in frame.columns:
        requested = rows[rows["currency"].astype(str).isin([currency, ""])].copy()
        if not requested.empty:
            rows = requested
    return _safe_float(rows.iloc[0].get("value"))


def market_is_open(reconciliation_manifest: dict[str, Any]) -> tuple[bool, str]:
    server_time = reconciliation_manifest.get("ibkr", {}).get("health", {}).get("server_time", "")
    if not server_time:
        return False, "missing_ibkr_server_time"
    return bool(is_nyse_rth(pd.Timestamp(server_time).to_pydatetime())), str(server_time)


def size_ticket(ticket: dict[str, Any], account_summary: pd.DataFrame, config: H1CAutoConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    sized = dict(ticket)
    original_quantity = float(ticket.get("quantity", 0.0) or 0.0)
    price = _safe_float(ticket.get("theoretical_entry_price"))
    available = account_value(account_summary, config.account, "AvailableFunds")
    buying_power = account_value(account_summary, config.account, "BuyingPower")
    reference_capital = None
    if config.sizing_mode == "buying_power_fraction":
        reference_capital = buying_power
    elif config.sizing_mode == "available_funds_fraction":
        reference_capital = available
    if str(ticket.get("action", "NONE")).upper() == "SELL" and price and reference_capital is not None:
        usable_capital = max(0.0, reference_capital * config.capital_fraction - config.reserve_cash_usd)
        quantity = math.floor(usable_capital / price)
        if quantity >= config.min_quantity:
            sized["quantity"] = float(quantity)
        else:
            sized["quantity"] = 0.0
            sized["action"] = "NONE"
            sized["status"] = "sizing_blocked"
            sized["reason"] = "sized quantity is below min_quantity"
    return sized, {
        "sizing_mode": config.sizing_mode,
        "original_quantity": original_quantity,
        "sized_quantity": float(sized.get("quantity", 0.0) or 0.0),
        "reference_capital": reference_capital,
        "capital_fraction": config.capital_fraction,
        "reserve_cash_usd": config.reserve_cash_usd,
        "theoretical_entry_price": price,
        "available_funds": available,
        "buying_power": buying_power,
    }


def write_sized_ticket(ticket: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "paper_ticket_sized.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(ticket, sort_keys=False), encoding="utf-8")
    return path


def ticket_reference_price(ticket: dict[str, Any]) -> float | None:
    action = str(ticket.get("action", "NONE")).upper()
    if action == "BUY":
        return _safe_float(ticket.get("theoretical_exit_price")) or _safe_float(ticket.get("theoretical_entry_price"))
    return _safe_float(ticket.get("theoretical_entry_price")) or _safe_float(ticket.get("theoretical_exit_price"))


def funds_check(ticket: dict[str, Any], account_summary: pd.DataFrame, config: H1CAutoConfig) -> dict[str, Any]:
    quantity = float(ticket.get("quantity", 0.0) or 0.0)
    price = ticket_reference_price(ticket)
    notional = quantity * (price or 0.0)
    available = account_value(account_summary, config.account, "AvailableFunds")
    buying_power = account_value(account_summary, config.account, "BuyingPower")
    errors: list[str] = []
    if config.require_account_summary and account_summary.empty:
        errors.append("account summary is missing")
    if quantity > 0 and price is None:
        errors.append("ticket reference price is missing")
    if config.max_order_notional_usd is not None and notional > config.max_order_notional_usd:
        errors.append(f"order notional {notional:.2f} exceeds max_order_notional_usd {config.max_order_notional_usd:.2f}")
    if available is None:
        if config.require_account_summary:
            errors.append("AvailableFunds is missing")
    elif available < config.min_available_funds_usd:
        errors.append(f"AvailableFunds {available:.2f} below minimum {config.min_available_funds_usd:.2f}")
    if buying_power is None:
        if config.require_account_summary:
            errors.append("BuyingPower is missing")
    elif buying_power < config.min_buying_power_usd:
        errors.append(f"BuyingPower {buying_power:.2f} below minimum {config.min_buying_power_usd:.2f}")
    return {
        "ok": not errors,
        "errors": errors,
        "available_funds": available,
        "buying_power": buying_power,
        "ticket_quantity": quantity,
        "ticket_price": price,
        "order_notional": notional,
    }


def load_state_snapshot(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def exit_due_status(state: dict[str, Any], server_time: str) -> dict[str, Any]:
    open_position = dict(state.get("open_position") or {})
    exit_timestamp = open_position.get("theoretical_exit_timestamp")
    if state.get("status") != "open":
        return {"due": False, "reason": "state_not_open", "exit_timestamp": exit_timestamp, "server_time": server_time}
    if not exit_timestamp:
        return {"due": False, "reason": "missing_theoretical_exit_timestamp", "exit_timestamp": exit_timestamp, "server_time": server_time}
    try:
        server_ts = pd.Timestamp(server_time)
        exit_ts = pd.Timestamp(exit_timestamp)
    except Exception as exc:  # noqa: BLE001
        return {"due": False, "reason": f"invalid_timestamp: {exc}", "exit_timestamp": exit_timestamp, "server_time": server_time}
    if server_ts.tzinfo is not None and exit_ts.tzinfo is None:
        exit_ts = exit_ts.tz_localize(server_ts.tzinfo)
    elif server_ts.tzinfo is None and exit_ts.tzinfo is not None:
        server_ts = server_ts.tz_localize(exit_ts.tzinfo)
    return {
        "due": bool(server_ts >= exit_ts),
        "reason": "exit_due" if server_ts >= exit_ts else "exit_not_due",
        "exit_timestamp": str(exit_ts),
        "server_time": str(server_ts),
    }


def build_exit_ticket(state: dict[str, Any], config: H1CAutoConfig) -> dict[str, Any]:
    open_position = dict(state.get("open_position") or {})
    quantity = abs(float(state.get("quantity", 0.0) or open_position.get("quantity", 0.0) or 0.0))
    return {
        "mode": "signal_only",
        "send_orders": False,
        "strategy_id": config.strategy_id,
        "account": config.account,
        "symbol": config.symbol,
        "signal_timestamp": open_position.get("signal_timestamp") or state.get("last_signal_timestamp"),
        "session": "",
        "bar_index": None,
        "entry_rule": "next_open",
        "exit_rule": open_position.get("exit_rule") or "fixed_horizon_open",
        "horizon_bars": open_position.get("horizon_bars"),
        "execution_timing": "fixed_horizon_exit",
        "theoretical_entry_timestamp": open_position.get("theoretical_entry_timestamp"),
        "theoretical_entry_price": open_position.get("theoretical_entry_price"),
        "theoretical_exit_timestamp": open_position.get("theoretical_exit_timestamp"),
        "theoretical_exit_price": open_position.get("theoretical_exit_price"),
        "desired_position_unit": 0.0,
        "action": "BUY",
        "quantity": quantity,
        "order_type": "MKT",
        "time_in_force": "DAY",
        "status": "paper_ticket_only",
        "reason": "H1c fixed-horizon short exit is due",
    }


def _utc_date(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.date().isoformat()


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def daily_entry_order_count(output_dir: str | Path, date_utc: str) -> int:
    root = Path(output_dir)
    if not root.exists():
        return 0
    total = 0
    for manifest_path in root.glob("*/manifest.yaml"):
        manifest = _read_yaml_if_exists(manifest_path)
        created = (manifest.get("run") or {}).get("created_at_utc")
        if not created:
            continue
        try:
            if _utc_date(created) != date_utc:
                continue
        except Exception:
            continue
        ticket = ((manifest.get("signal") or {}).get("ticket") or {})
        action = str(ticket.get("action") or "").upper()
        if action != "SELL":
            continue
        try:
            total += int(((manifest.get("execution") or {}).get("summary") or {}).get("submitted_orders") or 0)
        except (TypeError, ValueError):
            continue
    return total


def accounting_pnl_log_path(config: H1CAutoConfig) -> Path:
    raw = _read_yaml_if_exists(config.accounting_config_path)
    accounting = dict(raw.get("accounting", {}) or {})
    return Path(accounting.get("pnl_log_path", "results/paper/h1c_state/pnl_events.parquet"))


def daily_realized_pnl(pnl_log_path: str | Path, date_utc: str) -> float:
    path = Path(pnl_log_path)
    if not path.exists():
        return 0.0
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return 0.0
    if frame.empty or "created_at_utc" not in frame.columns or "realized_pnl" not in frame.columns:
        return 0.0
    created_dates = pd.to_datetime(frame["created_at_utc"], errors="coerce", utc=True).dt.date.astype(str)
    values = pd.to_numeric(frame.loc[created_dates.eq(date_utc), "realized_pnl"], errors="coerce").fillna(0.0)
    return float(values.sum())


def latest_slippage_metrics(pnl_log_path: str | Path) -> dict[str, float | None]:
    path = Path(pnl_log_path)
    if not path.exists():
        return {"entry_slippage_bps": None, "exit_slippage_bps": None}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {"entry_slippage_bps": None, "exit_slippage_bps": None}
    metrics: dict[str, float | None] = {}
    for column in ["entry_slippage_bps", "exit_slippage_bps"]:
        if column not in frame.columns:
            metrics[column] = None
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        metrics[column] = None if values.empty else float(values.iloc[-1])
    return metrics


def entry_safety_check(config: H1CAutoConfig, *, output_dir: str | Path, created_at_utc: str) -> dict[str, Any]:
    date_utc = _utc_date(created_at_utc)
    entry_orders_today = daily_entry_order_count(output_dir, date_utc)
    pnl_path = accounting_pnl_log_path(config)
    realized_pnl_today = daily_realized_pnl(pnl_path, date_utc)
    slippage = latest_slippage_metrics(pnl_path)
    issues: list[str] = []
    kill_switch_exists = config.kill_switch_path.exists()
    if not config.enabled:
        issues.append("auto.enabled is false")
    if kill_switch_exists:
        issues.append(f"kill switch file exists: {config.kill_switch_path.as_posix()}")
    if config.max_daily_entry_orders is not None and entry_orders_today >= config.max_daily_entry_orders:
        issues.append(f"daily entry order limit reached: {entry_orders_today}/{config.max_daily_entry_orders}")
    if config.max_daily_realized_loss_usd is not None and realized_pnl_today <= -abs(config.max_daily_realized_loss_usd):
        issues.append(f"daily realized loss limit reached: {realized_pnl_today:.2f} <= -{abs(config.max_daily_realized_loss_usd):.2f}")
    entry_slippage = slippage.get("entry_slippage_bps")
    if config.max_entry_slippage_bps is not None and entry_slippage is not None and abs(entry_slippage) >= config.max_entry_slippage_bps:
        issues.append(f"entry slippage limit reached: {entry_slippage:.2f} bps >= {config.max_entry_slippage_bps:.2f} bps")
    exit_slippage = slippage.get("exit_slippage_bps")
    if config.max_exit_slippage_bps is not None and exit_slippage is not None and abs(exit_slippage) >= config.max_exit_slippage_bps:
        issues.append(f"exit slippage limit reached: {exit_slippage:.2f} bps >= {config.max_exit_slippage_bps:.2f} bps")
    return {
        "ok": not issues,
        "issues": issues,
        "date_utc": date_utc,
        "enabled": config.enabled,
        "kill_switch_path": config.kill_switch_path.as_posix(),
        "kill_switch_exists": kill_switch_exists,
        "entry_orders_today": entry_orders_today,
        "max_daily_entry_orders": config.max_daily_entry_orders,
        "realized_pnl_today": realized_pnl_today,
        "max_daily_realized_loss_usd": config.max_daily_realized_loss_usd,
        "latest_entry_slippage_bps": entry_slippage,
        "max_entry_slippage_bps": config.max_entry_slippage_bps,
        "latest_exit_slippage_bps": exit_slippage,
        "max_exit_slippage_bps": config.max_exit_slippage_bps,
        "pnl_log_path": pnl_path.as_posix(),
    }


def reconciliation_drift_snapshot(reconciliation: dict[str, Any]) -> dict[str, Any]:
    target_qty = float(reconciliation.get("target_position_qty", 0.0) or 0.0)
    expected_qty = float(reconciliation.get("expected_state_quantity", 0.0) or 0.0)
    signed_expected = -abs(expected_qty) if expected_qty else 0.0
    return {
        "decision": reconciliation.get("decision"),
        "severity": reconciliation.get("severity"),
        "target_position_qty": target_qty,
        "expected_state_quantity": expected_qty,
        "signed_expected_position_qty": signed_expected,
        "position_qty_drift": target_qty - signed_expected,
        "target_open_orders": reconciliation.get("target_open_orders"),
        "account_open_orders": reconciliation.get("account_open_orders"),
        "unrelated_open_orders": reconciliation.get("unrelated_open_orders"),
        "state_transition_hint": reconciliation.get("state_transition_hint"),
    }


def attach_operational_metadata(manifest: dict[str, Any], *, started_monotonic: float, step_timings: dict[str, float]) -> None:
    manifest["latency"] = {
        "total_seconds": round(time.perf_counter() - started_monotonic, 3),
        "steps": {key: round(value, 3) for key, value in step_timings.items()},
    }
    drift: dict[str, Any] = {}
    if manifest.get("pre_trade_reconciliation"):
        drift["pre_trade"] = reconciliation_drift_snapshot(dict(manifest.get("pre_trade_reconciliation") or {}))
    if manifest.get("post_execution_reconciliation"):
        drift["post_execution"] = reconciliation_drift_snapshot(dict(manifest.get("post_execution_reconciliation") or {}))
    if drift:
        manifest["drift"] = drift


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# H1c auto runner",
        "",
        f"- Created UTC: `{manifest['run']['created_at_utc']}`",
        f"- Decision: `{manifest['decision']}`",
        f"- Reason: `{manifest['reason']}`",
        f"- Market open: `{manifest.get('market', {}).get('open', '')}`",
        f"- Pre-trade reconciliation: `{manifest.get('pre_trade_reconciliation', {}).get('decision', '')}`",
        f"- Signal action: `{manifest.get('signal', {}).get('ticket', {}).get('action', '')}`",
        f"- Funds OK: `{manifest.get('funds', {}).get('ok', '')}`",
        f"- Entry safety OK: `{manifest.get('entry_safety', {}).get('ok', '')}`",
        f"- Plan decision: `{manifest.get('order_plan', {}).get('summary', {}).get('decision', '')}`",
        f"- Submitted orders: `{manifest.get('execution', {}).get('summary', {}).get('submitted_orders', 0)}`",
        f"- Latency seconds: `{manifest.get('latency', {}).get('total_seconds', '')}`",
        "",
        "This runner is paper-only. It can submit paper orders only when market, reconciliation, signal, funds, planner, and executor guardrails all pass.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_h1c_auto(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    skip_download: bool | None = None,
    skip_cboe: bool | None = None,
    output_dir: str | Path | None = None,
) -> tuple[H1CAutoPaths, dict[str, Any]]:
    config = load_auto_config(config_path)
    created = utc_now()
    started_monotonic = time.perf_counter()
    step_timings: dict[str, float] = {}
    base_output_dir = Path(output_dir) if output_dir is not None else config.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (base_output_dir / "auto.lock").open("w", encoding="utf-8")
    root = base_output_dir / created.replace(":", "").replace("-", "")
    root.mkdir(parents=True, exist_ok=True)
    paths = H1CAutoPaths(output_dir=root, manifest_path=root / "manifest.yaml", report_path=root / "report.md")

    def finish_step(name: str, started: float) -> None:
        step_timings[name] = time.perf_counter() - started

    def write_manifest_and_report(manifest: dict[str, Any]) -> tuple[H1CAutoPaths, dict[str, Any]]:
        attach_operational_metadata(manifest, started_monotonic=started_monotonic, step_timings=step_timings)
        paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        _write_report(paths.report_path, manifest)
        return paths, manifest

    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        manifest = {
            "schema_version": 1,
            "run": {"run_type": "h1c_auto_runner", "created_at_utc": created, "status": "skipped"},
            "config": config.to_dict(),
            "decision": "lock_held",
            "reason": "another H1c auto runner instance is already active",
        }
        lock_handle.close()
        return write_manifest_and_report(manifest)

    try:
        effective_skip_download = config.default_skip_download if skip_download is None else skip_download
        effective_skip_cboe = config.default_skip_cboe if skip_cboe is None else skip_cboe

        step_started = time.perf_counter()
        pre_recon_paths, pre_recon_manifest = run_h1c_reconciliation(config_path=config.reconciliation_config_path, output_dir=root / "pre_trade_reconciliation")
        finish_step("pre_trade_reconciliation", step_started)
        is_open, server_time = market_is_open(pre_recon_manifest)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run": {"run_type": "h1c_auto_runner", "created_at_utc": created, "status": "complete"},
            "config": config.to_dict(),
            "market": {"open": is_open, "server_time": server_time},
            "pre_trade_reconciliation": pre_recon_manifest["reconciliation"],
            "paths": {"pre_trade_reconciliation": _paths_to_dict(pre_recon_paths)},
            "decision": "market_closed",
            "reason": "NYSE regular trading hours are closed",
        }
        if config.require_market_open and not is_open:
            return write_manifest_and_report(manifest)

        reconciliation_decision = pre_recon_manifest["reconciliation"]["decision"]
        if reconciliation_decision in {"FILL_DETECTED_PENDING_ENTRY", "FILL_DETECTED_PENDING_EXIT"}:
            if config.run_accounting_after_reconciliation:
                step_started = time.perf_counter()
                accounting_paths, accounting_manifest = run_h1c_accounting(
                    reconciliation_manifest_path=pre_recon_paths.manifest_path,
                    config_path=config.accounting_config_path,
                    output_dir=root / "pre_trade_accounting",
                )
                finish_step("pre_trade_accounting", step_started)
                manifest.update(
                    {
                        "paths": {**manifest["paths"], "accounting": _paths_to_dict(accounting_paths)},
                        "accounting": accounting_manifest.get("accounting", {}),
                        "decision": "accounting_updated",
                        "reason": f"pre-trade reconciliation detected fill: {reconciliation_decision}",
                    }
                )
            else:
                manifest.update(
                    {
                        "decision": "accounting_required",
                        "reason": f"pre-trade reconciliation detected fill but accounting is disabled: {reconciliation_decision}",
                    }
                )
            return write_manifest_and_report(manifest)

        if reconciliation_decision == "OK_OPEN":
            state = load_state_snapshot(pre_recon_paths.state_snapshot_path)
            exit_status = exit_due_status(state, server_time)
            manifest["exit_monitor"] = exit_status
            if not exit_status["due"]:
                manifest.update({"decision": "monitoring_open_position", "reason": exit_status["reason"]})
                return write_manifest_and_report(manifest)

            exit_ticket = build_exit_ticket(state, config)
            exit_ticket_path = write_sized_ticket(exit_ticket, root / "exit_ticket")
            manifest.update({"paths": {**manifest["paths"], "exit_ticket": exit_ticket_path.as_posix()}, "exit_ticket": exit_ticket})
            if float(exit_ticket.get("quantity", 0.0) or 0.0) <= 0.0 or not exit_ticket.get("theoretical_exit_timestamp"):
                manifest.update({"decision": "blocked_exit_ticket", "reason": "exit ticket is missing quantity or theoretical_exit_timestamp"})
                return write_manifest_and_report(manifest)

            step_started = time.perf_counter()
            plan_paths, plan_summary = create_h1c_order_plan(
                ticket_path=exit_ticket_path,
                reconciliation_manifest_path=pre_recon_paths.manifest_path,
                config_path=config.order_plan_config_path,
                output_dir=root / "exit_order_plan",
            )
            finish_step("exit_order_plan", step_started)
            manifest.update({"paths": {**manifest["paths"], "exit_order_plan": _paths_to_dict(plan_paths)}, "order_plan": {"summary": plan_summary}})
            if plan_summary["decision"] != "ready_for_review":
                manifest.update({"decision": "blocked_exit_order_plan", "reason": plan_summary.get("block_reason") or plan_summary["decision"]})
                return write_manifest_and_report(manifest)

            step_started = time.perf_counter()
            exec_paths, exec_summary = run_h1c_order_execution(
                plan_dir=plan_paths.output_dir,
                config_path=config.order_executor_config_path,
                execute=config.execute_orders,
                transmit_orders=config.transmit_orders,
                connect_preflight=True,
                output_dir=root / "exit_execution",
            )
            finish_step("exit_execution", step_started)
            manifest.update({"paths": {**manifest["paths"], "exit_execution": _paths_to_dict(exec_paths)}, "execution": {"summary": exec_summary}})
            if int(exec_summary.get("submitted_orders", 0)) <= 0:
                manifest.update({"decision": "exit_execution_dry_run_or_no_submission", "reason": "executor did not submit an exit order"})
                return write_manifest_and_report(manifest)

            if config.apply_state_after_submission:
                step_started = time.perf_counter()
                state_paths, state_summary = apply_ticket(ticket_path=exit_ticket_path, config_path=config.state_config_path, output_dir=root / "exit_state")
                finish_step("exit_state", step_started)
                manifest.update({"paths": {**manifest["paths"], "exit_state": _paths_to_dict(state_paths)}, "state": state_summary})
            step_started = time.perf_counter()
            post_recon_paths, post_recon_manifest = run_h1c_reconciliation(config_path=config.reconciliation_config_path, output_dir=root / "post_exit_reconciliation")
            finish_step("post_exit_reconciliation", step_started)
            manifest.update(
                {
                    "paths": {**manifest["paths"], "post_exit_reconciliation": _paths_to_dict(post_recon_paths)},
                    "post_execution_reconciliation": post_recon_manifest["reconciliation"],
                    "decision": "exit_submitted",
                    "reason": "paper exit order submitted and post-exit reconciliation captured",
                }
            )
            if config.run_accounting_after_reconciliation:
                step_started = time.perf_counter()
                accounting_paths, accounting_manifest = run_h1c_accounting(
                    reconciliation_manifest_path=post_recon_paths.manifest_path,
                    config_path=config.accounting_config_path,
                    output_dir=root / "exit_accounting",
                )
                finish_step("exit_accounting", step_started)
                manifest.update({"paths": {**manifest["paths"], "exit_accounting": _paths_to_dict(accounting_paths)}, "accounting": accounting_manifest.get("accounting", {})})
            return write_manifest_and_report(manifest)

        step_started = time.perf_counter()
        refresh_paths, refresh_manifest = run_paper_data_refresh(
            config_path=config.data_refresh_config_path,
            skip_download=effective_skip_download,
            skip_cboe=effective_skip_cboe,
            output_dir=root / "data_refresh",
        )
        finish_step("data_refresh", step_started)
        step_started = time.perf_counter()
        signal_paths, signal_summary = run_h1c_signal_runner(config_path=config.signal_runner_config_path, output_dir=root / "signal")
        finish_step("signal", step_started)
        raw_ticket = signal_summary["ticket"]
        account_summary = _account_summary_from_reconciliation(pre_recon_manifest)
        ticket, sizing = size_ticket(raw_ticket, account_summary, config)
        sized_ticket_path = write_sized_ticket(ticket, root / "sizing")
        funds = funds_check(ticket, account_summary, config)
        manifest.update(
            {
                "paths": {
                    **manifest["paths"],
                    "data_refresh": _paths_to_dict(refresh_paths),
                    "signal": _paths_to_dict(signal_paths),
                    "sized_ticket": sized_ticket_path.as_posix(),
                },
                "data_refresh": {"status": refresh_manifest.get("run", {}).get("status"), "date_window": refresh_manifest.get("date_window", {})},
                "signal": {"ticket": ticket, "raw_ticket": raw_ticket, "warnings": signal_summary.get("warnings", [])},
                "sizing": sizing,
                "funds": funds,
            }
        )

        action = str(ticket.get("action", "NONE")).upper()
        if action == "NONE":
            step_started = time.perf_counter()
            state_paths, state_summary = apply_ticket(ticket_path=sized_ticket_path, config_path=config.state_config_path, output_dir=root / "state")
            finish_step("state", step_started)
            manifest.update(
                {
                    "decision": "no_signal",
                    "reason": "H1c signal is NONE",
                    "paths": {**manifest["paths"], "state": _paths_to_dict(state_paths)},
                    "state": state_summary,
                }
            )
            return write_manifest_and_report(manifest)

        if reconciliation_decision != "OK_FLAT":
            manifest.update({"decision": "blocked_reconciliation", "reason": f"pre-trade reconciliation is {reconciliation_decision}"})
            return write_manifest_and_report(manifest)
        if not funds["ok"]:
            manifest.update({"decision": "blocked_funds", "reason": "; ".join(funds["errors"])})
            return write_manifest_and_report(manifest)

        entry_safety = entry_safety_check(config, output_dir=base_output_dir, created_at_utc=created)
        manifest["entry_safety"] = entry_safety
        if not entry_safety["ok"]:
            manifest.update({"decision": "blocked_entry_safety", "reason": "; ".join(entry_safety["issues"])})
            return write_manifest_and_report(manifest)

        step_started = time.perf_counter()
        plan_paths, plan_summary = create_h1c_order_plan(
            ticket_path=sized_ticket_path,
            reconciliation_manifest_path=pre_recon_paths.manifest_path,
            config_path=config.order_plan_config_path,
            output_dir=root / "order_plan",
        )
        finish_step("order_plan", step_started)
        manifest.update({"paths": {**manifest["paths"], "order_plan": _paths_to_dict(plan_paths)}, "order_plan": {"summary": plan_summary}})
        if plan_summary["decision"] != "ready_for_review":
            manifest.update({"decision": "blocked_order_plan", "reason": plan_summary.get("block_reason") or plan_summary["decision"]})
            return write_manifest_and_report(manifest)

        step_started = time.perf_counter()
        exec_paths, exec_summary = run_h1c_order_execution(
            plan_dir=plan_paths.output_dir,
            config_path=config.order_executor_config_path,
            execute=config.execute_orders,
            transmit_orders=config.transmit_orders,
            connect_preflight=True,
            output_dir=root / "execution",
        )
        finish_step("execution", step_started)
        manifest.update({"paths": {**manifest["paths"], "execution": _paths_to_dict(exec_paths)}, "execution": {"summary": exec_summary}})
        if int(exec_summary.get("submitted_orders", 0)) <= 0:
            manifest.update({"decision": "execution_dry_run_or_no_submission", "reason": "executor did not submit an order"})
            return write_manifest_and_report(manifest)

        if config.apply_state_after_submission:
            step_started = time.perf_counter()
            state_paths, state_summary = apply_ticket(ticket_path=sized_ticket_path, config_path=config.state_config_path, output_dir=root / "state")
            finish_step("state", step_started)
            manifest.update({"paths": {**manifest["paths"], "state": _paths_to_dict(state_paths)}, "state": state_summary})
        step_started = time.perf_counter()
        post_recon_paths, post_recon_manifest = run_h1c_reconciliation(config_path=config.reconciliation_config_path, output_dir=root / "post_execution_reconciliation")
        finish_step("post_execution_reconciliation", step_started)
        manifest.update(
            {
                "paths": {**manifest["paths"], "post_execution_reconciliation": _paths_to_dict(post_recon_paths)},
                "post_execution_reconciliation": post_recon_manifest["reconciliation"],
                "decision": "submitted",
                "reason": "paper order submitted and post-execution reconciliation captured",
            }
        )
        if config.run_accounting_after_reconciliation:
            step_started = time.perf_counter()
            accounting_paths, accounting_manifest = run_h1c_accounting(
                reconciliation_manifest_path=post_recon_paths.manifest_path,
                config_path=config.accounting_config_path,
                output_dir=root / "accounting",
            )
            finish_step("accounting", step_started)
            manifest.update({"paths": {**manifest["paths"], "accounting": _paths_to_dict(accounting_paths)}, "accounting": accounting_manifest.get("accounting", {})})
        return write_manifest_and_report(manifest)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Automatic paper-only H1c runner with market-open, reconciliation, funds, and duplicate-order guardrails")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-cboe", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, manifest = run_h1c_auto(
        config_path=args.config,
        skip_download=True if args.skip_download else None,
        skip_cboe=True if args.skip_cboe else None,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": _paths_to_dict(paths), "summary": manifest}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

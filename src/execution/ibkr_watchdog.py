from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from src.execution.ibkr_read_only import IBKRReadOnlyClient, load_ibkr_read_only_config, utc_now
from src.execution.operational_events import DEFAULT_OPERATIONAL_EVENTS_PATH, append_operational_event


DEFAULT_CONFIG_PATH = Path("configs/execution/ibkr_watchdog.yaml")


@dataclass(frozen=True)
class IBKRWatchdogTarget:
    name: str
    config_path: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "IBKRWatchdogTarget":
        name = str(raw.get("name", "") or "").strip()
        config_path = Path(str(raw.get("config_path", "") or "").strip())
        target = cls(name=name, config_path=config_path)
        target.validate()
        return target

    def validate(self) -> None:
        if not self.name:
            raise ValueError("watchdog target requires name")
        if not self.config_path.as_posix():
            raise ValueError("watchdog target requires config_path")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "config_path": self.config_path.as_posix()}


@dataclass(frozen=True)
class IBKRWatchdogConfig:
    targets: tuple[IBKRWatchdogTarget, ...]
    output_dir: Path
    status_path: Path
    events_path: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "IBKRWatchdogConfig":
        watchdog = dict(raw.get("watchdog", {}) or {})
        outputs = dict(raw.get("outputs", {}) or {})
        targets = tuple(IBKRWatchdogTarget.from_mapping(dict(item or {})) for item in watchdog.get("targets", []))
        config = cls(
            targets=targets,
            output_dir=Path(outputs.get("output_dir", "results/paper/ibkr_watchdog")),
            status_path=Path(outputs.get("status_path", "results/paper/ibkr_watchdog/status.yaml")),
            events_path=Path(outputs.get("events_path", DEFAULT_OPERATIONAL_EVENTS_PATH)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.targets:
            raise ValueError("ibkr watchdog requires at least one target")
        if not self.output_dir.as_posix():
            raise ValueError("output_dir is required")
        if not self.status_path.as_posix():
            raise ValueError("status_path is required")
        if not self.events_path.as_posix():
            raise ValueError("events_path is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "output_dir": self.output_dir.as_posix(),
            "status_path": self.status_path.as_posix(),
            "events_path": self.events_path.as_posix(),
        }


def load_watchdog_config(path: str | Path = DEFAULT_CONFIG_PATH) -> IBKRWatchdogConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return IBKRWatchdogConfig.from_mapping(raw)


def _check_target(
    target: IBKRWatchdogTarget,
    *,
    ib_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        connection_config = load_ibkr_read_only_config(target.config_path)
        result = {
            "name": target.name,
            "config_path": target.config_path.as_posix(),
            "host": connection_config.host,
            "port": connection_config.port,
            "client_id": connection_config.client_id,
            "expected_account": connection_config.expected_account,
            "trading_mode": connection_config.trading_mode,
            "ok": False,
            "account_ok": False,
            "connected": False,
            "managed_accounts": [],
            "server_time": None,
            "latency_ms": None,
            "error": None,
        }
        client = IBKRReadOnlyClient(connection_config, ib_factory=ib_factory)
        try:
            client.connect()
            health = client.health_check()
            accounts = list(health.get("managed_accounts", []))
            account_ok = not connection_config.expected_account or connection_config.expected_account in accounts
            result.update(
                {
                    "ok": bool(account_ok),
                    "account_ok": bool(account_ok),
                    "connected": bool(health.get("connected")),
                    "managed_accounts": accounts,
                    "server_time": health.get("server_time"),
                    "error": None if account_ok else f"expected account {connection_config.expected_account} not in managed accounts",
                }
            )
        finally:
            client.disconnect()
        return result | {"latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:  # noqa: BLE001
        return {
            "name": target.name,
            "config_path": target.config_path.as_posix(),
            "host": None,
            "port": None,
            "client_id": None,
            "expected_account": None,
            "trading_mode": None,
            "ok": False,
            "account_ok": False,
            "connected": False,
            "managed_accounts": [],
            "server_time": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": repr(exc),
        }


def write_watchdog_status(path: str | Path, payload: dict[str, Any]) -> None:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_watchdog_status(path: str | Path) -> dict[str, Any]:
    status_path = Path(path)
    if not status_path.exists():
        return {}
    try:
        raw = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": repr(exc)}
    return raw if isinstance(raw, dict) else {}


def _failure_signature(status: dict[str, Any]) -> dict[str, str]:
    checks = list(status.get("checks", []) or [])
    return {
        str(check.get("name") or check.get("config_path") or "target"): str(check.get("error") or "failed")
        for check in checks
        if not check.get("ok")
    }


def maybe_record_watchdog_event(
    config: IBKRWatchdogConfig,
    *,
    previous_status: dict[str, Any],
    current_status: dict[str, Any],
) -> dict[str, Any] | None:
    previous_ok = previous_status.get("ok") if isinstance(previous_status.get("ok"), bool) else None
    current_ok = bool(current_status.get("ok"))
    previous_signature = _failure_signature(previous_status)
    current_signature = _failure_signature(current_status)

    if current_ok and previous_ok is False:
        event_type = "ibkr_watchdog_recovered"
        severity = "info"
    elif not current_ok and (previous_ok is not False or previous_signature != current_signature):
        event_type = "ibkr_watchdog_degraded"
        severity = "critical"
    else:
        return None

    return append_operational_event(
        {
            "event_type": event_type,
            "component": "ibkr_watchdog",
            "severity": severity,
            "status": current_status.get("status"),
            "action": current_status.get("action"),
            "message": current_status.get("message"),
            "status_path": config.status_path.as_posix(),
            "failed_targets": list(current_signature.keys()),
            "previous_status": previous_status.get("status"),
            "previous_ok": previous_ok,
            "checks": current_status.get("checks", []),
        },
        path=config.events_path,
    )


def run_watchdog(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    ib_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    config = load_watchdog_config(config_path)
    previous_status = read_watchdog_status(config.status_path)
    checks = [_check_target(target, ib_factory=ib_factory) for target in config.targets]
    ok = bool(checks) and all(bool(check.get("ok")) for check in checks)
    status = {
        "created_at_utc": utc_now(),
        "status": "ok" if ok else "degraded",
        "ok": ok,
        "action": "none" if ok else "manual_intervention_required",
        "message": "IBKR API is reachable for all configured paper targets"
        if ok
        else "IBKR API is not healthy; inspect IB Gateway login/API state before trading",
        "checks": checks,
        "config": config.to_dict(),
    }
    try:
        event = maybe_record_watchdog_event(config, previous_status=previous_status, current_status=status)
        if event:
            status["event"] = {"recorded": True, "event_id": event["event_id"], "event_type": event["event_type"]}
    except Exception as exc:  # noqa: BLE001
        status.update(
            {
                "status": "degraded",
                "ok": False,
                "action": "manual_intervention_required",
                "message": "Operational event log write failed; inspect disk permissions before trading",
                "event_log_error": repr(exc),
            }
        )
    write_watchdog_status(config.status_path, status)
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only IBKR Gateway watchdog")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-fail", action="store_true", help="always exit 0 after writing status")
    args = parser.parse_args(argv)

    status = run_watchdog(config_path=args.config)
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    if not args.no_fail and not status["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

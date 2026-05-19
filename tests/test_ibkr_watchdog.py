from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import yaml

from src.execution.ibkr_watchdog import load_watchdog_config, run_watchdog


class FakeIB:
    def __init__(self, *, accounts: list[str] | None = None, fail_connect: bool = False) -> None:
        self.accounts = accounts or ["DU123"]
        self.fail_connect = fail_connect
        self.connected = False

    def connect(self, host: str, port: int, *, clientId: int, timeout: float, readonly: bool) -> None:
        if self.fail_connect:
            raise ConnectionRefusedError("paper gateway unavailable")
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return self.accounts

    def reqCurrentTime(self) -> datetime:
        return datetime(2026, 5, 19, 13, 30, tzinfo=timezone.utc)

    def accountSummary(self) -> list[SimpleNamespace]:
        return []

    def positions(self) -> list[SimpleNamespace]:
        return []

    def openTrades(self) -> list[SimpleNamespace]:
        return []


def _write_readonly_config(path, *, account: str = "DU123") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "connection": {
                    "broker": "ibkr",
                    "gateway": "ib_gateway",
                    "trading_mode": "paper",
                    "host": "127.0.0.1",
                    "port": 4002,
                    "client_id": 71,
                    "timeout_seconds": 1,
                    "expected_account": account,
                },
                "safety": {
                    "read_only": True,
                    "allow_orders": False,
                    "require_paper_account": True,
                    "require_paper_port": True,
                },
                "outputs": {"output_dir": "results/paper/ibkr_read_only"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_watchdog_config(path, readonly_path, status_path, events_path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "watchdog": {"targets": [{"name": "paper", "config_path": readonly_path.as_posix()}]},
                "outputs": {
                    "output_dir": status_path.parent.as_posix(),
                    "status_path": status_path.as_posix(),
                    "events_path": events_path.as_posix(),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_events(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_watchdog_writes_ok_status(tmp_path) -> None:
    readonly = tmp_path / "ibkr.yaml"
    watchdog = tmp_path / "watchdog.yaml"
    status_path = tmp_path / "status.yaml"
    events_path = tmp_path / "events.jsonl"
    _write_readonly_config(readonly)
    _write_watchdog_config(watchdog, readonly, status_path, events_path)

    status = run_watchdog(config_path=watchdog, ib_factory=lambda: FakeIB())

    assert status["ok"] is True
    assert status["checks"][0]["account_ok"] is True
    assert yaml.safe_load(status_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert _read_events(events_path) == []


def test_watchdog_marks_connection_failure_degraded(tmp_path) -> None:
    readonly = tmp_path / "ibkr.yaml"
    watchdog = tmp_path / "watchdog.yaml"
    status_path = tmp_path / "status.yaml"
    events_path = tmp_path / "events.jsonl"
    _write_readonly_config(readonly)
    _write_watchdog_config(watchdog, readonly, status_path, events_path)

    status = run_watchdog(config_path=watchdog, ib_factory=lambda: FakeIB(fail_connect=True))

    assert status["ok"] is False
    assert status["status"] == "degraded"
    assert status["action"] == "manual_intervention_required"
    assert "ConnectionRefusedError" in status["checks"][0]["error"]
    events = _read_events(events_path)
    assert [event["event_type"] for event in events] == ["ibkr_watchdog_degraded"]
    assert events[0]["severity"] == "critical"


def test_watchdog_records_recovery_after_degraded_status(tmp_path) -> None:
    readonly = tmp_path / "ibkr.yaml"
    watchdog = tmp_path / "watchdog.yaml"
    status_path = tmp_path / "status.yaml"
    events_path = tmp_path / "events.jsonl"
    _write_readonly_config(readonly)
    _write_watchdog_config(watchdog, readonly, status_path, events_path)

    run_watchdog(config_path=watchdog, ib_factory=lambda: FakeIB(fail_connect=True))
    status = run_watchdog(config_path=watchdog, ib_factory=lambda: FakeIB())

    assert status["ok"] is True
    events = _read_events(events_path)
    assert [event["event_type"] for event in events] == ["ibkr_watchdog_degraded", "ibkr_watchdog_recovered"]
    assert events[-1]["severity"] == "info"


def test_load_default_watchdog_config() -> None:
    config = load_watchdog_config()

    assert [target.name for target in config.targets] == ["h1c-paper", "c2-paper"]
    assert config.status_path.as_posix() == "results/paper/ibkr_watchdog/status.yaml"
    assert config.events_path.as_posix() == "results/paper/operational_events/events.jsonl"

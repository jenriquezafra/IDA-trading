from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.execution.h1c_auto_runner import run_h1c_auto
from src.execution.operational_events import DEFAULT_OPERATIONAL_EVENTS_PATH, append_operational_event
from src.execution.paper_state_store import utc_now


DEFAULT_CONFIG_PATH = Path("configs/execution/h1c_auto_daemon.yaml")


@dataclass(frozen=True)
class H1CAutoDaemonConfig:
    auto_runner_config_path: Path
    open_scan_interval_seconds: int
    pre_open_wakeup_minutes: int
    pre_open_scan_interval_seconds: int
    active_reconciliation_interval_seconds: int
    error_sleep_seconds: int
    max_error_sleep_seconds: int
    calendar: str
    skip_cboe: bool
    skip_download: bool
    output_dir: Path
    status_path: Path
    events_path: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "H1CAutoDaemonConfig":
        daemon = dict(raw.get("daemon", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            auto_runner_config_path=Path(daemon.get("auto_runner_config_path", "configs/execution/h1c_auto_runner.yaml")),
            open_scan_interval_seconds=int(daemon.get("open_scan_interval_seconds", 900)),
            pre_open_wakeup_minutes=int(daemon.get("pre_open_wakeup_minutes", 15)),
            pre_open_scan_interval_seconds=int(daemon.get("pre_open_scan_interval_seconds", 60)),
            active_reconciliation_interval_seconds=int(daemon.get("active_reconciliation_interval_seconds", 60)),
            error_sleep_seconds=int(daemon.get("error_sleep_seconds", 300)),
            max_error_sleep_seconds=int(daemon.get("max_error_sleep_seconds", 900)),
            calendar=str(daemon.get("calendar", "NYSE")).strip(),
            skip_cboe=bool(daemon.get("skip_cboe", True)),
            skip_download=bool(daemon.get("skip_download", False)),
            output_dir=Path(outputs.get("output_dir", "results/paper/h1c_auto_runner")),
            status_path=Path(outputs.get("status_path", "results/paper/h1c_auto_runner/daemon_status.yaml")),
            events_path=Path(outputs.get("events_path", DEFAULT_OPERATIONAL_EVENTS_PATH)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.open_scan_interval_seconds <= 0:
            raise ValueError("open_scan_interval_seconds must be positive")
        if self.pre_open_wakeup_minutes < 0:
            raise ValueError("pre_open_wakeup_minutes must be non-negative")
        if self.pre_open_scan_interval_seconds <= 0:
            raise ValueError("pre_open_scan_interval_seconds must be positive")
        if self.active_reconciliation_interval_seconds <= 0:
            raise ValueError("active_reconciliation_interval_seconds must be positive")
        if self.error_sleep_seconds <= 0:
            raise ValueError("error_sleep_seconds must be positive")
        if self.max_error_sleep_seconds < self.error_sleep_seconds:
            raise ValueError("max_error_sleep_seconds must be >= error_sleep_seconds")
        if not self.calendar:
            raise ValueError("calendar is required")
        if not self.events_path.as_posix():
            raise ValueError("events_path is required")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["auto_runner_config_path"] = self.auto_runner_config_path.as_posix()
        data["output_dir"] = self.output_dir.as_posix()
        data["status_path"] = self.status_path.as_posix()
        data["events_path"] = self.events_path.as_posix()
        return data


def load_daemon_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H1CAutoDaemonConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return H1CAutoDaemonConfig.from_mapping(raw)


def _calendar(name: str):
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError("pandas_market_calendars is required for adaptive H1c scheduling") from exc
    return mcal.get_calendar(name)


def _utc_timestamp(value: pd.Timestamp | str | None = None) -> pd.Timestamp:
    ts = pd.Timestamp.utcnow() if value is None else pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def next_scan_decision(now: pd.Timestamp, config: H1CAutoDaemonConfig) -> dict[str, Any]:
    now_utc = _utc_timestamp(now)
    calendar = _calendar(config.calendar)
    end = (now_utc + pd.Timedelta(days=14)).date().isoformat()
    schedule = calendar.schedule(start_date=now_utc.date().isoformat(), end_date=end)
    if schedule.empty:
        return {
            "should_run_now": False,
            "sleep_seconds": config.error_sleep_seconds,
            "reason": "calendar_schedule_empty",
            "market_open": False,
            "next_open_utc": None,
        }

    for _, row in schedule.iterrows():
        market_open = _utc_timestamp(row["market_open"])
        market_close = _utc_timestamp(row["market_close"])
        pre_open = market_open - pd.Timedelta(minutes=config.pre_open_wakeup_minutes)
        if market_open <= now_utc <= market_close:
            return {
                "should_run_now": True,
                "sleep_seconds": config.open_scan_interval_seconds,
                "reason": "market_open",
                "market_open": True,
                "next_open_utc": market_open.isoformat(),
                "market_close_utc": market_close.isoformat(),
            }
        if pre_open <= now_utc < market_open:
            seconds_to_open = max(1, int((market_open - now_utc).total_seconds()))
            return {
                "should_run_now": True,
                "sleep_seconds": min(config.pre_open_scan_interval_seconds, seconds_to_open),
                "reason": "pre_open_window",
                "market_open": False,
                "next_open_utc": market_open.isoformat(),
                "market_close_utc": market_close.isoformat(),
            }
        if now_utc < pre_open:
            return {
                "should_run_now": False,
                "sleep_seconds": max(1, int((pre_open - now_utc).total_seconds())),
                "reason": "waiting_for_pre_open_window",
                "market_open": False,
                "next_open_utc": market_open.isoformat(),
                "pre_open_utc": pre_open.isoformat(),
            }
    return {
        "should_run_now": False,
        "sleep_seconds": config.error_sleep_seconds,
        "reason": "no_future_market_open_found",
        "market_open": False,
        "next_open_utc": None,
    }


def error_sleep_seconds_for_streak(config: H1CAutoDaemonConfig, error_streak: int) -> int:
    streak = max(1, int(error_streak))
    return min(config.max_error_sleep_seconds, config.error_sleep_seconds * streak)


def reconciliation_has_live_activity(reconciliation: dict[str, Any]) -> bool:
    count_keys = [
        "target_open_orders",
        "account_nonzero_positions",
        "account_open_orders",
    ]
    for key in count_keys:
        try:
            if float(reconciliation.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    try:
        return abs(float(reconciliation.get("target_position_qty", 0) or 0)) > 0
    except (TypeError, ValueError):
        return False


def reconciliation_allows_active_scan(reconciliation: dict[str, Any]) -> bool:
    decision = str(reconciliation.get("decision", "") or "")
    severity = str(reconciliation.get("severity", "") or "")
    ok_decisions = {
        "OK_PENDING_ENTRY",
        "OK_OPEN",
        "OK_PENDING_EXIT",
        "FILL_DETECTED_PENDING_ENTRY",
        "FILL_DETECTED_PENDING_EXIT",
    }
    return decision in ok_decisions or (severity == "ok" and reconciliation_has_live_activity(reconciliation))


def reconciliation_allows_cycle(reconciliation: dict[str, Any]) -> bool:
    decision = str(reconciliation.get("decision", "") or "")
    return decision in {
        "OK_FLAT",
        "OK_OPEN",
        "FILL_DETECTED_PENDING_ENTRY",
        "FILL_DETECTED_PENDING_EXIT",
    }


def apply_post_run_sleep_policy(
    decision: dict[str, Any],
    runner_summary: dict[str, Any],
    config: H1CAutoDaemonConfig,
) -> dict[str, Any]:
    adjusted = dict(decision)
    base_sleep = int(adjusted["sleep_seconds"])
    pre_reconciliation = dict(runner_summary.get("pre_trade_reconciliation", {}) or {})
    post_reconciliation = dict(runner_summary.get("post_execution_reconciliation", {}) or {})
    live_activity = reconciliation_has_live_activity(post_reconciliation) or reconciliation_has_live_activity(pre_reconciliation)
    active_scan_allowed = reconciliation_allows_active_scan(post_reconciliation) or reconciliation_allows_active_scan(pre_reconciliation)
    cycle_allowed = reconciliation_allows_cycle(pre_reconciliation)
    adjusted["base_sleep_seconds"] = base_sleep
    adjusted["live_activity_detected"] = live_activity
    adjusted["active_scan_allowed"] = active_scan_allowed
    adjusted["cycle_allowed_by_reconciliation"] = cycle_allowed
    adjusted["sleep_policy_reason"] = adjusted.get("reason", "")
    if adjusted.get("market_open") and live_activity and active_scan_allowed:
        adjusted["sleep_seconds"] = min(base_sleep, config.active_reconciliation_interval_seconds)
        adjusted["sleep_policy_reason"] = "active_order_or_position_reconciliation"
    elif adjusted.get("market_open") and live_activity:
        adjusted["sleep_seconds"] = base_sleep
        adjusted["sleep_policy_reason"] = "blocked_reconciliation_uses_open_interval"
    return adjusted


def write_status(path: str | Path, payload: dict[str, Any]) -> None:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def record_daemon_event(
    config: H1CAutoDaemonConfig,
    *,
    component: str,
    event_type: str,
    severity: str,
    status: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    event = {
        "event_type": event_type,
        "component": component,
        "severity": severity,
        "status_path": config.status_path.as_posix(),
        "error_streak": status.get("error_streak"),
        "sleep_seconds": status.get("sleep_seconds") or dict(status.get("scheduler", {}) or {}).get("sleep_seconds"),
        "iteration_started_at_utc": status.get("iteration_started_at_utc"),
    }
    if "error" in status:
        event["error"] = status.get("error")
        event["error_type"] = status.get("error_type")
    if extra:
        event.update(extra)
    try:
        return append_operational_event(event, path=config.events_path)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "event_type": "operational_event_write_failed",
                    "component": component,
                    "severity": "critical",
                    "error": repr(exc),
                    "intended_event_type": event_type,
                    "status_path": config.status_path.as_posix(),
                    "created_at_utc": utc_now(),
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
        return None


def run_daemon(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    once: bool = False,
    max_iterations: int | None = None,
) -> None:
    config = load_daemon_config(config_path)
    iterations = 0
    error_streak = 0
    while True:
        iterations += 1
        started = utc_now()
        iteration_started_monotonic = time.perf_counter()
        try:
            previous_error_streak = error_streak
            decision = next_scan_decision(pd.Timestamp.now(tz="UTC"), config)
            runner_summary: dict[str, Any] = {}
            runner_paths: dict[str, str] = {}
            if decision["should_run_now"]:
                paths, manifest = run_h1c_auto(
                    config_path=config.auto_runner_config_path,
                    skip_download=config.skip_download,
                    skip_cboe=config.skip_cboe,
                    output_dir=config.output_dir,
                )
                runner_paths = {key: str(value) for key, value in asdict(paths).items()}
                runner_summary = {
                    "decision": manifest.get("decision"),
                    "reason": manifest.get("reason"),
                    "market": manifest.get("market", {}),
                    "pre_trade_reconciliation": manifest.get("pre_trade_reconciliation", {}),
                    "post_execution_reconciliation": manifest.get("post_execution_reconciliation", {}),
                    "execution": manifest.get("execution", {}).get("summary", {}),
                }
                decision = apply_post_run_sleep_policy(decision, runner_summary, config)
            error_streak = 0
            status = {
                "created_at_utc": utc_now(),
                "iteration_started_at_utc": started,
                "scheduler": decision,
                "runner_paths": runner_paths,
                "runner_summary": runner_summary,
                "latency": {"iteration_seconds": round(time.perf_counter() - iteration_started_monotonic, 3)},
                "error_streak": error_streak,
                "config": config.to_dict(),
            }
            write_status(config.status_path, status)
            if previous_error_streak > 0:
                event = record_daemon_event(
                    config,
                    component="h1c_auto_daemon",
                    event_type="daemon_recovered",
                    severity="info",
                    status=status,
                    extra={"previous_error_streak": previous_error_streak},
                )
                if event:
                    status["event"] = {"recorded": True, "event_id": event["event_id"], "event_type": event["event_type"]}
                    write_status(config.status_path, status)
            print(json.dumps(status, sort_keys=True, default=str), flush=True)
            if once or (max_iterations is not None and iterations >= max_iterations):
                return
            time.sleep(int(decision["sleep_seconds"]))
        except Exception as exc:  # noqa: BLE001
            error_streak += 1
            sleep_seconds = error_sleep_seconds_for_streak(config, error_streak)
            status = {
                "created_at_utc": utc_now(),
                "iteration_started_at_utc": started,
                "error": repr(exc),
                "error_type": type(exc).__name__,
                "error_streak": error_streak,
                "sleep_seconds": sleep_seconds,
                "latency": {"iteration_seconds": round(time.perf_counter() - iteration_started_monotonic, 3)},
                "config": config.to_dict(),
            }
            write_status(config.status_path, status)
            event = record_daemon_event(
                config,
                component="h1c_auto_daemon",
                event_type="daemon_error",
                severity="critical" if error_streak >= 3 else "warning",
                status=status,
            )
            if event:
                status["event"] = {"recorded": True, "event_id": event["event_id"], "event_type": event["event_type"]}
                write_status(config.status_path, status)
            print(json.dumps(status, sort_keys=True, default=str), flush=True)
            if once or (max_iterations is not None and iterations >= max_iterations):
                return
            time.sleep(sleep_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Adaptive scheduler daemon for the H1c paper auto runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--once", action="store_true", help="evaluate scheduler once and exit")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args(argv)
    run_daemon(config_path=args.config, once=args.once, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()

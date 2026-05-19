from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.execution.h1c_auto_daemon import (
    apply_post_run_sleep_policy,
    error_sleep_seconds_for_streak,
    load_daemon_config,
    next_scan_decision,
    record_daemon_event,
    write_status,
)
from src.execution.paper_state_store import utc_now
from src.execution.setup_signal_auto_runner import run_setup_signal_auto


DEFAULT_CONFIG_PATH = Path("configs/execution/c2_auto_daemon.yaml")


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
            runner_summary: dict[str, object] = {}
            runner_paths: dict[str, str] = {}
            if decision["should_run_now"]:
                paths, manifest = run_setup_signal_auto(
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
                    component="setup_signal_auto_daemon",
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
                component="setup_signal_auto_daemon",
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
    parser = argparse.ArgumentParser(description="Adaptive scheduler daemon for setup-signal paper auto runners")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--once", action="store_true", help="evaluate scheduler once and exit")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args(argv)
    run_daemon(config_path=args.config, once=args.once, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()

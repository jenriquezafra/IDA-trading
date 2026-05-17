from __future__ import annotations

import pandas as pd

from src.execution.h1c_auto_daemon import (
    H1CAutoDaemonConfig,
    apply_post_run_sleep_policy,
    error_sleep_seconds_for_streak,
    next_scan_decision,
    reconciliation_allows_active_scan,
    reconciliation_allows_cycle,
    reconciliation_has_live_activity,
)


def _config() -> H1CAutoDaemonConfig:
    return H1CAutoDaemonConfig.from_mapping(
        {
            "daemon": {
                "auto_runner_config_path": "configs/execution/h1c_auto_runner.yaml",
                "open_scan_interval_seconds": 900,
                "pre_open_wakeup_minutes": 15,
                "pre_open_scan_interval_seconds": 900,
                "active_reconciliation_interval_seconds": 60,
                "error_sleep_seconds": 300,
                "max_error_sleep_seconds": 900,
                "calendar": "NYSE",
            },
            "outputs": {"output_dir": "results/paper/h1c_auto_runner", "status_path": "results/paper/h1c_auto_runner/daemon_status.yaml"},
        }
    )


def test_next_scan_sleeps_until_pre_open_window() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 09:30:00Z"), _config())

    assert decision["should_run_now"] is False
    assert decision["reason"] == "waiting_for_pre_open_window"
    assert decision["sleep_seconds"] == 13500


def test_next_scan_runs_during_pre_open_window() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 13:20:00Z"), _config())

    assert decision["should_run_now"] is True
    assert decision["reason"] == "pre_open_window"
    assert decision["sleep_seconds"] == 600


def test_next_scan_runs_every_open_interval_during_market() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 15:00:00Z"), _config())

    assert decision["should_run_now"] is True
    assert decision["reason"] == "market_open"
    assert decision["sleep_seconds"] == 900


def test_post_run_policy_uses_fast_reconciliation_when_position_or_order_is_alive() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 15:00:00Z"), _config())

    adjusted = apply_post_run_sleep_policy(
        decision,
        {"pre_trade_reconciliation": {"decision": "OK_OPEN", "severity": "ok", "account_nonzero_positions": 1, "account_open_orders": 0}},
        _config(),
    )

    assert adjusted["sleep_seconds"] == 60
    assert adjusted["base_sleep_seconds"] == 900
    assert adjusted["sleep_policy_reason"] == "active_order_or_position_reconciliation"
    assert adjusted["live_activity_detected"] is True


def test_post_run_policy_keeps_open_interval_when_account_is_flat() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 15:00:00Z"), _config())

    adjusted = apply_post_run_sleep_policy(
        decision,
        {"pre_trade_reconciliation": {"account_nonzero_positions": 0, "account_open_orders": 0, "target_position_qty": 0.0}},
        _config(),
    )

    assert adjusted["sleep_seconds"] == 900
    assert adjusted["live_activity_detected"] is False


def test_post_run_policy_does_not_fast_scan_blocked_unrelated_activity() -> None:
    decision = next_scan_decision(pd.Timestamp("2026-05-11 15:00:00Z"), _config())

    adjusted = apply_post_run_sleep_policy(
        decision,
        {
            "pre_trade_reconciliation": {
                "decision": "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS",
                "severity": "block",
                "account_open_orders": 3,
            }
        },
        _config(),
    )

    assert adjusted["sleep_seconds"] == 900
    assert adjusted["active_scan_allowed"] is False
    assert adjusted["sleep_policy_reason"] == "blocked_reconciliation_uses_open_interval"


def test_reconciliation_has_live_activity_from_target_position_or_orders() -> None:
    assert reconciliation_has_live_activity({"target_position_qty": -10.0}) is True
    assert reconciliation_has_live_activity({"target_open_orders": 1}) is True
    assert reconciliation_has_live_activity({"account_nonzero_positions": 0, "account_open_orders": 0, "target_position_qty": 0.0}) is False


def test_reconciliation_cycle_and_active_scan_decisions() -> None:
    assert reconciliation_allows_cycle({"decision": "OK_FLAT"}) is True
    assert reconciliation_allows_cycle({"decision": "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS"}) is False
    assert reconciliation_allows_active_scan({"decision": "OK_OPEN", "target_position_qty": -1.0}) is True
    assert reconciliation_allows_active_scan({"decision": "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS", "account_open_orders": 1}) is False


def test_error_backoff_is_five_ten_fifteen_minutes() -> None:
    config = _config()

    assert error_sleep_seconds_for_streak(config, 1) == 300
    assert error_sleep_seconds_for_streak(config, 2) == 600
    assert error_sleep_seconds_for_streak(config, 3) == 900
    assert error_sleep_seconds_for_streak(config, 4) == 900

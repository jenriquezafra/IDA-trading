from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.execution.h1c_order_executor import plan_fingerprint, run_h1c_order_execution, validate_h1c_plan_for_execution
from src.execution.flatten_executor import IBKRFlattenExecutorConfig


def _executor_config() -> IBKRFlattenExecutorConfig:
    return IBKRFlattenExecutorConfig.from_mapping(
        {
            "connection": {"trading_mode": "paper", "host": "127.0.0.1", "port": 4002, "client_id": 73, "timeout_seconds": 1, "expected_account": "DU123"},
            "safety": {
                "execution_enabled": False,
                "allow_orders": False,
                "require_paper_account": True,
                "require_paper_port": True,
                "require_no_open_trades": True,
                "require_market_open": False,
                "require_account_confirmation": True,
                "require_fingerprint_confirmation": True,
                "require_env_confirmation": True,
                "env_confirmation_var": "IBKR_H1C_EXECUTION_CONFIRM",
                "max_orders": 1,
                "max_total_notional_at_avg_cost": 1000,
                "allowed_sec_types": ["STK"],
                "allowed_order_types": ["MKT"],
                "allowed_tifs": ["DAY"],
            },
            "outputs": {"output_dir": "unused"},
        }
    )


def _write_plan(root: Path, decision: str = "ready_for_review", action: str = "SELL") -> Path:
    plan_dir = root / "plan"
    plan_dir.mkdir()
    pd.DataFrame(
        [
            {
                "account": "DU123",
                "symbol": "QQQ",
                "sec_type": "STK",
                "currency": "USD",
                "action": action,
                "quantity": 1.0,
                "order_type": "MKT",
                "tif": "DAY",
                "outside_rth": False,
                "routing_exchange": "SMART",
                "primary_exchange": "NASDAQ",
                "transmit": False,
                "dry_run": True,
                "status": "planned",
                "approx_notional_at_ticket_price": 100.0,
            }
        ]
    ).to_parquet(plan_dir / "orders.parquet", index=False)
    (plan_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"run_type": "h1c_paper_order_plan", "created_at_utc": "2026-05-10T12:00:00Z"},
                "summary": {"decision": decision, "expected_account": "DU123", "dry_run": True, "transmit": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return plan_dir


def test_validate_h1c_plan_for_execution_accepts_ready_plan(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path)
    orders, preflight = validate_h1c_plan_for_execution(plan_dir=plan_dir, config=_executor_config())

    assert len(orders) == 1
    assert preflight["plan_fingerprint"] == plan_fingerprint(plan_dir)


def test_validate_h1c_plan_for_execution_accepts_buy_exit_plan(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path, action="BUY")

    orders, preflight = validate_h1c_plan_for_execution(plan_dir=plan_dir, config=_executor_config())

    assert orders.iloc[0]["action"] == "BUY"
    assert preflight["offline_valid"] is True


def test_validate_h1c_plan_for_execution_rejects_blocked_plan(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path, decision="blocked_reconciliation")

    with pytest.raises(ValueError, match="ready_for_review"):
        validate_h1c_plan_for_execution(plan_dir=plan_dir, config=_executor_config())


def test_run_h1c_order_execution_default_is_dry_run(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path)
    config_path = tmp_path / "executor.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "connection": {"trading_mode": "paper", "host": "127.0.0.1", "port": 4002, "client_id": 73, "timeout_seconds": 1, "expected_account": "DU123"},
                "safety": {
                    "execution_enabled": False,
                    "allow_orders": False,
                    "require_paper_account": True,
                    "require_paper_port": True,
                    "require_no_open_trades": True,
                    "require_market_open": False,
                    "require_account_confirmation": True,
                    "require_fingerprint_confirmation": True,
                    "require_env_confirmation": True,
                    "env_confirmation_var": "IBKR_H1C_EXECUTION_CONFIRM",
                    "max_orders": 1,
                    "max_total_notional_at_avg_cost": 1000,
                    "allowed_sec_types": ["STK"],
                    "allowed_order_types": ["MKT"],
                    "allowed_tifs": ["DAY"],
                },
                "outputs": {"output_dir": (tmp_path / "exec_runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    paths, summary = run_h1c_order_execution(plan_dir=plan_dir, config_path=config_path)

    assert paths.report_path.exists()
    assert summary["unlock"]["unlocked"] is False
    assert summary["submitted_orders"] == 0

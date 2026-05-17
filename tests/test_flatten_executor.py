from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from src.execution.flatten_executor import (
    IBKRFlattenExecutorConfig,
    plan_fingerprint,
    run_flatten_execution,
    validate_execution_unlock,
    validate_plan_for_execution,
)


def _config(**overrides: object) -> IBKRFlattenExecutorConfig:
    raw = {
        "connection": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 72,
            "timeout_seconds": 1,
            "trading_mode": "paper",
            "expected_account": "DU123",
        },
        "safety": {
            "execution_enabled": False,
            "allow_orders": False,
            "require_paper_account": True,
            "require_paper_port": True,
            "require_no_open_trades": True,
            "require_market_open": False,
            "allow_opg_outside_rth_submission": True,
            "require_account_confirmation": True,
            "require_fingerprint_confirmation": True,
            "require_env_confirmation": True,
            "env_confirmation_var": "IBKR_PAPER_EXECUTION_CONFIRM",
            "max_orders": 50,
            "max_total_notional_at_avg_cost": 100000,
            "allowed_sec_types": ["STK"],
            "allowed_order_types": ["MKT"],
            "allowed_tifs": ["DAY", "OPG"],
        },
        "outputs": {"output_dir": "unused"},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        raw[section][key] = value
    return IBKRFlattenExecutorConfig.from_mapping(raw)


def _write_config(path: Path, *, allow_orders: bool = False, execution_enabled: bool = False) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "connection": {
                    "host": "127.0.0.1",
                    "port": 4002,
                    "client_id": 72,
                    "timeout_seconds": 1,
                    "trading_mode": "paper",
                    "expected_account": "DU123",
                },
                "safety": {
                    "execution_enabled": execution_enabled,
                    "allow_orders": allow_orders,
                    "require_paper_account": True,
                    "require_paper_port": True,
                    "require_no_open_trades": True,
                    "require_market_open": False,
                    "allow_opg_outside_rth_submission": True,
                    "require_account_confirmation": True,
                    "require_fingerprint_confirmation": True,
                    "require_env_confirmation": True,
                    "env_confirmation_var": "IBKR_PAPER_EXECUTION_CONFIRM",
                    "max_orders": 50,
                    "max_total_notional_at_avg_cost": 100000,
                    "allowed_sec_types": ["STK"],
                    "allowed_order_types": ["MKT"],
                    "allowed_tifs": ["DAY", "OPG"],
                },
                "outputs": {"output_dir": "unused"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_plan(root: Path, *, decision: str = "ready_for_review", open_trades: int = 0) -> Path:
    plan_dir = root / "plan"
    plan_dir.mkdir()
    orders = pd.DataFrame(
        [
            {
                "account": "DU123",
                "con_id": 1,
                "symbol": "AAPL",
                "sec_type": "STK",
                "currency": "USD",
                "local_symbol": "AAPL",
                "trading_class": "NMS",
                "current_position": 3.0,
                "action": "SELL",
                "quantity": 3.0,
                "order_type": "MKT",
                "tif": "DAY",
                "outside_rth": False,
                "routing_exchange": "SMART",
                "primary_exchange": "NASDAQ",
                "transmit": False,
                "dry_run": True,
                "approx_notional_at_avg_cost": 300.0,
                "status": "planned",
            }
        ]
    )
    orders.to_parquet(plan_dir / "orders.parquet", index=False)
    (plan_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"run_type": "ibkr_flatten_plan", "created_at_utc": "2026-05-10T12:00:00Z"},
                "source": {"snapshot_dir": "snap"},
                "summary": {
                    "decision": decision,
                    "expected_account": "DU123",
                    "dry_run": True,
                    "transmit": False,
                    "open_trades": open_trades,
                    "unsupported_positions": 0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return plan_dir


class FakeIB:
    def __init__(self, server_time: str = "2026-05-11 15:00:00") -> None:
        self.connected = False
        self.connect_kwargs: dict[str, object] = {}
        self.placed: list[tuple[object, object]] = []
        self.server_time = server_time

    def connect(self, host: str, port: int, *, clientId: int, timeout: float, readonly: bool) -> None:
        self.connected = True
        self.connect_kwargs = {"host": host, "port": port, "clientId": clientId, "timeout": timeout, "readonly": readonly}

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return ["DU123"]

    def openTrades(self) -> list[object]:
        return []

    def reqCurrentTime(self) -> pd.Timestamp:
        return pd.Timestamp(self.server_time, tz="UTC").to_pydatetime()

    def placeOrder(self, contract: object, order: object) -> SimpleNamespace:
        self.placed.append((contract, order))
        return SimpleNamespace(order=SimpleNamespace(orderId=101, permId=202), orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=3, avgFillPrice=0))


def test_executor_config_rejects_live_or_nonpaper_port() -> None:
    with pytest.raises(ValueError, match="trading_mode=paper"):
        _config(connection__trading_mode="live")
    with pytest.raises(ValueError, match="paper mode"):
        _config(connection__port=7496)


def test_validate_plan_for_execution_accepts_ready_review_plan(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path)
    orders, preflight = validate_plan_for_execution(plan_dir=plan_dir, config=_config())

    assert len(orders) == 1
    assert preflight["offline_valid"] is True
    assert preflight["plan_fingerprint"] == plan_fingerprint(plan_dir)


def test_validate_plan_for_execution_accepts_opg_tif(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path)
    orders = pd.read_parquet(plan_dir / "orders.parquet")
    orders["tif"] = "OPG"
    orders["outside_rth"] = False
    orders.to_parquet(plan_dir / "orders.parquet", index=False)

    validated, preflight = validate_plan_for_execution(plan_dir=plan_dir, config=_config())

    assert validated.iloc[0]["tif"] == "OPG"
    assert preflight["offline_valid"] is True


def test_validate_plan_for_execution_rejects_blocked_plan(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path, decision="blocked_open_trades", open_trades=1)

    with pytest.raises(ValueError, match="ready_for_review"):
        validate_plan_for_execution(plan_dir=plan_dir, config=_config())


def test_execution_unlock_requires_all_confirmations(tmp_path: Path) -> None:
    plan_dir = _write_plan(tmp_path)
    _, preflight = validate_plan_for_execution(plan_dir=plan_dir, config=_config())

    with pytest.raises(ValueError, match="execution_enabled"):
        validate_execution_unlock(
            config=_config(),
            preflight=preflight,
            execute=True,
            transmit_orders=True,
            confirm_account="DU123",
            confirm_fingerprint=preflight["plan_fingerprint"],
            environ={"IBKR_PAPER_EXECUTION_CONFIRM": "DU123"},
        )


def test_run_flatten_execution_default_is_dry_run_without_connection(tmp_path: Path) -> None:
    config_path = tmp_path / "executor.yaml"
    _write_config(config_path)
    plan_dir = _write_plan(tmp_path)
    fake = FakeIB()

    paths, summary = run_flatten_execution(plan_dir=plan_dir, config_path=config_path, output_dir=tmp_path / "runs", ib_factory=lambda: fake)

    assert summary["unlock"]["unlocked"] is False
    assert summary["submitted_orders"] == 0
    assert fake.connect_kwargs == {}
    assert paths.report_path.exists()


def test_run_flatten_execution_submits_only_when_unlocked(tmp_path: Path) -> None:
    config_path = tmp_path / "executor.yaml"
    _write_config(config_path, allow_orders=True, execution_enabled=True)
    plan_dir = _write_plan(tmp_path)
    fingerprint = plan_fingerprint(plan_dir)
    fake = FakeIB()

    paths, summary = run_flatten_execution(
        plan_dir=plan_dir,
        config_path=config_path,
        execute=True,
        transmit_orders=True,
        confirm_account="DU123",
        confirm_fingerprint=fingerprint,
        output_dir=tmp_path / "runs",
        ib_factory=lambda: fake,
        environ={"IBKR_PAPER_EXECUTION_CONFIRM": "DU123"},
    )

    assert fake.connect_kwargs["readonly"] is False
    assert len(fake.placed) == 1
    _, order = fake.placed[0]
    assert order.tif == "DAY"
    assert order.outsideRth is False
    assert summary["submitted_orders"] == 1
    assert paths.submitted_orders_path.exists()


def test_opg_plan_can_pass_live_preflight_outside_rth_when_allowed(tmp_path: Path) -> None:
    config_path = tmp_path / "executor.yaml"
    _write_config(config_path)
    plan_dir = _write_plan(tmp_path)
    orders = pd.read_parquet(plan_dir / "orders.parquet")
    orders["tif"] = "OPG"
    orders["outside_rth"] = False
    orders.to_parquet(plan_dir / "orders.parquet", index=False)
    fake = FakeIB(server_time="2026-05-10 15:00:00")

    _, summary = run_flatten_execution(
        plan_dir=plan_dir,
        config_path=config_path,
        connect_preflight=True,
        output_dir=tmp_path / "runs",
        ib_factory=lambda: fake,
    )

    assert summary["live_preflight"]["nyse_rth_open"] is False
    assert summary["live_preflight"]["all_orders_opg"] is True
    assert summary["live_preflight"]["opg_outside_rth_submission_allowed"] is True
    assert summary["live_preflight"]["live_valid"] is True

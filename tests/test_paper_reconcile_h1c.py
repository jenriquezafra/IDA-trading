from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from src.execution.paper_reconcile_h1c import H1CReconciliationConfig, reconcile_state_snapshot, run_h1c_reconciliation


def _config(tmp_path: Path | None = None, **overrides: object) -> H1CReconciliationConfig:
    raw = {
        "reconciliation": {
            "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
            "account": "DU123",
            "symbol": "QQQ",
            "state_config_path": "state.yaml",
            "ibkr_config_path": "ibkr.yaml",
            "position_tolerance": 1e-6,
            "allow_unrelated_open_orders": False,
            "request_all_open_orders": True,
            "include_executions": True,
        },
        "outputs": {"output_dir": (tmp_path / "runs").as_posix() if tmp_path is not None else "runs"},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        raw[section][key] = value
    return H1CReconciliationConfig.from_mapping(raw)


def _state(status: str = "flat", quantity: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
        "account": "DU123",
        "symbol": "QQQ",
        "status": status,
        "position_unit": -1.0 if status == "open" else 0.0,
        "quantity": quantity,
        "desired_position_unit": -1.0 if status in {"pending_entry", "open"} else 0.0,
        "pending_ticket": None,
        "open_position": None,
        "last_signal_timestamp": "2026-05-08 14:00:00-04:00",
    }


def _positions(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )


def _open_trades(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )


def test_reconcile_flat_state_ok_when_account_has_no_target_exposure() -> None:
    result = reconcile_state_snapshot(state=_state(), positions=_positions([]), open_trades=_open_trades([]), config=_config())

    assert result["decision"] == "OK_FLAT"
    assert result["severity"] == "ok"


def test_reconcile_flat_state_blocks_target_position_drift() -> None:
    result = reconcile_state_snapshot(
        state=_state(),
        positions=_positions([{"account": "DU123", "symbol": "QQQ", "sec_type": "STK", "position": -1.0, "avg_cost": 100.0}]),
        open_trades=_open_trades([]),
        config=_config(),
    )

    assert result["decision"] == "DRIFT_POSITION_MISMATCH"
    assert result["severity"] == "block"


def test_reconcile_pending_entry_ok_when_sell_order_is_open() -> None:
    result = reconcile_state_snapshot(
        state=_state("pending_entry"),
        positions=_positions([]),
        open_trades=_open_trades(
            [
                {
                    "symbol": "QQQ",
                    "sec_type": "STK",
                    "action": "SELL",
                    "status": "PreSubmitted",
                    "total_quantity": 1.0,
                    "remaining": 1.0,
                }
            ]
        ),
        config=_config(),
    )

    assert result["decision"] == "OK_PENDING_ENTRY"


def test_reconcile_pending_exit_detects_flat_after_buy_fill() -> None:
    result = reconcile_state_snapshot(
        state=_state("pending_exit", quantity=1.0),
        positions=_positions([]),
        open_trades=_open_trades([]),
        config=_config(),
    )

    assert result["decision"] == "FILL_DETECTED_PENDING_EXIT"
    assert result["severity"] == "action_required"


def test_reconcile_pending_exit_ok_when_buy_order_is_open() -> None:
    result = reconcile_state_snapshot(
        state=_state("pending_exit", quantity=1.0),
        positions=_positions([{"account": "DU123", "symbol": "QQQ", "sec_type": "STK", "position": -1.0, "avg_cost": 100.0}]),
        open_trades=_open_trades(
            [
                {
                    "symbol": "QQQ",
                    "sec_type": "STK",
                    "action": "BUY",
                    "status": "Submitted",
                    "total_quantity": 1.0,
                    "remaining": 1.0,
                }
            ]
        ),
        config=_config(),
    )

    assert result["decision"] == "OK_PENDING_EXIT"


def test_reconcile_blocks_unrelated_open_orders_by_default() -> None:
    result = reconcile_state_snapshot(
        state=_state(),
        positions=_positions([]),
        open_trades=_open_trades(
            [
                {
                    "symbol": "AAPL",
                    "sec_type": "STK",
                    "action": "SELL",
                    "status": "Submitted",
                    "total_quantity": 1.0,
                    "remaining": 1.0,
                }
            ]
        ),
        config=_config(),
    )

    assert result["decision"] == "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS"
    assert result["severity"] == "block"


class FakeIB:
    def __init__(self) -> None:
        self.connected = False

    def connect(self, host: str, port: int, *, clientId: int, timeout: float, readonly: bool) -> None:
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return ["DU123"]

    def reqCurrentTime(self) -> datetime:
        return datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)

    def accountSummary(self) -> list[SimpleNamespace]:
        return []

    def positions(self) -> list[SimpleNamespace]:
        return []

    def openTrades(self) -> list[SimpleNamespace]:
        return []

    def reqAllOpenOrders(self) -> list[SimpleNamespace]:
        return []

    def reqExecutions(self) -> list[SimpleNamespace]:
        return []


def test_run_reconciliation_writes_report_from_read_only_snapshot(tmp_path: Path) -> None:
    state_config_path = tmp_path / "state_config.yaml"
    state_path = tmp_path / "state.yaml"
    ibkr_config_path = tmp_path / "ibkr.yaml"
    reconcile_config_path = tmp_path / "reconcile.yaml"
    state_path.write_text(yaml.safe_dump(_state(), sort_keys=False), encoding="utf-8")
    state_config_path.write_text(
        yaml.safe_dump(
            {
                "state": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "state_path": state_path.as_posix(),
                    "event_log_path": (tmp_path / "events.parquet").as_posix(),
                    "output_dir": (tmp_path / "state_runs").as_posix(),
                    "allow_send_orders_tickets": False,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ibkr_config_path.write_text(
        yaml.safe_dump(
            {
                "connection": {
                    "host": "127.0.0.1",
                    "port": 4002,
                    "client_id": 71,
                    "timeout_seconds": 1,
                    "trading_mode": "paper",
                    "expected_account": "DU123",
                },
                "safety": {"read_only": True, "allow_orders": False, "require_paper_account": True, "require_paper_port": True},
                "outputs": {"output_dir": (tmp_path / "ibkr").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    reconcile_config_path.write_text(
        yaml.safe_dump(
            {
                "reconciliation": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "state_config_path": state_config_path.as_posix(),
                    "ibkr_config_path": ibkr_config_path.as_posix(),
                },
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    paths, manifest = run_h1c_reconciliation(config_path=reconcile_config_path, ib_factory=lambda: FakeIB())

    assert paths.report_path.exists()
    assert paths.positions_path.exists()
    assert manifest["reconciliation"]["decision"] == "OK_FLAT"

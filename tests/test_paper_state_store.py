from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.execution.paper_state_store import apply_ticket


def _write_config(path: Path, tmp_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "state": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "state_path": (tmp_path / "state.yaml").as_posix(),
                    "event_log_path": (tmp_path / "events.parquet").as_posix(),
                    "output_dir": (tmp_path / "runs").as_posix(),
                    "allow_send_orders_tickets": False,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_ticket(path: Path, *, action: str = "NONE", send_orders: bool = False) -> None:
    signal = action == "SELL"
    exit_signal = action == "BUY"
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "signal_only",
                "send_orders": send_orders,
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "signal_timestamp": "2026-05-08 14:00:00-04:00",
                "session": "2026-05-08",
                "bar_index": 18,
                "entry_rule": "next_open",
                "exit_rule": "fixed_horizon_open",
                "horizon_bars": 6 if signal or exit_signal else None,
                "execution_timing": "next_bar_open_simulated",
                "theoretical_entry_timestamp": "2026-05-08 14:15:00-04:00" if signal or exit_signal else None,
                "theoretical_entry_price": 100.0 if signal or exit_signal else None,
                "theoretical_exit_timestamp": "2026-05-08 15:45:00-04:00" if signal or exit_signal else None,
                "theoretical_exit_price": 98.0 if signal or exit_signal else None,
                "desired_position_unit": -1.0 if signal else 0.0,
                "action": action,
                "quantity": 1.0 if signal or exit_signal else 0.0,
                "order_type": "MKT",
                "time_in_force": "DAY",
                "status": "paper_ticket_only" if signal or exit_signal else "no_signal",
                "reason": "test",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_apply_no_signal_creates_flat_state_and_event(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    _write_config(config_path, tmp_path)
    _write_ticket(ticket_path)

    paths, summary = apply_ticket(ticket_path=ticket_path, config_path=config_path)

    assert paths.state_path.exists()
    assert paths.event_log_path.exists()
    assert summary["state"]["status"] == "flat"
    assert summary["event"]["event_type"] == "flat_no_signal"


def test_apply_sell_ticket_creates_pending_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    _write_config(config_path, tmp_path)
    _write_ticket(ticket_path, action="SELL")

    _, summary = apply_ticket(ticket_path=ticket_path, config_path=config_path)
    events = pd.read_parquet(tmp_path / "events.parquet")

    assert summary["state"]["status"] == "pending_entry"
    assert summary["state"]["pending_ticket"]["action"] == "SELL"
    assert summary["event"]["event_type"] == "pending_entry_created"
    assert len(events) == 1


def test_duplicate_pending_ticket_is_ignored_but_logged(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    _write_config(config_path, tmp_path)
    _write_ticket(ticket_path, action="SELL")

    apply_ticket(ticket_path=ticket_path, config_path=config_path)
    _, summary = apply_ticket(ticket_path=ticket_path, config_path=config_path)

    assert summary["state"]["status"] == "pending_entry"
    assert summary["event"]["event_type"] == "duplicate_pending_ticket_ignored"
    assert len(pd.read_parquet(tmp_path / "events.parquet")) == 2


def test_apply_buy_ticket_creates_pending_exit_from_open_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    _write_config(config_path, tmp_path)
    (tmp_path / "state.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "status": "open",
                "position_unit": -1.0,
                "quantity": 1.0,
                "desired_position_unit": -1.0,
                "pending_ticket": None,
                "open_position": {"quantity": 1.0, "side": "SHORT"},
                "last_signal_timestamp": "2026-05-08 14:00:00-04:00",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_ticket(ticket_path, action="BUY")

    _, summary = apply_ticket(ticket_path=ticket_path, config_path=config_path)

    assert summary["state"]["status"] == "pending_exit"
    assert summary["state"]["pending_ticket"]["action"] == "BUY"
    assert summary["event"]["event_type"] == "pending_exit_created"


def test_rejects_ticket_that_would_send_orders(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    _write_config(config_path, tmp_path)
    _write_ticket(ticket_path, action="SELL", send_orders=True)

    with pytest.raises(ValueError, match="send_orders=false"):
        apply_ticket(ticket_path=ticket_path, config_path=config_path)

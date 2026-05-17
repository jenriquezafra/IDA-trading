from __future__ import annotations

from pathlib import Path

import yaml

from src.execution.h1c_order_plan import H1COrderPlanConfig, build_h1c_order_plan, create_h1c_order_plan


def _config(tmp_path: Path) -> H1COrderPlanConfig:
    return H1COrderPlanConfig.from_mapping(
        {
            "plan": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "allowed_reconciliation_decisions": ["OK_FLAT"],
                "max_quantity": 1,
            },
            "execution_policy": {"order_type": "MKT", "tif": "DAY", "outside_rth": False},
            "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
        }
    )


def _ticket(action: str = "SELL") -> dict[str, object]:
    active = action in {"SELL", "BUY"}
    return {
        "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
        "account": "DU123",
        "symbol": "QQQ",
        "signal_timestamp": "2026-05-08 14:00:00-04:00",
        "action": action,
        "quantity": 1.0 if active else 0.0,
        "theoretical_entry_price": 100.0 if active else None,
        "theoretical_exit_timestamp": "2026-05-08 15:45:00-04:00" if active else None,
        "theoretical_exit_price": 98.0 if active else None,
        "exit_rule": "fixed_horizon_open" if active else None,
        "horizon_bars": 6 if active else None,
    }


def test_build_h1c_order_plan_requires_ok_reconciliation(tmp_path: Path) -> None:
    orders, summary = build_h1c_order_plan(_ticket(), {"reconciliation": {"decision": "ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS"}}, _config(tmp_path))

    assert orders.empty
    assert summary["decision"] == "blocked_reconciliation"


def test_build_h1c_order_plan_creates_single_review_order(tmp_path: Path) -> None:
    orders, summary = build_h1c_order_plan(_ticket(), {"reconciliation": {"decision": "OK_FLAT"}}, _config(tmp_path))

    assert len(orders) == 1
    assert bool(orders.iloc[0]["transmit"]) is False
    assert summary["decision"] == "ready_for_review"


def test_build_h1c_order_plan_creates_buy_exit_on_ok_open(tmp_path: Path) -> None:
    orders, summary = build_h1c_order_plan(_ticket("BUY"), {"reconciliation": {"decision": "OK_OPEN"}}, _config(tmp_path))

    assert len(orders) == 1
    assert orders.iloc[0]["action"] == "BUY"
    assert orders.iloc[0]["intent"] == "h1c_short_exit"
    assert orders.iloc[0]["approx_notional_at_ticket_price"] == 98.0
    assert summary["decision"] == "ready_for_review"


def test_create_h1c_order_plan_writes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ticket_path = tmp_path / "ticket.yaml"
    reconciliation_path = tmp_path / "reconciliation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "plan": {"strategy_id": "qqq_15min_risk_off_short_h1c_v1", "account": "DU123", "symbol": "QQQ"},
                "execution_policy": {"order_type": "MKT", "tif": "DAY"},
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ticket_path.write_text(yaml.safe_dump(_ticket(), sort_keys=False), encoding="utf-8")
    reconciliation_path.write_text(yaml.safe_dump({"reconciliation": {"decision": "OK_FLAT"}}, sort_keys=False), encoding="utf-8")

    paths, summary = create_h1c_order_plan(ticket_path=ticket_path, reconciliation_manifest_path=reconciliation_path, config_path=config_path)

    assert paths.orders_path.exists()
    assert summary["decision"] == "ready_for_review"

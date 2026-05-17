from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.execution.h1c_auto_runner import H1CAutoConfig, account_value, build_exit_ticket, entry_safety_check, exit_due_status, funds_check, run_h1c_auto, size_ticket
from src.execution.h1c_order_executor import H1CExecutionPaths
from src.execution.paper_data_refresh import PaperDataRefreshPaths
from src.execution.paper_h1c_signal_runner import PaperH1CRunnerPaths
from src.execution.paper_reconcile_h1c import H1CReconciliationPaths


def _config(tmp_path: Path) -> H1CAutoConfig:
    return H1CAutoConfig.from_mapping(
        {
            "auto": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "paper_only": True,
                "require_market_open": True,
                "execute_orders": True,
                "transmit_orders": True,
                "min_available_funds_usd": 1000,
                "min_buying_power_usd": 1000,
                "max_order_notional_usd": None,
                "sizing_mode": "buying_power_fraction",
                "capital_fraction": 1.0,
                "reserve_cash_usd": 0,
                "min_quantity": 1,
            },
            "components": {
                "data_refresh_config_path": "refresh.yaml",
                "signal_runner_config_path": "signal.yaml",
                "state_config_path": "state.yaml",
                "reconciliation_config_path": "reconcile.yaml",
                "accounting_config_path": "accounting.yaml",
                "order_plan_config_path": "plan.yaml",
                "order_executor_config_path": "executor.yaml",
            },
            "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
        }
    )


def test_account_value_parses_ibkr_account_summary() -> None:
    frame = pd.DataFrame([{"account": "DU123", "tag": "BuyingPower", "value": "12,345.67", "currency": "USD"}])

    assert account_value(frame, "DU123", "BuyingPower") == 12345.67


def test_account_value_falls_back_to_account_base_currency() -> None:
    frame = pd.DataFrame([{"account": "DU123", "tag": "BuyingPower", "value": "751634.10", "currency": "EUR"}])

    assert account_value(frame, "DU123", "BuyingPower") == 751634.10


def test_funds_check_blocks_missing_or_low_buying_power(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ticket = {"quantity": 1.0, "theoretical_entry_price": 500.0}
    frame = pd.DataFrame(
        [
            {"account": "DU123", "tag": "AvailableFunds", "value": "2000", "currency": "USD"},
            {"account": "DU123", "tag": "BuyingPower", "value": "900", "currency": "USD"},
        ]
    )

    result = funds_check(ticket, frame, config)

    assert result["ok"] is False
    assert "BuyingPower" in result["errors"][0]


def test_size_ticket_uses_buying_power_fraction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ticket = {"action": "SELL", "quantity": 1.0, "theoretical_entry_price": 250.0}
    frame = pd.DataFrame(
        [
            {"account": "DU123", "tag": "AvailableFunds", "value": "1000", "currency": "USD"},
            {"account": "DU123", "tag": "BuyingPower", "value": "5000", "currency": "USD"},
        ]
    )

    sized, summary = size_ticket(ticket, frame, config)

    assert sized["quantity"] == 20.0
    assert summary["original_quantity"] == 1.0


def test_entry_safety_blocks_when_kill_switch_exists(tmp_path: Path) -> None:
    kill_switch = tmp_path / "pause"
    kill_switch.write_text("pause\n", encoding="utf-8")
    config = H1CAutoConfig.from_mapping(
        {
            "auto": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "paper_only": True,
                "kill_switch_path": kill_switch.as_posix(),
            },
            "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
        }
    )

    result = entry_safety_check(config, output_dir=tmp_path / "runs", created_at_utc="2026-05-11T14:00:00Z")

    assert result["ok"] is False
    assert result["kill_switch_exists"] is True
    assert "kill switch file exists" in result["issues"][0]


def test_entry_safety_blocks_at_daily_entry_limit(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "20260511T140000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"created_at_utc": "2026-05-11T14:00:00Z"},
                "signal": {"ticket": {"action": "SELL"}},
                "execution": {"summary": {"submitted_orders": 1}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = H1CAutoConfig.from_mapping(
        {
            "auto": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "paper_only": True,
                "max_daily_entry_orders": 1,
            },
            "outputs": {"output_dir": runs.as_posix()},
        }
    )

    result = entry_safety_check(config, output_dir=runs, created_at_utc="2026-05-11T15:00:00Z")

    assert result["ok"] is False
    assert result["entry_orders_today"] == 1
    assert "daily entry order limit reached" in result["issues"][0]


def test_entry_safety_blocks_when_recent_slippage_exceeds_limit(tmp_path: Path) -> None:
    pnl_path = tmp_path / "pnl.parquet"
    accounting_config_path = tmp_path / "accounting.yaml"
    pd.DataFrame(
        [
            {
                "created_at_utc": "2026-05-11T15:00:00Z",
                "event_type": "entry",
                "entry_slippage_bps": 31.0,
                "exit_slippage_bps": None,
                "realized_pnl": None,
            }
        ]
    ).to_parquet(pnl_path, index=False)
    accounting_config_path.write_text(
        yaml.safe_dump({"accounting": {"pnl_log_path": pnl_path.as_posix()}}, sort_keys=False),
        encoding="utf-8",
    )
    config = H1CAutoConfig.from_mapping(
        {
            "auto": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "paper_only": True,
                "max_entry_slippage_bps": 25,
            },
            "components": {"accounting_config_path": accounting_config_path.as_posix()},
            "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
        }
    )

    result = entry_safety_check(config, output_dir=tmp_path / "runs", created_at_utc="2026-05-11T16:00:00Z")

    assert result["ok"] is False
    assert result["latest_entry_slippage_bps"] == 31.0
    assert "entry slippage limit reached" in result["issues"][0]


def test_exit_due_status_detects_fixed_horizon_exit(tmp_path: Path) -> None:
    state = {
        "status": "open",
        "open_position": {"theoretical_exit_timestamp": "2026-05-11 15:45:00-04:00"},
    }

    result = exit_due_status(state, "2026-05-11 19:46:00+00:00")

    assert result["due"] is True
    assert result["reason"] == "exit_due"


def test_build_exit_ticket_covers_open_short_quantity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ticket = build_exit_ticket(
        {
            "status": "open",
            "quantity": 3.0,
            "last_signal_timestamp": "2026-05-11 14:00:00-04:00",
            "open_position": {
                "signal_timestamp": "2026-05-11 14:00:00-04:00",
                "theoretical_entry_timestamp": "2026-05-11 14:15:00-04:00",
                "theoretical_entry_price": 100.0,
                "theoretical_exit_timestamp": "2026-05-11 15:45:00-04:00",
                "theoretical_exit_price": 98.0,
                "horizon_bars": 6,
            },
        },
        config,
    )

    assert ticket["action"] == "BUY"
    assert ticket["quantity"] == 3.0
    assert ticket["desired_position_unit"] == 0.0


def _write_auto_config(path: Path, tmp_path: Path, *, run_accounting_after_reconciliation: bool = True) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "auto": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "paper_only": True,
                    "require_market_open": True,
                    "execute_orders": True,
                    "transmit_orders": True,
                    "run_accounting_after_reconciliation": run_accounting_after_reconciliation,
                },
                "components": {
                    "data_refresh_config_path": (tmp_path / "refresh.yaml").as_posix(),
                    "signal_runner_config_path": (tmp_path / "signal.yaml").as_posix(),
                    "state_config_path": (tmp_path / "state.yaml").as_posix(),
                    "reconciliation_config_path": (tmp_path / "reconcile.yaml").as_posix(),
                    "accounting_config_path": (tmp_path / "accounting.yaml").as_posix(),
                    "order_plan_config_path": (tmp_path / "plan.yaml").as_posix(),
                    "order_executor_config_path": (tmp_path / "executor.yaml").as_posix(),
                },
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _fake_recon(root: Path, decision: str = "OK_FLAT", server_time: str = "2026-05-10 12:00:00+00:00", state: dict[str, object] | None = None):
    output_dir = root / "recon"
    snapshot_dir = output_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"account": "DU123", "tag": "AvailableFunds", "value": "5000", "currency": "USD"},
            {"account": "DU123", "tag": "BuyingPower", "value": "5000", "currency": "USD"},
        ]
    ).to_parquet(snapshot_dir / "account_summary.parquet", index=False)
    paths = H1CReconciliationPaths(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.yaml",
        report_path=output_dir / "report.md",
        state_snapshot_path=output_dir / "state.yaml",
        positions_path=output_dir / "positions.parquet",
        open_trades_path=output_dir / "open_trades.parquet",
        executions_path=output_dir / "executions.parquet",
        ibkr_snapshot_dir=snapshot_dir,
    )
    paths.state_snapshot_path.write_text(yaml.safe_dump(state or {"status": "flat"}, sort_keys=False), encoding="utf-8")
    manifest = {
        "reconciliation": {"decision": decision, "severity": "ok" if decision == "OK_FLAT" else "block"},
        "ibkr": {"health": {"server_time": server_time}},
        "outputs": {"ibkr_snapshot_dir": snapshot_dir.as_posix()},
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return paths, manifest


def test_auto_runner_exits_before_refresh_when_market_closed(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "auto.yaml"
    _write_auto_config(config_path, tmp_path)

    def fake_reconcile(**kwargs):
        return _fake_recon(tmp_path, server_time="2026-05-10 12:00:00+00:00")

    def fail_refresh(**kwargs):
        raise AssertionError("refresh should not run when market is closed")

    monkeypatch.setattr("src.execution.h1c_auto_runner.run_h1c_reconciliation", fake_reconcile)
    monkeypatch.setattr("src.execution.h1c_auto_runner.run_paper_data_refresh", fail_refresh)

    _, manifest = run_h1c_auto(config_path=config_path)

    assert manifest["decision"] == "market_closed"


def test_auto_runner_blocks_duplicate_when_reconciliation_not_flat(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "auto.yaml"
    _write_auto_config(config_path, tmp_path)

    def fake_reconcile(**kwargs):
        return _fake_recon(tmp_path, decision="OK_PENDING_ENTRY", server_time="2026-05-11 15:00:00+00:00")

    def fake_refresh(**kwargs):
        root = tmp_path / "refresh"
        root.mkdir(exist_ok=True)
        return PaperDataRefreshPaths(output_dir=root, manifest_path=root / "manifest.yaml", report_path=root / "report.md"), {
            "run": {"status": "complete"},
            "date_window": {},
        }

    def fake_signal(**kwargs):
        root = tmp_path / "signal"
        root.mkdir(exist_ok=True)
        ticket = root / "paper_ticket.yaml"
        ticket.write_text(
            yaml.safe_dump(
                {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "action": "SELL",
                    "quantity": 1.0,
                    "theoretical_entry_price": 100.0,
                }
            ),
            encoding="utf-8",
        )
        paths = PaperH1CRunnerPaths(
            output_dir=root,
            signals_path=root / "signals.parquet",
            latest_signal_path=root / "latest_signal.yaml",
            ticket_path=ticket,
            manifest_path=root / "manifest.yaml",
            report_path=root / "report.md",
        )
        return paths, {"ticket": yaml.safe_load(ticket.read_text()), "warnings": []}

    monkeypatch.setattr("src.execution.h1c_auto_runner.run_h1c_reconciliation", fake_reconcile)
    monkeypatch.setattr("src.execution.h1c_auto_runner.run_paper_data_refresh", fake_refresh)
    monkeypatch.setattr("src.execution.h1c_auto_runner.run_h1c_signal_runner", fake_signal)

    _, manifest = run_h1c_auto(config_path=config_path)

    assert manifest["decision"] == "blocked_reconciliation"


def test_auto_runner_submits_exit_when_open_position_exit_is_due(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "auto.yaml"
    state_config_path = tmp_path / "state.yaml"
    plan_config_path = tmp_path / "plan.yaml"
    _write_auto_config(config_path, tmp_path, run_accounting_after_reconciliation=False)
    state = {
        "schema_version": 1,
        "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
        "account": "DU123",
        "symbol": "QQQ",
        "status": "open",
        "position_unit": -1.0,
        "quantity": 2.0,
        "desired_position_unit": -1.0,
        "pending_ticket": None,
        "open_position": {
            "quantity": 2.0,
            "side": "SHORT",
            "signal_timestamp": "2026-05-11 14:00:00-04:00",
            "theoretical_entry_timestamp": "2026-05-11 14:15:00-04:00",
            "theoretical_entry_price": 100.0,
            "theoretical_exit_timestamp": "2026-05-11 15:45:00-04:00",
            "theoretical_exit_price": 98.0,
            "exit_rule": "fixed_horizon_open",
            "horizon_bars": 6,
        },
        "last_signal_timestamp": "2026-05-11 14:00:00-04:00",
    }
    state_path = tmp_path / "h1c_state.yaml"
    state_path.write_text(yaml.safe_dump(state, sort_keys=False), encoding="utf-8")
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
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plan_config_path.write_text(
        yaml.safe_dump(
            {
                "plan": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "max_quantity": 10,
                },
                "execution_policy": {"order_type": "MKT", "tif": "DAY", "outside_rth": False},
                "outputs": {"output_dir": (tmp_path / "plans").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = {"count": 0}

    def fake_reconcile(**kwargs):
        calls["count"] += 1
        decision = "OK_OPEN" if calls["count"] == 1 else "OK_PENDING_EXIT"
        return _fake_recon(tmp_path, decision=decision, server_time="2026-05-11 19:46:00+00:00", state=state)

    def fake_execute(**kwargs):
        root = tmp_path / "exec"
        root.mkdir(exist_ok=True)
        paths = H1CExecutionPaths(
            output_dir=root,
            manifest_path=root / "manifest.yaml",
            preflight_path=root / "preflight.yaml",
            submitted_orders_path=root / "submitted_orders.parquet",
            report_path=root / "report.md",
        )
        return paths, {"submitted_orders": 1, "preflight": {}, "unlock": {}, "live_preflight": {}}

    monkeypatch.setattr("src.execution.h1c_auto_runner.run_h1c_reconciliation", fake_reconcile)
    monkeypatch.setattr("src.execution.h1c_auto_runner.run_h1c_order_execution", fake_execute)

    _, manifest = run_h1c_auto(config_path=config_path)

    updated_state = yaml.safe_load(state_path.read_text())
    assert manifest["decision"] == "exit_submitted"
    assert manifest["exit_ticket"]["action"] == "BUY"
    assert manifest["order_plan"]["summary"]["intent"] == "h1c_short_exit"
    assert updated_state["status"] == "pending_exit"

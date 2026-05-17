from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.execution.paper_accounting_h1c import H1CAccountingConfig, apply_accounting_to_state, run_h1c_accounting


def _state(status: str = "pending_entry") -> dict[str, object]:
    return {
        "schema_version": 1,
        "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
        "account": "DU123",
        "symbol": "QQQ",
        "status": status,
        "position_unit": 0.0,
        "quantity": 0.0,
        "desired_position_unit": -1.0,
        "pending_ticket": {
            "quantity": 1.0,
            "theoretical_entry_price": 100.0,
            "theoretical_entry_timestamp": "2026-05-08 14:15:00-04:00",
            "theoretical_exit_timestamp": "2026-05-08 15:45:00-04:00",
            "theoretical_exit_price": 98.0,
            "exit_rule": "fixed_horizon_open",
            "horizon_bars": 6,
            "signal_timestamp": "2026-05-08 14:00:00-04:00",
        },
        "open_position": None,
        "last_signal_timestamp": "2026-05-08 14:00:00-04:00",
    }


def _config(tmp_path: Path) -> H1CAccountingConfig:
    return H1CAccountingConfig.from_mapping(
        {
            "accounting": {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "account": "DU123",
                "symbol": "QQQ",
                "state_config_path": (tmp_path / "state_config.yaml").as_posix(),
                "pnl_log_path": (tmp_path / "pnl.parquet").as_posix(),
            },
            "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
        }
    )


def test_apply_accounting_marks_pending_entry_open_on_fill_detection(tmp_path: Path) -> None:
    reconciliation = {"reconciliation": {"decision": "FILL_DETECTED_PENDING_ENTRY", "target_position_qty": -1.0}}
    positions = pd.DataFrame([{"account": "DU123", "symbol": "QQQ", "position": -1.0, "avg_cost": 101.0}])

    updated, event, pnl = apply_accounting_to_state(
        state=_state(),
        reconciliation_manifest=reconciliation,
        positions=positions,
        executions=pd.DataFrame(),
        config=_config(tmp_path),
    )

    assert updated["status"] == "open"
    assert updated["quantity"] == 1.0
    assert event["event_type"] == "entry_fill_marked_open"
    assert pnl["event_type"] == "entry"
    assert pnl["entry_price"] == 101.0
    assert updated["open_position"]["theoretical_exit_timestamp"] == "2026-05-08 15:45:00-04:00"


def test_apply_accounting_marks_pending_exit_flat_and_records_pnl(tmp_path: Path) -> None:
    state = {
        **_state("pending_exit"),
        "position_unit": -1.0,
        "quantity": 1.0,
        "desired_position_unit": 0.0,
        "pending_ticket": {
            "quantity": 1.0,
            "theoretical_entry_price": 100.0,
            "theoretical_exit_timestamp": "2026-05-08 15:45:00-04:00",
            "theoretical_exit_price": 98.0,
        },
        "open_position": {
            "quantity": 1.0,
            "side": "SHORT",
            "entry_price": 101.0,
            "theoretical_entry_price": 100.0,
            "theoretical_exit_timestamp": "2026-05-08 15:45:00-04:00",
            "theoretical_exit_price": 98.0,
        },
    }
    reconciliation = {"reconciliation": {"decision": "FILL_DETECTED_PENDING_EXIT", "target_position_qty": 0.0}}
    executions = pd.DataFrame([{"account": "DU123", "symbol": "QQQ", "side": "BOT", "shares": 1.0, "price": 97.0, "realized_pnl": 4.0}])

    updated, event, pnl = apply_accounting_to_state(
        state=state,
        reconciliation_manifest=reconciliation,
        positions=pd.DataFrame(),
        executions=executions,
        config=_config(tmp_path),
    )

    assert updated["status"] == "flat"
    assert updated["open_position"] is None
    assert event["event_type"] == "exit_fill_marked_flat"
    assert pnl["event_type"] == "exit"
    assert pnl["exit_price"] == 97.0
    assert pnl["realized_pnl"] == 4.0


def test_run_h1c_accounting_writes_state_and_pnl_log(tmp_path: Path) -> None:
    state_config_path = tmp_path / "state_config.yaml"
    state_path = tmp_path / "state.yaml"
    event_log_path = tmp_path / "events.parquet"
    config_path = tmp_path / "accounting.yaml"
    reconciliation_manifest_path = tmp_path / "reconciliation.yaml"
    positions_path = tmp_path / "positions.parquet"
    executions_path = tmp_path / "executions.parquet"
    state_path.write_text(yaml.safe_dump(_state(), sort_keys=False), encoding="utf-8")
    state_config_path.write_text(
        yaml.safe_dump(
            {
                "state": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "state_path": state_path.as_posix(),
                    "event_log_path": event_log_path.as_posix(),
                    "output_dir": (tmp_path / "state_runs").as_posix(),
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"account": "DU123", "symbol": "QQQ", "position": -1.0, "avg_cost": 101.0}]).to_parquet(positions_path, index=False)
    pd.DataFrame().to_parquet(executions_path, index=False)
    reconciliation_manifest_path.write_text(
        yaml.safe_dump(
            {
                "reconciliation": {"decision": "FILL_DETECTED_PENDING_ENTRY", "target_position_qty": -1.0},
                "outputs": {"positions": positions_path.as_posix(), "executions": executions_path.as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "accounting": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "state_config_path": state_config_path.as_posix(),
                    "pnl_log_path": (tmp_path / "pnl.parquet").as_posix(),
                },
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    paths, manifest = run_h1c_accounting(reconciliation_manifest_path=reconciliation_manifest_path, config_path=config_path)

    assert paths.report_path.exists()
    assert paths.pnl_log_path.exists()
    assert manifest["accounting"]["event"]["event_type"] == "entry_fill_marked_open"

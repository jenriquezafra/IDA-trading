from __future__ import annotations

from pathlib import Path

from src.candidate_app.models import metric_value, normalize_metrics
from src.candidate_app.control import apply_control_action, candidate_control_snapshot
from src.candidate_app.service import (
    change_candidate_status,
    compare_candidates,
    create_candidate,
    create_paper_ledger_entry,
    get_candidate,
    list_candidate_records,
    list_paper_ledger,
    paper_ledger_summaries,
    prepare_store,
)


def candidate_payload(candidate_id: str = "test-candidate") -> dict:
    return {
        "id": candidate_id,
        "name": "Test Candidate",
        "strategy_type": "unit_test",
        "asset_universe": ["SPY", "QQQ"],
        "status": "candidate",
        "description": "Synthetic strategy candidate.",
        "metrics": {
            "cagr": 0.12,
            "annualized_return": 0.12,
            "sharpe": 1.4,
            "sortino": 2.0,
            "max_drawdown": -0.08,
            "volatility": 0.11,
            "win_rate": 0.56,
            "profit_factor": 1.31,
            "trade_count": 42,
            "turnover": 2.4,
            "estimated_costs_bps": 4.0,
            "estimated_slippage_bps": 1.2,
            "backtest_period_start": "2025-01-01",
            "backtest_period_end": "2026-01-01",
            "last_evaluated_at": "2026-05-16T10:00:00Z",
        },
    }


def test_create_candidate_and_read_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    prepare_store(db_path, seed=False)

    created = create_candidate(db_path, candidate_payload())
    records = list_candidate_records(db_path)

    assert created["id"] == "test-candidate"
    assert records[0]["name"] == "Test Candidate"
    assert metric_value(created, "sharpe") == 1.4
    assert created["metrics"]["trade_count"] == 42


def test_change_status_records_audit_log(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    create_candidate(db_path, candidate_payload())

    updated = change_candidate_status(
        db_path,
        "test-candidate",
        "paper_trading",
        actor="pytest",
        reason="promotion test",
    )
    detail = get_candidate(db_path, "test-candidate")

    assert updated["status"] == "paper_trading"
    assert updated["promoted_at"] is not None
    assert detail["audit_log"][0]["from_status"] == "candidate"
    assert detail["audit_log"][0]["to_status"] == "paper_trading"
    assert detail["audit_log"][0]["reason"] == "promotion test"


def test_filters_and_compare_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    create_candidate(db_path, candidate_payload("strong"))
    weak = candidate_payload("weak")
    weak["name"] = "Weak Candidate"
    weak["metrics"]["sharpe"] = 0.2
    weak["metrics"]["max_drawdown"] = -0.25
    create_candidate(db_path, weak)

    filtered = list_candidate_records(db_path, asset="SPY", sharpe_min=1.0, max_drawdown_min=-0.10)
    comparison = compare_candidates(db_path, ["strong", "weak"])

    assert [candidate["id"] for candidate in filtered] == ["strong"]
    assert [row["id"] for row in comparison] == ["strong", "weak"]
    assert comparison[0]["sharpe"] == 1.4


def test_normalize_metrics_preserves_required_keys() -> None:
    metrics = normalize_metrics({"sharpe": 1.1})

    assert metrics["sharpe"] == 1.1
    assert "max_drawdown" in metrics
    assert metrics["trade_count"] is None


def test_paper_ledger_tracks_pnl_by_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    payload = candidate_payload("paper-candidate")
    payload["status"] = "paper_trading"
    create_candidate(db_path, payload)

    create_paper_ledger_entry(
        db_path,
        {
            "entry_id": "entry-1",
            "candidate_id": "paper-candidate",
            "event_at": "2026-05-16T10:00:00Z",
            "event_type": "fill",
            "strategy_run_id": "run-1",
            "symbol": "SPY",
            "side": "long",
            "quantity": 10,
            "price": 500.0,
            "gross_pnl": 42.0,
            "fees": 2.0,
            "slippage_bps": 0.8,
            "net_pnl": 40.0,
            "exposure": 5000.0,
        },
    )
    create_paper_ledger_entry(
        db_path,
        {
            "entry_id": "entry-2",
            "candidate_id": "paper-candidate",
            "event_at": "2026-05-17T10:00:00Z",
            "event_type": "fill",
            "strategy_run_id": "run-2",
            "symbol": "SPY",
            "side": "long",
            "quantity": 10,
            "price": 501.0,
            "gross_pnl": -10.0,
            "fees": 2.0,
            "slippage_bps": 1.2,
            "net_pnl": -12.0,
            "exposure": 5010.0,
        },
    )

    entries = list_paper_ledger(db_path, candidate_id="paper-candidate")
    summaries = paper_ledger_summaries(db_path)
    detail = get_candidate(db_path, "paper-candidate")

    assert [entry["entry_id"] for entry in entries] == ["entry-1", "entry-2"]
    assert summaries[0]["candidate_id"] == "paper-candidate"
    assert summaries[0]["net_pnl"] == 28.0
    assert summaries[0]["fees"] == 4.0
    assert summaries[0]["win_rate"] == 0.5
    assert summaries[0]["pnl_curve"][-1]["cumulative_net_pnl"] == 28.0
    assert detail["paper_ledger_summary"]["max_pnl_drawdown"] == -12.0


def test_control_snapshot_excludes_manual_ledger_from_operational_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    prepare_store(db_path, seed=True)

    snapshot = candidate_control_snapshot("qqq-risk-off-credit-spread", root=tmp_path, db_path=db_path)

    assert snapshot["pnl"]["realized_pnl"] == 0.0
    assert snapshot["pnl"]["source_available"] is False
    assert snapshot["pnl"]["manual_ledger_affects_pnl"] is False
    assert snapshot["pnl"]["excluded_manual_ledger_count"] == 3
    assert snapshot["ledger"]["events"] == []
    assert snapshot["manual_ledger"]["count"] == 3
    assert snapshot["manual_ledger"]["affects_operational_pnl"] is False
    assert snapshot["manual_ledger"]["metrics"]["realized_pnl"] == 143.28
    assert any(alert["code"] == "PNL_LOG_MISSING" for alert in snapshot["alerts"])


def test_control_action_toggles_kill_switch(tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.sqlite"
    (tmp_path / "ops" / "kill_switches").mkdir(parents=True)
    (tmp_path / "results" / "paper" / "h1c_auto_runner").mkdir(parents=True)
    (tmp_path / "results" / "paper" / "h1c_state").mkdir(parents=True)
    (tmp_path / "results" / "paper" / "h1c_state" / "state.yaml").write_text(
        "strategy_id: qqq_15min_risk_off_short_h1c_v1\nstatus: flat\nquantity: 0\nsymbol: QQQ\n",
        encoding="utf-8",
    )

    paused = apply_control_action(
        "qqq-risk-off-credit-spread",
        action="pause",
        actor="pytest",
        reason="test pause",
        root=tmp_path,
        db_path=db_path,
    )
    resumed = apply_control_action(
        "qqq-risk-off-credit-spread",
        action="resume",
        actor="pytest",
        reason="test resume",
        root=tmp_path,
        db_path=db_path,
    )
    snapshot = candidate_control_snapshot("qqq-risk-off-credit-spread", root=tmp_path, db_path=db_path)

    assert paused["control"]["kill_switch_exists"] is True
    assert resumed["control"]["kill_switch_exists"] is False
    assert snapshot["overall_state"] in {"attention", "waiting", "running"}

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute

from src.candidate_app.api import (
    CandidateCreateRequest,
    PaperLedgerCreateRequest,
    RuntimeControlRequest,
    StatusChangeRequest,
    create_app,
)


def endpoint_for(app: FastAPI, path: str, method: str = "GET"):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"missing route {method} {path}")


def test_candidate_api_exposes_dashboard_routes(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "candidates.sqlite", seed=True)
    routes = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/" in routes
    assert "/static" in routes
    assert "/health" in routes
    assert "/metadata" in routes
    assert "/control-center" in routes
    assert "/control-center/connections" in routes
    assert "/control-center/{candidate_id}" in routes
    assert "/control-center/{candidate_id}/control" in routes
    assert "/control-center/{candidate_id}/runtime" in routes
    assert "/candidates" in routes
    assert "/candidates/paper-trading" in routes
    assert "/candidates/{candidate_id}" in routes
    assert "/candidates/{candidate_id}/status" in routes
    assert "/paper-ledger" in routes
    assert "/paper-ledger/summary" in routes
    assert "/compare" in routes


def test_candidate_api_renders_main_dashboard(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "candidates.sqlite", seed=True)
    response = endpoint_for(app, "/")()
    html = Path(response.path).read_text(encoding="utf-8")

    assert "Trading Strats Control Center" in html
    assert "Capital operativo" in html
    assert "Precio y señales" in html
    assert "chart-tooltip" in html
    assert "Ledger operativo" in html
    assert "Ledger manual / review" in html
    assert "Últimos runs" in html
    assert "Eventos de estado" in html
    assert "Operaciones y posición" in html
    assert "Posición actual" in html


def test_candidate_api_create_filter_and_status_flow(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "candidates.sqlite", seed=False)
    payload = {
        "id": "api-candidate",
        "name": "API Candidate",
        "strategy_type": "api_test",
        "asset_universe": ["SPY"],
        "metrics": {
            "cagr": 0.10,
            "annualized_return": 0.10,
            "sharpe": 1.25,
            "sortino": 1.6,
            "max_drawdown": -0.07,
            "volatility": 0.09,
            "win_rate": 0.54,
            "profit_factor": 1.2,
            "trade_count": 12,
            "turnover": 1.7,
            "estimated_costs_bps": 3.0,
            "estimated_slippage_bps": 0.9,
            "backtest_period_start": "2025-01-01",
            "backtest_period_end": "2026-01-01",
            "last_evaluated_at": "2026-05-16T10:00:00Z",
        },
    }

    created = endpoint_for(app, "/candidates", "POST")(CandidateCreateRequest(**payload))
    listed = endpoint_for(app, "/candidates")(
        status=None,
        strategy_type=None,
        asset="SPY",
        sharpe_min=1.0,
        sharpe_max=None,
        max_drawdown_min=None,
        max_drawdown_max=None,
    )
    status = endpoint_for(app, "/candidates/{candidate_id}/status", "PATCH")(
        "api-candidate",
        StatusChangeRequest(status="paper_trading", actor="pytest", reason="ready"),
    )
    detail = endpoint_for(app, "/candidates/{candidate_id}")("api-candidate")
    comparison = endpoint_for(app, "/compare")(candidate_id=["api-candidate"])
    ledger_entry = endpoint_for(app, "/paper-ledger", "POST")(
        PaperLedgerCreateRequest(
            entry_id="api-ledger-1",
            candidate_id="api-candidate",
            event_at="2026-05-17T10:00:00Z",
            event_type="fill",
            symbol="SPY",
            side="long",
            quantity=3,
            price=500.0,
            gross_pnl=15.0,
            fees=1.0,
            slippage_bps=0.7,
            net_pnl=14.0,
            exposure=1500.0,
        )
    )
    ledger = endpoint_for(app, "/paper-ledger")(candidate_id="api-candidate", active_only=False)
    ledger_summary = endpoint_for(app, "/paper-ledger/summary")(active_only=True)

    assert created["id"] == "api-candidate"
    assert listed[0]["id"] == "api-candidate"
    assert status["status"] == "paper_trading"
    assert detail["audit_log"][0]["to_status"] == "paper_trading"
    assert comparison[0]["sharpe"] == 1.25
    assert ledger_entry["entry_id"] == "api-ledger-1"
    assert ledger[0]["net_pnl"] == 14.0
    assert ledger_summary[0]["net_pnl"] == 14.0


def test_control_endpoint_reports_active_candidates() -> None:
    app = create_app(seed=False)

    snapshot = endpoint_for(app, "/control-center")()

    assert snapshot["summary"]["active_count"] >= 1
    assert "paper" in snapshot["sections"]
    assert "live" in snapshot["sections"]
    assert "connection" in snapshot
    assert snapshot["active_candidates"][0]["mode"] == "paper"
    assert "alerts" in snapshot["active_candidates"][0]
    assert "ledger" in snapshot["active_candidates"][0]
    assert "operations" in snapshot["active_candidates"][0]
    assert "position" in snapshot["active_candidates"][0]
    assert "current" in snapshot["active_candidates"][0]["position"]
    assert "timeline" in snapshot["active_candidates"][0]["position"]
    c2 = next(candidate for candidate in snapshot["active_candidates"] if candidate["candidate_id"] == "c2-googl-opening-bias-followthrough")
    assert c2["symbol"] == "GOOGL"
    assert c2["strategy_id"] == "c2_h9_googl_5min_opening_bias_followthrough_v1"


def test_connection_and_runtime_control_routes(tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "candidates.sqlite", seed=True)

    connections = endpoint_for(app, "/control-center/connections")()
    before = endpoint_for(app, "/control-center/{candidate_id}")("qqq-risk-off-credit-spread")
    desired_enabled = not before["control"]["kill_switch_exists"]
    updated = endpoint_for(app, "/control-center/{candidate_id}/runtime", "PATCH")(
        "qqq-risk-off-credit-spread",
        RuntimeControlRequest(
            enabled=desired_enabled,
            capital_mode="net_fraction",
            capital_value=0.25,
            capital_basis="buying_power_fraction",
            actor="pytest",
            notes="test runtime control",
        ),
    )

    assert connections["check_count"] >= 1
    assert updated["control"]["runtime"]["enabled"] is desired_enabled
    assert updated["control"]["runtime"]["capital_value"] == 0.25

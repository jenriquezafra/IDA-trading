from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.candidate_app.control import (
    apply_control_action,
    candidate_control_snapshot,
    connection_snapshot,
    control_center_snapshot,
    update_strategy_runtime_control,
)
from src.candidate_app.models import VALID_STATUSES
from src.candidate_app.service import (
    change_candidate_status,
    compare_candidates,
    create_candidate,
    create_paper_ledger_entry,
    get_candidate,
    list_candidate_records,
    list_paper_ledger,
    metadata,
    paper_ledger_summaries,
    paper_trading_candidates,
    prepare_store,
)
from src.candidate_app.store import DEFAULT_DB_PATH


WEB_ROOT = Path(__file__).resolve().parent / "web"


class CandidateCreateRequest(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    strategy_type: str = Field(min_length=1)
    asset_universe: list[str] = Field(default_factory=list)
    status: str = "candidate"
    created_at: str | None = None
    promoted_at: str | None = None
    description: str = ""
    backtest_summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    paper_trading_metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    actor: str = "dashboard"


class StatusChangeRequest(BaseModel):
    status: str
    actor: str = "dashboard"
    reason: str = ""
    note: str = ""


class PaperLedgerCreateRequest(BaseModel):
    entry_id: str | None = None
    candidate_id: str = Field(min_length=1)
    event_at: str | None = None
    event_type: str = Field(min_length=1)
    strategy_run_id: str | None = None
    symbol: str | None = None
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    gross_pnl: float | None = None
    fees: float | None = None
    slippage_bps: float | None = None
    net_pnl: float | None = None
    exposure: float | None = None
    currency: str = "USD"
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlActionRequest(BaseModel):
    action: str = Field(min_length=1)
    actor: str = "dashboard"
    reason: str = ""


class RuntimeControlRequest(BaseModel):
    enabled: bool
    capital_mode: str = "net_fraction"
    capital_value: float = 1.0
    capital_basis: str = "buying_power_fraction"
    actor: str = "dashboard"
    notes: str = ""
    apply_to_config: bool = False


def as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=f"candidate not found: {exc.args[0]}")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_app(db_path: str | Path = DEFAULT_DB_PATH, *, seed: bool = True) -> FastAPI:
    app = FastAPI(
        title="IDA Candidate Strategy API",
        version="0.1.0",
        description="Candidate-only dashboard API for paper-trading promotion and review.",
    )
    resolved_db_path = Path(db_path)

    if WEB_ROOT.exists():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    def ready() -> None:
        prepare_store(resolved_db_path, seed=seed)

    @app.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        index_path = WEB_ROOT / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web UI is not installed")
        return FileResponse(index_path)

    @app.get("/health")
    def health() -> dict[str, Any]:
        ready()
        return {"ok": True, "db_path": resolved_db_path.as_posix(), "metadata": metadata(resolved_db_path)}

    @app.get("/metadata")
    def get_metadata() -> dict[str, Any]:
        ready()
        return metadata(resolved_db_path)

    @app.get("/control-center")
    def get_control_center() -> dict[str, Any]:
        ready()
        try:
            return control_center_snapshot(db_path=resolved_db_path)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/control-center/connections")
    def get_connections() -> dict[str, Any]:
        ready()
        try:
            return connection_snapshot()
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/control-center/{candidate_id}")
    def get_control_candidate(candidate_id: str) -> dict[str, Any]:
        ready()
        try:
            return candidate_control_snapshot(candidate_id, db_path=resolved_db_path)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.post("/control-center/{candidate_id}/control")
    def post_control_action(candidate_id: str, request: ControlActionRequest) -> dict[str, Any]:
        ready()
        try:
            return apply_control_action(
                candidate_id,
                action=request.action,
                actor=request.actor,
                reason=request.reason,
                db_path=resolved_db_path,
            )
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.patch("/control-center/{candidate_id}/runtime")
    def patch_runtime_control(candidate_id: str, request: RuntimeControlRequest) -> dict[str, Any]:
        ready()
        try:
            return update_strategy_runtime_control(
                candidate_id,
                enabled=request.enabled,
                capital_mode=request.capital_mode,
                capital_value=request.capital_value,
                capital_basis=request.capital_basis,
                actor=request.actor,
                notes=request.notes,
                apply_to_config=request.apply_to_config,
                db_path=resolved_db_path,
            )
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/candidates")
    def get_candidates(
        status: str | None = None,
        strategy_type: str | None = None,
        asset: str | None = None,
        sharpe_min: float | None = None,
        sharpe_max: float | None = None,
        max_drawdown_min: float | None = None,
        max_drawdown_max: float | None = None,
    ) -> list[dict[str, Any]]:
        ready()
        return list_candidate_records(
            resolved_db_path,
            status=status,
            strategy_type=strategy_type,
            asset=asset,
            sharpe_min=sharpe_min,
            sharpe_max=sharpe_max,
            max_drawdown_min=max_drawdown_min,
            max_drawdown_max=max_drawdown_max,
        )

    @app.post("/candidates")
    def post_candidate(request: CandidateCreateRequest) -> dict[str, Any]:
        ready()
        payload = request.model_dump()
        actor = payload.pop("actor", "dashboard")
        try:
            return create_candidate(resolved_db_path, payload, actor=actor)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/candidates/paper-trading")
    def get_paper_trading_candidates() -> list[dict[str, Any]]:
        ready()
        return paper_trading_candidates(resolved_db_path)

    @app.get("/candidates/{candidate_id}")
    def get_candidate_detail(candidate_id: str) -> dict[str, Any]:
        ready()
        try:
            return get_candidate(resolved_db_path, candidate_id)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.patch("/candidates/{candidate_id}/status")
    def patch_candidate_status(candidate_id: str, request: StatusChangeRequest) -> dict[str, Any]:
        ready()
        if request.status not in VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"invalid candidate status: {request.status}")
        try:
            return change_candidate_status(
                resolved_db_path,
                candidate_id,
                request.status,
                actor=request.actor,
                reason=request.reason,
                note=request.note,
            )
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/paper-ledger")
    def get_paper_ledger(
        candidate_id: str | None = None,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        ready()
        return list_paper_ledger(resolved_db_path, candidate_id=candidate_id, active_only=active_only)

    @app.post("/paper-ledger")
    def post_paper_ledger_entry(request: PaperLedgerCreateRequest) -> dict[str, Any]:
        ready()
        try:
            return create_paper_ledger_entry(resolved_db_path, request.model_dump())
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/paper-ledger/summary")
    def get_paper_ledger_summary(active_only: bool = True) -> list[dict[str, Any]]:
        ready()
        return paper_ledger_summaries(resolved_db_path, active_only=active_only)

    @app.get("/compare")
    def get_compare(candidate_id: list[str] | None = Query(default=None)) -> list[dict[str, Any]]:
        ready()
        try:
            return compare_candidates(resolved_db_path, candidate_id or [])
        except Exception as exc:
            raise as_http_error(exc) from exc

    return app


app = create_app()

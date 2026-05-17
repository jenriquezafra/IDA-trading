from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.research_app.registry import DEFAULT_DB_PATH
from src.research_app.service import (
    PROJECT_ROOT,
    frame_to_records,
    h8_available_targets,
    h8c_position_snapshot,
    h8_probe_snapshot,
    h1c_operations_snapshot,
    index_registry,
    list_registry_records,
    load_registry_frames,
    parquet_preview,
    read_daemon_status,
    read_report_markdown,
    record_decision,
    registry_summary,
)


WEB_ROOT = Path(__file__).resolve().parent / "web"


class IndexRequest(BaseModel):
    db_path: str = str(DEFAULT_DB_PATH)
    results_dir: str = "results"
    reports_dir: str = "reports"
    reset: bool = False


class DecisionRequest(BaseModel):
    db_path: str = str(DEFAULT_DB_PATH)
    decision_type: str
    decision: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    run_id: str | None = None
    candidate_id: str | None = None
    rationale: str | None = None
    next_action: str | None = None
    human_owner: str | None = None


class H8RunRequest(BaseModel):
    target_symbol: str = Field(default="SPY", min_length=1, max_length=12)
    config_path: str = "configs/hmm_bayesian_regime_h8_spy_15min.yaml"


class H8cRunRequest(BaseModel):
    target_symbol: str = Field(default="QQQ", min_length=1, max_length=12)
    config_path: str = "configs/hmm_bayesian_regime_h8c_qqq_15min.yaml"


def as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def create_app() -> FastAPI:
    app = FastAPI(
        title="IDA Research API",
        version="0.1.0",
        description="Read-only research registry API plus explicit decision-log writes.",
    )
    if WEB_ROOT.exists():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        index_path = WEB_ROOT / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web UI is not installed")
        return FileResponse(index_path)

    @app.get("/health")
    def health(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return {
            "ok": True,
            "workspace": PROJECT_ROOT.as_posix(),
            "registry": registry_summary(db_path).__dict__,
            "daemon_status": read_daemon_status(),
        }

    @app.get("/registry/summary")
    def get_registry_summary(db_path: str = str(DEFAULT_DB_PATH)) -> dict[str, Any]:
        return registry_summary(db_path).__dict__

    @app.post("/registry/index")
    def post_registry_index(request: IndexRequest) -> dict[str, Any]:
        try:
            return index_registry(
                db_path=request.db_path,
                results_dir=request.results_dir,
                reports_dir=request.reports_dir,
                reset=request.reset,
            ).__dict__
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/registry/snapshot")
    def get_registry_snapshot(
        db_path: str = str(DEFAULT_DB_PATH),
        run_id: str | None = None,
        target: list[str] | None = Query(default=None),
        timeframe: list[str] | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> dict[str, Any]:
        return list_registry_records(
            db_path=db_path,
            run_id=run_id,
            targets=target,
            timeframes=timeframe,
            limit=limit,
        )

    @app.get("/runs")
    def get_runs(db_path: str = str(DEFAULT_DB_PATH), limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
        return frame_to_records(load_registry_frames(db_path).runs, limit=limit)

    @app.get("/candidates")
    def get_candidates(
        db_path: str = str(DEFAULT_DB_PATH),
        run_id: str | None = None,
        target: list[str] | None = Query(default=None),
        timeframe: list[str] | None = Query(default=None),
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return list_registry_records(
            db_path=db_path,
            run_id=run_id,
            targets=target,
            timeframes=timeframe,
            limit=limit,
        )["candidates"]

    @app.get("/artifacts")
    def get_artifacts(
        db_path: str = str(DEFAULT_DB_PATH),
        run_id: str | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return list_registry_records(db_path=db_path, run_id=run_id, limit=limit)["artifacts"]

    @app.get("/reports")
    def get_reports(
        db_path: str = str(DEFAULT_DB_PATH),
        run_id: str | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return list_registry_records(db_path=db_path, run_id=run_id, limit=limit)["reports"]

    @app.get("/reports/markdown")
    def get_report_markdown(path: str) -> dict[str, Any]:
        try:
            return {"path": path, "markdown": read_report_markdown(path)}
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/artifacts/parquet-preview")
    def get_parquet_preview(path: str, limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
        try:
            return parquet_preview(path, limit=limit)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/decisions")
    def get_decisions(db_path: str = str(DEFAULT_DB_PATH), limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
        return frame_to_records(load_registry_frames(db_path).decisions, limit=limit)

    @app.post("/decisions")
    def post_decision(request: DecisionRequest) -> dict[str, Any]:
        try:
            return record_decision(**request.model_dump())
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.get("/operations/daemon-status")
    def get_daemon_status(path: str = "results/paper/h1c_auto_runner/daemon_status.yaml") -> dict[str, Any]:
        return read_daemon_status(Path(path))

    @app.get("/operations/h1c")
    def get_h1c_operations(limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
        return h1c_operations_snapshot(limit=limit)

    @app.get("/hypotheses/h8/targets")
    def get_h8_targets(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        return h8_available_targets(limit=limit)

    @app.post("/hypotheses/h8/run")
    def post_h8_run(request: H8RunRequest) -> dict[str, Any]:
        try:
            return h8_probe_snapshot(target_symbol=request.target_symbol, config_path=request.config_path)
        except Exception as exc:
            raise as_http_error(exc) from exc

    @app.post("/hypotheses/h8c/run")
    def post_h8c_run(request: H8cRunRequest) -> dict[str, Any]:
        try:
            return h8c_position_snapshot(target_symbol=request.target_symbol, config_path=request.config_path)
        except Exception as exc:
            raise as_http_error(exc) from exc

    return app


app = create_app()

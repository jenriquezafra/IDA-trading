from __future__ import annotations

from src.research_app.api import create_app


def test_research_api_exposes_core_routes() -> None:
    app = create_app()
    routes = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/" in routes
    assert "/static" in routes
    assert "/health" in routes
    assert "/registry/summary" in routes
    assert "/registry/snapshot" in routes
    assert "/operations/daemon-status" in routes
    assert "/operations/h1c" in routes
    assert "/hypotheses/h8/targets" in routes
    assert "/hypotheses/h8/run" in routes
    assert "/hypotheses/h8c/run" in routes

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.research_app.registry import index_workspace
from src.research_app.service import (
    filter_candidates,
    frame_to_records,
    h8_available_targets,
    h1c_operations_snapshot,
    list_registry_records,
    parquet_preview,
    read_daemon_status,
    read_report_markdown,
    registry_summary,
)


def build_registry(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    run_dir = results / "15min_divergence" / "SPY"
    report_dir = reports / "15min_divergence" / "SPY"
    run_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "decision": "keep",
                "validation_status": "research_candidate",
                "test_net_primary": 1.25,
            }
        ]
    ).to_parquet(run_dir / "cross_asset_divergence_decisions.parquet", index=False)
    (report_dir / "cross_asset_divergence_search.md").write_text("# Report\n", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    index_workspace(results_dir=results, reports_dir=reports, db_path=db_path, reset=True)
    return db_path


def test_registry_service_summary_and_snapshot(tmp_path: Path) -> None:
    db_path = build_registry(tmp_path)

    summary = registry_summary(db_path)
    snapshot = list_registry_records(db_path=db_path)

    assert summary.exists
    assert summary.runs == 1
    assert summary.candidates == 1
    assert snapshot["candidates"][0]["candidate_id"] == "C1"
    assert snapshot["candidates"][0]["target_symbol"] == "SPY"


def test_filter_candidates_by_target_and_timeframe() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "C1", "target_symbol": "SPY", "timeframe": "15min"},
            {"candidate_id": "C2", "target_symbol": "QQQ", "timeframe": "30min"},
        ]
    )

    filtered = filter_candidates(candidates, targets=["SPY"], timeframes=["15min"])

    assert filtered["candidate_id"].tolist() == ["C1"]


def test_frame_to_records_converts_nan_to_none() -> None:
    records = frame_to_records(pd.DataFrame([{"candidate_id": "C1", "metric": float("nan")}]))

    assert records == [{"candidate_id": "C1", "metric": None}]


def test_workspace_preview_helpers(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    artifacts = tmp_path / "results"
    reports.mkdir()
    artifacts.mkdir()
    report = reports / "report.md"
    parquet = artifacts / "artifact.parquet"
    report.write_text("# Local Report\n", encoding="utf-8")
    pd.DataFrame([{"a": 1}]).to_parquet(parquet, index=False)

    assert read_report_markdown("reports/report.md", root=tmp_path) == "# Local Report\n"
    preview = parquet_preview("results/artifact.parquet", root=tmp_path)

    assert preview["columns"] == ["a"]
    assert preview["rows"] == [{"a": 1}]


def test_h8_available_targets_reads_feature_inventory(tmp_path: Path) -> None:
    feature_path = tmp_path / "data" / "features" / "SPY" / "15min" / "core" / "v1" / "features.parquet"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"timestamp": "2024-01-02 09:30:00-05:00", "session": "2024-01-02"},
            {"timestamp": "2024-01-03 09:30:00-05:00", "session": "2024-01-03"},
        ]
    ).to_parquet(feature_path, index=False)

    targets = h8_available_targets(root=tmp_path)

    assert targets[0]["target_symbol"] == "SPY"
    assert targets[0]["sessions"] == 2
    assert targets[0]["feature_version"] == "v1"


def test_report_preview_rejects_non_markdown(tmp_path: Path) -> None:
    path = tmp_path / "reports"
    path.mkdir()
    (path / "report.txt").write_text("not markdown", encoding="utf-8")

    with pytest.raises(ValueError, match="markdown"):
        read_report_markdown("reports/report.txt", root=tmp_path)


def test_read_daemon_status(tmp_path: Path) -> None:
    status = tmp_path / "daemon_status.yaml"
    status.write_text("runner_summary:\n  decision: no_signal\n", encoding="utf-8")

    payload = read_daemon_status("daemon_status.yaml", root=tmp_path)

    assert payload["available"]
    assert payload["runner_summary"]["decision"] == "no_signal"


def test_h1c_operations_snapshot_reads_orders_and_pnl(tmp_path: Path) -> None:
    state_dir = tmp_path / "results" / "paper" / "h1c_state"
    auto_dir = tmp_path / "results" / "paper" / "h1c_auto_runner" / "RUN1"
    signal_dir = auto_dir / "signal" / "SIG1"
    execution_dir = auto_dir / "execution" / "EXEC1"
    plan_dir = auto_dir / "order_plan" / "PLAN1"
    state_dir.mkdir(parents=True)
    signal_dir.mkdir(parents=True)
    execution_dir.mkdir(parents=True)
    plan_dir.mkdir(parents=True)
    (state_dir / "state.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "open",
                "quantity": 10.0,
                "position_unit": -1.0,
                "last_signal_timestamp": "2026-05-11 10:00:00-04:00",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"created_at_utc": "2026-05-11T14:00:00Z", "event_type": "entry", "realized_pnl": None}]).to_parquet(
        state_dir / "pnl_events.parquet",
        index=False,
    )
    pd.DataFrame([{"created_at_utc": "2026-05-11T14:00:00Z", "event_type": "entry_fill_marked_open"}]).to_parquet(
        state_dir / "events.parquet",
        index=False,
    )
    pd.DataFrame([{"account": "DU1", "symbol": "QQQ", "action": "SELL", "quantity": 10.0}]).to_parquet(plan_dir / "orders.parquet", index=False)
    (plan_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"run": {"created_at_utc": "2026-05-11T13:59:00Z"}, "summary": {"decision": "ready_for_review"}}),
        encoding="utf-8",
    )
    pd.DataFrame([{"account": "DU1", "symbol": "QQQ", "action": "SELL", "quantity": 10.0}]).to_parquet(
        execution_dir / "submitted_orders.parquet",
        index=False,
    )
    (execution_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"created_at_utc": "2026-05-11T14:00:00Z", "status": "submitted"},
                "preflight": {"plan_fingerprint": "abc"},
                "unlock": {"unlocked": True},
            }
        ),
        encoding="utf-8",
    )
    (auto_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"created_at_utc": "2026-05-11T14:00:00Z", "status": "complete"},
                "decision": "submitted",
                "reason": "paper order submitted",
                "signal": {"ticket": {"action": "SELL", "quantity": 10.0}},
                "execution": {"summary": {"submitted_orders": 1}},
            }
        ),
        encoding="utf-8",
    )
    (signal_dir / "latest_signal.yaml").write_text(
        yaml.safe_dump(
            {
                "timestamp": "2026-05-11 10:00:00-04:00",
                "session": "2026-05-11",
                "bar_index": 2,
                "target_open_next": 99.0,
                "target_can_open_trade": True,
                "target_ret_6": -0.01,
                "target_ret_12": 0.02,
                "risk_off_score": 0.20,
                "risk_off_min": 0.10,
                "prev_vix_z20": 0.30,
                "vix_z20_min": 0.20,
                "spread_credit_12": 0.01,
                "spread_credit_12_max": 0.0,
                "h1c_risk_off_pass": True,
                "h1c_vix_pass": True,
                "h1c_credit_pass": False,
                "h1c_can_enter": True,
                "h1c_signal_short": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    price_dir = tmp_path / "data" / "cleaned" / "15min" / "QQQ"
    price_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"timestamp": "2026-05-11 09:30:00-04:00", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000},
            {"timestamp": "2026-05-11 09:45:00-04:00", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1200},
        ]
    ).to_parquet(price_dir / "QQQ_15min_clean.parquet", index=False)

    snapshot = h1c_operations_snapshot(root=tmp_path)

    assert snapshot["summary"]["current_status"] == "open"
    assert snapshot["summary"]["submitted_orders"] == 1
    assert snapshot["planned_orders"][0]["plan_decision"] == "ready_for_review"
    assert snapshot["submitted_orders"][0]["execution_status"] == "submitted"
    assert snapshot["charts"]["price"][-1]["close"] == 101.5
    assert snapshot["charts"]["pnl"][0]["cumulative_realized_pnl"] == 0.0
    assert snapshot["signal_diagnostics"]["summary"]["passed_conditions"] == 4
    assert snapshot["signal_diagnostics"]["conditions"][1]["key"] == "target_ret_12"
    assert snapshot["signal_diagnostics"]["conditions"][1]["margin"] == -0.02

from __future__ import annotations

from pathlib import Path

import yaml

from src.execution import paper_cycle_runner
from src.execution.paper_cycle_runner import run_paper_cycle
from src.execution.paper_data_refresh import PaperDataRefreshPaths
from src.execution.paper_h1c_signal_runner import PaperH1CRunnerPaths
from src.execution.paper_state_store import PaperStatePaths


def test_paper_cycle_orchestrates_refresh_signal_and_state(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "cycle.yaml"
    refresh_config_path = tmp_path / "refresh.yaml"
    signal_config_path = tmp_path / "signal.yaml"
    state_config_path = tmp_path / "state.yaml"
    refresh_config_path.write_text("refresh: {}\n", encoding="utf-8")
    signal_config_path.write_text("runner: {}\n", encoding="utf-8")
    state_config_path.write_text("state: {}\n", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "cycle": {
                    "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                    "account": "DU123",
                    "symbol": "QQQ",
                    "refresh_data": True,
                    "apply_state": True,
                    "signal_only": True,
                    "default_skip_download": False,
                    "default_skip_cboe": True,
                    "default_refresh_dry_run": False,
                },
                "components": {
                    "data_refresh_config_path": refresh_config_path.as_posix(),
                    "signal_runner_config_path": signal_config_path.as_posix(),
                    "state_config_path": state_config_path.as_posix(),
                },
                "outputs": {"output_dir": (tmp_path / "cycle_runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_refresh(**kwargs):
        assert kwargs["config_path"] == refresh_config_path
        assert kwargs["skip_download"] is True
        assert kwargs["skip_cboe"] is True
        root = Path(kwargs["output_dir"]) / "refresh_run"
        root.mkdir(parents=True)
        paths = PaperDataRefreshPaths(output_dir=root, manifest_path=root / "manifest.yaml", report_path=root / "report.md")
        paths.manifest_path.write_text("run:\n  status: complete\n", encoding="utf-8")
        paths.report_path.write_text("# refresh\n", encoding="utf-8")
        return paths, {"run": {"status": "complete"}, "date_window": {"start_date": "2026-05-01", "end_date": "2026-05-08"}, "warnings": []}

    def fake_signal(**kwargs):
        assert kwargs["config_path"] == signal_config_path
        root = Path(kwargs["output_dir"]) / "signal_run"
        root.mkdir(parents=True)
        paths = PaperH1CRunnerPaths(
            output_dir=root,
            signals_path=root / "signals.parquet",
            latest_signal_path=root / "latest_signal.yaml",
            ticket_path=root / "paper_ticket.yaml",
            manifest_path=root / "manifest.yaml",
            report_path=root / "report.md",
        )
        paths.ticket_path.write_text("action: SELL\n", encoding="utf-8")
        paths.report_path.write_text("# signal\n", encoding="utf-8")
        return paths, {
            "ticket": {
                "signal_timestamp": "2026-05-08 14:00:00-04:00",
                "session": "2026-05-08",
                "action": "SELL",
                "quantity": 1.0,
                "status": "paper_ticket_only",
            },
            "thresholds": {"source_fold": 4},
            "warnings": [],
        }

    def fake_state(**kwargs):
        assert kwargs["ticket_path"].name == "paper_ticket.yaml"
        assert kwargs["config_path"] == state_config_path
        root = Path(kwargs["output_dir"]) / "state_run"
        root.mkdir(parents=True)
        paths = PaperStatePaths(
            output_dir=root,
            state_path=tmp_path / "state_out.yaml",
            event_log_path=tmp_path / "events.parquet",
            event_path=root / "event.yaml",
            report_path=root / "report.md",
        )
        paths.report_path.write_text("# state\n", encoding="utf-8")
        return paths, {
            "state": {"status": "pending_entry"},
            "event": {"event_type": "pending_entry_created", "previous_status": "flat", "new_status": "pending_entry"},
        }

    monkeypatch.setattr(paper_cycle_runner, "run_paper_data_refresh", fake_refresh)
    monkeypatch.setattr(paper_cycle_runner, "run_h1c_signal_runner", fake_signal)
    monkeypatch.setattr(paper_cycle_runner, "apply_ticket", fake_state)

    paths, manifest = run_paper_cycle(config_path=config_path, skip_download=True, skip_cboe=True)

    assert paths.manifest_path.exists()
    assert paths.report_path.exists()
    assert manifest["steps"]["refresh_data"] is True
    assert manifest["signal"]["latest"]["action"] == "SELL"
    assert manifest["state"]["event"]["event_type"] == "pending_entry_created"

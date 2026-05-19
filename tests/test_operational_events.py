from __future__ import annotations

import json

from src.execution.operational_events import append_operational_event, read_recent_operational_events


def test_append_operational_event_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "events.jsonl"

    event = append_operational_event(
        {
            "event_type": "daemon_error",
            "component": "h1c_auto_daemon",
            "severity": "warning",
            "error": "boom",
        },
        path=path,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["event_id"] == event["event_id"]
    assert stored["created_at_utc"].endswith("Z")
    assert stored["event_type"] == "daemon_error"


def test_read_recent_operational_events_returns_newest_first(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    for index in range(3):
        append_operational_event(
            {"event_type": f"event_{index}", "component": "test", "severity": "info"},
            path=path,
        )

    recent = read_recent_operational_events(path, limit=2)

    assert [event["event_type"] for event in recent] == ["event_2", "event_1"]

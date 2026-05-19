from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


DEFAULT_OPERATIONAL_EVENTS_PATH = Path("results/paper/operational_events/events.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_operational_event(
    event: Mapping[str, Any],
    *,
    path: str | Path = DEFAULT_OPERATIONAL_EVENTS_PATH,
) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("event_id", uuid4().hex)
    payload.setdefault("created_at_utc", utc_now())
    payload.setdefault("schema_version", 1)
    if not str(payload.get("event_type") or "").strip():
        raise ValueError("operational event requires event_type")
    if not str(payload.get("component") or "").strip():
        raise ValueError("operational event requires component")

    events_path = Path(path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")
    return payload


def read_recent_operational_events(
    path: str | Path = DEFAULT_OPERATIONAL_EVENTS_PATH,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    events_path = Path(path)
    if not events_path.exists():
        return []
    records: deque[dict[str, Any]] = deque(maxlen=max(0, int(limit)))
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
                records.append(raw if isinstance(raw, dict) else {"value": raw})
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "event_type": "operational_event_parse_error",
                        "component": "operational_events",
                        "severity": "warning",
                        "line_number": line_number,
                        "error": str(exc),
                    }
                )
    return list(reversed(records))

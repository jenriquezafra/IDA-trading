from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.strategy.spec import StrategySpec


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_run_id(*parts: str) -> str:
    normalized = "_".join(part.strip().upper().replace("/", "_") for part in parts if part.strip())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = "_".join(part.strip().lower().replace("/", "_") for part in parts if part.strip())
    return f"{stem}_{digest}"


def fingerprint_path(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    raw = f"{p.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return "statsha1:" + hashlib.sha1(raw).hexdigest()


@dataclass(frozen=True)
class ResearchManifest:
    schema_version: int
    run: dict[str, Any]
    strategy: dict[str, Any]
    data: dict[str, Any]
    costs: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run": self.run,
            "strategy": self.strategy,
            "data": self.data,
            "costs": self.costs,
            "artifacts": self.artifacts,
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return output


def manifest_from_strategy(
    strategy: StrategySpec,
    *,
    run_type: str,
    feature_path: str | Path,
    artifacts: dict[str, str] | None = None,
) -> ResearchManifest:
    path = Path(feature_path)
    data = {
        "target_symbol": strategy.target_symbol,
        "timeframe": strategy.timeframe,
        "feature_set_id": strategy.feature_set_id,
        "feature_path": path.as_posix(),
        "feature_fingerprint": fingerprint_path(path) if path.exists() else "MISSING",
        "split_policy_id": strategy.split_policy_id,
    }
    run_id = build_run_id(run_type, strategy.strategy_id, strategy.target_symbol, strategy.timeframe, strategy.alpha_id)
    return ResearchManifest(
        schema_version=1,
        run={
            "run_id": run_id,
            "run_type": run_type,
            "created_at_utc": utc_now(),
            "status": "draft",
        },
        strategy=strategy.to_dict(),
        data=data,
        costs={"cost_profile": strategy.cost_profile_id},
        artifacts=artifacts or {},
    )

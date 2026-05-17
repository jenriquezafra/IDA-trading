from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManifestValidation:
    path: Path
    ok: bool
    errors: tuple[str, ...]


REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "run")
REQUIRED_RUN_KEYS = ("run_id",)


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Cheap fingerprint for large legacy artifacts.

    This is not a content hash. It is meant for local indexing where reading
    every large parquet file would slow down the dashboard.
    """

    stat = Path(path).stat()
    raw = f"{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    return "statsha1:" + hashlib.sha1(raw).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {manifest_path}")
    return data


def validate_manifest(path: str | Path) -> ManifestValidation:
    manifest_path = Path(path)
    errors: list[str] = []
    try:
        data = load_manifest(manifest_path)
    except Exception as exc:
        return ManifestValidation(manifest_path, False, (str(exc),))

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            errors.append(f"missing top-level key: {key}")

    run = data.get("run")
    if not isinstance(run, dict):
        errors.append("run must be a mapping")
    else:
        for key in REQUIRED_RUN_KEYS:
            if not run.get(key):
                errors.append(f"missing run key: {key}")

    return ManifestValidation(manifest_path, not errors, tuple(errors))


def get_git_commit(cwd: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def get_git_branch(cwd: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def get_git_dirty(cwd: str | Path = ".") -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())

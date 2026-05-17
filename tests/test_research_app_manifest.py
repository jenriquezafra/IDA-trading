from __future__ import annotations

from pathlib import Path

from src.research_app.manifest import hash_file, load_manifest, validate_manifest


def test_load_and_validate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yml"
    manifest.write_text(
        """
schema_version: 1
run:
  run_id: RUN_TEST
data:
  instrument: SPY
  timeframe: 5min
""",
        encoding="utf-8",
    )

    loaded = load_manifest(manifest)
    validation = validate_manifest(manifest)

    assert loaded["run"]["run_id"] == "RUN_TEST"
    assert validation.ok
    assert validation.errors == ()


def test_manifest_validation_reports_missing_run_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yml"
    manifest.write_text("schema_version: 1\nrun: {}\n", encoding="utf-8")

    validation = validate_manifest(manifest)

    assert not validation.ok
    assert "missing run key: run_id" in validation.errors


def test_hash_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("abc", encoding="utf-8")

    assert hash_file(path) == hash_file(path)

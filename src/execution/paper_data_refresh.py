from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src import data_download
from src import feature_engineering_cross_asset
from src.cross_asset_data import align_symbols, aligned_panel_path, clean_symbol, load_yaml, resolve_symbols, symbol_paths, write_coverage_report
from src.data import cboe_risk_context
from src.research.manifest import fingerprint_path


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_data_refresh.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/data_refresh")


@dataclass(frozen=True)
class PaperDataRefreshConfig:
    target_symbol: str
    lab_config_path: Path
    feature_config_path: Path
    cboe_config_path: Path
    symbols: tuple[str, ...]
    lookback_days: int
    end_date: str | None
    download: bool
    clean: bool
    align: bool
    build_features: bool
    refresh_cboe: bool
    keep_incomplete_sessions_for_paper: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PaperDataRefreshConfig":
        refresh = dict(raw.get("refresh", {}))
        outputs = dict(raw.get("outputs", {}))
        symbols = tuple(str(symbol).upper() for symbol in refresh.get("symbols", []) if str(symbol).strip())
        config = cls(
            target_symbol=str(refresh.get("target_symbol", "QQQ")).strip().upper(),
            lab_config_path=Path(refresh.get("lab_config_path", "configs/hmm_lab_15min_expansion_repair.yaml")),
            feature_config_path=Path(refresh.get("feature_config_path", "configs/features/cross_asset_15min_liquid.yaml")),
            cboe_config_path=Path(refresh.get("cboe_config_path", "configs/data/cboe_risk_context.yaml")),
            symbols=symbols,
            lookback_days=int(refresh.get("lookback_days", 10)),
            end_date=None if refresh.get("end_date") in {None, ""} else str(refresh.get("end_date")),
            download=bool(refresh.get("download", True)),
            clean=bool(refresh.get("clean", True)),
            align=bool(refresh.get("align", True)),
            build_features=bool(refresh.get("build_features", True)),
            refresh_cboe=bool(refresh.get("refresh_cboe", False)),
            keep_incomplete_sessions_for_paper=bool(refresh.get("keep_incomplete_sessions_for_paper", True)),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.target_symbol:
            raise ValueError("target_symbol is required")
        if self.lookback_days < 0:
            raise ValueError("lookback_days must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ["lab_config_path", "feature_config_path", "cboe_config_path", "output_dir"]:
            data[key] = data[key].as_posix()
        data["symbols"] = list(self.symbols)
        return data


@dataclass(frozen=True)
class PaperDataRefreshPaths:
    output_dir: Path
    manifest_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_refresh_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperDataRefreshConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return PaperDataRefreshConfig.from_mapping(raw)


def _effective_lab_config(config: PaperDataRefreshConfig) -> dict[str, Any]:
    lab_config = copy.deepcopy(load_yaml(config.lab_config_path))
    lab_config.setdefault("lab", {})["target_symbol"] = config.target_symbol
    if config.keep_incomplete_sessions_for_paper:
        lab_config.setdefault("session", {})["drop_incomplete_sessions"] = False
    return lab_config


def _latest_clean_session(lab_config: dict[str, Any], symbol: str, target_symbol: str) -> str | None:
    path = symbol_paths(lab_config, symbol, target_symbol=target_symbol).cleaned_file
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["session"])
    if frame.empty:
        return None
    return str(frame["session"].max())


def _latest_feature_session(lab_config: dict[str, Any], feature_config_path: Path, target_symbol: str) -> str | None:
    path = _feature_output_path(lab_config, feature_config_path, target_symbol)
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["session"])
    if frame.empty:
        return None
    return str(frame["session"].max())


def _resolve_date_window(config: PaperDataRefreshConfig, lab_config: dict[str, Any]) -> tuple[str, str]:
    end = config.end_date or date.today().isoformat()
    latest = _latest_clean_session(lab_config, config.target_symbol, config.target_symbol)
    if latest is None:
        latest = _latest_feature_session(lab_config, config.feature_config_path, config.target_symbol)
    lab_start = str(lab_config.get("lab", {}).get("start_date", end))
    if latest:
        start = (pd.Timestamp(latest).date() - timedelta(days=config.lookback_days)).isoformat()
        start = max(start, lab_start)
    else:
        start = lab_start
    return start, end


def _merge_ohlcv(existing: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in [existing, incoming] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp", kind="stable").reset_index(drop=True)


def refresh_symbol_raw(
    lab_config: dict[str, Any],
    symbol: str,
    target_symbol: str,
    start_date: str,
    end_date: str,
    *,
    downloader: Callable[[dict[str, Any], str | None, str | None], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    from src.cross_asset_data import build_symbol_config

    data_download.load_dotenv()
    symbol_config = build_symbol_config(lab_config, symbol, target_symbol=target_symbol)
    paths = symbol_paths(lab_config, symbol, target_symbol=target_symbol)
    fetch = downloader or data_download.download_polygon_ohlcv
    incoming = fetch(symbol_config, start_date, end_date)
    existing = pd.read_parquet(paths.raw_file) if paths.raw_file.exists() else None
    merged = _merge_ohlcv(existing, incoming)
    paths.raw_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(paths.raw_file, index=False)
    return {
        "symbol": symbol.upper(),
        "raw_file": paths.raw_file.as_posix(),
        "downloaded_rows": int(len(incoming)),
        "merged_raw_rows": int(len(merged)),
        "start_timestamp": None if merged.empty else pd.to_datetime(merged["timestamp"]).min().isoformat(),
        "end_timestamp": None if merged.empty else pd.to_datetime(merged["timestamp"]).max().isoformat(),
    }


def _feature_output_path(lab_config: dict[str, Any], feature_config_path: Path, target_symbol: str) -> Path:
    feature_config = load_yaml(feature_config_path)
    return feature_engineering_cross_asset.feature_output_path(lab_config, feature_config, target_symbol)


def _build_features(lab_config: dict[str, Any], feature_config_path: Path, target_symbol: str) -> Path:
    feature_config = load_yaml(feature_config_path)
    panel = pd.read_parquet(aligned_panel_path(lab_config, target_symbol))
    features = feature_engineering_cross_asset.build_cross_asset_features(panel, lab_config, feature_config, target_symbol=target_symbol)
    output_path = feature_engineering_cross_asset.feature_output_path(lab_config, feature_config, target_symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    report = feature_engineering_cross_asset.build_report(features, feature_config, target_symbol, output_path)
    report_path = feature_engineering_cross_asset.report_output_path(lab_config, feature_config, target_symbol)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(feature_engineering_cross_asset.render_report(report, feature_config), encoding="utf-8")
    return output_path


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    symbols = manifest["symbols"]
    lines = [
        "# Paper data refresh",
        "",
        f"- Created UTC: `{manifest['run']['created_at_utc']}`",
        f"- Status: `{manifest['run']['status']}`",
        f"- Target: `{manifest['target_symbol']}`",
        f"- Date window: `{manifest['date_window']['start_date']} -> {manifest['date_window']['end_date']}`",
        f"- Symbols: `{len(symbols)}`",
        f"- Download: `{manifest['steps']['download']}`",
        f"- Clean: `{manifest['steps']['clean']}`",
        f"- Align: `{manifest['steps']['align']}`",
        f"- Build features: `{manifest['steps']['build_features']}`",
        f"- Refresh Cboe: `{manifest['steps']['refresh_cboe']}`",
        f"- Keep incomplete sessions for paper: `{manifest['keep_incomplete_sessions_for_paper']}`",
        "",
        "## Outputs",
        "",
        f"- Feature path: `{manifest['outputs'].get('features_path', '')}`",
        f"- Aligned panel: `{manifest['outputs'].get('aligned_panel_path', '')}`",
        f"- Cboe context: `{manifest['outputs'].get('cboe_context_path', '')}`",
        "",
        "## Symbol Status",
        "",
        "| symbol | downloaded_rows | merged_raw_rows | clean_rows | clean_end |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in manifest.get("symbol_rows", []):
        lines.append(
            f"| {row['symbol']} | {row.get('downloaded_rows', 0)} | {row.get('merged_raw_rows', 0)} | "
            f"{row.get('clean_rows', 0)} | {row.get('clean_end_timestamp') or ''} |"
        )
    if manifest.get("warnings"):
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in manifest["warnings"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_paper_data_refresh(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = False,
    skip_download: bool = False,
    skip_cboe: bool = False,
    output_dir: str | Path | None = None,
    downloader: Callable[[dict[str, Any], str | None, str | None], pd.DataFrame] | None = None,
) -> tuple[PaperDataRefreshPaths, dict[str, Any]]:
    config = load_refresh_config(config_path)
    lab_config = _effective_lab_config(config)
    if end_date is not None:
        config = PaperDataRefreshConfig.from_mapping({"refresh": {**config.to_dict(), "end_date": end_date}, "outputs": {"output_dir": config.output_dir.as_posix()}})
    resolved_start, resolved_end = _resolve_date_window(config, lab_config)
    if start_date is not None:
        resolved_start = start_date
    if end_date is not None:
        resolved_end = end_date
    symbols = resolve_symbols(lab_config, target_symbol=config.target_symbol, override_symbols=list(config.symbols) or None)

    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = PaperDataRefreshPaths(output_dir=root, manifest_path=root / "manifest.yaml", report_path=root / "report.md")
    root.mkdir(parents=True, exist_ok=True)

    steps = {
        "download": bool(config.download and not skip_download),
        "clean": bool(config.clean),
        "align": bool(config.align),
        "build_features": bool(config.build_features),
        "refresh_cboe": bool(config.refresh_cboe and not skip_cboe),
    }
    symbol_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if dry_run:
        steps = {key: False for key in steps}
        warnings.append("dry_run=true; no data files were modified")

    if steps["download"]:
        for symbol in symbols:
            symbol_rows.append(refresh_symbol_raw(lab_config, symbol, config.target_symbol, resolved_start, resolved_end, downloader=downloader))
    else:
        for symbol in symbols:
            paths_for_symbol = symbol_paths(lab_config, symbol, target_symbol=config.target_symbol)
            symbol_rows.append({"symbol": symbol, "raw_file": paths_for_symbol.raw_file.as_posix(), "downloaded_rows": 0, "merged_raw_rows": 0})

    if steps["clean"]:
        for row in symbol_rows:
            report = clean_symbol(lab_config, row["symbol"], target_symbol=config.target_symbol)
            row["clean_rows"] = int(report.rows_clean)
            row["clean_end_timestamp"] = report.end_timestamp
            row["clean_file"] = report.output_path.as_posix()

    coverage_report_path = None
    if not dry_run:
        coverage_rows = []
        from src.cross_asset_data import coverage_for_symbol

        for symbol in symbols:
            coverage_rows.append(coverage_for_symbol(lab_config, symbol, target_symbol=config.target_symbol))
        coverage_report_path = write_coverage_report(lab_config, coverage_rows, config.target_symbol)

    alignment_report = None
    if steps["align"]:
        alignment_report = align_symbols(lab_config, symbols, config.target_symbol)

    features_path = _feature_output_path(lab_config, config.feature_config_path, config.target_symbol)
    if steps["build_features"]:
        features_path = _build_features(lab_config, config.feature_config_path, config.target_symbol)

    cboe_outputs = None
    if steps["refresh_cboe"]:
        cboe_config = cboe_risk_context.load_config(config.cboe_config_path)
        cboe_config["end_date"] = resolved_end
        cboe_outputs = cboe_risk_context.run(cboe_config)

    outputs = {
        "features_path": features_path.as_posix(),
        "features_fingerprint": fingerprint_path(features_path) if features_path.exists() else "MISSING",
        "coverage_report_path": coverage_report_path.as_posix() if coverage_report_path is not None else "",
        "aligned_panel_path": alignment_report.output_path if alignment_report is not None else "",
        "cboe_context_path": cboe_outputs.context_path.as_posix() if cboe_outputs is not None else str(cboe_risk_context.load_config(config.cboe_config_path).get("paths", {}).get("context", "")),
    }
    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "paper_data_refresh",
            "created_at_utc": created,
            "status": "dry_run" if dry_run else "complete",
        },
        "target_symbol": config.target_symbol,
        "config": config.to_dict(),
        "date_window": {"start_date": resolved_start, "end_date": resolved_end},
        "symbols": symbols,
        "steps": steps,
        "keep_incomplete_sessions_for_paper": config.keep_incomplete_sessions_for_paper,
        "symbol_rows": symbol_rows,
        "outputs": outputs,
        "warnings": warnings,
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, manifest)
    return paths, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh paper-trading data/features for H1c")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-cboe", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, manifest = run_paper_data_refresh(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=args.dry_run,
        skip_download=args.skip_download,
        skip_cboe=args.skip_cboe,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": {key: str(value) for key, value in asdict(paths).items()}, "summary": manifest}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

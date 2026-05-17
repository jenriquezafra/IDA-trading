from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src import data_cleaning, data_download


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_SAFETY_COLUMNS = [
    "target_open_next",
    "next_open_timestamp",
    "target_crosses_session_close",
    "can_open_trade",
    "force_flat_bar",
    "trade_could_remain_open_past_close",
]


@dataclass(frozen=True)
class SymbolPaths:
    symbol: str
    raw_source_file: Path
    raw_file: Path
    cleaned_file: Path
    quality_report_file: Path


@dataclass(frozen=True)
class CoverageRow:
    symbol: str
    raw_file: str
    cleaned_file: str
    raw_rows: int
    clean_rows: int
    clean_sessions: int
    start_timestamp: str | None
    end_timestamp: str | None
    status: str


@dataclass(frozen=True)
class AlignmentReport:
    target_symbol: str
    universe_id: str
    timeframe: str
    missing_policy: str
    symbols: list[str]
    target_rows: int
    aligned_rows: int
    dropped_target_rows: int
    missing_vs_target: dict[str, int]
    output_path: str
    missing_detail_path: str | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _flatten_symbols(value: Any) -> list[str]:
    symbols: list[str] = []
    if value is None:
        return symbols
    if isinstance(value, str):
        return [value.upper()]
    if isinstance(value, list):
        for item in value:
            symbols.extend(_flatten_symbols(item))
        return symbols
    if isinstance(value, dict):
        for item in value.values():
            symbols.extend(_flatten_symbols(item))
        return symbols
    raise TypeError(f"Unsupported symbol container: {type(value)!r}")


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            output.append(symbol)
    return output


def resolve_symbols(config: dict[str, Any], target_symbol: str | None = None, override_symbols: list[str] | None = None) -> list[str]:
    if override_symbols:
        return _dedupe_symbols(override_symbols)

    lab_cfg = config.get("lab", {})
    target = (target_symbol or lab_cfg.get("target_symbol", "SPY")).upper()
    universe = load_yaml(lab_cfg.get("universe_config", "configs/universes/core_cross_asset.yaml"))
    symbols = _flatten_symbols(universe.get(lab_cfg.get("context_key", "context_universe_core")))
    symbols.extend(_flatten_symbols(lab_cfg.get("optional_symbols", [])))
    if lab_cfg.get("include_target_in_context", True):
        symbols.insert(0, target)
    return _dedupe_symbols(symbols)


def symbol_paths(config: dict[str, Any], symbol: str, target_symbol: str | None = None) -> SymbolPaths:
    lab_cfg = config.get("lab", {})
    paths_cfg = config.get("paths", {})
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    source_interval = config.get("polygon", {}).get("source_interval", timeframe)
    target = (target_symbol or lab_cfg.get("target_symbol", "SPY")).upper()
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    symbol_upper = symbol.upper()

    raw_root = Path(paths_cfg.get("raw_dir", "data/raw/polygon"))
    cleaned_root = Path(paths_cfg.get("cleaned_dir", "data/cleaned"))
    reports_root = Path(paths_cfg.get("data_coverage_dir", "reports/data_coverage"))

    return SymbolPaths(
        symbol=symbol_upper,
        raw_source_file=raw_root / source_interval / symbol_upper / f"{symbol_upper}_{source_interval}.parquet",
        raw_file=raw_root / timeframe / symbol_upper / f"{symbol_upper}_{timeframe}.parquet",
        cleaned_file=cleaned_root / timeframe / symbol_upper / f"{symbol_upper}_{timeframe}_clean.parquet",
        quality_report_file=reports_root / target / universe_id / f"{symbol_upper}_{timeframe}_quality.md",
    )


def build_symbol_config(config: dict[str, Any], symbol: str, target_symbol: str | None = None) -> dict[str, Any]:
    lab_cfg = config.get("lab", {})
    paths = symbol_paths(config, symbol, target_symbol=target_symbol)
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    start_date = lab_cfg.get("start_date")
    end_date = lab_cfg.get("end_date")

    return {
        "project": {
            "name": config.get("project", {}).get("name", "ida-trading"),
            "asset": symbol.upper(),
            "frequency": timeframe,
            "timezone": config.get("project", {}).get("timezone", "America/New_York"),
        },
        "paths": {
            "raw_data": str(paths.raw_file.parent),
            "cleaned_data": str(paths.cleaned_file.parent),
            "reports": str(paths.quality_report_file.parent),
        },
        "data": {
            "provider": lab_cfg.get("provider", "polygon"),
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "download_interval": timeframe,
            "input_file": str(paths.raw_file),
            "cleaned_file": str(paths.cleaned_file),
            "quality_report_file": str(paths.quality_report_file),
            "timestamp_col": "timestamp",
            "polygon": {
                "api_key_env": config.get("polygon", {}).get("api_key_env", "POLYGON_API_KEY"),
                "source_interval": config.get("polygon", {}).get("source_interval", timeframe),
                "raw_source_file": str(paths.raw_source_file),
                "adjusted": bool(config.get("polygon", {}).get("adjusted", True)),
                "default_years": int(config.get("polygon", {}).get("default_years", 5)),
            },
        },
        "session": config.get("session", {}),
        "calendar": config.get("calendar", {}),
        "quality": config.get("quality", {}),
        "labeling": config.get("labeling", {}),
        "backtest": config.get("backtest", {}),
    }


def download_symbol(
    config: dict[str, Any],
    symbol: str,
    target_symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    skip_existing: bool = False,
) -> Path:
    data_download.load_dotenv()
    symbol_config = build_symbol_config(config, symbol, target_symbol=target_symbol)
    output_path = Path(symbol_config["data"]["input_file"])
    if skip_existing and output_path.exists():
        return output_path

    provider = symbol_config["data"].get("provider", "polygon")
    if provider != "polygon":
        raise ValueError(f"cross_asset_data currently supports polygon only, got: {provider}")

    frame = data_download.download_polygon_ohlcv(symbol_config, start_date=start_date, end_date=end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path


def clean_symbol(config: dict[str, Any], symbol: str, target_symbol: str | None = None) -> data_cleaning.DataQualityReport:
    symbol_config = build_symbol_config(config, symbol, target_symbol=target_symbol)
    input_path = Path(symbol_config["data"]["input_file"])
    output_path = Path(symbol_config["data"]["cleaned_file"])
    report_path = Path(symbol_config["data"]["quality_report_file"])

    raw = data_cleaning.load_ohlcv(input_path)
    cleaned, report = data_cleaning.clean_ohlcv(raw, symbol_config, input_path, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    report_path.write_text(data_cleaning.render_quality_report(report), encoding="utf-8")
    return report


def _read_cleaned_symbol(config: dict[str, Any], symbol: str, target_symbol: str | None = None) -> pd.DataFrame:
    path = symbol_paths(config, symbol, target_symbol=target_symbol).cleaned_file
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found for {symbol}: {path}")
    return pd.read_parquet(path)


def coverage_for_symbol(config: dict[str, Any], symbol: str, target_symbol: str | None = None) -> CoverageRow:
    paths = symbol_paths(config, symbol, target_symbol=target_symbol)
    raw_rows = 0
    clean_rows = 0
    clean_sessions = 0
    start_timestamp = None
    end_timestamp = None
    status = "missing_raw"

    if paths.raw_file.exists():
        raw_rows = int(len(pd.read_parquet(paths.raw_file, columns=["timestamp"])))
        status = "raw_only"
    if paths.cleaned_file.exists():
        cleaned = pd.read_parquet(paths.cleaned_file, columns=["timestamp", "session"])
        clean_rows = int(len(cleaned))
        clean_sessions = int(cleaned["session"].nunique()) if not cleaned.empty else 0
        if not cleaned.empty:
            timestamps = pd.to_datetime(cleaned["timestamp"])
            start_timestamp = timestamps.min().isoformat()
            end_timestamp = timestamps.max().isoformat()
        status = "cleaned"

    return CoverageRow(
        symbol=symbol.upper(),
        raw_file=str(paths.raw_file),
        cleaned_file=str(paths.cleaned_file),
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        clean_sessions=clean_sessions,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        status=status,
    )


def write_coverage_report(config: dict[str, Any], rows: list[CoverageRow], target_symbol: str) -> Path:
    lab_cfg = config.get("lab", {})
    paths_cfg = config.get("paths", {})
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    output_dir = Path(paths_cfg.get("data_coverage_dir", "reports/data_coverage")) / target_symbol.upper() / universe_id
    metadata_dir = Path(paths_cfg.get("metadata_dir", "data/metadata"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([asdict(row) for row in rows])
    metadata_path = metadata_dir / f"{target_symbol.upper()}_{timeframe}_{universe_id}_coverage.parquet"
    frame.to_parquet(metadata_path, index=False)

    lines = [
        f"# Data Coverage - {target_symbol.upper()} {timeframe} {universe_id}",
        "",
        f"- Metadata: `{metadata_path}`",
        "",
        "| symbol | status | raw_rows | clean_rows | clean_sessions | start | end |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.symbol} | {row.status} | {row.raw_rows} | {row.clean_rows} | "
            f"{row.clean_sessions} | {row.start_timestamp or ''} | {row.end_timestamp or ''} |"
        )
    report_path = output_dir / f"{target_symbol.upper()}_{timeframe}_coverage.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _prefix_symbol_columns(frame: pd.DataFrame, symbol: str, keep_safety: bool = False) -> pd.DataFrame:
    base_columns = ["timestamp", "session", "bar_index"]
    symbol_columns = [column for column in OHLCV_COLUMNS if column in frame.columns]
    if keep_safety:
        symbol_columns.extend([column for column in TARGET_SAFETY_COLUMNS if column in frame.columns])

    output = frame.loc[:, base_columns + symbol_columns].copy()
    rename = {column: f"{symbol.upper()}__{column}" for column in symbol_columns}
    return output.rename(columns=rename)


def build_aligned_panel(
    frames: dict[str, pd.DataFrame],
    target_symbol: str,
    missing_policy: str = "drop_core_missing",
) -> tuple[pd.DataFrame, dict[str, int]]:
    symbols = _dedupe_symbols(list(frames.keys()))
    target = target_symbol.upper()
    if target not in frames:
        raise ValueError(f"target_symbol {target} is not present in frames")
    if missing_policy != "drop_core_missing":
        raise ValueError(f"Unsupported missing_policy: {missing_policy}")

    target_timestamps = set(pd.to_datetime(frames[target]["timestamp"]))
    common_timestamps = set(target_timestamps)
    missing_vs_target: dict[str, int] = {}
    for symbol in symbols:
        timestamps = set(pd.to_datetime(frames[symbol]["timestamp"]))
        missing_vs_target[symbol] = int(len(target_timestamps - timestamps))
        common_timestamps &= timestamps

    common_index = pd.Index(sorted(common_timestamps), name="timestamp")
    target_full = frames[target].sort_values(["session", "bar_index"], kind="stable").copy()
    if "target_open_next" not in target_full.columns and "open" in target_full.columns:
        target_full["target_open_next"] = target_full.groupby("session", sort=False)["open"].shift(-1)

    target_base = target_full[pd.to_datetime(target_full["timestamp"]).isin(common_index)].copy()
    target_base = target_base.sort_values(["timestamp", "session", "bar_index"], kind="stable")
    panel = _prefix_symbol_columns(target_base, target, keep_safety=True)

    for symbol in symbols:
        if symbol == target:
            continue
        symbol_frame = frames[symbol][pd.to_datetime(frames[symbol]["timestamp"]).isin(common_index)].copy()
        symbol_prefixed = _prefix_symbol_columns(symbol_frame, symbol, keep_safety=False)
        panel = panel.merge(symbol_prefixed, on=["timestamp", "session", "bar_index"], how="inner", sort=False)

    for symbol in symbols:
        panel[f"is_available_{symbol}"] = True

    return panel.sort_values(["timestamp", "session", "bar_index"], kind="stable").reset_index(drop=True), missing_vs_target


def aligned_panel_path(config: dict[str, Any], target_symbol: str) -> Path:
    lab_cfg = config.get("lab", {})
    paths_cfg = config.get("paths", {})
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    return Path(paths_cfg.get("aligned_dir", "data/aligned")) / target_symbol.upper() / timeframe / universe_id / "panel.parquet"


def write_alignment_report(config: dict[str, Any], report: AlignmentReport) -> Path:
    paths_cfg = config.get("paths", {})
    output_dir = Path(paths_cfg.get("alignment_dir", "reports/alignment")) / report.target_symbol / report.universe_id
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Alignment - {report.target_symbol} {report.timeframe} {report.universe_id}",
        "",
        f"- Missing policy: `{report.missing_policy}`",
        f"- Output: `{report.output_path}`",
        f"- Target rows: `{report.target_rows}`",
        f"- Aligned rows: `{report.aligned_rows}`",
        f"- Dropped target rows: `{report.dropped_target_rows}`",
        f"- Missing detail: `{report.missing_detail_path or ''}`",
        "",
        "| symbol | missing_vs_target |",
        "| --- | ---: |",
    ]
    for symbol in report.symbols:
        lines.append(f"| {symbol} | {report.missing_vs_target.get(symbol, 0)} |")
    output_path = output_dir / f"{report.target_symbol}_{report.timeframe}_alignment.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def missing_detail_frame(frames: dict[str, pd.DataFrame], target_symbol: str, symbols: list[str]) -> pd.DataFrame:
    target = target_symbol.upper()
    target_times = frames[target].loc[:, ["timestamp", "session"]].copy()
    target_times["timestamp"] = pd.to_datetime(target_times["timestamp"])
    target_times["hour"] = target_times["timestamp"].dt.strftime("%H:00")

    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        symbol_upper = symbol.upper()
        if symbol_upper == target:
            continue
        available = set(pd.to_datetime(frames[symbol_upper]["timestamp"]))
        missing = target_times.loc[~target_times["timestamp"].isin(available)].copy()
        if missing.empty:
            continue
        grouped = missing.groupby(["session", "hour"], as_index=False).size().rename(columns={"size": "missing_bars"})
        grouped.insert(0, "symbol", symbol_upper)
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["symbol", "session", "hour", "missing_bars"])
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "session", "hour"], kind="stable").reset_index(drop=True)


def missing_detail_path(config: dict[str, Any], target_symbol: str) -> Path:
    lab_cfg = config.get("lab", {})
    paths_cfg = config.get("paths", {})
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    return Path(paths_cfg.get("alignment_dir", "reports/alignment")) / target_symbol.upper() / universe_id / f"{target_symbol.upper()}_{timeframe}_missing_detail.parquet"


def align_symbols(config: dict[str, Any], symbols: list[str], target_symbol: str) -> AlignmentReport:
    lab_cfg = config.get("lab", {})
    timeframe = lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "5min"))
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    missing_policy = config.get("alignment", {}).get("missing_policy", "drop_core_missing")
    normalized_symbols = _dedupe_symbols([target_symbol, *symbols])
    frames = {symbol: _read_cleaned_symbol(config, symbol, target_symbol=target_symbol) for symbol in normalized_symbols}
    panel, missing_vs_target = build_aligned_panel(frames, target_symbol, missing_policy=missing_policy)
    output_path = aligned_panel_path(config, target_symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    detail_path = missing_detail_path(config, target_symbol)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    missing_detail_frame(frames, target_symbol, normalized_symbols).to_parquet(detail_path, index=False)

    report = AlignmentReport(
        target_symbol=target_symbol.upper(),
        universe_id=universe_id,
        timeframe=timeframe,
        missing_policy=missing_policy,
        symbols=normalized_symbols,
        target_rows=int(len(frames[target_symbol.upper()])),
        aligned_rows=int(len(panel)),
        dropped_target_rows=int(len(frames[target_symbol.upper()]) - len(panel)),
        missing_vs_target=missing_vs_target,
        output_path=str(output_path),
        missing_detail_path=str(detail_path),
    )
    write_alignment_report(config, report)
    return report


def run_command(
    command: str,
    config_path: str | Path,
    target_symbol: str | None = None,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    skip_existing: bool = False,
) -> None:
    config = load_yaml(config_path)
    target = (target_symbol or config.get("lab", {}).get("target_symbol", "SPY")).upper()
    resolved_symbols = resolve_symbols(config, target_symbol=target, override_symbols=symbols)

    if command in {"download", "run-all"}:
        for symbol in resolved_symbols:
            path = download_symbol(config, symbol, target_symbol=target, start_date=start_date, end_date=end_date, skip_existing=skip_existing)
            print(f"Downloaded {symbol}: {path}")

    if command in {"clean", "run-all"}:
        for symbol in resolved_symbols:
            report = clean_symbol(config, symbol, target_symbol=target)
            print(f"Cleaned {symbol}: {report.rows_clean} rows -> {report.output_path}")

    rows = [coverage_for_symbol(config, symbol, target_symbol=target) for symbol in resolved_symbols]
    coverage_path = write_coverage_report(config, rows, target)
    print(f"Coverage report: {coverage_path}")

    if command in {"align", "run-all"}:
        report = align_symbols(config, resolved_symbols, target)
        print(f"Aligned panel: {report.output_path} ({report.aligned_rows} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-asset data download, cleaning and alignment.")
    parser.add_argument("command", choices=["download", "clean", "align", "run-all"])
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional explicit symbol list.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    run_command(
        command=args.command,
        config_path=args.config,
        target_symbol=args.target,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()

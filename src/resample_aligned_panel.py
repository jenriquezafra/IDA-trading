from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.cross_asset_data import aligned_panel_path, load_yaml


OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ResampleReport:
    target_symbol: str
    source_timeframe: str
    target_timeframe: str
    universe_id: str
    factor: int
    source_rows: int
    output_rows: int
    source_sessions: int
    output_sessions: int
    dropped_buckets: int
    source_path: str
    output_path: str


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    value = timeframe.strip().lower()
    aliases = {"m": "min", "mins": "min", "minute": "min", "minutes": "min"}
    for suffix, replacement in aliases.items():
        if value.endswith(suffix) and not value.endswith(replacement):
            value = f"{value[: -len(suffix)]}{replacement}"
            break
    return pd.Timedelta(value)


def resample_factor(source_timeframe: str, target_timeframe: str) -> int:
    source_delta = timeframe_delta(source_timeframe)
    target_delta = timeframe_delta(target_timeframe)
    if target_delta <= source_delta:
        raise ValueError("target_timeframe must be larger than source_timeframe")
    ratio = target_delta / source_delta
    factor = int(ratio)
    if factor != ratio:
        raise ValueError(f"{target_timeframe} is not an integer multiple of {source_timeframe}")
    return factor


def _config_with_timeframe(config: dict[str, Any], timeframe: str) -> dict[str, Any]:
    copied = copy.deepcopy(config)
    copied.setdefault("lab", {})["timeframe"] = timeframe
    copied.setdefault("project", {})["frequency"] = timeframe
    return copied


def panel_symbols(panel: pd.DataFrame) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for column in panel.columns:
        if "__" not in column:
            continue
        symbol, field = column.split("__", 1)
        if field in OHLCV_FIELDS and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def resample_panel(
    panel: pd.DataFrame,
    target_symbol: str,
    source_timeframe: str,
    target_timeframe: str,
    drop_incomplete_buckets: bool = True,
) -> tuple[pd.DataFrame, ResampleReport]:
    factor = resample_factor(source_timeframe, target_timeframe)
    target = target_symbol.upper()
    ordered = panel.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True).copy()
    ordered["_bucket"] = ordered["bar_index"].astype(int) // factor

    symbols = panel_symbols(ordered)
    aggregations: dict[str, tuple[str, str]] = {
        "timestamp": ("timestamp", "first"),
        "source_first_bar": ("bar_index", "first"),
        "source_last_bar": ("bar_index", "last"),
        "source_rows": ("bar_index", "size"),
    }
    for symbol in symbols:
        aggregations[f"{symbol}__open"] = (f"{symbol}__open", "first")
        aggregations[f"{symbol}__high"] = (f"{symbol}__high", "max")
        aggregations[f"{symbol}__low"] = (f"{symbol}__low", "min")
        aggregations[f"{symbol}__close"] = (f"{symbol}__close", "last")
        aggregations[f"{symbol}__volume"] = (f"{symbol}__volume", "sum")
    for column in [column for column in ordered.columns if column.startswith("is_available_")]:
        aggregations[column] = (column, "all")

    grouped = ordered.groupby(["session", "_bucket"], sort=False)
    resampled = grouped.agg(**aggregations).reset_index()
    before_filter = len(resampled)
    if drop_incomplete_buckets:
        resampled = resampled[resampled["source_rows"].eq(factor)].copy()
    dropped_buckets = before_filter - len(resampled)

    resampled["bar_index"] = resampled.groupby("session", sort=False).cumcount().astype(int)
    resampled = resampled.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)

    target_open_col = f"{target}__open"
    if target_open_col not in resampled.columns:
        raise ValueError(f"Target open column missing after resample: {target_open_col}")
    next_open = resampled.groupby("session", sort=False)[target_open_col].shift(-1)
    next_timestamp = resampled.groupby("session", sort=False)["timestamp"].shift(-1)
    has_next = next_open.notna()
    resampled[f"{target}__target_open_next"] = next_open
    resampled[f"{target}__next_open_timestamp"] = next_timestamp
    resampled[f"{target}__target_crosses_session_close"] = ~has_next
    resampled[f"{target}__can_open_trade"] = has_next
    resampled[f"{target}__force_flat_bar"] = ~has_next
    resampled[f"{target}__trade_could_remain_open_past_close"] = ~has_next

    ordered_columns = ["timestamp", "session", "bar_index"]
    helper_columns = [
        f"{target}__target_open_next",
        f"{target}__next_open_timestamp",
        f"{target}__target_crosses_session_close",
        f"{target}__can_open_trade",
        f"{target}__force_flat_bar",
        f"{target}__trade_could_remain_open_past_close",
    ]
    for symbol in symbols:
        ordered_columns.extend([f"{symbol}__{field}" for field in OHLCV_FIELDS if f"{symbol}__{field}" in resampled.columns])
        if symbol == target:
            ordered_columns.extend(helper_columns)
    ordered_columns.extend([column for column in resampled.columns if column.startswith("is_available_")])
    resampled = resampled.loc[:, [column for column in ordered_columns if column in resampled.columns]]

    report = ResampleReport(
        target_symbol=target,
        source_timeframe=source_timeframe,
        target_timeframe=target_timeframe,
        universe_id="",
        factor=factor,
        source_rows=len(panel),
        output_rows=len(resampled),
        source_sessions=int(panel["session"].nunique()),
        output_sessions=int(resampled["session"].nunique()),
        dropped_buckets=dropped_buckets,
        source_path="",
        output_path="",
    )
    return resampled, report


def render_report(report: ResampleReport) -> str:
    return "\n".join(
        [
            f"# Aligned Panel Resample - {report.target_symbol} {report.source_timeframe} to {report.target_timeframe}",
            "",
            f"- Universe: `{report.universe_id}`",
            f"- Factor: `{report.factor}` source bars per output bar",
            f"- Source rows: `{report.source_rows}`",
            f"- Output rows: `{report.output_rows}`",
            f"- Source sessions: `{report.source_sessions}`",
            f"- Output sessions: `{report.output_sessions}`",
            f"- Dropped incomplete buckets: `{report.dropped_buckets}`",
            f"- Source: `{report.source_path}`",
            f"- Output: `{report.output_path}`",
            "",
        ]
    )


def report_output_path(config: dict[str, Any], target_symbol: str, source_timeframe: str, target_timeframe: str) -> Path:
    lab_cfg = config.get("lab", {})
    universe_id = str(lab_cfg.get("universe_id", "core_cross_asset_v1"))
    reports_root = Path(config.get("paths", {}).get("alignment_dir", "reports/alignment"))
    return reports_root / target_symbol.upper() / universe_id / f"{target_symbol.upper()}_{source_timeframe}_to_{target_timeframe}_resample.md"


def run(
    config_path: str | Path,
    source_timeframe: str,
    target_timeframe: str | None = None,
    target_symbol: str | None = None,
    drop_incomplete_buckets: bool = True,
) -> Path:
    config = load_yaml(config_path)
    lab_cfg = config.get("lab", {})
    target = (target_symbol or lab_cfg.get("target_symbol", "SPY")).upper()
    output_timeframe = target_timeframe or str(lab_cfg.get("timeframe", config.get("project", {}).get("frequency", "15min")))
    source_config = _config_with_timeframe(config, source_timeframe)
    target_config = _config_with_timeframe(config, output_timeframe)
    source_path = aligned_panel_path(source_config, target)
    output_path = aligned_panel_path(target_config, target)

    panel = pd.read_parquet(source_path)
    resampled, report = resample_panel(panel, target, source_timeframe, output_timeframe, drop_incomplete_buckets=drop_incomplete_buckets)
    report = ResampleReport(
        target_symbol=report.target_symbol,
        source_timeframe=report.source_timeframe,
        target_timeframe=report.target_timeframe,
        universe_id=str(lab_cfg.get("universe_id", "core_cross_asset_v1")),
        factor=report.factor,
        source_rows=report.source_rows,
        output_rows=report.output_rows,
        source_sessions=report.source_sessions,
        output_sessions=report.output_sessions,
        dropped_buckets=report.dropped_buckets,
        source_path=str(source_path),
        output_path=str(output_path),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(output_path, index=False)
    report_path = report_output_path(config, target, source_timeframe, output_timeframe)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(report), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample an aligned OHLCV panel to a larger bar timeframe.")
    parser.add_argument("--config", default="configs/hmm_lab_15min.yaml")
    parser.add_argument("--source-timeframe", default="5min")
    parser.add_argument("--target-timeframe", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--keep-incomplete-buckets", action="store_true")
    args = parser.parse_args()

    output_path = run(
        args.config,
        source_timeframe=args.source_timeframe,
        target_timeframe=args.target_timeframe,
        target_symbol=args.target,
        drop_incomplete_buckets=not args.keep_incomplete_buckets,
    )
    print(f"Resampled aligned panel written to: {output_path}")


if __name__ == "__main__":
    main()

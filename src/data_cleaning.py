from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.market_calendar import add_execution_safety_columns, calendar_session_mask, get_market_schedule


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class DataQualityReport:
    input_path: Path
    output_path: Path
    rows_raw: int
    rows_clean: int
    duplicate_timestamps: int
    critical_nan_rows: int
    invalid_price_rows: int
    negative_volume_rows: int
    extreme_range_rows: int
    out_of_session_rows: int
    non_trading_session_rows: int
    non_trading_sessions: list[str]
    half_day_sessions: list[str]
    dropped_half_day_rows: int
    incomplete_sessions: list[str]
    dropped_incomplete_rows: int
    target_crosses_session_close_rows: int
    cannot_open_trade_rows: int
    force_flat_rows: int
    start_timestamp: str | None
    end_timestamp: str | None

    @property
    def rows_dropped(self) -> int:
        return self.rows_raw - self.rows_clean


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(f"Unsupported input format: {suffix}. Use .csv or .parquet.")


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]

    aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    normalized = normalized.rename(columns={k: v for k, v in aliases.items() if k in normalized.columns})

    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    return normalized.loc[:, list(REQUIRED_COLUMNS) + [c for c in normalized.columns if c not in REQUIRED_COLUMNS]]


def normalize_timestamp(series: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"Found {bad_count} unparsable timestamps")

    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")

    return parsed.dt.tz_convert(timezone)


def _session_mask(timestamps: pd.Series, market_open: str, market_close: str, timestamp_label: str) -> pd.Series:
    times = timestamps.dt.strftime("%H:%M")
    if timestamp_label == "end":
        return (times > market_open) & (times <= market_close)
    if timestamp_label == "start":
        return (times >= market_open) & (times < market_close)
    raise ValueError("session.timestamp_label must be either 'start' or 'end'")


def clean_ohlcv(df: pd.DataFrame, config: dict[str, Any], input_path: Path, output_path: Path) -> tuple[pd.DataFrame, DataQualityReport]:
    data_cfg = config["data"]
    session_cfg = config["session"]
    quality_cfg = config.get("quality", {})

    cleaned = standardize_ohlcv_columns(df)
    cleaned["timestamp"] = normalize_timestamp(cleaned[data_cfg.get("timestamp_col", "timestamp")], config["project"]["timezone"])

    for col in PRICE_COLUMNS + ("volume",):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    rows_raw = len(cleaned)
    duplicate_mask = cleaned["timestamp"].duplicated(keep="first")
    nan_mask = cleaned.loc[:, REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_price_mask = (
        (cleaned.loc[:, PRICE_COLUMNS] <= 0).any(axis=1)
        | (cleaned["high"] < cleaned[["open", "close", "low"]].max(axis=1))
        | (cleaned["low"] > cleaned[["open", "close", "high"]].min(axis=1))
    )
    negative_volume_mask = cleaned["volume"] < 0

    bar_range_bps = ((cleaned["high"] / cleaned["low"]) - 1.0) * 10_000.0
    extreme_range_mask = bar_range_bps > float(quality_cfg.get("max_bar_range_bps", 1000.0))

    removal_mask = pd.Series(False, index=cleaned.index)
    if quality_cfg.get("drop_duplicate_timestamps", True):
        removal_mask |= duplicate_mask
    if quality_cfg.get("drop_critical_nan_rows", True):
        removal_mask |= nan_mask
    if quality_cfg.get("drop_invalid_price_rows", True):
        removal_mask |= invalid_price_mask
    if quality_cfg.get("drop_negative_volume_rows", True):
        removal_mask |= negative_volume_mask
    if quality_cfg.get("drop_extreme_range_rows", True):
        removal_mask |= extreme_range_mask

    cleaned = cleaned.loc[~removal_mask].copy()
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    calendar_cfg = config.get("calendar", {})
    use_calendar = bool(calendar_cfg.get("enabled", False))
    schedule = None
    non_trading_session_rows = 0
    non_trading_sessions: list[str] = []
    half_day_sessions: list[str] = []
    dropped_half_day_rows = 0

    if use_calendar and not cleaned.empty:
        start_date = cleaned["timestamp"].min().date().isoformat()
        end_date = cleaned["timestamp"].max().date().isoformat()
        schedule = get_market_schedule(config, start_date, end_date)
        session_mask, has_schedule = calendar_session_mask(
            cleaned["timestamp"],
            schedule,
            session_cfg.get("timestamp_label", "start"),
        )
        non_trading_session_rows = int((~has_schedule).sum())
        non_trading_sessions = sorted(cleaned.loc[~has_schedule, "timestamp"].dt.strftime("%Y-%m-%d").unique().tolist())
    else:
        session_mask = _session_mask(
            cleaned["timestamp"],
            session_cfg["market_open"],
            session_cfg["market_close"],
            session_cfg.get("timestamp_label", "start"),
        )

    out_of_session_rows = int((~session_mask).sum())
    if session_cfg.get("regular_session_only", True):
        cleaned = cleaned.loc[session_mask].copy()

    cleaned["session"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d")

    if use_calendar and schedule is not None and not cleaned.empty:
        half_day_sessions = schedule.loc[schedule["is_half_day"], "session"].tolist()
        half_day_mask = cleaned["session"].isin(half_day_sessions)
        if calendar_cfg.get("drop_half_days", True):
            dropped_half_day_rows = int(half_day_mask.sum())
            cleaned = cleaned.loc[~half_day_mask].copy()

    cleaned["bar_index"] = cleaned.groupby("session", sort=False).cumcount()

    expected_bars = int(session_cfg["expected_bars_per_session"])
    session_counts = cleaned.groupby("session", sort=True).size()
    if use_calendar and schedule is not None and not cleaned.empty:
        expected_by_session = schedule["expected_bars"].to_dict()
        incomplete_sessions = [
            session
            for session, count in session_counts.items()
            if count != int(expected_by_session.get(session, expected_bars))
        ]
    else:
        incomplete_sessions = session_counts[session_counts != expected_bars].index.tolist()

    incomplete_mask = cleaned["session"].isin(incomplete_sessions)
    dropped_incomplete_rows = int(incomplete_mask.sum()) if session_cfg.get("drop_incomplete_sessions", True) else 0
    if dropped_incomplete_rows:
        cleaned = cleaned.loc[~incomplete_mask].copy()
        cleaned["bar_index"] = cleaned.groupby("session", sort=False).cumcount()

    cleaned = cleaned.reset_index(drop=True)
    if not cleaned.empty:
        cleaned = add_execution_safety_columns(cleaned, config)

    target_crosses_session_close_rows = int(cleaned.get("target_crosses_session_close", pd.Series(dtype=bool)).sum())
    cannot_open_trade_rows = int((~cleaned.get("can_open_trade", pd.Series(dtype=bool))).sum())
    force_flat_rows = int(cleaned.get("force_flat_bar", pd.Series(dtype=bool)).sum())

    report = DataQualityReport(
        input_path=input_path,
        output_path=output_path,
        rows_raw=rows_raw,
        rows_clean=len(cleaned),
        duplicate_timestamps=int(duplicate_mask.sum()),
        critical_nan_rows=int(nan_mask.sum()),
        invalid_price_rows=int(invalid_price_mask.sum()),
        negative_volume_rows=int(negative_volume_mask.sum()),
        extreme_range_rows=int(extreme_range_mask.sum()),
        out_of_session_rows=out_of_session_rows,
        non_trading_session_rows=non_trading_session_rows,
        non_trading_sessions=non_trading_sessions,
        half_day_sessions=half_day_sessions,
        dropped_half_day_rows=dropped_half_day_rows,
        incomplete_sessions=incomplete_sessions,
        dropped_incomplete_rows=dropped_incomplete_rows,
        target_crosses_session_close_rows=target_crosses_session_close_rows,
        cannot_open_trade_rows=cannot_open_trade_rows,
        force_flat_rows=force_flat_rows,
        start_timestamp=cleaned["timestamp"].min().isoformat() if not cleaned.empty else None,
        end_timestamp=cleaned["timestamp"].max().isoformat() if not cleaned.empty else None,
    )
    return cleaned, report


def render_quality_report(report: DataQualityReport) -> str:
    incomplete = "\n".join(f"- {session}" for session in report.incomplete_sessions) or "- Ninguna"
    holidays = "\n".join(f"- {session}" for session in report.non_trading_sessions) or "- Ninguna"
    half_days = "\n".join(f"- {session}" for session in report.half_day_sessions) or "- Ninguna"

    return f"""# Data Quality Report

## Inputs

- Input: `{report.input_path}`
- Output: `{report.output_path}`
- Start timestamp: `{report.start_timestamp}`
- End timestamp: `{report.end_timestamp}`

## Row Counts

| Metric | Value |
| --- | ---: |
| Raw rows | {report.rows_raw} |
| Clean rows | {report.rows_clean} |
| Dropped rows | {report.rows_dropped} |

## Checks

| Check | Count |
| --- | ---: |
| Duplicate timestamps | {report.duplicate_timestamps} |
| Critical NaN rows | {report.critical_nan_rows} |
| Invalid price rows | {report.invalid_price_rows} |
| Negative volume rows | {report.negative_volume_rows} |
| Extreme range rows | {report.extreme_range_rows} |
| Out-of-session rows | {report.out_of_session_rows} |
| Rows on non-trading sessions | {report.non_trading_session_rows} |
| Dropped half-day rows | {report.dropped_half_day_rows} |
| Dropped incomplete-session rows | {report.dropped_incomplete_rows} |
| Rows whose target would cross session close | {report.target_crosses_session_close_rows} |
| Rows where a new trade cannot be opened | {report.cannot_open_trade_rows} |
| Force-flat bars | {report.force_flat_rows} |

## Non-Trading Sessions

{holidays}

## Half-Day Sessions

{half_days}

## Incomplete Sessions

{incomplete}
"""


def run(config_path: str | Path) -> DataQualityReport:
    config = load_config(config_path)
    input_path = Path(config["data"]["input_file"])
    output_path = Path(config["data"]["cleaned_file"])
    report_path = Path(config["data"]["quality_report_file"])

    raw = load_ohlcv(input_path)
    cleaned, report = clean_ohlcv(raw, config, input_path, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    report_path.write_text(render_quality_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and validate SPY 5-minute OHLCV data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report = run(args.config)
    print(f"Clean rows: {report.rows_clean}")
    print(f"Output: {report.output_path}")
    print(f"Report rows dropped: {report.rows_dropped}")


if __name__ == "__main__":
    main()

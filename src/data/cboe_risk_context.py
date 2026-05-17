from __future__ import annotations

import argparse
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yaml


VOL_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
DAILY_OPTIONS_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{date}_daily_options"

RATIO_NAME_MAP = {
    "TOTAL PUT/CALL RATIO": "total_put_call_ratio",
    "INDEX PUT/CALL RATIO": "index_put_call_ratio",
    "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp_put_call_ratio",
    "EQUITY PUT/CALL RATIO": "equity_put_call_ratio",
    "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "vix_put_call_ratio",
    "SPX + SPXW PUT/CALL RATIO": "spx_spxw_put_call_ratio",
}

CATEGORY_NAME_MAP = {
    "SUM OF ALL PRODUCTS": "total_options",
    "INDEX OPTIONS": "index_options",
    "EXCHANGE TRADED PRODUCTS": "etp_options",
    "EQUITY OPTIONS": "equity_options",
    "CBOE VOLATILITY INDEX (VIX)": "vix_options",
}


@dataclass(frozen=True)
class CboeRiskContextOutputs:
    volatility_path: Path
    put_call_path: Path
    context_path: Path
    report_path: Path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def fetch_text(url: str, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": "ida-trading/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cboe request failed with HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cboe request failed: {exc.reason}") from exc


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    return json.loads(fetch_text(url, timeout=timeout))


def trading_sessions(start_date: str, end_date: str, calendar_name: str = "NYSE") -> list[pd.Timestamp]:
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    return [pd.Timestamp(value).normalize() for value in schedule.index]


def next_trading_session_map(dates: pd.Series, calendar_name: str = "NYSE") -> pd.Series:
    clean_dates = pd.to_datetime(dates).dt.normalize()
    if clean_dates.empty:
        return pd.Series(dtype="datetime64[ns]")
    start = (clean_dates.min() - pd.Timedelta(days=3)).date().isoformat()
    end = (clean_dates.max() + pd.Timedelta(days=10)).date().isoformat()
    sessions = pd.DatetimeIndex(trading_sessions(start, end, calendar_name=calendar_name))
    positions = sessions.searchsorted(clean_dates + pd.Timedelta(days=1), side="left")
    values = [sessions[pos] if pos < len(sessions) else pd.NaT for pos in positions]
    return pd.Series(values, index=dates.index)


def parse_volatility_index_csv(symbol: str, text: str) -> pd.DataFrame:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame(columns=["date", f"{symbol.lower()}_close"])
    frame = pd.DataFrame(rows)
    frame.columns = [str(col).strip().upper() for col in frame.columns]
    if "DATE" not in frame:
        raise ValueError(f"{symbol} CSV is missing DATE")
    output = pd.DataFrame({"date": pd.to_datetime(frame["DATE"], format="%m/%d/%Y").dt.normalize()})
    lower = symbol.lower()
    if "OPEN" in frame:
        output[f"{lower}_open"] = pd.to_numeric(frame["OPEN"], errors="coerce")
    if "HIGH" in frame:
        output[f"{lower}_high"] = pd.to_numeric(frame["HIGH"], errors="coerce")
    if "LOW" in frame:
        output[f"{lower}_low"] = pd.to_numeric(frame["LOW"], errors="coerce")
    close_column = "CLOSE" if "CLOSE" in frame else symbol.upper()
    if close_column not in frame:
        raise ValueError(f"{symbol} CSV is missing close column")
    output[f"{lower}_close"] = pd.to_numeric(frame[close_column], errors="coerce")
    return output.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


def download_volatility_indices(symbols: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for symbol in symbols:
        frame = parse_volatility_index_csv(symbol, fetch_text(VOL_INDEX_URL.format(symbol=symbol.upper())))
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["date"])
    return merged.sort_values("date", kind="stable").reset_index(drop=True)


def parse_daily_options_stats(date: str | pd.Timestamp, payload: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {"date": pd.Timestamp(date).normalize()}
    for item in payload.get("ratios", []):
        name = str(item.get("name", "")).strip()
        column = RATIO_NAME_MAP.get(name)
        if column:
            row[column] = pd.to_numeric(item.get("value"), errors="coerce")

    for raw_category, prefix in CATEGORY_NAME_MAP.items():
        for item in payload.get(raw_category, []):
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            if name not in {"volume", "open_interest"}:
                continue
            for side in ("call", "put", "total"):
                row[f"{prefix}_{name}_{side}"] = pd.to_numeric(item.get(side), errors="coerce")
    return pd.DataFrame([row])


def _download_one_daily_options(date: str) -> pd.DataFrame | None:
    try:
        payload = fetch_json(DAILY_OPTIONS_URL.format(date=date))
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise
    return parse_daily_options_stats(date, payload)


def download_put_call_daily(start_date: str, end_date: str, calendar_name: str = "NYSE", max_workers: int = 8) -> pd.DataFrame:
    dates = [session.date().isoformat() for session in trading_sessions(start_date, end_date, calendar_name=calendar_name)]
    rows: list[pd.DataFrame] = []
    if max_workers <= 1:
        for date in dates:
            result = _download_one_daily_options(date)
            if result is not None:
                rows.append(result)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_one_daily_options, date): date for date in dates}
            for future in as_completed(futures):
                date = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"failed to download Cboe daily options stats for {date}") from exc
                if result is not None:
                    rows.append(result)
    if not rows:
        return pd.DataFrame(columns=["date", *RATIO_NAME_MAP.values()])
    return pd.concat(rows, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 2)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def build_risk_context(
    volatility: pd.DataFrame,
    put_call: pd.DataFrame,
    *,
    calendar_name: str = "NYSE",
    zscore_window: int = 20,
) -> pd.DataFrame:
    context = volatility.merge(put_call, on="date", how="outer").sort_values("date", kind="stable").reset_index(drop=True)
    if context.empty:
        return context

    for symbol in ("vix", "vix9d", "vix3m", "vvix"):
        close = f"{symbol}_close"
        if close in context:
            context[f"{symbol}_change_1"] = context[close].diff()
            context[f"{symbol}_ret_1"] = context[close].pct_change()
            context[f"{symbol}_z{zscore_window}"] = _rolling_zscore(context[close], zscore_window)

    if {"vix9d_close", "vix_close"}.issubset(context.columns):
        context["vix9d_vix_ratio"] = context["vix9d_close"] / context["vix_close"]
    if {"vix_close", "vix3m_close"}.issubset(context.columns):
        context["vix_vix3m_ratio"] = context["vix_close"] / context["vix3m_close"]

    for column in RATIO_NAME_MAP.values():
        if column in context:
            context[f"{column}_z{zscore_window}"] = _rolling_zscore(context[column], zscore_window)

    context.insert(0, "source_date", context.pop("date"))
    context.insert(1, "available_session", next_trading_session_map(context["source_date"], calendar_name=calendar_name))
    feature_cols = [col for col in context.columns if col not in {"source_date", "available_session"}]
    return context.rename(columns={col: f"prev_{col}" for col in feature_cols})


def _write_report(path: Path, volatility: pd.DataFrame, put_call: pd.DataFrame, context: pd.DataFrame, config: dict[str, Any]) -> None:
    lines = [
        "# Cboe risk context data",
        "",
        f"- Volatility rows: `{len(volatility)}`",
        f"- Put/call rows: `{len(put_call)}`",
        f"- Context rows: `{len(context)}`",
        f"- Start date: `{config.get('start_date')}`",
        f"- End date: `{config.get('end_date')}`",
        "",
        "## Columns",
        "",
        "```text",
        *context.columns.tolist(),
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: dict[str, Any]) -> CboeRiskContextOutputs:
    paths = config.get("paths", {})
    calendar_name = str(config.get("calendar", "NYSE"))
    symbols = [str(symbol).upper() for symbol in config.get("volatility_symbols", ["VIX", "VIX9D", "VIX3M", "VVIX"])]
    start_date = str(config["start_date"])
    end_date = str(config["end_date"])

    volatility = download_volatility_indices(symbols)
    volatility = volatility[(volatility["date"] >= pd.Timestamp(start_date)) & (volatility["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
    put_call = download_put_call_daily(start_date, end_date, calendar_name=calendar_name, max_workers=int(config.get("max_workers", 8)))
    context = build_risk_context(
        volatility,
        put_call,
        calendar_name=calendar_name,
        zscore_window=int(config.get("zscore_window", 20)),
    )

    outputs = CboeRiskContextOutputs(
        volatility_path=Path(paths.get("volatility", "data/external/cboe/volatility_indices_daily.parquet")),
        put_call_path=Path(paths.get("put_call", "data/external/cboe/put_call_ratios_daily.parquet")),
        context_path=Path(paths.get("context", "data/external/cboe/risk_context_daily.parquet")),
        report_path=Path(paths.get("report", "reports/data_external/cboe_risk_context.md")),
    )
    for path, frame in (
        (outputs.volatility_path, volatility),
        (outputs.put_call_path, put_call),
        (outputs.context_path, context),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    _write_report(outputs.report_path, volatility, put_call, context, config)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and process Cboe risk context data")
    parser.add_argument("--config", default="configs/data/cboe_risk_context.yaml")
    args = parser.parse_args(argv)
    outputs = run(load_config(args.config))
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

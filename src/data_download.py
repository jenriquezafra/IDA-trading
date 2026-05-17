from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path
from datetime import date, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd
import yaml
import yfinance as yf


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    if len(df.columns.names) >= 2 and "Price" in [str(name) for name in df.columns.names]:
        return df.droplevel([level for level in range(df.columns.nlevels) if level != 0], axis=1)

    return df.droplevel(-1, axis=1)


def download_yfinance_ohlcv(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    symbol = data_cfg.get("symbol", "SPY")
    period = data_cfg.get("download_period", "60d")
    interval = data_cfg.get("download_interval", "5m")

    raw = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned by yfinance for {symbol} period={period} interval={interval}")

    raw = _flatten_yfinance_columns(raw)
    raw = raw.reset_index()
    raw.columns = [str(col).strip().lower().replace(" ", "_") for col in raw.columns]

    timestamp_col = "datetime" if "datetime" in raw.columns else "date"
    raw = raw.rename(
        columns={
            timestamp_col: "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing required columns: {missing}")

    return raw.loc[:, required].sort_values("timestamp").reset_index(drop=True)


def _parse_minute_interval(interval: str) -> int:
    normalized = str(interval).strip().lower()
    if normalized.endswith("min"):
        normalized = normalized.removesuffix("min")
    elif normalized.endswith("m"):
        normalized = normalized.removesuffix("m")
    else:
        raise ValueError(f"Unsupported interval: {interval}. Use minute intervals like 1m or 5m.")

    minutes = int(normalized)
    if minutes <= 0:
        raise ValueError(f"Interval must be positive: {interval}")
    return minutes


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year - years)


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError(f"Missing required date field: {field_name}")
    return date.fromisoformat(str(value))


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    if start > end:
        raise ValueError(f"start_date must be <= end_date, got {start} > {end}")

    chunks: list[tuple[date, date]] = []
    current = start
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        chunk_end = min(month_end, end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _fetch_polygon_json(url: str, params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    query = urlencode(params)
    separator = "&" if urlparse(url).query else "?"
    request_url = f"{url}{separator}{query}" if query else url
    request = Request(request_url, headers={"User-Agent": "ida-trading/1.0"})

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Polygon request failed with HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Polygon request failed: {exc.reason}") from exc


def _append_api_key_to_next_url(next_url: str, api_key: str) -> tuple[str, dict[str, str]]:
    parsed = urlparse(next_url)
    url_without_query = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    if "apiKey=" in parsed.query:
        return url_without_query, {}
    return url_without_query, {"apiKey": api_key}


def _polygon_results_to_ohlcv(results: list[dict[str, Any]], timezone: str) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(results)
    required = {"t", "o", "h", "l", "c", "v"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Polygon response is missing aggregate fields: {missing}")

    output = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(timezone),
            "open": frame["o"],
            "high": frame["h"],
            "low": frame["l"],
            "close": frame["c"],
            "volume": frame["v"],
        }
    )
    return output.sort_values("timestamp").reset_index(drop=True)


def _resample_ohlcv_minutes(df: pd.DataFrame, output_interval: str) -> pd.DataFrame:
    output_minutes = _parse_minute_interval(output_interval)
    if df.empty or output_minutes == 1:
        return df

    indexed = df.sort_values("timestamp").set_index("timestamp")
    resampled = indexed.resample(
        f"{output_minutes}min",
        label="left",
        closed="left",
        origin="start_day",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled.reset_index()


def download_polygon_ohlcv(config: dict[str, Any], start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    data_cfg = config["data"]
    polygon_cfg = data_cfg.get("polygon", {})
    symbol = data_cfg.get("symbol", "SPY")
    timezone = config["project"].get("timezone", "America/New_York")
    output_interval = data_cfg.get("download_interval", "5m")
    source_interval = polygon_cfg.get("source_interval", "1m")
    adjusted = bool(polygon_cfg.get("adjusted", True))
    api_key_env = polygon_cfg.get("api_key_env", "POLYGON_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        raise RuntimeError(f"Missing Polygon API key. Set {api_key_env} in .env or in the shell environment.")

    end = _parse_date(end_date or data_cfg.get("end_date") or (date.today() - timedelta(days=1)), "end_date")
    start_default = _subtract_years(end, int(polygon_cfg.get("default_years", 5)))
    start = _parse_date(start_date or data_cfg.get("start_date") or start_default, "start_date")

    source_minutes = _parse_minute_interval(source_interval)
    _parse_minute_interval(output_interval)

    base_url = "https://api.polygon.io/v2/aggs/ticker"
    all_results: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        url = f"{base_url}/{symbol}/range/{source_minutes}/minute/{chunk_start.isoformat()}/{chunk_end.isoformat()}"
        params: dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50_000,
            "apiKey": api_key,
        }

        while url:
            payload = _fetch_polygon_json(url, params)
            status = str(payload.get("status", "")).upper()
            if status in {"ERROR", "NOT_AUTHORIZED"}:
                message = payload.get("error") or payload.get("message") or payload
                raise RuntimeError(f"Polygon returned {status}: {message}")

            all_results.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            if next_url:
                url, params = _append_api_key_to_next_url(next_url, api_key)
            else:
                url = ""

    source = _polygon_results_to_ohlcv(all_results, timezone)
    source = source.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    raw_source_file = polygon_cfg.get("raw_source_file")
    if raw_source_file:
        raw_source_path = Path(raw_source_file)
        raw_source_path.parent.mkdir(parents=True, exist_ok=True)
        source.to_parquet(raw_source_path, index=False)

    return _resample_ohlcv_minutes(source, output_interval)


def run(config_path: str | Path, start_date: str | None = None, end_date: str | None = None) -> Path:
    load_dotenv()
    config = load_config(config_path)
    provider = config["data"].get("provider", "yfinance")

    output_path = Path(config["data"]["input_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "yfinance":
        data = download_yfinance_ohlcv(config)
    elif provider == "polygon":
        data = download_polygon_ohlcv(config, start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Unsupported data provider: {provider}")

    data.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw SPY 5-minute OHLCV data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--start-date", help="Override configured historical start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Override configured historical end date, YYYY-MM-DD.")
    args = parser.parse_args()

    output_path = run(args.config, start_date=args.start_date, end_date=args.end_date)
    print(f"Downloaded raw data to: {output_path}")


if __name__ == "__main__":
    main()

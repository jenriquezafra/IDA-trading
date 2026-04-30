from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    provider = config["data"].get("provider", "yfinance")
    if provider != "yfinance":
        raise ValueError(f"Unsupported data provider: {provider}")

    output_path = Path(config["data"]["input_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = download_yfinance_ohlcv(config)
    data.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw SPY 5-minute OHLCV data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    output_path = run(args.config)
    print(f"Downloaded raw data to: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _grouped_shift(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return df.groupby("session", sort=False)[column].shift(periods)


def _grouped_rolling_sum(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _grouped_rolling_mean(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    safe_denominator = denominator.where(denominator.abs() > eps)
    return numerator / safe_denominator


def add_return_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    featured = df.copy()
    log_close = np.log(featured["close"])
    for window in windows:
        featured[f"ret_{window}"] = log_close - log_close.groupby(featured["session"], sort=False).shift(window)
    return featured


def add_realized_volatility_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    featured = df.copy()
    squared_ret = featured["ret_1"].pow(2)
    for window in windows:
        featured[f"rv_{window}"] = np.sqrt(_grouped_rolling_sum(squared_ret, featured["session"], window))
    return featured


def add_range_and_atr_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    featured = df.copy()
    featured["range"] = np.log(featured["high"] / featured["low"])

    prev_close = _grouped_shift(featured, "close", 1)
    high_low = featured["high"] - featured["low"]
    high_prev_close = (featured["high"] - prev_close).abs()
    low_prev_close = (featured["low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    featured["true_range"] = true_range / featured["close"]

    for window in windows:
        featured[f"atr_{window}"] = _grouped_rolling_mean(featured["true_range"], featured["session"], window)

    return featured.drop(columns=["true_range"])


def add_sma_and_trend_features(df: pd.DataFrame, sma_windows: list[int], trend_windows: list[int]) -> pd.DataFrame:
    featured = df.copy()
    for window in sma_windows:
        featured[f"sma_{window}"] = _grouped_rolling_mean(featured["close"], featured["session"], window)

    for window in trend_windows:
        sma_col = f"sma_{window}"
        if sma_col not in featured:
            featured[sma_col] = _grouped_rolling_mean(featured["close"], featured["session"], window)
        featured[f"trend_{window}"] = np.log(featured["close"] / featured[sma_col])

    return featured


def add_intraday_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    typical_price = (featured["high"] + featured["low"] + featured["close"]) / 3.0
    dollar_volume = typical_price * featured["volume"]
    grouped = featured.groupby("session", sort=False)
    cumulative_dollar_volume = dollar_volume.groupby(featured["session"], sort=False).cumsum()
    cumulative_volume = grouped["volume"].cumsum()

    featured["vwap"] = cumulative_dollar_volume / cumulative_volume.replace(0, np.nan)
    featured["dist_vwap"] = np.log(featured["close"] / featured["vwap"])
    return featured


def add_volatility_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if {"rv_3", "rv_12"}.issubset(featured.columns):
        featured["vol_ratio_3_12"] = _safe_ratio(featured["rv_3"], featured["rv_12"])
    if {"rv_6", "rv_24"}.issubset(featured.columns):
        featured["vol_ratio_6_24"] = _safe_ratio(featured["rv_6"], featured["rv_24"])
    return featured


def add_trend_efficiency_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    featured = df.copy()
    signed_ret = np.sign(featured["ret_1"])
    abs_ret = featured["ret_1"].abs()
    for window in windows:
        sum_ret = _grouped_rolling_sum(featured["ret_1"], featured["session"], int(window))
        sum_abs_ret = _grouped_rolling_sum(abs_ret, featured["session"], int(window))
        featured[f"signed_efficiency_{window}"] = _safe_ratio(sum_ret, sum_abs_ret)
        featured[f"dir_persistence_{window}"] = _grouped_rolling_mean(signed_ret, featured["session"], int(window))
    return featured


def add_intraday_location_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    featured = df.copy()
    atr_col = config.get("features", {}).get("location_atr_column", "atr_12")
    if atr_col not in featured.columns:
        raise ValueError(f"Missing ATR column for location features: {atr_col}")

    grouped = featured.groupby("session", sort=False)
    session_open = grouped["open"].transform("first")
    high_so_far = grouped["high"].cummax()
    low_so_far = grouped["low"].cummin()
    session_range = high_so_far - low_so_far
    atr = featured[atr_col]

    featured["dist_open"] = np.log(featured["close"] / session_open)
    featured["pos_session_range"] = _safe_ratio(featured["close"] - low_so_far, session_range)
    featured["dist_session_high_atr"] = _safe_ratio(np.log(featured["close"] / high_so_far), atr)
    featured["dist_session_low_atr"] = _safe_ratio(np.log(featured["close"] / low_so_far), atr)
    featured["intraday_runup"] = _safe_ratio(np.log(featured["close"] / low_so_far), atr)

    if "vwap" in featured.columns:
        featured["dist_vwap_atr"] = _safe_ratio(np.log(featured["close"] / featured["vwap"]), atr)
        vwap_window = int(config.get("features", {}).get("vwap_slope_window", 12))
        previous_vwap = featured.groupby("session", sort=False)["vwap"].shift(vwap_window)
        featured[f"vwap_slope_{vwap_window}"] = _safe_ratio(np.log(featured["vwap"] / previous_vwap), atr)
    return featured


def add_compression_expansion_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if "range" in featured.columns:
        range_mean_6 = _grouped_rolling_mean(featured["range"], featured["session"], 6)
        range_mean_24 = _grouped_rolling_mean(featured["range"], featured["session"], 24)
        featured["range_ratio_6_24"] = _safe_ratio(range_mean_6, range_mean_24)
    return featured


def add_intraday_drawdown_feature(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    intraday_peak = featured.groupby("session", sort=False)["close"].cummax()
    featured["intraday_drawdown"] = np.log(featured["close"] / intraday_peak)
    return featured


def add_relative_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    session_order = featured[["session"]].drop_duplicates().reset_index(drop=True)
    session_order["session_number"] = np.arange(len(session_order))
    featured = featured.merge(session_order, on="session", how="left")
    featured = featured.sort_values(["bar_index", "session_number"]).copy()

    prior_mean = (
        featured.groupby("bar_index", sort=False)["volume"]
        .expanding(min_periods=1)
        .mean()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    featured["rel_volume"] = featured["volume"] / prior_mean.replace(0, np.nan)
    featured = featured.sort_values(["session_number", "bar_index"]).drop(columns=["session_number"]).reset_index(drop=True)
    return featured


def add_time_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    featured = df.copy()
    feature_cfg = config.get("features", {})
    open_window_bars = int(feature_cfg.get("open_window_bars", 6))
    close_window_bars = int(feature_cfg.get("close_window_bars", 6))
    midday_start = feature_cfg.get("midday_start", "12:00")
    midday_end = feature_cfg.get("midday_end", "14:00")

    denominator = (featured["bars_in_session"] - 1).replace(0, np.nan)
    phase = 2.0 * np.pi * featured["bar_index"] / denominator
    featured["sin_time"] = np.sin(phase)
    featured["cos_time"] = np.cos(phase)

    frequency = pd.Timedelta(config["project"].get("frequency", "5min"))
    featured["minutes_to_close"] = (featured["bars_in_session"] - 1 - featured["bar_index"]) * (frequency / pd.Timedelta(minutes=1))
    featured["open_window"] = featured["bar_index"] < open_window_bars
    featured["close_window"] = featured["bar_index"] >= (featured["bars_in_session"] - close_window_bars)

    clock_time = featured["timestamp"].dt.strftime("%H:%M")
    featured["midday"] = (clock_time >= midday_start) & (clock_time < midday_end)
    return featured


def build_features(cleaned: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "session", "bar_index", "bars_in_session"}
    missing = sorted(required - set(cleaned.columns))
    if missing:
        raise ValueError(f"Cleaned data is missing required columns: {missing}")

    feature_cfg = config.get("features", {})
    featured = cleaned.sort_values(["session", "bar_index"]).reset_index(drop=True).copy()
    featured = add_return_features(featured, feature_cfg.get("return_windows", [1, 2, 3, 6, 12]))
    featured = add_realized_volatility_features(featured, feature_cfg.get("realized_vol_windows", [3, 6, 12, 24]))
    featured = add_range_and_atr_features(featured, feature_cfg.get("atr_windows", [6, 12]))
    featured = add_volatility_ratio_features(featured)
    featured = add_sma_and_trend_features(
        featured,
        feature_cfg.get("sma_windows", [6, 12, 24]),
        feature_cfg.get("trend_windows", [6, 12, 24]),
    )
    featured = add_trend_efficiency_features(featured, feature_cfg.get("efficiency_windows", [12]))
    featured = add_intraday_vwap_features(featured)
    featured = add_intraday_location_features(featured, config)
    featured = add_compression_expansion_features(featured)
    featured = add_intraday_drawdown_feature(featured)
    featured = add_relative_volume_feature(featured)
    featured = add_time_features(featured, config)
    return featured


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    input_path = Path(config["data"]["cleaned_file"])
    output_path = Path(config["data"]["features_file"])

    cleaned = pd.read_parquet(input_path)
    features = build_features(cleaned, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal base features from cleaned OHLCV data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    output_path = run(args.config)
    print(f"Features written to: {output_path}")


if __name__ == "__main__":
    main()

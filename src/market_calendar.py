from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_market_calendars as mcal


def _bar_frequency(config: dict[str, Any]) -> pd.Timedelta:
    return pd.Timedelta(config["project"].get("frequency", "5min"))


def _expected_bars(open_ts: pd.Timestamp, close_ts: pd.Timestamp, frequency: pd.Timedelta, timestamp_label: str) -> int:
    duration = close_ts - open_ts
    bars = int(duration / frequency)
    if timestamp_label == "end":
        return bars
    if timestamp_label == "start":
        return bars
    raise ValueError("session.timestamp_label must be either 'start' or 'end'")


def get_market_schedule(config: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    calendar_cfg = config.get("calendar", {})
    calendar_name = calendar_cfg.get("name", "NYSE")
    timezone = config["project"]["timezone"]
    timestamp_label = config["session"].get("timestamp_label", "start")
    full_day_bars = int(config["session"]["expected_bars_per_session"])

    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    if schedule.empty:
        return pd.DataFrame(
            columns=["session", "market_open", "market_close", "expected_bars", "is_half_day"]
        ).set_index("session", drop=False)

    local = schedule.copy()
    local["market_open"] = local["market_open"].dt.tz_convert(timezone)
    local["market_close"] = local["market_close"].dt.tz_convert(timezone)
    local["session"] = local.index.strftime("%Y-%m-%d")

    frequency = _bar_frequency(config)
    local["expected_bars"] = [
        _expected_bars(open_ts, close_ts, frequency, timestamp_label)
        for open_ts, close_ts in zip(local["market_open"], local["market_close"], strict=True)
    ]
    local["is_half_day"] = local["expected_bars"] < full_day_bars

    return local.set_index("session", drop=False)


def calendar_session_mask(timestamps: pd.Series, schedule: pd.DataFrame, timestamp_label: str) -> tuple[pd.Series, pd.Series]:
    sessions = timestamps.dt.strftime("%Y-%m-%d")
    schedule_lookup = schedule.reset_index(drop=True)
    lookup = pd.DataFrame({"timestamp": timestamps, "session": sessions})
    lookup = lookup.merge(
        schedule_lookup[["session", "market_open", "market_close"]],
        on="session",
        how="left",
    )

    has_schedule = lookup["market_open"].notna()
    if timestamp_label == "end":
        in_session = has_schedule & (lookup["timestamp"] > lookup["market_open"]) & (lookup["timestamp"] <= lookup["market_close"])
    elif timestamp_label == "start":
        in_session = has_schedule & (lookup["timestamp"] >= lookup["market_open"]) & (lookup["timestamp"] < lookup["market_close"])
    else:
        raise ValueError("session.timestamp_label must be either 'start' or 'end'")

    return pd.Series(in_session.to_numpy(), index=timestamps.index), pd.Series(has_schedule.to_numpy(), index=timestamps.index)


def add_execution_safety_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    annotated = df.copy()
    horizon = int(config.get("labeling", {}).get("horizon_bars", 2))
    no_new_trades_after = config.get("backtest", {}).get("no_new_trades_after", "15:45")
    force_flat_before = config.get("backtest", {}).get("force_flat_before", "15:55")

    annotated["bars_in_session"] = annotated.groupby("session", sort=False)["timestamp"].transform("size")
    annotated["next_open_timestamp"] = annotated.groupby("session", sort=False)["timestamp"].shift(-1)
    annotated["target_crosses_session_close"] = annotated["bar_index"] + horizon + 1 >= annotated["bars_in_session"]

    next_open_time = annotated["next_open_timestamp"].dt.strftime("%H:%M")
    annotated["can_open_trade"] = annotated["next_open_timestamp"].notna() & (next_open_time <= no_new_trades_after)

    bar_time = annotated["timestamp"].dt.strftime("%H:%M")
    annotated["force_flat_bar"] = bar_time >= force_flat_before
    annotated["trade_could_remain_open_past_close"] = annotated["target_crosses_session_close"]

    return annotated

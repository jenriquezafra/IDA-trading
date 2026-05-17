from __future__ import annotations

"""Build H3 earnings event datasets from audited raw vendor payloads."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal
import yaml


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_earnings_continuation_h3_v1.yaml")
DEFAULT_OUTPUT_PATH = Path("data/events/earnings/h3_v1/earnings_events.parquet")
TIMING_PARTITION_FILENAMES = {
    "pre_market": "earnings_events_pre_market.parquet",
    "after_market": "earnings_events_after_market.parquet",
    "excluded_timing": "earnings_events_excluded_timing.parquet",
}
EVENT_SCHEMA = {
    "event_id": "object",
    "symbol": "object",
    "vendor_event_id": "object",
    "report_timestamp_utc": "datetime64[ns, UTC]",
    "report_timestamp_et": "object",
    "report_timing": "object",
    "event_session": "object",
    "entry_timestamp": "datetime64[ns, UTC]",
    "exit_timestamp": "datetime64[ns, UTC]",
    "eps_actual": "float64",
    "eps_consensus": "float64",
    "consensus_snapshot_at_utc": "datetime64[ns, UTC]",
    "consensus_source": "object",
    "consensus_raw_hash": "object",
    "consensus_revision_policy": "object",
    "consensus_snapshot_is_pre_event": "bool",
    "eps_surprise": "float64",
    "eps_surprise_pct": "float64",
    "eps_surprise_z": "float64",
    "revenue_actual": "float64",
    "revenue_consensus": "float64",
    "revenue_surprise": "float64",
    "revenue_surprise_pct": "float64",
    "revenue_surprise_z": "float64",
    "missing_eps_actual": "bool",
    "missing_eps_consensus": "bool",
    "missing_revenue_actual": "bool",
    "missing_revenue_consensus": "bool",
    "eps_consensus_abs_below_floor": "bool",
    "revenue_consensus_abs_below_floor": "bool",
    "gap_open": "float64",
    "gap_prev_close": "float64",
    "gap_return": "float64",
    "recent_atr_return": "float64",
    "gap_atr": "float64",
    "missing_gap_open": "bool",
    "missing_gap_prev_close": "bool",
    "missing_recent_atr": "bool",
    "volume_30m": "float64",
    "expected_volume_30m": "float64",
    "opening_30m_bar_count": "float64",
    "rel_volume_30m": "float64",
    "missing_volume_30m": "bool",
    "missing_expected_volume_30m": "bool",
    "insufficient_opening_30m_bars": "bool",
    "vwap_30m": "float64",
    "range_high_30m": "float64",
    "range_low_30m": "float64",
    "close_30m": "float64",
    "missing_vwap_30m": "bool",
    "missing_range_30m": "bool",
    "missing_close_30m": "bool",
    "sector_id": "object",
    "sector_proxy": "object",
    "sector_return_30m": "float64",
    "peer_proxy_symbol": "object",
    "peer_proxy_fallback_used": "bool",
    "peer_proxy_return_30m": "float64",
    "missing_sector_mapping": "bool",
    "missing_sector_proxy_return_30m": "bool",
    "missing_peer_proxy_return_30m": "bool",
    "macro_day_flag": "bool",
    "simultaneous_peer_earnings_flag": "bool",
    "spread_bps_30m": "float64",
    "high_spread_flag": "bool",
    "binary_news_flag": "bool",
    "is_full_regular_session": "bool",
    "halt_flag": "bool",
    "suspected_halt_or_bad_session": "bool",
    "split_flag": "bool",
    "split_factor": "float64",
    "corporate_action_flag": "bool",
    "exclusion_flags": "object",
    "is_tradeable_v1": "bool",
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"H3 config must be a mapping: {path}")
    return raw


def load_raw_payload(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("results", raw.get("data", raw.get("rows", [])))
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("raw earnings payload must be a list or a mapping with results/data/rows")
    return [dict(row) for row in rows if isinstance(row, dict)]


def empty_events_frame() -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column, dtype in EVENT_SCHEMA.items():
        data[column] = pd.Series(dtype=dtype)
    return pd.DataFrame(data)


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_or_na(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _feature_config(config: dict[str, Any]) -> dict[str, Any]:
    features_cfg = dict(config.get("features", {}))
    fundamental_cfg = dict(features_cfg.get("fundamental", {}))
    return dict(fundamental_cfg.get("surprise_zscore", {}))


def _opening_reaction_config(config: dict[str, Any]) -> dict[str, Any]:
    features_cfg = dict(config.get("features", {}))
    opening_cfg = dict(features_cfg.get("opening_reaction", {}))
    return dict(opening_cfg.get("gap_atr", {}))


def _rel_volume_config(config: dict[str, Any]) -> dict[str, Any]:
    features_cfg = dict(config.get("features", {}))
    opening_cfg = dict(features_cfg.get("opening_reaction", {}))
    return dict(opening_cfg.get("rel_volume_30m", {}))


def _timeframe_minutes(value: Any) -> int:
    text = str(value or "5min").strip().lower()
    return max(int(pd.Timedelta(text).total_seconds() // 60), 1)


def _relative_surprise_pct(surprise: pd.Series, consensus: pd.Series, denominator_floor: float) -> pd.Series:
    surprise_values = pd.to_numeric(surprise, errors="coerce")
    consensus_values = pd.to_numeric(consensus, errors="coerce")
    denominator = consensus_values.abs()
    out = pd.Series(float("nan"), index=surprise.index, dtype="float64")
    valid = surprise_values.notna() & consensus_values.notna() & denominator.ge(float(denominator_floor))
    out.loc[valid] = surprise_values.loc[valid] / denominator.loc[valid]
    return out


def _prior_session_zscore(values: pd.Series, event_sessions: pd.Series, min_history: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sessions = event_sessions.astype("object")
    out = pd.Series(float("nan"), index=values.index, dtype="float64")
    history: list[float] = []

    for session in sorted(s for s in sessions.dropna().unique()):
        mask = sessions.eq(session)
        if len(history) >= min_history:
            history_series = pd.Series(history, dtype="float64")
            std = float(history_series.std(ddof=0))
            if std > 0:
                out.loc[mask] = (values.loc[mask] - float(history_series.mean())) / std
        history.extend(values.loc[mask].dropna().astype("float64").tolist())

    return out


def add_surprise_features(events: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    z_cfg = _feature_config(cfg)
    min_history = int(z_cfg.get("min_history", 20))
    eps_floor = float(z_cfg.get("eps_consensus_abs_floor", 0.01))
    revenue_floor = float(z_cfg.get("revenue_consensus_abs_floor", 1.0))

    enriched = events.copy()
    enriched["missing_eps_actual"] = enriched["eps_actual"].isna()
    enriched["missing_eps_consensus"] = enriched["eps_consensus"].isna()
    enriched["missing_revenue_actual"] = enriched["revenue_actual"].isna()
    enriched["missing_revenue_consensus"] = enriched["revenue_consensus"].isna()
    enriched["eps_consensus_abs_below_floor"] = enriched["eps_consensus"].notna() & enriched["eps_consensus"].abs().lt(eps_floor)
    enriched["revenue_consensus_abs_below_floor"] = enriched["revenue_consensus"].notna() & enriched["revenue_consensus"].abs().lt(revenue_floor)
    enriched["eps_surprise_pct"] = _relative_surprise_pct(enriched["eps_surprise"], enriched["eps_consensus"], eps_floor)
    enriched["revenue_surprise_pct"] = _relative_surprise_pct(enriched["revenue_surprise"], enriched["revenue_consensus"], revenue_floor)
    enriched["eps_surprise_z"] = _prior_session_zscore(enriched["eps_surprise_pct"], enriched["event_session"], min_history)
    enriched["revenue_surprise_z"] = _prior_session_zscore(enriched["revenue_surprise_pct"], enriched["event_session"], min_history)
    return enriched


def _append_exclusion_flags(existing_flags: Any, new_flags: list[str]) -> str:
    flags = [flag for flag in str(existing_flags or "").split(";") if flag]
    for flag in new_flags:
        if flag and flag not in flags:
            flags.append(flag)
    return ";".join(flags)


def _read_optional_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(table_path)
    if table_path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(table_path, lines=table_path.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported table format for H3 exclusion input: {path}")


def _normalise_session_values(frame: pd.DataFrame, candidates: tuple[str, ...] = ("event_session", "session", "date")) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column].astype(str)
    return pd.Series(dtype="object")


def _normalise_symbol_values(frame: pd.DataFrame) -> pd.Series:
    if "symbol" not in frame.columns:
        return pd.Series(dtype="object")
    return frame["symbol"].astype(str).str.upper().str.strip()


def _truthy_rows(frame: pd.DataFrame, flag_candidates: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in flag_candidates:
        if column in frame.columns:
            value = frame[column]
            if value.dtype == bool:
                return value.fillna(False)
            return value.fillna(False).astype(bool)
    return mask


def _panel_to_long_ohlcv(panel: pd.DataFrame) -> pd.DataFrame:
    required_long = {"symbol", "open", "high", "low", "close"}
    if required_long.issubset(panel.columns):
        keep = [
            column
            for column in ["timestamp", "session", "bar_index", "symbol", "open", "high", "low", "close", "volume", "bar_vwap", "vwap", "vw"]
            if column in panel.columns
        ]
        out = panel.loc[:, keep].copy()
        if "bar_vwap" not in out.columns:
            if "vwap" in out.columns:
                out["bar_vwap"] = out["vwap"]
            elif "vw" in out.columns:
                out["bar_vwap"] = out["vw"]
        return out.drop(columns=[column for column in ["vwap", "vw"] if column in out.columns])

    symbols: set[str] = set()
    for column in panel.columns:
        if "__" not in column:
            continue
        symbol, field = column.rsplit("__", 1)
        if field in {"open", "high", "low", "close", "volume", "bar_vwap", "vwap", "vw"}:
            symbols.add(symbol)

    frames: list[pd.DataFrame] = []
    for symbol in sorted(symbols):
        columns = {field: f"{symbol}__{field}" for field in ["open", "high", "low", "close"]}
        if not set(columns.values()).issubset(panel.columns):
            continue
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "open": panel[columns["open"]],
                "high": panel[columns["high"]],
                "low": panel[columns["low"]],
                "close": panel[columns["close"]],
            }
        )
        volume_column = f"{symbol}__volume"
        if volume_column in panel.columns:
            frame["volume"] = panel[volume_column]
        for vwap_field in ["bar_vwap", "vwap", "vw"]:
            vwap_column = f"{symbol}__{vwap_field}"
            if vwap_column in panel.columns:
                frame["bar_vwap"] = panel[vwap_column]
                break
        for column in ["timestamp", "session", "bar_index"]:
            if column in panel.columns:
                frame[column] = panel[column]
        frames.append(frame)

    if not frames:
        raise ValueError("OHLCV panel must be long with symbol/open/high/low/close or wide with SYMBOL__open/high/low/close")
    return pd.concat(frames, ignore_index=True)


def _normalise_ohlcv_panel(panel: pd.DataFrame, timezone: str) -> pd.DataFrame:
    long_panel = _panel_to_long_ohlcv(panel)
    if "timestamp" in long_panel.columns:
        long_panel["timestamp"] = pd.to_datetime(long_panel["timestamp"], utc=True)
    if "session" not in long_panel.columns:
        if "timestamp" not in long_panel.columns:
            raise ValueError("OHLCV panel must include session or timestamp")
        long_panel["session"] = long_panel["timestamp"].dt.tz_convert(timezone).dt.date.astype(str)
    long_panel["session"] = long_panel["session"].astype(str)
    long_panel["symbol"] = long_panel["symbol"].astype(str).str.upper().str.strip()
    for column in ["open", "high", "low", "close", "volume", "bar_index", "bar_vwap"]:
        if column not in long_panel.columns:
            continue
        long_panel[column] = pd.to_numeric(long_panel[column], errors="coerce")
    sort_columns = [column for column in ["symbol", "session", "timestamp"] if column in long_panel.columns]
    return long_panel.sort_values(sort_columns).reset_index(drop=True)


def _daily_bars_from_ohlcv_panel(panel: pd.DataFrame, timezone: str) -> pd.DataFrame:
    normalised = _normalise_ohlcv_panel(panel, timezone)
    daily = (
        normalised.groupby(["symbol", "session"], sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )
    return daily.sort_values(["symbol", "session"]).reset_index(drop=True)


def _daily_bars_with_atr(daily: pd.DataFrame, lookback_sessions: int, min_history_sessions: int) -> pd.DataFrame:
    out = daily.copy()
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)
    high_low = out["high"] - out["low"]
    high_prev_close = (out["high"] - out["prev_close"]).abs()
    low_prev_close = (out["low"] - out["prev_close"]).abs()
    out["true_range"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    reference_close = out["prev_close"].where(out["prev_close"].notna(), out["close"])
    out["atr_return"] = out["true_range"] / reference_close
    out["recent_atr_return"] = (
        out.groupby("symbol")["atr_return"]
        .transform(lambda series: series.rolling(window=lookback_sessions, min_periods=min_history_sessions).mean().shift(1))
    )
    return out


def add_gap_features(
    events: pd.DataFrame,
    intraday_panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = dict(cfg.get("data", {}))
    timezone = str(data_cfg.get("timestamp_timezone", "America/New_York"))
    gap_cfg = _opening_reaction_config(cfg)
    lookback_sessions = int(gap_cfg.get("atr_lookback_sessions", 20))
    min_history_sessions = int(gap_cfg.get("atr_min_history_sessions", lookback_sessions))

    daily = _daily_bars_with_atr(_daily_bars_from_ohlcv_panel(intraday_panel, timezone), lookback_sessions, min_history_sessions)
    keyed = daily.set_index(["symbol", "session"])
    enriched = events.copy()

    for column, default in {
        "gap_open": float("nan"),
        "gap_prev_close": float("nan"),
        "gap_return": float("nan"),
        "recent_atr_return": float("nan"),
        "gap_atr": float("nan"),
        "missing_gap_open": False,
        "missing_gap_prev_close": False,
        "missing_recent_atr": False,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default

    for index, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        session = str(row.get("event_session") or "").strip()
        flags: list[str] = []
        if not symbol or not session or (symbol, session) not in keyed.index:
            flags.extend(["missing_gap_open", "missing_gap_prev_close", "missing_recent_atr"])
            enriched.at[index, "missing_gap_open"] = True
            enriched.at[index, "missing_gap_prev_close"] = True
            enriched.at[index, "missing_recent_atr"] = True
        else:
            daily_row = keyed.loc[(symbol, session)]
            gap_open = float(daily_row["open"]) if pd.notna(daily_row["open"]) else float("nan")
            prev_close = float(daily_row["prev_close"]) if pd.notna(daily_row["prev_close"]) else float("nan")
            recent_atr = float(daily_row["recent_atr_return"]) if pd.notna(daily_row["recent_atr_return"]) else float("nan")
            enriched.at[index, "gap_open"] = gap_open
            enriched.at[index, "gap_prev_close"] = prev_close
            enriched.at[index, "recent_atr_return"] = recent_atr

            if pd.isna(gap_open):
                flags.append("missing_gap_open")
                enriched.at[index, "missing_gap_open"] = True
            if pd.isna(prev_close) or prev_close <= 0:
                flags.append("missing_gap_prev_close")
                enriched.at[index, "missing_gap_prev_close"] = True
            if pd.isna(recent_atr) or recent_atr <= 0:
                flags.append("missing_recent_atr")
                enriched.at[index, "missing_recent_atr"] = True

            if not flags:
                gap_return = gap_open / prev_close - 1.0
                enriched.at[index, "gap_return"] = gap_return
                enriched.at[index, "gap_atr"] = gap_return / recent_atr

        if flags:
            enriched.at[index, "exclusion_flags"] = _append_exclusion_flags(row.get("exclusion_flags"), flags)
            enriched.at[index, "is_tradeable_v1"] = False

    return enriched


def _opening_window_mask(
    panel: pd.DataFrame,
    timezone: str,
    regular_open: str,
    opening_window_minutes: int,
    timeframe_minutes: int,
) -> pd.Series:
    if "timestamp" in panel.columns:
        local_clock = panel["timestamp"].dt.tz_convert(timezone).dt.strftime("%H:%M")
        window_end = (pd.Timestamp(f"2000-01-01 {regular_open}") + pd.Timedelta(minutes=opening_window_minutes)).strftime("%H:%M")
        return local_clock.ge(regular_open) & local_clock.lt(window_end)
    if "bar_index" in panel.columns:
        expected_bars = max(int(opening_window_minutes // timeframe_minutes), 1)
        return panel["bar_index"].ge(0) & panel["bar_index"].lt(expected_bars)
    return pd.Series(False, index=panel.index)


def _opening_volumes_from_ohlcv_panel(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = dict(config.get("data", {}))
    session_policy = dict(data_cfg.get("timezone_session_policy", {}))
    events_cfg = dict(config.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    timezone = str(data_cfg.get("timestamp_timezone", "America/New_York"))
    regular_open = str(session_policy.get("regular_open", "09:30"))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    timeframe_minutes = _timeframe_minutes(config.get("timeframe", "5min"))
    expected_bars = max(int(opening_window_minutes // timeframe_minutes), 1)
    rel_cfg = _rel_volume_config(config)
    lookback_sessions = int(rel_cfg.get("lookback_sessions", 60))
    min_history_sessions = int(rel_cfg.get("min_history_sessions", lookback_sessions))

    normalised = _normalise_ohlcv_panel(panel, timezone)
    if "volume" not in normalised.columns:
        return pd.DataFrame(columns=["symbol", "session", "volume_30m", "opening_30m_bar_count", "expected_volume_30m"])

    opening = normalised.loc[
        _opening_window_mask(normalised, timezone, regular_open, opening_window_minutes, timeframe_minutes)
    ].copy()
    if opening.empty:
        return pd.DataFrame(columns=["symbol", "session", "volume_30m", "opening_30m_bar_count", "expected_volume_30m"])

    volumes = (
        opening.groupby(["symbol", "session"], sort=True)
        .agg(volume_30m=("volume", "sum"), opening_30m_bar_count=("volume", "count"))
        .reset_index()
        .sort_values(["symbol", "session"])
        .reset_index(drop=True)
    )
    volumes["opening_30m_expected_bar_count"] = float(expected_bars)
    volumes["expected_volume_30m"] = (
        volumes.groupby("symbol")["volume_30m"]
        .transform(lambda series: series.rolling(window=lookback_sessions, min_periods=min_history_sessions).median().shift(1))
    )
    return volumes


def add_volume_features(
    events: pd.DataFrame,
    intraday_panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    events_cfg = dict(cfg.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    timeframe_minutes = _timeframe_minutes(cfg.get("timeframe", "5min"))
    expected_bars = max(int(opening_window_minutes // timeframe_minutes), 1)

    volumes = _opening_volumes_from_ohlcv_panel(intraday_panel, cfg)
    keyed = volumes.set_index(["symbol", "session"]) if not volumes.empty else None
    enriched = events.copy()

    for column, default in {
        "volume_30m": float("nan"),
        "expected_volume_30m": float("nan"),
        "opening_30m_bar_count": float("nan"),
        "rel_volume_30m": float("nan"),
        "missing_volume_30m": False,
        "missing_expected_volume_30m": False,
        "insufficient_opening_30m_bars": False,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default

    for index, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        session = str(row.get("event_session") or "").strip()
        flags: list[str] = []
        if keyed is None or not symbol or not session or (symbol, session) not in keyed.index:
            flags.extend(["missing_volume_30m", "missing_expected_volume_30m", "insufficient_opening_30m_bars"])
            enriched.at[index, "missing_volume_30m"] = True
            enriched.at[index, "missing_expected_volume_30m"] = True
            enriched.at[index, "insufficient_opening_30m_bars"] = True
        else:
            volume_row = keyed.loc[(symbol, session)]
            volume_30m = float(volume_row["volume_30m"]) if pd.notna(volume_row["volume_30m"]) else float("nan")
            expected_volume = float(volume_row["expected_volume_30m"]) if pd.notna(volume_row["expected_volume_30m"]) else float("nan")
            bar_count = float(volume_row["opening_30m_bar_count"]) if pd.notna(volume_row["opening_30m_bar_count"]) else float("nan")
            enriched.at[index, "volume_30m"] = volume_30m
            enriched.at[index, "expected_volume_30m"] = expected_volume
            enriched.at[index, "opening_30m_bar_count"] = bar_count

            if pd.isna(volume_30m) or volume_30m <= 0:
                flags.append("missing_volume_30m")
                enriched.at[index, "missing_volume_30m"] = True
            if pd.isna(expected_volume) or expected_volume <= 0:
                flags.append("missing_expected_volume_30m")
                enriched.at[index, "missing_expected_volume_30m"] = True
            if pd.isna(bar_count) or bar_count < expected_bars:
                flags.append("insufficient_opening_30m_bars")
                enriched.at[index, "insufficient_opening_30m_bars"] = True

            if not flags:
                enriched.at[index, "rel_volume_30m"] = volume_30m / expected_volume

        if flags:
            enriched.at[index, "exclusion_flags"] = _append_exclusion_flags(row.get("exclusion_flags"), flags)
            enriched.at[index, "is_tradeable_v1"] = False

    return enriched


def _opening_range_from_ohlcv_panel(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = dict(config.get("data", {}))
    session_policy = dict(data_cfg.get("timezone_session_policy", {}))
    events_cfg = dict(config.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    timezone = str(data_cfg.get("timestamp_timezone", "America/New_York"))
    regular_open = str(session_policy.get("regular_open", "09:30"))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    timeframe_minutes = _timeframe_minutes(config.get("timeframe", "5min"))

    normalised = _normalise_ohlcv_panel(panel, timezone)
    opening = normalised.loc[
        _opening_window_mask(normalised, timezone, regular_open, opening_window_minutes, timeframe_minutes)
    ].copy()
    if opening.empty:
        return pd.DataFrame(
            columns=["symbol", "session", "vwap_30m", "range_high_30m", "range_low_30m", "close_30m", "opening_30m_bar_count"]
        )

    if "bar_vwap" not in opening.columns:
        opening["bar_vwap"] = opening["close"]
    if "volume" not in opening.columns:
        opening["volume"] = float("nan")
    opening["vwap_numerator"] = opening["bar_vwap"] * opening["volume"]

    stats = (
        opening.groupby(["symbol", "session"], sort=True)
        .agg(
            vwap_numerator=("vwap_numerator", "sum"),
            volume_30m=("volume", "sum"),
            range_high_30m=("high", "max"),
            range_low_30m=("low", "min"),
            close_30m=("close", "last"),
            opening_30m_bar_count=("close", "count"),
        )
        .reset_index()
        .sort_values(["symbol", "session"])
        .reset_index(drop=True)
    )
    valid_vwap = stats["volume_30m"].notna() & stats["volume_30m"].gt(0)
    stats["vwap_30m"] = float("nan")
    stats.loc[valid_vwap, "vwap_30m"] = stats.loc[valid_vwap, "vwap_numerator"] / stats.loc[valid_vwap, "volume_30m"]
    return stats.loc[:, ["symbol", "session", "vwap_30m", "range_high_30m", "range_low_30m", "close_30m", "opening_30m_bar_count"]]


def add_opening_range_features(
    events: pd.DataFrame,
    intraday_panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    events_cfg = dict(cfg.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    timeframe_minutes = _timeframe_minutes(cfg.get("timeframe", "5min"))
    expected_bars = max(int(opening_window_minutes // timeframe_minutes), 1)

    ranges = _opening_range_from_ohlcv_panel(intraday_panel, cfg)
    keyed = ranges.set_index(["symbol", "session"]) if not ranges.empty else None
    enriched = events.copy()

    for column, default in {
        "vwap_30m": float("nan"),
        "range_high_30m": float("nan"),
        "range_low_30m": float("nan"),
        "close_30m": float("nan"),
        "opening_30m_bar_count": float("nan"),
        "missing_vwap_30m": False,
        "missing_range_30m": False,
        "missing_close_30m": False,
        "insufficient_opening_30m_bars": False,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default

    for index, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        session = str(row.get("event_session") or "").strip()
        flags: list[str] = []
        if keyed is None or not symbol or not session or (symbol, session) not in keyed.index:
            flags.extend(["missing_vwap_30m", "missing_range_30m", "missing_close_30m", "insufficient_opening_30m_bars"])
            enriched.at[index, "missing_vwap_30m"] = True
            enriched.at[index, "missing_range_30m"] = True
            enriched.at[index, "missing_close_30m"] = True
            enriched.at[index, "insufficient_opening_30m_bars"] = True
        else:
            range_row = keyed.loc[(symbol, session)]
            vwap = float(range_row["vwap_30m"]) if pd.notna(range_row["vwap_30m"]) else float("nan")
            range_high = float(range_row["range_high_30m"]) if pd.notna(range_row["range_high_30m"]) else float("nan")
            range_low = float(range_row["range_low_30m"]) if pd.notna(range_row["range_low_30m"]) else float("nan")
            close_30m = float(range_row["close_30m"]) if pd.notna(range_row["close_30m"]) else float("nan")
            bar_count = float(range_row["opening_30m_bar_count"]) if pd.notna(range_row["opening_30m_bar_count"]) else float("nan")
            enriched.at[index, "vwap_30m"] = vwap
            enriched.at[index, "range_high_30m"] = range_high
            enriched.at[index, "range_low_30m"] = range_low
            enriched.at[index, "close_30m"] = close_30m
            enriched.at[index, "opening_30m_bar_count"] = bar_count

            if pd.isna(vwap):
                flags.append("missing_vwap_30m")
                enriched.at[index, "missing_vwap_30m"] = True
            if pd.isna(range_high) or pd.isna(range_low):
                flags.append("missing_range_30m")
                enriched.at[index, "missing_range_30m"] = True
            if pd.isna(close_30m):
                flags.append("missing_close_30m")
                enriched.at[index, "missing_close_30m"] = True
            if pd.isna(bar_count) or bar_count < expected_bars:
                flags.append("insufficient_opening_30m_bars")
                enriched.at[index, "insufficient_opening_30m_bars"] = True

        if flags:
            enriched.at[index, "exclusion_flags"] = _append_exclusion_flags(row.get("exclusion_flags"), flags)
            enriched.at[index, "is_tradeable_v1"] = False

    return enriched


def load_sector_map(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    records = raw.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"sector map records must be a list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        symbol = str(record.get("symbol") or "").upper().strip()
        if symbol:
            out[symbol] = dict(record)
    return out


def load_peer_proxy_map(path: str | Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    proxies = raw.get("sector_proxies", {})
    if not isinstance(proxies, dict):
        raise ValueError(f"peer proxy sector_proxies must be a mapping: {path}")
    return {str(sector): dict(value or {}) for sector, value in proxies.items()}


def _opening_returns_from_ohlcv_panel(
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = dict(config.get("data", {}))
    session_policy = dict(data_cfg.get("timezone_session_policy", {}))
    events_cfg = dict(config.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    timezone = str(data_cfg.get("timestamp_timezone", "America/New_York"))
    regular_open = str(session_policy.get("regular_open", "09:30"))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    timeframe_minutes = _timeframe_minutes(config.get("timeframe", "5min"))

    normalised = _normalise_ohlcv_panel(panel, timezone)
    opening = normalised.loc[
        _opening_window_mask(normalised, timezone, regular_open, opening_window_minutes, timeframe_minutes)
    ].copy()
    if opening.empty:
        return pd.DataFrame(columns=["symbol", "session", "open_30m", "close_30m", "proxy_return_30m", "opening_30m_bar_count"])

    returns = (
        opening.groupby(["symbol", "session"], sort=True)
        .agg(open_30m=("open", "first"), close_30m=("close", "last"), opening_30m_bar_count=("close", "count"))
        .reset_index()
        .sort_values(["symbol", "session"])
        .reset_index(drop=True)
    )
    valid = returns["open_30m"].notna() & returns["open_30m"].gt(0) & returns["close_30m"].notna()
    returns["proxy_return_30m"] = float("nan")
    returns.loc[valid, "proxy_return_30m"] = returns.loc[valid, "close_30m"] / returns.loc[valid, "open_30m"] - 1.0
    return returns


def add_sector_peer_features(
    events: pd.DataFrame,
    intraday_panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
    sector_map_path: str | Path | None = None,
    peer_proxy_path: str | Path | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = dict(cfg.get("data", {}))
    resolved_sector_map_path = sector_map_path or data_cfg.get("sector_mapping_path") or "configs/strategy/equity_earnings_continuation_h3_sector_map_v1.yaml"
    resolved_peer_proxy_path = peer_proxy_path or data_cfg.get("peer_proxy_path") or "configs/strategy/equity_earnings_continuation_h3_peer_proxy_v1.yaml"
    sector_map = load_sector_map(resolved_sector_map_path) if resolved_sector_map_path else {}
    peer_proxy_map = load_peer_proxy_map(resolved_peer_proxy_path) if resolved_peer_proxy_path else {}
    returns = _opening_returns_from_ohlcv_panel(intraday_panel, cfg)
    keyed_returns = returns.set_index(["symbol", "session"]) if not returns.empty else None

    enriched = events.copy()
    for column, default in {
        "sector_id": None,
        "sector_proxy": None,
        "sector_return_30m": float("nan"),
        "peer_proxy_symbol": None,
        "peer_proxy_fallback_used": False,
        "peer_proxy_return_30m": float("nan"),
        "missing_sector_mapping": False,
        "missing_sector_proxy_return_30m": False,
        "missing_peer_proxy_return_30m": False,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default

    for index, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        session = str(row.get("event_session") or "").strip()
        mapping = sector_map.get(symbol)
        flags: list[str] = []
        sector_id = None
        primary_proxy = None
        fallback_proxy = "SPY"

        if mapping:
            sector_id = str(mapping.get("sector_id") or "").strip() or None
            primary_proxy = str(mapping.get("sector_proxy") or "").upper().strip() or None
            peer_cfg = peer_proxy_map.get(str(sector_id), {}) if sector_id else {}
            primary_proxy = str(peer_cfg.get("primary_proxy") or primary_proxy or "").upper().strip() or None
            fallback_proxy = str(peer_cfg.get("fallback_proxy") or "SPY").upper().strip()
        else:
            flags.append("missing_sector_mapping")
            enriched.at[index, "missing_sector_mapping"] = True

        enriched.at[index, "sector_id"] = sector_id
        enriched.at[index, "sector_proxy"] = primary_proxy

        selected_proxy = None
        selected_return = float("nan")
        fallback_used = False
        candidates = [candidate for candidate in [primary_proxy, fallback_proxy] if candidate]
        for candidate_position, candidate in enumerate(dict.fromkeys(candidates)):
            if keyed_returns is None or not session or (candidate, session) not in keyed_returns.index:
                continue
            candidate_row = keyed_returns.loc[(candidate, session)]
            candidate_return = float(candidate_row["proxy_return_30m"]) if pd.notna(candidate_row["proxy_return_30m"]) else float("nan")
            if pd.notna(candidate_return):
                selected_proxy = candidate
                selected_return = candidate_return
                fallback_used = candidate_position > 0 or candidate != primary_proxy
                break

        if selected_proxy is None:
            flags.extend(["missing_sector_proxy_return_30m", "missing_peer_proxy_return_30m"])
            enriched.at[index, "missing_sector_proxy_return_30m"] = True
            enriched.at[index, "missing_peer_proxy_return_30m"] = True
        else:
            enriched.at[index, "peer_proxy_symbol"] = selected_proxy
            enriched.at[index, "peer_proxy_fallback_used"] = fallback_used
            enriched.at[index, "sector_return_30m"] = selected_return
            enriched.at[index, "peer_proxy_return_30m"] = selected_return

        if flags:
            enriched.at[index, "exclusion_flags"] = _append_exclusion_flags(row.get("exclusion_flags"), flags)
            enriched.at[index, "is_tradeable_v1"] = False

    return enriched


def _events_with_sector_labels(events: pd.DataFrame, sector_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    labelled = events.copy()
    if "sector_id" not in labelled.columns:
        labelled["sector_id"] = None
    if "sector_proxy" not in labelled.columns:
        labelled["sector_proxy"] = None
    for index, row in labelled.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        mapping = sector_map.get(symbol)
        if not mapping:
            continue
        if not row.get("sector_id"):
            labelled.at[index, "sector_id"] = mapping.get("sector_id")
        if not row.get("sector_proxy"):
            labelled.at[index, "sector_proxy"] = mapping.get("sector_proxy")
    return labelled


def _macro_sessions(macro_calendar: pd.DataFrame | None) -> set[str]:
    if macro_calendar is None or macro_calendar.empty:
        return set()
    sessions = _normalise_session_values(macro_calendar)
    mask = _truthy_rows(macro_calendar, ("macro_day_flag", "macro_dominant_session", "exclude"))
    return set(sessions.loc[mask].dropna().astype(str))


def _symbol_session_pairs(frame: pd.DataFrame | None, flag_candidates: tuple[str, ...]) -> set[tuple[str, str]]:
    if frame is None or frame.empty:
        return set()
    symbols = _normalise_symbol_values(frame)
    sessions = _normalise_session_values(frame)
    if symbols.empty or sessions.empty:
        return set()
    mask = _truthy_rows(frame, flag_candidates)
    return set(zip(symbols.loc[mask], sessions.loc[mask].astype(str), strict=False))


def _spread_by_symbol_session(quote_quality: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    if quote_quality is None or quote_quality.empty:
        return {}
    symbols = _normalise_symbol_values(quote_quality)
    sessions = _normalise_session_values(quote_quality)
    spread_col = next(
        (
            column
            for column in ("spread_bps_30m", "quoted_spread_bps", "median_spread_bps", "p90_spread_bps", "spread_bps")
            if column in quote_quality.columns
        ),
        None,
    )
    if symbols.empty or sessions.empty or spread_col is None:
        return {}
    spreads = pd.to_numeric(quote_quality[spread_col], errors="coerce")
    out: dict[tuple[str, str], float] = {}
    for symbol, session, spread in zip(symbols, sessions.astype(str), spreads, strict=False):
        if pd.notna(spread):
            out[(symbol, session)] = float(spread)
    return out


def add_exclusion_features(
    events: pd.DataFrame,
    config: dict[str, Any] | None = None,
    macro_calendar: pd.DataFrame | None = None,
    halt_events: pd.DataFrame | None = None,
    quote_quality: pd.DataFrame | None = None,
    binary_news: pd.DataFrame | None = None,
    sector_map_path: str | Path | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = dict(cfg.get("data", {}))
    costs_cfg = dict(cfg.get("costs", {}))
    exclusion_cfg = dict(cfg.get("exclusion_feature_policy", {}))
    spread_threshold_bps = float(
        exclusion_cfg.get(
            "high_spread_threshold_bps",
            dict(costs_cfg.get("round_trip_bps", {})).get("conservative", 10.0),
        )
    )
    resolved_sector_map_path = sector_map_path or data_cfg.get("sector_mapping_path") or "configs/strategy/equity_earnings_continuation_h3_sector_map_v1.yaml"
    sector_map = load_sector_map(resolved_sector_map_path) if resolved_sector_map_path else {}
    enriched = _events_with_sector_labels(events, sector_map)

    for column, default in {
        "macro_day_flag": False,
        "simultaneous_peer_earnings_flag": False,
        "spread_bps_30m": float("nan"),
        "high_spread_flag": False,
        "binary_news_flag": False,
        "halt_flag": False,
        "suspected_halt_or_bad_session": False,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default

    macro_session_set = _macro_sessions(macro_calendar)
    halt_pairs = _symbol_session_pairs(halt_events, ("halt_flag", "trading_halt", "exclude"))
    binary_news_pairs = _symbol_session_pairs(binary_news, ("binary_news_flag", "binary_news_event", "exclude"))
    spread_by_pair = _spread_by_symbol_session(quote_quality)

    peer_counts = (
        enriched.dropna(subset=["event_session", "sector_id"])
        .assign(symbol_norm=lambda frame: frame["symbol"].astype(str).str.upper().str.strip())
        .groupby(["event_session", "sector_id"])["symbol_norm"]
        .nunique()
    )

    for index, row in enriched.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        session = str(row.get("event_session") or "").strip()
        sector_id = row.get("sector_id")
        flags: list[str] = []

        if session in macro_session_set:
            enriched.at[index, "macro_day_flag"] = True
            flags.append("macro_dominant_session")

        if pd.notna(sector_id) and peer_counts.get((session, sector_id), 0) > 1:
            enriched.at[index, "simultaneous_peer_earnings_flag"] = True
            flags.append("simultaneous_core_peer_earnings")

        if (symbol, session) in halt_pairs:
            enriched.at[index, "halt_flag"] = True
            enriched.at[index, "suspected_halt_or_bad_session"] = True
            flags.append("trading_halt")

        spread = spread_by_pair.get((symbol, session))
        if spread is not None:
            enriched.at[index, "spread_bps_30m"] = spread
            if spread > spread_threshold_bps:
                enriched.at[index, "high_spread_flag"] = True
                flags.append("high_spread")

        if (symbol, session) in binary_news_pairs:
            enriched.at[index, "binary_news_flag"] = True
            flags.append("binary_news_event")

        if flags:
            enriched.at[index, "exclusion_flags"] = _append_exclusion_flags(row.get("exclusion_flags"), flags)
            enriched.at[index, "is_tradeable_v1"] = False

    return enriched


def _parse_vendor_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, ""):
        return pd.NaT
    try:
        return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")
    except (TypeError, ValueError):
        return pd.NaT


def _parse_report_timestamp(row: dict[str, Any], timezone: str) -> pd.Timestamp | pd.NaT:
    date_value = row.get("date")
    time_value = row.get("time")
    if date_value in (None, "") or time_value in (None, ""):
        return pd.NaT
    time_text = str(time_value).strip().lower()
    if time_text in {"bmo", "amc", "pre-market", "premarket", "after-market", "aftermarket"}:
        return pd.NaT
    try:
        return pd.Timestamp(f"{date_value} {time_value}", tz=timezone)
    except (TypeError, ValueError):
        return pd.NaT


def _classify_report_timing(report_timestamp_et: pd.Timestamp | pd.NaT, row: dict[str, Any]) -> str:
    if pd.isna(report_timestamp_et):
        text = str(row.get("time", "")).strip().lower()
        if text in {"bmo", "pre-market", "premarket"}:
            return "pre_market"
        if text in {"amc", "after-market", "aftermarket"}:
            return "after_market"
        return "unknown_time"
    clock = report_timestamp_et.time()
    if clock < pd.Timestamp("09:30").time():
        return "pre_market"
    if clock >= pd.Timestamp("16:00").time():
        return "after_market"
    return "during_session"


def _market_calendar(calendar_name: str) -> mcal.MarketCalendar:
    calendar = mcal.get_calendar("NYSE" if calendar_name.upper() == "XNYS" else calendar_name)
    return calendar


def _trading_schedule(start: str, end: str, calendar_name: str) -> pd.DataFrame:
    calendar = _market_calendar(calendar_name)
    schedule = calendar.schedule(start_date=start, end_date=end)
    return schedule


def _trading_sessions(start: str, end: str, calendar_name: str) -> pd.DatetimeIndex:
    schedule = _trading_schedule(start, end, calendar_name)
    return pd.DatetimeIndex(schedule.index).tz_localize(None)


def _event_session(report_date: str, report_timing: str, calendar_name: str) -> str | None:
    report_day = pd.Timestamp(report_date).normalize()
    sessions = _trading_sessions(
        (report_day - pd.Timedelta(days=7)).date().isoformat(),
        (report_day + pd.Timedelta(days=14)).date().isoformat(),
        calendar_name,
    )
    if report_timing == "pre_market":
        candidates = sessions[sessions >= report_day]
    elif report_timing == "after_market":
        candidates = sessions[sessions > report_day]
    else:
        return None
    if candidates.empty:
        return None
    return candidates[0].date().isoformat()


def _session_schedule_row(session: str | None, calendar_name: str) -> pd.Series | None:
    if not session:
        return None
    schedule = _trading_schedule(session, session, calendar_name)
    if schedule.empty:
        return None
    return schedule.iloc[0]


def _local_clock(timestamp: pd.Timestamp | pd.NaT, timezone: str) -> str | None:
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).tz_convert(timezone).strftime("%H:%M")


def _entry_exit_from_calendar(
    session: str | None,
    calendar_name: str,
    timezone: str,
    opening_window_minutes: int,
    regular_open: str,
    regular_close: str,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, bool]:
    row = _session_schedule_row(session, calendar_name)
    if row is None:
        return pd.NaT, pd.NaT, False

    market_open = pd.Timestamp(row["market_open"]).tz_convert("UTC")
    market_close = pd.Timestamp(row["market_close"]).tz_convert("UTC")
    entry_ts = market_open + pd.Timedelta(minutes=opening_window_minutes)
    is_full_regular_session = bool(
        _local_clock(market_open, timezone) == regular_open
        and _local_clock(market_close, timezone) == regular_close
    )
    return entry_ts, market_close, is_full_regular_session


def _event_id(symbol: str, vendor_event_id: str, report_date: str, report_time: Any) -> str:
    base = f"H3|{symbol}|{vendor_event_id}|{report_date}|{report_time}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


def normalize_benzinga_earnings(
    rows: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = dict(cfg.get("data", {}))
    events_cfg = dict(cfg.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    session_policy = dict(data_cfg.get("timezone_session_policy", {}))
    timezone = str(data_cfg.get("timestamp_timezone", "America/New_York"))
    calendar_name = str(data_cfg.get("calendar", "XNYS"))
    opening_window_minutes = int(entry_cfg.get("opening_window_minutes", 30))
    expected_entry_time = str(entry_cfg.get("entry_time", session_policy.get("entry_time", "10:00")))
    latest_allowed_entry_time = str(entry_cfg.get("latest_allowed_entry_time", session_policy.get("latest_allowed_entry_time", "10:05")))
    regular_open = str(session_policy.get("regular_open", "09:30"))
    regular_close = str(session_policy.get("regular_close", "16:00"))
    surprise_z_cfg = _feature_config(cfg)
    eps_consensus_abs_floor = float(surprise_z_cfg.get("eps_consensus_abs_floor", 0.01))
    revenue_consensus_abs_floor = float(surprise_z_cfg.get("revenue_consensus_abs_floor", 1.0))
    revision_rule = str(data_cfg.get("consensus_revision_policy", {}).get("rule", "hard_reject_without_pre_event_snapshot"))

    records: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(_first_present(row, ("ticker", "symbol")) or "").upper().strip()
        report_date = str(row.get("date") or "").strip()
        vendor_event_id = str(_first_present(row, ("benzinga_id", "id", "event_id")) or "").strip()
        raw_hash = _raw_hash(row)
        report_ts_et = _parse_report_timestamp(row, timezone)
        report_timing = _classify_report_timing(report_ts_et, row)
        report_ts_utc = report_ts_et.tz_convert("UTC") if not pd.isna(report_ts_et) else pd.NaT
        session = _event_session(report_date, report_timing, calendar_name) if report_date else None
        entry_ts, exit_ts, is_full_regular_session = _entry_exit_from_calendar(
            session,
            calendar_name,
            timezone,
            opening_window_minutes,
            regular_open,
            regular_close,
        )
        entry_clock = _local_clock(entry_ts, timezone)
        consensus_ts = _parse_vendor_timestamp(_first_present(row, ("last_updated", "updated")))
        consensus_is_pre_event = bool(not pd.isna(consensus_ts) and not pd.isna(report_ts_utc) and consensus_ts <= report_ts_utc)

        eps_actual = _float_or_na(_first_present(row, ("actual_eps", "eps")))
        eps_consensus = _float_or_na(_first_present(row, ("estimated_eps", "eps_est")))
        revenue_actual = _float_or_na(_first_present(row, ("actual_revenue", "revenue")))
        revenue_consensus = _float_or_na(_first_present(row, ("estimated_revenue", "revenue_est")))
        eps_surprise = eps_actual - eps_consensus if pd.notna(eps_actual) and pd.notna(eps_consensus) else float("nan")
        revenue_surprise = revenue_actual - revenue_consensus if pd.notna(revenue_actual) and pd.notna(revenue_consensus) else float("nan")
        missing_eps_actual = pd.isna(eps_actual)
        missing_eps_consensus = pd.isna(eps_consensus)
        missing_revenue_actual = pd.isna(revenue_actual)
        missing_revenue_consensus = pd.isna(revenue_consensus)
        eps_consensus_abs_below_floor = bool(pd.notna(eps_consensus) and abs(eps_consensus) < eps_consensus_abs_floor)
        revenue_consensus_abs_below_floor = bool(pd.notna(revenue_consensus) and abs(revenue_consensus) < revenue_consensus_abs_floor)

        flags: list[str] = []
        if not symbol:
            flags.append("missing_symbol")
        if report_timing not in {"pre_market", "after_market"}:
            flags.append(f"excluded_report_timing:{report_timing}")
        if pd.isna(report_ts_utc):
            flags.append("missing_report_timestamp")
        if missing_eps_consensus:
            flags.append("missing_pre_event_eps_consensus_snapshot")
        if missing_revenue_consensus:
            flags.append("missing_pre_event_revenue_consensus_snapshot")
        if eps_consensus_abs_below_floor:
            flags.append("eps_consensus_abs_below_floor")
        if revenue_consensus_abs_below_floor:
            flags.append("revenue_consensus_abs_below_floor")
        if not consensus_is_pre_event:
            flags.append("post_event_revised_consensus_without_snapshot")
        if missing_eps_actual:
            flags.append("missing_actual_eps")
        if missing_revenue_actual:
            flags.append("missing_actual_revenue")
        if session is None:
            flags.append("missing_event_session")
        if not is_full_regular_session:
            flags.append("non_full_regular_session")
        if pd.isna(entry_ts):
            flags.append("missing_regular_entry_timestamp")
        elif entry_clock != expected_entry_time:
            flags.append(f"unexpected_regular_entry_time:{entry_clock}")
        elif entry_clock > latest_allowed_entry_time:
            flags.append(f"entry_after_latest_allowed_time:{entry_clock}")
        if pd.isna(exit_ts):
            flags.append("missing_regular_exit_timestamp")

        records.append(
            {
                "event_id": _event_id(symbol, vendor_event_id, report_date, row.get("time")),
                "symbol": symbol,
                "vendor_event_id": vendor_event_id,
                "report_timestamp_utc": report_ts_utc,
                "report_timestamp_et": None if pd.isna(report_ts_et) else report_ts_et.isoformat(),
                "report_timing": report_timing,
                "event_session": session,
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "eps_actual": eps_actual,
                "eps_consensus": eps_consensus,
                "consensus_snapshot_at_utc": consensus_ts,
                "consensus_source": "polygon_benzinga_earnings",
                "consensus_raw_hash": raw_hash,
                "consensus_revision_policy": revision_rule,
                "consensus_snapshot_is_pre_event": consensus_is_pre_event,
                "eps_surprise": eps_surprise,
                "eps_surprise_pct": float("nan"),
                "eps_surprise_z": float("nan"),
                "revenue_actual": revenue_actual,
                "revenue_consensus": revenue_consensus,
                "revenue_surprise": revenue_surprise,
                "revenue_surprise_pct": float("nan"),
                "revenue_surprise_z": float("nan"),
                "missing_eps_actual": missing_eps_actual,
                "missing_eps_consensus": missing_eps_consensus,
                "missing_revenue_actual": missing_revenue_actual,
                "missing_revenue_consensus": missing_revenue_consensus,
                "eps_consensus_abs_below_floor": eps_consensus_abs_below_floor,
                "revenue_consensus_abs_below_floor": revenue_consensus_abs_below_floor,
                "gap_open": float("nan"),
                "gap_prev_close": float("nan"),
                "gap_return": float("nan"),
                "recent_atr_return": float("nan"),
                "gap_atr": float("nan"),
                "missing_gap_open": False,
                "missing_gap_prev_close": False,
                "missing_recent_atr": False,
                "volume_30m": float("nan"),
                "expected_volume_30m": float("nan"),
                "opening_30m_bar_count": float("nan"),
                "rel_volume_30m": float("nan"),
                "missing_volume_30m": False,
                "missing_expected_volume_30m": False,
                "insufficient_opening_30m_bars": False,
                "vwap_30m": float("nan"),
                "range_high_30m": float("nan"),
                "range_low_30m": float("nan"),
                "close_30m": float("nan"),
                "missing_vwap_30m": False,
                "missing_range_30m": False,
                "missing_close_30m": False,
                "sector_id": None,
                "sector_proxy": None,
                "sector_return_30m": float("nan"),
                "peer_proxy_symbol": None,
                "peer_proxy_fallback_used": False,
                "peer_proxy_return_30m": float("nan"),
                "missing_sector_mapping": False,
                "missing_sector_proxy_return_30m": False,
                "missing_peer_proxy_return_30m": False,
                "macro_day_flag": False,
                "simultaneous_peer_earnings_flag": False,
                "spread_bps_30m": float("nan"),
                "high_spread_flag": False,
                "binary_news_flag": False,
                "is_full_regular_session": is_full_regular_session,
                "halt_flag": False,
                "suspected_halt_or_bad_session": False,
                "split_flag": False,
                "split_factor": float("nan"),
                "corporate_action_flag": False,
                "exclusion_flags": ";".join(flags),
                "is_tradeable_v1": not flags,
            }
        )

    frame = pd.DataFrame(records) if records else empty_events_frame()
    frame = add_surprise_features(frame, cfg)
    for column in EVENT_SCHEMA:
        if column not in frame:
            frame[column] = empty_events_frame()[column]
    frame = frame.loc[:, list(EVENT_SCHEMA)]
    for column in ("report_timestamp_utc", "entry_timestamp", "exit_timestamp", "consensus_snapshot_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in (
        "consensus_snapshot_is_pre_event",
        "missing_eps_actual",
        "missing_eps_consensus",
        "missing_revenue_actual",
        "missing_revenue_consensus",
        "eps_consensus_abs_below_floor",
        "revenue_consensus_abs_below_floor",
        "missing_gap_open",
        "missing_gap_prev_close",
        "missing_recent_atr",
        "missing_volume_30m",
        "missing_expected_volume_30m",
        "insufficient_opening_30m_bars",
        "missing_vwap_30m",
        "missing_range_30m",
        "missing_close_30m",
        "peer_proxy_fallback_used",
        "missing_sector_mapping",
        "missing_sector_proxy_return_30m",
        "missing_peer_proxy_return_30m",
        "macro_day_flag",
        "simultaneous_peer_earnings_flag",
        "high_spread_flag",
        "binary_news_flag",
        "is_full_regular_session",
        "halt_flag",
        "suspected_halt_or_bad_session",
        "split_flag",
        "corporate_action_flag",
        "is_tradeable_v1",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame.sort_values(["event_session", "symbol", "event_id"], na_position="last").reset_index(drop=True)


def split_events_by_timing(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return explicit report-timing partitions used by H3 v1 screening."""
    if "report_timing" not in events.columns:
        raise ValueError("events frame must include report_timing")

    pre_market = events.loc[events["report_timing"].eq("pre_market")].copy()
    after_market = events.loc[events["report_timing"].eq("after_market")].copy()
    excluded = events.loc[~events["report_timing"].isin(["pre_market", "after_market"])].copy()
    return {
        "pre_market": pre_market.reset_index(drop=True),
        "after_market": after_market.reset_index(drop=True),
        "excluded_timing": excluded.reset_index(drop=True),
    }


def timing_partition_paths(output_path: str | Path) -> dict[str, Path]:
    output = Path(output_path)
    return {name: output.parent / filename for name, filename in TIMING_PARTITION_FILENAMES.items()}


def write_timing_partitions(events: pd.DataFrame, output_path: str | Path) -> dict[str, Path]:
    paths = timing_partition_paths(output_path)
    partitions = split_events_by_timing(events)
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        partitions[name].to_parquet(path, index=False)
    return paths


def build_earnings_events(
    raw_json_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    intraday_panel_path: str | Path | None = None,
    macro_calendar_path: str | Path | None = None,
    halt_events_path: str | Path | None = None,
    quote_quality_path: str | Path | None = None,
    binary_news_path: str | Path | None = None,
    *,
    write_partitions: bool = True,
) -> Path:
    config = load_config(config_path)
    rows = load_raw_payload(raw_json_path)
    events = normalize_benzinga_earnings(rows, config)
    if intraday_panel_path is not None:
        intraday_panel = pd.read_parquet(intraday_panel_path)
        events = add_gap_features(events, intraday_panel, config)
        events = add_volume_features(events, intraday_panel, config)
        events = add_opening_range_features(events, intraday_panel, config)
        events = add_sector_peer_features(events, intraday_panel, config)
    events = add_exclusion_features(
        events,
        config,
        macro_calendar=_read_optional_table(macro_calendar_path),
        halt_events=_read_optional_table(halt_events_path),
        quote_quality=_read_optional_table(quote_quality_path),
        binary_news=_read_optional_table(binary_news_path),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output, index=False)
    if write_partitions:
        write_timing_partitions(events, output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build H3 earnings_events.parquet from audited raw earnings payloads.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--raw-json", required=True, help="Raw Benzinga/Polygon earnings payload JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--intraday-panel", help="Optional adjusted intraday OHLCV panel parquet used to add opening reaction features.")
    parser.add_argument("--macro-calendar", help="Optional macro session exclusion table: parquet/csv/json/jsonl.")
    parser.add_argument("--halt-events", help="Optional symbol/session halt exclusion table: parquet/csv/json/jsonl.")
    parser.add_argument("--quote-quality", help="Optional symbol/session spread quality table: parquet/csv/json/jsonl.")
    parser.add_argument("--binary-news", help="Optional symbol/session binary-news exclusion table: parquet/csv/json/jsonl.")
    parser.add_argument("--no-timing-partitions", action="store_true", help="Do not write pre/after/excluded timing parquet partitions.")
    args = parser.parse_args(argv)

    output = build_earnings_events(
        args.raw_json,
        output_path=args.output,
        config_path=args.config,
        intraday_panel_path=args.intraday_panel,
        macro_calendar_path=args.macro_calendar,
        halt_events_path=args.halt_events,
        quote_quality_path=args.quote_quality,
        binary_news_path=args.binary_news,
        write_partitions=not args.no_timing_partitions,
    )
    print(f"Wrote H3 earnings events to: {output}")
    if not args.no_timing_partitions:
        for name, path in timing_partition_paths(output).items():
            print(f"Wrote H3 {name} events to: {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""H5 capitulation gap reversal strategy runner.

This module implements the shares-first version of H5. It deliberately avoids
options assumptions until the Massive options ingestion layer can provide
historical bid/ask, contract availability, and liquidity filters.
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds


DEFAULT_CONFIG_PATH = Path("configs/strategy/capitulation_gap_reversal_h5_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/capitulation_gap_reversal/5min/h5_v1")

PRIMARY_LABEL = "h5_primary"
ALL_CAPITULATION_LABEL = "h5_all_capitulation_liquid"
NO_CLOSE_LOCATION_LABEL = "h5_no_close_location_control"
WAIT_1D_LABEL = "h5_primary_wait_1d"
SHORT_CONTINUATION_LABEL = "h5_short_continuation_control"
CONTROL_LABELS = (
    ALL_CAPITULATION_LABEL,
    NO_CLOSE_LOCATION_LABEL,
    WAIT_1D_LABEL,
    SHORT_CONTINUATION_LABEL,
)


@dataclass(frozen=True)
class H5SignalConfig:
    gap_down_threshold: float
    daily_drop_threshold: float
    three_day_drop_threshold: float
    min_rel_volume: float
    rel_volume_window: int
    close_location_min: float
    min_price: float
    min_adv_dollar: float
    adv_window: int
    min_history_sessions: int
    require_next_open_above_event_low: bool


@dataclass(frozen=True)
class H5ExitSpec:
    exit_id: str
    max_hold_sessions: int
    profit_target_r: float | None


@dataclass(frozen=True)
class H5Config:
    strategy_id: str
    hypothesis_id: str
    timeframe: str
    provider: str
    symbols: tuple[str, ...]
    cleaned_dir: Path
    file_template: str
    timestamp_timezone: str
    signal: H5SignalConfig
    exits: tuple[H5ExitSpec, ...]
    cost_bps_values: tuple[float, ...]
    split_policy: dict[str, Any]
    output_dir: Path
    max_positions_per_session: int
    max_initial_risk_pct: float
    stop_buffer_bps: float
    controls: dict[str, Any]


@dataclass(frozen=True)
class H5Outputs:
    output_dir: Path
    coverage_path: Path
    daily_path: Path
    events_path: Path
    trades_path: Path
    portfolio_daily_path: Path
    monthly_path: Path
    summary_path: Path
    distribution_path: Path
    manifest_path: Path
    report_path: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H5Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"H5 config must be a mapping: {config_path}")

    data_cfg = dict(raw.get("data", {}))
    signal_cfg = dict(raw.get("signal", {}))
    exit_cfg = dict(raw.get("exit", {}))
    position_cfg = dict(raw.get("position", {}))
    costs_cfg = dict(raw.get("costs", {}))
    outputs_cfg = dict(raw.get("outputs", {}))

    symbols = tuple(str(symbol).upper() for symbol in data_cfg.get("symbols", []) if str(symbol).strip())
    if not symbols:
        raise ValueError("data.symbols must contain at least one symbol")

    raw_exits = exit_cfg.get("rules", [])
    if not isinstance(raw_exits, list) or not raw_exits:
        raise ValueError("exit.rules must be a non-empty list")
    exits: list[H5ExitSpec] = []
    for item in raw_exits:
        exit_id = str(item.get("exit_id", "")).strip()
        if not exit_id:
            raise ValueError("each exit rule needs exit_id")
        max_hold = int(item.get("max_hold_sessions", 5))
        if max_hold <= 0:
            raise ValueError("exit max_hold_sessions must be positive")
        target = item.get("profit_target_r")
        exits.append(
            H5ExitSpec(
                exit_id=exit_id,
                max_hold_sessions=max_hold,
                profit_target_r=None if target is None else float(target),
            )
        )

    round_trip_cfg = costs_cfg.get("round_trip_bps", [10.0, 25.0, 50.0])
    if isinstance(round_trip_cfg, dict):
        round_trip_cfg = list(round_trip_cfg.values())
    cost_bps_values = tuple(float(value) for value in round_trip_cfg)
    if not cost_bps_values:
        raise ValueError("costs.round_trip_bps cannot be empty")

    return H5Config(
        strategy_id=str(raw.get("strategy_id", "capitulation_gap_reversal_h5_v1")).strip(),
        hypothesis_id=str(raw.get("hypothesis_id", "H5")).strip(),
        timeframe=str(raw.get("timeframe", "5min")).strip(),
        provider=str(data_cfg.get("provider", "massive")).strip(),
        symbols=symbols,
        cleaned_dir=Path(data_cfg.get("cleaned_dir", "data/cleaned")),
        file_template=str(data_cfg.get("file_template", "{cleaned_dir}/{timeframe}/{symbol}/{symbol}_{timeframe}_clean.parquet")),
        timestamp_timezone=str(data_cfg.get("timestamp_timezone", "America/New_York")),
        signal=H5SignalConfig(
            gap_down_threshold=float(signal_cfg.get("gap_down_threshold", -0.20)),
            daily_drop_threshold=float(signal_cfg.get("daily_drop_threshold", -0.30)),
            three_day_drop_threshold=float(signal_cfg.get("three_day_drop_threshold", -0.45)),
            min_rel_volume=float(signal_cfg.get("min_rel_volume", 10.0)),
            rel_volume_window=int(signal_cfg.get("rel_volume_window", 20)),
            close_location_min=float(signal_cfg.get("close_location_min", 0.60)),
            min_price=float(signal_cfg.get("min_price", 5.0)),
            min_adv_dollar=float(signal_cfg.get("min_adv_dollar", 20_000_000.0)),
            adv_window=int(signal_cfg.get("adv_window", 20)),
            min_history_sessions=int(signal_cfg.get("min_history_sessions", 20)),
            require_next_open_above_event_low=bool(signal_cfg.get("require_next_open_above_event_low", True)),
        ),
        exits=tuple(exits),
        cost_bps_values=cost_bps_values,
        split_policy=dict(raw.get("split_policy", {})),
        output_dir=Path(outputs_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)),
        max_positions_per_session=int(position_cfg.get("max_positions_per_session", 5)),
        max_initial_risk_pct=float(position_cfg.get("max_initial_risk_pct", 0.25)),
        stop_buffer_bps=float(position_cfg.get("stop_buffer_bps", 0.0)),
        controls=dict(raw.get("controls", {})),
    )


def _with_output_dir(config: H5Config, output_dir: str | Path | None) -> H5Config:
    if output_dir is None:
        return config
    return H5Config(
        strategy_id=config.strategy_id,
        hypothesis_id=config.hypothesis_id,
        timeframe=config.timeframe,
        provider=config.provider,
        symbols=config.symbols,
        cleaned_dir=config.cleaned_dir,
        file_template=config.file_template,
        timestamp_timezone=config.timestamp_timezone,
        signal=config.signal,
        exits=config.exits,
        cost_bps_values=config.cost_bps_values,
        split_policy=config.split_policy,
        output_dir=Path(output_dir),
        max_positions_per_session=config.max_positions_per_session,
        max_initial_risk_pct=config.max_initial_risk_pct,
        stop_buffer_bps=config.stop_buffer_bps,
        controls=config.controls,
    )


def symbol_path(config: H5Config, symbol: str) -> Path:
    return Path(
        config.file_template.format(
            cleaned_dir=config.cleaned_dir.as_posix(),
            timeframe=config.timeframe,
            symbol=symbol.upper(),
        )
    )


def load_symbol_frame(path: str | Path, symbol: str, timezone: str) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    frame = pd.read_parquet(input_path).copy()
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{input_path} missing columns: {', '.join(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].dt.tz is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert(timezone)
    if "session" not in frame:
        frame["session"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    if "bar_index" not in frame:
        frame = frame.sort_values("timestamp", kind="stable")
        frame["bar_index"] = frame.groupby("session", sort=False).cumcount()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol.upper()
    return (
        frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values(["session", "bar_index", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def load_symbol_frames(config: H5Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for symbol in config.symbols:
        path = symbol_path(config, symbol)
        if not path.exists():
            rows.append(
                {
                    "symbol": symbol,
                    "path": path.as_posix(),
                    "status": "missing",
                    "intraday_rows": 0,
                    "daily_rows": 0,
                    "first_session": "",
                    "last_session": "",
                }
            )
            continue
        frame = load_symbol_frame(path, symbol, config.timestamp_timezone)
        frames[symbol] = frame
        rows.append(
            {
                "symbol": symbol,
                "path": path.as_posix(),
                "status": "loaded",
                "intraday_rows": int(len(frame)),
                "daily_rows": int(frame["session"].nunique()),
                "first_session": str(frame["session"].min()) if not frame.empty else "",
                "last_session": str(frame["session"].max()) if not frame.empty else "",
            }
        )
    if not frames:
        raise FileNotFoundError("No H5 symbol files were found for data.symbols")
    return frames, pd.DataFrame(rows)


def build_daily_frame(intraday: pd.DataFrame, config: H5Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session, group in intraday.groupby("session", sort=True):
        ordered = group.sort_values(["bar_index", "timestamp"], kind="stable")
        rows.append(
            {
                "symbol": str(ordered["symbol"].iloc[0]),
                "session": str(session),
                "first_timestamp": ordered["timestamp"].iloc[0],
                "last_timestamp": ordered["timestamp"].iloc[-1],
                "open": float(ordered["open"].iloc[0]),
                "high": float(ordered["high"].max()),
                "low": float(ordered["low"].min()),
                "close": float(ordered["close"].iloc[-1]),
                "volume": float(ordered["volume"].sum()),
                "bars": int(len(ordered)),
            }
        )
    daily = pd.DataFrame(rows).sort_values(["symbol", "session"], kind="stable").reset_index(drop=True)
    if daily.empty:
        return daily

    grouped = daily.groupby("symbol", sort=False)
    daily["prev_close"] = grouped["close"].shift(1)
    daily["close_3_sessions_ago"] = grouped["close"].shift(3)
    daily["daily_return"] = daily["close"] / daily["prev_close"] - 1.0
    daily["gap_return"] = daily["open"] / daily["prev_close"] - 1.0
    daily["three_day_return"] = daily["close"] / daily["close_3_sessions_ago"] - 1.0
    daily["dollar_volume"] = daily["close"] * daily["volume"]
    daily["prior_avg_volume"] = grouped["volume"].transform(
        lambda series: series.shift(1).rolling(config.signal.rel_volume_window, min_periods=config.signal.min_history_sessions).mean()
    )
    daily["prior_adv_dollar"] = grouped["dollar_volume"].transform(
        lambda series: series.shift(1).rolling(config.signal.adv_window, min_periods=config.signal.min_history_sessions).mean()
    )
    daily["rel_volume"] = daily["volume"] / daily["prior_avg_volume"]
    range_width = daily["high"] - daily["low"]
    daily["close_location"] = np.where(range_width.gt(0.0), (daily["close"] - daily["low"]) / range_width, np.nan)
    daily["next_session"] = grouped["session"].shift(-1)
    daily["next_open"] = grouped["open"].shift(-1)
    daily["next_open_timestamp"] = grouped["first_timestamp"].shift(-1)
    daily["history_sessions"] = grouped.cumcount()
    return daily


def build_daily_frames(frames: dict[str, pd.DataFrame], config: H5Config) -> pd.DataFrame:
    daily_parts = [build_daily_frame(frame, config) for frame in frames.values()]
    return pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()


def _raw_capitulation_mask(daily: pd.DataFrame, config: H5Config) -> pd.Series:
    signal = config.signal
    return (
        daily["gap_return"].le(signal.gap_down_threshold)
        | daily["daily_return"].le(signal.daily_drop_threshold)
        | daily["three_day_return"].le(signal.three_day_drop_threshold)
    )


def _liquidity_mask(daily: pd.DataFrame, config: H5Config) -> pd.Series:
    signal = config.signal
    mask = (
        daily["close"].ge(signal.min_price)
        & daily["prior_adv_dollar"].ge(signal.min_adv_dollar)
        & daily["history_sessions"].ge(signal.min_history_sessions)
        & daily["next_open"].notna()
    )
    if signal.require_next_open_above_event_low:
        mask &= daily["next_open"].ge(daily["low"])
    return mask


def _make_event_id(label: str, symbol: str, session: str) -> str:
    digest = hashlib.sha1(f"{label}|{symbol}|{session}".encode("utf-8")).hexdigest()[:10]
    return f"{label}|{symbol}|{session}|{digest}"


def _event_score(frame: pd.DataFrame) -> pd.Series:
    downside = frame[["gap_return", "daily_return", "three_day_return"]].min(axis=1).abs().fillna(0.0)
    rel_volume = frame["rel_volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    close_location = frame["close_location"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return downside + 0.02 * rel_volume + 0.20 * close_location


def label_events(daily: pd.DataFrame, label: str, mask: pd.Series) -> pd.DataFrame:
    columns = [
        "symbol",
        "session",
        "first_timestamp",
        "last_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "daily_return",
        "gap_return",
        "three_day_return",
        "prior_avg_volume",
        "prior_adv_dollar",
        "rel_volume",
        "close_location",
        "next_session",
        "next_open",
        "next_open_timestamp",
        "history_sessions",
    ]
    events = daily.loc[mask, columns].copy()
    if events.empty:
        return events
    events["event_session"] = events["session"].astype(str)
    events["label"] = label
    events["event_id"] = [
        _make_event_id(label, str(symbol), str(session))
        for symbol, session in zip(events["symbol"], events["event_session"], strict=True)
    ]
    events["event_score"] = _event_score(events)
    return events.sort_values(["event_session", "event_score", "symbol"], ascending=[True, False, True], kind="stable").reset_index(drop=True)


def build_base_events(daily: pd.DataFrame, config: H5Config) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    raw = _raw_capitulation_mask(daily, config)
    liquid = _liquidity_mask(daily, config)
    rel_volume = daily["rel_volume"].ge(config.signal.min_rel_volume)
    close_location = daily["close_location"].ge(config.signal.close_location_min)

    events = [
        label_events(daily, PRIMARY_LABEL, raw & liquid & rel_volume & close_location),
    ]
    if config.controls.get("all_capitulation_liquid", True):
        events.append(label_events(daily, ALL_CAPITULATION_LABEL, raw & liquid))
    if config.controls.get("no_close_location", True):
        events.append(label_events(daily, NO_CLOSE_LOCATION_LABEL, raw & liquid & rel_volume))
    return pd.concat([event for event in events if not event.empty], ignore_index=True) if any(not event.empty for event in events) else pd.DataFrame()


def _split_events(events: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events.loc[events["event_session"].astype(str).isin(sessions)].copy()


def _cap_positions(events: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    if events.empty or max_positions <= 0:
        return events
    return events.sort_values(["event_session", "event_score", "symbol"], ascending=[True, False, True], kind="stable").groupby(
        ["event_session", "label"], sort=False
    ).head(max_positions).reset_index(drop=True)


def build_labeled_events(events: pd.DataFrame, config: H5Config, folds: tuple[ResearchFold, ...]) -> tuple[pd.DataFrame, dict[tuple[int, str], tuple[str, ...]]]:
    labeled_parts: list[pd.DataFrame] = []
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    for fold in folds:
        for split, sessions in (
            ("train", fold.train_sessions),
            ("validation", fold.validation_sessions),
            ("test", fold.test_sessions),
        ):
            split_sessions[(fold.fold, split)] = tuple(sessions)
            part = _cap_positions(_split_events(events, sessions), config.max_positions_per_session)
            if part.empty:
                continue
            part["fold"] = int(fold.fold)
            part["split"] = split
            labeled_parts.append(part)
    labeled = pd.concat(labeled_parts, ignore_index=True) if labeled_parts else pd.DataFrame()
    return labeled, split_sessions


def _session_index(frame: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    sessions: list[str] = []
    grouped: dict[str, pd.DataFrame] = {}
    for session, group in frame.groupby("session", sort=True):
        session_str = str(session)
        sessions.append(session_str)
        grouped[session_str] = group.sort_values(["bar_index", "timestamp"], kind="stable").reset_index(drop=True)
    return sessions, grouped


def _resolve_exit_rule(exit_spec: H5ExitSpec, side: int, entry: float, stop: float) -> tuple[float | None, float]:
    risk = (entry - stop) if side > 0 else (stop - entry)
    if risk <= 0.0 or not np.isfinite(risk):
        return None, risk
    target = None
    if exit_spec.profit_target_r is not None:
        target = entry + side * float(exit_spec.profit_target_r) * risk
    return target, risk


def _scan_exit(
    bars: pd.DataFrame,
    *,
    side: int,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
) -> tuple[float, pd.Timestamp, str]:
    last = bars.iloc[-1]
    for _, bar in bars.iterrows():
        open_px = float(bar["open"])
        high_px = float(bar["high"])
        low_px = float(bar["low"])
        timestamp = pd.Timestamp(bar["timestamp"])
        if side > 0:
            if open_px <= stop_price:
                return open_px, timestamp, "gap_stop"
            if low_px <= stop_price:
                return stop_price, timestamp, "stop"
            if target_price is not None:
                if open_px >= target_price:
                    return open_px, timestamp, "gap_target"
                if high_px >= target_price:
                    return target_price, timestamp, "target"
        else:
            if open_px >= stop_price:
                return open_px, timestamp, "gap_stop"
            if high_px >= stop_price:
                return stop_price, timestamp, "stop"
            if target_price is not None:
                if open_px <= target_price:
                    return open_px, timestamp, "gap_target"
                if low_px <= target_price:
                    return target_price, timestamp, "target"
    return float(last["close"]), pd.Timestamp(last["timestamp"]), "time"


def simulate_event_trade(
    event: pd.Series,
    frames: dict[str, pd.DataFrame],
    exit_spec: H5ExitSpec,
    config: H5Config,
    *,
    side: int,
    entry_delay_sessions: int,
    trade_label: str,
) -> dict[str, Any] | None:
    symbol = str(event["symbol"]).upper()
    frame = frames.get(symbol)
    if frame is None or frame.empty:
        return None
    sessions, grouped = _session_index(frame)
    event_session = str(event["event_session"])
    if event_session not in grouped:
        return None
    event_pos = sessions.index(event_session)
    entry_pos = event_pos + int(entry_delay_sessions)
    if entry_pos >= len(sessions):
        return None
    entry_session = sessions[entry_pos]
    entry_bar = grouped[entry_session].iloc[0]
    entry_price = float(entry_bar["open"])
    if not np.isfinite(entry_price) or entry_price <= 0.0:
        return None

    buffer = float(config.stop_buffer_bps) / 10_000.0
    if side > 0:
        stop_price = float(event["low"]) * (1.0 - buffer)
        if config.signal.require_next_open_above_event_low and entry_price < float(event["low"]):
            return None
    else:
        stop_price = float(event["high"]) * (1.0 + buffer)

    target_price, risk_dollars = _resolve_exit_rule(exit_spec, side, entry_price, stop_price)
    if target_price is None and exit_spec.profit_target_r is not None:
        return None
    if risk_dollars <= 0.0:
        return None
    risk_pct = risk_dollars / entry_price
    if risk_pct <= 0.0 or risk_pct > float(config.max_initial_risk_pct):
        return None

    last_pos = min(entry_pos + int(exit_spec.max_hold_sessions) - 1, len(sessions) - 1)
    hold_sessions = sessions[entry_pos : last_pos + 1]
    bars = pd.concat([grouped[session] for session in hold_sessions], ignore_index=True)
    exit_price, exit_timestamp, exit_reason = _scan_exit(
        bars,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
    )
    gross_return = side * (exit_price / entry_price - 1.0)
    gross_r = side * (exit_price - entry_price) / risk_dollars
    return {
        "strategy_id": config.strategy_id,
        "label": trade_label,
        "source_label": str(event["label"]),
        "fold": int(event["fold"]),
        "split": str(event["split"]),
        "exit_id": exit_spec.exit_id,
        "event_id": str(event["event_id"]),
        "symbol": symbol,
        "session": event_session,
        "entry_session": entry_session,
        "exit_session": pd.Timestamp(exit_timestamp).strftime("%Y-%m-%d"),
        "entry_timestamp": pd.Timestamp(entry_bar["timestamp"]),
        "exit_timestamp": exit_timestamp,
        "side": "long" if side > 0 else "short",
        "entry_delay_sessions": int(entry_delay_sessions),
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "stop_price": stop_price,
        "target_price": np.nan if target_price is None else float(target_price),
        "risk_dollars": risk_dollars,
        "risk_pct": risk_pct,
        "gross_return": gross_return,
        "gross_r": gross_r,
        "exit_reason": exit_reason,
        "max_hold_sessions": int(exit_spec.max_hold_sessions),
        "profit_target_r": np.nan if exit_spec.profit_target_r is None else float(exit_spec.profit_target_r),
        "event_open": float(event["open"]),
        "event_high": float(event["high"]),
        "event_low": float(event["low"]),
        "event_close": float(event["close"]),
        "event_volume": float(event["volume"]),
        "gap_return": float(event["gap_return"]),
        "daily_return": float(event["daily_return"]),
        "three_day_return": float(event["three_day_return"]),
        "rel_volume": float(event["rel_volume"]),
        "close_location": float(event["close_location"]),
        "prior_adv_dollar": float(event["prior_adv_dollar"]),
    }


def simulate_trades(events: pd.DataFrame, frames: dict[str, pd.DataFrame], config: H5Config) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    enable_wait = bool(config.controls.get("wait_1d", True))
    enable_short = bool(config.controls.get("short_continuation", True))
    for _, event in events.iterrows():
        source_label = str(event["label"])
        for exit_spec in config.exits:
            rows.append(
                simulate_event_trade(
                    event,
                    frames,
                    exit_spec,
                    config,
                    side=1,
                    entry_delay_sessions=1,
                    trade_label=source_label,
                )
            )
            if source_label == PRIMARY_LABEL and enable_wait:
                rows.append(
                    simulate_event_trade(
                        event,
                        frames,
                        exit_spec,
                        config,
                        side=1,
                        entry_delay_sessions=2,
                        trade_label=WAIT_1D_LABEL,
                    )
                )
            if source_label == PRIMARY_LABEL and enable_short:
                rows.append(
                    simulate_event_trade(
                        event,
                        frames,
                        exit_spec,
                        config,
                        side=-1,
                        entry_delay_sessions=1,
                        trade_label=SHORT_CONTINUATION_LABEL,
                    )
                )
    clean_rows = [row for row in rows if row is not None]
    return pd.DataFrame(clean_rows)


def apply_costs(trades: pd.DataFrame, cost_bps_values: tuple[float, ...]) -> pd.DataFrame:
    if trades.empty:
        return trades
    frames: list[pd.DataFrame] = []
    for cost_bps in cost_bps_values:
        costed = trades.copy()
        costed["cost_bps_round_trip"] = float(cost_bps)
        costed["cost_return_gross"] = float(cost_bps) / 10_000.0
        costed["net_return"] = costed["gross_return"].astype(float) - costed["cost_return_gross"]
        costed["net_r"] = costed["gross_r"].astype(float) - (costed["cost_return_gross"] / costed["risk_pct"].astype(float))
        frames.append(costed)
    return pd.concat(frames, ignore_index=True)


def _profit_factor(values: pd.Series) -> float:
    profit = float(values[values > 0.0].sum())
    loss = float(-values[values < 0.0].sum())
    if loss == 0.0:
        return np.inf if profit > 0.0 else 0.0
    return profit / loss


def _daily_sharpe(daily: pd.Series) -> float:
    clean = daily.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(252.0) * clean.mean() / std)


def _max_drawdown(values: pd.Series) -> float:
    equity = values.fillna(0.0).cumsum()
    return float((equity.cummax() - equity).max()) if not equity.empty else 0.0


def _top_abs_share(group: pd.DataFrame, top_n: int = 5) -> float:
    by_session = group.groupby("session")["net_return"].sum()
    denom = float(by_session.abs().sum())
    if denom <= 0.0:
        return 0.0
    return float(by_session.abs().sort_values(ascending=False).head(top_n).sum() / denom)


def aggregate_trades(
    trades: pd.DataFrame,
    split_sessions: dict[tuple[int, str], tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    keys = ["strategy_id", "label", "fold", "split", "exit_id", "side", "cost_bps_round_trip"]
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for key_values, group in trades.groupby(keys, sort=False, dropna=False):
        key = dict(zip(keys, key_values, strict=True))
        sessions = split_sessions.get((int(key["fold"]), str(key["split"])), tuple(sorted(group["session"].astype(str).unique())))
        daily = pd.Series(0.0, index=pd.Index(sessions, name="session"))
        daily = daily.add(group.groupby("session")["net_return"].sum(), fill_value=0.0).sort_index()
        monthly = daily.groupby(pd.to_datetime(daily.index).strftime("%Y-%m")).sum()
        net = group["net_return"].astype(float)
        net_r = group["net_r"].astype(float)
        gross_r = group["gross_r"].astype(float)
        for session, value in daily.items():
            daily_rows.append({**key, "session": str(session), "net_return": float(value)})
        for month, value in monthly.items():
            monthly_rows.append({**key, "month": str(month), "net_return": float(value)})
        summary_rows.append(
            {
                **key,
                "trades": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "gross_return": float(group["gross_return"].sum()),
                "total_cost": float(group["cost_return_gross"].sum()),
                "net_return": float(net.sum()),
                "avg_trade_net": float(net.mean()) if len(net) else 0.0,
                "median_trade_net": float(net.median()) if len(net) else 0.0,
                "avg_net_r": float(net_r.mean()) if len(net_r) else 0.0,
                "median_net_r": float(net_r.median()) if len(net_r) else 0.0,
                "p75_net_r": float(net_r.quantile(0.75)) if len(net_r) else 0.0,
                "p95_net_r": float(net_r.quantile(0.95)) if len(net_r) else 0.0,
                "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
                "r_win_rate": float(net_r.gt(0.0).mean()) if len(net_r) else 0.0,
                "payoff_ratio": _profit_factor(net),
                "daily_sharpe": _daily_sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "sessions": int(len(daily)),
                "sessions_with_trades": int(group["session"].nunique()),
                "stopped_rate": float(group["exit_reason"].isin(["stop", "gap_stop"]).mean()) if len(group) else 0.0,
                "target_rate": float(group["exit_reason"].isin(["target", "gap_target"]).mean()) if len(group) else 0.0,
                "avg_risk_pct": float(group["risk_pct"].mean()) if len(group) else 0.0,
                "top5_abs_share": _top_abs_share(group),
            }
        )
        for metric in ("gross_return", "net_return", "gross_r", "net_r", "risk_pct"):
            values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if values.empty:
                continue
            distribution_rows.append(
                {
                    **key,
                    "metric": metric,
                    "count": int(values.shape[0]),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                    "without_top_1pct_mean": _mean_without_top_pct(values, 0.01),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), pd.DataFrame(monthly_rows), pd.DataFrame(distribution_rows)


def _mean_without_top_pct(values: pd.Series, pct: float) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float).sort_values()
    if clean.empty:
        return 0.0
    drop_n = int(np.ceil(len(clean) * float(pct)))
    if drop_n <= 0 or drop_n >= len(clean):
        return float(clean.mean())
    return float(clean.iloc[:-drop_n].mean())


def build_coverage(file_coverage: pd.DataFrame, daily: pd.DataFrame, events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in file_coverage.iterrows():
        symbol = str(row["symbol"])
        symbol_daily = daily.loc[daily["symbol"].eq(symbol)] if not daily.empty else pd.DataFrame()
        symbol_events = events.loc[events["symbol"].eq(symbol)] if not events.empty else pd.DataFrame()
        symbol_trades = trades.loc[trades["symbol"].eq(symbol)] if not trades.empty else pd.DataFrame()
        rows.append(
            {
                **row.to_dict(),
                "raw_capitulation_events": int(
                    symbol_events.loc[symbol_events["label"].isin([PRIMARY_LABEL, ALL_CAPITULATION_LABEL, NO_CLOSE_LOCATION_LABEL]), "event_id"].nunique()
                )
                if not symbol_events.empty
                else 0,
                "primary_events": int(symbol_events.loc[symbol_events["label"].eq(PRIMARY_LABEL), "event_id"].nunique()) if not symbol_events.empty else 0,
                "trade_rows": int(len(symbol_trades)),
                "first_daily_session": str(symbol_daily["session"].min()) if not symbol_daily.empty else "",
                "last_daily_session": str(symbol_daily["session"].max()) if not symbol_daily.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("symbol", kind="stable").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> list[str]:
    if frame.empty:
        return ["No rows."]
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    lines = ["| " + " | ".join(visible.columns) + " |", "| " + " | ".join(["---"] * len(visible.columns)) + " |"]
    for _, row in visible.iterrows():
        values: list[str] = []
        for column in visible.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}" if np.isfinite(value) else "")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _rollup(summary: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    part = summary.loc[
        summary["cost_bps_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if part.empty:
        return pd.DataFrame()
    grouped = (
        part.groupby(["split", "label", "exit_id", "side"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            trades=("trades", "sum"),
            symbols=("symbols", "max"),
            net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            avg_net_r=("avg_net_r", "mean"),
            p95_net_r=("p95_net_r", "max"),
            win_rate=("win_rate", "mean"),
            target_rate=("target_rate", "mean"),
            stopped_rate=("stopped_rate", "mean"),
            max_top5_abs_share=("top5_abs_share", "max"),
        )
    )
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "exit_id", "net_return"], ascending=[True, True, False], kind="stable").drop(columns="_split_order")


def _write_manifest(
    path: Path,
    config: H5Config,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: H5Outputs,
) -> None:
    config_file = Path(config_path)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("strategy", config.strategy_id, "h5_v1", config.timeframe),
            "run_type": "h5_capitulation_gap_reversal",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": config.strategy_id,
            "hypothesis_id": config.hypothesis_id,
            "phase": "H5 shares-first screening",
            "position": "single_name_long_only_with_short_continuation_control",
            "entry_rule": "next_regular_session_open_after_capitulation",
            "labels": [PRIMARY_LABEL, *CONTROL_LABELS],
            "exit_rules": [exit_spec.__dict__ for exit_spec in config.exits],
        },
        "data": {
            "provider": config.provider,
            "symbols": list(config.symbols),
            "cleaned_dir": config.cleaned_dir.as_posix(),
            "file_template": config.file_template,
            "config_path": config_file.as_posix(),
            "config_fingerprint": fingerprint_path(config_file) if config_file.exists() else "MISSING",
            "timeframe": config.timeframe,
            "split_policy": config.split_policy,
            "n_folds": len(folds),
        },
        "parameters": {
            "signal": config.signal.__dict__,
            "cost_bps_values": list(config.cost_bps_values),
            "max_positions_per_session": config.max_positions_per_session,
            "max_initial_risk_pct": config.max_initial_risk_pct,
            "stop_buffer_bps": config.stop_buffer_bps,
            "controls": config.controls,
        },
        "outputs": {
            "coverage": outputs.coverage_path.as_posix(),
            "daily": outputs.daily_path.as_posix(),
            "events": outputs.events_path.as_posix(),
            "trades": outputs.trades_path.as_posix(),
            "portfolio_daily": outputs.portfolio_daily_path.as_posix(),
            "monthly": outputs.monthly_path.as_posix(),
            "summary": outputs.summary_path.as_posix(),
            "distribution": outputs.distribution_path.as_posix(),
            "report": outputs.report_path.as_posix(),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _write_report(path: Path, config: H5Config, coverage: pd.DataFrame, summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    conservative_cost = 25.0 if 25.0 in config.cost_bps_values else float(config.cost_bps_values[-1])
    rollup = _rollup(summary, conservative_cost)
    lines = [
        "# H5 Capitulation Gap Reversal",
        "",
        "Shares-first implementation. Options are intentionally out of scope until historical contract selection and bid/ask data are ingested.",
        "",
        "## Coverage",
        "",
        *_markdown_table(
            coverage,
            ["symbol", "status", "intraday_rows", "daily_rows", "primary_events", "trade_rows", "first_session", "last_session"],
            limit=80,
        ),
        "",
        f"## Rollup, Cost={conservative_cost:g} bps Round Trip",
        "",
        *_markdown_table(
            rollup,
            [
                "split",
                "label",
                "exit_id",
                "side",
                "folds",
                "trades",
                "symbols",
                "net_return",
                "avg_trade_net",
                "avg_net_r",
                "p95_net_r",
                "win_rate",
                "target_rate",
                "stopped_rate",
                "max_top5_abs_share",
            ],
            limit=100,
        ),
        "",
        "## Research Contract",
        "",
        "- Entry is next regular-session open after the event session.",
        "- Primary label requires raw capitulation, liquidity, relative volume, close-location recovery, and next-open tradability.",
        "- Controls include all liquid capitulations, no-close-location, one-day-delayed entry, and short-continuation.",
        "- `net_r` converts notional cost into stop-risk units: `gross_r - cost_return / risk_pct`.",
        "- Distribution artifacts include mean without the top 1 pct of trades for outlier checks.",
        "",
        "## Config",
        "",
        f"- Strategy id: `{config.strategy_id}`",
        f"- Symbols: `{', '.join(config.symbols)}`",
        f"- Exits: `{', '.join(exit_spec.exit_id for exit_spec in config.exits)}`",
        f"- Costs bps: `{', '.join(str(value) for value in config.cost_bps_values)}`",
        "",
    ]
    if not distribution.empty:
        preview = distribution.loc[
            distribution["cost_bps_round_trip"].eq(conservative_cost)
            & distribution["metric"].eq("net_r")
            & distribution["split"].isin(["validation", "test"])
        ].copy()
        lines.extend(
            [
                "## Net R Distribution Preview",
                "",
                *_markdown_table(preview, ["split", "label", "exit_id", "side", "count", "mean", "p05", "median", "p95", "without_top_1pct_mean"], limit=80),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_strategy(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> H5Outputs:
    config = _with_output_dir(load_config(config_path), output_dir)
    frames, file_coverage = load_symbol_frames(config)
    daily = build_daily_frames(frames, config)
    base_events = build_base_events(daily, config)
    split_source = daily.loc[:, ["session"]].drop_duplicates().sort_values("session", kind="stable")
    folds = build_monthly_folds(split_source, config.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")
    labeled_events, split_sessions = build_labeled_events(base_events, config, folds)
    raw_trades = simulate_trades(labeled_events, frames, config)
    trades = apply_costs(raw_trades, config.cost_bps_values)
    summary, portfolio_daily, monthly, distribution = aggregate_trades(trades, split_sessions)
    coverage = build_coverage(file_coverage, daily, labeled_events, trades)

    root = config.output_dir
    outputs = H5Outputs(
        output_dir=root,
        coverage_path=root / "coverage.parquet",
        daily_path=root / "daily.parquet",
        events_path=root / "events.parquet",
        trades_path=root / "trades.parquet",
        portfolio_daily_path=root / "portfolio_daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        distribution_path=root / "distribution.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(outputs.coverage_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    labeled_events.to_parquet(outputs.events_path, index=False)
    trades.to_parquet(outputs.trades_path, index=False)
    portfolio_daily.to_parquet(outputs.portfolio_daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    distribution.to_parquet(outputs.distribution_path, index=False)
    _write_manifest(outputs.manifest_path, config, config_path, folds, outputs)
    _write_report(outputs.report_path, config, coverage, summary, distribution)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H5 capitulation gap reversal strategy")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH.as_posix())
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_strategy(args.config, output_dir=args.output_dir)
    print(outputs.report_path)


if __name__ == "__main__":
    main()

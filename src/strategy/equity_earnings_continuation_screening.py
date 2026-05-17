from __future__ import annotations

"""Phase 3 screening baselines for H3 earnings continuation."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.earnings_events_h3 import _normalise_ohlcv_panel
from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_earnings_continuation_h3_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/equity_earnings_continuation/5min/phase3_screening")

H3_SIGNAL_LABEL = "h3_candidate_screen"
EARNINGS_BEAT_LABEL = "earnings_beat_only"
GAP_MODERATE_LABEL = "gap_moderate_only"
INTRADAY_MOMENTUM_NO_EVENT_LABEL = "intraday_momentum_no_event"
RANDOM_CONTROL_LABEL = "random_same_frequency_by_ticker"
SAME_HOUR_CONTROL_LABEL = "same_hour_by_ticker"
SECTOR_EQUIVALENT_LABEL = "sector_proxy_equivalent"

EVENT_BASELINE_LABELS = (H3_SIGNAL_LABEL, EARNINGS_BEAT_LABEL, GAP_MODERATE_LABEL)
CONTROL_LABELS = (
    INTRADAY_MOMENTUM_NO_EVENT_LABEL,
    RANDOM_CONTROL_LABEL,
    SAME_HOUR_CONTROL_LABEL,
    SECTOR_EQUIVALENT_LABEL,
)


@dataclass(frozen=True)
class H3ScreeningConfig:
    strategy_id: str
    hypothesis_id: str
    timeframe: str
    earnings_events_path: Path
    intraday_panel_path: Path
    output_dir: Path
    split_policy: dict[str, Any]
    controls: dict[str, Any]
    cost_bps_values: tuple[float, ...]
    horizons: tuple[str, ...]
    timestamp_timezone: str
    entry_time: str
    latest_allowed_entry_time: str
    same_session_close_time: str
    random_seed: int
    signal_filters: list[dict[str, Any]]


@dataclass(frozen=True)
class H3ScreeningOutputs:
    output_dir: Path
    coverage_path: Path
    events_path: Path
    trades_path: Path
    daily_path: Path
    monthly_path: Path
    summary_path: Path
    distribution_path: Path
    manifest_path: Path
    report_path: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H3ScreeningConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"H3 screening config must be a mapping: {path}")

    data_cfg = dict(raw.get("data", {}))
    events_cfg = dict(raw.get("events", {}))
    entry_cfg = dict(events_cfg.get("entry", {}))
    exit_cfg = dict(events_cfg.get("exit", {}))
    costs_cfg = dict(raw.get("costs", {}))
    round_trip_costs = dict(costs_cfg.get("round_trip_bps", {}))
    outputs_cfg = dict(raw.get("outputs", {}))
    controls_cfg = dict(raw.get("controls", {}))
    session_policy = dict(data_cfg.get("timezone_session_policy", {}))

    cost_values = tuple(float(round_trip_costs[name]) for name in ("base", "conservative", "stress") if name in round_trip_costs)
    if not cost_values:
        cost_values = (0.0,)

    primary_exit = str(exit_cfg.get("primary_exit", "same_session_close")).strip()
    secondary = [str(value).strip() for value in exit_cfg.get("secondary_exits", [])]
    horizons = tuple(dict.fromkeys([primary_exit, *secondary]))
    if not horizons:
        horizons = ("same_session_close",)

    configured_output = outputs_cfg.get("screening_output_dir")
    if configured_output is None:
        base_output = Path(outputs_cfg.get("output_dir", DEFAULT_OUTPUT_DIR.parent))
        configured_output = base_output / "phase3_screening"

    return H3ScreeningConfig(
        strategy_id=str(raw.get("strategy_id", "equity_earnings_continuation_h3_v1")).strip(),
        hypothesis_id=str(raw.get("hypothesis_id", "H3")).strip(),
        timeframe=str(raw.get("timeframe", "5min")).strip(),
        earnings_events_path=Path(data_cfg.get("earnings_events_path", "data/events/earnings/h3_v1/earnings_events.parquet")),
        intraday_panel_path=Path(data_cfg.get("intraday_panel_path", "data/aligned/equities/5min/h3_v1/panel.parquet")),
        output_dir=Path(configured_output),
        split_policy=dict(raw.get("split_policy", {})),
        controls=controls_cfg,
        cost_bps_values=cost_values,
        horizons=horizons,
        timestamp_timezone=str(data_cfg.get("timestamp_timezone", "America/New_York")),
        entry_time=str(entry_cfg.get("entry_time", session_policy.get("entry_time", "10:00"))),
        latest_allowed_entry_time=str(entry_cfg.get("latest_allowed_entry_time", session_policy.get("latest_allowed_entry_time", "10:05"))),
        same_session_close_time=str(exit_cfg.get("primary_exit_time", session_policy.get("regular_close", "16:00"))),
        random_seed=int(controls_cfg.get("random_seed", 3003)),
        signal_filters=list(dict(raw.get("signal", {})).get("filters", [])),
    )


def _with_output_dir(config: H3ScreeningConfig, output_dir: str | Path | None) -> H3ScreeningConfig:
    if output_dir is None:
        return config
    return H3ScreeningConfig(
        strategy_id=config.strategy_id,
        hypothesis_id=config.hypothesis_id,
        timeframe=config.timeframe,
        earnings_events_path=config.earnings_events_path,
        intraday_panel_path=config.intraday_panel_path,
        output_dir=Path(output_dir),
        split_policy=config.split_policy,
        controls=config.controls,
        cost_bps_values=config.cost_bps_values,
        horizons=config.horizons,
        timestamp_timezone=config.timestamp_timezone,
        entry_time=config.entry_time,
        latest_allowed_entry_time=config.latest_allowed_entry_time,
        same_session_close_time=config.same_session_close_time,
        random_seed=config.random_seed,
        signal_filters=config.signal_filters,
    )


def load_events(path: str | Path) -> pd.DataFrame:
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(event_path)
    events = pd.read_parquet(event_path).copy()
    required = {"event_id", "symbol", "event_session"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(f"earnings events missing required columns: {', '.join(missing)}")
    events["symbol"] = events["symbol"].astype(str).str.upper().str.strip()
    events["event_session"] = events["event_session"].astype(str)
    if "entry_timestamp" in events:
        events["entry_timestamp"] = pd.to_datetime(events["entry_timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in events:
        events["exit_timestamp"] = pd.to_datetime(events["exit_timestamp"], utc=True, errors="coerce")
    return events.sort_values(["event_session", "symbol", "event_id"], kind="stable").reset_index(drop=True)


def load_panel(path: str | Path, config: H3ScreeningConfig) -> pd.DataFrame:
    panel_path = Path(path)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path)
    return _normalise_ohlcv_panel(panel, config.timestamp_timezone)


def _clock_inclusive(clock: pd.Series, start: str, end: str) -> pd.Series:
    return clock.ge(start) & clock.le(end)


def build_symbol_session_prices(panel: pd.DataFrame, config: H3ScreeningConfig) -> pd.DataFrame:
    required = {"symbol", "session", "open", "close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"intraday panel missing required columns: {', '.join(missing)}")

    prepared = panel.copy()
    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
        prepared["local_clock"] = prepared["timestamp"].dt.tz_convert(config.timestamp_timezone).dt.strftime("%H:%M")
    elif "bar_index" in prepared.columns:
        prepared["local_clock"] = ""
    else:
        raise KeyError("intraday panel must include timestamp or bar_index")
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper().str.strip()
    prepared["session"] = prepared["session"].astype(str)
    for column in ("open", "close"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.sort_values(["symbol", "session", "timestamp" if "timestamp" in prepared.columns else "bar_index"], kind="stable")

    rows: list[dict[str, Any]] = []
    for (symbol, session), group in prepared.groupby(["symbol", "session"], sort=False):
        group = group.copy()
        valid_open = group["open"].replace([np.inf, -np.inf], np.nan)
        valid_close = group["close"].replace([np.inf, -np.inf], np.nan)
        entry_candidates = group.loc[_clock_inclusive(group["local_clock"], config.entry_time, config.latest_allowed_entry_time)]
        close_candidates = group.loc[group["local_clock"].lt(config.same_session_close_time)]
        ten_thirty_candidates = group.loc[_clock_inclusive(group["local_clock"], "10:30", "10:35")]
        rows.append(
            {
                "symbol": str(symbol),
                "session": str(session),
                "first_timestamp": group["timestamp"].iloc[0] if "timestamp" in group else pd.NaT,
                "first_open": float(valid_open.iloc[0]) if len(valid_open) and pd.notna(valid_open.iloc[0]) else np.nan,
                "entry_timestamp": entry_candidates["timestamp"].iloc[0] if "timestamp" in entry_candidates and not entry_candidates.empty else pd.NaT,
                "entry_open": float(entry_candidates["open"].iloc[0]) if not entry_candidates.empty and pd.notna(entry_candidates["open"].iloc[0]) else np.nan,
                "same_session_close_timestamp": close_candidates["timestamp"].iloc[-1] if "timestamp" in close_candidates and not close_candidates.empty else pd.NaT,
                "same_session_close": float(close_candidates["close"].iloc[-1]) if not close_candidates.empty and pd.notna(close_candidates["close"].iloc[-1]) else np.nan,
                "t_plus_1_10_30_timestamp": ten_thirty_candidates["timestamp"].iloc[0] if "timestamp" in ten_thirty_candidates and not ten_thirty_candidates.empty else pd.NaT,
                "t_plus_1_10_30_px": float(ten_thirty_candidates["open"].iloc[0]) if not ten_thirty_candidates.empty and pd.notna(ten_thirty_candidates["open"].iloc[0]) else np.nan,
                "bars": int(len(group)),
            }
        )

    prices = pd.DataFrame(rows).sort_values(["symbol", "session"], kind="stable").reset_index(drop=True)
    prices["next_session"] = prices.groupby("symbol", sort=False)["session"].shift(-1)
    prices["t_plus_1_open_timestamp"] = prices.groupby("symbol", sort=False)["first_timestamp"].shift(-1)
    prices["t_plus_1_open"] = prices.groupby("symbol", sort=False)["first_open"].shift(-1)
    prices["t_plus_1_close_timestamp"] = prices.groupby("symbol", sort=False)["same_session_close_timestamp"].shift(-1)
    prices["t_plus_1_close"] = prices.groupby("symbol", sort=False)["same_session_close"].shift(-1)
    prices["next_10_30_timestamp"] = prices.groupby("symbol", sort=False)["t_plus_1_10_30_timestamp"].shift(-1)
    prices["next_10_30_px"] = prices.groupby("symbol", sort=False)["t_plus_1_10_30_px"].shift(-1)
    return prices


def _price_lookup(prices: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    return {
        (str(row["symbol"]).upper(), str(row["session"])): row
        for _, row in prices.iterrows()
    }


def _is_positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def _exit_for_horizon(row: pd.Series, horizon: str) -> tuple[float, pd.Timestamp | pd.NaT]:
    if horizon == "same_session_close":
        return float(row.get("same_session_close", np.nan)), row.get("same_session_close_timestamp", pd.NaT)
    if horizon == "t_plus_1_open":
        return float(row.get("t_plus_1_open", np.nan)), row.get("t_plus_1_open_timestamp", pd.NaT)
    if horizon == "t_plus_1_close":
        return float(row.get("t_plus_1_close", np.nan)), row.get("t_plus_1_close_timestamp", pd.NaT)
    if horizon == "t_plus_1_10_30":
        return float(row.get("next_10_30_px", np.nan)), row.get("next_10_30_timestamp", pd.NaT)
    raise ValueError(f"unsupported H3 screening horizon: {horizon}")


def _gross_return_for(lookup: dict[tuple[str, str], pd.Series], symbol: str, session: str, horizon: str) -> tuple[float, float, float, Any, Any]:
    row = lookup.get((str(symbol).upper(), str(session)))
    if row is None:
        return np.nan, np.nan, np.nan, pd.NaT, pd.NaT
    entry = float(row.get("entry_open", np.nan))
    exit_px, exit_ts = _exit_for_horizon(row, horizon)
    if not (_is_positive_finite(entry) and _is_positive_finite(exit_px)):
        return np.nan, entry, exit_px, row.get("entry_timestamp", pd.NaT), exit_ts
    return float(np.log(exit_px / entry)), entry, exit_px, row.get("entry_timestamp", pd.NaT), exit_ts


def _tradeable_events(events: pd.DataFrame) -> pd.Series:
    mask = events["event_session"].notna() & events["symbol"].astype(str).str.len().gt(0)
    if "is_tradeable_v1" in events.columns:
        mask &= events["is_tradeable_v1"].fillna(False).astype(bool)
    if "exclusion_flags" in events.columns:
        flags = events["exclusion_flags"].fillna("").astype(str).str.strip().str.lower()
        mask &= flags.isin(["", "none", "nan"])
    return mask.fillna(False)


def _numeric(events: pd.DataFrame, column: str) -> pd.Series:
    if column not in events:
        return pd.Series(np.nan, index=events.index, dtype="float64")
    return pd.to_numeric(events[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _gap_filter_config(config: H3ScreeningConfig) -> tuple[float, float]:
    for item in config.signal_filters:
        if str(item.get("field")) == "gap_atr" and str(item.get("operator")) == "between":
            return float(item.get("min_value", 0.25)), float(item.get("max_value", 2.50))
    return 0.25, 2.50


def _rel_volume_filter_config(config: H3ScreeningConfig) -> tuple[str, float]:
    for item in config.signal_filters:
        if str(item.get("field")) == "rel_volume_30m":
            return str(item.get("operator", ">=")), float(item.get("quantile", 0.60))
    return ">=", 0.60


def _sector_return_min(config: H3ScreeningConfig) -> float:
    for item in config.signal_filters:
        if str(item.get("field")) == "sector_return_30m":
            return float(item.get("value", -0.001))
    return -0.001


def fit_rel_volume_threshold(train_events: pd.DataFrame, config: H3ScreeningConfig) -> float:
    _, quantile = _rel_volume_filter_config(config)
    values = _numeric(train_events, "rel_volume_30m").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(quantile))


def h3_candidate_mask(events: pd.DataFrame, config: H3ScreeningConfig, rel_volume_threshold: float) -> pd.Series:
    gap_min, gap_max = _gap_filter_config(config)
    sector_min = _sector_return_min(config)
    mask = _tradeable_events(events)
    mask &= _numeric(events, "eps_surprise_z").gt(0.0)
    mask &= _numeric(events, "revenue_surprise_z").ge(0.0)
    mask &= _numeric(events, "gap_atr").between(gap_min, gap_max, inclusive="both")
    if np.isfinite(rel_volume_threshold):
        mask &= _numeric(events, "rel_volume_30m").ge(rel_volume_threshold)
    else:
        mask &= False
    mask &= _numeric(events, "close_30m").ge(_numeric(events, "vwap_30m"))
    mask &= _numeric(events, "sector_return_30m").ge(sector_min)
    return mask.fillna(False)


def earnings_beat_mask(events: pd.DataFrame) -> pd.Series:
    mask = _tradeable_events(events)
    mask &= _numeric(events, "eps_surprise").gt(0.0)
    mask &= _numeric(events, "revenue_surprise").ge(0.0)
    return mask.fillna(False)


def gap_moderate_mask(events: pd.DataFrame, config: H3ScreeningConfig) -> pd.Series:
    gap_min, gap_max = _gap_filter_config(config)
    mask = _tradeable_events(events)
    mask &= _numeric(events, "gap_atr").between(gap_min, gap_max, inclusive="both")
    return mask.fillna(False)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_000


def _base_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in [
            "event_id",
            "symbol",
            "event_session",
            "report_timing",
            "sector_id",
            "sector_proxy",
            "peer_proxy_symbol",
            "eps_surprise",
            "revenue_surprise",
            "eps_surprise_z",
            "revenue_surprise_z",
            "gap_atr",
            "rel_volume_30m",
            "close_30m",
            "vwap_30m",
            "sector_return_30m",
        ]
        if column in frame.columns
    ]
    return frame.loc[:, keep].copy()


def make_labeled_events(events: pd.DataFrame, label: str, fold: int, split: str) -> pd.DataFrame:
    out = _base_event_columns(events)
    out["label"] = label
    out["fold"] = int(fold)
    out["split"] = split
    out["source_event_id"] = out["event_id"].astype(str)
    out["is_synthetic_control"] = False
    return out


def _event_pairs(events: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(events["symbol"].astype(str).str.upper(), events["event_session"].astype(str), strict=False))


def _valid_price_sessions(
    prices: pd.DataFrame,
    *,
    symbol: str,
    sessions: tuple[str, ...],
    horizon: str,
    excluded_pairs: set[tuple[str, str]],
    require_opening_momentum: bool = False,
) -> pd.DataFrame:
    rows = prices.loc[
        prices["symbol"].eq(str(symbol).upper())
        & prices["session"].astype(str).isin(sessions)
        & ~prices.apply(lambda row: (str(row["symbol"]).upper(), str(row["session"])) in excluded_pairs, axis=1)
    ].copy()
    if rows.empty:
        return rows
    rows["entry_valid"] = rows["entry_open"].map(_is_positive_finite)
    exit_values = rows.apply(lambda row: _exit_for_horizon(row, horizon)[0], axis=1)
    rows["exit_valid"] = exit_values.map(_is_positive_finite)
    rows = rows.loc[rows["entry_valid"] & rows["exit_valid"]].copy()
    if require_opening_momentum:
        rows["opening_momentum_30m"] = np.log(rows["entry_open"].astype(float) / rows["first_open"].astype(float))
        rows = rows.loc[rows["opening_momentum_30m"].gt(0.0)]
    return rows


def sample_non_event_controls(
    reference_events: pd.DataFrame,
    prices: pd.DataFrame,
    source_events: pd.DataFrame,
    *,
    label: str,
    fold: int,
    split: str,
    sessions: tuple[str, ...],
    horizon: str,
    seed: int,
    require_opening_momentum: bool = False,
) -> pd.DataFrame:
    if reference_events.empty:
        return pd.DataFrame()
    excluded_pairs = _event_pairs(source_events)
    sampled_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for _, event in reference_events.iterrows():
        symbol = str(event["symbol"]).upper()
        candidates = _valid_price_sessions(
            prices,
            symbol=symbol,
            sessions=sessions,
            horizon=horizon,
            excluded_pairs=excluded_pairs,
            require_opening_momentum=require_opening_momentum,
        )
        if candidates.empty:
            continue
        selected = candidates.iloc[int(rng.integers(0, len(candidates)))]
        row = event.to_dict()
        row["event_id"] = f"{label}|{event.get('event_id')}|{selected['session']}"
        row["event_session"] = str(selected["session"])
        row["label"] = label
        row["fold"] = int(fold)
        row["split"] = split
        row["source_event_id"] = str(event.get("event_id"))
        row["is_synthetic_control"] = True
        sampled_rows.append(row)
    return pd.DataFrame(sampled_rows)


def _proxy_symbol(row: pd.Series) -> str:
    for column in ("peer_proxy_symbol", "sector_proxy"):
        value = row.get(column)
        if value not in (None, "") and pd.notna(value):
            return str(value).upper().strip()
    return "SPY"


def simulate_trades(
    labeled_events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    strategy_id: str,
    horizon: str,
    trade_symbol_mode: str = "event_symbol",
) -> pd.DataFrame:
    if labeled_events.empty:
        return pd.DataFrame()
    lookup = _price_lookup(prices)
    rows: list[dict[str, Any]] = []
    for _, event in labeled_events.iterrows():
        event_symbol = str(event["symbol"]).upper()
        session = str(event["event_session"])
        proxy_symbol = _proxy_symbol(event)
        trade_symbol = proxy_symbol if trade_symbol_mode == "sector_proxy" else event_symbol
        gross, entry_px, exit_px, entry_ts, exit_ts = _gross_return_for(lookup, trade_symbol, session, horizon)
        if not np.isfinite(gross):
            continue
        sector_gross, sector_entry, sector_exit, _, _ = _gross_return_for(lookup, proxy_symbol, session, horizon)
        index_gross, index_entry, index_exit, _, _ = _gross_return_for(lookup, "SPY", session, horizon)
        sector_residual = gross - sector_gross if np.isfinite(sector_gross) else np.nan
        index_residual = gross - index_gross if np.isfinite(index_gross) else np.nan
        rows.append(
            {
                "strategy_id": strategy_id,
                "label": str(event["label"]),
                "fold": int(event["fold"]),
                "split": str(event["split"]),
                "horizon": horizon,
                "event_id": str(event["event_id"]),
                "source_event_id": str(event.get("source_event_id", event["event_id"])),
                "is_synthetic_control": bool(event.get("is_synthetic_control", False)),
                "event_symbol": event_symbol,
                "trade_symbol": trade_symbol,
                "sector_id": event.get("sector_id", ""),
                "sector_proxy_symbol": proxy_symbol,
                "session": session,
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "side": "long",
                "entry_price": entry_px,
                "exit_price": exit_px,
                "gross_return": gross,
                "sector_gross_return": sector_gross,
                "index_gross_return": index_gross,
                "sector_residual_return": sector_residual,
                "index_residual_return": index_residual,
                "sector_entry_price": sector_entry,
                "sector_exit_price": sector_exit,
                "index_entry_price": index_entry,
                "index_exit_price": index_exit,
                "eps_surprise_z": event.get("eps_surprise_z", np.nan),
                "revenue_surprise_z": event.get("revenue_surprise_z", np.nan),
                "gap_atr": event.get("gap_atr", np.nan),
                "rel_volume_30m": event.get("rel_volume_30m", np.nan),
                "sector_return_30m": event.get("sector_return_30m", np.nan),
            }
        )
    return pd.DataFrame(rows)


def apply_costs(trades: pd.DataFrame, cost_bps_values: tuple[float, ...]) -> pd.DataFrame:
    if trades.empty:
        return trades
    frames: list[pd.DataFrame] = []
    for cost_bps in cost_bps_values:
        costed = trades.copy()
        costed["cost_bps_round_trip"] = float(cost_bps)
        costed["cost_return_gross"] = float(cost_bps) / 10_000.0
        costed["net_return"] = costed["gross_return"].astype(float) - costed["cost_return_gross"]
        costed["sector_residual_net_return"] = costed["net_return"] - costed["sector_gross_return"]
        costed["index_residual_net_return"] = costed["net_return"] - costed["index_gross_return"]
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
    keys = ["strategy_id", "label", "fold", "split", "horizon", "cost_bps_round_trip"]
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
        gross = group["gross_return"].astype(float)
        sector_residual_net = group["sector_residual_net_return"].astype(float)
        index_residual_net = group["index_residual_net_return"].astype(float)
        for session, value in daily.items():
            daily_rows.append({**key, "session": str(session), "net_return": float(value)})
        for month, value in monthly.items():
            monthly_rows.append({**key, "month": str(month), "net_return": float(value)})
        summary_rows.append(
            {
                **key,
                "trades": int(len(group)),
                "gross_return": float(gross.sum()),
                "total_cost": float(group["cost_return_gross"].sum()),
                "net_return": float(net.sum()),
                "sector_residual_net_return": float(sector_residual_net.sum(skipna=True)),
                "index_residual_net_return": float(index_residual_net.sum(skipna=True)),
                "avg_trade_net": float(net.mean()) if len(net) else 0.0,
                "median_trade_net": float(net.median()) if len(net) else 0.0,
                "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
                "payoff_ratio": _profit_factor(net),
                "daily_sharpe": _daily_sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "sessions": int(len(daily)),
                "sessions_with_trades": int(group["session"].nunique()),
                "symbols": int(group["trade_symbol"].nunique()),
                "active_sectors": int(group["sector_id"].replace("", np.nan).dropna().nunique()) if "sector_id" in group else 0,
                "top5_abs_share": _top_abs_share(group),
            }
        )
        for metric in ("gross_return", "net_return", "sector_residual_net_return", "index_residual_net_return"):
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
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), pd.DataFrame(monthly_rows), pd.DataFrame(distribution_rows)


def build_coverage(events: pd.DataFrame, prices: pd.DataFrame, labeled_events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"scope": "events", "metric": "rows", "value": int(len(events))},
        {"scope": "events", "metric": "tradeable_v1", "value": int(_tradeable_events(events).sum()) if not events.empty else 0},
        {"scope": "panel", "metric": "symbols", "value": int(prices["symbol"].nunique()) if not prices.empty else 0},
        {"scope": "panel", "metric": "symbol_sessions", "value": int(len(prices))},
    ]
    if not labeled_events.empty:
        counts = labeled_events.groupby(["label", "split"], sort=False).size().reset_index(name="value")
        rows.extend({"scope": "labeled_events", "metric": f"{row.label}:{row.split}", "value": int(row.value)} for row in counts.itertuples())
    if not trades.empty:
        counts = trades.groupby(["label", "split", "horizon"], sort=False).size().reset_index(name="value")
        rows.extend({"scope": "trades", "metric": f"{row.label}:{row.split}:{row.horizon}", "value": int(row.value)} for row in counts.itertuples())
    return pd.DataFrame(rows)


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[frame["event_session"].astype(str).isin(sessions)].copy()


def _panel_sessions_for_split(prices: pd.DataFrame, event_sessions: tuple[str, ...]) -> tuple[str, ...]:
    if prices.empty or not event_sessions:
        return tuple(event_sessions)
    months = set(pd.to_datetime(pd.Index(event_sessions)).strftime("%Y-%m"))
    sessions = (
        prices.loc[pd.to_datetime(prices["session"]).dt.strftime("%Y-%m").isin(months), "session"]
        .drop_duplicates()
        .astype(str)
        .sort_values()
        .tolist()
    )
    return tuple(sessions)


def build_screening_events_and_trades(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    config: H3ScreeningConfig,
    folds: tuple[ResearchFold, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], tuple[str, ...]], pd.DataFrame]:
    all_labeled_events: list[pd.DataFrame] = []
    all_trade_bases: list[pd.DataFrame] = []
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    threshold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_events = _split_frame(events, fold.train_sessions)
        rel_volume_threshold = fit_rel_volume_threshold(train_events.loc[_tradeable_events(train_events)], config)
        threshold_rows.append(
            {
                "fold": int(fold.fold),
                "threshold": "rel_volume_30m",
                "value": rel_volume_threshold,
                "fit_split": "train",
                "train_events": int(len(train_events)),
            }
        )

        for split, sessions in (
            ("train", fold.train_sessions),
            ("validation", fold.validation_sessions),
            ("test", fold.test_sessions),
        ):
            panel_sessions = _panel_sessions_for_split(prices, tuple(sessions))
            split_sessions[(fold.fold, split)] = panel_sessions
            split_events = _split_frame(events, sessions)
            labels = {H3_SIGNAL_LABEL: h3_candidate_mask(split_events, config, rel_volume_threshold)}
            if config.controls.get("earnings_beat_only", True):
                labels[EARNINGS_BEAT_LABEL] = earnings_beat_mask(split_events)
            if config.controls.get("gap_only", True):
                labels[GAP_MODERATE_LABEL] = gap_moderate_mask(split_events, config)
            labeled_by_name: dict[str, pd.DataFrame] = {}
            for label, mask in labels.items():
                labeled = make_labeled_events(split_events.loc[mask].copy(), label, fold.fold, split)
                labeled_by_name[label] = labeled
                if not labeled.empty:
                    all_labeled_events.append(labeled)
            reference = labeled_by_name[H3_SIGNAL_LABEL]
            for horizon in config.horizons:
                for label, labeled in labeled_by_name.items():
                    base = simulate_trades(
                        labeled,
                        prices,
                        strategy_id=config.strategy_id,
                        horizon=horizon,
                    )
                    if not base.empty:
                        all_trade_bases.append(base)
                if config.controls.get("sector_proxy_equivalent", True):
                    sector_base = simulate_trades(
                        reference,
                        prices,
                        strategy_id=config.strategy_id,
                        horizon=horizon,
                        trade_symbol_mode="sector_proxy",
                    )
                    if not sector_base.empty:
                        sector_base["label"] = SECTOR_EQUIVALENT_LABEL
                        all_trade_bases.append(sector_base)
                if config.controls.get("random_same_frequency_by_ticker", True):
                    sampled = sample_non_event_controls(
                        reference,
                        prices,
                        events,
                        label=RANDOM_CONTROL_LABEL,
                        fold=fold.fold,
                        split=split,
                        sessions=panel_sessions,
                        horizon=horizon,
                        seed=_stable_seed(config.random_seed, RANDOM_CONTROL_LABEL, fold.fold, split, horizon),
                    )
                    if not sampled.empty:
                        all_labeled_events.append(sampled)
                        base = simulate_trades(sampled, prices, strategy_id=config.strategy_id, horizon=horizon)
                        if not base.empty:
                            all_trade_bases.append(base)
                if config.controls.get("same_hour_by_ticker", True):
                    sampled = sample_non_event_controls(
                        reference,
                        prices,
                        events,
                        label=SAME_HOUR_CONTROL_LABEL,
                        fold=fold.fold,
                        split=split,
                        sessions=panel_sessions,
                        horizon=horizon,
                        seed=_stable_seed(config.random_seed, SAME_HOUR_CONTROL_LABEL, fold.fold, split, horizon),
                    )
                    if not sampled.empty:
                        all_labeled_events.append(sampled)
                        base = simulate_trades(sampled, prices, strategy_id=config.strategy_id, horizon=horizon)
                        if not base.empty:
                            all_trade_bases.append(base)
                if config.controls.get("intraday_momentum_no_event", True):
                    sampled = sample_non_event_controls(
                        reference,
                        prices,
                        events,
                        label=INTRADAY_MOMENTUM_NO_EVENT_LABEL,
                        fold=fold.fold,
                        split=split,
                        sessions=panel_sessions,
                        horizon=horizon,
                        seed=_stable_seed(config.random_seed, INTRADAY_MOMENTUM_NO_EVENT_LABEL, fold.fold, split, horizon),
                        require_opening_momentum=True,
                    )
                    if not sampled.empty:
                        all_labeled_events.append(sampled)
                        base = simulate_trades(sampled, prices, strategy_id=config.strategy_id, horizon=horizon)
                        if not base.empty:
                            all_trade_bases.append(base)

    labeled_events = pd.concat(all_labeled_events, ignore_index=True) if all_labeled_events else pd.DataFrame()
    trade_bases = pd.concat(all_trade_bases, ignore_index=True) if all_trade_bases else pd.DataFrame()
    trades = apply_costs(trade_bases, config.cost_bps_values)
    thresholds = pd.DataFrame(threshold_rows)
    return labeled_events, trades, split_sessions, thresholds


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
        part.groupby(["split", "label", "horizon"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            sector_residual_net_return=("sector_residual_net_return", "sum"),
            index_residual_net_return=("index_residual_net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            win_rate=("win_rate", "mean"),
            max_top5_abs_share=("top5_abs_share", "max"),
        )
    )
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "horizon", "net_return"], ascending=[True, True, False], kind="stable").drop(columns="_split_order")


def _write_manifest(
    path: Path,
    config: H3ScreeningConfig,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: H3ScreeningOutputs,
    thresholds: pd.DataFrame,
) -> None:
    config_file = Path(config_path)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("screening", config.strategy_id, "phase3", config.timeframe),
            "run_type": "h3_phase3_screening",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": config.strategy_id,
            "hypothesis_id": config.hypothesis_id,
            "phase": "Fase 3 - Screening baseline",
            "position": "single_name_long_only",
            "entry_rule": "first_bar_open_at_or_after_10_00_et",
            "exit_horizons": list(config.horizons),
            "labels": [H3_SIGNAL_LABEL, EARNINGS_BEAT_LABEL, GAP_MODERATE_LABEL, *CONTROL_LABELS],
        },
        "data": {
            "earnings_events_path": config.earnings_events_path.as_posix(),
            "earnings_events_fingerprint": fingerprint_path(config.earnings_events_path) if config.earnings_events_path.exists() else "MISSING",
            "intraday_panel_path": config.intraday_panel_path.as_posix(),
            "intraday_panel_fingerprint": fingerprint_path(config.intraday_panel_path) if config.intraday_panel_path.exists() else "MISSING",
            "config_path": config_file.as_posix(),
            "config_fingerprint": fingerprint_path(config_file) if config_file.exists() else "MISSING",
            "timeframe": config.timeframe,
            "split_policy": config.split_policy,
            "n_folds": len(folds),
            "fit_thresholds": thresholds.to_dict(orient="records") if not thresholds.empty else [],
        },
        "parameters": {
            "cost_bps_values": list(config.cost_bps_values),
            "controls": config.controls,
            "signal_filters": config.signal_filters,
        },
        "outputs": {
            "coverage": outputs.coverage_path.as_posix(),
            "events": outputs.events_path.as_posix(),
            "trades": outputs.trades_path.as_posix(),
            "daily": outputs.daily_path.as_posix(),
            "monthly": outputs.monthly_path.as_posix(),
            "summary": outputs.summary_path.as_posix(),
            "distribution": outputs.distribution_path.as_posix(),
            "report": outputs.report_path.as_posix(),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _write_report(path: Path, config: H3ScreeningConfig, coverage: pd.DataFrame, summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    conservative_cost = 10.0 if 10.0 in config.cost_bps_values else float(config.cost_bps_values[-1])
    rollup = _rollup(summary, conservative_cost)
    lines = [
        "# H3 Phase 3 screening",
        "",
        "This is a screening run, not the final H3 strategy runner. It compares the H3 candidate filter against event and timing controls.",
        "",
        "## Coverage",
        "",
        *_markdown_table(coverage, ["scope", "metric", "value"], limit=50),
        "",
        f"## Rollup, Cost={conservative_cost:g} bps Round Trip",
        "",
        *_markdown_table(
            rollup,
            [
                "split",
                "label",
                "horizon",
                "folds",
                "trades",
                "net_return",
                "sector_residual_net_return",
                "index_residual_net_return",
                "avg_trade_net",
                "win_rate",
                "max_top5_abs_share",
            ],
            limit=80,
        ),
        "",
        "## Distribution Contract",
        "",
        "- `distribution.parquet` stores mean, std, min, p05, p25, median, p75, p95 and max for gross, net and residual returns.",
        "- `sector_residual_net_return = net_return - sector_proxy_gross_return`.",
        "- `index_residual_net_return = net_return - SPY_gross_return`.",
        "",
        "## Read",
        "",
        "- H3 only advances if the candidate filter beats earnings-beat-only, gap-only, same-frequency timing controls and sector-equivalent exposure after costs.",
        "- Real interpretation remains blocked until Benzinga Earnings and exclusion tables are populated point-in-time.",
        "",
        "## Config",
        "",
        f"- Strategy id: `{config.strategy_id}`",
        f"- Horizons: `{', '.join(config.horizons)}`",
        f"- Costs bps: `{', '.join(str(value) for value in config.cost_bps_values)}`",
        "",
    ]
    if distribution.empty:
        lines.extend(["## Distribution Preview", "", "No rows."])
    else:
        preview = distribution.loc[
            distribution["cost_bps_round_trip"].eq(conservative_cost)
            & distribution["metric"].eq("net_return")
            & distribution["split"].isin(["validation", "test"])
        ].copy()
        lines.extend(
            [
                "## Distribution Preview",
                "",
                *_markdown_table(preview, ["split", "label", "horizon", "count", "mean", "p05", "median", "p95"], limit=40),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_screening(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> H3ScreeningOutputs:
    config = _with_output_dir(load_config(config_path), output_dir)
    events = load_events(config.earnings_events_path)
    panel = load_panel(config.intraday_panel_path, config)
    prices = build_symbol_session_prices(panel, config)
    folds = build_monthly_folds(events.rename(columns={"event_session": "session"}), config.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")

    root = config.output_dir
    outputs = H3ScreeningOutputs(
        output_dir=root,
        coverage_path=root / "coverage.parquet",
        events_path=root / "events.parquet",
        trades_path=root / "trades.parquet",
        daily_path=root / "daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        distribution_path=root / "distribution.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)

    labeled_events, trades, split_sessions, thresholds = build_screening_events_and_trades(events, prices, config, folds)
    summary, daily, monthly, distribution = aggregate_trades(trades, split_sessions)
    coverage = build_coverage(events, prices, labeled_events, trades)

    coverage.to_parquet(outputs.coverage_path, index=False)
    labeled_events.to_parquet(outputs.events_path, index=False)
    trades.to_parquet(outputs.trades_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    distribution.to_parquet(outputs.distribution_path, index=False)
    _write_manifest(outputs.manifest_path, config, config_path, folds, outputs, thresholds)
    _write_report(outputs.report_path, config, coverage, summary, distribution)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H3 phase 3 earnings continuation screening")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_screening(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

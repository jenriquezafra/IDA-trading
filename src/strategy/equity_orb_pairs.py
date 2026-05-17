from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_orb_pairs_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/equity_orb_pairs/5min")
SPREAD_LABEL = "orb_spread_breakout"
ASSET_BASELINE_LABEL = "orb_asset_baseline"
RANDOM_CONTROL_LABEL = "random_same_frequency_control"
SAME_HOUR_CONTROL_LABEL = "same_hour_control"
MARKET_BETA_CONTROL_LABEL = "market_beta_control"


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    asset_a: str
    asset_b: str
    rationale: str = ""


@dataclass(frozen=True)
class OrbWindow:
    window_id: str
    label: str
    range_bars: int


@dataclass(frozen=True)
class EquityOrbConfig:
    strategy_id: str
    hypothesis_id: str
    timeframe: str
    panel_path: Path
    pairs: tuple[PairSpec, ...]
    windows: tuple[OrbWindow, ...]
    horizons: tuple[int, ...]
    cost_bps_values: tuple[float, ...]
    split_policy: dict[str, Any]
    output_dir: Path
    controls: dict[str, Any]


@dataclass(frozen=True)
class EquityOrbOutputs:
    output_dir: Path
    coverage_path: Path
    events_path: Path
    trades_path: Path
    daily_path: Path
    monthly_path: Path
    summary_path: Path
    manifest_path: Path
    report_path: Path


def _as_tuple_int(values: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    out = tuple(int(value) for value in values)
    if any(value <= 0 for value in out):
        raise ValueError(f"{field_name} values must be positive")
    return out


def _as_tuple_float(values: Any, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    out = tuple(float(value) for value in values)
    if any(value < 0.0 for value in out):
        raise ValueError(f"{field_name} values must be non-negative")
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> EquityOrbConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    data = dict(raw.get("data", {}))
    orb = dict(raw.get("orb", {}))
    exit_cfg = dict(raw.get("exit", {}))
    costs = dict(raw.get("costs", {}))
    outputs = dict(raw.get("outputs", {}))
    pairs_raw = raw.get("pairs", [])
    windows_raw = orb.get("windows", [])
    if not isinstance(pairs_raw, list) or not pairs_raw:
        raise ValueError("pairs must be a non-empty list")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ValueError("orb.windows must be a non-empty list")
    pairs = tuple(
        PairSpec(
            pair_id=str(item["pair_id"]).strip(),
            asset_a=str(item["asset_a"]).strip().upper(),
            asset_b=str(item["asset_b"]).strip().upper(),
            rationale=str(item.get("rationale", "")).strip(),
        )
        for item in pairs_raw
    )
    windows = tuple(
        OrbWindow(
            window_id=str(item["window_id"]).strip(),
            label=str(item.get("label", item["window_id"])).strip(),
            range_bars=int(item["range_bars"]),
        )
        for item in windows_raw
    )
    if any(not pair.pair_id or not pair.asset_a or not pair.asset_b for pair in pairs):
        raise ValueError("each pair requires pair_id, asset_a and asset_b")
    if any(pair.asset_a == pair.asset_b for pair in pairs):
        raise ValueError("pair legs must be different")
    if any(not window.window_id or window.range_bars <= 0 for window in windows):
        raise ValueError("each ORB window requires window_id and positive range_bars")
    return EquityOrbConfig(
        strategy_id=str(raw.get("strategy_id", "")).strip() or "equity_orb_pairs_v1",
        hypothesis_id=str(raw.get("hypothesis_id", "")).strip() or "H2.2",
        timeframe=str(raw.get("timeframe", "5min")).strip(),
        panel_path=Path(data.get("panel_path", "")),
        pairs=pairs,
        windows=windows,
        horizons=_as_tuple_int(exit_cfg.get("horizon_bars"), field_name="exit.horizon_bars"),
        cost_bps_values=_as_tuple_float(costs.get("round_trip_bps_per_leg"), field_name="costs.round_trip_bps_per_leg"),
        split_policy=dict(raw.get("split_policy", {})),
        output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        controls=dict(raw.get("controls", {})),
    )


def required_symbols(config: EquityOrbConfig) -> tuple[str, ...]:
    symbols = {pair.asset_a for pair in config.pairs} | {pair.asset_b for pair in config.pairs} | {"SPY"}
    return tuple(sorted(symbols))


def load_panel(path: str | Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    panel_path = Path(path)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path).sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)
    required = {"timestamp", "session", "bar_index"}
    for symbol in symbols:
        required.update({f"{symbol}__open", f"{symbol}__close"})
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"panel missing required columns: {', '.join(missing)}")
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel["session"] = panel["session"].astype(str)
    panel["bar_index"] = panel["bar_index"].astype(int)
    return panel


def build_coverage(panel: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        open_col = f"{symbol}__open"
        close_col = f"{symbol}__close"
        available_col = f"is_available_{symbol}"
        available = panel[available_col].fillna(False).astype(bool) if available_col in panel else panel[[open_col, close_col]].notna().all(axis=1)
        symbol_frame = panel.loc[available, ["timestamp", "session"]]
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(panel)),
                "available_rows": int(available.sum()),
                "available_ratio": float(available.mean()) if len(available) else 0.0,
                "sessions": int(symbol_frame["session"].nunique()) if not symbol_frame.empty else 0,
                "first_timestamp": symbol_frame["timestamp"].min().isoformat() if not symbol_frame.empty else "",
                "last_timestamp": symbol_frame["timestamp"].max().isoformat() if not symbol_frame.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("symbol", kind="stable").reset_index(drop=True)


def _finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.where(values > 0.0)


def build_pair_frame(panel: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    asset_a = pair.asset_a
    asset_b = pair.asset_b
    out = panel.loc[:, ["timestamp", "session", "bar_index"]].copy()
    for symbol in (asset_a, asset_b, "SPY"):
        out[f"{symbol}_open"] = _finite_positive(panel[f"{symbol}__open"])
        out[f"{symbol}_close"] = _finite_positive(panel[f"{symbol}__close"])
        out[f"{symbol}_open_next"] = out.groupby("session", sort=False)[f"{symbol}_open"].shift(-1)
    out["pair_id"] = pair.pair_id
    out["asset_a"] = asset_a
    out["asset_b"] = asset_b
    out["spread_open"] = np.log(out[f"{asset_a}_open"]) - np.log(out[f"{asset_b}_open"])
    out["spread_close"] = np.log(out[f"{asset_a}_close"]) - np.log(out[f"{asset_b}_close"])
    out["entry_timestamp"] = out.groupby("session", sort=False)["timestamp"].shift(-1)
    return out


def add_orb_range(pair_frame: pd.DataFrame, window: OrbWindow) -> pd.DataFrame:
    range_rows = pair_frame["bar_index"].lt(window.range_bars) & pair_frame["spread_close"].notna()
    ranges = (
        pair_frame.loc[range_rows]
        .groupby("session", sort=False)["spread_close"]
        .agg(orb_high="max", orb_low="min", orb_observations="size")
        .reset_index()
    )
    out = pair_frame.merge(ranges, on="session", how="left", validate="many_to_one")
    out["orb_window"] = window.window_id
    out["orb_window_label"] = window.label
    out["orb_range_bars"] = int(window.range_bars)
    out["orb_width"] = out["orb_high"] - out["orb_low"]
    return out


def spread_directions(frame: pd.DataFrame) -> pd.Series:
    direction = pd.Series(0, index=frame.index, dtype="int8")
    eligible = frame["bar_index"].ge(frame["orb_range_bars"]) & frame["orb_observations"].ge(frame["orb_range_bars"])
    direction.loc[eligible & frame["spread_close"].gt(frame["orb_high"])] = 1
    direction.loc[eligible & frame["spread_close"].lt(frame["orb_low"])] = -1
    return direction


def failed_reversion_directions(frame: pd.DataFrame, value_column: str = "spread_close") -> pd.Series:
    direction = pd.Series(0, index=frame.index, dtype="int8")
    required = {"session", "bar_index", "orb_range_bars", "orb_observations", "orb_high", "orb_low", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing failed ORB columns: {', '.join(sorted(missing))}")
    ordered = frame.sort_values(["session", "bar_index"], kind="stable")
    for _, group in ordered.groupby("session", sort=False):
        active_side = 0
        for idx, row in group.iterrows():
            if int(row["bar_index"]) < int(row["orb_range_bars"]):
                continue
            if int(row["orb_observations"]) < int(row["orb_range_bars"]):
                continue
            value = row[value_column]
            high = row["orb_high"]
            low = row["orb_low"]
            if not np.isfinite(value) or not np.isfinite(high) or not np.isfinite(low):
                continue
            if value > high:
                active_side = 1
                continue
            if value < low:
                active_side = -1
                continue
            if active_side > 0:
                direction.at[idx] = -1
                break
            if active_side < 0:
                direction.at[idx] = 1
                break
    return direction


def asset_directions(panel: pd.DataFrame, symbol: str, window: OrbWindow) -> pd.DataFrame:
    out = panel.loc[:, ["timestamp", "session", "bar_index"]].copy()
    out["asset_symbol"] = symbol
    out["asset_open"] = _finite_positive(panel[f"{symbol}__open"])
    out["asset_close"] = _finite_positive(panel[f"{symbol}__close"])
    out["asset_open_next"] = out.groupby("session", sort=False)["asset_open"].shift(-1)
    range_rows = out["bar_index"].lt(window.range_bars) & out["asset_close"].notna()
    ranges = (
        out.loc[range_rows]
        .groupby("session", sort=False)["asset_close"]
        .agg(orb_high="max", orb_low="min", orb_observations="size")
        .reset_index()
    )
    out = out.merge(ranges, on="session", how="left", validate="many_to_one")
    out["orb_window"] = window.window_id
    out["orb_window_label"] = window.label
    out["orb_range_bars"] = int(window.range_bars)
    out["entry_timestamp"] = out.groupby("session", sort=False)["timestamp"].shift(-1)
    direction = pd.Series(0, index=out.index, dtype="int8")
    eligible = out["bar_index"].ge(window.range_bars) & out["orb_observations"].ge(window.range_bars)
    direction.loc[eligible & out["asset_close"].gt(out["orb_high"])] = 1
    direction.loc[eligible & out["asset_close"].lt(out["orb_low"])] = -1
    out["direction"] = direction
    return out


def failed_asset_directions(panel: pd.DataFrame, symbol: str, window: OrbWindow) -> pd.DataFrame:
    out = asset_directions(panel, symbol, window).drop(columns=["direction"])
    out["direction"] = failed_reversion_directions(out, value_column="asset_close")
    return out


def _pair_valid_mask(frame: pd.DataFrame, horizon: int) -> pd.Series:
    asset_a = str(frame["asset_a"].iloc[0])
    asset_b = str(frame["asset_b"].iloc[0])
    required = [
        f"{asset_a}_open_next",
        f"{asset_b}_open_next",
        f"{asset_a}_open",
        f"{asset_b}_open",
        "entry_timestamp",
    ]
    valid = frame[required].notna().all(axis=1)
    for column in [f"{asset_a}_open", f"{asset_b}_open", "SPY_open"]:
        valid &= frame.groupby("session", sort=False)[column].shift(-(int(horizon) + 1)).notna()
    return valid.fillna(False)


def _asset_valid_mask(frame: pd.DataFrame, horizon: int) -> pd.Series:
    valid = frame[["asset_open_next", "entry_timestamp"]].notna().all(axis=1)
    valid &= frame.groupby("session", sort=False)["asset_open"].shift(-(int(horizon) + 1)).notna()
    return valid.fillna(False)


def simulate_pair_base(
    frame: pd.DataFrame,
    directions: pd.Series,
    *,
    label: str,
    fold: int,
    split: str,
    horizon: int,
    strategy_id: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    asset_a = str(frame["asset_a"].iloc[0])
    asset_b = str(frame["asset_b"].iloc[0])
    prepared = frame.reset_index(drop=True).copy()
    prepared["direction"] = directions.reindex(frame.index).fillna(0).astype("int8").to_numpy()
    valid = _pair_valid_mask(prepared, horizon).to_numpy()
    direction_values = prepared["direction"].to_numpy(dtype=int)
    candidate = valid & (direction_values != 0)
    for symbol in (asset_a, asset_b, "SPY"):
        prepared[f"{symbol}_exit_open"] = prepared.groupby("session", sort=False)[f"{symbol}_open"].shift(-(int(horizon) + 1))
    prepared["exit_timestamp"] = prepared.groupby("session", sort=False)["timestamp"].shift(-(int(horizon) + 1))
    selected = prepared.loc[candidate].groupby("session", sort=False).head(1).copy()
    if selected.empty:
        return pd.DataFrame()
    direction = selected["direction"].astype(int)
    entry_a = selected[f"{asset_a}_open_next"].astype(float)
    entry_b = selected[f"{asset_b}_open_next"].astype(float)
    exit_a = selected[f"{asset_a}_exit_open"].astype(float)
    exit_b = selected[f"{asset_b}_exit_open"].astype(float)
    leg_a_return = np.log(exit_a / entry_a)
    leg_b_return = np.log(exit_b / entry_b)
    spread_return = direction.astype(float) * (leg_a_return - leg_b_return)
    gross_return = 0.5 * spread_return
    return pd.DataFrame(
        {
            "strategy_id": strategy_id,
            "label": label,
            "instrument_type": "pair",
            "fold": int(fold),
            "split": split,
            "pair_id": selected["pair_id"].to_numpy(),
            "asset_a": asset_a,
            "asset_b": asset_b,
            "asset_symbol": "",
            "orb_window": selected["orb_window"].to_numpy(),
            "orb_window_label": selected["orb_window_label"].to_numpy(),
            "orb_range_bars": selected["orb_range_bars"].astype(int).to_numpy(),
            "horizon_bars": int(horizon),
            "session": selected["session"].to_numpy(),
            "signal_timestamp": selected["timestamp"].to_numpy(),
            "entry_timestamp": selected["entry_timestamp"].to_numpy(),
            "exit_timestamp": selected["exit_timestamp"].to_numpy(),
            "bar_index": selected["bar_index"].astype(int).to_numpy(),
            "direction": direction.to_numpy(),
            "side": np.where(direction.gt(0), "long_spread", "short_spread"),
            "asset_a_entry_px": entry_a.to_numpy(),
            "asset_a_exit_px": exit_a.to_numpy(),
            "asset_b_entry_px": entry_b.to_numpy(),
            "asset_b_exit_px": exit_b.to_numpy(),
            "leg_a_return": leg_a_return.to_numpy(),
            "leg_b_return": leg_b_return.to_numpy(),
            "spread_return": spread_return.to_numpy(),
            "gross_return": gross_return.to_numpy(),
            "orb_high": selected["orb_high"].astype(float).to_numpy(),
            "orb_low": selected["orb_low"].astype(float).to_numpy(),
            "orb_width": selected["orb_width"].astype(float).to_numpy(),
            "spread_close": selected["spread_close"].astype(float).to_numpy(),
        }
    )


def simulate_asset_base(
    frame: pd.DataFrame,
    *,
    pair_id: str,
    asset_role: str,
    fold: int,
    split: str,
    horizon: int,
    strategy_id: str,
) -> pd.DataFrame:
    prepared = frame.reset_index(drop=True).copy()
    valid = _asset_valid_mask(prepared, horizon).to_numpy()
    direction_values = prepared["direction"].fillna(0).astype("int8").to_numpy(dtype=int)
    candidate = valid & (direction_values != 0)
    prepared["asset_exit_open"] = prepared.groupby("session", sort=False)["asset_open"].shift(-(int(horizon) + 1))
    prepared["exit_timestamp"] = prepared.groupby("session", sort=False)["timestamp"].shift(-(int(horizon) + 1))
    selected = prepared.loc[candidate].groupby("session", sort=False).head(1).copy()
    if selected.empty:
        return pd.DataFrame()
    direction = selected["direction"].astype(int)
    entry_px = selected["asset_open_next"].astype(float)
    exit_px = selected["asset_exit_open"].astype(float)
    gross_return = direction.astype(float) * np.log(exit_px / entry_px)
    return pd.DataFrame(
        {
            "strategy_id": strategy_id,
            "label": ASSET_BASELINE_LABEL,
            "instrument_type": "single_asset",
            "fold": int(fold),
            "split": split,
            "pair_id": pair_id,
            "asset_a": "",
            "asset_b": "",
            "asset_symbol": selected["asset_symbol"].to_numpy(),
            "asset_role": asset_role,
            "orb_window": selected["orb_window"].to_numpy(),
            "orb_window_label": selected["orb_window_label"].to_numpy(),
            "orb_range_bars": selected["orb_range_bars"].astype(int).to_numpy(),
            "horizon_bars": int(horizon),
            "session": selected["session"].to_numpy(),
            "signal_timestamp": selected["timestamp"].to_numpy(),
            "entry_timestamp": selected["entry_timestamp"].to_numpy(),
            "exit_timestamp": selected["exit_timestamp"].to_numpy(),
            "bar_index": selected["bar_index"].astype(int).to_numpy(),
            "direction": direction.to_numpy(),
            "side": np.where(direction.gt(0), "long", "short"),
            "asset_entry_px": entry_px.to_numpy(),
            "asset_exit_px": exit_px.to_numpy(),
            "gross_return": gross_return.to_numpy(),
            "orb_high": selected["orb_high"].astype(float).to_numpy(),
            "orb_low": selected["orb_low"].astype(float).to_numpy(),
        }
    )


def market_beta_base(candidate: pd.DataFrame) -> pd.DataFrame:
    if candidate.empty:
        return pd.DataFrame()
    out = candidate.copy()
    out["label"] = MARKET_BETA_CONTROL_LABEL
    out["instrument_type"] = "market_beta_control"
    out["asset_symbol"] = "SPY"
    out["asset_entry_px"] = out["SPY_entry_px"]
    out["asset_exit_px"] = out["SPY_exit_px"]
    out["gross_return"] = out["direction"].astype(float) * np.log(out["SPY_exit_px"].astype(float) / out["SPY_entry_px"].astype(float))
    return out


def add_market_prices_to_pair_base(base: pd.DataFrame, frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if base.empty:
        return base
    prepared = frame.reset_index(drop=True).copy()
    prepared["SPY_exit_open"] = prepared.groupby("session", sort=False)["SPY_open"].shift(-(int(horizon) + 1))
    keyed = prepared.loc[:, ["session", "timestamp", "SPY_open_next", "SPY_exit_open"]].rename(
        columns={"timestamp": "signal_timestamp", "SPY_open_next": "SPY_entry_px", "SPY_exit_open": "SPY_exit_px"}
    )
    out = base.merge(keyed, on=["session", "signal_timestamp"], how="left", validate="many_to_one")
    return out


def sample_control_directions(
    frame: pd.DataFrame,
    candidate_base: pd.DataFrame,
    *,
    horizon: int,
    mode: str,
    seed: int,
) -> pd.Series:
    direction = pd.Series(0, index=frame.index, dtype="int8")
    if candidate_base.empty:
        return direction
    valid = _pair_valid_mask(frame.reset_index(drop=True), horizon)
    if mode == "same_hour":
        hours = pd.to_datetime(candidate_base["signal_timestamp"]).dt.hour.unique().tolist()
        valid &= pd.to_datetime(frame["timestamp"]).dt.hour.isin(hours)
    valid_positions = np.flatnonzero(valid.to_numpy())
    if len(valid_positions) == 0:
        return direction
    sample_count = min(int(len(candidate_base)), int(len(valid_positions)))
    rng = np.random.default_rng(seed)
    selected = rng.choice(valid_positions, size=sample_count, replace=False)
    candidate_directions = candidate_base["direction"].astype("int8").to_numpy()
    if len(candidate_directions) >= sample_count:
        assigned = rng.choice(candidate_directions, size=sample_count, replace=False)
    else:
        assigned = rng.choice(candidate_directions, size=sample_count, replace=True)
    direction.iloc[selected] = assigned
    return direction


def apply_costs(base: pd.DataFrame, cost_bps_values: tuple[float, ...]) -> pd.DataFrame:
    if base.empty:
        return base
    frames: list[pd.DataFrame] = []
    for cost_bps in cost_bps_values:
        costed = base.copy()
        costed["cost_bps_per_leg_round_trip"] = float(cost_bps)
        costed["cost_return_gross"] = float(cost_bps) / 10_000.0
        costed["net_return"] = costed["gross_return"].astype(float) - costed["cost_return_gross"]
        frames.append(costed)
    return pd.concat(frames, ignore_index=True)


def _profit_factor(values: pd.Series) -> float:
    profit = float(values[values > 0.0].sum())
    loss = float(-values[values < 0.0].sum())
    if loss == 0.0:
        return np.inf if profit > 0.0 else 0.0
    return profit / loss


def _daily_sharpe(daily: pd.Series) -> float:
    if len(daily) < 2:
        return 0.0
    std = float(daily.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(252.0) * daily.mean() / std)


def _max_drawdown(values: pd.Series) -> float:
    equity = values.fillna(0.0).cumsum()
    if equity.empty:
        return 0.0
    return float((equity.cummax() - equity).max())


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["session"].astype(str).isin(sessions)].copy()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000_000


def aggregate_trades(trades: pd.DataFrame, split_sessions: dict[tuple[int, str], tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = [
        "strategy_id",
        "label",
        "instrument_type",
        "pair_id",
        "asset_symbol",
        "asset_role",
        "orb_window",
        "orb_range_bars",
        "fold",
        "split",
        "horizon_bars",
        "cost_bps_per_leg_round_trip",
        "range_quality_filter",
        "range_quality_label",
    ]
    for key in keys:
        if key not in trades:
            trades[key] = ""
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    grouped = trades.groupby(keys, sort=False, dropna=False)
    for key_values, group in grouped:
        key = dict(zip(keys, key_values, strict=True))
        sessions = split_sessions.get((int(key["fold"]), str(key["split"])), tuple(sorted(group["session"].astype(str).unique())))
        daily = pd.Series(0.0, index=pd.Index(sessions, name="session"))
        daily = daily.add(group.groupby("session")["net_return"].sum(), fill_value=0.0).sort_index()
        monthly = daily.groupby(pd.to_datetime(daily.index).strftime("%Y-%m")).sum()
        for session, value in daily.items():
            daily_rows.append({**key, "session": str(session), "net_return": float(value)})
        for month, value in monthly.items():
            monthly_rows.append({**key, "month": str(month), "net_return": float(value)})
        net = group["net_return"].astype(float)
        gross = group["gross_return"].astype(float)
        summary_rows.append(
            {
                **key,
                "trades": int(len(group)),
                "gross_return": float(gross.sum()),
                "total_cost": float(group["cost_return_gross"].sum()),
                "net_return": float(net.sum()),
                "avg_trade_net": float(net.mean()) if len(net) else 0.0,
                "profit_factor": _profit_factor(net),
                "daily_sharpe": _daily_sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
                "sessions": int(len(daily)),
                "sessions_with_trades": int(group["session"].nunique()),
                "top5_abs_share": _top_abs_share(group),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), pd.DataFrame(monthly_rows)


def _top_abs_share(group: pd.DataFrame, top_n: int = 5) -> float:
    by_session = group.groupby("session")["net_return"].sum()
    denom = float(by_session.abs().sum())
    if denom <= 0.0:
        return 0.0
    return float(by_session.abs().sort_values(ascending=False).head(top_n).sum() / denom)


def build_events(pair_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in pair_frames:
        directions = spread_directions(frame)
        events = frame.loc[directions.ne(0), ["pair_id", "asset_a", "asset_b", "orb_window", "orb_window_label", "orb_range_bars", "session", "timestamp", "bar_index", "spread_close", "orb_high", "orb_low", "orb_width"]].copy()
        events["direction"] = directions.loc[events.index].to_numpy(dtype=int)
        events["side"] = np.where(events["direction"].gt(0), "long_spread", "short_spread")
        rows.append(events)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_strategy(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> EquityOrbOutputs:
    config = load_config(config_path)
    if output_dir is not None:
        config = EquityOrbConfig(
            strategy_id=config.strategy_id,
            hypothesis_id=config.hypothesis_id,
            timeframe=config.timeframe,
            panel_path=config.panel_path,
            pairs=config.pairs,
            windows=config.windows,
            horizons=config.horizons,
            cost_bps_values=config.cost_bps_values,
            split_policy=config.split_policy,
            output_dir=Path(output_dir),
            controls=config.controls,
        )
    symbols = required_symbols(config)
    panel = load_panel(config.panel_path, symbols)
    coverage = build_coverage(panel, symbols)
    folds = build_monthly_folds(panel, config.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")

    root = config.output_dir
    outputs = EquityOrbOutputs(
        output_dir=root,
        coverage_path=root / "coverage.parquet",
        events_path=root / "events.parquet",
        trades_path=root / "trades.parquet",
        daily_path=root / "daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)

    all_trades: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    random_seed = int(config.controls.get("random_seed", 2202))

    pair_window_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for pair in config.pairs:
        base_pair = build_pair_frame(panel, pair)
        for window in config.windows:
            frame = add_orb_range(base_pair, window)
            pair_window_frames[(pair.pair_id, window.window_id)] = frame
            event_frames.append(frame)
    asset_symbols = tuple(sorted({pair.asset_a for pair in config.pairs} | {pair.asset_b for pair in config.pairs}))
    asset_window_frames: dict[tuple[str, str], pd.DataFrame] = {
        (asset, window.window_id): asset_directions(panel, asset, window)
        for asset in asset_symbols
        for window in config.windows
    }

    for fold in folds:
        for split, sessions in (
            ("train", fold.train_sessions),
            ("validation", fold.validation_sessions),
            ("test", fold.test_sessions),
        ):
            split_sessions[(fold.fold, split)] = tuple(sessions)
            for pair in config.pairs:
                for window in config.windows:
                    full_pair_frame = pair_window_frames[(pair.pair_id, window.window_id)]
                    pair_frame = _split_frame(full_pair_frame, sessions)
                    pair_direction = spread_directions(pair_frame)
                    for horizon in config.horizons:
                        candidate_base = simulate_pair_base(
                            pair_frame,
                            pair_direction,
                            label=SPREAD_LABEL,
                            fold=fold.fold,
                            split=split,
                            horizon=horizon,
                            strategy_id=config.strategy_id,
                        )
                        if not candidate_base.empty:
                            all_trades.append(apply_costs(candidate_base, config.cost_bps_values))
                        if config.controls.get("random_same_frequency", True) and not candidate_base.empty:
                            random_direction = sample_control_directions(
                                pair_frame,
                                candidate_base,
                                horizon=horizon,
                                mode="random",
                                seed=_stable_seed(random_seed, "random", fold.fold, horizon, pair.pair_id, window.window_id),
                            )
                            random_base = simulate_pair_base(
                                pair_frame,
                                random_direction,
                                label=RANDOM_CONTROL_LABEL,
                                fold=fold.fold,
                                split=split,
                                horizon=horizon,
                                strategy_id=config.strategy_id,
                            )
                            if not random_base.empty:
                                all_trades.append(apply_costs(random_base, config.cost_bps_values))
                        if config.controls.get("same_hour", True) and not candidate_base.empty:
                            same_hour_direction = sample_control_directions(
                                pair_frame,
                                candidate_base,
                                horizon=horizon,
                                mode="same_hour",
                                seed=_stable_seed(random_seed, "same_hour", fold.fold, horizon, pair.pair_id, window.window_id),
                            )
                            same_hour_base = simulate_pair_base(
                                pair_frame,
                                same_hour_direction,
                                label=SAME_HOUR_CONTROL_LABEL,
                                fold=fold.fold,
                                split=split,
                                horizon=horizon,
                                strategy_id=config.strategy_id,
                            )
                            if not same_hour_base.empty:
                                all_trades.append(apply_costs(same_hour_base, config.cost_bps_values))
                        if config.controls.get("market_beta", True) and not candidate_base.empty:
                            market_ready = add_market_prices_to_pair_base(candidate_base, pair_frame, horizon)
                            market_base = market_beta_base(market_ready.dropna(subset=["SPY_entry_px", "SPY_exit_px"]))
                            if not market_base.empty:
                                all_trades.append(apply_costs(market_base, config.cost_bps_values))
                        if config.controls.get("directional_orb_baseline", True):
                            for asset_role, asset in (("asset_a", pair.asset_a), ("asset_b", pair.asset_b)):
                                asset_frame = _split_frame(asset_window_frames[(asset, window.window_id)], sessions)
                                asset_base = simulate_asset_base(
                                    asset_frame,
                                    pair_id=pair.pair_id,
                                    asset_role=asset_role,
                                    fold=fold.fold,
                                    split=split,
                                    horizon=horizon,
                                    strategy_id=config.strategy_id,
                                )
                                if not asset_base.empty:
                                    all_trades.append(apply_costs(asset_base, config.cost_bps_values))

    events = build_events(event_frames)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary, daily, monthly = aggregate_trades(trades, split_sessions)
    coverage.to_parquet(outputs.coverage_path, index=False)
    events.to_parquet(outputs.events_path, index=False)
    trades.to_parquet(outputs.trades_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    _write_manifest(outputs.manifest_path, config, config_path, folds, outputs, symbols)
    _write_report(outputs.report_path, config, coverage, summary)
    return outputs


def _write_manifest(
    path: Path,
    config: EquityOrbConfig,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: EquityOrbOutputs,
    symbols: tuple[str, ...],
) -> None:
    config_file = Path(config_path)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("strategy", config.strategy_id, "PAIRS", config.timeframe),
            "run_type": "strategy_backtest",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": config.strategy_id,
            "hypothesis_id": config.hypothesis_id,
            "entry_rule": "next_open",
            "exit_rule": "fixed_horizon_open",
            "position": "dollar_neutral_pair",
            "spread": "log(asset_a) - log(asset_b)",
            "trade_policy": "first_breakout_per_session",
        },
        "data": {
            "panel_path": config.panel_path.as_posix(),
            "panel_fingerprint": fingerprint_path(config.panel_path) if config.panel_path.exists() else "MISSING",
            "config_path": config_file.as_posix(),
            "config_fingerprint": fingerprint_path(config_file) if config_file.exists() else "MISSING",
            "symbols": list(symbols),
            "timeframe": config.timeframe,
            "split_policy": config.split_policy,
            "n_folds": len(folds),
        },
        "parameters": {
            "pairs": [pair.__dict__ for pair in config.pairs],
            "orb_windows": [window.__dict__ for window in config.windows],
            "horizons": list(config.horizons),
            "cost_bps_values": list(config.cost_bps_values),
            "controls": config.controls,
        },
        "outputs": {
            "coverage": outputs.coverage_path.as_posix(),
            "events": outputs.events_path.as_posix(),
            "trades": outputs.trades_path.as_posix(),
            "daily": outputs.daily_path.as_posix(),
            "monthly": outputs.monthly_path.as_posix(),
            "summary": outputs.summary_path.as_posix(),
            "report": outputs.report_path.as_posix(),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


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


def _candidate_rollup(summary: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    candidate = summary[
        summary["label"].eq(SPREAD_LABEL)
        & summary["cost_bps_per_leg_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame()
    grouped = (
        candidate.groupby(["split", "pair_id", "orb_window", "horizon_bars"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            max_top5_abs_share=("top5_abs_share", "max"),
        )
    )
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "net_return"], ascending=[True, False], kind="stable").drop(columns="_split_order")


def _control_rollup(summary: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    part = summary[
        summary["cost_bps_per_leg_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if part.empty:
        return pd.DataFrame()
    grouped = (
        part.groupby(["split", "label"], as_index=False)
        .agg(
            groups=("pair_id", "count"),
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            profit_factor=("profit_factor", "mean"),
        )
    )
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "net_return"], ascending=[True, False], kind="stable").drop(columns="_split_order")


def _write_report(path: Path, config: EquityOrbConfig, coverage: pd.DataFrame, summary: pd.DataFrame) -> None:
    candidate = _candidate_rollup(summary, cost_bps=2.0)
    controls = _control_rollup(summary, cost_bps=2.0)
    validation_top = candidate[candidate["split"].eq("validation")].sort_values("net_return", ascending=False, kind="stable")
    test_top = candidate[candidate["split"].eq("test")].sort_values("net_return", ascending=False, kind="stable")
    best_validation = float(validation_top["net_return"].max()) if not validation_top.empty else np.nan
    best_test = float(test_top["net_return"].max()) if not test_top.empty else np.nan
    if np.isfinite(best_validation) and np.isfinite(best_test) and best_validation < 0.0 and best_test < 0.0:
        decision = "Initial read: H2.2 continuation ORB is rejected at this cost level; every pair/window/horizon is negative in validation and test."
    elif np.isfinite(best_validation) and best_validation > 0.0:
        decision = "Initial read: H2.2 has at least one positive validation pocket; inspect controls and fold concentration before adding filters."
    else:
        decision = "Initial read: H2.2 needs manual review; candidate summary is incomplete or mixed."
    lines = [
        "# Equity ORB pairs diagnostic",
        "",
        "This is the first H2.2 diagnostic: ORB on relative log spreads with dollar-neutral pair returns.",
        "Signals use spread close outside the opening range, enter on next open, and exit on fixed horizons.",
        "",
        "## Initial Decision",
        "",
        f"- {decision}",
        f"- Best validation net return at 2 bps: `{best_validation:.4f}`" if np.isfinite(best_validation) else "- Best validation net return at 2 bps: `n/a`",
        f"- Best test net return at 2 bps: `{best_test:.4f}`" if np.isfinite(best_test) else "- Best test net return at 2 bps: `n/a`",
        "",
        "## Data Coverage",
        "",
        *_markdown_table(coverage, ["symbol", "available_ratio", "sessions", "first_timestamp", "last_timestamp"], limit=30),
        "",
        "## Candidate Rollup, Cost=2 bps Per Leg Round Trip",
        "",
        *_markdown_table(candidate, ["split", "pair_id", "orb_window", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "max_top5_abs_share"], limit=40),
        "",
        "## Validation Top Candidates",
        "",
        *_markdown_table(validation_top, ["pair_id", "orb_window", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "max_top5_abs_share"], limit=20),
        "",
        "## Test Top Candidates",
        "",
        *_markdown_table(test_top, ["pair_id", "orb_window", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "max_top5_abs_share"], limit=20),
        "",
        "## Control Rollup, Cost=2 bps Per Leg Round Trip",
        "",
        *_markdown_table(controls, ["split", "label", "groups", "trades", "net_return", "avg_trade_net", "profit_factor"], limit=40),
        "",
        "## Read",
        "",
        "- Treat this as a screening run. H2.2 is only interesting if spread ORB beats directional ORB and timing controls after costs.",
        "- If validation looks promising, inspect fold-level concentration before adding H2.1/H2.3 filters.",
        "- If validation does not beat controls, reject or pivot to failed ORB before considering options.",
        "",
        "## Config",
        "",
        f"- Strategy id: `{config.strategy_id}`",
        f"- Pairs: `{', '.join(pair.pair_id for pair in config.pairs)}`",
        f"- Horizons: `{', '.join(str(value) for value in config.horizons)}`",
        f"- ORB windows: `{', '.join(window.window_id for window in config.windows)}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H2.2 equity ORB pairs diagnostic")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_strategy(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

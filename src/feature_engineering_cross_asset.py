from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.cross_asset_data import aligned_panel_path, resolve_symbols


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
TARGET_HELPER_FIELDS = [
    "target_open_next",
    "next_open_timestamp",
    "target_crosses_session_close",
    "can_open_trade",
    "force_flat_bar",
    "trade_could_remain_open_past_close",
]


@dataclass(frozen=True)
class FeatureBuildReport:
    target_symbol: str
    feature_set_version: str
    rows: int
    columns: int
    hmm_feature_columns: int
    missing_hmm_feature_columns: list[str]
    output_path: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _symbol_col(symbol: str, field: str) -> str:
    return f"{symbol.upper()}__{field}"


def _ret_col(symbol: str, window: int) -> str:
    return f"ret_{symbol.upper()}_{int(window)}"


def _range_col(symbol: str) -> str:
    return f"range_{symbol.upper()}"


def _range_ratio_col(symbol: str, short_window: int, long_window: int) -> str:
    return f"range_ratio_{symbol.upper()}_{short_window}_{long_window}"


def _range_ratio_windows(feature_cfg: dict[str, Any], default_short: int = 6, default_long: int = 24) -> list[tuple[int, int]]:
    configured = feature_cfg.get("range_ratio_windows")
    pairs: list[tuple[int, int]] = []
    if configured:
        for item in configured:
            if isinstance(item, dict):
                short = int(item.get("short", default_short))
                long = int(item.get("long", default_long))
            else:
                short, long = [int(value) for value in item]
            pairs.append((short, long))
    else:
        pairs.append(
            (
                int(feature_cfg.get("range_ratio_short_window", default_short)),
                int(feature_cfg.get("range_ratio_long_window", default_long)),
            )
        )

    seen: set[tuple[int, int]] = set()
    output: list[tuple[int, int]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            output.append(pair)
    return output


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > eps)


def _grouped_shift(series: pd.Series, sessions: pd.Series, periods: int) -> pd.Series:
    return series.groupby(sessions, sort=False).shift(periods)


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


def _grouped_rolling_std(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _grouped_rolling_min(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .min()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _grouped_rolling_max(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _row_mean(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in features.columns]
    if not available:
        return pd.Series(np.nan, index=features.index)
    return features[available].mean(axis=1)


def _row_std(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in features.columns]
    if len(available) < 2:
        return pd.Series(np.nan, index=features.index)
    return features[available].std(axis=1)


def _row_count_positive(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in features.columns]
    if not available:
        return pd.Series(np.nan, index=features.index)
    return (features[available] > 0.0).sum(axis=1)


def _group_symbols(feature_cfg: dict[str, Any], group_name: str, symbols: list[str]) -> list[str]:
    configured = [str(symbol).upper() for symbol in feature_cfg.get("groups", {}).get(group_name, [])]
    available = set(symbols)
    return [symbol for symbol in configured if symbol in available]


def _mean_ret(features: pd.DataFrame, symbols: list[str], window: int) -> pd.Series:
    return _row_mean(features, [_ret_col(symbol, window) for symbol in symbols])


def _with_columns(base: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    if not columns:
        return base
    return pd.concat([base, pd.DataFrame(columns, index=base.index)], axis=1)


def _expanding_mean_by_bar_id(values: pd.Series, sessions: pd.Series, bar_index: pd.Series, min_periods: int = 1) -> pd.Series:
    session_order = pd.Series(pd.factorize(sessions, sort=False)[0], index=values.index)
    frame = pd.DataFrame(
        {
            "value": values.astype(float),
            "session_order": session_order,
            "bar_index": bar_index.astype(int),
            "_row": np.arange(len(values)),
        },
        index=values.index,
    ).sort_values(["bar_index", "session_order", "_row"], kind="stable")
    prior_mean = (
        frame.groupby("bar_index", sort=False)["value"]
        .expanding(min_periods=max(int(min_periods), 1))
        .mean()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    frame["prior_mean"] = prior_mean.to_numpy()
    return frame.sort_values("_row", kind="stable")["prior_mean"].reset_index(drop=True)


def _consecutive_count(flags: pd.Series, sessions: pd.Series) -> pd.Series:
    result = pd.Series(0, index=flags.index, dtype="int64")
    clean_flags = flags.fillna(False).astype(bool)
    for _, index_values in clean_flags.groupby(sessions, sort=False).groups.items():
        run = 0
        for idx in index_values:
            run = run + 1 if bool(clean_flags.loc[idx]) else 0
            result.loc[idx] = run
    return result


def _previous_session_close(close: pd.Series, sessions: pd.Series) -> pd.Series:
    session_close = close.groupby(sessions, sort=False).last()
    previous_by_session = session_close.shift(1)
    return sessions.map(previous_by_session).astype(float)


def _compute_vwap(panel: pd.DataFrame, symbol: str) -> pd.Series:
    high = panel[_symbol_col(symbol, "high")]
    low = panel[_symbol_col(symbol, "low")]
    close = panel[_symbol_col(symbol, "close")]
    volume = panel[_symbol_col(symbol, "volume")]
    typical = (high + low + close) / 3.0
    dollar_volume = typical * volume
    cumulative_dollar = dollar_volume.groupby(panel["session"], sort=False).cumsum()
    cumulative_volume = volume.groupby(panel["session"], sort=False).cumsum()
    return cumulative_dollar / cumulative_volume.replace(0, np.nan)


def add_per_symbol_features(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    symbols: list[str],
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    featured = features.copy()
    sessions = panel["session"]
    return_windows = [int(window) for window in feature_cfg.get("return_windows", [1, 3, 6, 12, 24])]
    ratio_windows = _range_ratio_windows(feature_cfg)

    new_columns: dict[str, Any] = {}
    for symbol in symbols:
        close_col = _symbol_col(symbol, "close")
        high_col = _symbol_col(symbol, "high")
        low_col = _symbol_col(symbol, "low")
        if close_col not in panel:
            continue

        log_close = np.log(panel[close_col])
        for window in return_windows:
            new_columns[_ret_col(symbol, window)] = log_close - _grouped_shift(log_close, sessions, window)

        session_open = panel[_symbol_col(symbol, "open")].groupby(sessions, sort=False).transform("first")
        new_columns[f"dist_open_{symbol}"] = np.log(panel[close_col] / session_open)

        range_col = _range_col(symbol)
        symbol_range = np.log(panel[high_col] / panel[low_col])
        new_columns[range_col] = symbol_range
        for short_window, long_window in ratio_windows:
            range_short = _grouped_rolling_mean(symbol_range, sessions, short_window)
            range_long = _grouped_rolling_mean(symbol_range, sessions, long_window)
            new_columns[_range_ratio_col(symbol, short_window, long_window)] = _safe_ratio(range_short, range_long)
        ret_12 = new_columns.get(_ret_col(symbol, 12))
        new_columns[f"absret_{symbol}_12"] = ret_12.abs() if ret_12 is not None else np.nan
        vwap = _compute_vwap(panel, symbol)
        new_columns[f"vwap_{symbol}"] = vwap
        new_columns[f"above_vwap_{symbol}"] = panel[close_col] > vwap

    return _with_columns(featured, new_columns)


def add_target_features(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    target_symbol: str,
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    featured = features.copy()
    target = target_symbol.upper()
    sessions = panel["session"]
    efficiency_windows = [int(value) for value in feature_cfg.get("efficiency_windows", [feature_cfg.get("efficiency_window", 12)])]
    vwap_slope_windows = [int(value) for value in feature_cfg.get("vwap_slope_windows", [feature_cfg.get("vwap_slope_window", 12)])]
    atr_windows = [int(value) for value in feature_cfg.get("atr_windows", [12])]
    ratio_windows = _range_ratio_windows(feature_cfg)

    new_columns: dict[str, Any] = {}
    for window in feature_cfg.get("return_windows", [1, 3, 6, 12, 24]):
        source = _ret_col(target, int(window))
        if source in featured:
            new_columns[f"target_ret_{int(window)}"] = featured[source]

    target_ret_1 = new_columns["target_ret_1"]
    for efficiency_window in efficiency_windows:
        sum_ret = _grouped_rolling_sum(target_ret_1, sessions, efficiency_window)
        sum_abs_ret = _grouped_rolling_sum(target_ret_1.abs(), sessions, efficiency_window)
        new_columns[f"target_signed_efficiency_{efficiency_window}"] = _safe_ratio(sum_ret, sum_abs_ret)
        new_columns[f"target_dir_persistence_{efficiency_window}"] = _grouped_rolling_mean(np.sign(target_ret_1), sessions, efficiency_window)

    target_open = panel[_symbol_col(target, "open")]
    target_high = panel[_symbol_col(target, "high")]
    target_low = panel[_symbol_col(target, "low")]
    target_close = panel[_symbol_col(target, "close")]
    session_open = target_open.groupby(sessions, sort=False).transform("first")
    high_so_far = target_high.groupby(sessions, sort=False).cummax()
    low_so_far = target_low.groupby(sessions, sort=False).cummin()
    session_range = high_so_far - low_so_far

    previous_close = _grouped_shift(target_close, sessions, 1)
    true_range = pd.concat(
        [
            target_high - target_low,
            (target_high - previous_close).abs(),
            (target_low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    new_columns["target_true_range_pct"] = true_range / target_close
    for atr_window in atr_windows:
        new_columns[f"target_atr_{atr_window}"] = _grouped_rolling_mean(true_range / target_close, sessions, atr_window)
    atr_12 = pd.Series(new_columns.get("target_atr_12", new_columns[f"target_atr_{atr_windows[0]}"]), index=panel.index)
    for short_window, long_window in ratio_windows:
        source = _range_ratio_col(target, short_window, long_window)
        if source in featured:
            new_columns[f"target_range_ratio_{short_window}_{long_window}"] = featured[source]
    new_columns["target_dist_open"] = np.log(target_close / session_open)
    new_columns["target_pos_session_range"] = _safe_ratio(target_close - low_so_far, session_range)
    new_columns["target_dist_vwap_atr"] = _safe_ratio(np.log(target_close / featured[f"vwap_{target}"]), atr_12)
    for atr_window in atr_windows:
        atr = pd.Series(new_columns[f"target_atr_{atr_window}"], index=panel.index)
        new_columns[f"target_dist_vwap_atr_{atr_window}"] = _safe_ratio(np.log(target_close / featured[f"vwap_{target}"]), atr)
    for vwap_slope_window in vwap_slope_windows:
        new_columns[f"target_vwap_slope_{vwap_slope_window}"] = _safe_ratio(
            np.log(featured[f"vwap_{target}"] / _grouped_shift(featured[f"vwap_{target}"], sessions, vwap_slope_window)),
            atr_12,
        )
    new_columns["target_intraday_runup"] = _safe_ratio(np.log(target_close / low_so_far), atr_12)
    new_columns["target_intraday_drawdown"] = np.log(target_close / target_close.groupby(sessions, sort=False).cummax())
    return _with_columns(featured, new_columns)


def add_target_setup_features(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    target_symbol: str,
    feature_cfg: dict[str, Any],
) -> pd.DataFrame:
    featured = features.copy()
    target = target_symbol.upper()
    sessions = panel["session"]
    bar_index = panel["bar_index"].astype(int)
    frequency = pd.Timedelta(feature_cfg.get("frequency", "5min"))
    setup_cfg = feature_cfg.get("setup_features", {})
    expected_min_periods = int(setup_cfg.get("expected_by_bar_min_periods", 20))
    opening_range_bars = [int(value) for value in setup_cfg.get("opening_range_bars", [3, 6, 12])]
    breakout_windows = [int(value) for value in setup_cfg.get("rolling_breakout_windows", [12, 24])]
    rv_windows = [int(value) for value in setup_cfg.get("realized_vol_windows", [3, 6, 12, 24])]
    breakout_persistence_windows = [int(value) for value in setup_cfg.get("breakout_persistence_windows", [2, 3])]
    volume_accel_windows = [int(value) for value in setup_cfg.get("volume_accel_windows", [2, 4])]

    target_open = panel[_symbol_col(target, "open")].astype(float)
    target_high = panel[_symbol_col(target, "high")].astype(float)
    target_low = panel[_symbol_col(target, "low")].astype(float)
    target_close = panel[_symbol_col(target, "close")].astype(float)
    target_volume = panel[_symbol_col(target, "volume")].astype(float)
    session_open = target_open.groupby(sessions, sort=False).transform("first")
    previous_close = _previous_session_close(target_close, sessions)
    previous_high_by_session = target_high.groupby(sessions, sort=False).max().shift(1)
    previous_low_by_session = target_low.groupby(sessions, sort=False).min().shift(1)
    previous_session_high = sessions.map(previous_high_by_session).astype(float)
    previous_session_low = sessions.map(previous_low_by_session).astype(float)
    previous_session_range = previous_session_high - previous_session_low
    high_so_far = target_high.groupby(sessions, sort=False).cummax()
    low_so_far = target_low.groupby(sessions, sort=False).cummin()
    bars_in_session = bar_index.groupby(sessions, sort=False).transform("size")
    minutes_per_bar = frequency / pd.Timedelta(minutes=1)

    bar_range = (target_high - target_low).replace(0.0, np.nan)
    close_location_value = _safe_ratio(target_close - target_low, bar_range)
    clv = _safe_ratio((2.0 * target_close) - target_high - target_low, bar_range)
    body = (target_close - target_open).abs()
    upper_wick = target_high - pd.concat([target_open, target_close], axis=1).max(axis=1)
    lower_wick = pd.concat([target_open, target_close], axis=1).min(axis=1) - target_low
    ret_1 = featured.get("target_ret_1", np.log(target_close / _grouped_shift(target_close, sessions, 1)))
    cumulative_volume = target_volume.groupby(sessions, sort=False).cumsum()

    expected_volume = _expanding_mean_by_bar_id(target_volume, sessions, bar_index, expected_min_periods)
    expected_cum_volume = _expanding_mean_by_bar_id(cumulative_volume, sessions, bar_index, expected_min_periods)
    rel_volume = _safe_ratio(target_volume.reset_index(drop=True), expected_volume)
    rel_cum_volume = _safe_ratio(cumulative_volume.reset_index(drop=True), expected_cum_volume)
    minutes_from_open = bar_index * float(minutes_per_bar)
    minutes_to_close = (bars_in_session - 1 - bar_index) * float(minutes_per_bar)

    gap = np.log(session_open / previous_close)
    gap_denom = (session_open - previous_close).abs().replace(0.0, np.nan)
    gap_fill_progress = pd.Series(
        np.select(
            [gap > 0.0, gap < 0.0],
            [(session_open - target_close) / gap_denom, (target_close - session_open) / gap_denom],
            default=np.nan,
        ),
        index=panel.index,
    )

    new_columns: dict[str, Any] = {
        "target_minutes_from_open": minutes_from_open,
        "target_minutes_to_close": minutes_to_close,
        "target_first_30m": minutes_from_open < 30.0,
        "target_first_60m": minutes_from_open < 60.0,
        "target_last_60m": minutes_to_close < 60.0,
        "target_lunch": panel["timestamp"].dt.strftime("%H:%M").between(
            str(setup_cfg.get("lunch_start", "12:00")),
            str(setup_cfg.get("lunch_end", "14:00")),
            inclusive="left",
        ),
        "target_prev_session_close": previous_close,
        "target_prev_session_high": previous_session_high,
        "target_prev_session_low": previous_session_low,
        "target_overnight_ret": gap,
        "target_abs_overnight_ret": gap.abs(),
        "target_gap_fill_progress": gap_fill_progress,
        "target_pos_prev_session_range": _safe_ratio(target_close - previous_session_low, previous_session_range),
        "target_close_location_bar": close_location_value,
        "target_clv": clv,
        "target_bar_efficiency": _safe_ratio(body, bar_range),
        "target_upper_wick_ratio": _safe_ratio(upper_wick, bar_range),
        "target_lower_wick_ratio": _safe_ratio(lower_wick, bar_range),
        "target_rel_volume_by_bar": rel_volume,
        "target_rel_cum_volume_by_bar": rel_cum_volume,
        "target_signed_rel_volume": np.sign(target_close - target_open) * rel_volume,
        "target_clv_rel_volume": clv.reset_index(drop=True) * rel_volume,
        "target_effort_vs_result": rel_volume / _safe_ratio(body.reset_index(drop=True), bar_range.reset_index(drop=True)).replace(0.0, np.nan),
        "target_consecutive_up_bars": _consecutive_count(target_close > target_open, sessions),
        "target_consecutive_down_bars": _consecutive_count(target_close < target_open, sessions),
        "target_new_session_high": target_high >= high_so_far,
        "target_new_session_low": target_low <= low_so_far,
    }

    setup_atr_column = str(setup_cfg.get("atr_column", "target_atr_12"))
    setup_atr = featured.get(setup_atr_column, featured.get("target_atr_12", pd.Series(np.nan, index=panel.index))).astype(float)
    new_columns["target_dist_prev_close_atr"] = _safe_ratio(np.log(target_close / previous_close), setup_atr)
    new_columns["target_dist_prev_high_atr"] = _safe_ratio(np.log(target_close / previous_session_high), setup_atr)
    new_columns["target_dist_prev_low_atr"] = _safe_ratio(np.log(target_close / previous_session_low), setup_atr)

    rel_volume_series = pd.Series(rel_volume, index=panel.index)
    for window in volume_accel_windows:
        prior_mean = _grouped_shift(_grouped_rolling_mean(rel_volume_series, sessions, window), sessions, 1)
        new_columns[f"target_rel_volume_accel_{window}"] = _safe_ratio(rel_volume_series, prior_mean)

    squared_ret = ret_1.astype(float).pow(2)
    for window in rv_windows:
        rv = np.sqrt(_grouped_rolling_sum(squared_ret, sessions, window))
        expected_rv = _expanding_mean_by_bar_id(rv, sessions, bar_index, expected_min_periods)
        new_columns[f"target_rv_{window}"] = rv
        new_columns[f"target_rv_{window}_rel_by_bar"] = _safe_ratio(rv.reset_index(drop=True), expected_rv)
    if "target_rv_12" in new_columns:
        rv_12 = pd.Series(new_columns["target_rv_12"], index=panel.index)
        new_columns["target_absret_12_to_rv_12"] = _safe_ratio(featured["target_ret_12"].abs(), rv_12)
        new_columns["target_vol_of_vol_12"] = rv_12 - _grouped_shift(rv_12, sessions, 12)
    if "target_rv_6" in new_columns and "target_rv_24" in new_columns:
        new_columns["target_rv_ratio_6_24"] = _safe_ratio(
            pd.Series(new_columns["target_rv_6"], index=panel.index),
            pd.Series(new_columns["target_rv_24"], index=panel.index),
        )

    atr = featured.get("target_atr_12", pd.Series(np.nan, index=panel.index)).astype(float)
    for bars in opening_range_bars:
        high_name = f"target_or_{bars}_high"
        low_name = f"target_or_{bars}_low"
        complete = bar_index >= (bars - 1)
        opening_high = target_high.groupby(sessions, sort=False).transform(lambda values, n=bars: values.iloc[:n].max())
        opening_low = target_low.groupby(sessions, sort=False).transform(lambda values, n=bars: values.iloc[:n].min())
        opening_high = opening_high.where(complete)
        opening_low = opening_low.where(complete)
        new_columns[high_name] = opening_high
        new_columns[low_name] = opening_low
        opening_range = opening_high - opening_low
        above_opening_high = target_close > opening_high
        below_opening_low = target_close < opening_low
        new_columns[f"target_dist_or_{bars}_high_atr"] = _safe_ratio(np.log(target_close / opening_high), atr)
        new_columns[f"target_dist_or_{bars}_low_atr"] = _safe_ratio(np.log(target_close / opening_low), atr)
        new_columns[f"target_or_{bars}_range_atr"] = _safe_ratio(opening_range / target_close, setup_atr)
        new_columns[f"target_breakout_margin_or_{bars}_high_atr"] = _safe_ratio(np.log(target_close / opening_high), setup_atr)
        new_columns[f"target_breakdown_margin_or_{bars}_low_atr"] = _safe_ratio(np.log(opening_low / target_close), setup_atr)
        new_columns[f"target_above_or_{bars}_high"] = above_opening_high
        new_columns[f"target_below_or_{bars}_low"] = below_opening_low
        high_attempts = (target_high > opening_high).where(complete, False).astype(int).groupby(sessions, sort=False).cumsum()
        low_attempts = (target_low < opening_low).where(complete, False).astype(int).groupby(sessions, sort=False).cumsum()
        new_columns[f"target_breakout_attempt_count_or_{bars}_high"] = high_attempts
        new_columns[f"target_breakout_attempt_count_or_{bars}_low"] = low_attempts
        for persistence_window in breakout_persistence_windows:
            new_columns[f"target_above_or_{bars}_high_persist_{persistence_window}"] = _grouped_rolling_sum(
                above_opening_high.astype(float), sessions, persistence_window
            )
            new_columns[f"target_below_or_{bars}_low_persist_{persistence_window}"] = _grouped_rolling_sum(
                below_opening_low.astype(float), sessions, persistence_window
            )

    for window in breakout_windows:
        rolling_high = _grouped_rolling_max(target_high, sessions, window)
        rolling_low = _grouped_rolling_min(target_low, sessions, window)
        previous_rolling_high = _grouped_shift(pd.Series(rolling_high, index=panel.index), sessions, 1)
        previous_rolling_low = _grouped_shift(pd.Series(rolling_low, index=panel.index), sessions, 1)
        breaks_high = target_high > previous_rolling_high
        breaks_low = target_low < previous_rolling_low
        new_columns[f"target_roll_high_{window}_prev"] = previous_rolling_high
        new_columns[f"target_roll_low_{window}_prev"] = previous_rolling_low
        new_columns[f"target_breaks_roll_high_{window}"] = breaks_high
        new_columns[f"target_breaks_roll_low_{window}"] = breaks_low
        new_columns[f"target_breakout_margin_roll_high_{window}_atr"] = _safe_ratio(np.log(target_close / previous_rolling_high), setup_atr)
        new_columns[f"target_breakdown_margin_roll_low_{window}_atr"] = _safe_ratio(np.log(previous_rolling_low / target_close), setup_atr)
        new_columns[f"target_failed_breakout_high_{window}"] = breaks_high & (target_close < previous_rolling_high)
        new_columns[f"target_failed_breakout_low_{window}"] = breaks_low & (target_close > previous_rolling_low)

    return _with_columns(featured, new_columns)


def add_relative_returns(features: pd.DataFrame, symbols: list[str], target_symbol: str, feature_cfg: dict[str, Any]) -> pd.DataFrame:
    featured = features.copy()
    windows = [6, 12, 24]
    pairs = [
        ("QQQ", "SPY"),
        ("IWM", "SPY"),
        ("DIA", "SPY"),
        ("XLK", "SPY"),
        ("XLP", "SPY"),
        ("TLT", "SPY"),
        ("IEF", "SPY"),
        ("GLD", "SPY"),
        ("HYG", "LQD"),
    ]
    target = target_symbol.upper()
    if target != "SPY":
        pairs.extend([(target, "SPY"), (target, "QQQ")])

    available_symbols = set(symbols)
    new_columns: dict[str, Any] = {}
    for left, right in pairs:
        if left not in available_symbols or right not in available_symbols:
            continue
        for window in windows:
            left_col = _ret_col(left, window)
            right_col = _ret_col(right, window)
            if left_col in featured and right_col in featured:
                new_columns[f"relret_{left}_{right}_{window}"] = featured[left_col] - featured[right_col]
        left_open = f"dist_open_{left}"
        right_open = f"dist_open_{right}"
        if left_open in featured and right_open in featured:
            new_columns[f"relopen_{left}_{right}"] = featured[left_open] - featured[right_open]
    return _with_columns(featured, new_columns)


def add_spreads_and_breadth(features: pd.DataFrame, panel: pd.DataFrame, symbols: list[str], feature_cfg: dict[str, Any]) -> pd.DataFrame:
    featured = features.copy()
    indices = _group_symbols(feature_cfg, "indices", symbols)
    sectors = _group_symbols(feature_cfg, "sectors", symbols)
    growth = _group_symbols(feature_cfg, "growth", symbols)
    cyclicals = _group_symbols(feature_cfg, "cyclicals", symbols)
    defensives = _group_symbols(feature_cfg, "defensives", symbols)
    bonds = _group_symbols(feature_cfg, "bonds", symbols)
    breadth_windows = [int(window) for window in feature_cfg.get("breadth_windows", [6, 12, 24])]

    new_columns: dict[str, Any] = {}
    for window in breadth_windows:
        new_columns[f"positive_index_count_{window}"] = _row_count_positive(featured, [_ret_col(symbol, window) for symbol in indices])
        new_columns[f"positive_sector_count_{window}"] = _row_count_positive(featured, [_ret_col(symbol, window) for symbol in sectors])
        rel_strength = [
            featured[_ret_col(symbol, window)] - featured[_ret_col("SPY", window)]
            for symbol in sectors
            if _ret_col(symbol, window) in featured and _ret_col("SPY", window) in featured
        ]
        new_columns[f"sector_rel_strength_count_{window}"] = (pd.concat(rel_strength, axis=1) > 0.0).sum(axis=1) if rel_strength else np.nan
        sector_returns = [_ret_col(symbol, window) for symbol in sectors if _ret_col(symbol, window) in featured]
        if sector_returns:
            new_columns[f"leadership_concentration_score_{window}"] = featured[sector_returns].max(axis=1) - featured[sector_returns].median(axis=1)

    sector_above_cols = [
        featured[f"above_vwap_{symbol}"].rename(symbol) for symbol in sectors if f"above_vwap_{symbol}" in featured
    ]
    new_columns["sector_above_vwap_count"] = pd.concat(sector_above_cols, axis=1).sum(axis=1) if sector_above_cols else np.nan
    index_above_cols = [featured[f"above_vwap_{symbol}"].rename(symbol) for symbol in indices if f"above_vwap_{symbol}" in featured]
    new_columns["index_above_vwap_count"] = pd.concat(index_above_cols, axis=1).sum(axis=1) if index_above_cols else np.nan

    index_open_cols = [f"dist_open_{symbol}" for symbol in indices if f"dist_open_{symbol}" in featured]
    sector_open_cols = [f"dist_open_{symbol}" for symbol in sectors if f"dist_open_{symbol}" in featured]
    new_columns["positive_index_count_open"] = _row_count_positive(featured, index_open_cols)
    new_columns["positive_sector_count_open"] = _row_count_positive(featured, sector_open_cols)
    if sector_open_cols and "dist_open_SPY" in featured:
        new_columns["sector_rel_strength_count_open"] = (featured[sector_open_cols].sub(featured["dist_open_SPY"], axis=0) > 0.0).sum(axis=1)

    for window in [12, 24]:
        new_columns[f"spread_growth_defensive_{window}"] = _mean_ret(featured, growth, window) - _mean_ret(featured, defensives, window)
        new_columns[f"spread_cyclicals_defensive_{window}"] = _mean_ret(featured, cyclicals, window) - _mean_ret(featured, defensives, window)
        new_columns[f"spread_tech_broad_{window}"] = _mean_ret(featured, [symbol for symbol in ["QQQ", "XLK"] if symbol in symbols], window) - _mean_ret(
            featured, [symbol for symbol in ["SPY", "IWM", "DIA"] if symbol in symbols], window
        )
        new_columns[f"spread_equity_bonds_{window}"] = _mean_ret(featured, indices, window) - _mean_ret(featured, bonds, window)
        if {"SPY", "GLD"}.issubset(symbols):
            new_columns[f"spread_equity_gold_{window}"] = featured[_ret_col("SPY", window)] - featured[_ret_col("GLD", window)]
        if {"HYG", "LQD"}.issubset(symbols):
            new_columns[f"spread_credit_{window}"] = featured[_ret_col("HYG", window)] - featured[_ret_col("LQD", window)]

    if "dist_open_SPY" in featured:
        growth_open = _row_mean(featured, [f"dist_open_{symbol}" for symbol in growth])
        defensive_open = _row_mean(featured, [f"dist_open_{symbol}" for symbol in defensives])
        cyclicals_open = _row_mean(featured, [f"dist_open_{symbol}" for symbol in cyclicals])
        broad_open = _row_mean(featured, [f"dist_open_{symbol}" for symbol in indices])
        spread_growth_defensive_open = growth_open - defensive_open
        new_columns["spread_growth_defensive_open"] = spread_growth_defensive_open
        new_columns["spread_cyclicals_defensive_open"] = cyclicals_open - defensive_open
        new_columns["spread_tech_broad_open"] = _row_mean(featured, [column for column in ["dist_open_QQQ", "dist_open_XLK"] if column in featured]) - broad_open
        new_columns["risk_on_open_confirmation"] = (
            _safe_ratio(new_columns["positive_index_count_open"], pd.Series(max(len(indices), 1), index=featured.index))
            + _safe_ratio(new_columns["positive_sector_count_open"], pd.Series(max(len(sectors), 1), index=featured.index))
            + spread_growth_defensive_open
        )

    return _with_columns(featured, new_columns)


def add_volatility_and_scores(features: pd.DataFrame, symbols: list[str], feature_cfg: dict[str, Any]) -> pd.DataFrame:
    featured = features.copy()
    short_window = int(feature_cfg.get("range_ratio_short_window", 6))
    long_window = int(feature_cfg.get("range_ratio_long_window", 24))
    ratio_windows = _range_ratio_windows(feature_cfg)
    indices = _group_symbols(feature_cfg, "indices", symbols)
    sectors = _group_symbols(feature_cfg, "sectors", symbols)
    growth = _group_symbols(feature_cfg, "growth", symbols)
    cyclicals = _group_symbols(feature_cfg, "cyclicals", symbols)
    defensives = _group_symbols(feature_cfg, "defensives", symbols)

    sector_range_cols = [_range_col(symbol) for symbol in sectors]
    all_risk_range_ratio_cols = [_range_ratio_col(symbol, short_window, long_window) for symbol in indices + sectors]

    for pair_short, pair_long in ratio_windows:
        index_ratio_cols = [_range_ratio_col(symbol, pair_short, pair_long) for symbol in indices]
        featured[f"market_range_ratio_{pair_short}_{pair_long}"] = _row_mean(featured, index_ratio_cols)
    featured["sector_range_dispersion_12"] = _row_std(featured, sector_range_cols)
    featured["cross_asset_vol_expansion_score"] = _row_mean(featured, all_risk_range_ratio_cols)

    index_ret_12 = _mean_ret(featured, indices, 12)
    defensive_ret_12 = _mean_ret(featured, defensives, 12)
    growth_ret_12 = _mean_ret(featured, growth, 12)
    cyclical_ret_12 = _mean_ret(featured, cyclicals, 12)
    haven_rel = _row_mean(
        featured,
        [column for column in ["relret_TLT_SPY_12", "relret_IEF_SPY_12", "relret_GLD_SPY_12"] if column in featured],
    )
    market_range = featured[f"market_range_ratio_{short_window}_{long_window}"]
    spread_credit = featured.get("spread_credit_12", pd.Series(np.nan, index=featured.index))
    spread_growth_defensive = featured.get("spread_growth_defensive_12", pd.Series(np.nan, index=featured.index))
    spread_equity_bonds = featured.get("spread_equity_bonds_12", pd.Series(np.nan, index=featured.index))

    featured["risk_on_score"] = index_ret_12 + spread_growth_defensive + spread_credit + spread_equity_bonds - market_range * 0.001
    featured["risk_off_score"] = -index_ret_12 - spread_credit + haven_rel + market_range * 0.001
    featured["defensive_rotation_score"] = defensive_ret_12 - pd.concat([growth_ret_12, cyclical_ret_12], axis=1).mean(axis=1) + haven_rel
    featured["narrow_rally_score"] = (
        featured.get("relret_QQQ_SPY_12", pd.Series(np.nan, index=featured.index))
        + featured.get("relret_XLK_SPY_12", pd.Series(np.nan, index=featured.index))
        - featured.get("relret_IWM_SPY_12", pd.Series(np.nan, index=featured.index))
        + featured.get("leadership_concentration_score_12", pd.Series(np.nan, index=featured.index))
    )

    sign_cols = [_ret_col(symbol, 12) for symbol in indices + sectors if _ret_col(symbol, 12) in featured]
    if sign_cols and _ret_col("SPY", 12) in featured:
        spy_sign = np.sign(featured[_ret_col("SPY", 12)])
        conflict = pd.concat([(np.sign(featured[column]) != spy_sign).rename(column) for column in sign_cols], axis=1)
        featured["cross_asset_signal_conflict_score"] = conflict.mean(axis=1)
    else:
        featured["cross_asset_signal_conflict_score"] = np.nan
    featured["chop_score"] = (
        -featured["target_signed_efficiency_12"].abs()
        - featured["target_dir_persistence_12"].abs()
        + featured["cross_asset_signal_conflict_score"]
    )
    featured["intraday_stress_score"] = pd.concat([featured["risk_off_score"], featured["cross_asset_vol_expansion_score"]], axis=1).mean(axis=1)
    return featured


def build_cross_asset_features(
    panel: pd.DataFrame,
    lab_config: dict[str, Any],
    feature_config: dict[str, Any],
    target_symbol: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    target = (target_symbol or lab_config.get("lab", {}).get("target_symbol", "SPY")).upper()
    resolved_symbols = symbols or resolve_symbols(lab_config, target_symbol=target)
    panel_sorted = panel.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)

    features = panel_sorted.loc[:, INDEX_COLUMNS].copy()
    for helper in TARGET_HELPER_FIELDS:
        source = _symbol_col(target, helper)
        if source in panel_sorted:
            features[f"target_{helper}" if not helper.startswith("target_") else helper] = panel_sorted[source]

    features = add_per_symbol_features(features, panel_sorted, resolved_symbols, feature_config)
    features = add_target_features(features, panel_sorted, target, feature_config)
    features = add_target_setup_features(features, panel_sorted, target, feature_config)
    features = add_relative_returns(features, resolved_symbols, target, feature_config)
    features = add_spreads_and_breadth(features, panel_sorted, resolved_symbols, feature_config)
    features = add_volatility_and_scores(features, resolved_symbols, feature_config)
    return features


def feature_output_path(lab_config: dict[str, Any], feature_config: dict[str, Any], target_symbol: str) -> Path:
    lab_cfg = lab_config.get("lab", {})
    timeframe = lab_cfg.get("timeframe", lab_config.get("project", {}).get("frequency", "5min"))
    universe_id = lab_cfg.get("universe_id", "core_cross_asset_v1")
    version = feature_config.get("feature_set_version", "cross_asset_v1")
    return Path("data/features") / target_symbol.upper() / timeframe / universe_id / version / "features.parquet"


def report_output_path(lab_config: dict[str, Any], feature_config: dict[str, Any], target_symbol: str) -> Path:
    version = feature_config.get("feature_set_version", "cross_asset_v1")
    return Path(lab_config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / f"cross_asset_features_{version}.md"


def render_report(report: FeatureBuildReport, feature_config: dict[str, Any]) -> str:
    missing = "\n".join(f"- `{column}`" for column in report.missing_hmm_feature_columns) or "- Ninguna"
    feature_columns = "\n".join(f"- `{column}`" for column in feature_config.get("hmm_feature_columns", []))
    return f"""# Cross-Asset Features - {report.target_symbol} {report.feature_set_version}

## Summary

- Output: `{report.output_path}`
- Rows: `{report.rows}`
- Columns: `{report.columns}`
- HMM feature columns configured: `{report.hmm_feature_columns}`
- Missing configured HMM columns: `{len(report.missing_hmm_feature_columns)}`

## Missing HMM Columns

{missing}

## HMM Feature Columns

{feature_columns}
"""


def build_report(features: pd.DataFrame, feature_config: dict[str, Any], target_symbol: str, output_path: Path) -> FeatureBuildReport:
    hmm_columns = list(feature_config.get("hmm_feature_columns", []))
    missing = [column for column in hmm_columns if column not in features.columns]
    return FeatureBuildReport(
        target_symbol=target_symbol.upper(),
        feature_set_version=feature_config.get("feature_set_version", "cross_asset_v1"),
        rows=int(len(features)),
        columns=int(len(features.columns)),
        hmm_feature_columns=len(hmm_columns),
        missing_hmm_feature_columns=missing,
        output_path=str(output_path),
    )


def run(config_path: str | Path, features_config_path: str | Path, target_symbol: str | None = None) -> Path:
    lab_config = load_yaml(config_path)
    feature_config = load_yaml(features_config_path)
    target = (target_symbol or lab_config.get("lab", {}).get("target_symbol", "SPY")).upper()
    panel_path = aligned_panel_path(lab_config, target)
    panel = pd.read_parquet(panel_path)
    features = build_cross_asset_features(panel, lab_config, feature_config, target_symbol=target)

    output_path = feature_output_path(lab_config, feature_config, target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    report = build_report(features, feature_config, target, output_path)
    report_path = report_output_path(lab_config, feature_config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(report, feature_config), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal cross-asset features from an aligned panel.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--features-config", default="configs/features/cross_asset_v1.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    output_path = run(args.config, args.features_config, args.target)
    print(f"Cross-asset features written to: {output_path}")


if __name__ == "__main__":
    main()

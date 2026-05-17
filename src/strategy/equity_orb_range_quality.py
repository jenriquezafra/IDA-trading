from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds
from src.strategy.equity_orb_pairs import (
    EquityOrbConfig,
    EquityOrbOutputs,
    _markdown_table,
    _split_frame,
    _stable_seed,
    add_market_prices_to_pair_base,
    add_orb_range,
    aggregate_trades,
    apply_costs,
    build_coverage,
    build_events,
    build_pair_frame,
    load_config as load_base_config,
    load_panel,
    market_beta_base,
    required_symbols,
    sample_control_directions,
    simulate_pair_base,
    spread_directions,
)


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_orb_range_quality_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/equity_orb_range_quality/5min")
QUALITY_LABEL = "range_quality_orb_spread_breakout"
UNFILTERED_REFERENCE_LABEL = "unfiltered_continuation_reference"
RANDOM_CONTROL_LABEL = "range_quality_random_same_frequency_control"
SAME_HOUR_CONTROL_LABEL = "range_quality_same_hour_control"
MARKET_BETA_CONTROL_LABEL = "range_quality_market_beta_control"


@dataclass(frozen=True)
class RangeQualityFilter:
    filter_id: str
    label: str
    min_percentile: float
    max_percentile: float


@dataclass(frozen=True)
class RangeQualityConfig:
    base: EquityOrbConfig
    filters: tuple[RangeQualityFilter, ...]


@dataclass(frozen=True)
class RangeQualityOutputs(EquityOrbOutputs):
    thresholds_path: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RangeQualityConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    base = load_base_config(config_path)
    quality = dict(raw.get("range_quality", {}))
    filters_raw = quality.get("filters", [])
    if not isinstance(filters_raw, list) or not filters_raw:
        raise ValueError("range_quality.filters must be a non-empty list")
    filters: list[RangeQualityFilter] = []
    for item in filters_raw:
        min_pct = float(item.get("min_percentile", 0.0))
        max_pct = float(item.get("max_percentile", 1.0))
        if min_pct < 0.0 or max_pct > 1.0 or min_pct > max_pct:
            raise ValueError("range_quality filter percentiles must satisfy 0 <= min <= max <= 1")
        filter_id = str(item["filter_id"]).strip()
        if not filter_id:
            raise ValueError("range_quality filter_id cannot be empty")
        filters.append(
            RangeQualityFilter(
                filter_id=filter_id,
                label=str(item.get("label", filter_id)).strip() or filter_id,
                min_percentile=min_pct,
                max_percentile=max_pct,
            )
        )
    return RangeQualityConfig(base=base, filters=tuple(filters))


def _with_output_dir(config: RangeQualityConfig, output_dir: str | Path | None) -> RangeQualityConfig:
    if output_dir is None:
        return config
    base = config.base
    return RangeQualityConfig(
        base=EquityOrbConfig(
            strategy_id=base.strategy_id,
            hypothesis_id=base.hypothesis_id,
            timeframe=base.timeframe,
            panel_path=base.panel_path,
            pairs=base.pairs,
            windows=base.windows,
            horizons=base.horizons,
            cost_bps_values=base.cost_bps_values,
            split_policy=base.split_policy,
            output_dir=Path(output_dir),
            controls=base.controls,
        ),
        filters=config.filters,
    )


def _session_widths(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "orb_width", "orb_observations", "orb_range_bars"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing range-quality columns: {', '.join(sorted(missing))}")
    widths = (
        frame.loc[:, ["session", "orb_width", "orb_observations", "orb_range_bars"]]
        .drop_duplicates("session")
        .copy()
    )
    widths["orb_width"] = pd.to_numeric(widths["orb_width"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    widths = widths[
        widths["orb_width"].gt(0.0)
        & widths["orb_observations"].ge(widths["orb_range_bars"])
    ].copy()
    return widths.loc[:, ["session", "orb_width"]]


def fit_width_thresholds(
    frame: pd.DataFrame,
    train_sessions: tuple[str, ...],
    filters: tuple[RangeQualityFilter, ...],
    *,
    fold: int,
    pair_id: str,
    orb_window: str,
) -> pd.DataFrame:
    widths = _session_widths(frame)
    train_widths = widths.loc[widths["session"].astype(str).isin(train_sessions), "orb_width"].astype(float)
    rows: list[dict[str, Any]] = []
    for quality_filter in filters:
        if train_widths.empty:
            lower = np.nan
            upper = np.nan
        else:
            lower = float(train_widths.quantile(quality_filter.min_percentile))
            upper = float(train_widths.quantile(quality_filter.max_percentile))
        rows.append(
            {
                "fold": int(fold),
                "pair_id": pair_id,
                "orb_window": orb_window,
                "range_quality_filter": quality_filter.filter_id,
                "range_quality_label": quality_filter.label,
                "min_percentile": float(quality_filter.min_percentile),
                "max_percentile": float(quality_filter.max_percentile),
                "lower_width": lower,
                "upper_width": upper,
                "train_sessions": int(train_widths.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def apply_width_filter(frame: pd.DataFrame, directions: pd.Series, threshold: pd.Series) -> pd.Series:
    out = directions.copy()
    lower = float(threshold["lower_width"])
    upper = float(threshold["upper_width"])
    if not np.isfinite(lower) or not np.isfinite(upper):
        out.loc[:] = 0
        return out
    widths = _session_widths(frame).set_index("session")["orb_width"]
    eligible_sessions = widths.loc[widths.ge(lower) & widths.le(upper)].index.astype(str)
    out.loc[~frame["session"].astype(str).isin(eligible_sessions)] = 0
    return out


def _add_quality_metadata(base: pd.DataFrame, threshold: pd.Series) -> pd.DataFrame:
    if base.empty:
        return base
    out = base.copy()
    out["range_quality_filter"] = str(threshold["range_quality_filter"])
    out["range_quality_label"] = str(threshold["range_quality_label"])
    out["range_quality_min_percentile"] = float(threshold["min_percentile"])
    out["range_quality_max_percentile"] = float(threshold["max_percentile"])
    out["range_quality_lower_width"] = float(threshold["lower_width"])
    out["range_quality_upper_width"] = float(threshold["upper_width"])
    return out


def _add_unfiltered_metadata(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base
    out = base.copy()
    out["range_quality_filter"] = "unfiltered"
    out["range_quality_label"] = "unfiltered continuation reference"
    return out


def run_strategy(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> RangeQualityOutputs:
    config = _with_output_dir(load_config(config_path), output_dir)
    base_config = config.base
    symbols = required_symbols(base_config)
    panel = load_panel(base_config.panel_path, symbols)
    coverage = build_coverage(panel, symbols)
    folds = build_monthly_folds(panel, base_config.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")

    root = base_config.output_dir
    outputs = RangeQualityOutputs(
        output_dir=root,
        coverage_path=root / "coverage.parquet",
        events_path=root / "events.parquet",
        trades_path=root / "trades.parquet",
        daily_path=root / "daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
        thresholds_path=root / "range_quality_thresholds.parquet",
    )
    root.mkdir(parents=True, exist_ok=True)

    pair_window_frames: dict[tuple[str, str], pd.DataFrame] = {}
    event_frames: list[pd.DataFrame] = []
    for pair in base_config.pairs:
        base_pair = build_pair_frame(panel, pair)
        for window in base_config.windows:
            frame = add_orb_range(base_pair, window)
            pair_window_frames[(pair.pair_id, window.window_id)] = frame
            event_frames.append(frame)

    all_trades: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    random_seed = int(base_config.controls.get("random_seed", 2404))

    for fold in folds:
        fold_thresholds: dict[tuple[str, str], pd.DataFrame] = {}
        for pair in base_config.pairs:
            for window in base_config.windows:
                full_pair_frame = pair_window_frames[(pair.pair_id, window.window_id)]
                thresholds = fit_width_thresholds(
                    full_pair_frame,
                    fold.train_sessions,
                    config.filters,
                    fold=fold.fold,
                    pair_id=pair.pair_id,
                    orb_window=window.window_id,
                )
                fold_thresholds[(pair.pair_id, window.window_id)] = thresholds
                threshold_frames.append(thresholds)

        for split, sessions in (
            ("train", fold.train_sessions),
            ("validation", fold.validation_sessions),
            ("test", fold.test_sessions),
        ):
            split_sessions[(fold.fold, split)] = tuple(sessions)
            for pair in base_config.pairs:
                for window in base_config.windows:
                    full_pair_frame = pair_window_frames[(pair.pair_id, window.window_id)]
                    pair_frame = _split_frame(full_pair_frame, sessions)
                    base_direction = spread_directions(pair_frame)
                    thresholds = fold_thresholds[(pair.pair_id, window.window_id)]
                    for horizon in base_config.horizons:
                        if base_config.controls.get("unfiltered_reference", True):
                            reference_base = simulate_pair_base(
                                pair_frame,
                                base_direction,
                                label=UNFILTERED_REFERENCE_LABEL,
                                fold=fold.fold,
                                split=split,
                                horizon=horizon,
                                strategy_id=base_config.strategy_id,
                            )
                            if not reference_base.empty:
                                all_trades.append(apply_costs(_add_unfiltered_metadata(reference_base), base_config.cost_bps_values))

                        for _, threshold in thresholds.iterrows():
                            filtered_direction = apply_width_filter(pair_frame, base_direction, threshold)
                            candidate_base = simulate_pair_base(
                                pair_frame,
                                filtered_direction,
                                label=QUALITY_LABEL,
                                fold=fold.fold,
                                split=split,
                                horizon=horizon,
                                strategy_id=base_config.strategy_id,
                            )
                            if not candidate_base.empty:
                                candidate_base = _add_quality_metadata(candidate_base, threshold)
                                all_trades.append(apply_costs(candidate_base, base_config.cost_bps_values))
                            if candidate_base.empty:
                                continue
                            if base_config.controls.get("random_same_frequency", True):
                                random_direction = sample_control_directions(
                                    pair_frame,
                                    candidate_base,
                                    horizon=horizon,
                                    mode="random",
                                    seed=_stable_seed(
                                        random_seed,
                                        "random",
                                        fold.fold,
                                        horizon,
                                        pair.pair_id,
                                        window.window_id,
                                        threshold["range_quality_filter"],
                                    ),
                                )
                                random_base = simulate_pair_base(
                                    pair_frame,
                                    random_direction,
                                    label=RANDOM_CONTROL_LABEL,
                                    fold=fold.fold,
                                    split=split,
                                    horizon=horizon,
                                    strategy_id=base_config.strategy_id,
                                )
                                if not random_base.empty:
                                    all_trades.append(apply_costs(_add_quality_metadata(random_base, threshold), base_config.cost_bps_values))
                            if base_config.controls.get("same_hour", True):
                                same_hour_direction = sample_control_directions(
                                    pair_frame,
                                    candidate_base,
                                    horizon=horizon,
                                    mode="same_hour",
                                    seed=_stable_seed(
                                        random_seed,
                                        "same_hour",
                                        fold.fold,
                                        horizon,
                                        pair.pair_id,
                                        window.window_id,
                                        threshold["range_quality_filter"],
                                    ),
                                )
                                same_hour_base = simulate_pair_base(
                                    pair_frame,
                                    same_hour_direction,
                                    label=SAME_HOUR_CONTROL_LABEL,
                                    fold=fold.fold,
                                    split=split,
                                    horizon=horizon,
                                    strategy_id=base_config.strategy_id,
                                )
                                if not same_hour_base.empty:
                                    all_trades.append(apply_costs(_add_quality_metadata(same_hour_base, threshold), base_config.cost_bps_values))
                            if base_config.controls.get("market_beta", True):
                                market_ready = add_market_prices_to_pair_base(candidate_base, pair_frame, horizon)
                                market_base = market_beta_base(market_ready.dropna(subset=["SPY_entry_px", "SPY_exit_px"]))
                                if not market_base.empty:
                                    market_base["label"] = MARKET_BETA_CONTROL_LABEL
                                    all_trades.append(apply_costs(_add_quality_metadata(market_base, threshold), base_config.cost_bps_values))

    events = build_events(event_frames)
    thresholds = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary, daily, monthly = aggregate_trades(trades, split_sessions)
    coverage.to_parquet(outputs.coverage_path, index=False)
    events.to_parquet(outputs.events_path, index=False)
    thresholds.to_parquet(outputs.thresholds_path, index=False)
    trades.to_parquet(outputs.trades_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    _write_manifest(outputs.manifest_path, config, config_path, folds, outputs, symbols)
    _write_report(outputs.report_path, config, coverage, thresholds, summary)
    return outputs


def _candidate_rollup(summary: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    candidate = summary[
        summary["label"].eq(QUALITY_LABEL)
        & summary["cost_bps_per_leg_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame()
    grouped = (
        candidate.groupby(["split", "range_quality_filter", "range_quality_label", "pair_id", "orb_window", "horizon_bars"], as_index=False)
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


def _filter_rollup(summary: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    part = summary[
        summary["cost_bps_per_leg_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if part.empty:
        return pd.DataFrame()
    grouped = (
        part.groupby(["split", "label", "range_quality_filter"], as_index=False)
        .agg(
            groups=("pair_id", "count"),
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            profit_factor=("profit_factor", "mean"),
        )
    )
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "label", "net_return"], ascending=[True, True, False], kind="stable").drop(columns="_split_order")


def _matched_validation_test(candidate: pd.DataFrame) -> pd.DataFrame:
    if candidate.empty:
        return pd.DataFrame()
    keys = ["range_quality_filter", "pair_id", "orb_window", "horizon_bars"]
    validation = candidate[candidate["split"].eq("validation")].loc[:, keys + ["net_return", "positive_folds"]]
    test = candidate[candidate["split"].eq("test")].loc[:, keys + ["net_return", "positive_folds"]]
    matched = validation.merge(test, on=keys, how="inner", suffixes=("_validation", "_test"))
    if matched.empty:
        return matched
    return matched.sort_values(["net_return_validation", "net_return_test"], ascending=[False, False], kind="stable")


def _positive_pocket_control_check(summary: pd.DataFrame, matched: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty or matched.empty:
        return pd.DataFrame()
    positive = matched[
        matched["net_return_validation"].gt(0.0)
        & matched["net_return_test"].gt(0.0)
    ].copy()
    if positive.empty:
        return pd.DataFrame()
    keys = ["range_quality_filter", "pair_id", "orb_window", "horizon_bars"]
    labels = [
        QUALITY_LABEL,
        RANDOM_CONTROL_LABEL,
        SAME_HOUR_CONTROL_LABEL,
        MARKET_BETA_CONTROL_LABEL,
    ]
    part = summary[
        summary["cost_bps_per_leg_round_trip"].eq(float(cost_bps))
        & summary["split"].isin(["validation", "test"])
        & summary["label"].isin(labels)
    ].copy()
    if part.empty:
        return pd.DataFrame()
    grouped = (
        part.groupby(["split", "label", *keys], as_index=False)
        .agg(net_return=("net_return", "sum"), trades=("trades", "sum"), max_top5_abs_share=("top5_abs_share", "max"))
    )
    rows: list[dict[str, Any]] = []
    for _, pocket in positive.iterrows():
        key_filter = np.logical_and.reduce([grouped[key].eq(pocket[key]) for key in keys])
        row: dict[str, Any] = {key: pocket[key] for key in keys}
        for split in ("validation", "test"):
            split_rows = grouped.loc[key_filter & grouped["split"].eq(split)]
            candidate_row = split_rows.loc[split_rows["label"].eq(QUALITY_LABEL)]
            row[f"{split}_candidate_net"] = float(candidate_row["net_return"].sum()) if not candidate_row.empty else np.nan
            row[f"{split}_candidate_trades"] = int(candidate_row["trades"].sum()) if not candidate_row.empty else 0
            row[f"{split}_candidate_top5_abs_share"] = float(candidate_row["max_top5_abs_share"].max()) if not candidate_row.empty else np.nan
            for label, suffix in (
                (RANDOM_CONTROL_LABEL, "random_net"),
                (SAME_HOUR_CONTROL_LABEL, "same_hour_net"),
                (MARKET_BETA_CONTROL_LABEL, "market_beta_net"),
            ):
                control_row = split_rows.loc[split_rows["label"].eq(label)]
                row[f"{split}_{suffix}"] = float(control_row["net_return"].sum()) if not control_row.empty else np.nan
            control_values = [
                row[f"{split}_random_net"],
                row[f"{split}_same_hour_net"],
                row[f"{split}_market_beta_net"],
            ]
            finite_controls = [value for value in control_values if np.isfinite(value)]
            row[f"{split}_best_control_net"] = max(finite_controls) if finite_controls else np.nan
            row[f"{split}_beats_best_control"] = bool(
                np.isfinite(row[f"{split}_candidate_net"])
                and np.isfinite(row[f"{split}_best_control_net"])
                and row[f"{split}_candidate_net"] > row[f"{split}_best_control_net"]
            )
        row["passes_control_and_concentration"] = bool(
            row["validation_beats_best_control"]
            and row["test_beats_best_control"]
            and np.isfinite(row["validation_candidate_top5_abs_share"])
            and np.isfinite(row["test_candidate_top5_abs_share"])
            and row["validation_candidate_top5_abs_share"] <= 0.50
            and row["test_candidate_top5_abs_share"] <= 0.50
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes_control_and_concentration", "validation_candidate_net", "test_candidate_net"], ascending=[False, False, False], kind="stable")


def _write_report(
    path: Path,
    config: RangeQualityConfig,
    coverage: pd.DataFrame,
    thresholds: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    candidate = _candidate_rollup(summary, cost_bps=2.0)
    filter_rollup = _filter_rollup(summary, cost_bps=2.0)
    matched = _matched_validation_test(candidate)
    validation_top = candidate[candidate["split"].eq("validation")].sort_values("net_return", ascending=False, kind="stable")
    test_top = candidate[candidate["split"].eq("test")].sort_values("net_return", ascending=False, kind="stable")
    best_validation = float(validation_top["net_return"].max()) if not validation_top.empty else np.nan
    best_test = float(test_top["net_return"].max()) if not test_top.empty else np.nan
    matched_positive = matched[
        matched["net_return_validation"].gt(0.0)
        & matched["net_return_test"].gt(0.0)
    ] if not matched.empty else pd.DataFrame()
    control_check = _positive_pocket_control_check(summary, matched, cost_bps=2.0)
    promoted_pockets = control_check[control_check["passes_control_and_concentration"]] if not control_check.empty else pd.DataFrame()
    if not promoted_pockets.empty:
        decision = "Initial read: H2.4 has at least one positive validation/test pocket that beats controls and concentration gates; inspect before promotion."
    elif not matched_positive.empty:
        decision = "Initial read: H2.4 is not promoted; positive pockets exist, but they fail control or concentration checks."
    elif np.isfinite(best_validation) and best_validation < 0.0 and np.isfinite(best_test) and best_test < 0.0:
        decision = "Initial read: H2.4 is rejected at this cost level; pre-registered range-quality filters remain negative in validation and test."
    else:
        decision = "Initial read: H2.4 needs manual review; validation/test quality pockets are mixed or incomplete."
    lines = [
        "# Equity ORB range-quality diagnostic",
        "",
        "This is the first H2.4 diagnostic: ORB continuation on relative log spreads conditioned by opening-range width.",
        "Range-width thresholds are fit on each fold's train sessions and then applied unchanged to validation and test.",
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
        "## Filters",
        "",
        *_markdown_table(
            pd.DataFrame([quality_filter.__dict__ for quality_filter in config.filters]),
            ["filter_id", "label", "min_percentile", "max_percentile"],
            limit=20,
        ),
        "",
        "## Candidate Rollup, Cost=2 bps Per Leg Round Trip",
        "",
        *_markdown_table(
            candidate,
            [
                "split",
                "range_quality_filter",
                "pair_id",
                "orb_window",
                "horizon_bars",
                "folds",
                "trades",
                "net_return",
                "avg_trade_net",
                "positive_folds",
                "max_top5_abs_share",
            ],
            limit=50,
        ),
        "",
        "## Matched Validation/Test Candidates",
        "",
        *_markdown_table(
            matched,
            [
                "range_quality_filter",
                "pair_id",
                "orb_window",
                "horizon_bars",
                "net_return_validation",
                "net_return_test",
                "positive_folds_validation",
                "positive_folds_test",
            ],
            limit=30,
        ),
        "",
        "## Positive Pocket Control Check",
        "",
        *_markdown_table(
            control_check,
            [
                "range_quality_filter",
                "pair_id",
                "orb_window",
                "horizon_bars",
                "validation_candidate_net",
                "validation_best_control_net",
                "validation_candidate_top5_abs_share",
                "test_candidate_net",
                "test_best_control_net",
                "test_candidate_top5_abs_share",
                "passes_control_and_concentration",
            ],
            limit=20,
        ),
        "",
        "## Validation Top Candidates",
        "",
        *_markdown_table(
            validation_top,
            ["range_quality_filter", "pair_id", "orb_window", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds"],
            limit=20,
        ),
        "",
        "## Test Top Candidates",
        "",
        *_markdown_table(
            test_top,
            ["range_quality_filter", "pair_id", "orb_window", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds"],
            limit=20,
        ),
        "",
        "## Filter And Control Rollup, Cost=2 bps Per Leg Round Trip",
        "",
        *_markdown_table(filter_rollup, ["split", "label", "range_quality_filter", "groups", "trades", "net_return", "avg_trade_net", "profit_factor"], limit=80),
        "",
        "## Threshold Sample",
        "",
        *_markdown_table(
            thresholds,
            ["fold", "pair_id", "orb_window", "range_quality_filter", "lower_width", "upper_width", "train_sessions"],
            limit=30,
        ),
        "",
        "## Read",
        "",
        "- Treat H2.4 as a diagnostic filter over H2.2, not as a standalone strategy.",
        "- A filter is only interesting if the same pre-registered bucket is positive in validation and test after costs and does not lose to random/same-hour controls.",
        "- Do not add more range buckets from test results; if these filters fail, park equity ORB.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path,
    config: RangeQualityConfig,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: RangeQualityOutputs,
    symbols: tuple[str, ...],
) -> None:
    base = config.base
    config_file = Path(config_path)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("strategy", base.strategy_id, "PAIRS", base.timeframe),
            "run_type": "strategy_backtest",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": base.strategy_id,
            "hypothesis_id": base.hypothesis_id,
            "entry_rule": "next_open",
            "exit_rule": "fixed_horizon_open",
            "position": "dollar_neutral_pair",
            "spread": "log(asset_a) - log(asset_b)",
            "trade_policy": "first_breakout_per_session",
            "range_quality": "opening_range_width_train_percentiles",
        },
        "data": {
            "panel_path": base.panel_path.as_posix(),
            "panel_fingerprint": fingerprint_path(base.panel_path) if base.panel_path.exists() else "MISSING",
            "config_path": config_file.as_posix(),
            "config_fingerprint": fingerprint_path(config_file) if config_file.exists() else "MISSING",
            "symbols": list(symbols),
            "timeframe": base.timeframe,
            "split_policy": base.split_policy,
            "n_folds": len(folds),
        },
        "parameters": {
            "pairs": [pair.__dict__ for pair in base.pairs],
            "orb_windows": [window.__dict__ for window in base.windows],
            "horizons": list(base.horizons),
            "cost_bps_values": list(base.cost_bps_values),
            "range_quality_filters": [quality_filter.__dict__ for quality_filter in config.filters],
            "controls": base.controls,
        },
        "outputs": {
            "coverage": outputs.coverage_path.as_posix(),
            "events": outputs.events_path.as_posix(),
            "range_quality_thresholds": outputs.thresholds_path.as_posix(),
            "trades": outputs.trades_path.as_posix(),
            "daily": outputs.daily_path.as_posix(),
            "monthly": outputs.monthly_path.as_posix(),
            "summary": outputs.summary_path.as_posix(),
            "report": outputs.report_path.as_posix(),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H2.4 equity ORB range-quality diagnostic")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_strategy(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

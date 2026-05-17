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
    OrbWindow,
    PairSpec,
    _markdown_table,
    _split_frame,
    _stable_seed,
    add_market_prices_to_pair_base,
    add_orb_range,
    aggregate_trades,
    apply_costs,
    asset_directions,
    build_coverage,
    build_pair_frame,
    failed_asset_directions,
    failed_reversion_directions,
    load_config as load_base_config,
    load_panel,
    market_beta_base,
    required_symbols,
    sample_control_directions,
    simulate_asset_base,
    simulate_pair_base,
    spread_directions,
)


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_orb_failed_pairs_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/equity_orb_failed_pairs/5min")
FAILED_LABEL = "failed_orb_reversion"
CONTINUATION_REFERENCE_LABEL = "continuation_orb_reference"
RANDOM_CONTROL_LABEL = "random_same_frequency_control"
SAME_HOUR_CONTROL_LABEL = "same_hour_control"
MARKET_BETA_CONTROL_LABEL = "market_beta_control"
ASSET_FAILED_BASELINE_LABEL = "failed_orb_asset_baseline"


@dataclass(frozen=True)
class FailedOrbOutputs(EquityOrbOutputs):
    pass


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> EquityOrbConfig:
    return load_base_config(path)


def build_failed_events(pair_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in pair_frames:
        directions = failed_reversion_directions(frame)
        events = frame.loc[
            directions.ne(0),
            [
                "pair_id",
                "asset_a",
                "asset_b",
                "orb_window",
                "orb_window_label",
                "orb_range_bars",
                "session",
                "timestamp",
                "bar_index",
                "spread_close",
                "orb_high",
                "orb_low",
                "orb_width",
            ],
        ].copy()
        events["direction"] = directions.loc[events.index].to_numpy(dtype=int)
        events["side"] = np.where(events["direction"].gt(0), "long_spread", "short_spread")
        rows.append(events)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_strategy(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> FailedOrbOutputs:
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
    outputs = FailedOrbOutputs(
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
    random_seed = int(config.controls.get("random_seed", 2505))

    pair_window_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for pair in config.pairs:
        base_pair = build_pair_frame(panel, pair)
        for window in config.windows:
            frame = add_orb_range(base_pair, window)
            pair_window_frames[(pair.pair_id, window.window_id)] = frame
            event_frames.append(frame)

    asset_symbols = tuple(sorted({pair.asset_a for pair in config.pairs} | {pair.asset_b for pair in config.pairs}))
    failed_asset_frames: dict[tuple[str, str], pd.DataFrame] = {
        (asset, window.window_id): failed_asset_directions(panel, asset, window)
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
                    failed_direction = failed_reversion_directions(pair_frame)
                    continuation_direction = spread_directions(pair_frame)
                    for horizon in config.horizons:
                        candidate_base = simulate_pair_base(
                            pair_frame,
                            failed_direction,
                            label=FAILED_LABEL,
                            fold=fold.fold,
                            split=split,
                            horizon=horizon,
                            strategy_id=config.strategy_id,
                        )
                        if not candidate_base.empty:
                            all_trades.append(apply_costs(candidate_base, config.cost_bps_values))
                        if config.controls.get("continuation_reference", True):
                            continuation_base = simulate_pair_base(
                                pair_frame,
                                continuation_direction,
                                label=CONTINUATION_REFERENCE_LABEL,
                                fold=fold.fold,
                                split=split,
                                horizon=horizon,
                                strategy_id=config.strategy_id,
                            )
                            if not continuation_base.empty:
                                all_trades.append(apply_costs(continuation_base, config.cost_bps_values))
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
                                market_base["label"] = MARKET_BETA_CONTROL_LABEL
                                all_trades.append(apply_costs(market_base, config.cost_bps_values))
                        if config.controls.get("directional_failed_orb_baseline", True):
                            for asset_role, asset in (("asset_a", pair.asset_a), ("asset_b", pair.asset_b)):
                                asset_frame = _split_frame(failed_asset_frames[(asset, window.window_id)], sessions)
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
                                    asset_base["label"] = ASSET_FAILED_BASELINE_LABEL
                                    all_trades.append(apply_costs(asset_base, config.cost_bps_values))

    events = build_failed_events(event_frames)
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


def _candidate_rollup(summary: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    candidate = summary[
        summary["label"].eq(FAILED_LABEL)
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
    if np.isfinite(best_validation) and np.isfinite(best_test) and best_validation > 0.0 and best_test > 0.0:
        decision = "Initial read: H2.5 has at least one positive validation/test pocket; inspect controls, folds and concentration before promotion."
    elif np.isfinite(best_validation) and best_validation < 0.0 and np.isfinite(best_test) and best_test < 0.0:
        decision = "Initial read: H2.5 failed ORB is rejected at this cost level; every pair/window/horizon is negative in validation and test."
    else:
        decision = "Initial read: H2.5 needs manual review; candidate summary is incomplete or mixed."
    lines = [
        "# Equity failed ORB pairs diagnostic",
        "",
        "This is the first H2.5 diagnostic: failed ORB / reversion on relative log spreads.",
        "A failed breakout is a close outside the opening range followed by a close back inside the range.",
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
        "- Treat this as a screening run. Failed ORB only matters if it beats continuation, timing controls and directional failed-ORB baselines after costs.",
        "- If validation is promising but test is weak, do not add filters until fold-level concentration is inspected.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path,
    config: EquityOrbConfig,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: FailedOrbOutputs,
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
            "trade_policy": "first_failed_breakout_per_session",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H2.5 equity failed ORB pairs diagnostic")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_strategy(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

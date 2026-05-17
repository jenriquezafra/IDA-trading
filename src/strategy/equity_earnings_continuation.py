from __future__ import annotations

"""H3 v1 earnings continuation strategy runner."""

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
from src.strategy.equity_earnings_continuation_screening import (
    H3ScreeningConfig,
    aggregate_trades,
    apply_costs,
    build_coverage,
    build_symbol_session_prices,
    fit_rel_volume_threshold,
    h3_candidate_mask,
    load_config as load_screening_config,
    load_events,
    load_panel,
    make_labeled_events,
    simulate_trades,
    _tradeable_events,
)


DEFAULT_CONFIG_PATH = Path("configs/strategy/equity_earnings_continuation_h3_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/strategy/equity_earnings_continuation/5min/h3_v1")

PRIMARY_LABEL = "h3_v1_primary"
SENSITIVITY_LABEL = "h3_v1_sensitivity"


@dataclass(frozen=True)
class H3StrategyConfig:
    base: H3ScreeningConfig
    output_dir: Path
    primary_horizon: str
    sensitivity_horizons: tuple[str, ...]
    max_positions_per_session: int
    max_symbol_weight_per_session: float


@dataclass(frozen=True)
class H3StrategyOutputs:
    output_dir: Path
    coverage_path: Path
    events_path: Path
    trades_path: Path
    daily_path: Path
    monthly_path: Path
    summary_path: Path
    distribution_path: Path
    thresholds_path: Path
    manifest_path: Path
    report_path: Path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> H3StrategyConfig:
    config_path = Path(path)
    base = load_screening_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"H3 strategy config must be a mapping: {path}")
    outputs_cfg = dict(raw.get("outputs", {}))
    events_cfg = dict(raw.get("events", {}))
    exit_cfg = dict(events_cfg.get("exit", {}))
    position_cfg = dict(raw.get("position", {}))
    output_dir = Path(outputs_cfg.get("strategy_output_dir", Path(outputs_cfg.get("output_dir", DEFAULT_OUTPUT_DIR.parent)) / "h3_v1"))
    primary_horizon = str(exit_cfg.get("primary_exit", "same_session_close")).strip()
    sensitivity_horizons = tuple(
        horizon
        for horizon in [str(value).strip() for value in exit_cfg.get("secondary_exits", [])]
        if horizon and horizon != primary_horizon
    )
    return H3StrategyConfig(
        base=base,
        output_dir=output_dir,
        primary_horizon=primary_horizon,
        sensitivity_horizons=sensitivity_horizons,
        max_positions_per_session=int(position_cfg.get("max_positions_per_session", 10)),
        max_symbol_weight_per_session=float(position_cfg.get("max_symbol_weight_per_session", 0.20)),
    )


def _with_output_dir(config: H3StrategyConfig, output_dir: str | Path | None) -> H3StrategyConfig:
    if output_dir is None:
        return config
    return H3StrategyConfig(
        base=config.base,
        output_dir=Path(output_dir),
        primary_horizon=config.primary_horizon,
        sensitivity_horizons=config.sensitivity_horizons,
        max_positions_per_session=config.max_positions_per_session,
        max_symbol_weight_per_session=config.max_symbol_weight_per_session,
    )


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[frame["event_session"].astype(str).isin(sessions)].copy()


def _rank_candidates(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    ranked = events.copy()
    for column in ("rel_volume_30m", "eps_surprise_z", "gap_atr"):
        if column not in ranked:
            ranked[column] = np.nan
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked.sort_values(
        ["event_session", "rel_volume_30m", "eps_surprise_z", "gap_atr", "symbol"],
        ascending=[True, False, False, True, True],
        kind="stable",
    )


def select_strategy_events(
    events: pd.DataFrame,
    config: H3StrategyConfig,
    *,
    fold: int,
    split: str,
    rel_volume_threshold: float,
) -> pd.DataFrame:
    mask = h3_candidate_mask(events, config.base, rel_volume_threshold)
    selected = _rank_candidates(events.loc[mask].copy())
    if selected.empty:
        return selected
    if config.max_positions_per_session > 0:
        selected = selected.groupby("event_session", sort=False).head(config.max_positions_per_session).copy()
    labelled = make_labeled_events(selected, PRIMARY_LABEL, fold, split)
    labelled["rel_volume_threshold"] = rel_volume_threshold
    labelled["max_positions_per_session"] = int(config.max_positions_per_session)
    return labelled


def build_strategy_events_and_trades(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    config: H3StrategyConfig,
    folds: tuple[ResearchFold, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], tuple[str, ...]], pd.DataFrame]:
    selected_events: list[pd.DataFrame] = []
    trade_bases: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}

    for fold in folds:
        train_events = _split_frame(events, fold.train_sessions)
        rel_volume_threshold = fit_rel_volume_threshold(train_events.loc[_tradeable_events(train_events)], config.base)
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
            split_sessions[(fold.fold, split)] = tuple(sessions)
            split_events = _split_frame(events, sessions)
            candidates = select_strategy_events(
                split_events,
                config,
                fold=fold.fold,
                split=split,
                rel_volume_threshold=rel_volume_threshold,
            )
            if candidates.empty:
                continue
            selected_events.append(candidates)
            primary = simulate_trades(
                candidates,
                prices,
                strategy_id=config.base.strategy_id,
                horizon=config.primary_horizon,
            )
            if not primary.empty:
                trade_bases.append(primary)
            for horizon in config.sensitivity_horizons:
                sensitivity_events = candidates.copy()
                sensitivity_events["label"] = SENSITIVITY_LABEL
                sensitivity = simulate_trades(
                    sensitivity_events,
                    prices,
                    strategy_id=config.base.strategy_id,
                    horizon=horizon,
                )
                if not sensitivity.empty:
                    trade_bases.append(sensitivity)

    labeled = pd.concat(selected_events, ignore_index=True) if selected_events else pd.DataFrame()
    raw_trades = pd.concat(trade_bases, ignore_index=True) if trade_bases else pd.DataFrame()
    trades = apply_costs(raw_trades, config.base.cost_bps_values)
    thresholds = pd.DataFrame(threshold_rows)
    return labeled, trades, split_sessions, thresholds


def _write_manifest(
    path: Path,
    config: H3StrategyConfig,
    config_path: str | Path,
    folds: tuple[ResearchFold, ...],
    outputs: H3StrategyOutputs,
    thresholds: pd.DataFrame,
) -> None:
    config_file = Path(config_path)
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("strategy", config.base.strategy_id, "h3_v1", config.base.timeframe),
            "run_type": "h3_v1_strategy_backtest",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": config.base.strategy_id,
            "hypothesis_id": config.base.hypothesis_id,
            "phase": "Fase 4 - Estrategia H3 v1",
            "side": "long_only",
            "entry_rule": "first_bar_open_at_or_after_10_00_et",
            "primary_exit": config.primary_horizon,
            "sensitivity_exits": list(config.sensitivity_horizons),
            "labels": [PRIMARY_LABEL, SENSITIVITY_LABEL],
            "max_positions_per_session": config.max_positions_per_session,
            "max_symbol_weight_per_session": config.max_symbol_weight_per_session,
        },
        "data": {
            "earnings_events_path": config.base.earnings_events_path.as_posix(),
            "earnings_events_fingerprint": fingerprint_path(config.base.earnings_events_path) if config.base.earnings_events_path.exists() else "MISSING",
            "intraday_panel_path": config.base.intraday_panel_path.as_posix(),
            "intraday_panel_fingerprint": fingerprint_path(config.base.intraday_panel_path) if config.base.intraday_panel_path.exists() else "MISSING",
            "config_path": config_file.as_posix(),
            "config_fingerprint": fingerprint_path(config_file) if config_file.exists() else "MISSING",
            "timeframe": config.base.timeframe,
            "split_policy": config.base.split_policy,
            "n_folds": len(folds),
            "fit_thresholds": thresholds.to_dict(orient="records") if not thresholds.empty else [],
        },
        "parameters": {
            "cost_bps_values": list(config.base.cost_bps_values),
            "signal_filters": config.base.signal_filters,
        },
        "outputs": {
            "coverage": outputs.coverage_path.as_posix(),
            "events": outputs.events_path.as_posix(),
            "trades": outputs.trades_path.as_posix(),
            "daily": outputs.daily_path.as_posix(),
            "monthly": outputs.monthly_path.as_posix(),
            "summary": outputs.summary_path.as_posix(),
            "distribution": outputs.distribution_path.as_posix(),
            "thresholds": outputs.thresholds_path.as_posix(),
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
    grouped["_label_order"] = grouped["label"].map({PRIMARY_LABEL: 0, SENSITIVITY_LABEL: 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "_label_order", "horizon"], kind="stable").drop(columns=["_split_order", "_label_order"])


def _write_report(path: Path, config: H3StrategyConfig, coverage: pd.DataFrame, summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    conservative_cost = 10.0 if 10.0 in config.base.cost_bps_values else float(config.base.cost_bps_values[-1])
    rollup = _rollup(summary, conservative_cost)
    lines = [
        "# H3 v1 strategy backtest",
        "",
        "This is the first H3 strategy runner. Primary selection uses same-session close only; T+1 exits are sensitivity rows.",
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
        "## Contract",
        "",
        f"- Primary label: `{PRIMARY_LABEL}` with horizon `{config.primary_horizon}`.",
        f"- Sensitivity label: `{SENSITIVITY_LABEL}` with horizons `{', '.join(config.sensitivity_horizons) or 'none'}`.",
        "- Train-fitted thresholds are recorded in `thresholds.parquet` and manifest.",
        "- Real interpretation remains blocked until Benzinga Earnings and exclusion inputs are point-in-time.",
        "",
    ]
    if not distribution.empty:
        preview = distribution.loc[
            distribution["cost_bps_round_trip"].eq(conservative_cost)
            & distribution["metric"].eq("net_return")
            & distribution["split"].isin(["validation", "test"])
        ].copy()
        lines.extend(["## Distribution Preview", "", *_markdown_table(preview, ["split", "label", "horizon", "count", "mean", "p05", "median", "p95"], limit=40)])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_strategy(config_path: str | Path = DEFAULT_CONFIG_PATH, output_dir: str | Path | None = None) -> H3StrategyOutputs:
    config = _with_output_dir(load_config(config_path), output_dir)
    events = load_events(config.base.earnings_events_path)
    panel = load_panel(config.base.intraday_panel_path, config.base)
    prices = build_symbol_session_prices(panel, config.base)
    folds = build_monthly_folds(events.rename(columns={"event_session": "session"}), config.base.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")

    root = config.output_dir
    outputs = H3StrategyOutputs(
        output_dir=root,
        coverage_path=root / "coverage.parquet",
        events_path=root / "events.parquet",
        trades_path=root / "trades.parquet",
        daily_path=root / "daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        distribution_path=root / "distribution.parquet",
        thresholds_path=root / "thresholds.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)

    selected_events, trades, split_sessions, thresholds = build_strategy_events_and_trades(events, prices, config, folds)
    summary, daily, monthly, distribution = aggregate_trades(trades, split_sessions)
    coverage = build_coverage(events, prices, selected_events, trades)

    coverage.to_parquet(outputs.coverage_path, index=False)
    selected_events.to_parquet(outputs.events_path, index=False)
    trades.to_parquet(outputs.trades_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    distribution.to_parquet(outputs.distribution_path, index=False)
    thresholds.to_parquet(outputs.thresholds_path, index=False)
    _write_manifest(outputs.manifest_path, config, config_path, folds, outputs, thresholds)
    _write_report(outputs.report_path, config, coverage, summary, distribution)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H3 v1 earnings continuation strategy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    outputs = run_strategy(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

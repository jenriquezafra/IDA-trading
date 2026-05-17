from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.alpha.risk_off_eda import DEFAULT_FEATURES_PATH, DEFAULT_HORIZONS, DEFAULT_RISK_CONTEXT_PATH, load_eda_frame
from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds


DEFAULT_OUTPUT_DIR = Path("results/strategy/risk_off_short/QQQ/15min")
DEFAULT_SPLIT_POLICY = {"train_months": 24, "validation_months": 6, "test_months": 6, "step_months": 6, "embargo_sessions": 1}
DEFAULT_COST_BPS = (1.0, 2.0, 5.0)
CANDIDATE_LABEL = "target_breakdown__risk_off__vix_pressure"


@dataclass(frozen=True)
class RiskOffThresholds:
    risk_off_min: float
    vix_z20_min: float
    active_hours: tuple[int, ...]


@dataclass(frozen=True)
class RiskOffStrategyOutputs:
    output_dir: Path
    trades_path: Path
    daily_path: Path
    monthly_path: Path
    summary_path: Path
    manifest_path: Path
    report_path: Path


def _finite_quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def fit_thresholds(train: pd.DataFrame, risk_off_quantile: float = 0.70, vix_quantile: float = 0.70) -> RiskOffThresholds:
    risk_off_min = _finite_quantile(train, "risk_off_score", risk_off_quantile)
    vix_z20_min = _finite_quantile(train, "prev_vix_z20", vix_quantile)
    candidate = candidate_signal(train, risk_off_min=risk_off_min, vix_z20_min=vix_z20_min)
    active_hours = tuple(sorted(int(value) for value in train.loc[candidate, "hour"].dropna().unique().tolist()))
    return RiskOffThresholds(risk_off_min=risk_off_min, vix_z20_min=vix_z20_min, active_hours=active_hours)


def candidate_signal(frame: pd.DataFrame, *, risk_off_min: float, vix_z20_min: float) -> pd.Series:
    return (
        frame["target_ret_6"].lt(0.0)
        & frame["target_ret_12"].lt(0.0)
        & frame["risk_off_score"].ge(risk_off_min)
        & frame["prev_vix_z20"].ge(vix_z20_min)
    ).fillna(False)


def control_masks(frame: pd.DataFrame, thresholds: RiskOffThresholds, *, horizon: int, random_seed: int = 42) -> dict[str, pd.Series]:
    candidate = candidate_signal(frame, risk_off_min=thresholds.risk_off_min, vix_z20_min=thresholds.vix_z20_min)
    target_breakdown = (frame["target_ret_6"].lt(0.0) & frame["target_ret_12"].lt(0.0)).fillna(False)
    risk_off_top = frame["risk_off_score"].ge(thresholds.risk_off_min).fillna(False)
    vix_pressure = frame["prev_vix_z20"].ge(thresholds.vix_z20_min).fillna(False)
    same_hour = frame["hour"].isin(thresholds.active_hours)
    valid = _valid_exit_mask(frame, horizon)
    rng = np.random.default_rng(random_seed + int(horizon) + len(frame))
    random_mask = pd.Series(False, index=frame.index)
    candidate_count = int((candidate & valid).sum())
    valid_indices = np.flatnonzero(valid.to_numpy())
    if candidate_count > 0 and len(valid_indices) >= candidate_count:
        random_mask.iloc[rng.choice(valid_indices, size=candidate_count, replace=False)] = True
    return {
        CANDIDATE_LABEL: candidate,
        "target_breakdown": target_breakdown,
        "risk_off_top30": risk_off_top,
        "vix_pressure": vix_pressure,
        "same_hour_short_control": same_hour,
        "random_same_count_control": random_mask,
        "always_flat": pd.Series(False, index=frame.index),
    }


def _valid_exit_mask(frame: pd.DataFrame, horizon: int) -> pd.Series:
    entry = frame["target_open_next"].replace([np.inf, -np.inf], np.nan).astype(float)
    exit_px = frame.groupby("session", sort=False)["target_open_next"].shift(-int(horizon)).replace([np.inf, -np.inf], np.nan).astype(float)
    valid = entry.gt(0.0) & exit_px.gt(0.0)
    if "target_can_open_trade" in frame:
        valid &= frame["target_can_open_trade"].fillna(False).astype(bool)
    return valid.fillna(False)


def simulate_trades(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    label: str,
    fold: int,
    split: str,
    horizon: int,
    cost_bps: float,
    thresholds: RiskOffThresholds,
) -> pd.DataFrame:
    return simulate_trades_for_costs(
        frame,
        signal,
        label=label,
        fold=fold,
        split=split,
        horizon=horizon,
        cost_bps_values=(cost_bps,),
        thresholds=thresholds,
    )


def simulate_trades_for_costs(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    label: str,
    fold: int,
    split: str,
    horizon: int,
    cost_bps_values: tuple[float, ...],
    thresholds: RiskOffThresholds,
) -> pd.DataFrame:
    if label == "always_flat":
        return pd.DataFrame()
    signal_values = signal.reindex(frame.index).fillna(False).astype(bool).to_numpy()
    prepared = frame.reset_index(drop=True).copy()
    prepared["_entry_px"] = prepared["target_open_next"].astype(float)
    prepared["_exit_px"] = prepared.groupby("session", sort=False)["target_open_next"].shift(-int(horizon)).astype(float)
    if "target_next_open_timestamp" in prepared:
        prepared["_entry_timestamp"] = prepared["target_next_open_timestamp"]
        prepared["_exit_timestamp"] = prepared.groupby("session", sort=False)["target_next_open_timestamp"].shift(-int(horizon))
    else:
        prepared["_entry_timestamp"] = prepared.groupby("session", sort=False)["timestamp"].shift(-1)
        prepared["_exit_timestamp"] = prepared.groupby("session", sort=False)["timestamp"].shift(-(int(horizon) + 1))
    valid_values = _valid_exit_mask(prepared, horizon).to_numpy()
    candidate_values = signal_values & valid_values

    entry_values = prepared["_entry_px"].to_numpy(dtype=float)
    exit_values = prepared["_exit_px"].to_numpy(dtype=float)
    session_values = prepared["session"].to_numpy()
    timestamp_values = prepared["timestamp"].to_numpy()
    entry_timestamp_values = prepared["_entry_timestamp"].to_numpy()
    exit_timestamp_values = prepared["_exit_timestamp"].to_numpy()
    bar_values = prepared["bar_index"].to_numpy(dtype=int)
    risk_off_values = prepared["risk_off_score"].to_numpy(dtype=float)
    vix_values = prepared["prev_vix_z20"].to_numpy(dtype=float)
    target_ret_6_values = prepared["target_ret_6"].to_numpy(dtype=float)
    target_ret_12_values = prepared["target_ret_12"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for positions in prepared.groupby("session", sort=False).indices.values():
        next_allowed_pos = 0
        local_candidates = np.flatnonzero(candidate_values[positions])
        for local_pos in local_candidates:
            if local_pos < next_allowed_pos:
                continue
            pos = int(positions[local_pos])
            entry_px = float(entry_values[pos])
            exit_px = float(exit_values[pos])
            gross = -float(np.log(exit_px / entry_px))
            rows.append(
                {
                    "strategy_id": "risk_off_short_v1",
                    "label": label,
                    "fold": int(fold),
                    "split": split,
                    "horizon_bars": int(horizon),
                    "session": session_values[pos],
                    "signal_timestamp": timestamp_values[pos],
                    "entry_timestamp": entry_timestamp_values[pos],
                    "exit_timestamp": exit_timestamp_values[pos],
                    "bar_index": int(bar_values[pos]),
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "side": "short",
                    "gross_return": gross,
                    "risk_off_min": thresholds.risk_off_min,
                    "vix_z20_min": thresholds.vix_z20_min,
                    "risk_off_score": risk_off_values[pos],
                    "prev_vix_z20": vix_values[pos],
                    "target_ret_6": target_ret_6_values[pos],
                    "target_ret_12": target_ret_12_values[pos],
                }
            )
            next_allowed_pos = local_pos + int(horizon) + 1
    base = pd.DataFrame(rows)
    if base.empty:
        return base
    by_cost: list[pd.DataFrame] = []
    for cost_bps in cost_bps_values:
        costed = base.copy()
        cost_return = float(cost_bps) / 10_000.0
        costed["cost_bps"] = float(cost_bps)
        costed["cost_return"] = cost_return
        costed["net_return"] = costed["gross_return"] - cost_return
        by_cost.append(costed)
    return pd.concat(by_cost, ignore_index=True)


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


def aggregate_trades(trades: pd.DataFrame, split_sessions: dict[tuple[int, str], tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["strategy_id", "label", "fold", "split", "horizon_bars", "cost_bps"]
    daily_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for key_values, group in trades.groupby(keys, sort=False):
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
                "total_cost": float(group["cost_return"].sum()),
                "net_return": float(net.sum()),
                "avg_trade_net": float(net.mean()) if len(net) else 0.0,
                "profit_factor": _profit_factor(net),
                "daily_sharpe": _daily_sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
                "worst_day": str(daily.idxmin()) if len(daily) else "",
                "worst_day_net": float(daily.min()) if len(daily) else 0.0,
                "best_day": str(daily.idxmax()) if len(daily) else "",
                "best_day_net": float(daily.max()) if len(daily) else 0.0,
                "sessions": int(len(daily)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), pd.DataFrame(monthly_rows)


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["session"].astype(str).isin(sessions)].copy()


def run_strategy(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cost_bps_values: tuple[float, ...] = DEFAULT_COST_BPS,
    split_policy: dict[str, Any] | None = None,
) -> RiskOffStrategyOutputs:
    split_policy = dict(split_policy or DEFAULT_SPLIT_POLICY)
    frame = load_eda_frame(features_path, risk_context_path, horizons)
    folds = build_monthly_folds(frame, split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")

    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    all_trades: list[pd.DataFrame] = []
    for fold in folds:
        train = _split_frame(frame, fold.train_sessions)
        thresholds = fit_thresholds(train)
        for split, sessions in (
            ("train", fold.train_sessions),
            ("validation", fold.validation_sessions),
            ("test", fold.test_sessions),
        ):
            split_sessions[(fold.fold, split)] = tuple(sessions)
            split_frame = _split_frame(frame, sessions)
            for horizon in horizons:
                masks = control_masks(split_frame, thresholds, horizon=horizon, random_seed=10_000 + fold.fold)
                for label, signal in masks.items():
                    trades = simulate_trades_for_costs(
                        split_frame,
                        signal,
                        label=label,
                        fold=fold.fold,
                        split=split,
                        horizon=horizon,
                        cost_bps_values=cost_bps_values,
                        thresholds=thresholds,
                    )
                    if not trades.empty:
                        all_trades.append(trades)

    trades_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary, daily, monthly = aggregate_trades(trades_frame, split_sessions)
    root = Path(output_dir)
    outputs = RiskOffStrategyOutputs(
        output_dir=root,
        trades_path=root / "trades.parquet",
        daily_path=root / "daily.parquet",
        monthly_path=root / "monthly.parquet",
        summary_path=root / "summary.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    trades_frame.to_parquet(outputs.trades_path, index=False)
    daily.to_parquet(outputs.daily_path, index=False)
    monthly.to_parquet(outputs.monthly_path, index=False)
    summary.to_parquet(outputs.summary_path, index=False)
    _write_manifest(outputs.manifest_path, Path(features_path), Path(risk_context_path), folds, split_policy)
    _write_report(outputs.report_path, summary)
    return outputs


def _write_manifest(path: Path, features_path: Path, risk_context_path: Path, folds: tuple[ResearchFold, ...], split_policy: dict[str, Any]) -> None:
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("strategy", "risk_off_short_v1", "QQQ", "15min"),
            "run_type": "strategy_backtest",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": "risk_off_short_v1",
            "hypothesis": "risk_off_short_continuation",
            "candidate_label": CANDIDATE_LABEL,
            "entry_rule": "next_open",
            "exit_rule": "fixed_horizon_open",
            "position": "short",
            "overlap_policy": "skip_new_entries_while_trade_open",
            "threshold_fit": "train_fold_quantiles",
        },
        "data": {
            "features_path": features_path.as_posix(),
            "features_fingerprint": fingerprint_path(features_path) if features_path.exists() else "MISSING",
            "risk_context_path": risk_context_path.as_posix(),
            "risk_context_fingerprint": fingerprint_path(risk_context_path) if risk_context_path.exists() else "MISSING",
            "split_policy": split_policy,
            "n_folds": len(folds),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 16) -> list[str]:
    if frame.empty:
        return ["No rows."]
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    integer_columns = {column: pd.api.types.is_integer_dtype(visible[column]) for column in visible.columns}
    lines = ["| " + " | ".join(visible.columns) + " |", "| " + " | ".join(["---"] * len(visible.columns)) + " |"]
    for _, row in visible.iterrows():
        values: list[str] = []
        for column in visible.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif integer_columns[column]:
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _candidate_rollup(summary: pd.DataFrame) -> pd.DataFrame:
    candidate = summary[
        summary["label"].eq(CANDIDATE_LABEL) & summary["cost_bps"].eq(2.0) & summary["split"].isin(["validation", "test"])
    ]
    if candidate.empty:
        return pd.DataFrame()
    grouped = (
        candidate.groupby(["split", "horizon_bars"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
        )
    )
    grouped["avg_trade_net"] = grouped["net_return"] / grouped["trades"].replace(0, np.nan)
    grouped["_split_order"] = grouped["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return grouped.sort_values(["_split_order", "horizon_bars"], kind="stable").drop(columns="_split_order")


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    rollup = _candidate_rollup(summary)
    validation_h4 = summary[
        summary["split"].eq("validation") & summary["cost_bps"].eq(2.0) & summary["horizon_bars"].eq(4)
    ].sort_values("net_return", ascending=False, kind="stable")
    validation_h6 = summary[
        summary["split"].eq("validation") & summary["cost_bps"].eq(2.0) & summary["horizon_bars"].eq(6)
    ].sort_values("net_return", ascending=False, kind="stable")
    validation_candidate = summary[
        summary["split"].eq("validation") & summary["label"].eq(CANDIDATE_LABEL) & summary["cost_bps"].eq(2.0)
    ].sort_values(["horizon_bars", "fold"], kind="stable")
    test_candidate = summary[
        summary["split"].eq("test") & summary["label"].eq(CANDIDATE_LABEL) & summary["cost_bps"].eq(2.0)
    ].sort_values(["horizon_bars", "fold"], kind="stable")
    lines = [
        "# Risk-off short strategy diagnostic",
        "",
        "This is a rule-based strategy diagnostic, not an ML model.",
        "Thresholds are fit on train folds and evaluated on validation/test.",
        "",
        "## Candidate aggregate, cost=2 bps",
        "",
        *_markdown_table(
            rollup,
            ["split", "horizon_bars", "folds", "trades", "net_return", "avg_trade_net", "positive_folds"],
            limit=12,
        ),
        "",
        "## Read",
        "",
        "- h=6 is the only candidate horizon positive in both aggregate validation and aggregate test at 2 bps.",
        "- h=6 is still not production-ready: validation has losing folds and some controls are competitive in specific windows.",
        "- h=2 and h=3 fail validation; h=4 is roughly flat in aggregate validation.",
        "",
        "## Validation candidate by horizon, cost=2 bps",
        "",
        *_markdown_table(
            validation_candidate,
            ["fold", "horizon_bars", "trades", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown", "win_rate"],
            limit=40,
        ),
        "",
        "## Validation controls, h=4, cost=2 bps",
        "",
        *_markdown_table(
            validation_h4,
            ["label", "fold", "trades", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown", "win_rate"],
            limit=40,
        ),
        "",
        "## Validation controls, h=6, cost=2 bps",
        "",
        *_markdown_table(
            validation_h6,
            ["label", "fold", "trades", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown", "win_rate"],
            limit=40,
        ),
        "",
        "## Test candidate by horizon, cost=2 bps",
        "",
        *_markdown_table(
            test_candidate,
            ["fold", "horizon_bars", "trades", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown", "win_rate"],
            limit=40,
        ),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run risk-off short strategy diagnostic")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args(argv)
    outputs = run_strategy(
        features_path=args.features,
        risk_context_path=args.risk_context,
        output_dir=args.output_dir,
        horizons=tuple(args.horizons),
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

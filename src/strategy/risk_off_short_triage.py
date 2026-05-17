from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.alpha.risk_off_eda import DEFAULT_FEATURES_PATH, DEFAULT_RISK_CONTEXT_PATH, load_eda_frame
from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.promotion import DEFAULT_PROMOTION_GATES, evaluate_promotion_gates, rollup_by_cost
from src.research.splits import build_monthly_folds
from src.strategy.risk_off_short import (
    CANDIDATE_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_POLICY,
    aggregate_trades,
    control_masks,
    _daily_sharpe,
    _markdown_table,
    _max_drawdown,
    _profit_factor,
    _split_frame,
    candidate_signal,
    fit_thresholds,
    simulate_trades,
    simulate_trades_for_costs,
)


DEFAULT_HORIZON = 6
DEFAULT_COST_BPS = 2.0
DEFAULT_STRESS_COST_BPS = 5.0
DEFAULT_TRIAGE_DIR = DEFAULT_OUTPUT_DIR / "triage"
DEFAULT_THRESHOLD_QUANTILES = (0.60, 0.65, 0.70, 0.75, 0.80)


@dataclass(frozen=True)
class RiskOffTriageOutputs:
    output_dir: Path
    report_path: Path
    manifest_path: Path
    controls_rollup_path: Path
    candidate_by_fold_path: Path
    hour_summary_path: Path
    bucket_summary_path: Path
    weekday_summary_path: Path
    session_concentration_path: Path
    threshold_sensitivity_path: Path
    selected_threshold_confirmation_path: Path
    selected_threshold_trades_path: Path
    selected_threshold_controls_path: Path
    selected_threshold_concentration_path: Path
    promotion_gates_path: Path
    promotion_decision_path: Path


def enrich_trade_times(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    signal_time = pd.to_datetime(frame["signal_timestamp"])
    session_date = pd.to_datetime(frame["session"])
    frame["signal_hour"] = signal_time.dt.hour
    frame["signal_minute"] = signal_time.dt.minute
    frame["weekday"] = session_date.dt.day_name()
    frame["weekday_num"] = session_date.dt.dayofweek
    frame["intraday_bucket"] = pd.Categorical(
        np.select(
            [
                frame["bar_index"].lt(6),
                frame["bar_index"].between(6, 11),
                frame["bar_index"].between(12, 17),
            ],
            ["open_0930_1045", "late_morning_1100_1215", "midday_1230_1345"],
            default="afternoon_1400_close",
        ),
        categories=["open_0930_1045", "late_morning_1100_1215", "midday_1230_1345", "afternoon_1400_close"],
        ordered=True,
    )
    return frame


def trade_group_summary(trades: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    columns = [*by, "trades", "net_return", "avg_trade_net", "profit_factor", "win_rate"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key_values, group in trades.groupby(by, dropna=False, observed=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        net = group["net_return"].astype(float)
        rows.append(
            {
                **dict(zip(by, key_values, strict=True)),
                "trades": int(len(group)),
                "net_return": float(net.sum()),
                "avg_trade_net": float(net.mean()),
                "profit_factor": _profit_factor(net),
                "win_rate": float(net.gt(0.0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def control_rollup(summary: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON, cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    filtered = summary[summary["horizon_bars"].eq(horizon) & summary["cost_bps"].eq(cost_bps) & summary["split"].isin(["validation", "test"])]
    if filtered.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (split, label), group in filtered.groupby(["split", "label"], sort=False):
        trades = int(group["trades"].sum())
        net_return = float(group["net_return"].sum())
        rows.append(
            {
                "split": split,
                "label": label,
                "folds": int(group["fold"].nunique()),
                "trades": trades,
                "net_return": net_return,
                "avg_trade_net": net_return / trades if trades else 0.0,
                "positive_folds": int(group["net_return"].gt(0.0).sum()),
                "min_fold_return": float(group["net_return"].min()),
                "max_fold_return": float(group["net_return"].max()),
                "mean_daily_sharpe": float(group["daily_sharpe"].mean()),
                "max_fold_drawdown": float(group["max_drawdown"].max()),
            }
        )
    result = pd.DataFrame(rows)
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "net_return"], ascending=[True, False], kind="stable").drop(columns="_split_order")


def candidate_by_fold(summary: pd.DataFrame, *, horizon: int = DEFAULT_HORIZON, cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    columns = [
        "split",
        "fold",
        "trades",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "win_rate",
    ]
    result = summary[
        summary["label"].eq(CANDIDATE_LABEL)
        & summary["horizon_bars"].eq(horizon)
        & summary["cost_bps"].eq(cost_bps)
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "fold"], kind="stable").loc[:, columns]


def session_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "fold",
        "sessions_with_trades",
        "net_return",
        "top1_abs_share",
        "top5_abs_share",
        "best_session",
        "best_session_net",
        "worst_session",
        "worst_session_net",
        "positive_sessions",
        "negative_sessions",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    daily = trades.groupby(["split", "fold", "session"], as_index=False)["net_return"].sum()
    rows: list[dict[str, Any]] = []
    for (split, fold), group in daily.groupby(["split", "fold"], sort=True):
        net = group["net_return"].astype(float)
        abs_net = net.abs().sort_values(ascending=False)
        abs_total = float(abs_net.sum())
        best_idx = net.idxmax()
        worst_idx = net.idxmin()
        rows.append(
            {
                "split": split,
                "fold": int(fold),
                "sessions_with_trades": int(len(group)),
                "net_return": float(net.sum()),
                "top1_abs_share": float(abs_net.iloc[0] / abs_total) if abs_total else 0.0,
                "top5_abs_share": float(abs_net.head(5).sum() / abs_total) if abs_total else 0.0,
                "best_session": str(group.loc[best_idx, "session"]),
                "best_session_net": float(group.loc[best_idx, "net_return"]),
                "worst_session": str(group.loc[worst_idx, "session"]),
                "worst_session_net": float(group.loc[worst_idx, "net_return"]),
                "positive_sessions": int(net.gt(0.0).sum()),
                "negative_sessions": int(net.lt(0.0).sum()),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "fold"], kind="stable").drop(columns="_split_order")


def _sort_with_split_order(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or "split" not in frame:
        return frame
    sorted_frame = frame.copy()
    sorted_frame["_split_order"] = sorted_frame["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return sorted_frame.sort_values(["_split_order", *columns], kind="stable").drop(columns="_split_order")


def validation_threshold_sensitivity(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    split_policy: dict[str, Any] | None = None,
    risk_quantiles: tuple[float, ...] = DEFAULT_THRESHOLD_QUANTILES,
    vix_quantiles: tuple[float, ...] = DEFAULT_THRESHOLD_QUANTILES,
    horizon: int = DEFAULT_HORIZON,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    policy = dict(split_policy or DEFAULT_SPLIT_POLICY)
    frame = load_eda_frame(features_path, risk_context_path, (horizon,))
    folds = build_monthly_folds(frame, policy)
    rows: list[dict[str, Any]] = []
    for risk_q in risk_quantiles:
        for vix_q in vix_quantiles:
            fold_returns: list[float] = []
            trade_counts: list[int] = []
            for fold in folds:
                train = _split_frame(frame, fold.train_sessions)
                validation = _split_frame(frame, fold.validation_sessions)
                thresholds = fit_thresholds(train, risk_off_quantile=risk_q, vix_quantile=vix_q)
                signal = candidate_signal(validation, risk_off_min=thresholds.risk_off_min, vix_z20_min=thresholds.vix_z20_min)
                trades = simulate_trades(
                    validation,
                    signal,
                    label=CANDIDATE_LABEL,
                    fold=fold.fold,
                    split="validation",
                    horizon=horizon,
                    cost_bps=cost_bps,
                    thresholds=thresholds,
                )
                fold_returns.append(float(trades["net_return"].sum()) if not trades.empty else 0.0)
                trade_counts.append(int(len(trades)))
            total_trades = int(sum(trade_counts))
            total_net = float(sum(fold_returns))
            rows.append(
                {
                    "risk_off_quantile": float(risk_q),
                    "vix_quantile": float(vix_q),
                    "folds": int(len(folds)),
                    "trades": total_trades,
                    "net_return": total_net,
                    "avg_trade_net": total_net / total_trades if total_trades else 0.0,
                    "positive_folds": int(sum(value > 0.0 for value in fold_returns)),
                    "min_fold_return": float(min(fold_returns)) if fold_returns else 0.0,
                    "max_fold_return": float(max(fold_returns)) if fold_returns else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["net_return", "positive_folds"], ascending=[False, False], kind="stable")


def selected_threshold_confirmation(
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    split_policy: dict[str, Any] | None = None,
    horizon: int = DEFAULT_HORIZON,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    policy = dict(split_policy or DEFAULT_SPLIT_POLICY)
    frame = load_eda_frame(features_path, risk_context_path, (horizon,))
    folds = build_monthly_folds(frame, policy)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train = _split_frame(frame, fold.train_sessions)
        thresholds = fit_thresholds(train, risk_off_quantile=risk_off_quantile, vix_quantile=vix_quantile)
        for split, sessions in (("validation", fold.validation_sessions), ("test", fold.test_sessions)):
            split_frame = _split_frame(frame, sessions)
            signal = candidate_signal(split_frame, risk_off_min=thresholds.risk_off_min, vix_z20_min=thresholds.vix_z20_min)
            trades = simulate_trades(
                split_frame,
                signal,
                label=CANDIDATE_LABEL,
                fold=fold.fold,
                split=split,
                horizon=horizon,
                cost_bps=cost_bps,
                thresholds=thresholds,
            )
            net = trades["net_return"].astype(float) if not trades.empty else pd.Series(dtype=float)
            daily = trades.groupby("session")["net_return"].sum() if not trades.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "risk_off_quantile": float(risk_off_quantile),
                    "vix_quantile": float(vix_quantile),
                    "split": split,
                    "fold": int(fold.fold),
                    "trades": int(len(trades)),
                    "net_return": float(net.sum()) if len(net) else 0.0,
                    "avg_trade_net": float(net.mean()) if len(net) else 0.0,
                    "profit_factor": _profit_factor(net),
                    "daily_sharpe": _daily_sharpe(daily),
                    "max_drawdown": _max_drawdown(daily),
                    "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
                }
            )
    result = pd.DataFrame(rows)
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "fold"], kind="stable").drop(columns="_split_order")


def selected_threshold_control_backtest(
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    split_policy: dict[str, Any] | None = None,
    horizon: int = DEFAULT_HORIZON,
    cost_bps_values: tuple[float, ...] = (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy = dict(split_policy or DEFAULT_SPLIT_POLICY)
    frame = load_eda_frame(features_path, risk_context_path, (horizon,))
    folds = build_monthly_folds(frame, policy)
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    all_trades: list[pd.DataFrame] = []
    for fold in folds:
        train = _split_frame(frame, fold.train_sessions)
        thresholds = fit_thresholds(train, risk_off_quantile=risk_off_quantile, vix_quantile=vix_quantile)
        for split, sessions in (("validation", fold.validation_sessions), ("test", fold.test_sessions)):
            split_sessions[(fold.fold, split)] = tuple(sessions)
            split_frame = _split_frame(frame, sessions)
            masks = control_masks(split_frame, thresholds, horizon=horizon, random_seed=20_000 + fold.fold)
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
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary, _, _ = aggregate_trades(trades, split_sessions)
    candidate = trades[trades["label"].eq(CANDIDATE_LABEL) & trades["cost_bps"].eq(DEFAULT_COST_BPS)].copy() if not trades.empty else trades
    concentration = session_concentration(enrich_trade_times(candidate)) if not candidate.empty else pd.DataFrame()
    return trades, summary, concentration


def _selected_control_rollup(summary: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    return rollup_by_cost(summary, cost_bps=cost_bps)


def _candidate_trades(trades: pd.DataFrame, *, horizon: int, cost_bps: float) -> pd.DataFrame:
    return trades[
        trades["label"].eq(CANDIDATE_LABEL)
        & trades["horizon_bars"].eq(horizon)
        & trades["cost_bps"].eq(cost_bps)
        & trades["split"].isin(["validation", "test"])
    ].copy()


def _write_manifest(
    path: Path,
    source_dir: Path,
    outputs: RiskOffTriageOutputs,
    split_policy: dict[str, Any],
    promotion_decision: dict[str, Any],
) -> None:
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("triage", "risk_off_short_h6", "QQQ", "15min"),
            "run_type": "strategy_triage",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "strategy_id": "risk_off_short_v1",
            "candidate_label": CANDIDATE_LABEL,
            "horizon_bars": DEFAULT_HORIZON,
            "cost_bps": DEFAULT_COST_BPS,
        },
        "promotion": {
            "status": promotion_decision.get("status", "not_evaluated"),
            "failed_gates": promotion_decision.get("failed_gates", []),
        },
        "data": {
            "strategy_dir": source_dir.as_posix(),
            "summary_fingerprint": fingerprint_path(source_dir / "summary.parquet"),
            "trades_fingerprint": fingerprint_path(source_dir / "trades.parquet"),
            "split_policy": split_policy,
        },
        "outputs": {key: value.as_posix() for key, value in outputs.__dict__.items() if key.endswith("_path")},
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _write_report(
    path: Path,
    controls: pd.DataFrame,
    folds: pd.DataFrame,
    hour: pd.DataFrame,
    bucket: pd.DataFrame,
    weekday: pd.DataFrame,
    concentration: pd.DataFrame,
    sensitivity: pd.DataFrame,
    confirmation: pd.DataFrame,
    selected_controls: pd.DataFrame,
    promotion_gates: pd.DataFrame,
    promotion_decision: dict[str, Any],
) -> None:
    best_sensitivity = sensitivity.head(8) if not sensitivity.empty else sensitivity
    current = sensitivity[
        sensitivity["risk_off_quantile"].eq(0.70) & sensitivity["vix_quantile"].eq(0.70)
    ] if not sensitivity.empty else sensitivity
    candidate_validation = controls[controls["split"].eq("validation") & controls["label"].eq(CANDIDATE_LABEL)]
    candidate_test = controls[controls["split"].eq("test") & controls["label"].eq(CANDIDATE_LABEL)]
    validation_net = float(candidate_validation["net_return"].iloc[0]) if not candidate_validation.empty else 0.0
    test_net = float(candidate_test["net_return"].iloc[0]) if not candidate_test.empty else 0.0
    validation_positive_folds = int(candidate_validation["positive_folds"].iloc[0]) if not candidate_validation.empty else 0
    test_positive_folds = int(candidate_test["positive_folds"].iloc[0]) if not candidate_test.empty else 0
    confirmation_agg = (
        confirmation.groupby("split", as_index=False)
        .agg(
            trades=("trades", "sum"),
            net_return=("net_return", "sum"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
        )
        if not confirmation.empty
        else pd.DataFrame(columns=["split", "trades", "net_return", "positive_folds"])
    )
    selected_validation = confirmation_agg[confirmation_agg["split"].eq("validation")]
    selected_test = confirmation_agg[confirmation_agg["split"].eq("test")]
    selected_validation_net = float(selected_validation["net_return"].iloc[0]) if not selected_validation.empty else 0.0
    selected_test_net = float(selected_test["net_return"].iloc[0]) if not selected_test.empty else 0.0
    selected_test_positive = int(selected_test["positive_folds"].iloc[0]) if not selected_test.empty else 0
    selected_test_trades = int(selected_test["trades"].iloc[0]) if not selected_test.empty else 0
    selected_validation_trades = int(selected_validation["trades"].iloc[0]) if not selected_validation.empty else 0
    max_top5_share = (
        float(concentration.loc[concentration["split"].eq("validation"), "top5_abs_share"].max()) if not concentration.empty else 0.0
    )
    failed_gates = promotion_decision.get("failed_gates", [])
    lines = [
        "# Risk-off short h=6 triage",
        "",
        "This triage uses h=6, 2 bps round-trip cost, validation/test only.",
        "Threshold sensitivity is validation-only; test is not used to choose thresholds.",
        "",
        "## Read",
        "",
        f"- Candidate validation net: `{validation_net:.4f}` with `{validation_positive_folds}/5` positive folds.",
        f"- Candidate test net: `{test_net:.4f}` with `{test_positive_folds}/5` positive folds.",
        f"- Validation-selected q80/q80 confirmation: validation `{selected_validation_net:.4f}` over `{selected_validation_trades}` trades; test `{selected_test_net:.4f}` with `{selected_test_positive}/5` positive folds and `{selected_test_trades}` test trades.",
        f"- Main caution: fold-level concentration is high; validation top-5 session absolute share reaches `{max_top5_share:.2f}`.",
        f"- Promotion decision: `{promotion_decision.get('status', 'not_evaluated')}`.",
        "",
        "## Controls rollup",
        "",
        *_markdown_table(
            controls,
            ["split", "label", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "mean_daily_sharpe", "max_fold_drawdown"],
            limit=20,
        ),
        "",
        "## Candidate by fold",
        "",
        *_markdown_table(
            folds,
            ["split", "fold", "trades", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown", "win_rate"],
            limit=20,
        ),
        "",
        "## By intraday bucket",
        "",
        *_markdown_table(bucket, ["split", "intraday_bucket", "trades", "net_return", "avg_trade_net", "profit_factor", "win_rate"], limit=20),
        "",
        "## By signal hour",
        "",
        *_markdown_table(hour, ["split", "signal_hour", "trades", "net_return", "avg_trade_net", "profit_factor", "win_rate"], limit=30),
        "",
        "## By weekday",
        "",
        *_markdown_table(weekday, ["split", "weekday", "trades", "net_return", "avg_trade_net", "profit_factor", "win_rate"], limit=20),
        "",
        "## Session concentration",
        "",
        *_markdown_table(
            concentration,
            [
                "split",
                "fold",
                "sessions_with_trades",
                "net_return",
                "top1_abs_share",
                "top5_abs_share",
                "best_session",
                "best_session_net",
                "worst_session",
                "worst_session_net",
            ],
            limit=20,
        ),
        "",
        "## Threshold sensitivity, validation only",
        "",
        "Current q70/q70 row:",
        "",
        *_markdown_table(
            current,
            ["risk_off_quantile", "vix_quantile", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "min_fold_return"],
            limit=3,
        ),
        "",
        "Top validation rows:",
        "",
        *_markdown_table(
            best_sensitivity,
            ["risk_off_quantile", "vix_quantile", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "min_fold_return"],
            limit=8,
        ),
        "",
        "## Validation-selected threshold confirmation",
        "",
        "The threshold pair below is selected from validation sensitivity and then checked once on test.",
        "",
        *_markdown_table(
            confirmation,
            [
                "risk_off_quantile",
                "vix_quantile",
                "split",
                "fold",
                "trades",
                "net_return",
                "avg_trade_net",
                "profit_factor",
                "daily_sharpe",
                "max_drawdown",
                "win_rate",
            ],
            limit=20,
        ),
        "",
        "## Selected-threshold controls, cost=2 bps",
        "",
        *_markdown_table(
            _selected_control_rollup(selected_controls, cost_bps=DEFAULT_COST_BPS),
            ["split", "label", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "mean_daily_sharpe", "max_fold_drawdown"],
            limit=20,
        ),
        "",
        "## Promotion gates",
        "",
        "Promotion gates are hard rules for moving a candidate from research to freeze review.",
        "",
        *_markdown_table(
            promotion_gates,
            ["gate_id", "status", "observed", "threshold", "rationale"],
            limit=40,
        ),
        "",
        "Failed gates:",
        "",
        ", ".join(f"`{gate}`" for gate in failed_gates) if failed_gates else "None.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_triage(
    *,
    strategy_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_TRIAGE_DIR,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    split_policy: dict[str, Any] | None = None,
    run_sensitivity: bool = True,
) -> RiskOffTriageOutputs:
    policy = dict(split_policy or DEFAULT_SPLIT_POLICY)
    source = Path(strategy_dir)
    root = Path(output_dir)
    summary_path = source / "summary.parquet"
    trades_path = source / "trades.parquet"
    if not summary_path.exists() or not trades_path.exists():
        raise FileNotFoundError(f"missing strategy artifacts under {source}")

    summary = pd.read_parquet(summary_path)
    trades = pd.read_parquet(trades_path)
    candidate = enrich_trade_times(_candidate_trades(trades, horizon=DEFAULT_HORIZON, cost_bps=DEFAULT_COST_BPS))
    controls = control_rollup(summary)
    folds = candidate_by_fold(summary)
    hour = trade_group_summary(candidate, ["split", "signal_hour"]).sort_values(["split", "signal_hour"], kind="stable")
    hour = _sort_with_split_order(hour, ["signal_hour"])
    bucket = trade_group_summary(candidate, ["split", "intraday_bucket"])
    bucket = _sort_with_split_order(bucket, ["intraday_bucket"])
    weekday = trade_group_summary(candidate, ["split", "weekday_num", "weekday"])
    weekday = _sort_with_split_order(weekday, ["weekday_num"]).drop(columns="weekday_num", errors="ignore")
    concentration = session_concentration(candidate)
    if run_sensitivity:
        sensitivity = validation_threshold_sensitivity(
            features_path=features_path,
            risk_context_path=risk_context_path,
            split_policy=policy,
        )
        selected = sensitivity.iloc[0]
        confirmation = selected_threshold_confirmation(
            risk_off_quantile=float(selected["risk_off_quantile"]),
            vix_quantile=float(selected["vix_quantile"]),
            features_path=features_path,
            risk_context_path=risk_context_path,
            split_policy=policy,
        )
        selected_trades, selected_controls, selected_concentration = selected_threshold_control_backtest(
            risk_off_quantile=float(selected["risk_off_quantile"]),
            vix_quantile=float(selected["vix_quantile"]),
            features_path=features_path,
            risk_context_path=risk_context_path,
            split_policy=policy,
        )
        promotion_gates, promotion_decision = evaluate_promotion_gates(selected_controls, selected_concentration, candidate_label=CANDIDATE_LABEL)
    else:
        sensitivity = pd.DataFrame(
            columns=[
                "risk_off_quantile",
                "vix_quantile",
                "folds",
                "trades",
                "net_return",
                "avg_trade_net",
                "positive_folds",
                "min_fold_return",
                "max_fold_return",
            ]
        )
        confirmation = pd.DataFrame(
            columns=[
                "risk_off_quantile",
                "vix_quantile",
                "split",
                "fold",
                "trades",
                "net_return",
                "avg_trade_net",
                "profit_factor",
                "daily_sharpe",
                "max_drawdown",
                "win_rate",
            ]
        )
        selected_trades = pd.DataFrame()
        selected_controls = pd.DataFrame()
        selected_concentration = pd.DataFrame()
        promotion_gates, promotion_decision = evaluate_promotion_gates(selected_controls, selected_concentration, candidate_label=CANDIDATE_LABEL)

    outputs = RiskOffTriageOutputs(
        output_dir=root,
        report_path=root / "report.md",
        manifest_path=root / "manifest.yaml",
        controls_rollup_path=root / "controls_rollup.parquet",
        candidate_by_fold_path=root / "candidate_by_fold.parquet",
        hour_summary_path=root / "hour_summary.parquet",
        bucket_summary_path=root / "bucket_summary.parquet",
        weekday_summary_path=root / "weekday_summary.parquet",
        session_concentration_path=root / "session_concentration.parquet",
        threshold_sensitivity_path=root / "threshold_sensitivity.parquet",
        selected_threshold_confirmation_path=root / "selected_threshold_confirmation.parquet",
        selected_threshold_trades_path=root / "selected_threshold_trades.parquet",
        selected_threshold_controls_path=root / "selected_threshold_controls.parquet",
        selected_threshold_concentration_path=root / "selected_threshold_concentration.parquet",
        promotion_gates_path=root / "promotion_gates.parquet",
        promotion_decision_path=root / "promotion_decision.yaml",
    )
    root.mkdir(parents=True, exist_ok=True)
    controls.to_parquet(outputs.controls_rollup_path, index=False)
    folds.to_parquet(outputs.candidate_by_fold_path, index=False)
    hour.to_parquet(outputs.hour_summary_path, index=False)
    bucket.to_parquet(outputs.bucket_summary_path, index=False)
    weekday.to_parquet(outputs.weekday_summary_path, index=False)
    concentration.to_parquet(outputs.session_concentration_path, index=False)
    sensitivity.to_parquet(outputs.threshold_sensitivity_path, index=False)
    confirmation.to_parquet(outputs.selected_threshold_confirmation_path, index=False)
    selected_trades.to_parquet(outputs.selected_threshold_trades_path, index=False)
    selected_controls.to_parquet(outputs.selected_threshold_controls_path, index=False)
    selected_concentration.to_parquet(outputs.selected_threshold_concentration_path, index=False)
    promotion_gates.to_parquet(outputs.promotion_gates_path, index=False)
    outputs.promotion_decision_path.write_text(yaml.safe_dump(promotion_decision, sort_keys=False), encoding="utf-8")
    _write_report(
        outputs.report_path,
        controls,
        folds,
        hour,
        bucket,
        weekday,
        concentration,
        sensitivity,
        confirmation,
        selected_controls,
        promotion_gates,
        promotion_decision,
    )
    _write_manifest(outputs.manifest_path, source, outputs, policy, promotion_decision)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run risk-off short h=6 triage")
    parser.add_argument("--strategy-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_TRIAGE_DIR))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    outputs = run_triage(
        strategy_dir=args.strategy_dir,
        output_dir=args.output_dir,
        features_path=args.features,
        risk_context_path=args.risk_context,
        run_sensitivity=not args.skip_sensitivity,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

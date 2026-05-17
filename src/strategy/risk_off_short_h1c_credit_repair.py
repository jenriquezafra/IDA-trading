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
from src.research.splits import ResearchFold, build_monthly_folds
from src.strategy.risk_off_short import (
    CANDIDATE_LABEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_POLICY,
    RiskOffThresholds,
    _finite_quantile,
    _markdown_table,
    _valid_exit_mask,
    aggregate_trades,
    control_masks,
    fit_thresholds,
    simulate_trades_for_costs,
)
from src.strategy.risk_off_short_triage import (
    DEFAULT_COST_BPS,
    DEFAULT_HORIZON,
    DEFAULT_STRESS_COST_BPS,
    enrich_trade_times,
    session_concentration,
)


DEFAULT_H1C_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "h1c_credit_repair"
DEFAULT_H1C_RISK_QUANTILES = (0.50, 0.55, 0.60)
DEFAULT_H1C_VIX_QUANTILES = (0.40, 0.45, 0.50)
DEFAULT_H1C_CREDIT_POLICIES = (
    "credit_q50",
    "credit_spread_lte_0",
    "relret_hyg_lqd_lte_0",
    "credit_q50_plus_025iqr",
    "credit_q50_minus_025iqr",
    "credit_q50_or_risk_on_low_q50",
    "credit_spread_lte_0_or_risk_on_low_q50",
    "credit_spread_lte_0_and_defensive_high_q50",
)
EXTRA_COST_BPS = (7.5, 10.0)


@dataclass(frozen=True)
class CreditRepairFilter:
    policy: str
    thresholds: dict[str, float | str]


@dataclass(frozen=True)
class H1CCreditRepairOutputs:
    output_dir: Path
    report_path: Path
    manifest_path: Path
    validation_sweep_path: Path
    validation_gates_path: Path
    selected_variant_path: Path
    selected_trades_path: Path
    selected_controls_path: Path
    selected_concentration_path: Path
    selected_gates_path: Path
    selected_cost_sensitivity_path: Path
    selected_decision_path: Path


def h1c_variant_id(risk_off_quantile: float, vix_quantile: float, credit_policy: str) -> str:
    return f"riskq{int(round(risk_off_quantile * 100)):02d}__vixq{int(round(vix_quantile * 100)):02d}__{credit_policy}"


def _clean_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def _iqr(frame: pd.DataFrame, column: str) -> float:
    values = _clean_values(frame, column)
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def is_policy_interpretable(policy: str) -> bool:
    return policy in {
        "credit_spread_lte_0",
        "relret_hyg_lqd_lte_0",
        "credit_q50_plus_025iqr",
        "credit_q50_or_risk_on_low_q50",
        "credit_spread_lte_0_or_risk_on_low_q50",
        "credit_spread_lte_0_and_defensive_high_q50",
    }


def credit_policy_rank(policy: str) -> int:
    ranks = {
        "credit_spread_lte_0": 0,
        "relret_hyg_lqd_lte_0": 1,
        "credit_spread_lte_0_or_risk_on_low_q50": 2,
        "credit_spread_lte_0_and_defensive_high_q50": 3,
        "credit_q50_plus_025iqr": 4,
        "credit_q50_or_risk_on_low_q50": 5,
        "credit_q50": 6,
        "credit_q50_minus_025iqr": 7,
    }
    return ranks.get(policy, 99)


def fit_credit_repair_filter(train: pd.DataFrame, policy: str) -> CreditRepairFilter:
    median_credit = _finite_quantile(train, "spread_credit_12", 0.50)
    credit_iqr = _iqr(train, "spread_credit_12")
    thresholds: dict[str, float | str] = {}
    if policy == "credit_q50":
        thresholds["spread_credit_12_max"] = median_credit
    elif policy == "credit_spread_lte_0":
        thresholds["spread_credit_12_max"] = 0.0
    elif policy == "relret_hyg_lqd_lte_0":
        thresholds["relret_HYG_LQD_12_max"] = 0.0
    elif policy == "credit_q50_plus_025iqr":
        thresholds["spread_credit_12_max"] = median_credit + 0.25 * credit_iqr
    elif policy == "credit_q50_minus_025iqr":
        thresholds["spread_credit_12_max"] = median_credit - 0.25 * credit_iqr
    elif policy == "credit_q50_or_risk_on_low_q50":
        thresholds["spread_credit_12_max"] = median_credit
        thresholds["risk_on_score_max"] = _finite_quantile(train, "risk_on_score", 0.50)
        thresholds["combine"] = "or"
    elif policy == "credit_spread_lte_0_or_risk_on_low_q50":
        thresholds["spread_credit_12_max"] = 0.0
        thresholds["risk_on_score_max"] = _finite_quantile(train, "risk_on_score", 0.50)
        thresholds["combine"] = "or"
    elif policy == "credit_spread_lte_0_and_defensive_high_q50":
        thresholds["spread_credit_12_max"] = 0.0
        thresholds["defensive_rotation_score_min"] = _finite_quantile(train, "defensive_rotation_score", 0.50)
        thresholds["combine"] = "and"
    else:
        raise ValueError(f"unsupported H1c credit policy: {policy}")
    return CreditRepairFilter(policy=policy, thresholds=thresholds)


def apply_credit_repair_filter(frame: pd.DataFrame, fitted_filter: CreditRepairFilter) -> pd.Series:
    thresholds = fitted_filter.thresholds
    policy = fitted_filter.policy
    if policy == "relret_hyg_lqd_lte_0":
        return frame["relret_HYG_LQD_12"].le(float(thresholds["relret_HYG_LQD_12_max"])).fillna(False)

    credit = frame["spread_credit_12"].le(float(thresholds["spread_credit_12_max"])).fillna(False)
    if policy in {"credit_q50", "credit_spread_lte_0", "credit_q50_plus_025iqr", "credit_q50_minus_025iqr"}:
        return credit

    if policy in {"credit_q50_or_risk_on_low_q50", "credit_spread_lte_0_or_risk_on_low_q50"}:
        risk_on = frame["risk_on_score"].le(float(thresholds["risk_on_score_max"])).fillna(False)
        return credit | risk_on

    if policy == "credit_spread_lte_0_and_defensive_high_q50":
        defensive = frame["defensive_rotation_score"].ge(float(thresholds["defensive_rotation_score_min"])).fillna(False)
        return credit & defensive

    raise ValueError(f"unsupported H1c credit policy: {policy}")


def h1c_control_masks(
    frame: pd.DataFrame,
    thresholds: RiskOffThresholds,
    fitted_filter: CreditRepairFilter,
    *,
    horizon: int,
    random_seed: int,
) -> dict[str, pd.Series]:
    masks = control_masks(frame, thresholds, horizon=horizon, random_seed=random_seed)
    candidate = masks[CANDIDATE_LABEL] & apply_credit_repair_filter(frame, fitted_filter)
    masks[CANDIDATE_LABEL] = candidate

    valid = _valid_exit_mask(frame, horizon)
    policy_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(fitted_filter.policy))
    rng = np.random.default_rng(random_seed + int(horizon) + len(frame) + policy_seed)
    random_mask = pd.Series(False, index=frame.index)
    candidate_count = int((candidate & valid).sum())
    valid_indices = np.flatnonzero(valid.to_numpy())
    if candidate_count > 0 and len(valid_indices) >= candidate_count:
        random_mask.iloc[rng.choice(valid_indices, size=candidate_count, replace=False)] = True
    masks["random_same_count_control"] = random_mask
    return masks


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["session"].astype(str).isin(sessions)].copy()


def run_h1c_variant_backtest(
    frame: pd.DataFrame,
    folds: tuple[ResearchFold, ...],
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    credit_policy: str,
    splits: tuple[str, ...],
    cost_bps_values: tuple[float, ...] = (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS),
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    all_trades: list[pd.DataFrame] = []
    for fold in folds:
        train = _split_frame(frame, fold.train_sessions)
        thresholds = fit_thresholds(train, risk_off_quantile=risk_off_quantile, vix_quantile=vix_quantile)
        fitted_filter = fit_credit_repair_filter(train, credit_policy)
        session_map = {
            "validation": fold.validation_sessions,
            "test": fold.test_sessions,
        }
        for split in splits:
            sessions = session_map[split]
            split_sessions[(fold.fold, split)] = tuple(sessions)
            split_frame = _split_frame(frame, sessions)
            masks = h1c_control_masks(
                split_frame,
                thresholds,
                fitted_filter,
                horizon=horizon,
                random_seed=50_000 + fold.fold,
            )
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
    selected = (
        trades[trades["label"].eq(CANDIDATE_LABEL) & trades["cost_bps"].eq(DEFAULT_COST_BPS)].copy()
        if not trades.empty
        else pd.DataFrame()
    )
    concentration = session_concentration(enrich_trade_times(selected)) if not selected.empty else pd.DataFrame()
    vid = h1c_variant_id(risk_off_quantile, vix_quantile, credit_policy)
    for artifact in (trades, summary, concentration):
        if not artifact.empty:
            artifact.insert(0, "variant_id", vid)
            artifact.insert(1, "risk_off_quantile", float(risk_off_quantile))
            artifact.insert(2, "vix_quantile", float(vix_quantile))
            artifact.insert(3, "credit_policy", credit_policy)
            artifact.insert(4, "credit_policy_rank", credit_policy_rank(credit_policy))
            artifact.insert(5, "credit_policy_interpretable", is_policy_interpretable(credit_policy))
    return trades, summary, concentration


def _candidate_metric(frame: pd.DataFrame, split: str) -> pd.Series:
    match = frame[frame["split"].eq(split) & frame["label"].eq(CANDIDATE_LABEL)]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def _split_concentration(concentration: pd.DataFrame, split: str) -> tuple[int, float]:
    if concentration.empty:
        return 0, 1.0
    subset = concentration[concentration["split"].eq(split)]
    if subset.empty:
        return 0, 1.0
    return int(subset["sessions_with_trades"].min()), float(subset["top5_abs_share"].max())


def validation_sweep_row(
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    credit_policy: str,
    summary: pd.DataFrame,
    concentration: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    primary = rollup_by_cost(summary, cost_bps=DEFAULT_COST_BPS)
    stress = rollup_by_cost(summary, cost_bps=DEFAULT_STRESS_COST_BPS)
    candidate = _candidate_metric(primary, "validation")
    stress_candidate = _candidate_metric(stress, "validation")
    controls = primary[primary["split"].eq("validation") & ~primary["label"].eq(CANDIDATE_LABEL)]
    net = float(candidate.get("net_return", 0.0)) if not candidate.empty else 0.0
    best_control = float(controls["net_return"].max()) if not controls.empty else np.nan
    min_sessions, max_top5 = _split_concentration(concentration, "validation")
    return {
        "variant_id": h1c_variant_id(risk_off_quantile, vix_quantile, credit_policy),
        "risk_off_quantile": float(risk_off_quantile),
        "vix_quantile": float(vix_quantile),
        "credit_policy": credit_policy,
        "credit_policy_rank": credit_policy_rank(credit_policy),
        "credit_policy_interpretable": is_policy_interpretable(credit_policy),
        "validation_status": decision["status"],
        "failed_gate_count": int(len(decision.get("failed_gates", []))),
        "failed_gates": ",".join(decision.get("failed_gates", [])),
        "validation_trades": int(candidate.get("trades", 0)) if not candidate.empty else 0,
        "validation_net_return": net,
        "validation_stress_net_return": float(stress_candidate.get("net_return", 0.0)) if not stress_candidate.empty else 0.0,
        "validation_avg_trade_net_bps": float(candidate.get("avg_trade_net", 0.0)) * 10_000.0 if not candidate.empty else 0.0,
        "validation_positive_folds": int(candidate.get("positive_folds", 0)) if not candidate.empty else 0,
        "validation_best_control_net": best_control,
        "validation_control_edge": net - best_control if np.isfinite(best_control) else np.nan,
        "validation_min_sessions_per_fold": min_sessions,
        "validation_max_top5_abs_share": max_top5,
    }


def select_h1c_validation_variant(sweep: pd.DataFrame) -> pd.Series:
    if sweep.empty:
        return pd.Series(dtype=object)
    ranked = sweep.copy()
    ranked["passed_validation_gates"] = ranked["validation_status"].eq("freeze_review")
    ranked = ranked.sort_values(
        [
            "passed_validation_gates",
            "credit_policy_interpretable",
            "credit_policy_rank",
            "failed_gate_count",
            "validation_net_return",
            "validation_control_edge",
            "validation_positive_folds",
            "validation_trades",
        ],
        ascending=[False, False, True, True, False, False, False, False],
        kind="stable",
    )
    return ranked.iloc[0]


def build_cost_sensitivity(summary: pd.DataFrame, cost_bps_values: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost_bps in cost_bps_values:
        rollup = rollup_by_cost(summary, cost_bps=float(cost_bps))
        for split in ("validation", "test"):
            candidate = _candidate_metric(rollup, split)
            if candidate.empty:
                continue
            controls = rollup[rollup["split"].eq(split) & ~rollup["label"].eq(CANDIDATE_LABEL)]
            best_control = float(controls["net_return"].max()) if not controls.empty else np.nan
            net = float(candidate["net_return"])
            rows.append(
                {
                    "split": split,
                    "cost_bps": float(cost_bps),
                    "trades": int(candidate["trades"]),
                    "net_return": net,
                    "avg_trade_net_bps": float(candidate["avg_trade_net"]) * 10_000.0,
                    "positive_folds": int(candidate["positive_folds"]),
                    "best_control_net": best_control,
                    "control_edge": net - best_control if np.isfinite(best_control) else np.nan,
                    "mean_daily_sharpe": float(candidate["mean_daily_sharpe"]),
                    "max_fold_drawdown": float(candidate["max_fold_drawdown"]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "cost_bps"], kind="stable").drop(columns="_split_order")


def decide_h1c(
    validation_sweep: pd.DataFrame,
    selected_decision: dict[str, Any],
    selected_cost_sensitivity: pd.DataFrame,
) -> dict[str, Any]:
    failed: list[str] = []
    warnings: list[str] = []
    passing = validation_sweep[validation_sweep["validation_status"].eq("freeze_review")] if not validation_sweep.empty else pd.DataFrame()
    interpretable_passing = passing[passing["credit_policy_interpretable"].eq(True)] if not passing.empty else pd.DataFrame()
    if interpretable_passing.empty:
        failed.append("no_interpretable_credit_policy_passed_validation")
    if selected_decision.get("status") != "freeze_review":
        failed.append("selected_variant_failed_full_promotion_gates")
    distinct_policies = int(interpretable_passing["credit_policy"].nunique()) if not interpretable_passing.empty else 0
    if distinct_policies < 2:
        warnings.append("interpretable_credit_support_less_than_2_policies")
    stress_5 = selected_cost_sensitivity[selected_cost_sensitivity["cost_bps"].eq(DEFAULT_STRESS_COST_BPS)]
    if stress_5.empty or not stress_5["net_return"].gt(0.0).all():
        failed.append("selected_variant_not_positive_at_5bps")
    extra_75 = selected_cost_sensitivity[selected_cost_sensitivity["cost_bps"].eq(7.5)]
    if not extra_75.empty and not extra_75["net_return"].gt(0.0).all():
        warnings.append("selected_variant_not_positive_at_7_5bps")
    extra_10 = selected_cost_sensitivity[selected_cost_sensitivity["cost_bps"].eq(10.0)]
    if not extra_10.empty and not extra_10["net_return"].gt(0.0).all():
        warnings.append("selected_variant_not_positive_at_10bps")
    status = "credit_repaired" if not failed else "needs_more_research"
    return {
        "status": status,
        "summary": "H1c found an interpretable credit repair candidate." if status == "credit_repaired" else "H1c did not repair the credit fragility.",
        "failed_checks": failed,
        "warnings": warnings,
        "validation_pass_count": int(len(passing)),
        "interpretable_validation_pass_count": int(len(interpretable_passing)),
        "interpretable_passed_credit_policies": sorted(interpretable_passing["credit_policy"].unique().tolist()) if not interpretable_passing.empty else [],
        "selected_promotion_status": selected_decision.get("status", "not_evaluated"),
    }


def _write_report(
    path: Path,
    validation_sweep: pd.DataFrame,
    selected: pd.Series,
    selected_controls: pd.DataFrame,
    selected_concentration: pd.DataFrame,
    selected_gates: pd.DataFrame,
    selected_cost_sensitivity: pd.DataFrame,
    repair_decision: dict[str, Any],
) -> None:
    passing = validation_sweep[validation_sweep["validation_status"].eq("freeze_review")] if not validation_sweep.empty else pd.DataFrame()
    interpretable_passing = passing[passing["credit_policy_interpretable"].eq(True)] if not passing.empty else pd.DataFrame()
    selected_rollup = rollup_by_cost(selected_controls, cost_bps=DEFAULT_COST_BPS)
    lines = [
        "# Risk-off short H1c credit repair",
        "",
        "Selection is validation-only. Test is shown only for the validation-selected variant.",
        "",
        "## Read",
        "",
        f"- Variants evaluated: `{len(validation_sweep)}`.",
        f"- Variants passing validation gates: `{len(passing)}`.",
        f"- Interpretable credit variants passing validation gates: `{len(interpretable_passing)}`.",
        f"- Selected validation variant: `{selected.get('variant_id', '')}`.",
        f"- Repair decision: `{repair_decision.get('status', 'not_evaluated')}`.",
        f"- Warnings: `{', '.join(repair_decision.get('warnings', [])) if repair_decision.get('warnings') else 'none'}`.",
        "",
        "## Top Validation Variants",
        "",
        *_markdown_table(
            validation_sweep.head(30),
            [
                "variant_id",
                "validation_status",
                "credit_policy",
                "credit_policy_interpretable",
                "failed_gate_count",
                "validation_trades",
                "validation_net_return",
                "validation_stress_net_return",
                "validation_avg_trade_net_bps",
                "validation_positive_folds",
                "validation_max_top5_abs_share",
            ],
            limit=30,
        ),
        "",
        "## Interpretable Passing Variants",
        "",
        *_markdown_table(
            interpretable_passing.head(30),
            [
                "variant_id",
                "credit_policy",
                "validation_trades",
                "validation_net_return",
                "validation_avg_trade_net_bps",
                "validation_max_top5_abs_share",
            ],
            limit=30,
        ),
        "",
        "## Selected Variant Controls",
        "",
        *_markdown_table(
            selected_rollup,
            ["split", "label", "folds", "trades", "net_return", "avg_trade_net", "positive_folds", "mean_daily_sharpe", "max_fold_drawdown"],
            limit=20,
        ),
        "",
        "## Selected Variant Concentration",
        "",
        *_markdown_table(
            selected_concentration,
            ["split", "fold", "sessions_with_trades", "net_return", "top1_abs_share", "top5_abs_share", "best_session", "best_session_net", "worst_session", "worst_session_net"],
            limit=20,
        ),
        "",
        "## Selected Variant Gates",
        "",
        *_markdown_table(selected_gates, ["gate_id", "status", "observed", "threshold", "rationale"], limit=40),
        "",
        "## Selected Cost Sensitivity",
        "",
        *_markdown_table(
            selected_cost_sensitivity,
            ["split", "cost_bps", "trades", "net_return", "avg_trade_net_bps", "positive_folds", "control_edge"],
            limit=20,
        ),
        "",
        "## Repair Decision",
        "",
        "```yaml",
        yaml.safe_dump(repair_decision, sort_keys=False).strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    outputs: H1CCreditRepairOutputs,
    selected: pd.Series,
    repair_decision: dict[str, Any],
    feature_path: Path,
    risk_context_path: Path,
) -> None:
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("h1c_credit_repair", "risk_off_short_h6", "QQQ", "15min"),
            "run_type": "h1c_credit_repair",
            "created_at_utc": utc_now(),
            "status": repair_decision["status"],
        },
        "strategy": {
            "candidate_label": CANDIDATE_LABEL,
            "horizon_bars": DEFAULT_HORIZON,
            "primary_cost_bps": DEFAULT_COST_BPS,
            "stress_cost_bps": DEFAULT_STRESS_COST_BPS,
            "hypothesis_variant": "H1c credit repair",
        },
        "selected_validation_variant": selected.to_dict() if not selected.empty else {},
        "repair_decision": repair_decision,
        "data": {
            "features_path": feature_path.as_posix(),
            "features_fingerprint": fingerprint_path(feature_path) if feature_path.exists() else "MISSING",
            "risk_context_path": risk_context_path.as_posix(),
            "risk_context_fingerprint": fingerprint_path(risk_context_path) if risk_context_path.exists() else "MISSING",
            "split_policy": DEFAULT_SPLIT_POLICY,
        },
        "outputs": {key: value.as_posix() for key, value in outputs.__dict__.items() if key.endswith("_path")},
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def run_h1c_credit_repair(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    output_dir: str | Path = DEFAULT_H1C_OUTPUT_DIR,
    risk_quantiles: tuple[float, ...] = DEFAULT_H1C_RISK_QUANTILES,
    vix_quantiles: tuple[float, ...] = DEFAULT_H1C_VIX_QUANTILES,
    credit_policies: tuple[str, ...] = DEFAULT_H1C_CREDIT_POLICIES,
) -> H1CCreditRepairOutputs:
    feature_path = Path(features_path)
    context_path = Path(risk_context_path)
    frame = load_eda_frame(feature_path, context_path, (DEFAULT_HORIZON,))
    folds = build_monthly_folds(frame, DEFAULT_SPLIT_POLICY)
    if not folds:
        raise ValueError("split policy produced no folds")

    rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []
    for risk_q in risk_quantiles:
        for vix_q in vix_quantiles:
            for policy in credit_policies:
                _, summary, concentration = run_h1c_variant_backtest(
                    frame,
                    folds,
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    credit_policy=policy,
                    splits=("validation",),
                )
                gates, decision = evaluate_promotion_gates(
                    summary,
                    concentration,
                    candidate_label=CANDIDATE_LABEL,
                    splits=("validation",),
                )
                row = validation_sweep_row(
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    credit_policy=policy,
                    summary=summary,
                    concentration=concentration,
                    decision=decision,
                )
                rows.append(row)
                if not gates.empty:
                    gates = gates.copy()
                    gates.insert(0, "variant_id", row["variant_id"])
                    gates.insert(1, "risk_off_quantile", float(risk_q))
                    gates.insert(2, "vix_quantile", float(vix_q))
                    gates.insert(3, "credit_policy", policy)
                    gate_rows.append(gates)

    validation_sweep = pd.DataFrame(rows)
    if not validation_sweep.empty:
        validation_sweep["_status_order"] = validation_sweep["validation_status"].map({"freeze_review": 0, "continue_research": 1}).fillna(9)
        validation_sweep = validation_sweep.sort_values(
            [
                "_status_order",
                "credit_policy_interpretable",
                "credit_policy_rank",
                "failed_gate_count",
                "validation_net_return",
                "validation_control_edge",
                "validation_positive_folds",
                "validation_trades",
            ],
            ascending=[True, False, True, True, False, False, False, False],
            kind="stable",
        ).drop(columns="_status_order")
    validation_gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    selected = select_h1c_validation_variant(validation_sweep)

    selected_trades = pd.DataFrame()
    selected_controls = pd.DataFrame()
    selected_concentration = pd.DataFrame()
    selected_gates = pd.DataFrame()
    selected_cost_sensitivity = pd.DataFrame()
    selected_decision: dict[str, Any] = {
        "status": "not_evaluated",
        "summary": "No validation variant was selected.",
        "failed_gates": [],
        "gate_config": DEFAULT_PROMOTION_GATES,
    }
    if not selected.empty:
        selected_trades, selected_controls, selected_concentration = run_h1c_variant_backtest(
            frame,
            folds,
            risk_off_quantile=float(selected["risk_off_quantile"]),
            vix_quantile=float(selected["vix_quantile"]),
            credit_policy=str(selected["credit_policy"]),
            splits=("validation", "test"),
            cost_bps_values=(DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS, *EXTRA_COST_BPS),
        )
        selected_gates, selected_decision = evaluate_promotion_gates(
            selected_controls,
            selected_concentration,
            candidate_label=CANDIDATE_LABEL,
        )
        selected_cost_sensitivity = build_cost_sensitivity(
            selected_controls,
            (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS, *EXTRA_COST_BPS),
        )
    repair_decision = decide_h1c(validation_sweep, selected_decision, selected_cost_sensitivity)

    root = Path(output_dir)
    outputs = H1CCreditRepairOutputs(
        output_dir=root,
        report_path=root / "report.md",
        manifest_path=root / "manifest.yaml",
        validation_sweep_path=root / "validation_sweep.parquet",
        validation_gates_path=root / "validation_gates.parquet",
        selected_variant_path=root / "selected_variant.yaml",
        selected_trades_path=root / "selected_trades.parquet",
        selected_controls_path=root / "selected_controls.parquet",
        selected_concentration_path=root / "selected_concentration.parquet",
        selected_gates_path=root / "selected_gates.parquet",
        selected_cost_sensitivity_path=root / "selected_cost_sensitivity.parquet",
        selected_decision_path=root / "selected_decision.yaml",
    )
    root.mkdir(parents=True, exist_ok=True)
    validation_sweep.to_parquet(outputs.validation_sweep_path, index=False)
    validation_gates.to_parquet(outputs.validation_gates_path, index=False)
    selected_trades.to_parquet(outputs.selected_trades_path, index=False)
    selected_controls.to_parquet(outputs.selected_controls_path, index=False)
    selected_concentration.to_parquet(outputs.selected_concentration_path, index=False)
    selected_gates.to_parquet(outputs.selected_gates_path, index=False)
    selected_cost_sensitivity.to_parquet(outputs.selected_cost_sensitivity_path, index=False)
    outputs.selected_variant_path.write_text(yaml.safe_dump(selected.to_dict() if not selected.empty else {}, sort_keys=False), encoding="utf-8")
    outputs.selected_decision_path.write_text(
        yaml.safe_dump({"promotion": selected_decision, "repair": repair_decision}, sort_keys=False),
        encoding="utf-8",
    )
    _write_report(
        outputs.report_path,
        validation_sweep,
        selected,
        selected_controls,
        selected_concentration,
        selected_gates,
        selected_cost_sensitivity,
        repair_decision,
    )
    _write_manifest(outputs.manifest_path, outputs, selected, repair_decision, feature_path, context_path)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H1c risk-off short credit repair")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_H1C_OUTPUT_DIR))
    args = parser.parse_args(argv)
    outputs = run_h1c_credit_repair(features_path=args.features, risk_context_path=args.risk_context, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

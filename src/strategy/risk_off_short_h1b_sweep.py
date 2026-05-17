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


DEFAULT_H1B_SWEEP_DIR = DEFAULT_OUTPUT_DIR / "h1b_concentration_sweep"
DEFAULT_H1B_RISK_QUANTILES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_H1B_VIX_QUANTILES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
DEFAULT_H1B_FILTER_POLICIES = (
    "none",
    "vix_cap_q90",
    "vix_cap_q95",
    "risk_on_low_q40",
    "risk_on_low_q50",
    "credit_weak_q40",
    "credit_weak_q50",
    "defensive_high_q60",
    "defensive_high_q70",
    "breadth_weak_12_lte1",
    "breadth_weak_12_lte2",
    "stress_high_q50",
    "vix_term_high_q50",
    "below_vwap",
    "risk_on_low_q50__vix_cap_q90",
    "credit_weak_q50__vix_cap_q90",
    "defensive_high_q60__vix_cap_q90",
    "breadth_weak_12_lte2__vix_cap_q90",
    "vix_term_high_q50__vix_cap_q90",
    "risk_on_low_q50__credit_weak_q50",
)


@dataclass(frozen=True)
class H1BFilter:
    policy: str
    thresholds: dict[str, float | int | str]


@dataclass(frozen=True)
class RiskOffH1BSweepOutputs:
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
    selected_decision_path: Path


def h1b_variant_id(risk_off_quantile: float, vix_quantile: float, filter_policy: str) -> str:
    risk_token = f"riskq{int(round(risk_off_quantile * 100)):02d}"
    vix_token = f"vixq{int(round(vix_quantile * 100)):02d}"
    return f"{risk_token}__{vix_token}__{filter_policy}"


def _policy_components(policy: str) -> tuple[str, ...]:
    return tuple(component for component in policy.split("__") if component and component != "none")


def _quantile_from_component(component: str) -> float:
    if "_q" not in component:
        raise ValueError(f"filter component has no quantile suffix: {component}")
    return float(int(component.rsplit("_q", 1)[1]) / 100.0)


def fit_h1b_filter(train: pd.DataFrame, policy: str) -> H1BFilter:
    thresholds: dict[str, float | int | str] = {}
    for component in _policy_components(policy):
        if component.startswith("vix_cap_q"):
            thresholds[component] = _finite_quantile(train, "prev_vix_z20", _quantile_from_component(component))
        elif component.startswith("risk_on_low_q"):
            thresholds[component] = _finite_quantile(train, "risk_on_score", _quantile_from_component(component))
        elif component.startswith("credit_weak_q"):
            thresholds[component] = _finite_quantile(train, "spread_credit_12", _quantile_from_component(component))
        elif component.startswith("defensive_high_q"):
            thresholds[component] = _finite_quantile(train, "defensive_rotation_score", _quantile_from_component(component))
        elif component.startswith("stress_high_q"):
            thresholds[component] = _finite_quantile(train, "intraday_stress_score", _quantile_from_component(component))
        elif component.startswith("vix_term_high_q"):
            thresholds[component] = _finite_quantile(train, "prev_vix9d_vix_ratio", _quantile_from_component(component))
        elif component.startswith("breadth_weak_12_lte"):
            thresholds[component] = int(component.rsplit("lte", 1)[1])
        elif component == "below_vwap":
            thresholds[component] = "target_dist_vwap_atr <= 0"
        else:
            raise ValueError(f"unsupported H1b filter component: {component}")
    return H1BFilter(policy=policy, thresholds=thresholds)


def apply_h1b_filter(frame: pd.DataFrame, fitted_filter: H1BFilter) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for component, threshold in fitted_filter.thresholds.items():
        if component.startswith("vix_cap_q"):
            condition = frame["prev_vix_z20"].le(float(threshold))
        elif component.startswith("risk_on_low_q"):
            condition = frame["risk_on_score"].le(float(threshold))
        elif component.startswith("credit_weak_q"):
            condition = frame["spread_credit_12"].le(float(threshold))
        elif component.startswith("defensive_high_q"):
            condition = frame["defensive_rotation_score"].ge(float(threshold))
        elif component.startswith("stress_high_q"):
            condition = frame["intraday_stress_score"].ge(float(threshold))
        elif component.startswith("vix_term_high_q"):
            condition = frame["prev_vix9d_vix_ratio"].ge(float(threshold))
        elif component.startswith("breadth_weak_12_lte"):
            condition = frame["positive_index_count_12"].le(int(threshold))
        elif component == "below_vwap":
            condition = frame["target_dist_vwap_atr"].le(0.0)
        else:
            raise ValueError(f"unsupported H1b filter component: {component}")
        mask &= condition.fillna(False)
    return mask.fillna(False)


def h1b_control_masks(
    frame: pd.DataFrame,
    thresholds: RiskOffThresholds,
    fitted_filter: H1BFilter,
    *,
    horizon: int,
    random_seed: int,
) -> dict[str, pd.Series]:
    masks = control_masks(frame, thresholds, horizon=horizon, random_seed=random_seed)
    candidate = masks[CANDIDATE_LABEL] & apply_h1b_filter(frame, fitted_filter)
    masks[CANDIDATE_LABEL] = candidate

    valid = _valid_exit_mask(frame, horizon)
    filter_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(fitted_filter.policy))
    rng = np.random.default_rng(random_seed + int(horizon) + len(frame) + filter_seed)
    random_mask = pd.Series(False, index=frame.index)
    candidate_count = int((candidate & valid).sum())
    valid_indices = np.flatnonzero(valid.to_numpy())
    if candidate_count > 0 and len(valid_indices) >= candidate_count:
        random_mask.iloc[rng.choice(valid_indices, size=candidate_count, replace=False)] = True
    masks["random_same_count_control"] = random_mask
    return masks


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["session"].astype(str).isin(sessions)].copy()


def run_h1b_variant_backtest(
    frame: pd.DataFrame,
    folds: tuple[ResearchFold, ...],
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    filter_policy: str,
    splits: tuple[str, ...],
    cost_bps_values: tuple[float, ...] = (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS),
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_sessions: dict[tuple[int, str], tuple[str, ...]] = {}
    all_trades: list[pd.DataFrame] = []
    for fold in folds:
        train = _split_frame(frame, fold.train_sessions)
        thresholds = fit_thresholds(train, risk_off_quantile=risk_off_quantile, vix_quantile=vix_quantile)
        fitted_filter = fit_h1b_filter(train, filter_policy)
        session_map = {
            "validation": fold.validation_sessions,
            "test": fold.test_sessions,
        }
        for split in splits:
            sessions = session_map[split]
            split_sessions[(fold.fold, split)] = tuple(sessions)
            split_frame = _split_frame(frame, sessions)
            masks = h1b_control_masks(
                split_frame,
                thresholds,
                fitted_filter,
                horizon=horizon,
                random_seed=40_000 + fold.fold,
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

    vid = h1b_variant_id(risk_off_quantile, vix_quantile, filter_policy)
    for artifact in (trades, summary, concentration):
        if not artifact.empty:
            artifact.insert(0, "variant_id", vid)
            artifact.insert(1, "risk_off_quantile", float(risk_off_quantile))
            artifact.insert(2, "vix_quantile", float(vix_quantile))
            artifact.insert(3, "filter_policy", filter_policy)
    return trades, summary, concentration


def h1b_validation_sweep_row(
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    filter_policy: str,
    summary: pd.DataFrame,
    concentration: pd.DataFrame,
    gates: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    primary = rollup_by_cost(summary, cost_bps=DEFAULT_COST_BPS)
    stress = rollup_by_cost(summary, cost_bps=DEFAULT_STRESS_COST_BPS)
    candidate = primary[primary["label"].eq(CANDIDATE_LABEL) & primary["split"].eq("validation")]
    stress_candidate = stress[stress["label"].eq(CANDIDATE_LABEL) & stress["split"].eq("validation")]
    best_control = primary[~primary["label"].eq(CANDIDATE_LABEL) & primary["split"].eq("validation")]
    row = candidate.iloc[0] if not candidate.empty else pd.Series(dtype=object)
    stress_row = stress_candidate.iloc[0] if not stress_candidate.empty else pd.Series(dtype=object)
    net = float(row.get("net_return", 0.0)) if not row.empty else 0.0
    avg_trade_bps = float(row.get("avg_trade_net", 0.0)) * 10_000.0 if not row.empty else 0.0
    stress_net = float(stress_row.get("net_return", 0.0)) if not stress_row.empty else 0.0
    best_control_net = float(best_control["net_return"].max()) if not best_control.empty else np.nan
    min_sessions = int(concentration["sessions_with_trades"].min()) if not concentration.empty else 0
    max_top5 = float(concentration["top5_abs_share"].max()) if not concentration.empty else 1.0
    gate_cfg = DEFAULT_PROMOTION_GATES
    failed_gates = decision.get("failed_gates", [])
    return {
        "variant_id": h1b_variant_id(risk_off_quantile, vix_quantile, filter_policy),
        "risk_off_quantile": float(risk_off_quantile),
        "vix_quantile": float(vix_quantile),
        "filter_policy": filter_policy,
        "validation_status": decision["status"],
        "failed_gate_count": int(len(failed_gates)),
        "failed_gates": ",".join(failed_gates),
        "validation_trades": int(row.get("trades", 0)) if not row.empty else 0,
        "validation_net_return": net,
        "validation_stress_net_return": stress_net,
        "validation_avg_trade_net_bps": avg_trade_bps,
        "validation_positive_folds": int(row.get("positive_folds", 0)) if not row.empty else 0,
        "validation_best_control_net": best_control_net,
        "validation_control_edge": net - best_control_net if np.isfinite(best_control_net) else np.nan,
        "validation_min_sessions_per_fold": min_sessions,
        "validation_max_top5_abs_share": max_top5,
        "validation_concentration_repaired": bool(min_sessions >= int(gate_cfg["min_sessions_per_fold"]) and max_top5 <= float(gate_cfg["max_top5_abs_share"])),
        "validation_avg_trade_gate_pass": bool(avg_trade_bps >= float(gate_cfg["min_avg_trade_net_bps"])),
        "validation_stress_gate_pass": bool(stress_net > float(gate_cfg["min_validation_stress_net_return"])),
        "passed_gate_count": int(gates["status"].eq("pass").sum()) if not gates.empty else 0,
    }


def select_h1b_validation_variant(sweep: pd.DataFrame) -> pd.Series:
    if sweep.empty:
        return pd.Series(dtype=object)
    ranked = sweep.copy()
    ranked["passed_validation_gates"] = ranked["validation_status"].eq("freeze_review")
    ranked = ranked.sort_values(
        [
            "passed_validation_gates",
            "validation_concentration_repaired",
            "validation_avg_trade_gate_pass",
            "validation_stress_gate_pass",
            "failed_gate_count",
            "validation_max_top5_abs_share",
            "validation_net_return",
            "validation_control_edge",
            "validation_positive_folds",
            "validation_trades",
        ],
        ascending=[False, False, False, False, True, True, False, False, False, False],
        kind="stable",
    )
    return ranked.iloc[0]


def _write_manifest(path: Path, outputs: RiskOffH1BSweepOutputs, selected: pd.Series, feature_path: Path, risk_context_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("h1b_concentration_sweep", "risk_off_short_h6", "QQQ", "15min"),
            "run_type": "h1b_concentration_sweep",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "strategy": {
            "candidate_label": CANDIDATE_LABEL,
            "horizon_bars": DEFAULT_HORIZON,
            "primary_cost_bps": DEFAULT_COST_BPS,
            "stress_cost_bps": DEFAULT_STRESS_COST_BPS,
            "hypothesis_variant": "H1b concentration repair",
        },
        "selected_validation_variant": selected.to_dict() if not selected.empty else {},
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


def _write_report(
    path: Path,
    validation_sweep: pd.DataFrame,
    selected: pd.Series,
    selected_controls: pd.DataFrame,
    selected_concentration: pd.DataFrame,
    selected_gates: pd.DataFrame,
    selected_decision: dict[str, Any],
) -> None:
    passing = validation_sweep[validation_sweep["validation_status"].eq("freeze_review")] if not validation_sweep.empty else pd.DataFrame()
    concentration_repaired = (
        validation_sweep[validation_sweep["validation_concentration_repaired"].eq(True)] if not validation_sweep.empty else pd.DataFrame()
    )
    selected_rollup = rollup_by_cost(selected_controls, cost_bps=DEFAULT_COST_BPS)
    failed_gates = selected_decision.get("failed_gates", [])
    top_columns = [
        "variant_id",
        "validation_status",
        "failed_gate_count",
        "filter_policy",
        "validation_trades",
        "validation_net_return",
        "validation_stress_net_return",
        "validation_avg_trade_net_bps",
        "validation_positive_folds",
        "validation_min_sessions_per_fold",
        "validation_max_top5_abs_share",
        "validation_concentration_repaired",
    ]
    lines = [
        "# Risk-off short H1b concentration repair sweep",
        "",
        "Selection is validation-only. Test is shown only for the validation-selected variant.",
        "",
        "## Read",
        "",
        f"- Variants evaluated: `{len(validation_sweep)}`.",
        f"- Variants passing validation gates: `{len(passing)}`.",
        f"- Variants repairing validation concentration: `{len(concentration_repaired)}`.",
        f"- Selected validation variant: `{selected.get('variant_id', '')}`.",
        f"- Final selected-variant decision after test confirmation: `{selected_decision.get('status', 'not_evaluated')}`.",
        "",
        "## Top Validation Variants",
        "",
        *_markdown_table(validation_sweep.head(25), top_columns, limit=25),
        "",
        "## Best Concentration-Repaired Variants",
        "",
        *_markdown_table(
            concentration_repaired.sort_values(
                ["validation_avg_trade_gate_pass", "failed_gate_count", "validation_net_return"],
                ascending=[False, True, False],
                kind="stable",
            ).head(25),
            top_columns,
            limit=25,
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
        "Failed gates:",
        "",
        ", ".join(f"`{gate}`" for gate in failed_gates) if failed_gates else "None.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_h1b_sweep(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    output_dir: str | Path = DEFAULT_H1B_SWEEP_DIR,
    risk_quantiles: tuple[float, ...] = DEFAULT_H1B_RISK_QUANTILES,
    vix_quantiles: tuple[float, ...] = DEFAULT_H1B_VIX_QUANTILES,
    filter_policies: tuple[str, ...] = DEFAULT_H1B_FILTER_POLICIES,
) -> RiskOffH1BSweepOutputs:
    feature_path = Path(features_path)
    context_path = Path(risk_context_path)
    frame = load_eda_frame(feature_path, context_path, (DEFAULT_HORIZON,))
    folds = build_monthly_folds(frame, DEFAULT_SPLIT_POLICY)
    if not folds:
        raise ValueError("split policy produced no folds")

    validation_rows: list[dict[str, Any]] = []
    validation_gate_rows: list[pd.DataFrame] = []
    for risk_q in risk_quantiles:
        for vix_q in vix_quantiles:
            for filter_policy in filter_policies:
                _, summary, concentration = run_h1b_variant_backtest(
                    frame,
                    folds,
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    filter_policy=filter_policy,
                    splits=("validation",),
                )
                gates, decision = evaluate_promotion_gates(
                    summary,
                    concentration,
                    candidate_label=CANDIDATE_LABEL,
                    splits=("validation",),
                )
                row = h1b_validation_sweep_row(
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    filter_policy=filter_policy,
                    summary=summary,
                    concentration=concentration,
                    gates=gates,
                    decision=decision,
                )
                validation_rows.append(row)
                if not gates.empty:
                    gates = gates.copy()
                    gates.insert(0, "variant_id", row["variant_id"])
                    gates.insert(1, "risk_off_quantile", float(risk_q))
                    gates.insert(2, "vix_quantile", float(vix_q))
                    gates.insert(3, "filter_policy", filter_policy)
                    validation_gate_rows.append(gates)

    validation_sweep = pd.DataFrame(validation_rows)
    if not validation_sweep.empty:
        validation_sweep = validation_sweep.sort_values(
            [
                "validation_status",
                "validation_concentration_repaired",
                "validation_avg_trade_gate_pass",
                "validation_stress_gate_pass",
                "failed_gate_count",
                "validation_max_top5_abs_share",
                "validation_net_return",
                "validation_control_edge",
                "validation_positive_folds",
                "validation_trades",
            ],
            ascending=[True, False, False, False, True, True, False, False, False, False],
            kind="stable",
        )
        validation_sweep["_status_order"] = validation_sweep["validation_status"].map({"freeze_review": 0, "continue_research": 1}).fillna(9)
        validation_sweep = validation_sweep.sort_values(
            [
                "_status_order",
                "validation_concentration_repaired",
                "validation_avg_trade_gate_pass",
                "validation_stress_gate_pass",
                "failed_gate_count",
                "validation_max_top5_abs_share",
                "validation_net_return",
                "validation_control_edge",
                "validation_positive_folds",
                "validation_trades",
            ],
            ascending=[True, False, False, False, True, True, False, False, False, False],
            kind="stable",
        ).drop(columns="_status_order")
    validation_gates = pd.concat(validation_gate_rows, ignore_index=True) if validation_gate_rows else pd.DataFrame()
    selected = select_h1b_validation_variant(validation_sweep)

    selected_trades = pd.DataFrame()
    selected_controls = pd.DataFrame()
    selected_concentration = pd.DataFrame()
    selected_gates = pd.DataFrame()
    selected_decision: dict[str, Any] = {
        "status": "not_evaluated",
        "summary": "No validation variant was selected.",
        "failed_gates": [],
        "gate_config": DEFAULT_PROMOTION_GATES,
    }
    if not selected.empty:
        selected_trades, selected_controls, selected_concentration = run_h1b_variant_backtest(
            frame,
            folds,
            risk_off_quantile=float(selected["risk_off_quantile"]),
            vix_quantile=float(selected["vix_quantile"]),
            filter_policy=str(selected["filter_policy"]),
            splits=("validation", "test"),
        )
        selected_gates, selected_decision = evaluate_promotion_gates(
            selected_controls,
            selected_concentration,
            candidate_label=CANDIDATE_LABEL,
        )

    root = Path(output_dir)
    outputs = RiskOffH1BSweepOutputs(
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
        selected_decision_path=root / "selected_decision.yaml",
    )
    root.mkdir(parents=True, exist_ok=True)
    validation_sweep.to_parquet(outputs.validation_sweep_path, index=False)
    validation_gates.to_parquet(outputs.validation_gates_path, index=False)
    selected_trades.to_parquet(outputs.selected_trades_path, index=False)
    selected_controls.to_parquet(outputs.selected_controls_path, index=False)
    selected_concentration.to_parquet(outputs.selected_concentration_path, index=False)
    selected_gates.to_parquet(outputs.selected_gates_path, index=False)
    outputs.selected_variant_path.write_text(yaml.safe_dump(selected.to_dict() if not selected.empty else {}, sort_keys=False), encoding="utf-8")
    outputs.selected_decision_path.write_text(yaml.safe_dump(selected_decision, sort_keys=False), encoding="utf-8")
    _write_report(outputs.report_path, validation_sweep, selected, selected_controls, selected_concentration, selected_gates, selected_decision)
    _write_manifest(outputs.manifest_path, outputs, selected, feature_path, context_path)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run risk-off short H1b concentration repair sweep")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_H1B_SWEEP_DIR))
    args = parser.parse_args(argv)
    outputs = run_h1b_sweep(features_path=args.features, risk_context_path=args.risk_context, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

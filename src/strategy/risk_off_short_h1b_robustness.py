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
from src.strategy import StrategySpec
from src.strategy.freeze_risk_off_short_h1b import DEFAULT_FREEZE_DIR, DEFAULT_STRATEGY_SPEC_PATH
from src.strategy.risk_off_short import CANDIDATE_LABEL, DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_POLICY, _markdown_table
from src.strategy.risk_off_short_h1b_sweep import (
    h1b_variant_id,
    run_h1b_variant_backtest,
)
from src.strategy.risk_off_short_triage import DEFAULT_COST_BPS, DEFAULT_HORIZON, DEFAULT_STRESS_COST_BPS, enrich_trade_times


DEFAULT_ROBUSTNESS_DIR = DEFAULT_OUTPUT_DIR / "robustness" / "qqq_15min_risk_off_short_h1b_v1"
DEFAULT_EXTRA_COST_BPS = (7.5, 10.0)
LOCAL_GRID_STEP = 0.05


@dataclass(frozen=True)
class H1BRobustnessOutputs:
    output_dir: Path
    report_path: Path
    manifest_path: Path
    local_threshold_sweep_path: Path
    local_threshold_gates_path: Path
    cost_sensitivity_path: Path
    subperiod_summary_path: Path
    fold_stability_path: Path
    robustness_decision_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def _fingerprint_or_missing(path: Path) -> str:
    return fingerprint_path(path) if path.exists() else "MISSING"


def _alpha_rule(raw_spec: dict[str, Any], column: str) -> dict[str, Any]:
    rules = raw_spec.get("alpha", {}).get("rules", [])
    for rule in rules:
        if isinstance(rule, dict) and rule.get("column") == column:
            return rule
    raise ValueError(f"missing alpha rule for column: {column}")


def selected_quantiles_from_spec(raw_spec: dict[str, Any]) -> dict[str, float]:
    return {
        "risk_off_quantile": float(_alpha_rule(raw_spec, "risk_off_score")["quantile"]),
        "vix_quantile": float(_alpha_rule(raw_spec, "prev_vix_z20")["quantile"]),
        "credit_quantile": float(_alpha_rule(raw_spec, "spread_credit_12")["quantile"]),
    }


def local_quantile_grid(center: float, *, step: float = LOCAL_GRID_STEP) -> tuple[float, ...]:
    values = {round(float(np.clip(center + offset, 0.01, 0.99)), 4) for offset in (-step, 0.0, step)}
    return tuple(sorted(values))


def credit_filter_policy(credit_quantile: float) -> str:
    return f"credit_weak_q{int(round(float(credit_quantile) * 100)):02d}"


def _candidate_metric(frame: pd.DataFrame, split: str, label: str = CANDIDATE_LABEL) -> pd.Series:
    match = frame[frame["split"].eq(split) & frame["label"].eq(label)]
    return match.iloc[0] if not match.empty else pd.Series(dtype=object)


def _split_concentration(concentration: pd.DataFrame, split: str) -> tuple[int, float]:
    if concentration.empty:
        return 0, 1.0
    subset = concentration[concentration["split"].eq(split)]
    if subset.empty:
        return 0, 1.0
    return int(subset["sessions_with_trades"].min()), float(subset["top5_abs_share"].max())


def local_sweep_row(
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    credit_quantile: float,
    selected_quantiles: dict[str, float],
    summary: pd.DataFrame,
    concentration: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    primary = rollup_by_cost(summary, cost_bps=DEFAULT_COST_BPS)
    stress = rollup_by_cost(summary, cost_bps=DEFAULT_STRESS_COST_BPS)
    filter_policy = credit_filter_policy(credit_quantile)
    row: dict[str, Any] = {
        "variant_id": h1b_variant_id(risk_off_quantile, vix_quantile, filter_policy),
        "risk_off_quantile": float(risk_off_quantile),
        "vix_quantile": float(vix_quantile),
        "credit_quantile": float(credit_quantile),
        "filter_policy": filter_policy,
        "is_anchor": bool(
            np.isclose(risk_off_quantile, selected_quantiles["risk_off_quantile"])
            and np.isclose(vix_quantile, selected_quantiles["vix_quantile"])
            and np.isclose(credit_quantile, selected_quantiles["credit_quantile"])
        ),
        "status": decision["status"],
        "failed_gate_count": int(len(decision.get("failed_gates", []))),
        "failed_gates": ",".join(decision.get("failed_gates", [])),
    }
    for split in ("validation", "test"):
        candidate = _candidate_metric(primary, split)
        stress_candidate = _candidate_metric(stress, split)
        best_control = primary[primary["split"].eq(split) & ~primary["label"].eq(CANDIDATE_LABEL)]
        net = float(candidate.get("net_return", 0.0)) if not candidate.empty else 0.0
        best_control_net = float(best_control["net_return"].max()) if not best_control.empty else np.nan
        min_sessions, max_top5 = _split_concentration(concentration, split)
        row.update(
            {
                f"{split}_trades": int(candidate.get("trades", 0)) if not candidate.empty else 0,
                f"{split}_net_return": net,
                f"{split}_stress_net_return": float(stress_candidate.get("net_return", 0.0)) if not stress_candidate.empty else 0.0,
                f"{split}_avg_trade_net_bps": float(candidate.get("avg_trade_net", 0.0)) * 10_000.0 if not candidate.empty else 0.0,
                f"{split}_positive_folds": int(candidate.get("positive_folds", 0)) if not candidate.empty else 0,
                f"{split}_best_control_net": best_control_net,
                f"{split}_control_edge": net - best_control_net if np.isfinite(best_control_net) else np.nan,
                f"{split}_min_sessions_per_fold": min_sessions,
                f"{split}_max_top5_abs_share": max_top5,
            }
        )
    return row


def build_local_threshold_sweep(
    frame: pd.DataFrame,
    selected_quantiles: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_monthly_folds(frame, DEFAULT_SPLIT_POLICY)
    rows: list[dict[str, Any]] = []
    gate_rows: list[pd.DataFrame] = []
    for risk_q in local_quantile_grid(selected_quantiles["risk_off_quantile"]):
        for vix_q in local_quantile_grid(selected_quantiles["vix_quantile"]):
            for credit_q in local_quantile_grid(selected_quantiles["credit_quantile"]):
                filter_policy = credit_filter_policy(credit_q)
                _, summary, concentration = run_h1b_variant_backtest(
                    frame,
                    folds,
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    filter_policy=filter_policy,
                    splits=("validation", "test"),
                )
                gates, decision = evaluate_promotion_gates(summary, concentration, candidate_label=CANDIDATE_LABEL)
                row = local_sweep_row(
                    risk_off_quantile=float(risk_q),
                    vix_quantile=float(vix_q),
                    credit_quantile=float(credit_q),
                    selected_quantiles=selected_quantiles,
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
                    gates.insert(3, "credit_quantile", float(credit_q))
                    gates.insert(4, "filter_policy", filter_policy)
                    gates.insert(5, "is_anchor", row["is_anchor"])
                    gate_rows.append(gates)
    sweep = pd.DataFrame(rows)
    if not sweep.empty:
        sweep["_status_order"] = sweep["status"].map({"freeze_review": 0, "continue_research": 1}).fillna(9)
        sweep = sweep.sort_values(
            [
                "_status_order",
                "is_anchor",
                "failed_gate_count",
                "validation_net_return",
                "test_net_return",
                "validation_max_top5_abs_share",
            ],
            ascending=[True, False, True, False, False, True],
            kind="stable",
        ).drop(columns="_status_order")
    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    return sweep, gates


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


def build_subperiod_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["split", "year", "trades", "net_return", "avg_trade_net_bps", "positive_trades", "win_rate"])
    candidate = trades[trades["label"].eq(CANDIDATE_LABEL) & trades["cost_bps"].eq(DEFAULT_COST_BPS)].copy()
    if candidate.empty:
        return pd.DataFrame(columns=["split", "year", "trades", "net_return", "avg_trade_net_bps", "positive_trades", "win_rate"])
    candidate["year"] = pd.to_datetime(candidate["session"]).dt.year
    rows: list[dict[str, Any]] = []
    for (split, year), group in candidate.groupby(["split", "year"], sort=True):
        net = group["net_return"].astype(float)
        rows.append(
            {
                "split": str(split),
                "year": int(year),
                "trades": int(len(group)),
                "net_return": float(net.sum()),
                "avg_trade_net_bps": float(net.mean() * 10_000.0),
                "positive_trades": int(net.gt(0.0).sum()),
                "win_rate": float(net.gt(0.0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "year"], kind="stable").drop(columns="_split_order")


def build_fold_stability(summary: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    candidate = summary[
        summary["label"].eq(CANDIDATE_LABEL)
        & summary["cost_bps"].eq(DEFAULT_COST_BPS)
        & summary["split"].isin(["validation", "test"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame()
    candidate["avg_trade_net_bps"] = candidate["avg_trade_net"].astype(float) * 10_000.0
    columns = [
        "split",
        "fold",
        "trades",
        "net_return",
        "avg_trade_net_bps",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "win_rate",
    ]
    result = candidate.loc[:, columns].merge(
        concentration.loc[:, ["split", "fold", "sessions_with_trades", "top1_abs_share", "top5_abs_share"]],
        on=["split", "fold"],
        how="left",
    )
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "fold"], kind="stable").drop(columns="_split_order")


def decide_robustness(local_sweep: pd.DataFrame, cost_sensitivity: pd.DataFrame, fold_stability: pd.DataFrame) -> dict[str, Any]:
    failed_checks: list[str] = []
    warnings: list[str] = []

    anchor = local_sweep[local_sweep["is_anchor"].eq(True)] if not local_sweep.empty else pd.DataFrame()
    anchor_status = str(anchor["status"].iloc[0]) if not anchor.empty else "missing"
    local_passes = local_sweep[local_sweep["status"].eq("freeze_review")] if not local_sweep.empty else pd.DataFrame()
    pass_count = int(len(local_passes))
    pass_rate = float(pass_count / len(local_sweep)) if len(local_sweep) else 0.0
    distinct_passed_risk = int(local_passes["risk_off_quantile"].nunique()) if not local_passes.empty else 0
    distinct_passed_vix = int(local_passes["vix_quantile"].nunique()) if not local_passes.empty else 0
    distinct_passed_credit = int(local_passes["credit_quantile"].nunique()) if not local_passes.empty else 0

    if anchor_status != "freeze_review":
        failed_checks.append("anchor_variant_failed_promotion_gates")
    if pass_count < 5:
        failed_checks.append("local_threshold_support_lt_5_variants")
    if min(distinct_passed_risk, distinct_passed_vix, distinct_passed_credit) < 2:
        failed_checks.append("local_threshold_support_not_spread_across_dimensions")

    stress_5 = cost_sensitivity[cost_sensitivity["cost_bps"].eq(DEFAULT_STRESS_COST_BPS)]
    if stress_5.empty or not stress_5["net_return"].gt(0.0).all():
        failed_checks.append("stress_5bps_not_positive_all_splits")
    extra_75 = cost_sensitivity[cost_sensitivity["cost_bps"].eq(7.5)]
    if not extra_75.empty and not extra_75["net_return"].gt(0.0).all():
        warnings.append("extra_stress_7_5bps_not_positive_all_splits")
    extra_10 = cost_sensitivity[cost_sensitivity["cost_bps"].eq(10.0)]
    if not extra_10.empty and not extra_10["net_return"].gt(0.0).all():
        warnings.append("extra_stress_10bps_not_positive_all_splits")

    if fold_stability.empty:
        failed_checks.append("missing_fold_stability")
    else:
        if int(fold_stability["trades"].min()) <= 0:
            failed_checks.append("empty_candidate_fold")
        if float(fold_stability["top5_abs_share"].max()) > float(DEFAULT_PROMOTION_GATES["max_top5_abs_share"]):
            failed_checks.append("fold_concentration_above_gate")

    status = "paper_candidate" if not failed_checks else ("needs_more_research" if anchor_status == "freeze_review" else "reject_or_park")
    return {
        "status": status,
        "summary": "Pre-paper robustness checks passed." if status == "paper_candidate" else "Pre-paper robustness found blocking issues.",
        "failed_checks": failed_checks,
        "warnings": warnings,
        "local_threshold_pass_count": pass_count,
        "local_threshold_variant_count": int(len(local_sweep)),
        "local_threshold_pass_rate": pass_rate,
        "distinct_passed_risk_quantiles": distinct_passed_risk,
        "distinct_passed_vix_quantiles": distinct_passed_vix,
        "distinct_passed_credit_quantiles": distinct_passed_credit,
    }


def _write_report(
    path: Path,
    strategy: StrategySpec,
    selected_quantiles: dict[str, float],
    local_sweep: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    subperiod_summary: pd.DataFrame,
    fold_stability: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    anchor = local_sweep[local_sweep["is_anchor"].eq(True)] if not local_sweep.empty else pd.DataFrame()
    lines = [
        "# Risk-off short H1b pre-paper robustness",
        "",
        "This report starts from the frozen StrategySpec. It is not a search for a new best variant.",
        "",
        "## Read",
        "",
        f"- Strategy: `{strategy.strategy_id}`.",
        f"- Anchor quantiles: risk `{selected_quantiles['risk_off_quantile']:.2f}`, VIX `{selected_quantiles['vix_quantile']:.2f}`, credit `{selected_quantiles['credit_quantile']:.2f}`.",
        f"- Local variants evaluated: `{decision['local_threshold_variant_count']}`.",
        f"- Local variants passing full gates: `{decision['local_threshold_pass_count']}`.",
        f"- Decision: `{decision['status']}`.",
        f"- Warnings: `{', '.join(decision.get('warnings', [])) if decision.get('warnings') else 'none'}`.",
        "",
        "## Anchor Variant",
        "",
        *_markdown_table(
            anchor,
            [
                "variant_id",
                "status",
                "failed_gate_count",
                "validation_trades",
                "validation_net_return",
                "validation_stress_net_return",
                "validation_avg_trade_net_bps",
                "validation_max_top5_abs_share",
                "test_trades",
                "test_net_return",
                "test_stress_net_return",
                "test_avg_trade_net_bps",
                "test_max_top5_abs_share",
            ],
            limit=5,
        ),
        "",
        "## Local Threshold Sweep",
        "",
        *_markdown_table(
            local_sweep.head(27),
            [
                "variant_id",
                "is_anchor",
                "status",
                "failed_gate_count",
                "validation_net_return",
                "validation_avg_trade_net_bps",
                "validation_max_top5_abs_share",
                "test_net_return",
                "test_avg_trade_net_bps",
                "test_max_top5_abs_share",
            ],
            limit=27,
        ),
        "",
        "## Cost Sensitivity",
        "",
        *_markdown_table(
            cost_sensitivity,
            ["split", "cost_bps", "trades", "net_return", "avg_trade_net_bps", "positive_folds", "control_edge"],
            limit=20,
        ),
        "",
        "## Subperiod Summary",
        "",
        *_markdown_table(
            subperiod_summary,
            ["split", "year", "trades", "net_return", "avg_trade_net_bps", "win_rate"],
            limit=20,
        ),
        "",
        "## Fold Stability",
        "",
        *_markdown_table(
            fold_stability,
            ["split", "fold", "trades", "net_return", "avg_trade_net_bps", "daily_sharpe", "max_drawdown", "sessions_with_trades", "top5_abs_share"],
            limit=20,
        ),
        "",
        "## Decision",
        "",
        "```yaml",
        yaml.safe_dump(decision, sort_keys=False).strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    outputs: H1BRobustnessOutputs,
    strategy: StrategySpec,
    raw_spec: dict[str, Any],
    strategy_spec_path: Path,
    freeze_dir: Path,
    feature_path: Path,
    context_path: Path,
    decision: dict[str, Any],
) -> None:
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("pre_paper_robustness", strategy.strategy_id, strategy.target_symbol, strategy.timeframe),
            "run_type": "pre_paper_robustness",
            "created_at_utc": utc_now(),
            "status": decision["status"],
        },
        "strategy": strategy.to_dict(),
        "alpha": raw_spec.get("alpha", {}),
        "decision": decision,
        "data": {
            "features_path": feature_path.as_posix(),
            "features_fingerprint": _fingerprint_or_missing(feature_path),
            "risk_context_path": context_path.as_posix(),
            "risk_context_fingerprint": _fingerprint_or_missing(context_path),
            "split_policy": DEFAULT_SPLIT_POLICY,
        },
        "source_artifacts": {
            "strategy_spec": {
                "path": strategy_spec_path.as_posix(),
                "fingerprint": _fingerprint_or_missing(strategy_spec_path),
            },
            "freeze_manifest": {
                "path": (freeze_dir / "manifest.yaml").as_posix(),
                "fingerprint": _fingerprint_or_missing(freeze_dir / "manifest.yaml"),
            },
        },
        "outputs": {key: value.as_posix() for key, value in outputs.__dict__.items() if key.endswith("_path")},
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def run_h1b_robustness(
    *,
    strategy_spec_path: str | Path = DEFAULT_STRATEGY_SPEC_PATH,
    freeze_dir: str | Path = DEFAULT_FREEZE_DIR,
    output_dir: str | Path = DEFAULT_ROBUSTNESS_DIR,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    extra_cost_bps: tuple[float, ...] = DEFAULT_EXTRA_COST_BPS,
) -> H1BRobustnessOutputs:
    spec_path = Path(strategy_spec_path)
    raw_spec = _load_yaml(spec_path)
    strategy = StrategySpec.from_yaml(spec_path)
    selected_quantiles = selected_quantiles_from_spec(raw_spec)
    feature_path = Path(features_path)
    context_path = Path(risk_context_path)
    frame = load_eda_frame(feature_path, context_path, (DEFAULT_HORIZON,))
    folds = build_monthly_folds(frame, DEFAULT_SPLIT_POLICY)
    if not folds:
        raise ValueError("split policy produced no folds")

    local_sweep, local_gates = build_local_threshold_sweep(frame, selected_quantiles)
    selected_filter = credit_filter_policy(selected_quantiles["credit_quantile"])
    selected_trades, selected_summary, selected_concentration = run_h1b_variant_backtest(
        frame,
        folds,
        risk_off_quantile=selected_quantiles["risk_off_quantile"],
        vix_quantile=selected_quantiles["vix_quantile"],
        filter_policy=selected_filter,
        splits=("validation", "test"),
        cost_bps_values=tuple(float(value) for value in (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS, *extra_cost_bps)),
    )
    cost_sensitivity = build_cost_sensitivity(
        selected_summary,
        tuple(float(value) for value in (DEFAULT_COST_BPS, DEFAULT_STRESS_COST_BPS, *extra_cost_bps)),
    )
    subperiod_summary = build_subperiod_summary(enrich_trade_times(selected_trades))
    fold_stability = build_fold_stability(selected_summary, selected_concentration)
    decision = decide_robustness(local_sweep, cost_sensitivity, fold_stability)

    root = Path(output_dir)
    outputs = H1BRobustnessOutputs(
        output_dir=root,
        report_path=root / "report.md",
        manifest_path=root / "manifest.yaml",
        local_threshold_sweep_path=root / "local_threshold_sweep.parquet",
        local_threshold_gates_path=root / "local_threshold_gates.parquet",
        cost_sensitivity_path=root / "cost_sensitivity.parquet",
        subperiod_summary_path=root / "subperiod_summary.parquet",
        fold_stability_path=root / "fold_stability.parquet",
        robustness_decision_path=root / "robustness_decision.yaml",
    )
    root.mkdir(parents=True, exist_ok=True)
    local_sweep.to_parquet(outputs.local_threshold_sweep_path, index=False)
    local_gates.to_parquet(outputs.local_threshold_gates_path, index=False)
    cost_sensitivity.to_parquet(outputs.cost_sensitivity_path, index=False)
    subperiod_summary.to_parquet(outputs.subperiod_summary_path, index=False)
    fold_stability.to_parquet(outputs.fold_stability_path, index=False)
    outputs.robustness_decision_path.write_text(yaml.safe_dump(decision, sort_keys=False), encoding="utf-8")
    _write_report(outputs.report_path, strategy, selected_quantiles, local_sweep, cost_sensitivity, subperiod_summary, fold_stability, decision)
    _write_manifest(outputs.manifest_path, outputs, strategy, raw_spec, spec_path, Path(freeze_dir), feature_path, context_path, decision)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H1b risk-off short pre-paper robustness checks")
    parser.add_argument("--strategy-spec", default=str(DEFAULT_STRATEGY_SPEC_PATH))
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROBUSTNESS_DIR))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    args = parser.parse_args(argv)
    outputs = run_h1b_robustness(
        strategy_spec_path=args.strategy_spec,
        freeze_dir=args.freeze_dir,
        output_dir=args.output_dir,
        features_path=args.features,
        risk_context_path=args.risk_context,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

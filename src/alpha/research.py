from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.alpha.specs import AlphaResearchPlan, AlphaSpec, alpha_position, fit_confirmation_gates, load_alpha_research_plan, thresholds_for_spec
from src.backtesting import BacktestMetrics, evaluate_positions
from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds


DEFAULT_COST_BPS = {
    "ibkr_tiered_10000": 1.0,
    "bps_1": 1.0,
    "bps_2": 2.0,
    "bps_5": 5.0,
}


@dataclass(frozen=True)
class AlphaResearchArtifacts:
    output_dir: Path
    validation_path: Path
    test_path: Path
    decisions_path: Path
    manifest_path: Path
    report_path: Path


def cost_bps(cost_profile: str) -> float:
    if cost_profile in DEFAULT_COST_BPS:
        return DEFAULT_COST_BPS[cost_profile]
    if cost_profile.startswith("bps_"):
        return float(cost_profile.removeprefix("bps_"))
    raise ValueError(f"unknown cost profile: {cost_profile}")


def load_features(plan: AlphaResearchPlan) -> pd.DataFrame:
    path = plan.feature_path()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"timestamp", "session", "bar_index", "target_open_next", *plan.required_feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature data is missing required columns: {missing}")
    return frame.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)


def add_forward_returns(frame: pd.DataFrame, horizons: set[int]) -> pd.DataFrame:
    output = frame.copy()
    entry = output["target_open_next"].replace([np.inf, -np.inf], np.nan).astype(float)
    grouped_next_open = output.groupby("session", sort=False)["target_open_next"]
    for horizon in sorted(horizons):
        exit_px = grouped_next_open.shift(-int(horizon)).replace([np.inf, -np.inf], np.nan).astype(float)
        valid = entry.gt(0.0) & exit_px.gt(0.0)
        if "target_can_open_trade" in output:
            valid &= output["target_can_open_trade"].fillna(False).astype(bool)
        output[f"fwd_ret_{int(horizon)}"] = np.where(valid, np.log(exit_px / entry), np.nan)
    return output


def _split_frame(frame: pd.DataFrame, sessions: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["session"].astype(str).isin(sessions)].copy()


def _cost_profiles(plan: AlphaResearchPlan) -> tuple[str, ...]:
    return tuple(dict.fromkeys([plan.primary_cost_profile, plan.conservative_cost_profile, plan.stress_cost_profile]))


def _candidate_id(fold: ResearchFold, spec: AlphaSpec, horizon: int, threshold: float) -> str:
    return f"fold{fold.fold}__{spec.alpha_id}__h{int(horizon)}__thr{float(threshold):g}"


def _metric_row(
    *,
    plan: AlphaResearchPlan,
    fold: ResearchFold,
    spec: AlphaSpec,
    split: str,
    horizon: int,
    threshold: float,
    cost_profile: str,
    metrics: BacktestMetrics,
) -> dict[str, Any]:
    return {
        "research_id": plan.research_id,
        "target_symbol": plan.target_symbol,
        "timeframe": plan.timeframe,
        "feature_set_id": plan.feature_set_id,
        "fold": fold.fold,
        "alpha_id": spec.alpha_id,
        "family": spec.family,
        "split": split,
        "horizon_bars": int(horizon),
        "threshold": float(threshold),
        "cost_profile": cost_profile,
        "cost_bps": cost_bps(cost_profile),
        "candidate_id": _candidate_id(fold, spec, horizon, threshold),
        **metrics.to_dict(),
    }


def evaluate_alpha_grid(plan: AlphaResearchPlan, frame: pd.DataFrame, folds: tuple[ResearchFold, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = {horizon for spec in plan.alphas for horizon in spec.horizons}
    prepared = add_forward_returns(frame, horizons)
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for fold in folds:
        validation = _split_frame(prepared, fold.validation_sessions)
        test = _split_frame(prepared, fold.test_sessions)
        if validation.empty or test.empty:
            continue
        gates = fit_confirmation_gates(validation, plan.alphas)
        for spec in plan.alphas:
            thresholds = thresholds_for_spec(validation, spec)
            for horizon in spec.horizons:
                return_column = f"fwd_ret_{int(horizon)}"
                for threshold in thresholds:
                    for split_name, split_frame, rows in (("validation", validation, validation_rows), ("test", test, test_rows)):
                        position = alpha_position(split_frame, spec, threshold, gates)
                        for cost_profile in _cost_profiles(plan):
                            metrics = evaluate_positions(
                                split_frame,
                                position,
                                return_column=return_column,
                                cost_bps=cost_bps(cost_profile),
                            )
                            rows.append(
                                _metric_row(
                                    plan=plan,
                                    fold=fold,
                                    spec=spec,
                                    split=split_name,
                                    horizon=horizon,
                                    threshold=threshold,
                                    cost_profile=cost_profile,
                                    metrics=metrics,
                                )
                            )
    return pd.DataFrame(validation_rows), pd.DataFrame(test_rows)


def build_decisions(plan: AlphaResearchPlan, validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    gates = plan.promotion_gates
    primary = validation[validation["cost_profile"].eq(plan.primary_cost_profile)].copy()
    conservative = validation[validation["cost_profile"].eq(plan.conservative_cost_profile)].loc[:, ["candidate_id", "net_return"]].rename(
        columns={"net_return": "conservative_net_return"}
    )
    stress = validation[validation["cost_profile"].eq(plan.stress_cost_profile)].loc[:, ["candidate_id", "net_return"]].rename(
        columns={"net_return": "stress_net_return"}
    )
    primary = primary.merge(conservative, on="candidate_id", how="left", validate="one_to_one")
    primary = primary.merge(stress, on="candidate_id", how="left", validate="one_to_one")
    primary["decision"] = "rejected_validation_failed"
    primary["failure_reasons"] = ""

    failures: list[str] = []
    for _, row in primary.iterrows():
        row_failures: list[str] = []
        if float(row["trades"]) < float(gates.get("min_trades", 50)):
            row_failures.append("insufficient_trades")
        if bool(gates.get("require_positive_primary_cost", True)) and float(row["net_return"]) <= 0.0:
            row_failures.append("nonpositive_primary_net")
        conservative_net = row.get("conservative_net_return", np.nan)
        if bool(gates.get("require_nonnegative_conservative_cost", True)) and pd.notna(conservative_net) and float(conservative_net) < 0.0:
            row_failures.append("negative_conservative_net")
        if float(row["profit_factor"]) < float(gates.get("min_profit_factor", 1.10)):
            row_failures.append("weak_profit_factor")
        if float(row["daily_sharpe"]) < float(gates.get("min_daily_sharpe", 1.0)):
            row_failures.append("weak_daily_sharpe")
        failures.append(",".join(row_failures))

    primary["failure_reasons"] = failures
    primary.loc[primary["failure_reasons"].eq(""), "decision"] = "accepted_validation_candidate"
    selected_cols = [
        "candidate_id",
        "research_id",
        "target_symbol",
        "timeframe",
        "feature_set_id",
        "fold",
        "alpha_id",
        "family",
        "horizon_bars",
        "threshold",
        "cost_profile",
        "trades",
        "net_return",
        "conservative_net_return",
        "stress_net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "decision",
        "failure_reasons",
    ]
    return primary.loc[:, selected_cols].sort_values(
        ["decision", "net_return", "daily_sharpe"],
        ascending=[True, False, False],
        kind="stable",
    )


def _manifest(plan: AlphaResearchPlan, feature_path: Path, artifacts: AlphaResearchArtifacts, folds: tuple[ResearchFold, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("alpha_research", plan.research_id, plan.target_symbol, plan.timeframe, plan.feature_set_id),
            "run_type": "alpha_research",
            "created_at_utc": utc_now(),
            "status": "complete",
        },
        "research": {
            "research_id": plan.research_id,
            "target_symbol": plan.target_symbol,
            "timeframe": plan.timeframe,
            "feature_set_id": plan.feature_set_id,
            "split_policy_id": plan.split_policy_id,
            "n_alphas": len(plan.alphas),
            "n_folds": len(folds),
        },
        "data": {
            "feature_path": feature_path.as_posix(),
            "feature_fingerprint": fingerprint_path(feature_path) if feature_path.exists() else "MISSING",
        },
        "costs": {
            "primary": plan.primary_cost_profile,
            "conservative": plan.conservative_cost_profile,
            "stress": plan.stress_cost_profile,
        },
        "artifacts": {
            "validation": artifacts.validation_path.as_posix(),
            "test": artifacts.test_path.as_posix(),
            "candidate_decisions": artifacts.decisions_path.as_posix(),
            "report": artifacts.report_path.as_posix(),
        },
    }


def _write_report(path: Path, plan: AlphaResearchPlan, folds: tuple[ResearchFold, ...], decisions: pd.DataFrame) -> None:
    accepted = int(decisions["decision"].eq("accepted_validation_candidate").sum()) if not decisions.empty else 0
    top = decisions.head(10) if not decisions.empty else pd.DataFrame()
    lines = [
        f"# Alpha research - {plan.research_id}",
        "",
        f"- Target: `{plan.target_symbol}`",
        f"- Timeframe: `{plan.timeframe}`",
        f"- Feature set: `{plan.feature_set_id}`",
        f"- Alphas: `{len(plan.alphas)}`",
        f"- Folds: `{len(folds)}`",
        f"- Accepted validation candidates: `{accepted}`",
        "",
        "## Top decisions",
        "",
    ]
    if top.empty:
        lines.append("No decisions generated.")
    else:
        columns = [
            "candidate_id",
            "decision",
            "trades",
            "net_return",
            "profit_factor",
            "daily_sharpe",
            "failure_reasons",
        ]
        visible = top.loc[:, [column for column in columns if column in top.columns]].copy()
        lines.append("| " + " | ".join(visible.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(visible.columns)) + " |")
        for _, row in visible.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in visible.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_alpha_research(plan: AlphaResearchPlan, *, output_dir: str | Path | None = None, write: bool = True) -> AlphaResearchArtifacts:
    features = load_features(plan)
    folds = build_monthly_folds(features, plan.split_policy)
    if not folds:
        raise ValueError("split policy produced no folds")
    validation, test = evaluate_alpha_grid(plan, features, folds)
    decisions = build_decisions(plan, validation)

    root = Path(output_dir) if output_dir is not None else plan.output_dir()
    artifacts = AlphaResearchArtifacts(
        output_dir=root,
        validation_path=root / "validation.parquet",
        test_path=root / "test.parquet",
        decisions_path=root / "candidate_decisions.parquet",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    if write:
        root.mkdir(parents=True, exist_ok=True)
        validation.to_parquet(artifacts.validation_path, index=False)
        test.to_parquet(artifacts.test_path, index=False)
        decisions.to_parquet(artifacts.decisions_path, index=False)
        artifacts.manifest_path.write_text(
            yaml.safe_dump(_manifest(plan, plan.feature_path(), artifacts, folds), sort_keys=False),
            encoding="utf-8",
        )
        _write_report(artifacts.report_path, plan, folds, decisions)
    return artifacts


def dry_run_summary(plan: AlphaResearchPlan) -> dict[str, Any]:
    features_path = plan.feature_path()
    summary = {
        "research_id": plan.research_id,
        "target_symbol": plan.target_symbol,
        "timeframe": plan.timeframe,
        "feature_set_id": plan.feature_set_id,
        "feature_path": features_path.as_posix(),
        "feature_path_exists": features_path.exists(),
        "output_dir": plan.output_dir().as_posix(),
        "alphas": [alpha.alpha_id for alpha in plan.alphas],
        "required_feature_columns": list(plan.required_feature_columns),
        "split_policy": plan.split_policy,
        "cost_profiles": list(_cost_profiles(plan)),
    }
    if features_path.exists():
        features = load_features(plan)
        summary["rows"] = int(len(features))
        summary["folds"] = len(build_monthly_folds(features, plan.split_policy))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run IDA alpha research")
    parser.add_argument("--config", default="configs/alpha/alpha_research_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    plan = load_alpha_research_plan(args.config)
    if args.dry_run:
        print(json.dumps(dry_run_summary(plan), indent=2, sort_keys=True))
        return
    artifacts = run_alpha_research(plan, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in artifacts.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

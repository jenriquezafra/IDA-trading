from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.alpha.risk_off_eda import DEFAULT_FEATURES_PATH, DEFAULT_RISK_CONTEXT_PATH, load_eda_frame
from src.research.manifest import build_run_id, fingerprint_path, utc_now
from src.research.splits import ResearchFold, build_monthly_folds
from src.strategy import StrategySpec
from src.strategy.risk_off_short import DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_POLICY, fit_thresholds
from src.strategy.risk_off_short_h1c_credit_repair import DEFAULT_H1C_OUTPUT_DIR, fit_credit_repair_filter
from src.strategy.risk_off_short_triage import DEFAULT_HORIZON


DEFAULT_STRATEGY_SPEC_PATH = Path("configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml")
DEFAULT_FREEZE_DIR = DEFAULT_OUTPUT_DIR / "freeze_review" / "qqq_15min_risk_off_short_h1c_v1"


@dataclass(frozen=True)
class RiskOffH1CFreezeOutputs:
    output_dir: Path
    strategy_spec_snapshot_path: Path
    fold_thresholds_path: Path
    frozen_decision_path: Path
    manifest_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def _fingerprint_or_missing(path: Path) -> str:
    return fingerprint_path(path) if path.exists() else "MISSING"


def _session_bounds(sessions: tuple[str, ...]) -> tuple[str, str, int]:
    if not sessions:
        return "", "", 0
    return str(sessions[0]), str(sessions[-1]), len(sessions)


def build_h1c_fold_thresholds(
    frame: pd.DataFrame,
    folds: tuple[ResearchFold, ...],
    *,
    risk_off_quantile: float,
    vix_quantile: float,
    credit_policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train = frame[frame["session"].astype(str).isin(fold.train_sessions)].copy()
        thresholds = fit_thresholds(train, risk_off_quantile=risk_off_quantile, vix_quantile=vix_quantile)
        fitted_filter = fit_credit_repair_filter(train, credit_policy)
        train_start, train_end, train_sessions = _session_bounds(fold.train_sessions)
        validation_start, validation_end, validation_sessions = _session_bounds(fold.validation_sessions)
        test_start, test_end, test_sessions = _session_bounds(fold.test_sessions)
        row: dict[str, Any] = {
            "fold": int(fold.fold),
            "risk_off_quantile": float(risk_off_quantile),
            "vix_quantile": float(vix_quantile),
            "credit_policy": credit_policy,
            "risk_off_min": float(thresholds.risk_off_min),
            "vix_z20_min": float(thresholds.vix_z20_min),
            "active_hours": ",".join(str(hour) for hour in thresholds.active_hours),
            "train_start": train_start,
            "train_end": train_end,
            "train_sessions": train_sessions,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "validation_sessions": validation_sessions,
            "test_start": test_start,
            "test_end": test_end,
            "test_sessions": test_sessions,
        }
        for key, value in fitted_filter.thresholds.items():
            row[f"{key}_threshold"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _artifact_fingerprints(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": path.as_posix(),
            "fingerprint": _fingerprint_or_missing(path),
        }
        for name, path in paths.items()
    }


def freeze_h1c_strategy(
    *,
    strategy_spec_path: str | Path = DEFAULT_STRATEGY_SPEC_PATH,
    h1c_dir: str | Path = DEFAULT_H1C_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_FREEZE_DIR,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
) -> RiskOffH1CFreezeOutputs:
    spec_path = Path(strategy_spec_path)
    h1c_root = Path(h1c_dir)
    root = Path(output_dir)
    feature_path = Path(features_path)
    context_path = Path(risk_context_path)

    raw_spec = _load_yaml(spec_path)
    strategy = StrategySpec.from_yaml(spec_path)
    selected_variant = _load_yaml(h1c_root / "selected_variant.yaml")
    selected_decision = _load_yaml(h1c_root / "selected_decision.yaml")
    promotion_decision = dict(selected_decision.get("promotion", {}))
    repair_decision = dict(selected_decision.get("repair", {}))
    expected_variant = str(raw_spec.get("alpha", {}).get("selected_variant_id", ""))
    actual_variant = str(selected_variant.get("variant_id", ""))
    if expected_variant and actual_variant and expected_variant != actual_variant:
        raise ValueError(f"strategy spec variant {expected_variant} does not match selected variant {actual_variant}")
    if promotion_decision.get("status") != "freeze_review":
        raise ValueError(f"H1c selected promotion is not freeze_review: {promotion_decision.get('status')}")
    if repair_decision.get("status") != "credit_repaired":
        raise ValueError(f"H1c repair status is not credit_repaired: {repair_decision.get('status')}")

    frame = load_eda_frame(feature_path, context_path, (DEFAULT_HORIZON,))
    folds = build_monthly_folds(frame, DEFAULT_SPLIT_POLICY)
    thresholds = build_h1c_fold_thresholds(
        frame,
        folds,
        risk_off_quantile=float(selected_variant["risk_off_quantile"]),
        vix_quantile=float(selected_variant["vix_quantile"]),
        credit_policy=str(selected_variant["credit_policy"]),
    )

    outputs = RiskOffH1CFreezeOutputs(
        output_dir=root,
        strategy_spec_snapshot_path=root / "strategy_spec.yaml",
        fold_thresholds_path=root / "fold_thresholds.parquet",
        frozen_decision_path=root / "freeze_review_decision.yaml",
        manifest_path=root / "manifest.yaml",
    )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec_path, outputs.strategy_spec_snapshot_path)
    thresholds.to_parquet(outputs.fold_thresholds_path, index=False)

    frozen_decision = {
        "status": "freeze_review",
        "strategy_id": strategy.strategy_id,
        "selected_variant_id": actual_variant,
        "selected_with": "validation_only",
        "test_usage": "confirmation_only",
        "promotion_status": promotion_decision.get("status"),
        "repair_status": repair_decision.get("status"),
        "warnings": repair_decision.get("warnings", []),
        "next_required_step": "pre_paper_robustness_review",
    }
    outputs.frozen_decision_path.write_text(yaml.safe_dump(frozen_decision, sort_keys=False), encoding="utf-8")

    source_artifacts = {
        "source_strategy_spec": spec_path,
        "features": feature_path,
        "risk_context": context_path,
        "h1c_report": h1c_root / "report.md",
        "h1c_manifest": h1c_root / "manifest.yaml",
        "h1c_validation_sweep": h1c_root / "validation_sweep.parquet",
        "h1c_validation_gates": h1c_root / "validation_gates.parquet",
        "h1c_selected_variant": h1c_root / "selected_variant.yaml",
        "h1c_selected_trades": h1c_root / "selected_trades.parquet",
        "h1c_selected_controls": h1c_root / "selected_controls.parquet",
        "h1c_selected_concentration": h1c_root / "selected_concentration.parquet",
        "h1c_selected_gates": h1c_root / "selected_gates.parquet",
        "h1c_selected_cost_sensitivity": h1c_root / "selected_cost_sensitivity.parquet",
        "h1c_selected_decision": h1c_root / "selected_decision.yaml",
        "frozen_strategy_spec": outputs.strategy_spec_snapshot_path,
        "frozen_fold_thresholds": outputs.fold_thresholds_path,
        "frozen_decision": outputs.frozen_decision_path,
    }
    manifest = {
        "schema_version": 1,
        "run": {
            "run_id": build_run_id("freeze_review", strategy.strategy_id, strategy.target_symbol, strategy.timeframe),
            "run_type": "freeze_review",
            "created_at_utc": utc_now(),
            "status": "freeze_review",
        },
        "strategy": strategy.to_dict(),
        "alpha": raw_spec.get("alpha", {}),
        "selection": {
            "source_dir": h1c_root.as_posix(),
            "selected_variant": selected_variant,
            "promotion_decision": promotion_decision,
            "repair_decision": repair_decision,
        },
        "data": {
            "features_path": feature_path.as_posix(),
            "features_fingerprint": _fingerprint_or_missing(feature_path),
            "risk_context_path": context_path.as_posix(),
            "risk_context_fingerprint": _fingerprint_or_missing(context_path),
            "split_policy": DEFAULT_SPLIT_POLICY,
            "fold_thresholds_path": outputs.fold_thresholds_path.as_posix(),
            "fold_thresholds_fingerprint": _fingerprint_or_missing(outputs.fold_thresholds_path),
        },
        "artifacts": _artifact_fingerprints(source_artifacts),
        "review": {
            "required_before_paper": [
                "pre-paper robustness on H1c StrategySpec",
                "higher-cost review: 10 bps is negative",
                "paper runner wired to StrategySpec",
            ]
        },
    }
    outputs.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze H1c risk-off short StrategySpec and artifacts")
    parser.add_argument("--strategy-spec", default=str(DEFAULT_STRATEGY_SPEC_PATH))
    parser.add_argument("--h1c-dir", default=str(DEFAULT_H1C_OUTPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    args = parser.parse_args(argv)
    outputs = freeze_h1c_strategy(
        strategy_spec_path=args.strategy_spec,
        h1c_dir=args.h1c_dir,
        output_dir=args.output_dir,
        features_path=args.features,
        risk_context_path=args.risk_context,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

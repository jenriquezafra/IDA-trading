from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPORT_ARTIFACTS = [
    "reports/baseline_status.md",
    "reports/baseline_status.parquet",
    "reports/baseline_trades.parquet",
    "reports/backtest_trades.parquet",
    "reports/leakage_audit.md",
    "reports/leakage_audit.parquet",
    "reports/hmm_state_economics.md",
    "reports/hmm_state_economics/*.parquet",
    "reports/hmm_feature_lab.md",
    "reports/hmm_feature_lab/*.parquet",
    "reports/hmm_stability.md",
    "reports/hmm_stability/*.parquet",
    "reports/hmm_candidate_diagnostics.md",
    "reports/hmm_candidate_diagnostics/*.parquet",
    "reports/hmm_candidate_thresholds.md",
    "reports/hmm_candidate_thresholds/*.parquet",
    "reports/walkforward_summary.md",
    "reports/walkforward_folds_summary.md",
    "reports/walkforward/fold_summary.parquet",
]

DATA_INPUTS = [
    "data/raw/spy_5min.parquet",
    "data/cleaned/spy_5min_clean.parquet",
    "data/features/features_base.parquet",
    "data/features/hmm_features.parquet",
    "data/features/labels.parquet",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_existing(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        paths.extend(path for path in matched if path.is_file())
    return sorted(dict.fromkeys(paths))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _optional_read_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    return pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()


def _first_record(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if not frame.empty else {}


def build_frozen_results(
    baseline_status: pd.DataFrame,
    selected_test_results: pd.DataFrame,
    candidate_decisions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in baseline_status.iterrows():
        rows.append(
            {
                "result_group": "baseline_status",
                "id": row["strategy"],
                "candidate_id": None,
                "source": row.get("source"),
                "status": row.get("status"),
                "accepted": row.get("status") == "candidate_requires_more_evidence",
                "cost_fragile": False,
                "cost_bps": row.get("cost_bps"),
                "trades": row.get("trades"),
                "net_return": row.get("net_return"),
                "daily_sharpe_net": row.get("daily_sharpe_net"),
                "profit_factor_net": row.get("profit_factor_net"),
                "avg_trade_net": row.get("avg_trade_net"),
                "max_drawdown": row.get("max_drawdown"),
                "folds_positive": row.get("folds_positive"),
                "folds_negative": row.get("folds_negative"),
                "notes": "Legacy predictive/static/walk-forward baseline.",
            }
        )

    decisions = (
        candidate_decisions.set_index("candidate_id").to_dict("index")
        if not candidate_decisions.empty and "candidate_id" in candidate_decisions
        else {}
    )
    for _, row in selected_test_results.iterrows():
        candidate_id = row["candidate_id"]
        decision = decisions.get(candidate_id, {})
        rows.append(
            {
                "result_group": "hmm_candidate_threshold",
                "id": f"{candidate_id}__c{float(row['cost_bps']):g}",
                "candidate_id": candidate_id,
                "source": "reports/hmm_candidate_thresholds/selected_test_results.parquet",
                "status": row.get("candidate_status"),
                "accepted": bool(decision.get("accepted", False)),
                "cost_fragile": bool(decision.get("cost_fragile", False)),
                "cost_bps": row.get("cost_bps"),
                "trades": row.get("total_trades"),
                "net_return": row.get("total_net_return"),
                "daily_sharpe_net": row.get("median_daily_sharpe"),
                "profit_factor_net": row.get("median_profit_factor"),
                "avg_trade_net": row.get("avg_trade_net"),
                "max_drawdown": row.get("max_drawdown_abs"),
                "folds_positive": row.get("positive_folds"),
                "folds_negative": row.get("negative_folds"),
                "notes": "SPY-only HMM threshold result selected on validation, reported on test.",
            }
        )

    return pd.DataFrame(rows)


def _cleaned_data_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    frame = pd.read_parquet(path, columns=["timestamp", "session"])
    timestamps = pd.to_datetime(frame["timestamp"])
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "start": timestamps.min().isoformat() if len(timestamps) else None,
        "end": timestamps.max().isoformat() if len(timestamps) else None,
    }


def _leakage_summary(leakage: pd.DataFrame) -> dict[str, Any]:
    if leakage.empty or "status" not in leakage:
        return {"available": False}
    status_counts = leakage["status"].value_counts(dropna=False).to_dict()
    return {
        "available": True,
        "rows": int(len(leakage)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "critical_failures": int((leakage["status"].astype(str).str.upper() != "PASS").sum()),
    }


def _best_candidate(candidate_decisions: pd.DataFrame, selected_test_results: pd.DataFrame) -> dict[str, Any]:
    if candidate_decisions.empty:
        return {}
    ordered = candidate_decisions.sort_values(["source_rank"], kind="stable")
    best = ordered.iloc[0].to_dict()
    candidate_id = best["candidate_id"]
    selected = selected_test_results[selected_test_results["candidate_id"] == candidate_id]
    best["selected_test_rows"] = selected.sort_values("cost_bps").to_dict("records")
    return best


def build_summary(
    config: dict[str, Any],
    baseline_status: pd.DataFrame,
    selected_test_results: pd.DataFrame,
    candidate_decisions: pd.DataFrame,
    leakage: pd.DataFrame,
    artifact_paths: list[Path],
    data_paths: list[Path],
) -> dict[str, Any]:
    accepted = int(candidate_decisions["accepted"].sum()) if "accepted" in candidate_decisions else 0
    cost_fragile = int(candidate_decisions["cost_fragile"].sum()) if "cost_fragile" in candidate_decisions else 0
    baseline_status_counts = (
        baseline_status["status"].value_counts(dropna=False).to_dict() if "status" in baseline_status else {}
    )
    source_hashes = {
        str(path): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(artifact_paths + data_paths)
        if path.exists()
    }
    return {
        "baseline_id": "spy_only_hmm",
        "status": "rejected_cost_fragile",
        "target_symbol": config.get("data", {}).get("symbol", "SPY"),
        "timeframe": config.get("project", {}).get("frequency", "5min"),
        "provider": config.get("data", {}).get("provider", "polygon"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "economic_conclusion": (
            "SPY-only direct directional edge is not accepted. Legacy predictive baselines are rejected; "
            "best HMM fallback is valid at 1 bps but cost-fragile at 2 bps."
        ),
        "cost_profiles_bps": {
            "base": config.get("backtest", {}).get("base_round_trip_cost_bps"),
            "conservative": config.get("backtest", {}).get("conservative_round_trip_cost_bps"),
            "stress": config.get("backtest", {}).get("stress_round_trip_cost_bps"),
        },
        "data_summary": _cleaned_data_summary(Path(config.get("data", {}).get("cleaned_file", ""))),
        "baseline_status_counts": {str(key): int(value) for key, value in baseline_status_counts.items()},
        "candidate_decision_counts": {
            "rows": int(len(candidate_decisions)),
            "accepted": accepted,
            "cost_fragile": cost_fragile,
        },
        "best_fallback_candidate": _best_candidate(candidate_decisions, selected_test_results),
        "leakage_audit": _leakage_summary(leakage),
        "freeze_policy": {
            "branch": "SPY-only HMM",
            "allowed_future_changes": "Only verified bug fixes or report regeneration with identical methodology.",
            "not_allowed": "Further parameter/feature/threshold optimization on SPY-only without a new written hypothesis.",
        },
        "source_hashes": source_hashes,
    }


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in display.columns) + " |")
    return "\n".join(lines)


def render_report(
    summary: dict[str, Any],
    baseline_status: pd.DataFrame,
    candidate_decisions: pd.DataFrame,
    results: pd.DataFrame,
) -> str:
    compact_baseline_cols = [
        "strategy",
        "cost_bps",
        "trades",
        "net_return",
        "daily_sharpe_net",
        "profit_factor_net",
        "avg_trade_net",
        "max_drawdown",
        "folds_positive",
        "folds_negative",
        "status",
    ]
    candidate_cols = [
        "source_rank",
        "candidate_id",
        "feature_set",
        "status_1bps",
        "status_2bps",
        "accepted",
        "cost_fragile",
        "test_net_1bps",
        "test_net_2bps",
        "test_drawdown_1bps",
        "test_drawdown_2bps",
    ]
    result_counts = results["result_group"].value_counts().to_dict() if not results.empty else {}
    return f"""# Frozen SPY-only HMM Baseline

## Status

- Baseline id: `spy_only_hmm`
- Target: `{summary["target_symbol"]}`
- Timeframe: `{summary["timeframe"]}`
- Provider: `{summary["provider"]}`
- Final status: `rejected_cost_fragile`
- Generated UTC: `{summary["generated_at_utc"]}`

## Conclusion

{summary["economic_conclusion"]}

The frozen baseline is a comparison target for the cross-asset HMM branch. It is not an accepted strategy.

## Data Snapshot

- Cleaned file: `{summary["data_summary"].get("path")}`
- Rows: `{summary["data_summary"].get("rows")}`
- Sessions: `{summary["data_summary"].get("sessions")}`
- Start: `{summary["data_summary"].get("start")}`
- End: `{summary["data_summary"].get("end")}`

## Legacy Baselines

{markdown_table(baseline_status, compact_baseline_cols)}

## HMM Candidate Decisions

{markdown_table(candidate_decisions, candidate_cols)}

## Leakage Audit

- Available: `{summary["leakage_audit"].get("available")}`
- Rows: `{summary["leakage_audit"].get("rows")}`
- Critical failures: `{summary["leakage_audit"].get("critical_failures")}`
- Status counts: `{summary["leakage_audit"].get("status_counts")}`

## Frozen Outputs

- `baselines/spy_only_hmm/config.yaml`
- `baselines/spy_only_hmm/results.parquet`
- `baselines/spy_only_hmm/summary.json`
- `baselines/spy_only_hmm/source_artifacts/`
- `reports/baseline_spy_only_frozen.md`

Result row groups: `{result_counts}`

## Freeze Policy

- Allowed future changes: {summary["freeze_policy"]["allowed_future_changes"]}
- Not allowed: {summary["freeze_policy"]["not_allowed"]}
"""


def copy_artifacts(paths: list[Path], output_dir: Path) -> None:
    artifact_root = output_dir / "source_artifacts"
    for source in paths:
        destination = artifact_root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def freeze_spy_only_baseline(
    config_path: str | Path = "configs/base.yaml",
    output_dir: str | Path = "baselines/spy_only_hmm",
    report_path: str | Path = "reports/baseline_spy_only_frozen.md",
) -> Path:
    config_path = Path(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    baseline_status = _optional_read_parquet("reports/baseline_status.parquet")
    selected_test_results = _optional_read_parquet("reports/hmm_candidate_thresholds/selected_test_results.parquet")
    candidate_decisions = _optional_read_parquet("reports/hmm_candidate_thresholds/candidate_decisions.parquet")
    leakage = _optional_read_parquet("reports/leakage_audit.parquet")

    artifact_paths = expand_existing(REPORT_ARTIFACTS)
    data_paths = expand_existing(DATA_INPUTS)
    results = build_frozen_results(baseline_status, selected_test_results, candidate_decisions)
    summary = build_summary(config, baseline_status, selected_test_results, candidate_decisions, leakage, artifact_paths, data_paths)

    shutil.copy2(config_path, output_path / "config.yaml")
    results.to_parquet(output_path / "results.parquet", index=False)
    (output_path / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    copy_artifacts(artifact_paths, output_path)

    report = render_report(summary, baseline_status, candidate_decisions, results)
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(report, encoding="utf-8")
    (output_path / "baseline_spy_only_frozen.md").write_text(report, encoding="utf-8")
    return report_destination.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the current SPY-only HMM baseline.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="baselines/spy_only_hmm")
    parser.add_argument("--report", default="reports/baseline_spy_only_frozen.md")
    args = parser.parse_args()

    report = freeze_spy_only_baseline(args.config, args.output_dir, args.report)
    print(f"Frozen baseline report written to: {report}")


if __name__ == "__main__":
    main()

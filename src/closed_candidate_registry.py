from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.hmm_state_interpretability_cross_asset import _markdown_table


DECISION_PATTERNS = [
    "results/*/*/*_decisions.parquet",
    "results/*/*_decisions.parquet",
]

REGISTRY_COLUMNS = [
    "candidate_id",
    "target_symbol",
    "experiment",
    "family",
    "variant",
    "side",
    "horizon_bars",
    "hour_filter_name",
    "fold",
    "decision",
    "closed_status",
    "close_reason",
    "reopen_requires",
    "validation_status",
    "test_net_primary",
    "test_avg_trade_net_primary",
    "test_trades_primary",
    "test_net_stress",
    "test_net_delta_vs_random_primary",
    "test_net_delta_vs_breakout_primary",
    "source_path",
]


def collect_decision_paths(results_root: str | Path = "results") -> list[Path]:
    root = Path(results_root)
    paths: list[Path] = []
    for pattern in DECISION_PATTERNS:
        paths.extend(root.glob(pattern.removeprefix("results/")))
    return sorted({path for path in paths if path.is_file() and path.name.endswith("_decisions.parquet")})


def family_from_path(path: Path) -> str:
    name = path.name.removesuffix("_decisions.parquet")
    return {
        "excess_reversion": "excess_reversion",
        "cross_asset_divergence": "cross_asset_divergence",
        "volatility_expansion": "volatility_expansion",
        "volatility_expansion_robustness": "volatility_expansion",
        "volatility_expansion_holdout": "volatility_expansion",
        "operable_candidate": "operable_candidate",
        "operable_alpha": "operable_alpha",
        "setup_signal": "setup_signal",
    }.get(name, name)


def experiment_from_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def target_from_path(path: Path) -> str:
    parent = path.parent.name
    return parent.upper() if parent else "UNKNOWN"


def _optional_diagnostic_path(path: Path) -> Path | None:
    candidates = sorted(path.parent.glob("*failure_attribution.parquet"))
    return candidates[0] if candidates else None


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out


def _decision_stage(path: Path) -> str:
    name = path.name.removesuffix("_decisions.parquet")
    if name.endswith("_holdout"):
        return "holdout"
    if name.endswith("_robustness"):
        return "robustness"
    return "search"


def infer_closed_status(row: pd.Series | dict[str, Any]) -> str:
    decision = str(row.get("decision", ""))
    reason = str(row.get("diagnostic_reasons", ""))
    primary = _safe_float(row.get("test_net_primary", row.get("primary_net", np.nan)))
    stress = _safe_float(row.get("test_net_stress", row.get("stress_net", np.nan)))
    delta_random = _safe_float(row.get("test_net_delta_vs_random_primary", row.get("net_delta_vs_random", np.nan)))

    if decision in {"holdout_pass", "robustness_candidate"}:
        return "accepted_candidate"
    if decision in {"holdout_provisional", "robustness_provisional"}:
        if np.isfinite(primary) and primary <= 0.0:
            return "rejected_no_edge"
        if np.isfinite(stress) and stress <= 0.0:
            return "rejected_cost_fragile"
        return "accepted_research_only"
    if decision in {"holdout_failed", "robustness_failed"}:
        if np.isfinite(primary) and primary <= 0.0:
            return "rejected_no_edge"
        if np.isfinite(stress) and stress <= 0.0:
            return "rejected_cost_fragile"
        return "rejected_unstable"
    if decision == "accepted_candidate":
        return "accepted_candidate"
    if decision == "research_candidate":
        return "accepted_research_only"
    if decision == "cost_fragile" or "stress_cost_fragility" in reason or (primary > 0.0 and np.isfinite(stress) and stress <= 0.0):
        return "rejected_cost_fragile"
    if decision == "rejected_validation_failed":
        return "rejected_unstable"
    if "random_control_stronger" in reason or (np.isfinite(delta_random) and delta_random <= 0.0 and primary > 0.0):
        return "rejected_unstable"
    if np.isfinite(primary) and primary <= 0.0:
        return "rejected_no_edge"
    if decision.startswith("rejected"):
        return "rejected_no_edge"
    return "accepted_research_only"


def close_reason(row: pd.Series | dict[str, Any]) -> str:
    failed_checks = str(row.get("failed_checks", "") or "").strip()
    if failed_checks:
        return failed_checks
    diagnostic = str(row.get("diagnostic_reasons", "") or "").strip()
    if diagnostic:
        return diagnostic
    validation = str(row.get("validation_status", "") or "").strip()
    decision = str(row.get("decision", "") or "").strip()
    primary = row.get("test_net_primary", np.nan)
    stress = row.get("test_net_stress", np.nan)
    avg = row.get("test_avg_trade_net_primary", np.nan)
    trades = row.get("test_trades_primary", np.nan)
    parts = [part for part in [decision, validation] if part]
    metric_parts = []
    if pd.notna(primary):
        metric_parts.append(f"primary_net={float(primary):.6f}")
    if pd.notna(avg):
        metric_parts.append(f"avg_trade={float(avg):.6f}")
    if pd.notna(trades):
        metric_parts.append(f"trades={int(trades)}")
    if pd.notna(stress):
        metric_parts.append(f"stress_net={float(stress):.6f}")
    return "; ".join([*parts, ", ".join(metric_parts)]).strip("; ")


def reopen_requires(status: str) -> str:
    return {
        "rejected_cost_fragile": "Nueva hipotesis que aumente avg_trade/stress; no basta con reoptimizar thresholds.",
        "rejected_no_edge": "Nueva hipotesis economica, target o timeframe; no repetir misma familia.",
        "rejected_unstable": "Nueva compuerta de estabilidad definida en validation y verificada en test.",
        "accepted_research_only": "Resolver controles/frecuencia/concentracion antes de tratarlo como operable.",
        "accepted_candidate": "Mantener en registro vivo; no es rama cerrada.",
    }.get(status, "Nueva hipotesis explicita y evidencia incremental.")


def _merge_diagnostics(decisions: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or diagnostics.empty or "candidate_id" not in diagnostics.columns:
        return decisions
    cols = [
        column
        for column in [
            "candidate_id",
            "diagnostic_reasons",
            "primary_effective_cost_bps",
            "stress_effective_cost_bps",
            "primary_net",
            "primary_avg_trade",
            "primary_trades",
            "stress_net",
            "net_delta_vs_random",
            "net_delta_vs_breakout",
        ]
        if column in diagnostics.columns
    ]
    return decisions.merge(diagnostics.loc[:, cols], on="candidate_id", how="left", validate="many_to_one")


def _fill_column_from(output: pd.DataFrame, target: str, source: str) -> None:
    if source not in output.columns:
        return
    if target not in output.columns:
        output[target] = output[source]
        return
    output[target] = output[target].where(output[target].notna(), output[source])


def _normalize_stage_columns(output: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == "holdout":
        if "holdout_status" in output.columns:
            output["decision"] = output.get("decision", output["holdout_status"]).fillna(output["holdout_status"])
            output["validation_status"] = "posterior_holdout"
        mappings = {
            "holdout_net_primary": "test_net_primary",
            "holdout_avg_trade_primary": "test_avg_trade_net_primary",
            "holdout_trades_primary": "test_trades_primary",
            "holdout_net_stress": "test_net_stress",
        }
        for source, target in mappings.items():
            _fill_column_from(output, target, source)
    elif stage == "robustness":
        if "robustness_status" in output.columns:
            output["decision"] = output.get("decision", output["robustness_status"]).fillna(output["robustness_status"])
            output["validation_status"] = "frozen_robustness"
        _fill_column_from(output, "test_avg_trade_net_primary", "test_avg_trade_primary")
    return output


def normalize_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_parquet(path)
    diagnostic_path = _optional_diagnostic_path(path)
    if diagnostic_path is not None:
        decisions = _merge_diagnostics(decisions, pd.read_parquet(diagnostic_path))
    if decisions.empty:
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    output = decisions.copy()
    stage = _decision_stage(path)
    output = _normalize_stage_columns(output, stage)
    output["source_path"] = str(path)
    output["target_symbol"] = target_from_path(path)
    output["experiment"] = experiment_from_path(path)
    output["family"] = family_from_path(path)
    for column in ["variant", "side", "horizon_bars", "hour_filter_name", "fold", "validation_status", "decision"]:
        if column not in output.columns:
            output[column] = "all" if column == "hour_filter_name" else np.nan
    output["closed_status"] = output.apply(infer_closed_status, axis=1)
    output["close_reason"] = output.apply(close_reason, axis=1)
    output["reopen_requires"] = output["closed_status"].map(reopen_requires)
    for column in REGISTRY_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    return output.loc[:, REGISTRY_COLUMNS].copy()


def build_registry(paths: list[Path]) -> pd.DataFrame:
    frames = [normalize_decisions(path) for path in paths]
    registry = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REGISTRY_COLUMNS)
    if registry.empty:
        return registry
    stage_priority = registry["source_path"].map(lambda value: {"holdout": 3, "robustness": 2, "search": 1}[_decision_stage(Path(str(value)))])
    registry = registry.assign(_stage_priority=stage_priority)
    registry = registry.sort_values(
        ["target_symbol", "experiment", "family", "candidate_id", "_stage_priority"],
        ascending=[True, True, True, True, False],
        kind="stable",
    )
    registry = registry.drop_duplicates(["candidate_id", "target_symbol", "experiment", "family"], keep="first").drop(columns=["_stage_priority"])
    return registry.sort_values(["target_symbol", "experiment", "family", "closed_status", "candidate_id"], kind="stable").reset_index(drop=True)


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def write_registry_yaml(registry: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in registry.to_dict(orient="records"):
        records.append({key: _yaml_safe(value) for key, value in row.items()})
    path.write_text(yaml.safe_dump({"candidates": records}, sort_keys=False, allow_unicode=False), encoding="utf-8")


def render_negative_log(registry: pd.DataFrame, *, max_rows: int = 200) -> str:
    if registry.empty:
        body = "No closed candidates found."
        status = pd.DataFrame()
        family = pd.DataFrame()
    else:
        closed = registry[registry["closed_status"].ne("accepted_candidate")].copy()
        status = closed["closed_status"].value_counts().rename_axis("closed_status").reset_index(name="rows")
        family = (
            closed.groupby(["target_symbol", "experiment", "family", "closed_status"], as_index=False)
            .agg(candidates=("candidate_id", "nunique"))
            .sort_values(["target_symbol", "experiment", "family", "closed_status"], kind="stable")
        )
        cols = [
            "candidate_id",
            "target_symbol",
            "experiment",
            "family",
            "variant",
            "side",
            "horizon_bars",
            "hour_filter_name",
            "closed_status",
            "close_reason",
            "reopen_requires",
        ]
        body = _markdown_table(closed.loc[:, [column for column in cols if column in closed.columns]].head(max_rows), max_rows=max_rows)
    return f"""# Negative Branches Log

## Status Counts

{_markdown_table(status, max_rows=max_rows)}

## Family Counts

{_markdown_table(family, max_rows=max_rows)}

## Closed Candidates

{body}
"""


def write_reports(registry: pd.DataFrame, reports_dir: Path) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = [reports_dir / "negative_branches_log.md"]
    paths[0].write_text(render_negative_log(registry), encoding="utf-8")
    if registry.empty:
        return paths
    for target, group in registry.groupby("target_symbol", sort=False):
        target_dir = reports_dir / str(target).upper()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "closed_candidates.md"
        path.write_text(render_negative_log(group), encoding="utf-8")
        paths.append(path)
    return paths


def run(results_root: str | Path = "results") -> tuple[Path, Path]:
    paths = collect_decision_paths(results_root)
    registry = build_registry(paths)
    parquet_path = Path("results") / "candidate_registry.parquet"
    yaml_path = Path("experiments") / "candidate_registry.yaml"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(parquet_path, index=False)
    write_registry_yaml(registry, yaml_path)
    write_reports(registry, Path("reports"))
    return yaml_path, parquet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a registry of accepted and closed strategy candidates.")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    yaml_path, parquet_path = run(args.results_root)
    print(f"Candidate registry written to: {yaml_path}")
    print(f"Candidate registry parquet written to: {parquet_path}")


if __name__ == "__main__":
    main()

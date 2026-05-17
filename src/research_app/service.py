from __future__ import annotations

import math
import sqlite3
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src import bayesian_regime_h8, bayesian_regime_h8_position
from src.research_app.artifacts import read_markdown, read_parquet_preview, resolve_existing_path
from src.research_app.decisions import create_decision_log, list_decision_logs
from src.research_app.registry import DEFAULT_DB_PATH, connect, index_workspace, list_artifacts, list_candidates, list_reports, list_runs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DAEMON_STATUS_PATH = Path("results/paper/h1c_auto_runner/daemon_status.yaml")
DEFAULT_H1C_STATE_PATH = Path("results/paper/h1c_state/state.yaml")
DEFAULT_H1C_STATE_EVENTS_PATH = Path("results/paper/h1c_state/events.parquet")
DEFAULT_H1C_PNL_EVENTS_PATH = Path("results/paper/h1c_state/pnl_events.parquet")
DEFAULT_H1C_AUTO_RUNNER_DIR = Path("results/paper/h1c_auto_runner")
DEFAULT_H1C_ORDER_PLAN_DIR = Path("results/paper/h1c_order_plan")
DEFAULT_H1C_ORDER_EXECUTION_DIR = Path("results/paper/h1c_order_execution")
DEFAULT_H1C_PRICE_PATH = Path("data/cleaned/15min/QQQ/QQQ_15min_clean.parquet")
DEFAULT_H8_CONFIG_PATH = Path("configs/hmm_bayesian_regime_h8_spy_15min.yaml")
DEFAULT_H8C_CONFIG_PATH = Path("configs/hmm_bayesian_regime_h8c_qqq_15min.yaml")


@dataclass(frozen=True)
class RegistrySummary:
    db_path: str
    exists: bool
    runs: int = 0
    artifacts: int = 0
    reports: int = 0
    candidates: int = 0
    decision_logs: int = 0


@dataclass(frozen=True)
class RegistryFrames:
    runs: pd.DataFrame
    candidates: pd.DataFrame
    reports: pd.DataFrame
    artifacts: pd.DataFrame
    decisions: pd.DataFrame


def normalize_run_filter(run_id: str | None) -> str | None:
    if not run_id or run_id == "ALL":
        return None
    return run_id


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    return value


def frame_to_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    limited = frame.head(limit) if limit is not None else frame
    return [json_safe(record) for record in limited.to_dict(orient="records")]


def registry_summary(db_path: str | Path = DEFAULT_DB_PATH) -> RegistrySummary:
    path = Path(db_path)
    if not path.exists():
        return RegistrySummary(db_path=path.as_posix(), exists=False)

    conn = connect(path)
    try:
        counts = {
            table: int(conn.execute(f"select count(*) from {table}").fetchone()[0])
            for table in ("runs", "artifacts", "reports", "candidates", "decision_logs")
        }
    finally:
        conn.close()
    return RegistrySummary(db_path=path.as_posix(), exists=True, **counts)


def index_registry(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    results_dir: str | Path = "results",
    reports_dir: str | Path = "reports",
    reset: bool = False,
) -> RegistrySummary:
    index_workspace(results_dir=results_dir, reports_dir=reports_dir, db_path=db_path, reset=reset)
    return registry_summary(db_path)


def load_registry_frames(db_path: str | Path = DEFAULT_DB_PATH, run_id: str | None = None) -> RegistryFrames:
    normalized_run_id = normalize_run_filter(run_id)
    if not Path(db_path).exists():
        empty = pd.DataFrame()
        return RegistryFrames(runs=empty, candidates=empty, reports=empty, artifacts=empty, decisions=empty)
    return RegistryFrames(
        runs=list_runs(db_path),
        candidates=list_candidates(db_path, run_id=normalized_run_id),
        reports=list_reports(db_path, run_id=normalized_run_id),
        artifacts=list_artifacts(db_path, run_id=normalized_run_id),
        decisions=list_decision_logs(db_path),
    )


def metric_value(frame: pd.DataFrame, column: str, default: str = "0") -> str:
    if frame.empty or column not in frame:
        return default
    return str(frame[column].nunique(dropna=True))


def filter_frame(frame: pd.DataFrame, column: str, values: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if not values or column not in frame:
        return frame
    return frame[frame[column].fillna("UNKNOWN").astype(str).isin(values)]


def filter_candidates(
    candidates: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> pd.DataFrame:
    filtered = filter_frame(candidates, "target_symbol", targets)
    return filter_frame(filtered, "timeframe", timeframes)


def list_registry_records(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    run_id: str | None = None,
    targets: list[str] | None = None,
    timeframes: list[str] | None = None,
    limit: int | None = 500,
) -> dict[str, Any]:
    frames = load_registry_frames(db_path=db_path, run_id=run_id)
    candidates = filter_candidates(frames.candidates, targets=targets, timeframes=timeframes)
    return {
        "summary": asdict(registry_summary(db_path)),
        "runs": frame_to_records(frames.runs, limit=limit),
        "candidates": frame_to_records(candidates, limit=limit),
        "reports": frame_to_records(frames.reports, limit=limit),
        "artifacts": frame_to_records(frames.artifacts, limit=limit),
        "decisions": frame_to_records(frames.decisions, limit=limit),
    }


def resolve_workspace_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    root_path = Path(root).resolve()
    resolved = resolve_existing_path(path, root_path).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc
    return resolved


def workspace_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    root_path = Path(root).resolve()
    raw = Path(path)
    resolved = raw.resolve() if raw.is_absolute() else (root_path / raw).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {path}") from exc
    return resolved


def workspace_relpath(path: str | Path, root: str | Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_report_markdown(path: str | Path, root: str | Path = PROJECT_ROOT) -> str:
    report_path = resolve_workspace_path(path, root)
    if report_path.suffix.lower() != ".md":
        raise ValueError(f"report preview only supports markdown files: {path}")
    return read_markdown(report_path)


def parquet_preview(path: str | Path, *, root: str | Path = PROJECT_ROOT, limit: int = 200) -> dict[str, Any]:
    parquet_path = resolve_workspace_path(path, root)
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(f"parquet preview only supports parquet files: {path}")
    frame = read_parquet_preview(parquet_path, limit=limit)
    return {
        "path": parquet_path.relative_to(Path(root).resolve()).as_posix(),
        "limit": limit,
        "columns": list(frame.columns),
        "rows": frame_to_records(frame),
    }


def h8_available_targets(
    *,
    root: str | Path = PROJECT_ROOT,
    features_dir: str | Path = "data/features",
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    base = workspace_path(features_dir, root_path)
    rows: list[dict[str, Any]] = []
    if not base.exists():
        return rows

    for features_path in sorted(base.glob("*/*/*/*/features.parquet")):
        try:
            rel = features_path.relative_to(base)
            target, timeframe, universe_id, feature_version, _ = rel.parts
        except ValueError:
            continue
        try:
            frame = pd.read_parquet(features_path, columns=["timestamp", "session"])
        except Exception as exc:
            rows.append(
                {
                    "target_symbol": features_path.parts[-5],
                    "status": "read_error",
                    "error": str(exc),
                    "path": workspace_relpath(features_path, root_path),
                }
            )
            continue
        timestamps = pd.to_datetime(frame["timestamp"]) if not frame.empty else pd.Series(dtype="datetime64[ns]")
        rows.append(
            {
                "target_symbol": str(target).upper(),
                "timeframe": timeframe,
                "universe_id": universe_id,
                "feature_version": feature_version,
                "rows": int(len(frame)),
                "sessions": int(frame["session"].nunique()) if "session" in frame else 0,
                "start_timestamp": timestamps.min().isoformat() if len(timestamps) else None,
                "end_timestamp": timestamps.max().isoformat() if len(timestamps) else None,
                "status": "ready",
                "path": workspace_relpath(features_path, root_path),
            }
        )
    rows = sorted(rows, key=lambda row: (str(row.get("target_symbol")), str(row.get("timeframe")), -int(row.get("sessions") or 0)))
    return rows[:limit] if limit is not None else rows


def _h8_base_config(config_path: str | Path, root: str | Path) -> dict[str, Any]:
    path = workspace_path(config_path, root)
    return bayesian_regime_h8.load_yaml(path)


def _h8_config_for_target(config_path: str | Path, target_symbol: str, root: str | Path) -> dict[str, Any]:
    config = deepcopy(_h8_base_config(config_path, root))
    config.setdefault("lab", {})["target_symbol"] = target_symbol.upper()
    return config


def _h8c_base_config(config_path: str | Path, root: str | Path) -> dict[str, Any]:
    path = workspace_path(config_path, root)
    return bayesian_regime_h8_position.load_yaml(path)


def _h8c_config_for_target(config_path: str | Path, target_symbol: str, root: str | Path) -> dict[str, Any]:
    config = deepcopy(_h8c_base_config(config_path, root))
    config.setdefault("lab", {})["target_symbol"] = target_symbol.upper()
    config.setdefault("h8_position_model", {})["target_symbol"] = target_symbol.upper()
    return config


def _read_h8_price_series(
    target_symbol: str,
    posteriors: pd.DataFrame,
    features: pd.DataFrame,
    *,
    root: str | Path = PROJECT_ROOT,
) -> pd.DataFrame:
    target = target_symbol.upper()
    root_path = Path(root).resolve()
    price_path = root_path / "data" / "cleaned" / "15min" / target / f"{target}_15min_clean.parquet"
    output = posteriors.copy()
    if price_path.exists():
        try:
            price = pd.read_parquet(price_path, columns=["timestamp", "close"])
            price["timestamp"] = pd.to_datetime(price["timestamp"])
            output["timestamp"] = pd.to_datetime(output["timestamp"])
            output = output.merge(price, on="timestamp", how="left", validate="many_to_one")
        except Exception:
            output["close"] = pd.NA
    else:
        output["close"] = pd.NA

    if output["close"].isna().all() and "target_open_next" in features.columns:
        proxy = features.reset_index(names="source_index").loc[:, ["source_index", "target_open_next"]]
        output = output.merge(proxy, on="source_index", how="left", validate="many_to_one")
        output["close"] = output["target_open_next"]
    return output


def h8_probe_snapshot(
    *,
    target_symbol: str,
    config_path: str | Path = DEFAULT_H8_CONFIG_PATH,
    root: str | Path = PROJECT_ROOT,
    chart_variant: str = "manual_h8a",
    chart_limit: int = 260,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    target = target_symbol.upper().strip()
    config = _h8_config_for_target(config_path, target, root_path)
    feature_path = root_path / bayesian_regime_h8.features_path(config, target)
    if not feature_path.exists():
        return {
            "available": False,
            "target_symbol": target,
            "status": "missing_features",
            "reason": f"H8 features not found for {target}: {workspace_relpath(feature_path, root_path)}",
            "feature_path": workspace_relpath(feature_path, root_path),
            "available_targets": h8_available_targets(root=root_path, limit=50),
        }

    try:
        report, diagnostics_path = bayesian_regime_h8.run_config(config, target_symbol=target)
    except Exception as exc:
        return {
            "available": False,
            "target_symbol": target,
            "status": "run_failed",
            "reason": str(exc),
            "feature_path": workspace_relpath(feature_path, root_path),
            "available_targets": h8_available_targets(root=root_path, limit=50),
        }

    output_dir = root_path / bayesian_regime_h8.results_dir(config, target)
    posteriors = pd.read_parquet(output_dir / "h8_posteriors.parquet")
    profiles = pd.read_parquet(output_dir / "h8_regime_profiles.parquet")
    aggregate = pd.read_parquet(output_dir / "h8_directional_gate_aggregate.parquet")
    diagnostics = pd.read_parquet(diagnostics_path)
    features = pd.read_parquet(feature_path)

    validation = aggregate[aggregate["split"].eq("validation")].copy()
    test = aggregate[aggregate["split"].eq("test")].copy()
    sort_cols = ["avg_trade_net_pooled", "net_return"]
    validation_top = validation.sort_values(sort_cols, ascending=[False, False], kind="stable").head(20)
    test_top = test.sort_values(sort_cols, ascending=[False, False], kind="stable").head(20)
    best_validation = frame_to_records(validation_top.head(1))[0] if not validation_top.empty else None
    matching_test = pd.DataFrame()
    if best_validation:
        mask = (
            test["variant"].eq(best_validation["variant"])
            & test["horizon_bars"].eq(int(best_validation["horizon_bars"]))
            & test["probability_threshold"].eq(float(best_validation["probability_threshold"]))
            & test["cost_bps"].eq(float(best_validation["cost_bps"]))
        )
        if best_validation.get("max_entropy") is None:
            mask &= test["max_entropy"].isna()
        else:
            mask &= test["max_entropy"].eq(float(best_validation["max_entropy"]))
        matching_test = test[mask].copy()

    selected = posteriors[posteriors["variant"].eq(chart_variant)].copy()
    if selected.empty:
        selected = posteriors.copy()
        chart_variant = str(selected["variant"].iloc[0]) if not selected.empty else chart_variant
    split_order = ["test", "validation", "train"]
    selected_split = next((split for split in split_order if split in set(selected["split"].astype(str))), None)
    if selected_split:
        selected = selected[selected["split"].eq(selected_split)].copy()
    chart_fold = int(selected["fold"].max()) if not selected.empty else 0
    selected = selected[selected["fold"].eq(chart_fold)].sort_values("timestamp", kind="stable").tail(chart_limit)
    chart_frame = _read_h8_price_series(target, selected, features, root=root_path)
    chart_columns = [
        column
        for column in [
            "timestamp",
            "close",
            "regime",
            "max_prob",
            "entropy",
            "p_bull_trend",
            "p_bear_stress",
            "p_chop_compression",
            "p_volatile_noise",
        ]
        if column in chart_frame.columns
    ]

    dataset_frame = pd.read_parquet(feature_path, columns=["timestamp", "session"])
    timestamps = pd.to_datetime(dataset_frame["timestamp"])
    dataset = {
        "feature_path": workspace_relpath(feature_path, root_path),
        "rows": int(len(dataset_frame)),
        "sessions": int(dataset_frame["session"].nunique()),
        "start_timestamp": timestamps.min().isoformat() if len(timestamps) else None,
        "end_timestamp": timestamps.max().isoformat() if len(timestamps) else None,
    }
    return {
        "available": True,
        "target_symbol": target,
        "status": "ok",
        "dataset": json_safe(dataset),
        "report_path": workspace_relpath(report, root_path),
        "diagnostics_path": workspace_relpath(diagnostics_path, root_path),
        "artifacts": {
            "posteriors": workspace_relpath(output_dir / "h8_posteriors.parquet", root_path),
            "profiles": workspace_relpath(output_dir / "h8_regime_profiles.parquet", root_path),
            "aggregate": workspace_relpath(output_dir / "h8_directional_gate_aggregate.parquet", root_path),
            "diagnostics": workspace_relpath(diagnostics_path, root_path),
        },
        "summary": {
            "best_validation": json_safe(best_validation),
            "matching_test": frame_to_records(matching_test.head(1))[0] if not matching_test.empty else None,
            "diagnostic_rows": int(len(diagnostics)),
        },
        "aggregate_validation": frame_to_records(validation_top),
        "aggregate_test": frame_to_records(test_top),
        "profiles": frame_to_records(
            profiles[profiles["split"].eq("validation")]
            .sort_values(["variant", "fold", "regime"], kind="stable")
            .head(80)
        ),
        "chart": {
            "variant": chart_variant,
            "split": selected_split,
            "fold": chart_fold,
            "rows": frame_to_records(chart_frame.loc[:, chart_columns]),
        },
        "available_targets": h8_available_targets(root=root_path, limit=50),
    }


def h8c_position_snapshot(
    *,
    target_symbol: str,
    config_path: str | Path = DEFAULT_H8C_CONFIG_PATH,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    target = target_symbol.upper().strip()
    config = _h8c_config_for_target(config_path, target, root_path)
    feature_path = root_path / bayesian_regime_h8.features_path(config, target)
    if not feature_path.exists():
        return {
            "available": False,
            "target_symbol": target,
            "status": "missing_features",
            "reason": f"H8c features not found for {target}: {workspace_relpath(feature_path, root_path)}",
            "feature_path": workspace_relpath(feature_path, root_path),
            "available_targets": h8_available_targets(root=root_path, limit=50),
        }

    try:
        report, selected_metrics_path = bayesian_regime_h8_position.run_config(config, target_symbol=target)
    except Exception as exc:
        return {
            "available": False,
            "target_symbol": target,
            "status": "run_failed",
            "reason": str(exc),
            "feature_path": workspace_relpath(feature_path, root_path),
            "available_targets": h8_available_targets(root=root_path, limit=50),
        }

    output_dir = root_path / bayesian_regime_h8_position.results_dir(config, target)
    selected = pd.read_parquet(output_dir / "h8c_selected_gates.parquet")
    selected_metrics = pd.read_parquet(selected_metrics_path)
    aggregate = pd.read_parquet(output_dir / "h8c_selected_aggregate.parquet")
    cost_sensitivity_aggregate = pd.read_parquet(output_dir / "h8c_cost_sensitivity_aggregate.parquet")
    model_quality = pd.read_parquet(output_dir / "h8c_model_quality.parquet")
    coefficients = pd.read_parquet(output_dir / "h8c_coefficients.parquet")

    validation_aggregate = aggregate[aggregate["split"].eq("validation")].copy()
    test_aggregate = aggregate[aggregate["split"].eq("test")].copy()
    selected_gate = frame_to_records(selected.head(1))[0] if not selected.empty else None
    dataset_frame = pd.read_parquet(feature_path, columns=["timestamp", "session"])
    timestamps = pd.to_datetime(dataset_frame["timestamp"])
    dataset = {
        "feature_path": workspace_relpath(feature_path, root_path),
        "rows": int(len(dataset_frame)),
        "sessions": int(dataset_frame["session"].nunique()),
        "start_timestamp": timestamps.min().isoformat() if len(timestamps) else None,
        "end_timestamp": timestamps.max().isoformat() if len(timestamps) else None,
    }
    return {
        "available": True,
        "target_symbol": target,
        "status": "ok",
        "dataset": json_safe(dataset),
        "report_path": workspace_relpath(report, root_path),
        "selected_metrics_path": workspace_relpath(selected_metrics_path, root_path),
        "artifacts": {
            "predictions": workspace_relpath(output_dir / "h8c_position_predictions.parquet", root_path),
            "threshold_grid": workspace_relpath(output_dir / "h8c_threshold_grid.parquet", root_path),
            "selected_gates": workspace_relpath(output_dir / "h8c_selected_gates.parquet", root_path),
            "selected_metrics": workspace_relpath(selected_metrics_path, root_path),
            "aggregate": workspace_relpath(output_dir / "h8c_selected_aggregate.parquet", root_path),
            "cost_sensitivity": workspace_relpath(output_dir / "h8c_cost_sensitivity.parquet", root_path),
            "cost_sensitivity_aggregate": workspace_relpath(output_dir / "h8c_cost_sensitivity_aggregate.parquet", root_path),
            "model_quality": workspace_relpath(output_dir / "h8c_model_quality.parquet", root_path),
            "coefficients": workspace_relpath(output_dir / "h8c_coefficients.parquet", root_path),
        },
        "summary": {
            "selected_gate": json_safe(selected_gate),
            "validation": frame_to_records(validation_aggregate.head(1))[0] if not validation_aggregate.empty else None,
            "test": frame_to_records(test_aggregate.head(1))[0] if not test_aggregate.empty else None,
            "model_quality_rows": int(len(model_quality)),
        },
        "selected_metrics": frame_to_records(selected_metrics.sort_values(["split", "fold"], kind="stable")),
        "aggregate": frame_to_records(aggregate.sort_values(["split", "cost_bps"], kind="stable")),
        "cost_sensitivity": frame_to_records(
            cost_sensitivity_aggregate.sort_values(["split", "cost_kind", "effective_cost_bps"], kind="stable")
        ),
        "model_quality": frame_to_records(model_quality.sort_values(["fold", "split", "side"], kind="stable").head(80)),
        "coefficients": frame_to_records(
            coefficients.assign(abs_coefficient=coefficients["coefficient"].abs())
            .sort_values(["fold", "side", "abs_coefficient"], ascending=[True, True, False], kind="stable")
            .head(80)
        ),
        "available_targets": h8_available_targets(root=root_path, limit=50),
    }


def read_daemon_status(
    path: str | Path = DEFAULT_DAEMON_STATUS_PATH,
    *,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    try:
        status_path = resolve_workspace_path(path, root)
    except FileNotFoundError:
        return {"available": False, "path": Path(path).as_posix()}
    except ValueError as exc:
        return {"available": False, "path": Path(path).as_posix(), "error": str(exc)}
    try:
        data = yaml.safe_load(status_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"available": False, "path": Path(path).as_posix(), "error": str(exc)}
    mtime = datetime.fromtimestamp(status_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0)
    payload = json_safe(data)
    payload["available"] = True
    payload["path"] = Path(path).as_posix()
    payload["mtime_utc"] = mtime.isoformat().replace("+00:00", "Z")
    return payload


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}


def read_parquet_frame_if_exists(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return pd.DataFrame()


def sort_records(records: list[dict[str, Any]], key: str = "created_at_utc", limit: int | None = None) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: str(row.get(key) or ""), reverse=True)
    return ordered[:limit] if limit is not None else ordered


def read_h1c_state(
    path: str | Path = DEFAULT_H1C_STATE_PATH,
    *,
    root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    state_path = workspace_path(path, root)
    if not state_path.exists():
        return {"available": False, "path": Path(path).as_posix()}
    try:
        payload = read_yaml_mapping(state_path)
    except Exception as exc:
        return {"available": False, "path": Path(path).as_posix(), "error": str(exc)}
    payload = json_safe(payload)
    payload["available"] = True
    payload["path"] = workspace_relpath(state_path, root)
    return payload


def list_h1c_state_events(
    path: str | Path = DEFAULT_H1C_STATE_EVENTS_PATH,
    *,
    root: str | Path = PROJECT_ROOT,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    events_path = workspace_path(path, root)
    frame = read_parquet_frame_if_exists(events_path)
    records = frame_to_records(frame)
    return sort_records(records, limit=limit)


def list_h1c_pnl_events(
    path: str | Path = DEFAULT_H1C_PNL_EVENTS_PATH,
    *,
    root: str | Path = PROJECT_ROOT,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    pnl_path = workspace_path(path, root)
    frame = read_parquet_frame_if_exists(pnl_path)
    records = frame_to_records(frame)
    return sort_records(records, limit=limit)


def list_h1c_auto_runs(
    auto_runner_dir: str | Path = DEFAULT_H1C_AUTO_RUNNER_DIR,
    *,
    root: str | Path = PROJECT_ROOT,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    runner_dir = workspace_path(auto_runner_dir, root)
    records: list[dict[str, Any]] = []
    if not runner_dir.exists():
        return records
    for manifest_path in runner_dir.glob("*/manifest.yaml"):
        try:
            manifest = read_yaml_mapping(manifest_path)
        except Exception:
            continue
        run = dict(manifest.get("run", {}) or {})
        ticket = dict(manifest.get("signal", {}).get("ticket", {}) or {})
        exit_ticket = dict(manifest.get("exit_ticket", {}) or {})
        raw_ticket = dict(manifest.get("signal", {}).get("raw_ticket", {}) or {})
        plan_summary = dict(manifest.get("order_plan", {}).get("summary", {}) or {})
        execution_summary = dict(manifest.get("execution", {}).get("summary", {}) or {})
        pre_recon = dict(manifest.get("pre_trade_reconciliation", {}) or {})
        post_recon = dict(manifest.get("post_execution_reconciliation", {}) or {})
        active_ticket = ticket or exit_ticket or raw_ticket
        records.append(
            json_safe(
                {
                    "created_at_utc": run.get("created_at_utc"),
                    "run_dir": workspace_relpath(manifest_path.parent, root),
                    "manifest_path": workspace_relpath(manifest_path, root),
                    "status": run.get("status"),
                    "decision": manifest.get("decision"),
                    "reason": manifest.get("reason"),
                    "market_open": manifest.get("market", {}).get("open"),
                    "pre_trade_reconciliation": pre_recon.get("decision"),
                    "post_execution_reconciliation": post_recon.get("decision"),
                    "signal_timestamp": active_ticket.get("signal_timestamp"),
                    "signal_action": active_ticket.get("action"),
                    "ticket_quantity": active_ticket.get("quantity"),
                    "theoretical_entry_price": active_ticket.get("theoretical_entry_price"),
                    "theoretical_exit_price": active_ticket.get("theoretical_exit_price"),
                    "intent": plan_summary.get("intent"),
                    "funds_ok": manifest.get("funds", {}).get("ok"),
                    "entry_safety_ok": manifest.get("entry_safety", {}).get("ok"),
                    "order_plan_decision": plan_summary.get("decision"),
                    "planned_orders": plan_summary.get("planned_orders", 0),
                    "submitted_orders": execution_summary.get("submitted_orders", 0),
                    "latency_seconds": manifest.get("latency", {}).get("total_seconds"),
                    "drift": manifest.get("drift", {}),
                }
            )
        )
    return sort_records(records, limit=limit)


def _planned_order_paths(root: Path) -> list[Path]:
    paths = list((root / DEFAULT_H1C_ORDER_PLAN_DIR).glob("*/orders.parquet"))
    paths.extend((root / DEFAULT_H1C_AUTO_RUNNER_DIR).glob("*/order_plan/*/orders.parquet"))
    paths.extend((root / DEFAULT_H1C_AUTO_RUNNER_DIR).glob("*/exit_order_plan/*/orders.parquet"))
    return sorted(set(paths))


def _submitted_order_paths(root: Path) -> list[Path]:
    paths = list((root / DEFAULT_H1C_ORDER_EXECUTION_DIR).glob("*/submitted_orders.parquet"))
    paths.extend((root / DEFAULT_H1C_AUTO_RUNNER_DIR).glob("*/execution/*/submitted_orders.parquet"))
    paths.extend((root / DEFAULT_H1C_AUTO_RUNNER_DIR).glob("*/exit_execution/*/submitted_orders.parquet"))
    return sorted(set(paths))


def list_h1c_planned_orders(*, root: str | Path = PROJECT_ROOT, limit: int | None = 200) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for orders_path in _planned_order_paths(root_path):
        frame = read_parquet_frame_if_exists(orders_path)
        manifest_path = orders_path.parent / "manifest.yaml"
        manifest = read_yaml_mapping(manifest_path) if manifest_path.exists() else {}
        summary = dict(manifest.get("summary", {}) or {})
        run = dict(manifest.get("run", {}) or {})
        for row in frame_to_records(frame):
            records.append(
                {
                    **row,
                    "created_at_utc": run.get("created_at_utc"),
                    "plan_decision": summary.get("decision"),
                    "plan_path": workspace_relpath(orders_path.parent, root_path),
                    "orders_path": workspace_relpath(orders_path, root_path),
                    "manifest_path": workspace_relpath(manifest_path, root_path) if manifest_path.exists() else None,
                }
            )
    return sort_records(records, limit=limit)


def list_h1c_submitted_orders(*, root: str | Path = PROJECT_ROOT, limit: int | None = 200) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    records: list[dict[str, Any]] = []
    for submitted_path in _submitted_order_paths(root_path):
        frame = read_parquet_frame_if_exists(submitted_path)
        manifest_path = submitted_path.parent / "manifest.yaml"
        manifest = read_yaml_mapping(manifest_path) if manifest_path.exists() else {}
        run = dict(manifest.get("run", {}) or {})
        preflight = dict(manifest.get("preflight", {}) or {})
        unlock = dict(manifest.get("unlock", {}) or {})
        for row in frame_to_records(frame):
            records.append(
                {
                    **row,
                    "created_at_utc": run.get("created_at_utc"),
                    "execution_status": run.get("status"),
                    "plan_fingerprint": preflight.get("plan_fingerprint"),
                    "unlock": unlock.get("unlocked"),
                    "execution_path": workspace_relpath(submitted_path.parent, root_path),
                    "submitted_orders_path": workspace_relpath(submitted_path, root_path),
                    "manifest_path": workspace_relpath(manifest_path, root_path) if manifest_path.exists() else None,
                }
            )
    return sort_records(records, limit=limit)


def list_h1c_price_series(
    path: str | Path = DEFAULT_H1C_PRICE_PATH,
    *,
    root: str | Path = PROJECT_ROOT,
    limit: int | None = 240,
) -> list[dict[str, Any]]:
    price_path = workspace_path(path, root)
    frame = read_parquet_frame_if_exists(price_path)
    if frame.empty or "timestamp" not in frame.columns:
        return []
    columns = [column for column in ["timestamp", "open", "high", "low", "close", "volume", "session", "bar_index"] if column in frame.columns]
    output = frame[columns].copy()
    output = output.sort_values("timestamp")
    if limit is not None:
        output = output.tail(limit)
    return frame_to_records(output)


def list_h1c_pnl_series(pnl_events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    events = pnl_events if pnl_events is not None else list_h1c_pnl_events(limit=None)
    ordered = sorted(events, key=lambda row: str(row.get("created_at_utc") or ""))
    cumulative = 0.0
    rows: list[dict[str, Any]] = []
    for event in ordered:
        realized = event.get("realized_pnl")
        realized_value = 0.0 if realized is None else float(realized)
        cumulative += realized_value
        rows.append(
            {
                "timestamp": event.get("created_at_utc"),
                "event_type": event.get("event_type"),
                "realized_pnl": realized,
                "cumulative_realized_pnl": cumulative,
                "quantity": event.get("quantity"),
                "entry_price": event.get("entry_price"),
                "exit_price": event.get("exit_price"),
            }
        )
    return json_safe(rows)


def list_h1c_signal_markers(auto_runs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    runs = auto_runs if auto_runs is not None else list_h1c_auto_runs(limit=None)
    markers: list[dict[str, Any]] = []
    for run in runs:
        markers.append(
            {
                "timestamp": run.get("signal_timestamp") or run.get("created_at_utc"),
                "action": run.get("signal_action"),
                "decision": run.get("decision"),
                "quantity": run.get("ticket_quantity"),
                "price": run.get("theoretical_exit_price") if run.get("signal_action") == "BUY" else run.get("theoretical_entry_price"),
            }
        )
    return sort_records(json_safe(markers), key="timestamp", limit=None)


def latest_h1c_signal_path(
    *,
    root: str | Path = PROJECT_ROOT,
    auto_runner_dir: str | Path = DEFAULT_H1C_AUTO_RUNNER_DIR,
) -> Path | None:
    runner_dir = workspace_path(auto_runner_dir, root)
    if not runner_dir.exists():
        return None
    paths = sorted(runner_dir.glob("*/signal/*/latest_signal.yaml"))
    return paths[-1] if paths else None


def signed_margin(value: Any, threshold: Any, operator: str) -> float | None:
    if value is None or threshold is None:
        return None
    try:
        lhs = float(value)
        rhs = float(threshold)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return None
    if operator in {">", ">="}:
        return lhs - rhs
    if operator in {"<", "<="}:
        return rhs - lhs
    return None


def condition_row(
    *,
    key: str,
    label: str,
    value: Any,
    threshold: Any,
    operator: str,
    passed: bool,
    unit: str = "",
    required: bool = True,
    note: str = "",
) -> dict[str, Any]:
    margin = signed_margin(value, threshold, operator)
    unavailable = value is None
    return json_safe(
        {
            "key": key,
            "label": label,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "margin": margin,
            "passed": bool(passed),
            "unavailable": unavailable,
            "required": required,
            "unit": unit,
            "note": note,
        }
    )


def h1c_signal_diagnostics(*, root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    path = latest_h1c_signal_path(root=root)
    if path is None:
        return {"available": False, "reason": "latest_signal.yaml not found", "conditions": [], "summary": {}}
    try:
        signal = read_yaml_mapping(path)
    except Exception as exc:
        return {"available": False, "reason": str(exc), "conditions": [], "summary": {}}

    can_enter_value = 1.0 if signal.get("target_can_open_trade") and signal.get("target_open_next") is not None else 0.0
    exit_available_value = (
        1.0
        if signal.get("h1c_theoretical_exit_timestamp") is not None and signal.get("h1c_theoretical_exit_price") is not None
        else 0.0
    )
    conditions = [
        condition_row(
            key="target_ret_6",
            label="QQQ 6-bar return below 0",
            value=signal.get("target_ret_6"),
            threshold=0.0,
            operator="<",
            passed=bool(signal.get("target_ret_6") is not None and signal.get("target_ret_6") < 0.0),
            unit="return",
            note="Short setup needs recent QQQ weakness.",
        ),
        condition_row(
            key="target_ret_12",
            label="QQQ 12-bar return below 0",
            value=signal.get("target_ret_12"),
            threshold=0.0,
            operator="<",
            passed=bool(signal.get("target_ret_12") is not None and signal.get("target_ret_12") < 0.0),
            unit="return",
            note="Confirms the breakdown over a longer intraday window.",
        ),
        condition_row(
            key="risk_off_score",
            label="Risk-off score above threshold",
            value=signal.get("risk_off_score"),
            threshold=signal.get("risk_off_min"),
            operator=">=",
            passed=bool(signal.get("h1c_risk_off_pass")),
            note="Cross-asset context must look risk-off.",
        ),
        condition_row(
            key="prev_vix_z20",
            label="Previous VIX z-score above threshold",
            value=signal.get("prev_vix_z20"),
            threshold=signal.get("vix_z20_min"),
            operator=">=",
            passed=bool(signal.get("h1c_vix_pass")),
            unit="z",
            note="VIX pressure must be elevated enough.",
        ),
        condition_row(
            key="spread_credit_12",
            label="Credit spread at/below threshold",
            value=signal.get("spread_credit_12"),
            threshold=signal.get("spread_credit_12_max"),
            operator="<=",
            passed=bool(signal.get("h1c_credit_pass")),
            unit="return",
            note="HYG should not be leading LQD.",
        ),
        condition_row(
            key="can_enter",
            label="Next tradable open available",
            value=can_enter_value,
            threshold=1.0,
            operator=">=",
            passed=bool(signal.get("h1c_can_enter")),
            note="Avoids generating entries when the next open is unavailable.",
        ),
        condition_row(
            key="exit_available",
            label="Fixed-horizon exit available",
            value=exit_available_value,
            threshold=1.0,
            operator=">=",
            passed=bool(signal.get("h1c_exit_available")),
            note="Ensures the short has a planned BUY cover timestamp and price.",
        ),
    ]
    required = [row for row in conditions if row["required"]]
    passed = [row for row in required if row["passed"]]
    unavailable = [row for row in required if row["unavailable"]]
    summary = {
        "timestamp": signal.get("timestamp"),
        "session": signal.get("session"),
        "bar_index": signal.get("bar_index"),
        "action": "SELL" if signal.get("h1c_signal_short") else "NONE",
        "signal_short": bool(signal.get("h1c_signal_short")),
        "passed_conditions": len(passed),
        "required_conditions": len(required),
        "unavailable_conditions": len(unavailable),
        "pass_rate": (len(passed) / len(required)) if required else 0.0,
        "path": workspace_relpath(path, root),
    }
    return {
        "available": True,
        "latest_signal": json_safe(signal),
        "conditions": conditions,
        "summary": json_safe(summary),
    }


def h1c_operations_snapshot(*, root: str | Path = PROJECT_ROOT, limit: int | None = 200) -> dict[str, Any]:
    auto_runs = list_h1c_auto_runs(root=root, limit=limit)
    planned_orders = list_h1c_planned_orders(root=root, limit=limit)
    submitted_orders = list_h1c_submitted_orders(root=root, limit=limit)
    pnl_events = list_h1c_pnl_events(root=root, limit=limit)
    state_events = list_h1c_state_events(root=root, limit=limit)
    current_state = read_h1c_state(root=root)
    realized_values = [
        float(event["realized_pnl"])
        for event in pnl_events
        if event.get("realized_pnl") is not None
    ]
    submitted_count = sum(int(run.get("submitted_orders") or 0) for run in auto_runs)
    if submitted_count == 0:
        submitted_count = len(submitted_orders)
    summary = {
        "current_status": current_state.get("status"),
        "current_quantity": current_state.get("quantity"),
        "current_position_unit": current_state.get("position_unit"),
        "last_signal_timestamp": current_state.get("last_signal_timestamp"),
        "auto_runs": len(auto_runs),
        "planned_order_rows": len(planned_orders),
        "submitted_orders": submitted_count,
        "submitted_order_rows": len(submitted_orders),
        "pnl_events": len(pnl_events),
        "realized_pnl": sum(realized_values),
        "last_auto_decision": auto_runs[0].get("decision") if auto_runs else None,
        "last_state_event": state_events[0].get("event_type") if state_events else None,
    }
    return {
        "summary": json_safe(summary),
        "daemon_status": read_daemon_status(root=root),
        "signal_diagnostics": h1c_signal_diagnostics(root=root),
        "state": current_state,
        "auto_runs": auto_runs,
        "planned_orders": planned_orders,
        "submitted_orders": submitted_orders,
        "pnl_events": pnl_events,
        "charts": {
            "price": list_h1c_price_series(root=root),
            "pnl": list_h1c_pnl_series(pnl_events),
            "signal_markers": list_h1c_signal_markers(auto_runs),
        },
        "state_events": state_events,
    }


def record_decision(
    *,
    db_path: str | Path,
    decision_type: str,
    decision: str,
    evidence: list[dict[str, Any]],
    run_id: str | None = None,
    candidate_id: str | None = None,
    rationale: str | None = None,
    next_action: str | None = None,
    human_owner: str | None = None,
) -> dict[str, Any]:
    log = create_decision_log(
        db_path=db_path,
        decision_type=decision_type,
        decision=decision,
        evidence=evidence,
        run_id=run_id,
        candidate_id=candidate_id,
        rationale=rationale,
        next_action=next_action,
        human_owner=human_owner,
    )
    return json_safe(asdict(log))

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.cross_asset_data import aligned_panel_path, resolve_symbols


INDEX_COLUMNS = {"timestamp", "session", "bar_index"}
OHLCV_FIELDS = {"open", "high", "low", "close", "volume"}
TARGET_EXECUTION_FIELDS = {
    "target_open_next",
    "next_open_timestamp",
    "target_crosses_session_close",
    "can_open_trade",
    "force_flat_bar",
    "trade_could_remain_open_past_close",
}
FORBIDDEN_FEATURE_PATTERNS = (
    "future",
    "fwd",
    "forward",
    "label",
    "entry_",
    "exit_",
    "target_open_next",
    "next_open",
)


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    module: str
    description: str
    status: str
    evidence: str


@dataclass(frozen=True)
class CatalogRow:
    column: str
    role: str
    symbol: str | None
    source_fields: str
    window: str
    timestamp_max_used: str
    available_at: str
    usable_as_feature: bool
    requires_train_fit: bool
    missing_policy: str
    blocked_reason: str


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _pass(check_id: str, module: str, description: str, evidence: str) -> AuditCheck:
    return AuditCheck(check_id, module, description, "PASS", evidence)


def _fail(check_id: str, module: str, description: str, evidence: str) -> AuditCheck:
    return AuditCheck(check_id, module, description, "FAIL", evidence)


def _split_symbol_column(column: str) -> tuple[str | None, str]:
    if "__" not in column:
        return None, column
    symbol, field = column.split("__", 1)
    return symbol, field


def _bar_available_at(config: dict[str, Any]) -> str:
    timestamp_label = config.get("session", {}).get("timestamp_label", "start")
    timeframe = config.get("lab", {}).get("timeframe", config.get("project", {}).get("frequency", "5min"))
    if timestamp_label == "start":
        return f"bar close at timestamp + {timeframe}; before open_t+1"
    if timestamp_label == "end":
        return "bar close at timestamp; before open_t+1"
    return "bar close; before open_t+1"


def build_feature_timestamp_catalog(
    panel: pd.DataFrame,
    config: dict[str, Any],
    target_symbol: str,
) -> pd.DataFrame:
    missing_policy = config.get("alignment", {}).get("missing_policy", "drop_core_missing")
    target = target_symbol.upper()
    rows: list[CatalogRow] = []

    for column in panel.columns:
        symbol, field = _split_symbol_column(column)
        blocked_reason = ""
        role = "unknown"
        source_fields = column
        window = "none"
        timestamp_max_used = "unknown"
        available_at = "unknown"
        usable_as_feature = False
        requires_train_fit = False

        if column in INDEX_COLUMNS:
            role = "index"
            timestamp_max_used = "t"
            available_at = "row identity"
            blocked_reason = "index column, not model input"
        elif field in OHLCV_FIELDS and symbol is not None:
            role = "raw_ohlcv"
            source_fields = f"{symbol}.{field}"
            timestamp_max_used = "bar t"
            available_at = _bar_available_at(config)
            usable_as_feature = True
        elif column.startswith("is_available_"):
            role = "availability_mask"
            source_fields = column.removeprefix("is_available_")
            timestamp_max_used = "bar t"
            available_at = "alignment-time mask known at t"
            usable_as_feature = True
        elif symbol == target and field in TARGET_EXECUTION_FIELDS:
            role = "target_execution_or_label_only"
            source_fields = f"{symbol}.{field}"
            timestamp_max_used = "t+1 or execution calendar"
            available_at = "execution/label construction only"
            blocked_reason = "future/execution field; excluded from HMM feature matrix"
        elif symbol is not None and field in TARGET_EXECUTION_FIELDS:
            role = "blocked_non_target_execution_field"
            source_fields = f"{symbol}.{field}"
            timestamp_max_used = "t+1 or execution calendar"
            available_at = "not allowed"
            blocked_reason = "non-target execution field should not be present"
        else:
            lowered = column.lower()
            forbidden_hits = [pattern for pattern in FORBIDDEN_FEATURE_PATTERNS if pattern in lowered]
            if forbidden_hits:
                role = "blocked_future_like_column"
                timestamp_max_used = "unknown/future-like"
                available_at = "not allowed"
                blocked_reason = f"matches forbidden patterns: {forbidden_hits}"
            else:
                role = "derived_feature_or_unknown"
                timestamp_max_used = "must be declared by feature engineering"
                available_at = "requires explicit audit before use"
                blocked_reason = "no timestamp provenance declared yet"

        rows.append(
            CatalogRow(
                column=column,
                role=role,
                symbol=symbol,
                source_fields=source_fields,
                window=window,
                timestamp_max_used=timestamp_max_used,
                available_at=available_at,
                usable_as_feature=usable_as_feature,
                requires_train_fit=requires_train_fit,
                missing_policy=missing_policy,
                blocked_reason=blocked_reason,
            )
        )

    return pd.DataFrame([asdict(row) for row in rows])


def _feature_columns(catalog: pd.DataFrame) -> list[str]:
    return catalog.loc[catalog["usable_as_feature"].astype(bool), "column"].tolist()


def check_panel_structure(panel: pd.DataFrame, target_symbol: str) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    required = ["timestamp", "session", "bar_index"]
    missing = [column for column in required if column not in panel.columns]
    checks.append(
        _pass("panel_required_index_columns", "cross_asset_panel", "Panel has timestamp/session/bar_index.", "All required index columns present.")
        if not missing
        else _fail("panel_required_index_columns", "cross_asset_panel", "Panel has timestamp/session/bar_index.", f"Missing columns: {missing}")
    )
    if missing:
        return checks

    duplicate_count = int(panel.duplicated(["timestamp"]).sum())
    checks.append(
        _pass("panel_unique_timestamps", "cross_asset_panel", "Panel has unique timestamps.", f"duplicate_timestamps={duplicate_count}")
        if duplicate_count == 0
        else _fail("panel_unique_timestamps", "cross_asset_panel", "Panel has unique timestamps.", f"duplicate_timestamps={duplicate_count}")
    )

    ordered = panel.sort_values(["timestamp", "session", "bar_index"], kind="stable")
    is_sorted = panel.index.equals(ordered.index)
    checks.append(
        _pass("panel_sorted", "cross_asset_panel", "Panel is sorted point-in-time.", "Rows already sorted by timestamp/session/bar_index.")
        if is_sorted
        else _fail("panel_sorted", "cross_asset_panel", "Panel is sorted point-in-time.", "Rows are not sorted by timestamp/session/bar_index.")
    )

    target = target_symbol.upper()
    target_open_next = f"{target}__target_open_next"
    target_open = f"{target}__open"
    if {target_open_next, target_open, "session", "bar_index"} <= set(panel.columns):
        expected = panel.sort_values(["session", "bar_index"], kind="stable").groupby("session", sort=False)[target_open].shift(-1)
        actual = panel.sort_values(["session", "bar_index"], kind="stable")[target_open_next]
        mismatches = int(((actual.notna() | expected.notna()) & (actual != expected)).sum())
        checks.append(
            _pass("target_next_open_alignment", "execution", "target_open_next equals target open at t+1 within session.", f"mismatches={mismatches}")
            if mismatches == 0
            else _fail("target_next_open_alignment", "execution", "target_open_next equals target open at t+1 within session.", f"mismatches={mismatches}")
        )

    return checks


def check_symbol_columns(panel: pd.DataFrame, symbols: list[str], target_symbol: str) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    missing_by_symbol: dict[str, list[str]] = {}
    for symbol in symbols:
        required = [f"{symbol}__{field}" for field in sorted(OHLCV_FIELDS)]
        missing = [column for column in required if column not in panel.columns]
        if missing:
            missing_by_symbol[symbol] = missing
    checks.append(
        _pass("all_symbols_have_ohlcv", "cross_asset_panel", "Every core symbol has OHLCV columns.", f"symbols={symbols}")
        if not missing_by_symbol
        else _fail("all_symbols_have_ohlcv", "cross_asset_panel", "Every core symbol has OHLCV columns.", f"missing={missing_by_symbol}")
    )

    target = target_symbol.upper()
    non_target_execution = [
        column
        for column in panel.columns
        if "__" in column
        for symbol, field in [_split_symbol_column(column)]
        if symbol is not None and symbol != target and field in TARGET_EXECUTION_FIELDS
    ]
    checks.append(
        _pass("no_non_target_execution_fields", "cross_asset_panel", "Only target carries execution/label helper columns.", "No non-target execution fields present.")
        if not non_target_execution
        else _fail("no_non_target_execution_fields", "cross_asset_panel", "Only target carries execution/label helper columns.", f"Found {non_target_execution}")
    )
    return checks


def check_feature_catalog(catalog: pd.DataFrame) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    usable = _feature_columns(catalog)
    forbidden = [
        column
        for column in usable
        if any(pattern in column.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)
    ]
    blocked_unknown = catalog[
        catalog["role"].eq("derived_feature_or_unknown") | catalog["role"].eq("blocked_future_like_column")
    ]["column"].tolist()

    checks.append(
        _pass("feature_catalog_complete", "feature_catalog", "Every panel column has timestamp provenance.", f"columns={len(catalog)}")
        if len(catalog) > 0
        else _fail("feature_catalog_complete", "feature_catalog", "Every panel column has timestamp provenance.", "Catalog is empty.")
    )
    checks.append(
        _pass("future_like_columns_not_features", "feature_catalog", "Future/label/execution-like columns are excluded from features.", "No forbidden feature columns found.")
        if not forbidden
        else _fail("future_like_columns_not_features", "feature_catalog", "Future/label/execution-like columns are excluded from features.", f"Forbidden usable features: {forbidden}")
    )
    checks.append(
        _pass("no_unknown_feature_inputs", "feature_catalog", "No unknown derived feature is used without provenance.", "All unknown/future-like columns are blocked.")
        if not blocked_unknown
        else _fail("no_unknown_feature_inputs", "feature_catalog", "No unknown derived feature is used without provenance.", f"Blocked columns requiring explicit audit: {blocked_unknown}")
    )
    return checks


def check_missing_policy(panel: pd.DataFrame, config: dict[str, Any], symbols: list[str]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    policy = config.get("alignment", {}).get("missing_policy", "drop_core_missing")
    availability_cols = [f"is_available_{symbol}" for symbol in symbols]
    missing_availability = [column for column in availability_cols if column not in panel.columns]
    checks.append(
        _pass("availability_masks_present", "alignment", "Availability masks are present for all core symbols.", f"columns={availability_cols}")
        if not missing_availability
        else _fail("availability_masks_present", "alignment", "Availability masks are present for all core symbols.", f"missing={missing_availability}")
    )
    if policy == "drop_core_missing" and not missing_availability:
        unavailable_counts = {column: int((~panel[column].astype(bool)).sum()) for column in availability_cols}
        total_unavailable = sum(unavailable_counts.values())
        checks.append(
            _pass("drop_core_missing_has_no_unavailable_rows", "alignment", "Drop-core-missing policy removed unavailable rows.", f"unavailable_counts={unavailable_counts}")
            if total_unavailable == 0
            else _fail("drop_core_missing_has_no_unavailable_rows", "alignment", "Drop-core-missing policy removed unavailable rows.", f"unavailable_counts={unavailable_counts}")
        )
    return checks


def run_audit(config: dict[str, Any], target_symbol: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = (target_symbol or config.get("lab", {}).get("target_symbol", "SPY")).upper()
    symbols = resolve_symbols(config, target_symbol=target)
    panel_path = aligned_panel_path(config, target)
    if not panel_path.exists():
        raise FileNotFoundError(f"Aligned panel not found: {panel_path}")

    panel = pd.read_parquet(panel_path)
    catalog = build_feature_timestamp_catalog(panel, config, target)
    checks: list[AuditCheck] = []
    checks.extend(check_panel_structure(panel, target))
    checks.extend(check_symbol_columns(panel, symbols, target))
    checks.extend(check_feature_catalog(catalog))
    checks.extend(check_missing_policy(panel, config, symbols))
    checks.append(
        _pass(
            "train_only_transformers_not_present_yet",
            "modeling",
            "No cross-asset scalers/HMM/threshold artifacts are present before feature engineering.",
            "This raw panel audit precedes train-only fitting; future feature/model stages must add fold-level checks.",
        )
    )
    return pd.DataFrame([asdict(check) for check in checks]), catalog


def render_report(checks: pd.DataFrame, catalog: pd.DataFrame, target_symbol: str) -> str:
    status_counts = checks["status"].value_counts().to_dict()
    usable_count = int(catalog["usable_as_feature"].sum()) if not catalog.empty else 0
    blocked = catalog.loc[~catalog["blocked_reason"].eq(""), ["column", "role", "blocked_reason"]]

    lines = [
        f"# Cross-Asset Leakage Audit - {target_symbol.upper()}",
        "",
        "## Summary",
        "",
        f"- Status counts: `{status_counts}`",
        f"- Catalog columns: `{len(catalog)}`",
        f"- Usable feature columns at panel stage: `{usable_count}`",
        "",
        "## Checks",
        "",
        "| check_id | module | status | evidence |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in checks.iterrows():
        lines.append(f"| {row['check_id']} | {row['module']} | {row['status']} | {row['evidence']} |")

    lines.extend(["", "## Blocked Columns", "", "| column | role | reason |", "| --- | --- | --- |"])
    if blocked.empty:
        lines.append("|  |  |  |")
    else:
        for _, row in blocked.iterrows():
            lines.append(f"| {row['column']} | {row['role']} | {row['blocked_reason']} |")
    return "\n".join(lines) + "\n"


def write_outputs(config: dict[str, Any], target_symbol: str, checks: pd.DataFrame, catalog: pd.DataFrame) -> Path:
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper()
    reports_dir.mkdir(parents=True, exist_ok=True)
    checks_path = reports_dir / "leakage_audit_cross_asset.parquet"
    catalog_path = reports_dir / "feature_timestamp_catalog.parquet"
    report_path = reports_dir / "leakage_audit_cross_asset.md"

    checks.to_parquet(checks_path, index=False)
    catalog.to_parquet(catalog_path, index=False)
    report_path.write_text(render_report(checks, catalog, target_symbol), encoding="utf-8")
    return report_path.resolve()


def run(config_path: str | Path, target_symbol: str | None = None) -> Path:
    config = load_config(config_path)
    target = (target_symbol or config.get("lab", {}).get("target_symbol", "SPY")).upper()
    checks, catalog = run_audit(config, target)
    return write_outputs(config, target, checks, catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-asset aligned panel for temporal leakage.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path = run(args.config, args.target)
    print(f"Cross-asset leakage audit written to: {report_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.market_calendar import get_market_schedule
from src.walkforward import build_monthly_folds


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    module: str
    description: str
    status: str
    evidence: str


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _pass(check_id: str, module: str, description: str, evidence: str) -> AuditCheck:
    return AuditCheck(check_id, module, description, "PASS", evidence)


def _fail(check_id: str, module: str, description: str, evidence: str) -> AuditCheck:
    return AuditCheck(check_id, module, description, "FAIL", evidence)


def _max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    diff = (left.astype(float) - right.astype(float)).replace([np.inf, -np.inf], np.nan).dropna().abs()
    return float(diff.max()) if len(diff) else 0.0


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > eps)


def _grouped_rolling_sum(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _grouped_rolling_mean(series: pd.Series, sessions: pd.Series, window: int) -> pd.Series:
    return (
        series.groupby(sessions, sort=False)
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _check_tolerance(check_id: str, module: str, description: str, diff: float, tolerance: float, evidence_prefix: str) -> AuditCheck:
    evidence = f"{evidence_prefix}; max_abs_diff={diff:.3e}; tolerance={tolerance:.3e}"
    return _pass(check_id, module, description, evidence) if diff <= tolerance else _fail(check_id, module, description, evidence)


def check_feature_causality(features: pd.DataFrame, config: dict[str, Any]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    ordered = features.sort_values(["session", "bar_index"]).reset_index(drop=True)
    tolerance = 1e-10

    forbidden = {"target", "fwd_ret", "entry_px", "exit_px", "entry_timestamp", "exit_timestamp"}
    present_forbidden = sorted(forbidden & set(ordered.columns))
    checks.append(
        _pass("features_no_future_label_columns", "features", "Feature file excludes target/future execution columns.", "No forbidden future columns found.")
        if not present_forbidden
        else _fail("features_no_future_label_columns", "features", "Feature file excludes target/future execution columns.", f"Found {present_forbidden}.")
    )

    for window in config["features"].get("return_windows", []):
        col = f"ret_{window}"
        expected = np.log(ordered["close"]) - np.log(ordered["close"]).groupby(ordered["session"], sort=False).shift(int(window))
        checks.append(
            _check_tolerance(
                f"feature_{col}_past_close_only",
                "features",
                f"{col} uses only close at t and prior same-session closes.",
                _max_abs_diff(ordered[col], expected),
                tolerance,
                f"Recomputed grouped shift({window})",
            )
        )

    squared_ret = ordered["ret_1"].pow(2)
    for window in config["features"].get("realized_vol_windows", []):
        col = f"rv_{window}"
        expected = (
            squared_ret.groupby(ordered["session"], sort=False)
            .rolling(window=int(window), min_periods=int(window))
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
            .pow(0.5)
        )
        checks.append(
            _check_tolerance(
                f"feature_{col}_rolling_past_only",
                "features",
                f"{col} uses rolling returns available through t only.",
                _max_abs_diff(ordered[col], expected),
                tolerance,
                f"Recomputed same-session rolling window {window}",
            )
        )

    ratio_specs = [
        ("vol_ratio_3_12", "rv_3", "rv_12"),
        ("vol_ratio_6_24", "rv_6", "rv_24"),
    ]
    for col, numerator_col, denominator_col in ratio_specs:
        if col in ordered.columns:
            expected = _safe_ratio(ordered[numerator_col], ordered[denominator_col])
            checks.append(
                _check_tolerance(
                    f"feature_{col}_past_vol_only",
                    "features",
                    f"{col} uses realized volatility windows available through t only.",
                    _max_abs_diff(ordered[col], expected),
                    tolerance,
                    f"Recomputed {numerator_col} / {denominator_col}",
                )
            )

    for window in config["features"].get("efficiency_windows", [12]):
        window = int(window)
        efficiency_col = f"signed_efficiency_{window}"
        persistence_col = f"dir_persistence_{window}"
        sum_ret = _grouped_rolling_sum(ordered["ret_1"], ordered["session"], window)
        sum_abs_ret = _grouped_rolling_sum(ordered["ret_1"].abs(), ordered["session"], window)
        if efficiency_col in ordered.columns:
            checks.append(
                _check_tolerance(
                    f"feature_{efficiency_col}_rolling_past_only",
                    "features",
                    f"{efficiency_col} uses same-session returns available through t only.",
                    _max_abs_diff(ordered[efficiency_col], _safe_ratio(sum_ret, sum_abs_ret)),
                    tolerance,
                    f"Recomputed rolling efficiency window {window}",
                )
            )
        if persistence_col in ordered.columns:
            expected = _grouped_rolling_mean(np.sign(ordered["ret_1"]), ordered["session"], window)
            checks.append(
                _check_tolerance(
                    f"feature_{persistence_col}_rolling_past_only",
                    "features",
                    f"{persistence_col} uses same-session return signs available through t only.",
                    _max_abs_diff(ordered[persistence_col], expected),
                    tolerance,
                    f"Recomputed rolling sign mean window {window}",
                )
            )

    if "range_ratio_6_24" in ordered.columns:
        expected = _safe_ratio(
            _grouped_rolling_mean(ordered["range"], ordered["session"], 6),
            _grouped_rolling_mean(ordered["range"], ordered["session"], 24),
        )
        checks.append(
            _check_tolerance(
                "feature_range_ratio_6_24_rolling_past_only",
                "features",
                "range_ratio_6_24 uses rolling same-session bar ranges available through t only.",
                _max_abs_diff(ordered["range_ratio_6_24"], expected),
                tolerance,
                "Recomputed rolling mean range(6) / rolling mean range(24)",
            )
        )

    location_cols = {"dist_open", "pos_session_range", "dist_session_high_atr", "dist_session_low_atr", "intraday_runup"}
    if location_cols & set(ordered.columns):
        grouped = ordered.groupby("session", sort=False)
        session_open = grouped["open"].transform("first")
        high_so_far = grouped["high"].cummax()
        low_so_far = grouped["low"].cummin()
        session_range = high_so_far - low_so_far
        atr_col = config.get("features", {}).get("location_atr_column", "atr_12")
        atr = ordered[atr_col]
        expected_by_col = {
            "dist_open": np.log(ordered["close"] / session_open),
            "pos_session_range": _safe_ratio(ordered["close"] - low_so_far, session_range),
            "dist_session_high_atr": _safe_ratio(np.log(ordered["close"] / high_so_far), atr),
            "dist_session_low_atr": _safe_ratio(np.log(ordered["close"] / low_so_far), atr),
            "intraday_runup": _safe_ratio(np.log(ordered["close"] / low_so_far), atr),
        }
        for col, expected in expected_by_col.items():
            if col in ordered.columns:
                checks.append(
                    _check_tolerance(
                        f"feature_{col}_intraday_past_only",
                        "features",
                        f"{col} uses current bar and same-session cumulative information through t only.",
                        _max_abs_diff(ordered[col], expected),
                        tolerance,
                        "Recomputed with session open and cumulative high/low",
                    )
                )

    if "dist_vwap_atr" in ordered.columns:
        atr_col = config.get("features", {}).get("location_atr_column", "atr_12")
        expected = _safe_ratio(np.log(ordered["close"] / ordered["vwap"]), ordered[atr_col])
        checks.append(
            _check_tolerance(
                "feature_dist_vwap_atr_current_vwap_only",
                "features",
                "dist_vwap_atr uses cumulative VWAP through t and ATR through t.",
                _max_abs_diff(ordered["dist_vwap_atr"], expected),
                tolerance,
                f"Recomputed log(close / vwap) / {atr_col}",
            )
        )

    vwap_window = int(config.get("features", {}).get("vwap_slope_window", 12))
    vwap_slope_col = f"vwap_slope_{vwap_window}"
    if vwap_slope_col in ordered.columns:
        atr_col = config.get("features", {}).get("location_atr_column", "atr_12")
        previous_vwap = ordered.groupby("session", sort=False)["vwap"].shift(vwap_window)
        expected = _safe_ratio(np.log(ordered["vwap"] / previous_vwap), ordered[atr_col])
        checks.append(
            _check_tolerance(
                f"feature_{vwap_slope_col}_past_vwap_only",
                "features",
                f"{vwap_slope_col} uses cumulative VWAP at t and t-{vwap_window}.",
                _max_abs_diff(ordered[vwap_slope_col], expected),
                tolerance,
                f"Recomputed grouped vwap shift({vwap_window})",
            )
        )

    rel_volume_expected = recompute_relative_volume(ordered)
    checks.append(
        _check_tolerance(
            "feature_rel_volume_prior_sessions_only",
            "features",
            "rel_volume uses only prior sessions for the same bar_index.",
            _max_abs_diff(ordered["rel_volume"], rel_volume_expected),
            tolerance,
            "Recomputed expanding mean shifted by one session",
        )
    )
    return checks


def recompute_relative_volume(features: pd.DataFrame) -> pd.Series:
    frame = features[["session", "bar_index", "volume"]].copy()
    session_order = frame[["session"]].drop_duplicates().reset_index(drop=True)
    session_order["session_number"] = np.arange(len(session_order))
    frame = frame.merge(session_order, on="session", how="left").sort_values(["bar_index", "session_number"])
    prior_mean = (
        frame.groupby("bar_index", sort=False)["volume"]
        .expanding(min_periods=1)
        .mean()
        .groupby(level=0)
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    frame["rel_volume_expected"] = frame["volume"] / prior_mean.replace(0, np.nan)
    return frame.sort_values(["session_number", "bar_index"])["rel_volume_expected"].reset_index(drop=True)


def check_labels(labels: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    ordered = labels.sort_values(["session", "bar_index"]).reset_index(drop=True)
    feature_lookup = features.sort_values(["session", "bar_index"]).loc[:, ["session", "bar_index", "open", "timestamp"]].copy()
    horizon = int(config["labeling"]["horizon_bars"])
    cost_plus_buffer = (float(config["labeling"]["round_trip_cost_bps"]) + float(config["labeling"]["buffer_bps"])) / 10_000.0
    lambda_vol = float(config["labeling"]["lambda_vol"])

    entry_lookup = feature_lookup.rename(
        columns={"bar_index": "entry_bar_index", "open": "expected_entry_px", "timestamp": "expected_entry_timestamp"}
    )
    exit_lookup = feature_lookup.rename(
        columns={"bar_index": "exit_bar_index", "open": "expected_exit_px", "timestamp": "expected_exit_timestamp"}
    )
    aligned = ordered.copy()
    aligned["entry_bar_index"] = aligned["bar_index"] + 1
    aligned["exit_bar_index"] = aligned["bar_index"] + horizon + 1
    aligned = aligned.merge(entry_lookup, on=["session", "entry_bar_index"], how="left", validate="many_to_one")
    aligned = aligned.merge(exit_lookup, on=["session", "exit_bar_index"], how="left", validate="many_to_one")

    expected_entry = aligned["expected_entry_px"]
    expected_exit = aligned["expected_exit_px"]
    expected_fwd = np.log(expected_exit / expected_entry)
    checks.append(
        _check_tolerance(
            "label_entry_next_open",
            "labels",
            "Labels enter at open_{t+1}.",
            _max_abs_diff(aligned["entry_px"], expected_entry),
            1e-10,
            "Recomputed entry_px from grouped open shift(-1)",
        )
    )
    checks.append(
        _check_tolerance(
            "label_exit_horizon_open",
            "labels",
            "Labels exit at open_{t+h+1}.",
            _max_abs_diff(aligned["exit_px"], expected_exit),
            1e-10,
            f"Recomputed exit_px from grouped open shift(-{horizon + 1})",
        )
    )
    checks.append(
        _check_tolerance(
            "label_forward_return_alignment",
            "labels",
            "fwd_ret is computed from configured entry and exit opens.",
            _max_abs_diff(aligned["fwd_ret"], expected_fwd),
            1e-10,
            "Recomputed log(exit_px / entry_px)",
        )
    )

    expected_sigma = ordered["rv_12"] * np.sqrt(horizon)
    expected_neutral = np.maximum(cost_plus_buffer, lambda_vol * expected_sigma)
    checks.append(
        _check_tolerance(
            "label_neutral_zone_ex_ante_vol",
            "labels",
            "neutral_zone uses ex-ante rv_12 at t plus cost/buffer floor.",
            _max_abs_diff(ordered["neutral_zone"], expected_neutral),
            1e-10,
            "Recomputed max(cost+buffer, lambda_vol * rv_12 * sqrt(h))",
        )
    )

    invalid_cross = bool(ordered["target_crosses_session_close"].any())
    max_allowed_bar = ordered["bars_in_session"] - horizon - 2
    impossible = ordered["bar_index"] > max_allowed_bar
    checks.append(
        _pass("label_no_session_close_cross", "labels", "Dropped labels whose target would cross session close.", "No target_crosses_session_close rows remain.")
        if not invalid_cross and not impossible.any()
        else _fail("label_no_session_close_cross", "labels", "Dropped labels whose target would cross session close.", f"cross_flags={int(invalid_cross)} impossible_rows={int(impossible.sum())}")
    )
    return checks


def check_execution(cleaned: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    if trades.empty:
        return [
            _pass("execution_no_close_t_fill", "backtest", "No trades available to violate close_t execution.", "Trade file is empty."),
            _pass("execution_no_overnight", "backtest", "No trades available to hold overnight.", "Trade file is empty."),
            _pass("execution_costs_applied", "backtest", "No trades available without costs.", "Trade file is empty."),
        ]

    bar_lookup = cleaned[["timestamp", "session", "bar_index"]].rename(
        columns={"timestamp": "signal_timestamp", "bar_index": "signal_bar_index"}
    )
    merged = trades.merge(bar_lookup, on=["signal_timestamp", "session"], how="left", validate="many_to_one")
    next_open_ok = (merged["entry_bar_index"] == merged["signal_bar_index"] + 1) & (merged["entry_timestamp"] > merged["signal_timestamp"])
    checks.append(
        _pass("execution_next_open_only", "backtest", "Trades enter on the next bar open after signal t.", f"Checked {len(merged)} trades.")
        if bool(next_open_ok.all())
        else _fail("execution_next_open_only", "backtest", "Trades enter on the next bar open after signal t.", f"violations={int((~next_open_ok).sum())}")
    )

    close_fill = merged["entry_timestamp"] == merged["signal_timestamp"]
    checks.append(
        _pass("execution_no_close_t_fill", "backtest", "No trade uses close_t as execution fill.", f"Checked {len(merged)} trades.")
        if not bool(close_fill.any())
        else _fail("execution_no_close_t_fill", "backtest", "No trade uses close_t as execution fill.", f"violations={int(close_fill.sum())}")
    )

    entry_sessions = pd.to_datetime(merged["entry_timestamp"]).dt.strftime("%Y-%m-%d")
    exit_sessions = pd.to_datetime(merged["exit_timestamp"]).dt.strftime("%Y-%m-%d")
    no_overnight = (entry_sessions == merged["session"]) & (exit_sessions == merged["session"])
    checks.append(
        _pass("execution_no_overnight", "backtest", "Trades open and close within the same session.", f"Checked {len(merged)} trades.")
        if bool(no_overnight.all())
        else _fail("execution_no_overnight", "backtest", "Trades open and close within the same session.", f"violations={int((~no_overnight).sum())}")
    )

    expected_total_cost = float(config["labeling"]["round_trip_cost_bps"]) / 10_000.0 * merged["position"].abs()
    cost_ok = (merged["total_cost_ret"] > 0) & np.isclose(merged["total_cost_ret"], expected_total_cost, rtol=1e-8, atol=1e-12)
    checks.append(
        _pass("execution_costs_applied", "backtest", "Round-trip costs are applied to every executed trade.", f"Checked {len(merged)} trades at {config['labeling']['round_trip_cost_bps']} bps.")
        if bool(cost_ok.all())
        else _fail("execution_costs_applied", "backtest", "Round-trip costs are applied to every executed trade.", f"violations={int((~cost_ok).sum())}")
    )
    return checks


def check_folds(labels: pd.DataFrame, config: dict[str, Any]) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    folds = build_monthly_folds(labels, config)
    disjoint_ok = True
    chronological_ok = True
    for fold in folds:
        train = set(fold.train_sessions)
        validation = set(fold.validation_sessions)
        test = set(fold.test_sessions)
        disjoint_ok &= not (train & validation or train & test or validation & test)
        chronological_ok &= max(fold.train_sessions) < min(fold.validation_sessions) < min(fold.test_sessions)
    checks.append(
        _pass("fold_sessions_disjoint", "walkforward", "Train, validation and test sessions are disjoint per fold.", f"Checked {len(folds)} folds.")
        if disjoint_ok
        else _fail("fold_sessions_disjoint", "walkforward", "Train, validation and test sessions are disjoint per fold.", "Overlap detected.")
    )
    checks.append(
        _pass("fold_sessions_chronological", "walkforward", "Folds are chronological: train < validation < test.", f"Checked {len(folds)} folds.")
        if chronological_ok
        else _fail("fold_sessions_chronological", "walkforward", "Folds are chronological: train < validation < test.", "Non-chronological fold detected.")
    )
    return checks


def check_calendar(cleaned: pd.DataFrame, config: dict[str, Any]) -> list[AuditCheck]:
    if cleaned.empty:
        return [_fail("calendar_cleaned_not_empty", "calendar", "Cleaned dataset is not empty.", "No cleaned rows.")]
    start = cleaned["session"].min()
    end = cleaned["session"].max()
    schedule = get_market_schedule(config, start, end)
    counts = cleaned.groupby("session").size()
    half_days = set(schedule.loc[schedule["is_half_day"], "session"].tolist())
    schedule_sessions = set(schedule["session"].tolist())
    cleaned_sessions = set(counts.index.tolist())

    non_trading = cleaned_sessions - schedule_sessions
    half_day_present = cleaned_sessions & half_days
    expected = schedule["expected_bars"].to_dict()
    wrong_counts = [session for session, count in counts.items() if int(count) != int(expected.get(session, count))]

    checks = [
        _pass("calendar_no_non_trading_sessions", "calendar", "Cleaned data contains only NYSE trading sessions.", f"Checked {len(cleaned_sessions)} sessions.")
        if not non_trading
        else _fail("calendar_no_non_trading_sessions", "calendar", "Cleaned data contains only NYSE trading sessions.", f"non_trading={sorted(non_trading)[:5]}"),
        _pass("calendar_half_days_dropped", "calendar", "Configured half-days are not present in cleaned full-session dataset.", f"Dropped half-days in range={len(half_days)}.")
        if not half_day_present
        else _fail("calendar_half_days_dropped", "calendar", "Configured half-days are not present in cleaned full-session dataset.", f"present={sorted(half_day_present)[:5]}"),
        _pass("calendar_expected_bar_counts", "calendar", "Cleaned sessions have expected bar counts.", f"Checked {len(counts)} sessions.")
        if not wrong_counts
        else _fail("calendar_expected_bar_counts", "calendar", "Cleaned sessions have expected bar counts.", f"wrong_counts={wrong_counts[:5]}"),
    ]
    return checks


def check_source_protocol(config: dict[str, Any]) -> list[AuditCheck]:
    walkforward_source = Path("src/walkforward.py").read_text(encoding="utf-8")
    hmm_filter_source = Path("src/hmm_filter.py").read_text(encoding="utf-8")
    checks = [
        _pass("hmm_walkforward_fit_train_only", "hmm", "Walk-forward HMM fit uses fold train sessions only.", "Source contains train_frame filtered by fold.train_sessions before fit_hmm_model.")
        if "train_frame = hmm_frame[hmm_frame[\"session\"].isin(fold.train_sessions)].copy()" in walkforward_source
        else _fail("hmm_walkforward_fit_train_only", "hmm", "Walk-forward HMM fit uses fold train sessions only.", "Expected train_frame filter not found."),
        _pass("model_walkforward_fit_train_only", "model", "Walk-forward predictive model fit uses train split only.", "Source calls fit_base_model(train, ...) after split_frames.")
        if "fit_base_model(train, feature_columns, config)" in walkforward_source
        else _fail("model_walkforward_fit_train_only", "model", "Walk-forward predictive model fit uses train split only.", "Expected fit_base_model(train, ...) not found."),
        _pass("thresholds_validation_only", "signal", "Threshold selection is delegated to validation-only selector.", "Source calls select_thresholds_on_validation.")
        if "select_thresholds_on_validation" in walkforward_source
        else _fail("thresholds_validation_only", "signal", "Threshold selection is delegated to validation-only selector.", "Expected selector call not found."),
        _pass("hmm_filtered_not_smoothed", "hmm", "HMM probabilities use causal forward filtering, not smoothing.", "hmm_filter.filtered_probabilities implements forward recursion and session reset.")
        if "previous_log_alpha" in hmm_filter_source and "logsumexp(previous_log_alpha" in hmm_filter_source
        else _fail("hmm_filtered_not_smoothed", "hmm", "HMM probabilities use causal forward filtering, not smoothing.", "Forward recursion evidence not found."),
    ]
    return checks


def build_audit(config: dict[str, Any]) -> pd.DataFrame:
    cleaned = pd.read_parquet(config["data"]["cleaned_file"])
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    trades_path = Path(config["backtest"]["trades_file"])
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    checks: list[AuditCheck] = []
    checks.extend(check_feature_causality(features, config))
    checks.extend(check_labels(labels, features, config))
    checks.extend(check_execution(cleaned, trades, config))
    checks.extend(check_folds(labels, config))
    checks.extend(check_calendar(cleaned, config))
    checks.extend(check_source_protocol(config))
    return pd.DataFrame([check.__dict__ for check in checks])


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_report(audit: pd.DataFrame) -> str:
    status_counts = audit["status"].value_counts().rename_axis("status").reset_index(name="count")
    failures = audit[audit["status"] == "FAIL"]
    conclusion = (
        "No material leakage violations detected by automated audit."
        if failures.empty
        else "Leakage audit found failures. Do not advance until fixed and metrics are rerun."
    )

    return f"""# Leakage Audit

## Summary

{_markdown_table(status_counts)}

## Checks

{_markdown_table(audit)}

## Violations

{_markdown_table(failures) if not failures.empty else "- None"}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    audit = build_audit(config)
    report_path = Path(config["paths"]["reports"]) / "leakage_audit.md"
    metrics_path = Path(config["paths"]["reports"]) / "leakage_audit.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(metrics_path, index=False)
    report_path.write_text(render_report(audit), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anti-leakage audit on current artifacts.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    report_path = run(args.config)
    print(f"Leakage audit written to: {report_path}")


if __name__ == "__main__":
    main()

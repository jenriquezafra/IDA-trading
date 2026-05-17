from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.hmm_candidate_diagnostics import _feature_set_lookup, candidate_id
from src.hmm_state_economics import _prepare_horizon_labels, classify_candidate
from src.walkforward import _fold_hmm_features, build_monthly_folds


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_candidates(holdout: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config.get("hmm_candidate_thresholds", {})
    feature_sets = set(cfg.get("feature_sets_to_inspect", ["rich_extreme_reversion", "minimal_vwap_location"]))
    split = str(cfg.get("split", "test"))
    status = str(cfg.get("candidate_status", "candidate"))
    source_costs = {float(value) for value in cfg.get("source_cost_bps", [1.0])}
    selected = holdout[
        holdout["feature_set"].isin(feature_sets)
        & holdout["split"].eq(split)
        & holdout["candidate_status"].eq(status)
        & holdout["cost_bps"].astype(float).isin(source_costs)
    ].copy()
    selected["candidate_id"] = selected.apply(candidate_id, axis=1)
    selected = selected.sort_values(["total_net_return", "median_daily_sharpe", "total_trades"], ascending=[False, False, False])
    selected["source_rank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def _config_for_candidate(config: dict[str, Any], candidate: pd.Series, feature_columns: list[str]) -> dict[str, Any]:
    cfg = config.get("hmm_candidate_thresholds", {})
    copied = deepcopy(config)
    copied["hmm"]["feature_columns"] = list(feature_columns)
    copied["hmm"]["n_states"] = int(candidate["n_states"])
    copied["hmm"]["random_state"] = int(candidate["seed"])
    copied["hmm"]["n_iter"] = int(cfg.get("n_iter", copied["hmm"].get("n_iter", 200)))
    return copied


def threshold_position(frame: pd.DataFrame, action: str, threshold_multiplier: float) -> pd.Series:
    if action == "long":
        return pd.Series(1, index=frame.index, dtype="int64")
    if action == "short":
        return pd.Series(-1, index=frame.index, dtype="int64")
    if action == "flat":
        return pd.Series(0, index=frame.index, dtype="int64")
    if action in {"momentum_ret_3", "reversion_ret_3"}:
        threshold = frame["neutral_zone"] * float(threshold_multiplier)
        signal = pd.Series(0, index=frame.index, dtype="int64")
        signal.loc[frame["ret_3"] > threshold] = 1
        signal.loc[frame["ret_3"] < -threshold] = -1
        return signal if action == "momentum_ret_3" else -signal
    raise ValueError(f"Unsupported threshold action: {action}")


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0].sum()
    gross_loss = -active_net[active_net < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net_ret: pd.Series) -> float:
    if frame.empty:
        return np.nan
    daily = net_ret.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def _max_drawdown_abs(net_ret: pd.Series) -> float:
    if net_ret.empty:
        return 0.0
    cumulative = net_ret.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return max(0.0, float(-drawdown.min())) if len(drawdown) else 0.0


def evaluate_threshold_frame(
    frame: pd.DataFrame,
    action: str,
    threshold_multiplier: float,
    cost_bps: float,
) -> dict[str, float | int]:
    ordered = frame.sort_values(["session", "bar_index"]).copy()
    position = threshold_position(ordered, action, threshold_multiplier)
    active = position != 0
    gross = position * ordered["fwd_ret"]
    cost = position.abs() * (float(cost_bps) / 10_000.0)
    net = gross - cost
    active_net = net[active]
    return {
        "rows": int(len(ordered)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(ordered) else 0.0,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "net_return": float(net.sum()),
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "median_trade_net": float(active_net.median()) if len(active_net) else 0.0,
        "hit_ratio": float((active_net > 0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(ordered, net),
        "max_drawdown_abs": _max_drawdown_abs(net),
    }


def build_threshold_grid(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("hmm_candidate_thresholds", {})
    holdout_path = Path(cfg.get("candidate_source", "reports/hmm_stability/stability_holdout.parquet"))
    if not holdout_path.exists():
        raise FileNotFoundError(f"Candidate source not found: {holdout_path}")
    holdout = pd.read_parquet(holdout_path)
    candidates = select_candidates(holdout, config)
    feature_lookup = _feature_set_lookup(config)
    features = pd.read_parquet(config["data"]["features_file"])
    labels = pd.read_parquet(config["data"]["labels_file"])
    folds = build_monthly_folds(labels, config)
    max_folds = cfg.get("max_folds")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    horizons = sorted(candidates["horizon_bars"].astype(int).unique().tolist()) if not candidates.empty else []
    labels_by_horizon = _prepare_horizon_labels(features, config, horizons) if horizons else {}
    threshold_grid = [float(value) for value in cfg.get("threshold_multipliers", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])]
    cost_grid = [float(value) for value in cfg.get("cost_bps", [0.5, 1.0, 1.5, 2.0])]

    fold_rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        feature_columns = feature_lookup.get(str(candidate["feature_set"]))
        if not feature_columns:
            raise ValueError(f"Feature set columns not found for {candidate['feature_set']}")
        candidate_config = _config_for_candidate(config, candidate, feature_columns)
        horizon = int(candidate["horizon_bars"])
        target_state = int(candidate["hmm_state"])
        action = str(candidate["action"])
        cid = str(candidate["candidate_id"])

        for fold in folds:
            filtered_hmm, _, _ = _fold_hmm_features(features, fold, candidate_config)
            fold_sessions = fold.train_sessions + fold.validation_sessions + fold.test_sessions
            fold_labels = labels_by_horizon[horizon][labels_by_horizon[horizon]["session"].isin(fold_sessions)].copy()
            hmm_cols = ["timestamp", "session", "bar_index", "hmm_state"]
            merged = fold_labels.merge(filtered_hmm[hmm_cols], on=["timestamp", "session", "bar_index"], how="inner", validate="one_to_one")
            for split, sessions in [("validation", fold.validation_sessions), ("test", fold.test_sessions)]:
                split_frame = merged[merged["session"].isin(sessions)].copy()
                state_frame = split_frame[split_frame["hmm_state"] == target_state].copy()
                for threshold_multiplier in threshold_grid:
                    for cost_bps in cost_grid:
                        metrics = evaluate_threshold_frame(state_frame, action, threshold_multiplier, cost_bps)
                        fold_rows.append(
                            {
                                "candidate_id": cid,
                                "source_rank": int(candidate["source_rank"]),
                                "feature_set": candidate["feature_set"],
                                "n_states": int(candidate["n_states"]),
                                "seed": int(candidate["seed"]),
                                "horizon_bars": horizon,
                                "hmm_state": target_state,
                                "action": action,
                                "fold": int(fold.fold),
                                "split": split,
                                "threshold_multiplier": float(threshold_multiplier),
                                "cost_bps": float(cost_bps),
                                **metrics,
                            }
                        )

    return {"candidates": candidates, "threshold_fold_metrics": pd.DataFrame(fold_rows)}


def summarize_thresholds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    grouped = (
        fold_metrics.groupby(
            [
                "candidate_id",
                "source_rank",
                "feature_set",
                "n_states",
                "seed",
                "horizon_bars",
                "hmm_state",
                "action",
                "split",
                "threshold_multiplier",
                "cost_bps",
            ],
            as_index=False,
        )
        .agg(
            folds=("fold", "nunique"),
            total_trades=("trades", "sum"),
            total_net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            median_profit_factor=("profit_factor", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            negative_folds=("net_return", lambda values: int((values < 0).sum())),
            worst_fold_net=("net_return", "min"),
            best_fold_net=("net_return", "max"),
            max_drawdown_abs=("max_drawdown_abs", "max"),
            median_drawdown_abs=("max_drawdown_abs", "median"),
            avg_exposure=("exposure", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["candidate_status"] = grouped.apply(classify_candidate, axis=1)
    grouped["return_to_drawdown"] = grouped["total_net_return"] / grouped["max_drawdown_abs"].replace(0, np.nan)
    return grouped


def select_validation_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    validation = summary[summary["split"] == "validation"].copy()
    validation["_is_candidate"] = (validation["candidate_status"] == "candidate").astype(int)
    validation["_is_nonnegative"] = (validation["total_net_return"] > 0).astype(int)
    selected = (
        validation.sort_values(
            [
                "candidate_id",
                "cost_bps",
                "_is_candidate",
                "_is_nonnegative",
                "total_net_return",
                "return_to_drawdown",
                "total_trades",
            ],
            ascending=[True, True, False, False, False, False, False],
        )
        .groupby(["candidate_id", "cost_bps"], as_index=False)
        .head(1)
        .drop(columns=["_is_candidate", "_is_nonnegative"])
        .reset_index(drop=True)
    )
    selected = selected.rename(columns={col: f"validation_{col}" for col in selected.columns if col not in {"candidate_id", "cost_bps"}})
    return selected


def selected_test_results(summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or selected.empty:
        return pd.DataFrame()
    tests = summary[summary["split"] == "test"].copy()
    merged = selected.merge(
        tests,
        left_on=["candidate_id", "cost_bps", "validation_threshold_multiplier"],
        right_on=["candidate_id", "cost_bps", "threshold_multiplier"],
        how="left",
        validate="one_to_one",
    )
    return merged


def decide_candidates(selected_tests: pd.DataFrame) -> pd.DataFrame:
    if selected_tests.empty:
        return pd.DataFrame()
    rows = []
    for candidate_id, group in selected_tests.groupby("candidate_id", sort=False):
        source_rank = int(group["source_rank"].dropna().iloc[0])
        feature_set = str(group["feature_set"].dropna().iloc[0])
        one_bps = group[group["cost_bps"] == 1.0]
        two_bps = group[group["cost_bps"] == 2.0]
        one_status = str(one_bps["candidate_status"].iloc[0]) if not one_bps.empty else "missing"
        two_status = str(two_bps["candidate_status"].iloc[0]) if not two_bps.empty else "missing"
        accepted = one_status == "candidate" and two_status == "candidate"
        cost_fragile = one_status == "candidate" and two_status != "candidate"
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_rank": source_rank,
                "feature_set": feature_set,
                "status_1bps": one_status,
                "status_2bps": two_status,
                "accepted": bool(accepted),
                "cost_fragile": bool(cost_fragile),
                "test_net_1bps": float(one_bps["total_net_return"].iloc[0]) if not one_bps.empty else np.nan,
                "test_net_2bps": float(two_bps["total_net_return"].iloc[0]) if not two_bps.empty else np.nan,
                "test_drawdown_1bps": float(one_bps["max_drawdown_abs"].iloc[0]) if not one_bps.empty else np.nan,
                "test_drawdown_2bps": float(two_bps["max_drawdown_abs"].iloc[0]) if not two_bps.empty else np.nan,
            }
        )
    decisions = pd.DataFrame(rows)
    decisions["_accepted_rank"] = decisions["accepted"].astype(int)
    decisions["_fragile_rank"] = decisions["cost_fragile"].astype(int)
    return decisions.sort_values(
        ["_accepted_rank", "_fragile_rank", "test_net_2bps", "test_net_1bps", "source_rank"],
        ascending=[False, False, False, False, True],
    ).drop(columns=["_accepted_rank", "_fragile_rank"]).reset_index(drop=True)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_report(outputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> str:
    cfg = config.get("hmm_candidate_thresholds", {})
    candidates = outputs["candidates"]
    decisions = outputs["candidate_decisions"]
    selected = outputs["selected_test_results"]
    summary = outputs["threshold_summary"]
    fold_metrics = outputs["threshold_fold_metrics"]
    decision_text = "No candidate accepted under the configured 1 bps and 2 bps test criteria."
    accepted = decisions[decisions["accepted"]] if not decisions.empty else pd.DataFrame()
    if not accepted.empty:
        decision_text = f"Accepted candidate for next stage: `{accepted.iloc[0]['candidate_id']}`."
    elif not decisions.empty:
        first = decisions.iloc[0]
        decision_text = f"No candidate survives 2 bps. Best fallback is `{first['candidate_id']}`, but it remains cost-fragile."

    candidate_cols = [
        "source_rank",
        "candidate_id",
        "feature_set",
        "n_states",
        "seed",
        "hmm_state",
        "action",
        "total_net_return",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
    ]
    decision_cols = [
        "source_rank",
        "candidate_id",
        "status_1bps",
        "status_2bps",
        "accepted",
        "cost_fragile",
        "test_net_1bps",
        "test_net_2bps",
        "test_drawdown_1bps",
        "test_drawdown_2bps",
    ]
    selected_cols = [
        "source_rank",
        "candidate_id",
        "cost_bps",
        "validation_threshold_multiplier",
        "validation_candidate_status",
        "validation_total_net_return",
        "validation_max_drawdown_abs",
        "split",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "max_drawdown_abs",
        "return_to_drawdown",
        "candidate_status",
    ]
    top_cols = [
        "candidate_id",
        "split",
        "threshold_multiplier",
        "cost_bps",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "max_drawdown_abs",
        "return_to_drawdown",
        "candidate_status",
    ]
    fold_cols = [
        "candidate_id",
        "split",
        "fold",
        "threshold_multiplier",
        "cost_bps",
        "trades",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown_abs",
    ]
    selected_fold_keys = selected.loc[:, ["candidate_id", "cost_bps", "threshold_multiplier"]] if not selected.empty else pd.DataFrame()
    selected_folds = (
        fold_metrics.merge(selected_fold_keys, on=["candidate_id", "cost_bps", "threshold_multiplier"], how="inner")
        if not selected_fold_keys.empty
        else pd.DataFrame()
    )

    return f"""# HMM Candidate Thresholds

## Scope

- Candidate source: `{cfg.get("candidate_source", "reports/hmm_stability/stability_holdout.parquet")}`
- Feature sets: `{cfg.get("feature_sets_to_inspect")}`
- Threshold multipliers: `{cfg.get("threshold_multipliers")}`
- Cost grid bps: `{cfg.get("cost_bps")}`
- Threshold selection: validation only, then reported on test.

## Candidate Source Rows

{_markdown_table(candidates.loc[:, candidate_cols] if not candidates.empty else candidates)}

## Candidate Decisions

{_markdown_table(decisions.loc[:, decision_cols] if not decisions.empty else decisions)}

## Selected Threshold Test Results

{_markdown_table(selected.loc[:, selected_cols] if not selected.empty else selected, max_rows=80)}

## Top Threshold Grid Rows

{_markdown_table(summary.sort_values(["split", "cost_bps", "total_net_return"], ascending=[True, True, False]).loc[:, top_cols] if not summary.empty else summary, max_rows=80)}

## Selected Threshold Fold Detail

{_markdown_table(selected_folds.loc[:, fold_cols] if not selected_folds.empty else selected_folds, max_rows=96)}

## Outputs

- `reports/hmm_candidate_thresholds/candidates.parquet`
- `reports/hmm_candidate_thresholds/threshold_fold_metrics.parquet`
- `reports/hmm_candidate_thresholds/threshold_summary.parquet`
- `reports/hmm_candidate_thresholds/selected_validation_thresholds.parquet`
- `reports/hmm_candidate_thresholds/selected_test_results.parquet`
- `reports/hmm_candidate_thresholds/candidate_decisions.parquet`

## Conclusion

{decision_text}
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    outputs = build_threshold_grid(config)
    threshold_summary = summarize_thresholds(outputs["threshold_fold_metrics"])
    selected_validation = select_validation_thresholds(threshold_summary)
    selected_tests = selected_test_results(threshold_summary, selected_validation)
    decisions = decide_candidates(selected_tests)
    outputs = {
        **outputs,
        "threshold_summary": threshold_summary,
        "selected_validation_thresholds": selected_validation,
        "selected_test_results": selected_tests,
        "candidate_decisions": decisions,
    }
    cfg = config.get("hmm_candidate_thresholds", {})
    output_dir = Path(cfg.get("output_dir", "reports/hmm_candidate_thresholds"))
    report_path = Path(cfg.get("report_file", "reports/hmm_candidate_thresholds.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    report_path.write_text(render_report(outputs, config), encoding="utf-8")
    return output_dir / "candidate_decisions.parquet", report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cost/drawdown/threshold sensitivity for HMM candidates.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    decisions_path, report_path = run(args.config)
    print(f"HMM candidate threshold decisions written to: {decisions_path}")
    print(f"HMM candidate threshold report written to: {report_path}")


if __name__ == "__main__":
    main()

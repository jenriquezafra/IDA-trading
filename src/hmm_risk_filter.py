from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import (
    attach_forward_returns,
    build_forward_returns,
    enrich_posteriors_with_state_metadata,
    filter_posteriors_for_economics,
    filter_posteriors_to_stable_combos,
)
from src.hmm_state_interpretability_cross_asset import _markdown_table


STRATEGIES = ("always_flat", "momentum_simple", "reversion_simple", "vwap_location", "supervised_simple")
FILTERS = (
    "no_filter",
    "only_risk_on",
    "exclude_risk_off",
    "exclude_high_vol",
    "exclude_chop",
    "exclude_stress",
    "reduce_stress",
)
STRESS_LABELS = {"risk_off_stress", "high_volatility_expansion"}
RISK_ON_LABELS = {"risk_on_trend", "tech_led_narrow_rally"}


def _risk_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hmm_risk_filter", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "hmm_risk_filter.md"


def _path_from_template(template: str, target_symbol: str) -> Path:
    return Path(template.format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0].sum()
    gross_loss = -active_net[active_net < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    if frame.empty:
        return np.nan
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def _max_drawdown(net: pd.Series) -> float:
    if net.empty:
        return 0.0
    equity = net.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def _max_daily_abs_net_share(frame: pd.DataFrame, net: pd.Series) -> float:
    if frame.empty or net.empty:
        return np.nan
    daily = net.groupby(frame["session"]).sum()
    denom = daily.abs().sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(daily.abs().max() / denom)


def _signed_threshold(values: pd.Series, threshold: float) -> pd.Series:
    signal = pd.Series(0.0, index=values.index)
    finite = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    signal.loc[finite > float(threshold)] = 1.0
    signal.loc[finite < -float(threshold)] = -1.0
    return signal


def base_position(frame: pd.DataFrame, strategy: str, threshold: float) -> pd.Series:
    if strategy == "always_flat":
        return pd.Series(0.0, index=frame.index)
    if strategy == "momentum_simple":
        return _signed_threshold(frame["target_ret_3"], threshold)
    if strategy == "reversion_simple":
        return -_signed_threshold(frame["target_ret_3"], threshold)
    if strategy == "vwap_location":
        if "target_dist_vwap_atr" not in frame:
            return pd.Series(0.0, index=frame.index)
        return -_signed_threshold(frame["target_dist_vwap_atr"], threshold)
    if strategy == "supervised_simple":
        if "supervised_score" not in frame:
            return pd.Series(0.0, index=frame.index)
        return _signed_threshold(frame["supervised_score"], threshold)
    raise ValueError(f"Unsupported strategy: {strategy}")


def filter_multiplier(frame: pd.DataFrame, filter_name: str) -> pd.Series:
    labels = frame["proposed_label"].astype(str)
    multiplier = pd.Series(1.0, index=frame.index)
    if filter_name == "no_filter":
        return multiplier
    if filter_name == "only_risk_on":
        return labels.isin(RISK_ON_LABELS).astype(float)
    if filter_name == "exclude_risk_off":
        return (~labels.eq("risk_off_stress")).astype(float)
    if filter_name == "exclude_high_vol":
        return (~labels.eq("high_volatility_expansion")).astype(float)
    if filter_name == "exclude_chop":
        return (~labels.eq("chop_neutral")).astype(float)
    if filter_name == "exclude_stress":
        return (~labels.isin(STRESS_LABELS)).astype(float)
    if filter_name == "reduce_stress":
        multiplier.loc[labels.isin(STRESS_LABELS)] = 0.5
        return multiplier
    raise ValueError(f"Unsupported filter_name: {filter_name}")


def same_hour_multiplier(frame: pd.DataFrame, selected_hours: tuple[int, ...]) -> pd.Series:
    if not selected_hours:
        return pd.Series(0.0, index=frame.index)
    return frame["hour"].isin(selected_hours).astype(float)


def evaluate_position(frame: pd.DataFrame, position: pd.Series, cost_bps: float) -> dict[str, float | int]:
    position = position.astype(float).fillna(0.0)
    active = position.abs() > 0
    gross = position * frame["fwd_ret"].astype(float)
    cost = position.abs() * (float(cost_bps) / 10_000.0)
    net = gross - cost
    active_net = net[active]
    active_gross = gross[active]
    sessions = int(frame["session"].nunique()) if "session" in frame else 0
    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "turnover": float(active.sum() / sessions) if sessions else 0.0,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "net_return": float(net.sum()),
        "avg_trade_gross": float(active_gross.mean()) if len(active_gross) else 0.0,
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": _max_drawdown(net),
        "max_daily_abs_net_share": _max_daily_abs_net_share(frame, net),
    }


def filter_id(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}__fold{int(row['fold'])}"
        f"__{row['strategy']}__{row['filter_name']}__h{int(row['horizon_bars'])}"
        f"__c{float(row['cost_bps']):g}__thr{float(row['threshold']):g}"
    )


def load_candidate_combos(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _risk_cfg(config)
    source = _path_from_template(
        str(cfg.get("combo_source", "results/{target_symbol}/state_rules_selected_specs.parquet")),
        target_symbol,
    )
    frame = pd.read_parquet(source)
    if frame.empty:
        return pd.DataFrame()
    combos = frame.loc[:, ["feature_set", "n_states", "seed", "fold"]].drop_duplicates().reset_index(drop=True)
    max_combos = cfg.get("max_combos")
    if max_combos:
        combos = combos.head(int(max_combos)).copy()
    return combos


def attach_supervised_scores(frame: pd.DataFrame, config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _risk_cfg(config)
    path_template = cfg.get("supervised_predictions")
    if not path_template:
        return frame
    path = _path_from_template(str(path_template), target_symbol)
    if not path.exists():
        return frame
    predictions = pd.read_parquet(path)
    if "score" not in predictions:
        return frame
    score = predictions.loc[:, ["timestamp", "session", "bar_index", "score"]].rename(columns={"score": "supervised_score"})
    return frame.merge(score, on=["timestamp", "session", "bar_index"], how="left", validate="many_to_one")


def build_filter_dataset(config: dict[str, Any], target_symbol: str, combos: pd.DataFrame) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    results_dir = results_output_dir(config, target_symbol)
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config))
    state_names = pd.read_parquet(results_dir / "state_name_grid.parquet")
    stability_grid = pd.read_parquet(results_dir / "state_stability_grid.parquet")
    posteriors = filter_posteriors_for_economics(pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet"), config)
    posteriors = filter_posteriors_to_stable_combos(posteriors, state_names, stability_grid, config)
    posteriors = posteriors.merge(combos, on=["feature_set", "n_states", "seed", "fold"], how="inner")
    if posteriors.empty:
        return pd.DataFrame()
    horizons = [int(value) for value in _risk_cfg(config).get("horizons", [1, 3, 6, 12])]
    forward_returns = build_forward_returns(features, horizons)
    enriched = enrich_posteriors_with_state_metadata(posteriors, state_names, stability_grid)
    merged = attach_forward_returns(enriched, forward_returns)
    indexed_features = features.reset_index(names="source_index")
    extra_cols = ["source_index", "target_dist_vwap_atr"]
    available = [col for col in extra_cols if col in indexed_features.columns]
    if "target_dist_vwap_atr" in available:
        merged = merged.merge(indexed_features.loc[:, available], on="source_index", how="left", validate="many_to_one")
    return attach_supervised_scores(merged, config, target_symbol)


def thresholds_for_strategy(config: dict[str, Any], strategy: str) -> list[float]:
    cfg = _risk_cfg(config)
    if strategy in {"momentum_simple", "reversion_simple"}:
        return [float(value) for value in cfg.get("return_thresholds", [0.0, 0.0001, 0.0002, 0.0005])]
    if strategy == "vwap_location":
        return [float(value) for value in cfg.get("vwap_thresholds", [0.0, 0.5, 1.0, 1.5])]
    if strategy == "supervised_simple":
        return [float(value) for value in cfg.get("supervised_thresholds", [0.0, 0.02, 0.05, 0.10])]
    return [0.0]


def split_combo_frame(merged: pd.DataFrame, combo: pd.Series, split: str, horizon: int) -> pd.DataFrame:
    mask = (
        merged["feature_set"].eq(combo["feature_set"])
        & merged["n_states"].eq(int(combo["n_states"]))
        & merged["seed"].eq(int(combo["seed"]))
        & merged["fold"].eq(int(combo["fold"]))
        & merged["split"].eq(split)
        & merged["horizon_bars"].eq(int(horizon))
    )
    return merged.loc[mask].copy()


def evaluate_filter_triplet(
    frame: pd.DataFrame,
    combo: pd.Series,
    split: str,
    strategy: str,
    filter_name: str,
    horizon: int,
    cost_bps: float,
    threshold: float,
    selected_hours: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    base = base_position(frame, strategy, threshold)
    hmm_mult = filter_multiplier(frame, filter_name)
    hours = selected_hours if selected_hours is not None else tuple(sorted(int(value) for value in frame.loc[hmm_mult > 0, "hour"].dropna().unique().tolist()))
    hour_mult = same_hour_multiplier(frame, hours)
    rows = []
    for bucket, position in (
        ("base", base),
        ("hmm_filter", base * hmm_mult),
        ("same_hour_control", base * hour_mult),
        ("always_flat", pd.Series(0.0, index=frame.index)),
    ):
        row = {
            "feature_set": combo["feature_set"],
            "n_states": int(combo["n_states"]),
            "seed": int(combo["seed"]),
            "fold": int(combo["fold"]),
            "split": split,
            "strategy": strategy,
            "filter_name": filter_name,
            "bucket": bucket,
            "horizon_bars": int(horizon),
            "cost_bps": float(cost_bps),
            "threshold": float(threshold),
            "selected_hours": ",".join(str(hour) for hour in hours),
        }
        row["filter_id"] = filter_id(row)
        rows.append({**row, **evaluate_position(frame, position, cost_bps)})
    return add_filter_deltas(pd.DataFrame(rows))


def add_filter_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = ["filter_id", "feature_set", "n_states", "seed", "fold", "split", "strategy", "filter_name", "horizon_bars", "cost_bps", "threshold"]
    filtered = rows[rows["bucket"].eq("hmm_filter")].copy()
    for bucket in ["base", "same_hour_control", "always_flat"]:
        control = rows[rows["bucket"].eq(bucket)].loc[:, [*keys, "net_return", "daily_sharpe", "max_drawdown", "turnover", "avg_trade_net"]].rename(
            columns={metric: f"{bucket}_{metric}" for metric in ["net_return", "daily_sharpe", "max_drawdown", "turnover", "avg_trade_net"]}
        )
        filtered = filtered.merge(control, on=keys, how="left", validate="one_to_one")
    filtered["net_return_delta_vs_base"] = filtered["net_return"] - filtered["base_net_return"]
    filtered["daily_sharpe_delta_vs_base"] = filtered["daily_sharpe"] - filtered["base_daily_sharpe"]
    filtered["drawdown_reduction_vs_base"] = filtered["base_max_drawdown"] - filtered["max_drawdown"]
    filtered["turnover_reduction_vs_base"] = filtered["base_turnover"] - filtered["turnover"]
    filtered["opportunity_cost_vs_base"] = np.maximum(filtered["base_net_return"] - filtered["net_return"], 0.0)
    filtered["net_return_delta_vs_same_hour"] = filtered["net_return"] - filtered["same_hour_control_net_return"]
    filtered["daily_sharpe_delta_vs_same_hour"] = filtered["daily_sharpe"] - filtered["same_hour_control_daily_sharpe"]
    filtered["drawdown_reduction_vs_same_hour"] = filtered["same_hour_control_max_drawdown"] - filtered["max_drawdown"]
    delta_cols = [
        "net_return_delta_vs_base",
        "daily_sharpe_delta_vs_base",
        "drawdown_reduction_vs_base",
        "turnover_reduction_vs_base",
        "opportunity_cost_vs_base",
        "net_return_delta_vs_same_hour",
        "daily_sharpe_delta_vs_same_hour",
        "drawdown_reduction_vs_same_hour",
    ]
    return rows.merge(filtered.loc[:, [*keys, *delta_cols]], on=keys, how="left", validate="many_to_one")


def classify_filter_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _risk_cfg(config)
    if row["bucket"] != "hmm_filter":
        return "control"
    if row["strategy"] == "always_flat":
        return "flat_reference"
    if int(row["trades"]) < int(cfg.get("min_trades", 50)):
        return "rejected_insufficient_trades"
    improves_sharpe = row["daily_sharpe_delta_vs_base"] > float(cfg.get("min_sharpe_improvement", 0.0))
    improves_drawdown = row["drawdown_reduction_vs_base"] > float(cfg.get("min_drawdown_reduction", 0.0))
    if not (improves_sharpe or improves_drawdown):
        return "rejected_no_quality_improvement"
    if bool(cfg.get("require_nonnegative_net", True)) and row["net_return"] <= 0:
        return "rejected_negative_net"
    if bool(cfg.get("require_same_hour_improvement", True)) and row["daily_sharpe_delta_vs_same_hour"] <= 0 and row["drawdown_reduction_vs_same_hour"] <= 0:
        return "rejected_no_same_hour_edge"
    if row["profit_factor"] <= float(cfg.get("min_profit_factor", 1.0)):
        return "rejected_weak_profit_factor"
    return "risk_filter_candidate"


def add_filter_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["filter_status"] = rows.apply(lambda row: classify_filter_row(row, config), axis=1)
    return rows


def validation_grid(merged: pd.DataFrame, combos: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _risk_cfg(config)
    split = str(cfg.get("candidate_split", "validation"))
    horizons = [int(value) for value in cfg.get("horizons", [1, 3, 6, 12])]
    costs = [float(value) for value in cfg.get("cost_bps", [1.0, 2.0, 5.0])]
    strategies = [str(value) for value in cfg.get("strategies", STRATEGIES) if str(value) != "always_flat"]
    filters = [str(value) for value in cfg.get("filters", FILTERS) if str(value) != "no_filter"]
    rows: list[dict[str, Any]] = []
    for _, combo in combos.iterrows():
        for horizon in horizons:
            frame = split_combo_frame(merged, combo, split, horizon)
            if frame.empty:
                continue
            for strategy in strategies:
                if strategy == "supervised_simple" and "supervised_score" not in frame:
                    continue
                for threshold in thresholds_for_strategy(config, strategy):
                    base = base_position(frame, strategy, threshold)
                    base_metrics_by_cost = {cost_bps: evaluate_position(frame, base, cost_bps) for cost_bps in costs}
                    flat_metrics_by_cost = {cost_bps: evaluate_position(frame, pd.Series(0.0, index=frame.index), cost_bps) for cost_bps in costs}
                    for filter_name in filters:
                        hmm_mult = filter_multiplier(frame, filter_name)
                        selected_hours = tuple(sorted(int(value) for value in frame.loc[hmm_mult > 0, "hour"].dropna().unique().tolist()))
                        hour_mult = same_hour_multiplier(frame, selected_hours)
                        for cost_bps in costs:
                            base_metrics = base_metrics_by_cost[cost_bps]
                            hmm_metrics = evaluate_position(frame, base * hmm_mult, cost_bps)
                            same_hour_metrics = evaluate_position(frame, base * hour_mult, cost_bps)
                            flat_metrics = flat_metrics_by_cost[cost_bps]
                            deltas = {
                                "net_return_delta_vs_base": float(hmm_metrics["net_return"]) - float(base_metrics["net_return"]),
                                "daily_sharpe_delta_vs_base": float(hmm_metrics["daily_sharpe"]) - float(base_metrics["daily_sharpe"]),
                                "drawdown_reduction_vs_base": float(base_metrics["max_drawdown"]) - float(hmm_metrics["max_drawdown"]),
                                "turnover_reduction_vs_base": float(base_metrics["turnover"]) - float(hmm_metrics["turnover"]),
                                "opportunity_cost_vs_base": max(float(base_metrics["net_return"]) - float(hmm_metrics["net_return"]), 0.0),
                                "net_return_delta_vs_same_hour": float(hmm_metrics["net_return"]) - float(same_hour_metrics["net_return"]),
                                "daily_sharpe_delta_vs_same_hour": float(hmm_metrics["daily_sharpe"]) - float(same_hour_metrics["daily_sharpe"]),
                                "drawdown_reduction_vs_same_hour": float(same_hour_metrics["max_drawdown"]) - float(hmm_metrics["max_drawdown"]),
                            }
                            hours_text = ",".join(str(hour) for hour in selected_hours)
                            for bucket, metrics in (
                                ("base", base_metrics),
                                ("hmm_filter", hmm_metrics),
                                ("same_hour_control", same_hour_metrics),
                                ("always_flat", flat_metrics),
                            ):
                                row = {
                                    "feature_set": combo["feature_set"],
                                    "n_states": int(combo["n_states"]),
                                    "seed": int(combo["seed"]),
                                    "fold": int(combo["fold"]),
                                    "split": split,
                                    "strategy": strategy,
                                    "filter_name": filter_name,
                                    "bucket": bucket,
                                    "horizon_bars": int(horizon),
                                    "cost_bps": float(cost_bps),
                                    "threshold": float(threshold),
                                    "selected_hours": hours_text,
                                }
                                row["filter_id"] = filter_id(row)
                                rows.append({**row, **metrics, **deltas})
    grid = pd.DataFrame(rows)
    return add_filter_status(grid, config) if not grid.empty else grid


def select_validation_filters(grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    filtered = grid[grid["bucket"].eq("hmm_filter")].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["_is_candidate"] = filtered["filter_status"].eq("risk_filter_candidate").astype(int)
    filtered["_is_nonnegative"] = (filtered["net_return"] > 0).astype(int)
    sort_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "cost_bps",
        "_is_candidate",
        "_is_nonnegative",
        "daily_sharpe_delta_vs_base",
        "drawdown_reduction_vs_base",
        "net_return_delta_vs_same_hour",
        "net_return",
        "threshold",
    ]
    selected = (
        filtered.sort_values(
            sort_cols,
            ascending=[True, True, True, True, True, True, True, True, False, False, False, False, False, False, True],
            kind="stable",
        )
        .drop_duplicates(["feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "cost_bps"])
        .copy()
    )
    selected = selected.sort_values(
        ["feature_set", "n_states", "seed", "fold", "_is_candidate", "daily_sharpe_delta_vs_base", "drawdown_reduction_vs_base", "net_return"],
        ascending=[True, True, True, True, False, False, False, False],
        kind="stable",
    )
    max_filters = int(_risk_cfg(config).get("max_filters_per_combo", 8))
    selected = selected.groupby(["feature_set", "n_states", "seed", "fold"], as_index=False, sort=False).head(max_filters)
    selected_ids = selected["filter_id"].drop_duplicates()
    return grid[grid["filter_id"].isin(selected_ids)].reset_index(drop=True)


def selected_specs(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    filtered = selected[selected["bucket"].eq("hmm_filter")].copy()
    cols = [
        "filter_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "cost_bps",
        "threshold",
        "selected_hours",
    ]
    return filtered.loc[:, cols].drop_duplicates("filter_id").reset_index(drop=True)


def evaluate_specs_on_split(merged: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    frames = []
    for _, spec in specs.iterrows():
        frame = split_combo_frame(merged, spec, split, int(spec["horizon_bars"]))
        if frame.empty:
            continue
        hours = tuple(int(value) for value in str(spec["selected_hours"]).split(",") if value != "")
        frames.append(
            evaluate_filter_triplet(
                frame,
                spec,
                split,
                str(spec["strategy"]),
                str(spec["filter_name"]),
                int(spec["horizon_bars"]),
                float(spec["cost_bps"]),
                float(spec["threshold"]),
                hours,
            )
        )
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return add_filter_status(rows, config) if not rows.empty else rows


def fold_stability(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    filtered = rows[(rows["bucket"].eq("hmm_filter")) & (rows["filter_status"].eq("risk_filter_candidate"))].copy()
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.groupby(["split", "strategy", "filter_name", "cost_bps"], as_index=False)
        .agg(
            candidate_rows=("filter_id", "nunique"),
            folds_present=("fold", "nunique"),
            combos_present=("feature_set", "nunique"),
            median_net_return=("net_return", "median"),
            median_sharpe_delta=("daily_sharpe_delta_vs_base", "median"),
            median_drawdown_reduction=("drawdown_reduction_vs_base", "median"),
        )
        .sort_values(["split", "candidate_rows", "median_sharpe_delta"], ascending=[True, False, False])
    )


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    combos: pd.DataFrame,
    selected_validation: pd.DataFrame,
    selected_test: pd.DataFrame,
    stability: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _risk_cfg(config)
    val = selected_validation[selected_validation["bucket"].eq("hmm_filter")].copy() if not selected_validation.empty else pd.DataFrame()
    test = selected_test[selected_test["bucket"].eq("hmm_filter")].copy() if not selected_test.empty else pd.DataFrame()
    val_counts = val["filter_status"].value_counts().rename_axis("filter_status").reset_index(name="rows") if not val.empty else pd.DataFrame()
    test_counts = test["filter_status"].value_counts().rename_axis("filter_status").reset_index(name="rows") if not test.empty else pd.DataFrame()
    top_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "cost_bps",
        "threshold",
        "selected_hours",
        "trades",
        "turnover",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "daily_sharpe_delta_vs_base",
        "drawdown_reduction_vs_base",
        "turnover_reduction_vs_base",
        "opportunity_cost_vs_base",
        "daily_sharpe_delta_vs_same_hour",
        "drawdown_reduction_vs_same_hour",
        "filter_status",
    ]
    def _ranked_display(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        ranked = frame.copy()
        ranked["_is_candidate"] = ranked["filter_status"].eq("risk_filter_candidate").astype(int)
        ranked = ranked.sort_values(
            ["_is_candidate", "daily_sharpe_delta_vs_base", "drawdown_reduction_vs_base", "net_return"],
            ascending=[False, False, False, False],
            kind="stable",
        )
        return ranked.drop(columns=["_is_candidate"])

    candidate_test = test[test["filter_status"].eq("risk_filter_candidate")] if not test.empty else pd.DataFrame()
    conclusion = (
        "Some HMM risk filters improve validation and still pass on test; evaluate cost sensitivity before accepting."
        if not candidate_test.empty
        else "No selected HMM risk filter remains a test candidate under the configured gates."
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    return f"""# HMM Risk Filter - {target_symbol.upper()}

## Scope

- Combo source: `{cfg.get("combo_source", "results/{target_symbol}/state_rules_selected_specs.parquet")}`
- Combos evaluated: `{len(combos)}`
- Strategies: `{cfg.get("strategies", list(STRATEGIES))}`
- Filters: `{cfg.get("filters", list(FILTERS))}`
- Horizons: `{cfg.get("horizons", [1, 3, 6, 12])}`
- Costs bps: `{cfg.get("cost_bps", [1.0, 2.0, 5.0])}`
- Threshold selection used: `validation only`
- Test selection used: `no`
- Same-hour control: hours permitted by the HMM filter in validation, then frozen.

## Validation Status Counts

{_markdown_table(val_counts)}

## Test Status Counts

{_markdown_table(test_counts)}

## Top Validation Filters

{_markdown_table(_ranked_display(val).loc[:, [col for col in top_cols if col in val.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Test Sanity For Selected Filters

{_markdown_table(_ranked_display(test).loc[:, [col for col in top_cols if col in test.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Fold Stability

{_markdown_table(stability, max_rows=40)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    combos = load_candidate_combos(config, target)
    merged = build_filter_dataset(config, target, combos) if not combos.empty else pd.DataFrame()
    grid = validation_grid(merged, combos, config) if not merged.empty else pd.DataFrame()
    selected_validation = select_validation_filters(grid, config) if not grid.empty else pd.DataFrame()
    specs = selected_specs(selected_validation)
    selected_test = evaluate_specs_on_split(merged, specs, str(_risk_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()
    stability = fold_stability(pd.concat([selected_validation, selected_test], ignore_index=True) if not selected_validation.empty or not selected_test.empty else pd.DataFrame())

    outputs = {
        "risk_filter_threshold_grid": results_dir / "risk_filter_threshold_grid.parquet",
        "risk_filter_comparison": results_dir / "risk_filter_comparison.parquet",
        "risk_filter_validation": results_dir / "risk_filter_validation.parquet",
        "risk_filter_test": results_dir / "risk_filter_test.parquet",
        "risk_filter_selected_specs": results_dir / "risk_filter_selected_specs.parquet",
        "risk_filter_fold_stability": results_dir / "risk_filter_fold_stability.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(outputs["risk_filter_threshold_grid"], index=False)
    pd.concat([selected_validation, selected_test], ignore_index=True).to_parquet(outputs["risk_filter_comparison"], index=False)
    selected_validation.to_parquet(outputs["risk_filter_validation"], index=False)
    selected_test.to_parquet(outputs["risk_filter_test"], index=False)
    specs.to_parquet(outputs["risk_filter_selected_specs"], index=False)
    stability.to_parquet(outputs["risk_filter_fold_stability"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, combos, selected_validation, selected_test, stability, outputs), encoding="utf-8")
    return report_path, outputs["risk_filter_comparison"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HMM states as risk/no-trade filters.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, comparison_path = run(args.config, args.target)
    print(f"HMM risk filter report written to: {report_path}")
    print(f"Risk filter comparison written to: {comparison_path}")


if __name__ == "__main__":
    main()

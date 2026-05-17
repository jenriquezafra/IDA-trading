from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import (
    STATE_KEYS,
    attach_forward_returns,
    build_forward_returns,
    enrich_posteriors_with_state_metadata,
    filter_posteriors_for_economics,
    filter_posteriors_to_stable_combos,
)
from src.hmm_state_interpretability_cross_asset import _markdown_table


RULE_TYPES = ("long_momentum", "short_momentum", "mean_reversion", "no_trade", "reduce_risk")
BUCKETS = ("hmm_state_rule", "no_hmm_equivalent", "same_hour_control")
SPEC_KEYS = ["feature_set", "n_states", "seed", "fold", "hmm_state"]


def _rules_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("state_rules_cross_asset", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "state_rules_cross_asset.md"


def _path_from_template(template: str, target_symbol: str) -> Path:
    return Path(template.format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def _prob_column(state: int) -> str:
    return f"hmm_p{int(state)}"


def _candidate_state_id(row: pd.Series) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}"
        f"__fold{int(row['fold'])}__state{int(row['hmm_state'])}"
    )


def _rule_id(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['candidate_state_id']}__{row['rule_type']}__h{int(row['horizon_bars'])}"
        f"__c{float(row['cost_bps']):g}__p{float(row['hmm_prob_threshold']):g}"
        f"__s{float(row['signal_threshold']):g}"
    )


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


def position_for_rule(frame: pd.DataFrame, rule_type: str, signal_threshold: float, source_action: str) -> pd.Series:
    signal = frame["target_ret_3"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    threshold = float(signal_threshold)
    position = pd.Series(0.0, index=frame.index)
    if rule_type == "long_momentum":
        position.loc[signal > threshold] = 1.0
    elif rule_type == "short_momentum":
        position.loc[signal < -threshold] = -1.0
    elif rule_type == "mean_reversion":
        position.loc[signal > threshold] = -1.0
        position.loc[signal < -threshold] = 1.0
    elif rule_type == "no_trade":
        position.loc[:] = 0.0
    elif rule_type == "reduce_risk":
        if source_action == "long":
            position.loc[:] = 0.5
        elif source_action == "short":
            position.loc[:] = -0.5
        elif source_action == "momentum":
            position.loc[signal > threshold] = 0.5
            position.loc[signal < -threshold] = -0.5
        elif source_action == "reversion":
            position.loc[signal > threshold] = -0.5
            position.loc[signal < -threshold] = 0.5
    else:
        raise ValueError(f"Unsupported rule_type: {rule_type}")
    return position


def evaluate_rule_metrics(frame: pd.DataFrame, rule_type: str, signal_threshold: float, cost_bps: float, source_action: str) -> dict[str, float | int]:
    position = position_for_rule(frame, rule_type, signal_threshold, source_action)
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
        "median_trade_net": float(active_net.median()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": _max_drawdown(net),
        "top_hour_pct": float(frame["hour"].value_counts(normalize=True).iloc[0]) if len(frame) else np.nan,
        "top_session_pct": float(frame["session"].value_counts(normalize=True).iloc[0]) if len(frame) else np.nan,
        "max_daily_abs_net_share": _max_daily_abs_net_share(frame, net),
    }


def load_candidate_states(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _rules_cfg(config)
    source = _path_from_template(
        str(cfg.get("candidate_source", "results/{target_symbol}/state_economic_candidates.parquet")),
        target_symbol,
    )
    candidates = pd.read_parquet(source)
    if candidates.empty:
        return pd.DataFrame()
    candidate_split = str(cfg.get("candidate_split", "validation"))
    candidates = candidates[candidates["split"].eq(candidate_split)].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["candidate_state_id"] = candidates.apply(_candidate_state_id, axis=1)
    candidates = candidates.sort_values(
        ["avg_trade_net", "profit_factor", "daily_sharpe", "cost_bps"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    state_cols = [
        "candidate_state_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "hmm_state",
        "local_state_id",
        "proposed_label",
        "stability_status",
    ]
    best_cols = {
        "action": "source_action",
        "horizon_bars": "source_horizon_bars",
        "cost_bps": "source_cost_bps",
        "avg_trade_net": "source_avg_trade_net",
        "profit_factor": "source_profit_factor",
        "daily_sharpe": "source_daily_sharpe",
    }
    states = candidates.drop_duplicates("candidate_state_id").loc[:, [*state_cols, *best_cols.keys()]].rename(columns=best_cols)
    max_states = cfg.get("max_candidate_states")
    if max_states:
        states = states.head(int(max_states)).copy()
    return states.reset_index(drop=True)


def _metadata_from_spec(spec: pd.Series, split: str, horizon: int, cost_bps: float, rule_type: str, prob_threshold: float, signal_threshold: float, bucket: str) -> dict[str, Any]:
    row = {
        "candidate_state_id": spec["candidate_state_id"],
        "feature_set": spec["feature_set"],
        "n_states": int(spec["n_states"]),
        "seed": int(spec["seed"]),
        "fold": int(spec["fold"]),
        "hmm_state": int(spec["hmm_state"]),
        "local_state_id": spec["local_state_id"],
        "proposed_label": spec["proposed_label"],
        "stability_status": spec["stability_status"],
        "source_action": spec["source_action"],
        "source_horizon_bars": int(spec["source_horizon_bars"]),
        "source_cost_bps": float(spec["source_cost_bps"]),
        "split": split,
        "horizon_bars": int(horizon),
        "cost_bps": float(cost_bps),
        "rule_type": rule_type,
        "hmm_prob_threshold": float(prob_threshold),
        "signal_threshold": float(signal_threshold),
        "bucket": bucket,
    }
    row["rule_id"] = _rule_id(row)
    return row


def _split_frame(merged: pd.DataFrame, spec: pd.Series, split: str, horizon: int) -> pd.DataFrame:
    mask = (
        merged["feature_set"].eq(spec["feature_set"])
        & merged["n_states"].eq(int(spec["n_states"]))
        & merged["seed"].eq(int(spec["seed"]))
        & merged["fold"].eq(int(spec["fold"]))
        & merged["split"].eq(split)
        & merged["horizon_bars"].eq(int(horizon))
    )
    return merged.loc[mask].copy()


def _state_frame(split_frame: pd.DataFrame, spec: pd.Series, prob_threshold: float) -> pd.DataFrame:
    state = int(spec["hmm_state"])
    prob_col = _prob_column(state)
    if prob_col not in split_frame:
        return split_frame.iloc[0:0].copy()
    return split_frame[(split_frame["hmm_state"].eq(state)) & (split_frame[prob_col].fillna(0.0) >= float(prob_threshold))].copy()


def _hours_for_state(split_frame: pd.DataFrame, spec: pd.Series, prob_threshold: float) -> tuple[int, ...]:
    state = _state_frame(split_frame, spec, prob_threshold)
    if state.empty:
        return tuple()
    return tuple(sorted(int(value) for value in state["hour"].dropna().unique().tolist()))


def evaluate_rule_buckets(
    split_frame: pd.DataFrame,
    spec: pd.Series,
    split: str,
    horizon: int,
    cost_bps: float,
    rule_type: str,
    prob_threshold: float,
    signal_threshold: float,
    selected_hours: tuple[int, ...] | None,
) -> pd.DataFrame:
    source_action = str(spec["source_action"])
    state_frame = _state_frame(split_frame, spec, prob_threshold)
    hours = selected_hours if selected_hours is not None else tuple(sorted(int(value) for value in state_frame["hour"].dropna().unique().tolist()))
    same_hour_frame = split_frame[split_frame["hour"].isin(hours)].copy() if hours else split_frame.iloc[0:0].copy()
    rows = []
    for bucket, frame in (
        ("hmm_state_rule", state_frame),
        ("no_hmm_equivalent", split_frame),
        ("same_hour_control", same_hour_frame),
    ):
        meta = _metadata_from_spec(spec, split, horizon, cost_bps, rule_type, prob_threshold, signal_threshold, bucket)
        meta["selected_hours"] = ",".join(str(hour) for hour in hours)
        rows.append({**meta, **evaluate_rule_metrics(frame, rule_type, signal_threshold, cost_bps, source_action)})
    output = pd.DataFrame(rows)
    return add_control_deltas(output)


def add_control_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rule_keys = [
        "rule_id",
        "candidate_state_id",
        "split",
        "horizon_bars",
        "cost_bps",
        "rule_type",
        "hmm_prob_threshold",
        "signal_threshold",
    ]
    state = rows[rows["bucket"].eq("hmm_state_rule")].copy()
    for bucket in ["no_hmm_equivalent", "same_hour_control"]:
        controls = rows[rows["bucket"].eq(bucket)].loc[:, [*rule_keys, "avg_trade_net", "net_return", "profit_factor", "daily_sharpe"]].rename(
            columns={
                "avg_trade_net": f"{bucket}_avg_trade_net",
                "net_return": f"{bucket}_net_return",
                "profit_factor": f"{bucket}_profit_factor",
                "daily_sharpe": f"{bucket}_daily_sharpe",
            }
        )
        state = state.merge(controls, on=rule_keys, how="left", validate="one_to_one")
    state["avg_trade_net_vs_no_hmm"] = state["avg_trade_net"] - state["no_hmm_equivalent_avg_trade_net"]
    state["avg_trade_net_vs_same_hour"] = state["avg_trade_net"] - state["same_hour_control_avg_trade_net"]
    state["net_return_vs_no_hmm"] = state["net_return"] - state["no_hmm_equivalent_net_return"]
    state["net_return_vs_same_hour"] = state["net_return"] - state["same_hour_control_net_return"]
    delta_cols = [
        "avg_trade_net_vs_no_hmm",
        "avg_trade_net_vs_same_hour",
        "net_return_vs_no_hmm",
        "net_return_vs_same_hour",
    ]
    return rows.merge(state.loc[:, [*rule_keys, *delta_cols]], on=rule_keys, how="left", validate="many_to_one")


def classify_rule_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _rules_cfg(config)
    if row["bucket"] != "hmm_state_rule":
        return "control"
    if row["rule_type"] == "no_trade":
        return "flat_reference"
    if int(row["trades"]) < int(cfg.get("min_trades", 50)):
        return "rejected_insufficient_trades"
    if row["net_return"] <= 0 or row["avg_trade_net"] <= 0:
        return "rejected_negative_net"
    if row["profit_factor"] <= float(cfg.get("min_profit_factor", 1.10)):
        return "rejected_weak_profit_factor"
    if row["daily_sharpe"] <= float(cfg.get("min_daily_sharpe", 1.0)):
        return "rejected_weak_sharpe"
    if bool(cfg.get("require_no_hmm_improvement", True)) and row["avg_trade_net_vs_no_hmm"] <= 0:
        return "rejected_no_no_hmm_edge"
    if bool(cfg.get("require_same_hour_improvement", True)) and row["avg_trade_net_vs_same_hour"] <= 0:
        return "rejected_no_same_hour_edge"
    if row["top_hour_pct"] > float(cfg.get("max_top_hour_pct", 0.40)):
        return "rejected_hour_concentration"
    if row["top_session_pct"] > float(cfg.get("max_top_session_pct", 0.12)):
        return "rejected_session_concentration"
    if not pd.isna(row["max_daily_abs_net_share"]) and row["max_daily_abs_net_share"] > float(cfg.get("max_daily_abs_net_share", 0.50)):
        return "rejected_extreme_day_concentration"
    return "rule_candidate"


def add_rule_status(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = rows.copy()
    rows["rule_status"] = rows.apply(lambda row: classify_rule_row(row, config), axis=1)
    return rows


def build_rule_dataset(config: dict[str, Any], target_symbol: str, candidate_states: pd.DataFrame) -> pd.DataFrame:
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    results_dir = results_output_dir(config, target_symbol)
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config))
    state_names = pd.read_parquet(results_dir / "state_name_grid.parquet")
    stability_grid = pd.read_parquet(results_dir / "state_stability_grid.parquet")
    posteriors = filter_posteriors_for_economics(pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet"), config)
    posteriors = filter_posteriors_to_stable_combos(posteriors, state_names, stability_grid, config)
    if candidate_states.empty:
        return pd.DataFrame()
    combo_keys = candidate_states.loc[:, ["feature_set", "n_states", "seed", "fold"]].drop_duplicates()
    posteriors = posteriors.merge(combo_keys, on=["feature_set", "n_states", "seed", "fold"], how="inner")
    if posteriors.empty:
        return pd.DataFrame()
    horizons = [int(value) for value in _rules_cfg(config).get("horizons", [1, 3, 6, 12])]
    forward_returns = build_forward_returns(features, horizons)
    enriched = enrich_posteriors_with_state_metadata(posteriors, state_names, stability_grid)
    return attach_forward_returns(enriched, forward_returns)


def validation_threshold_grid(merged: pd.DataFrame, candidate_states: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _rules_cfg(config)
    candidate_split = str(cfg.get("candidate_split", "validation"))
    horizons = [int(value) for value in cfg.get("horizons", [1, 3, 6, 12])]
    costs = [float(value) for value in cfg.get("cost_bps", [1.0, 2.0, 5.0])]
    rule_types = [str(value) for value in cfg.get("rule_types", RULE_TYPES)]
    prob_thresholds = [float(value) for value in cfg.get("hmm_prob_thresholds", [0.0, 0.5, 0.6, 0.7, 0.8])]
    signal_thresholds = [float(value) for value in cfg.get("signal_thresholds", [0.0, 0.0001, 0.0002, 0.0005])]
    rows: list[dict[str, Any]] = []
    for _, spec in candidate_states.iterrows():
        for horizon in horizons:
            split_frame = _split_frame(merged, spec, candidate_split, horizon)
            if split_frame.empty:
                continue
            no_hmm_cache: dict[tuple[str, float, float], dict[str, float | int]] = {}
            for prob_threshold in prob_thresholds:
                state_frame = _state_frame(split_frame, spec, prob_threshold)
                selected_hours = tuple(sorted(int(value) for value in state_frame["hour"].dropna().unique().tolist())) if not state_frame.empty else tuple()
                same_hour_frame = split_frame[split_frame["hour"].isin(selected_hours)].copy() if selected_hours else split_frame.iloc[0:0].copy()
                same_hour_cache: dict[tuple[str, float, float], dict[str, float | int]] = {}
                for rule_type in rule_types:
                    for signal_threshold in signal_thresholds:
                        for cost_bps in costs:
                            metric_key = (rule_type, float(signal_threshold), float(cost_bps))
                            state_metrics = evaluate_rule_metrics(state_frame, rule_type, signal_threshold, cost_bps, str(spec["source_action"]))
                            no_hmm_metrics = no_hmm_cache.get(metric_key)
                            if no_hmm_metrics is None:
                                no_hmm_metrics = evaluate_rule_metrics(split_frame, rule_type, signal_threshold, cost_bps, str(spec["source_action"]))
                                no_hmm_cache[metric_key] = no_hmm_metrics
                            same_hour_metrics = same_hour_cache.get(metric_key)
                            if same_hour_metrics is None:
                                same_hour_metrics = evaluate_rule_metrics(same_hour_frame, rule_type, signal_threshold, cost_bps, str(spec["source_action"]))
                                same_hour_cache[metric_key] = same_hour_metrics

                            deltas = {
                                "avg_trade_net_vs_no_hmm": float(state_metrics["avg_trade_net"]) - float(no_hmm_metrics["avg_trade_net"]),
                                "avg_trade_net_vs_same_hour": float(state_metrics["avg_trade_net"]) - float(same_hour_metrics["avg_trade_net"]),
                                "net_return_vs_no_hmm": float(state_metrics["net_return"]) - float(no_hmm_metrics["net_return"]),
                                "net_return_vs_same_hour": float(state_metrics["net_return"]) - float(same_hour_metrics["net_return"]),
                            }
                            hours_text = ",".join(str(hour) for hour in selected_hours)
                            for bucket, metrics in (
                                ("hmm_state_rule", state_metrics),
                                ("no_hmm_equivalent", no_hmm_metrics),
                                ("same_hour_control", same_hour_metrics),
                            ):
                                meta = _metadata_from_spec(
                                    spec,
                                    candidate_split,
                                    horizon,
                                    cost_bps,
                                    rule_type,
                                    prob_threshold,
                                    signal_threshold,
                                    bucket,
                                )
                                meta["selected_hours"] = hours_text
                                rows.append({**meta, **metrics, **deltas})
    grid = pd.DataFrame(rows)
    return add_rule_status(grid, config) if not grid.empty else grid


def select_validation_rules(validation_grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation_grid.empty:
        return pd.DataFrame()
    state_rows = validation_grid[validation_grid["bucket"].eq("hmm_state_rule")].copy()
    if state_rows.empty:
        return pd.DataFrame()
    state_rows["_is_candidate"] = state_rows["rule_status"].eq("rule_candidate").astype(int)
    state_rows["_is_nonnegative"] = ((state_rows["net_return"] > 0) & (state_rows["avg_trade_net"] > 0)).astype(int)
    sort_cols = [
        "candidate_state_id",
        "rule_type",
        "horizon_bars",
        "cost_bps",
        "_is_candidate",
        "_is_nonnegative",
        "avg_trade_net_vs_no_hmm",
        "avg_trade_net_vs_same_hour",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "hmm_prob_threshold",
        "signal_threshold",
    ]
    selected = (
        state_rows.sort_values(
            sort_cols,
            ascending=[True, True, True, True, False, False, False, False, False, False, False, True, True],
            kind="stable",
        )
        .drop_duplicates(["candidate_state_id", "rule_type", "horizon_bars", "cost_bps"])
        .copy()
    )
    selected = selected.sort_values(
        ["candidate_state_id", "_is_candidate", "avg_trade_net_vs_no_hmm", "avg_trade_net", "profit_factor", "daily_sharpe"],
        ascending=[True, False, False, False, False, False],
        kind="stable",
    )
    max_rules = int(_rules_cfg(config).get("max_rules_per_state", 2))
    selected = selected.groupby("candidate_state_id", as_index=False, sort=False).head(max_rules).copy()
    selected["selection_rank_within_state"] = selected.groupby("candidate_state_id").cumcount() + 1
    selected = selected.drop(columns=["_is_candidate", "_is_nonnegative"])
    rule_ids = selected["rule_id"].drop_duplicates()
    return validation_grid[validation_grid["rule_id"].isin(rule_ids)].reset_index(drop=True)


def selected_rule_specs(selected_validation: pd.DataFrame) -> pd.DataFrame:
    if selected_validation.empty:
        return pd.DataFrame()
    state_rows = selected_validation[selected_validation["bucket"].eq("hmm_state_rule")].copy()
    cols = [
        "rule_id",
        "candidate_state_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "hmm_state",
        "local_state_id",
        "proposed_label",
        "stability_status",
        "source_action",
        "source_horizon_bars",
        "source_cost_bps",
        "horizon_bars",
        "cost_bps",
        "rule_type",
        "hmm_prob_threshold",
        "signal_threshold",
        "selected_hours",
        "selection_rank_within_state",
    ]
    return state_rows.loc[:, [column for column in cols if column in state_rows.columns]].drop_duplicates("rule_id").reset_index(drop=True)


def evaluate_selected_on_split(merged: pd.DataFrame, specs: pd.DataFrame, split: str, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    frames = []
    for _, spec in specs.iterrows():
        split_frame = _split_frame(merged, spec, split, int(spec["horizon_bars"]))
        hours = tuple(int(value) for value in str(spec["selected_hours"]).split(",") if value != "")
        if split_frame.empty:
            continue
        evaluated = evaluate_rule_buckets(
            split_frame,
            spec,
            split,
            int(spec["horizon_bars"]),
            float(spec["cost_bps"]),
            str(spec["rule_type"]),
            float(spec["hmm_prob_threshold"]),
            float(spec["signal_threshold"]),
            hours,
        )
        if "selection_rank_within_state" in spec:
            evaluated["selection_rank_within_state"] = int(spec["selection_rank_within_state"])
        frames.append(evaluated)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return add_rule_status(rows, config) if not rows.empty else rows


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    candidate_states: pd.DataFrame,
    selected_validation: pd.DataFrame,
    selected_test: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _rules_cfg(config)
    validation_state = selected_validation[selected_validation["bucket"].eq("hmm_state_rule")].copy() if not selected_validation.empty else pd.DataFrame()
    test_state = selected_test[selected_test["bucket"].eq("hmm_state_rule")].copy() if not selected_test.empty else pd.DataFrame()
    validation_counts = validation_state["rule_status"].value_counts().rename_axis("rule_status").reset_index(name="rows") if not validation_state.empty else pd.DataFrame()
    test_counts = test_state["rule_status"].value_counts().rename_axis("rule_status").reset_index(name="rows") if not test_state.empty else pd.DataFrame()
    top_cols = [
        "rule_id",
        "proposed_label",
        "rule_type",
        "horizon_bars",
        "cost_bps",
        "hmm_prob_threshold",
        "signal_threshold",
        "selected_hours",
        "trades",
        "turnover",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "avg_trade_net_vs_no_hmm",
        "avg_trade_net_vs_same_hour",
        "rule_status",
    ]
    selected_rule_ids = validation_state["rule_id"].drop_duplicates().tolist() if not validation_state.empty else []
    candidate_test = test_state[test_state["rule_status"].eq("rule_candidate")] if not test_state.empty else pd.DataFrame()
    conclusion = (
        "Some selected validation rules still pass the same rule gates on test. Treat them as candidates for cost sensitivity and risk-filter work, not accepted strategies."
        if not candidate_test.empty
        else "Selected validation rules do not produce robust test rule candidates under the same gates. Keep them as hypotheses for filtering/risk work only."
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    return f"""# State Rules Cross-Asset - {target_symbol.upper()}

## Scope

- Candidate source: `{cfg.get("candidate_source", "results/{target_symbol}/state_economic_candidates.parquet")}`
- Candidate states: `{len(candidate_states)}`
- Rule types: `{cfg.get("rule_types", list(RULE_TYPES))}`
- Horizons: `{cfg.get("horizons", [1, 3, 6, 12])}`
- Costs bps: `{cfg.get("cost_bps", [1.0, 2.0, 5.0])}`
- HMM probability thresholds: `{cfg.get("hmm_prob_thresholds", [0.0, 0.5, 0.6, 0.7, 0.8])}`
- Signal thresholds: `{cfg.get("signal_thresholds", [0.0, 0.0001, 0.0002, 0.0005])}`
- Max rules per state: `{cfg.get("max_rules_per_state", 2)}`
- Threshold selection used: `validation only`
- Test selection used: `no`

## Selected Rule Counts

- Threshold grid rows: `{len(threshold_grid)}`
- Selected rule ids: `{len(selected_rule_ids)}`

## Validation Status Counts

{_markdown_table(validation_counts)}

## Test Status Counts

{_markdown_table(test_counts)}

## Top Validation Rules

{_markdown_table(validation_state.sort_values(["rule_status", "avg_trade_net_vs_no_hmm", "avg_trade_net"], ascending=[True, False, False]).loc[:, [column for column in top_cols if column in validation_state.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Test Sanity For Selected Rules

{_markdown_table(test_state.sort_values(["rule_status", "avg_trade_net_vs_no_hmm", "avg_trade_net"], ascending=[True, False, False]).loc[:, [column for column in top_cols if column in test_state.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    candidate_states = load_candidate_states(config, target)
    merged = build_rule_dataset(config, target, candidate_states) if not candidate_states.empty else pd.DataFrame()
    threshold_grid = validation_threshold_grid(merged, candidate_states, config) if not merged.empty else pd.DataFrame()
    selected_validation = select_validation_rules(threshold_grid, config) if not threshold_grid.empty else pd.DataFrame()
    specs = selected_rule_specs(selected_validation)
    selected_test = evaluate_selected_on_split(merged, specs, str(_rules_cfg(config).get("test_split", "test")), config) if not specs.empty else pd.DataFrame()

    outputs = {
        "state_rules_threshold_grid": results_dir / "state_rules_threshold_grid.parquet",
        "state_rules_validation": results_dir / "state_rules_validation.parquet",
        "state_rules_test": results_dir / "state_rules_test.parquet",
        "state_rules_selected_specs": results_dir / "state_rules_selected_specs.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    threshold_grid.to_parquet(outputs["state_rules_threshold_grid"], index=False)
    selected_validation.to_parquet(outputs["state_rules_validation"], index=False)
    selected_test.to_parquet(outputs["state_rules_test"], index=False)
    specs.to_parquet(outputs["state_rules_selected_specs"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, candidate_states, selected_validation, selected_test, threshold_grid, outputs), encoding="utf-8")
    return report_path, outputs["state_rules_validation"], outputs["state_rules_test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simple falsifiable rules from cross-asset HMM state candidates.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, validation_path, test_path = run(args.config, args.target)
    print(f"State rules report written to: {report_path}")
    print(f"Validation rules written to: {validation_path}")
    print(f"Test rules written to: {test_path}")


if __name__ == "__main__":
    main()

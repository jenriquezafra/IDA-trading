from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _feature_set_version, _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table


STATE_KEYS = ["feature_set", "n_states", "seed", "fold", "hmm_state"]
MERGE_KEYS = ["source_index", "timestamp", "session", "bar_index"]
ACTIONS = ("long", "short", "momentum", "reversion", "flat")


def _economics_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hmm_state_economics_cross_asset", {})


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "hmm_state_economics_cross_asset.md"


def _local_state_id(feature_set: str, n_states: int, seed: int, fold: int, state: int) -> str:
    return f"{feature_set}__k{int(n_states)}__seed{int(seed)}__fold{int(fold)}__state{int(state)}"


def build_forward_returns(features: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    required = {"timestamp", "session", "bar_index", "target_open_next", "target_ret_3"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Features data is missing required forward-return columns: {missing}")

    frame = features.sort_values(["session", "bar_index"], kind="stable").reset_index(names="source_index").copy()
    frame["bars_in_session"] = frame.groupby("session", sort=False)["bar_index"].transform("size")
    frame["hour"] = pd.to_datetime(frame["timestamp"]).dt.hour
    entry = frame["target_open_next"].astype(float)

    rows = []
    for horizon in horizons:
        horizon = int(horizon)
        exit_px = entry.groupby(frame["session"], sort=False).shift(-horizon)
        valid = (
            entry.notna()
            & exit_px.notna()
            & (entry > 0)
            & (exit_px > 0)
            & (frame["bar_index"].astype(int) + horizon + 1 < frame["bars_in_session"].astype(int))
        )
        out = frame.loc[valid, ["source_index", "timestamp", "session", "bar_index", "hour", "target_ret_3"]].copy()
        out["horizon_bars"] = horizon
        out["entry_px"] = entry.loc[valid].to_numpy()
        out["exit_px"] = exit_px.loc[valid].to_numpy()
        out["fwd_ret"] = np.log(out["exit_px"] / out["entry_px"])
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def enrich_posteriors_with_state_metadata(
    posteriors: pd.DataFrame,
    state_names: pd.DataFrame,
    stability_grid: pd.DataFrame,
) -> pd.DataFrame:
    state_meta = state_names.loc[
        :,
        [
            *STATE_KEYS,
            "local_state_id",
            "proposed_label",
            "reference_label",
            "train_validation_label_match",
            "top_positive_features",
            "top_negative_features",
        ],
    ].copy()
    stability_meta = stability_grid.loc[
        :,
        [
            "feature_set",
            "n_states",
            "proposed_label",
            "status",
            "k_present_for_label",
            "avg_state_frequency",
            "min_seed_alignment_cosine",
            "worst_leave_one_removed_abs_z_share",
        ],
    ].rename(columns={"status": "stability_status", "avg_state_frequency": "label_avg_state_frequency"})
    frame = posteriors.merge(state_meta, on=STATE_KEYS, how="left", validate="many_to_one")
    frame = frame.merge(stability_meta, on=["feature_set", "n_states", "proposed_label"], how="left", validate="many_to_one")
    frame["local_state_id"] = frame["local_state_id"].fillna(
        frame.apply(lambda row: _local_state_id(row["feature_set"], int(row["n_states"]), int(row["seed"]), int(row["fold"]), int(row["hmm_state"])), axis=1)
    )
    frame["proposed_label"] = frame["proposed_label"].fillna("uninterpretable_noise")
    frame["stability_status"] = frame["stability_status"].fillna("not_in_stability_grid")
    return frame


def attach_forward_returns(posteriors: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    merged = posteriors.merge(forward_returns, on=MERGE_KEYS, how="inner", validate="many_to_many")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged


def filter_posteriors_for_economics(posteriors: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _economics_cfg(config)
    selected = posteriors.copy()
    feature_sets = cfg.get("selected_feature_sets")
    n_states = cfg.get("selected_n_states")
    seeds = cfg.get("selected_seeds")
    if feature_sets:
        selected = selected[selected["feature_set"].isin([str(value) for value in feature_sets])].copy()
    if n_states:
        selected = selected[selected["n_states"].astype(int).isin([int(value) for value in n_states])].copy()
    if seeds:
        selected = selected[selected["seed"].astype(int).isin([int(value) for value in seeds])].copy()
    if selected.empty:
        raise ValueError("No posterior rows match hmm_state_economics_cross_asset selection filters")
    return selected


def filter_posteriors_to_stable_combos(
    posteriors: pd.DataFrame,
    state_names: pd.DataFrame,
    stability_grid: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    cfg = _economics_cfg(config)
    include_statuses = set(str(value) for value in cfg.get("include_stability_statuses", ["stable_profile_candidate"]))
    stable_labels = stability_grid[stability_grid["status"].isin(include_statuses)].loc[:, ["feature_set", "n_states", "proposed_label"]].drop_duplicates()
    if stable_labels.empty:
        return posteriors.iloc[0:0].copy()
    stable_states = state_names.merge(stable_labels, on=["feature_set", "n_states", "proposed_label"], how="inner")
    combo_keys = stable_states.loc[:, ["feature_set", "n_states", "seed", "fold"]].drop_duplicates()
    if combo_keys.empty:
        return posteriors.iloc[0:0].copy()
    return posteriors.merge(combo_keys, on=["feature_set", "n_states", "seed", "fold"], how="inner")


def _position_for_action(frame: pd.DataFrame, action: str) -> pd.Series:
    if action == "long":
        return pd.Series(1, index=frame.index, dtype="int64")
    if action == "short":
        return pd.Series(-1, index=frame.index, dtype="int64")
    if action == "flat":
        return pd.Series(0, index=frame.index, dtype="int64")
    if action in {"momentum", "reversion"}:
        direction = np.sign(frame["target_ret_3"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        position = pd.Series(direction.astype(int), index=frame.index)
        return position if action == "momentum" else -position
    raise ValueError(f"Unsupported action: {action}")


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


def evaluate_action_metrics(frame: pd.DataFrame, action: str, cost_bps: float) -> dict[str, float | int]:
    position = _position_for_action(frame, action)
    active = position != 0
    gross = position * frame["fwd_ret"].astype(float)
    cost = position.abs() * (float(cost_bps) / 10_000.0)
    net = gross - cost
    active_gross = gross[active]
    active_net = net[active]
    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "net_return": float(net.sum()),
        "mean_fwd_ret": float(frame["fwd_ret"].mean()) if len(frame) else np.nan,
        "median_fwd_ret": float(frame["fwd_ret"].median()) if len(frame) else np.nan,
        "avg_trade_gross": float(active_gross.mean()) if len(active_gross) else 0.0,
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "median_trade_net": float(active_net.median()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else np.nan,
        "skew": float(active_net.skew()) if len(active_net) >= 3 else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": _max_drawdown(net),
        "top_hour_pct": float(frame["hour"].value_counts(normalize=True).iloc[0]) if len(frame) else np.nan,
        "top_session_pct": float(frame["session"].value_counts(normalize=True).iloc[0]) if len(frame) else np.nan,
        "max_daily_abs_net_share": _max_daily_abs_net_share(frame, net),
    }


def summarize_forward_returns(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "split",
        "hmm_state",
        "local_state_id",
        "proposed_label",
        "stability_status",
        "horizon_bars",
    ]
    for key, group in merged.groupby(group_cols, sort=False):
        fwd = group["fwd_ret"].astype(float)
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "rows": int(len(group)),
                "mean_fwd_ret": float(fwd.mean()),
                "median_fwd_ret": float(fwd.median()),
                "p10_fwd_ret": float(fwd.quantile(0.10)),
                "p90_fwd_ret": float(fwd.quantile(0.90)),
                "hit_rate_long": float((fwd > 0).mean()),
                "skew_fwd_ret": float(fwd.skew()) if len(fwd) >= 3 else np.nan,
                "top_hour_pct": float(group["hour"].value_counts(normalize=True).iloc[0]),
                "top_session_pct": float(group["session"].value_counts(normalize=True).iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _base_metadata(state_frame: pd.DataFrame, horizon: int, cost_bps: float, action: str, bucket: str) -> dict[str, Any]:
    first = state_frame.iloc[0]
    return {
        "feature_set": first["feature_set"],
        "n_states": int(first["n_states"]),
        "seed": int(first["seed"]),
        "fold": int(first["fold"]),
        "split": first["split"],
        "hmm_state": int(first["hmm_state"]),
        "local_state_id": first["local_state_id"],
        "proposed_label": first["proposed_label"],
        "stability_status": first["stability_status"],
        "horizon_bars": int(horizon),
        "cost_bps": float(cost_bps),
        "action": action,
        "bucket": bucket,
    }


def build_economic_diagnostics(state_merged: pd.DataFrame, control_merged: pd.DataFrame, actions: list[str], costs: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_group_cols = ["feature_set", "n_states", "seed", "fold", "split", "horizon_bars"]
    state_group_cols = [*split_group_cols, "hmm_state"]
    split_lookup = {key: frame for key, frame in control_merged.groupby(split_group_cols, sort=False)}
    unconditional_cache: dict[tuple[Any, ...], dict[str, float | int]] = {}

    for state_key, state_frame in state_merged.groupby(state_group_cols, sort=False):
        if state_frame.empty:
            continue
        split_key = state_key[: len(split_group_cols)]
        split_frame = split_lookup.get(split_key)
        if split_frame is None or split_frame.empty:
            continue
        state_hours = set(state_frame["hour"].unique().tolist())
        same_hours = split_frame[split_frame["hour"].isin(state_hours)].copy()
        same_hours_ex_state = same_hours[same_hours["hmm_state"] != int(state_frame["hmm_state"].iloc[0])].copy()
        horizon = int(state_frame["horizon_bars"].iloc[0])
        for cost_bps in costs:
            for action in actions:
                for bucket, bucket_frame in (
                    ("state", state_frame),
                    ("unconditional", split_frame),
                    ("same_hour_all_states", same_hours),
                    ("same_hour_ex_state", same_hours_ex_state),
                ):
                    metadata = _base_metadata(state_frame, horizon, cost_bps, action, bucket)
                    if bucket == "unconditional":
                        cache_key = (*split_key, float(cost_bps), action)
                        metrics = unconditional_cache.get(cache_key)
                        if metrics is None:
                            metrics = evaluate_action_metrics(bucket_frame, action, cost_bps)
                            unconditional_cache[cache_key] = metrics
                    else:
                        metrics = evaluate_action_metrics(bucket_frame, action, cost_bps)
                    rows.append({**metadata, **metrics})
    diagnostics = pd.DataFrame(rows)
    return add_control_deltas(diagnostics)


def add_control_deltas(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    key_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "split",
        "hmm_state",
        "horizon_bars",
        "cost_bps",
        "action",
    ]
    state = diagnostics[diagnostics["bucket"] == "state"].copy()
    controls = diagnostics[diagnostics["bucket"].isin(["unconditional", "same_hour_ex_state"])].copy()
    control_metrics = ["avg_trade_net", "net_return", "profit_factor", "daily_sharpe"]
    for bucket in ["unconditional", "same_hour_ex_state"]:
        renamed = controls[controls["bucket"] == bucket].loc[:, [*key_cols, *control_metrics]].rename(
            columns={metric: f"{bucket}_{metric}" for metric in control_metrics}
        )
        state = state.merge(renamed, on=key_cols, how="left", validate="one_to_one")
    state["avg_trade_net_vs_unconditional"] = state["avg_trade_net"] - state["unconditional_avg_trade_net"]
    state["avg_trade_net_vs_same_hour_ex_state"] = state["avg_trade_net"] - state["same_hour_ex_state_avg_trade_net"]
    state["net_return_vs_unconditional"] = state["net_return"] - state["unconditional_net_return"]
    state["net_return_vs_same_hour_ex_state"] = state["net_return"] - state["same_hour_ex_state_net_return"]

    enriched = diagnostics.merge(
        state.loc[
            :,
            [
                *key_cols,
                "avg_trade_net_vs_unconditional",
                "avg_trade_net_vs_same_hour_ex_state",
                "net_return_vs_unconditional",
                "net_return_vs_same_hour_ex_state",
            ],
        ],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    return enriched


def classify_economic_row(row: pd.Series, config: dict[str, Any]) -> str:
    cfg = _economics_cfg(config)
    if row["bucket"] != "state":
        return "control"
    if row["action"] == "flat":
        return "flat_reference"
    if row.get("stability_status") != "stable_profile_candidate":
        return "rejected_unstable_profile"
    if int(row["trades"]) < int(cfg.get("min_trades", 50)):
        return "rejected_insufficient_trades"
    if row["net_return"] <= 0 or row["avg_trade_net"] <= 0:
        return "rejected_negative_net"
    if row["profit_factor"] <= float(cfg.get("min_profit_factor", 1.10)):
        return "rejected_weak_profit_factor"
    if row["daily_sharpe"] <= float(cfg.get("min_daily_sharpe", 1.0)):
        return "rejected_weak_sharpe"
    if bool(cfg.get("require_same_hour_improvement", True)) and row["avg_trade_net_vs_same_hour_ex_state"] <= 0:
        return "rejected_no_same_hour_edge"
    if row["top_hour_pct"] > float(cfg.get("max_top_hour_pct", 0.35)):
        return "rejected_hour_concentration"
    if row["top_session_pct"] > float(cfg.get("max_top_session_pct", 0.10)):
        return "rejected_session_concentration"
    if not pd.isna(row["max_daily_abs_net_share"]) and row["max_daily_abs_net_share"] > float(cfg.get("max_daily_abs_net_share", 0.50)):
        return "rejected_extreme_day_concentration"
    return "economic_candidate"


def add_economic_status(diagnostics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    diagnostics = diagnostics.copy()
    diagnostics["economic_status"] = diagnostics.apply(lambda row: classify_economic_row(row, config), axis=1)
    return diagnostics


def select_validation_candidates(diagnostics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split = str(_economics_cfg(config).get("candidate_split", "validation"))
    candidates = diagnostics[
        (diagnostics["bucket"] == "state")
        & (diagnostics["split"] == split)
        & (diagnostics["economic_status"] == "economic_candidate")
    ].copy()
    return candidates.sort_values(["avg_trade_net", "profit_factor", "daily_sharpe"], ascending=[False, False, False]).reset_index(drop=True)


def candidate_holdout(candidates: pd.DataFrame, diagnostics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    holdout_split = str(_economics_cfg(config).get("holdout_split", "test"))
    keys = ["feature_set", "n_states", "seed", "fold", "hmm_state", "horizon_bars", "cost_bps", "action"]
    selected_keys = candidates.loc[:, keys].drop_duplicates()
    holdout = diagnostics[(diagnostics["bucket"] == "state") & (diagnostics["split"] == holdout_split)].merge(selected_keys, on=keys, how="inner")
    return holdout.sort_values(["avg_trade_net", "profit_factor", "daily_sharpe"], ascending=[False, False, False]).reset_index(drop=True)


def non_operable_stable_states(diagnostics: pd.DataFrame) -> pd.DataFrame:
    state_rows = diagnostics[(diagnostics["bucket"] == "state") & (diagnostics["split"] == "validation")].copy()
    stable = state_rows[state_rows["stability_status"] == "stable_profile_candidate"].copy()
    if stable.empty:
        return pd.DataFrame()
    grouped = (
        stable.groupby(["feature_set", "n_states", "seed", "fold", "hmm_state", "local_state_id", "proposed_label"], as_index=False)
        .agg(
            statuses=("economic_status", lambda values: ",".join(sorted(set(str(value) for value in values)))),
            candidate_actions=("economic_status", lambda values: int((values == "economic_candidate").sum())),
            best_avg_trade_net=("avg_trade_net", "max"),
            best_profit_factor=("profit_factor", "max"),
            best_daily_sharpe=("daily_sharpe", "max"),
        )
    )
    return grouped[grouped["candidate_actions"] == 0].reset_index(drop=True)


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    diagnostics: pd.DataFrame,
    forward_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    holdout: pd.DataFrame,
    non_operable: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _economics_cfg(config)
    state_rows = diagnostics[diagnostics["bucket"] == "state"].copy()
    status_counts = state_rows["economic_status"].value_counts().rename_axis("economic_status").reset_index(name="rows") if not state_rows.empty else pd.DataFrame()
    validation = state_rows[state_rows["split"] == str(cfg.get("candidate_split", "validation"))].copy()
    top_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "hmm_state",
        "proposed_label",
        "horizon_bars",
        "cost_bps",
        "action",
        "trades",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "avg_trade_net_vs_same_hour_ex_state",
        "top_hour_pct",
        "top_session_pct",
        "max_daily_abs_net_share",
        "economic_status",
    ]
    fwd_cols = [
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "hmm_state",
        "proposed_label",
        "stability_status",
        "horizon_bars",
        "rows",
        "mean_fwd_ret",
        "median_fwd_ret",
        "hit_rate_long",
        "top_hour_pct",
    ]
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "Validation contains state/action economic candidates. Treat them as hypotheses for simple rule construction; test rows are holdout diagnostics only."
        if not candidates.empty
        else "No state/action passed the configured validation economic gates after costs and same-hour controls."
    )
    return f"""# HMM State Economics Cross-Asset - {target_symbol.upper()}

## Scope

- Feature version: `{_feature_set_version(config)}`
- Horizons: `{cfg.get("horizons")}`
- Costs bps: `{cfg.get("cost_bps")}`
- Actions: `{cfg.get("actions")}`
- Selected feature sets: `{cfg.get("selected_feature_sets", "all")}`
- Selected K: `{cfg.get("selected_n_states", "all")}`
- Selected seeds: `{cfg.get("selected_seeds", "all")}`
- Candidate split: `{cfg.get("candidate_split", "validation")}`
- Holdout split: `{cfg.get("holdout_split", "test")}`
- Test selection used: `no`

## Economic Status Counts

{_markdown_table(status_counts)}

## Validation Candidates

{_markdown_table(candidates.loc[:, [column for column in top_cols if column in candidates.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Candidate Holdout Sanity

{_markdown_table(holdout.loc[:, [column for column in top_cols if column in holdout.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Top Validation Diagnostics

{_markdown_table(validation.sort_values(["avg_trade_net", "profit_factor"], ascending=[False, False]).loc[:, [column for column in top_cols if column in validation.columns]], max_rows=int(cfg.get("report_top_rows", 40)))}

## Forward Return Summary Sample

{_markdown_table(forward_summary.sort_values(["mean_fwd_ret"], ascending=False).loc[:, [column for column in fwd_cols if column in forward_summary.columns]], max_rows=30)}

## Stable But Non-Operable States

{_markdown_table(non_operable, max_rows=40)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    cfg = _economics_cfg(config)
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    results_dir = results_output_dir(config, target)

    features = pd.read_parquet(features_input_path(config, target, feature_config))
    state_names = pd.read_parquet(results_dir / "state_name_grid.parquet")
    stability_grid = pd.read_parquet(results_dir / "state_stability_grid.parquet")
    posteriors = filter_posteriors_for_economics(pd.read_parquet(results_dir / "hmm_feature_lab_cross_asset_posteriors.parquet"), config)
    posteriors = filter_posteriors_to_stable_combos(posteriors, state_names, stability_grid, config)
    if posteriors.empty:
        raise ValueError("No posterior rows remain after stable-combo filtering")

    horizons = [int(value) for value in cfg.get("horizons", [1, 3, 6, 12])]
    costs = [float(value) for value in cfg.get("cost_bps", [1.0, 2.0, 5.0])]
    actions = [str(value) for value in cfg.get("actions", ACTIONS)]
    forward_returns = build_forward_returns(features, horizons)
    enriched_posteriors = enrich_posteriors_with_state_metadata(posteriors, state_names, stability_grid)
    merged = attach_forward_returns(enriched_posteriors, forward_returns)
    include_statuses = set(str(value) for value in cfg.get("include_stability_statuses", ["stable_profile_candidate"]))
    state_merged = merged[merged["stability_status"].isin(include_statuses)].copy()
    forward_summary = summarize_forward_returns(state_merged)
    diagnostics = add_economic_status(build_economic_diagnostics(state_merged, merged, actions, costs), config)
    candidates = select_validation_candidates(diagnostics, config)
    holdout = candidate_holdout(candidates, diagnostics, config)
    non_operable = non_operable_stable_states(diagnostics)

    outputs = {
        "state_forward_returns": results_dir / "state_forward_returns.parquet",
        "state_economic_diagnostics": results_dir / "state_economic_diagnostics.parquet",
        "state_economic_candidates": results_dir / "state_economic_candidates.parquet",
        "state_economic_candidate_holdout": results_dir / "state_economic_candidate_holdout.parquet",
        "state_non_operable": results_dir / "state_non_operable.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    forward_summary.to_parquet(outputs["state_forward_returns"], index=False)
    diagnostics.to_parquet(outputs["state_economic_diagnostics"], index=False)
    candidates.to_parquet(outputs["state_economic_candidates"], index=False)
    holdout.to_parquet(outputs["state_economic_candidate_holdout"], index=False)
    non_operable.to_parquet(outputs["state_non_operable"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, diagnostics, forward_summary, candidates, holdout, non_operable, outputs), encoding="utf-8")
    return report_path, outputs["state_economic_diagnostics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose forward-return economics for cross-asset HMM states.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, diagnostics_path = run(args.config, args.target)
    print(f"HMM state economics report written to: {report_path}")
    print(f"State economic diagnostics written to: {diagnostics_path}")


if __name__ == "__main__":
    main()

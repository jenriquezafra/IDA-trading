from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _lab_cfg, _target_symbol, features_input_path, load_yaml, results_output_dir
from src.hmm_risk_filter import (
    base_position,
    build_filter_dataset,
    filter_multiplier,
    load_candidate_combos,
    split_combo_frame,
    thresholds_for_strategy,
)
from src.hmm_state_interpretability_cross_asset import _markdown_table


KEY_COLS = ["feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "threshold"]
DEFAULT_IBKR_SOURCE_URL = "https://brokerage.ibkr.com/en/pricing/commissions-stocks.php"


def _sensitivity_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("candidate_cost_sensitivity_cross_asset", {})


def _path_from_template(template: str, target_symbol: str) -> Path:
    return Path(template.format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return (
        Path(config.get("paths", {}).get("reports_dir", "reports"))
        / target_symbol.upper()
        / "candidate_cost_sensitivity_cross_asset.md"
    )


def _profit_factor(active_net: pd.Series) -> float:
    gross_profit = active_net[active_net > 0].sum()
    gross_loss = -active_net[active_net < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def max_drawdown(net: pd.Series) -> float:
    if net.empty:
        return 0.0
    equity = net.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def max_drawdown_duration(values: pd.Series) -> int:
    equity = values.fillna(0.0).cumsum()
    if equity.empty:
        return 0
    underwater = (equity.cummax() - equity) > 1e-12
    longest = current = 0
    for flag in underwater:
        current = current + 1 if bool(flag) else 0
        longest = max(longest, current)
    return int(longest)


def _top_abs_share(values: pd.Series) -> float:
    abs_values = values.abs()
    denom = abs_values.sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(abs_values.max() / denom)


def _worst_group(values: pd.Series) -> tuple[str, float]:
    if values.empty:
        return "", np.nan
    idx = values.idxmin()
    return str(idx), float(values.loc[idx])


def ibkr_order_cost_usd(
    trade_value_usd: float,
    price_usd: float,
    plan: str,
    ibkr_cfg: dict[str, Any],
    *,
    is_sell: bool,
) -> float:
    if trade_value_usd <= 0 or price_usd <= 0:
        return 0.0
    shares = trade_value_usd / price_usd
    if plan == "fixed":
        commission = max(
            shares * float(ibkr_cfg.get("fixed_commission_per_share_usd", 0.005)),
            float(ibkr_cfg.get("fixed_min_commission_per_order_usd", 1.0)),
        )
    elif plan == "tiered":
        commission = max(
            shares * float(ibkr_cfg.get("tiered_commission_per_share_usd", 0.0035)),
            float(ibkr_cfg.get("tiered_min_commission_per_order_usd", 0.35)),
        )
        commission += shares * float(ibkr_cfg.get("tiered_clearing_per_share_per_side_usd", 0.00020))
    else:
        raise ValueError(f"Unsupported IBKR plan: {plan}")

    commission = min(commission, trade_value_usd * float(ibkr_cfg.get("max_commission_pct_trade_value", 0.01)))
    if is_sell:
        sec_fee = trade_value_usd * float(ibkr_cfg.get("sec_fee_rate_on_sell", 0.0000206))
        taf_fee = min(
            shares * float(ibkr_cfg.get("finra_taf_per_share_on_sell_usd", 0.000195)),
            float(ibkr_cfg.get("finra_taf_cap_usd", 9.79)),
        )
        commission += sec_fee + taf_fee
    return float(commission)


def cost_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = _sensitivity_cfg(config)
    scenarios: list[dict[str, Any]] = []
    for cost in cfg.get("cost_bps", [1.0, 2.0, 5.0]):
        cost_value = float(cost)
        scenarios.append({"cost_scenario": f"bps_{cost_value:g}", "cost_kind": "bps", "cost_bps": cost_value})

    ibkr_cfg = cfg.get("ibkr", {})
    if ibkr_cfg.get("enabled", True):
        for plan in ibkr_cfg.get("plans", ["tiered"]):
            for notional in ibkr_cfg.get("notionals_usd", [5000, 10000, 25000]):
                notional_value = float(notional)
                scenarios.append(
                    {
                        "cost_scenario": f"ibkr_{plan}_{notional_value:g}",
                        "cost_kind": "ibkr",
                        "ibkr_plan": str(plan),
                        "notional_usd": notional_value,
                        "ibkr": ibkr_cfg,
                    }
                )
    return scenarios


def _bps_cost_return(position: pd.Series, cost_bps: float) -> pd.Series:
    return position.abs() * (float(cost_bps) / 10_000.0)


def _ibkr_cost_return(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> pd.Series:
    ibkr_cfg = scenario["ibkr"]
    full_notional = float(scenario["notional_usd"])
    prices = frame.get("target_open_next", pd.Series(np.nan, index=frame.index)).astype(float)
    spread_slippage = float(ibkr_cfg.get("spread_slippage_bps_round_trip", 1.0))
    size = position.astype(float).abs().to_numpy()
    price = prices.to_numpy()
    active = (size > 0) & np.isfinite(price) & (price > 0)
    costs = np.zeros(len(frame), dtype=float)
    if not active.any():
        return pd.Series(costs, index=frame.index, dtype=float)

    trade_value = full_notional * size[active]
    shares = trade_value / price[active]
    cap = trade_value * float(ibkr_cfg.get("max_commission_pct_trade_value", 0.01))
    plan = str(scenario["ibkr_plan"])
    if plan == "fixed":
        side_commission = np.maximum(
            shares * float(ibkr_cfg.get("fixed_commission_per_share_usd", 0.005)),
            float(ibkr_cfg.get("fixed_min_commission_per_order_usd", 1.0)),
        )
    elif plan == "tiered":
        side_commission = np.maximum(
            shares * float(ibkr_cfg.get("tiered_commission_per_share_usd", 0.0035)),
            float(ibkr_cfg.get("tiered_min_commission_per_order_usd", 0.35)),
        )
        side_commission = side_commission + shares * float(ibkr_cfg.get("tiered_clearing_per_share_per_side_usd", 0.00020))
    else:
        raise ValueError(f"Unsupported IBKR plan: {plan}")
    side_commission = np.minimum(side_commission, cap)
    sec_fee = trade_value * float(ibkr_cfg.get("sec_fee_rate_on_sell", 0.0000206))
    taf_fee = np.minimum(
        shares * float(ibkr_cfg.get("finra_taf_per_share_on_sell_usd", 0.000195)),
        float(ibkr_cfg.get("finra_taf_cap_usd", 9.79)),
    )
    execution_cost = trade_value * spread_slippage / 10_000.0
    costs[active] = ((2.0 * side_commission) + sec_fee + taf_fee + execution_cost) / full_notional
    return pd.Series(costs, index=frame.index, dtype=float)


def scenario_cost_return(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> pd.Series:
    if scenario["cost_kind"] == "bps":
        return _bps_cost_return(position, float(scenario["cost_bps"]))
    if scenario["cost_kind"] == "ibkr":
        return _ibkr_cost_return(frame, position, scenario)
    raise ValueError(f"Unsupported cost kind: {scenario['cost_kind']}")


def evaluate_position_with_cost(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> dict[str, float | int | str]:
    position = position.astype(float).fillna(0.0)
    active = position.abs() > 0
    gross = position * frame["fwd_ret"].astype(float)
    cost = scenario_cost_return(frame, position, scenario)
    net = gross - cost
    active_net = net[active]
    sessions = int(frame["session"].nunique()) if "session" in frame else 0
    daily = net.groupby(frame["session"]).sum()
    monthly = net.groupby(pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m")).sum()
    hourly = net.groupby(frame["hour"]).sum()
    states = net.groupby(frame["proposed_label"].astype(str)).sum()
    worst_day, worst_day_net = _worst_group(daily)
    worst_month, worst_month_net = _worst_group(monthly)
    abs_position_sum = float(position.abs().sum())
    effective_cost_bps = float(cost.sum() / abs_position_sum * 10_000.0) if abs_position_sum > 0 else np.nan
    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "turnover": float(active.sum() / sessions) if sessions else 0.0,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "effective_cost_bps": effective_cost_bps,
        "net_return": float(net.sum()),
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": max_drawdown(net),
        "drawdown_duration_bars": max_drawdown_duration(net),
        "drawdown_duration_days": max_drawdown_duration(daily),
        "worst_day": worst_day,
        "worst_day_net": worst_day_net,
        "worst_month": worst_month,
        "worst_month_net": worst_month_net,
        "top_day_abs_net_share": _top_abs_share(daily),
        "top_month_abs_net_share": _top_abs_share(monthly),
        "top_hour_abs_net_share": _top_abs_share(hourly),
        "top_state_abs_net_share": _top_abs_share(states),
    }


def _candidate_id(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}__fold{int(row['fold'])}"
        f"__{row['strategy']}__{row['filter_name']}__h{int(row['horizon_bars'])}__thr{float(row['threshold']):g}"
    )


def _family_id(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['feature_set']}__{row['strategy']}__{row['filter_name']}__h{int(row['horizon_bars'])}__thr{float(row['threshold']):g}"


def load_validation_candidate_specs(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _sensitivity_cfg(config)
    selected_path = _path_from_template(str(cfg.get("selected_specs", "results/{target_symbol}/risk_filter_selected_specs.parquet")), target_symbol)
    validation_path = _path_from_template(str(cfg.get("validation_results", "results/{target_symbol}/risk_filter_validation.parquet")), target_symbol)
    specs = pd.read_parquet(selected_path)
    validation = pd.read_parquet(validation_path)
    candidates = validation[(validation["bucket"].eq("hmm_filter")) & (validation["filter_status"].eq("risk_filter_candidate"))].copy()
    if candidates.empty:
        return pd.DataFrame()
    metric_cols = [
        "filter_id",
        "net_return",
        "daily_sharpe",
        "profit_factor",
        "avg_trade_net",
        "max_drawdown",
        "drawdown_reduction_vs_base",
        "daily_sharpe_delta_vs_base",
        "daily_sharpe_delta_vs_same_hour",
    ]
    specs = specs[specs["filter_id"].isin(candidates["filter_id"])].merge(
        candidates.loc[:, [col for col in metric_cols if col in candidates.columns]].rename(
            columns={col: f"validation_selected_{col}" for col in metric_cols if col != "filter_id"}
        ),
        on="filter_id",
        how="left",
        validate="one_to_one",
    )
    specs["candidate_id"] = specs.apply(_candidate_id, axis=1)
    specs["family_id"] = specs.apply(_family_id, axis=1)
    specs = specs.drop_duplicates("filter_id").sort_values(
        ["validation_selected_daily_sharpe_delta_vs_base", "validation_selected_net_return"],
        ascending=[False, False],
        kind="stable",
    )
    specs = specs.drop_duplicates("candidate_id").copy()
    max_candidates = cfg.get("max_candidates")
    if max_candidates:
        specs = specs.head(int(max_candidates)).copy()
    return specs.reset_index(drop=True)


def _ensure_price_column(merged: pd.DataFrame, config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    if "target_open_next" in merged.columns or merged.empty:
        return merged
    feature_config = load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))
    features = pd.read_parquet(features_input_path(config, target_symbol, feature_config)).reset_index(names="source_index")
    cols = [col for col in ["source_index", "target_open_next"] if col in features.columns]
    if len(cols) < 2:
        return merged
    return merged.merge(features.loc[:, cols], on="source_index", how="left", validate="many_to_one")


def build_candidate_dataset(config: dict[str, Any], target_symbol: str, specs: pd.DataFrame) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    combo_cols = ["feature_set", "n_states", "seed", "fold"]
    combos = specs.loc[:, combo_cols].drop_duplicates().reset_index(drop=True)
    available_combos = load_candidate_combos(config, target_symbol)
    combos = combos.merge(available_combos, on=combo_cols, how="inner")
    return _ensure_price_column(build_filter_dataset(config, target_symbol, combos), config, target_symbol)


def threshold_variants(config: dict[str, Any], strategy: str, selected_threshold: float) -> pd.DataFrame:
    cfg = _sensitivity_cfg(config)
    selected = float(selected_threshold)
    values: dict[float, str] = {selected: "selected"}
    for value in thresholds_for_strategy(config, strategy):
        values[round(float(value), 12)] = "coarse_grid"
    for mult in cfg.get("threshold_multipliers", [0.5, 0.75, 1.0, 1.25, 1.5]):
        values[round(max(selected * float(mult), 0.0), 12)] = f"mult_{float(mult):g}"
    additive_cfg = cfg.get("threshold_additive", {})
    for delta in additive_cfg.get(strategy, additive_cfg.get("default", [])):
        values[round(max(selected + float(delta), 0.0), 12)] = f"delta_{float(delta):g}"
    return pd.DataFrame(
        [{"threshold": float(value), "threshold_variant": variant} for value, variant in sorted(values.items())]
    )


def _selected_hours(spec: pd.Series) -> tuple[int, ...]:
    return tuple(int(value) for value in str(spec.get("selected_hours", "")).split(",") if value != "")


def candidate_position(frame: pd.DataFrame, spec: pd.Series, threshold: float) -> pd.Series:
    return base_position(frame, str(spec["strategy"]), float(threshold)) * filter_multiplier(frame, str(spec["filter_name"]))


def evaluate_spec(
    merged: pd.DataFrame,
    spec: pd.Series,
    split: str,
    horizon: int,
    threshold: float,
    scenario: dict[str, Any],
    *,
    threshold_variant: str = "selected",
    horizon_variant: str = "selected",
) -> dict[str, Any] | None:
    frame = split_combo_frame(merged, spec, split, int(horizon))
    if frame.empty:
        return None
    position = candidate_position(frame, spec, float(threshold))
    metadata = {
        "filter_id": spec["filter_id"],
        "candidate_id": spec["candidate_id"],
        "family_id": spec["family_id"],
        "feature_set": spec["feature_set"],
        "n_states": int(spec["n_states"]),
        "seed": int(spec["seed"]),
        "fold": int(spec["fold"]),
        "split": split,
        "strategy": spec["strategy"],
        "filter_name": spec["filter_name"],
        "selected_horizon_bars": int(spec["horizon_bars"]),
        "horizon_bars": int(horizon),
        "horizon_variant": horizon_variant,
        "selected_threshold": float(spec["threshold"]),
        "threshold": float(threshold),
        "threshold_variant": threshold_variant,
        "selected_cost_bps": float(spec["cost_bps"]),
        "selected_hours": ",".join(str(hour) for hour in _selected_hours(spec)),
        "cost_scenario": scenario["cost_scenario"],
        "cost_kind": scenario["cost_kind"],
        "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
        "ibkr_plan": scenario.get("ibkr_plan", ""),
        "notional_usd": float(scenario.get("notional_usd", np.nan)),
    }
    return {**metadata, **evaluate_position_with_cost(frame, position, scenario)}


def build_sensitivity_tables(
    merged: pd.DataFrame,
    specs: pd.DataFrame,
    config: dict[str, Any],
    target_symbol: str | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = _sensitivity_cfg(config)
    splits = [str(cfg.get("candidate_split", "validation")), str(cfg.get("test_split", "test"))]
    scenarios = cost_scenarios(config)
    cost_rows: list[dict[str, Any]] = []

    for _, spec in specs.iterrows():
        for split in splits:
            for scenario in scenarios:
                row = evaluate_spec(merged, spec, split, int(spec["horizon_bars"]), float(spec["threshold"]), scenario)
                if row is not None:
                    cost_rows.append(row)
    cost_frame = pd.DataFrame(cost_rows)
    threshold_frame = build_threshold_sensitivity_from_validation_grid(specs, config, target_symbol)
    horizon_frame = build_horizon_sensitivity_from_validation_grid(specs, config, target_symbol)
    drawdown_cols = [
        "filter_id",
        "candidate_id",
        "family_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "split",
        "strategy",
        "filter_name",
        "horizon_bars",
        "threshold",
        "cost_scenario",
        "cost_kind",
        "configured_cost_bps",
        "effective_cost_bps",
        "notional_usd",
        "trades",
        "turnover",
        "net_return",
        "daily_sharpe",
        "profit_factor",
        "avg_trade_net",
        "max_drawdown",
        "drawdown_duration_bars",
        "drawdown_duration_days",
        "worst_day",
        "worst_day_net",
        "worst_month",
        "worst_month_net",
        "top_day_abs_net_share",
        "top_month_abs_net_share",
        "top_hour_abs_net_share",
        "top_state_abs_net_share",
    ]
    drawdown = cost_frame.loc[:, [col for col in drawdown_cols if col in cost_frame.columns]].copy() if not cost_frame.empty else pd.DataFrame()
    concentration = build_concentration_analysis(cost_frame)
    decisions = label_candidates(cost_frame, threshold_frame, horizon_frame, config)
    return {
        "candidate_cost_sensitivity": cost_frame,
        "candidate_threshold_sensitivity": threshold_frame,
        "candidate_horizon_sensitivity": horizon_frame,
        "candidate_drawdown_analysis": drawdown,
        "candidate_concentration_analysis": concentration,
        "candidate_decisions": decisions,
    }


def _validation_grid(config: dict[str, Any], target_symbol: str | None = None) -> pd.DataFrame:
    target = _target_symbol(config, target_symbol)
    cfg = _sensitivity_cfg(config)
    path = _path_from_template(str(cfg.get("threshold_grid", "results/{target_symbol}/risk_filter_threshold_grid.parquet")), target)
    grid = pd.read_parquet(path)
    grid = grid[(grid["bucket"].eq("hmm_filter")) & (grid["split"].eq(str(cfg.get("candidate_split", "validation"))))].copy()
    grid["cost_scenario"] = grid["cost_bps"].map(lambda value: f"bps_{float(value):g}")
    return grid


def build_threshold_sensitivity_from_validation_grid(specs: pd.DataFrame, config: dict[str, Any], target_symbol: str | None = None) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    grid = _validation_grid(config, target_symbol).drop(columns=["filter_id"], errors="ignore")
    keys = ["feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "cost_bps"]
    spec_cols = [
        "filter_id",
        "candidate_id",
        "family_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "horizon_bars",
        "cost_bps",
        "threshold",
    ]
    joined = specs.loc[:, spec_cols].rename(columns={"threshold": "selected_threshold"}).merge(grid, on=keys, how="left")
    joined = joined.dropna(subset=["threshold"]).copy()
    joined["threshold_variant"] = np.where(
        np.isclose(joined["threshold"], joined["selected_threshold"], rtol=0.0, atol=1e-12),
        "selected",
        "coarse_grid",
    )
    joined["horizon_variant"] = "selected"
    return joined.reset_index(drop=True)


def build_horizon_sensitivity_from_validation_grid(specs: pd.DataFrame, config: dict[str, Any], target_symbol: str | None = None) -> pd.DataFrame:
    if specs.empty:
        return pd.DataFrame()
    grid = _validation_grid(config, target_symbol).drop(columns=["filter_id"], errors="ignore")
    keys = ["feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "cost_bps", "threshold"]
    spec_cols = [
        "filter_id",
        "candidate_id",
        "family_id",
        "feature_set",
        "n_states",
        "seed",
        "fold",
        "strategy",
        "filter_name",
        "cost_bps",
        "horizon_bars",
        "threshold",
    ]
    joined = specs.loc[:, spec_cols].rename(columns={"horizon_bars": "selected_horizon_bars"}).merge(grid, on=keys, how="left")
    joined = joined.dropna(subset=["horizon_bars"]).copy()
    joined["horizon_variant"] = np.where(joined["horizon_bars"].astype(int).eq(joined["selected_horizon_bars"].astype(int)), "selected", "alternate")
    joined["selected_threshold"] = joined["threshold"]
    joined["threshold_variant"] = "selected"
    return joined.reset_index(drop=True)


def build_concentration_analysis(cost_frame: pd.DataFrame) -> pd.DataFrame:
    if cost_frame.empty:
        return pd.DataFrame()
    test = cost_frame[cost_frame["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in test.groupby(["family_id", "cost_scenario"], sort=False):
        family_id, cost_scenario = keys
        fold_net = group.groupby("filter_id")["net_return"].sum()
        worst_filter, worst_filter_net = _worst_group(fold_net)
        rows.append(
            {
                "family_id": family_id,
                "cost_scenario": cost_scenario,
                "candidate_rows": int(group["filter_id"].nunique()),
                "top_filter_abs_net_share": _top_abs_share(fold_net),
                "worst_filter_id": worst_filter,
                "worst_filter_net": worst_filter_net,
                "median_top_day_abs_net_share": float(group["top_day_abs_net_share"].median()),
                "median_top_hour_abs_net_share": float(group["top_hour_abs_net_share"].median()),
                "median_top_state_abs_net_share": float(group["top_state_abs_net_share"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cost_scenario", "top_filter_abs_net_share"], ascending=[True, False])


def _passes(row: pd.Series | None, cfg: dict[str, Any]) -> bool:
    if row is None or row.empty:
        return False
    top_day_share = row.get("top_day_abs_net_share", row.get("max_daily_abs_net_share", 0.0))
    return bool(
        row["net_return"] > 0
        and row["daily_sharpe"] >= float(cfg.get("min_daily_sharpe", 1.0))
        and row["profit_factor"] >= float(cfg.get("min_profit_factor", 1.10))
        and row["avg_trade_net"] > 0
        and row["max_drawdown"] <= float(cfg.get("max_drawdown", 0.20))
        and top_day_share <= float(cfg.get("max_top_day_abs_net_share", 0.40))
    )


def _row_for_scenario(cost_frame: pd.DataFrame, filter_id: str, scenario: str) -> pd.Series | None:
    rows = cost_frame[(cost_frame["filter_id"].eq(filter_id)) & (cost_frame["split"].eq("test")) & (cost_frame["cost_scenario"].eq(scenario))]
    if rows.empty:
        return None
    return rows.iloc[0]


def _pass_rate(frame: pd.DataFrame, filter_id: str, scenario: str, cfg: dict[str, Any]) -> float:
    rows = frame[(frame["filter_id"].eq(filter_id)) & (frame["cost_scenario"].eq(scenario))].copy()
    if rows.empty:
        return 0.0
    passes = rows.apply(lambda row: _passes(row, cfg), axis=1)
    return float(passes.mean())


def label_candidates(cost_frame: pd.DataFrame, threshold_frame: pd.DataFrame, horizon_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if cost_frame.empty:
        return pd.DataFrame()
    cfg = _sensitivity_cfg(config).get("acceptance", {})
    primary = str(cfg.get("primary_cost_scenario", "bps_1"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    ibkr_required = str(cfg.get("ibkr_required_scenario", "ibkr_tiered_10000"))
    min_threshold_pass_rate = float(cfg.get("min_threshold_pass_rate", 0.60))
    min_horizon_pass_rate = float(cfg.get("min_horizon_pass_rate", 0.50))
    rows = []
    candidates = cost_frame[cost_frame["split"].eq("test")].drop_duplicates("filter_id")
    for _, candidate in candidates.iterrows():
        filter_id = str(candidate["filter_id"])
        primary_row = _row_for_scenario(cost_frame, filter_id, primary)
        conservative_row = _row_for_scenario(cost_frame, filter_id, conservative)
        ibkr_row = _row_for_scenario(cost_frame, filter_id, ibkr_required)
        passes_primary = _passes(primary_row, cfg)
        passes_conservative = _passes(conservative_row, cfg)
        passes_ibkr = _passes(ibkr_row, cfg)
        threshold_pass_rate = _pass_rate(threshold_frame, filter_id, conservative, cfg)
        horizon_pass_rate = _pass_rate(horizon_frame, filter_id, conservative, cfg)
        threshold_ok = threshold_pass_rate >= min_threshold_pass_rate
        horizon_ok = horizon_pass_rate >= min_horizon_pass_rate
        if passes_primary and passes_conservative and passes_ibkr and threshold_ok and horizon_ok:
            label = "accepted"
        elif passes_primary and not passes_conservative:
            label = "cost-fragile"
        elif passes_conservative and (not threshold_ok or not horizon_ok):
            label = "unstable"
        else:
            label = "rejected"
        rows.append(
            {
                "filter_id": filter_id,
                "candidate_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "feature_set": candidate["feature_set"],
                "n_states": int(candidate["n_states"]),
                "seed": int(candidate["seed"]),
                "fold": int(candidate["fold"]),
                "strategy": candidate["strategy"],
                "filter_name": candidate["filter_name"],
                "horizon_bars": int(candidate["horizon_bars"]),
                "threshold": float(candidate["threshold"]),
                "primary_cost_scenario": primary,
                "conservative_cost_scenario": conservative,
                "ibkr_required_scenario": ibkr_required,
                "passes_primary": passes_primary,
                "passes_conservative": passes_conservative,
                "passes_ibkr": passes_ibkr,
                "threshold_pass_rate": threshold_pass_rate,
                "horizon_pass_rate": horizon_pass_rate,
                "decision_label": label,
                "test_net_primary": float(primary_row["net_return"]) if primary_row is not None else np.nan,
                "test_net_conservative": float(conservative_row["net_return"]) if conservative_row is not None else np.nan,
                "test_net_ibkr": float(ibkr_row["net_return"]) if ibkr_row is not None else np.nan,
                "test_sharpe_conservative": float(conservative_row["daily_sharpe"]) if conservative_row is not None else np.nan,
                "test_profit_factor_conservative": float(conservative_row["profit_factor"]) if conservative_row is not None else np.nan,
                "test_max_drawdown_conservative": float(conservative_row["max_drawdown"]) if conservative_row is not None else np.nan,
                "test_effective_cost_bps_ibkr": float(ibkr_row["effective_cost_bps"]) if ibkr_row is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision_label", "test_sharpe_conservative", "test_net_conservative"],
        ascending=[True, False, False],
        kind="stable",
    )


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    specs: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    outputs: dict[str, Path],
) -> str:
    cfg = _sensitivity_cfg(config)
    decisions = tables["candidate_decisions"]
    cost_frame = tables["candidate_cost_sensitivity"]
    concentration = tables["candidate_concentration_analysis"]
    decision_counts = decisions["decision_label"].value_counts().rename_axis("decision_label").reset_index(name="rows") if not decisions.empty else pd.DataFrame()
    top_decisions = decisions.head(int(cfg.get("report_top_rows", 40))) if not decisions.empty else pd.DataFrame()
    cost_summary = (
        cost_frame[cost_frame["split"].eq("test")]
        .groupby(["cost_scenario"], as_index=False)
        .agg(
            candidate_rows=("filter_id", "nunique"),
            median_net_return=("net_return", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_effective_cost_bps=("effective_cost_bps", "median"),
            positive_net_rate=("net_return", lambda values: float((values > 0).mean())),
        )
        .sort_values("cost_scenario")
        if not cost_frame.empty
        else pd.DataFrame()
    )
    ibkr_cfg = cfg.get("ibkr", {})
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = (
        "At least one validation-selected candidate passes the configured cost, drawdown, threshold and horizon sensitivity gates."
        if not decisions.empty and decisions["decision_label"].eq("accepted").any()
        else "No candidate is accepted under the configured IBKR-aware sensitivity gates; promote only as research candidates."
    )
    source_url = ibkr_cfg.get("source_url", DEFAULT_IBKR_SOURCE_URL)
    return f"""# Candidate Cost Sensitivity Cross-Asset - {target_symbol.upper()}

## Scope

- Candidate source: validation-selected HMM risk filters from `results/{{target_symbol}}/risk_filter_selected_specs.parquet`.
- Validation candidates evaluated: `{len(specs)}`
- Abstract round-trip costs bps: `{cfg.get("cost_bps", [1.0, 2.0, 5.0])}`
- IBKR enabled: `{ibkr_cfg.get("enabled", True)}`
- IBKR plans: `{ibkr_cfg.get("plans", ["tiered"])}`
- IBKR notionals USD: `{ibkr_cfg.get("notionals_usd", [5000, 10000, 25000])}`
- IBKR source: `{source_url}`
- Threshold selection: frozen from validation; test is diagnostic only.

## IBKR Cost Assumptions

- US stocks/ETFs, IBKR Pro Tiered first tier: USD 0.0035/share, minimum USD 0.35/order, maximum 1% of trade value.
- US stocks/ETFs, IBKR Pro Fixed: USD 0.005/share, minimum USD 1.00/order, maximum 1% of trade value.
- Tiered scenario includes configured clearing pass-through and sell-side regulatory fees.
- Execution friction adds `{ibkr_cfg.get("spread_slippage_bps_round_trip", 1.0)}` bps round-trip for spread/slippage.
- These are research assumptions, not a substitute for live order estimates.

## Decision Counts

{_markdown_table(decision_counts)}

## Test Cost Summary

{_markdown_table(cost_summary, max_rows=80)}

## Top Candidate Decisions

{_markdown_table(top_decisions, max_rows=int(cfg.get("report_top_rows", 40)))}

## Concentration Summary

{_markdown_table(concentration.head(40) if not concentration.empty else concentration, max_rows=40)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    specs = load_validation_candidate_specs(config, target)
    merged = build_candidate_dataset(config, target, specs) if not specs.empty else pd.DataFrame()
    tables = build_sensitivity_tables(merged, specs, config, target) if not merged.empty else {
        "candidate_cost_sensitivity": pd.DataFrame(),
        "candidate_threshold_sensitivity": pd.DataFrame(),
        "candidate_horizon_sensitivity": pd.DataFrame(),
        "candidate_drawdown_analysis": pd.DataFrame(),
        "candidate_concentration_analysis": pd.DataFrame(),
        "candidate_decisions": pd.DataFrame(),
    }
    outputs = {
        "candidate_cost_sensitivity": results_dir / "candidate_cost_sensitivity.parquet",
        "candidate_threshold_sensitivity": results_dir / "candidate_threshold_sensitivity.parquet",
        "candidate_horizon_sensitivity": results_dir / "candidate_horizon_sensitivity.parquet",
        "candidate_drawdown_analysis": results_dir / "candidate_drawdown_analysis.parquet",
        "candidate_concentration_analysis": results_dir / "candidate_concentration_analysis.parquet",
        "candidate_decisions": results_dir / "candidate_decisions.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(outputs[name], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, specs, tables, outputs), encoding="utf-8")
    return report_path, outputs["candidate_decisions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-asset HMM candidates under cost, threshold and drawdown stress.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, decisions_path = run(args.config, args.target)
    print(f"Candidate cost sensitivity report written to: {report_path}")
    print(f"Candidate decisions written to: {decisions_path}")


if __name__ == "__main__":
    main()

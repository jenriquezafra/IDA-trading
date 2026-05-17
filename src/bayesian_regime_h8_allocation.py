from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import bayesian_regime_h8
from src.hmm_lab import LabFold
from src.hmm_state_economics_cross_asset import build_forward_returns


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
MERGE_KEYS = ["source_index", "timestamp", "session", "bar_index"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    return bayesian_regime_h8.load_yaml(path)


def _alloc_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("h8_probability_allocation", {})


def _cost_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("candidate_cost_sensitivity_cross_asset", {})


def _target_symbol(config: dict[str, Any], target_symbol: str | None = None) -> str:
    return (target_symbol or config.get("lab", {}).get("target_symbol") or _alloc_cfg(config).get("target_symbol") or "QQQ").upper()


def results_dir(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _alloc_cfg(config)
    if cfg.get("results_dir"):
        return Path(str(cfg["results_dir"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("results_dir", "results")) / target_symbol.upper() / "h8_probability_allocation"


def report_path(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _alloc_cfg(config)
    if cfg.get("report_file"):
        return Path(str(cfg["report_file"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "h8_probability_allocation.md"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
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


def cost_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    cfg = _cost_cfg(config)
    scenarios: list[dict[str, Any]] = []
    for cost in cfg.get("cost_bps", [1.0, 2.0, 5.0]):
        cost_value = float(cost)
        scenarios.append({"cost_scenario": f"bps_{cost_value:g}", "cost_kind": "bps", "round_trip_bps": cost_value})

    ibkr_cfg = cfg.get("ibkr", {})
    if ibkr_cfg.get("enabled", False):
        for plan in ibkr_cfg.get("plans", ["tiered"]):
            for notional in ibkr_cfg.get("notionals_usd", [10_000]):
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
    if names is None:
        return scenarios
    wanted = {str(name) for name in names}
    return [scenario for scenario in scenarios if str(scenario["cost_scenario"]) in wanted]


def _base_target_position(frame: pd.DataFrame, method: str) -> pd.Series:
    bull = frame.get("p_bull_trend", pd.Series(0.0, index=frame.index)).astype(float).fillna(0.0)
    bear = frame.get("p_bear_stress", pd.Series(0.0, index=frame.index)).astype(float).fillna(0.0)
    edge = bull - bear
    if method == "edge":
        return edge.clip(-1.0, 1.0)
    if method == "confidence_edge":
        confidence = frame.get("max_prob", pd.Series(1.0, index=frame.index)).astype(float).fillna(0.0)
        return (edge * confidence).clip(-1.0, 1.0)
    if method == "dominant_probability":
        probability_cols = [column for column in frame.columns if column.startswith("p_") and column not in {"p_long_profit", "p_short_profit"}]
        dominant = frame.loc[:, probability_cols].idxmax(axis=1) if probability_cols else pd.Series("", index=frame.index)
        out = pd.Series(0.0, index=frame.index)
        out.loc[dominant.eq("p_bull_trend") & bull.gt(bear)] = bull.loc[dominant.eq("p_bull_trend") & bull.gt(bear)]
        out.loc[dominant.eq("p_bear_stress") & bear.gt(bull)] = -bear.loc[dominant.eq("p_bear_stress") & bear.gt(bull)]
        return out.clip(-1.0, 1.0)
    raise ValueError(f"Unsupported allocation method: {method}")


def target_position(
    frame: pd.DataFrame,
    *,
    method: str,
    min_abs_position: float = 0.0,
    max_entropy: float | None = None,
) -> pd.Series:
    target = _base_target_position(frame, method)
    if max_entropy is not None:
        target = target.where(frame["entropy"].astype(float) <= float(max_entropy), 0.0)
    floor = float(min_abs_position)
    if floor > 0.0:
        target = target.where(target.abs() >= floor, 0.0)
    return target.clip(-1.0, 1.0)


def executable_position(
    target: pd.Series,
    sessions: pd.Series,
    *,
    smoothing_alpha: float = 1.0,
    rebalance_threshold: float = 0.0,
) -> pd.Series:
    alpha = float(smoothing_alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    threshold = float(rebalance_threshold)
    output = pd.Series(0.0, index=target.index, dtype=float)
    for _, idx in sessions.groupby(sessions, sort=False).groups.items():
        prev = 0.0
        for label in idx:
            raw = float(target.loc[label])
            desired = raw if alpha == 1.0 or raw == 0.0 else prev + alpha * (raw - prev)
            if abs(desired - prev) < threshold and desired != 0.0:
                current = prev
            else:
                current = desired
            output.loc[label] = current
            prev = current
    return output.clip(-1.0, 1.0)


def turnover_series(position: pd.Series, sessions: pd.Series) -> tuple[pd.Series, pd.Series]:
    entry_turnover = pd.Series(0.0, index=position.index, dtype=float)
    exit_turnover = pd.Series(0.0, index=position.index, dtype=float)
    for _, idx in sessions.groupby(sessions, sort=False).groups.items():
        session_pos = position.loc[idx].astype(float)
        previous = session_pos.shift(1).fillna(0.0)
        entry_turnover.loc[idx] = (session_pos - previous).abs()
        if len(session_pos):
            exit_turnover.loc[session_pos.index[-1]] = abs(float(session_pos.iloc[-1]))
    return entry_turnover, exit_turnover


def _one_way_ibkr_cost_return(price: pd.Series, signed_delta: pd.Series, scenario: dict[str, Any]) -> pd.Series:
    ibkr_cfg = scenario["ibkr"]
    full_notional = float(scenario["notional_usd"])
    delta = signed_delta.astype(float).fillna(0.0)
    px = price.astype(float).replace([np.inf, -np.inf], np.nan)
    active = delta.abs().gt(0.0) & px.gt(0.0)
    costs = pd.Series(0.0, index=delta.index, dtype=float)
    if not active.any():
        return costs

    trade_value = full_notional * delta.loc[active].abs()
    shares = trade_value / px.loc[active]
    plan = str(scenario["ibkr_plan"])
    if plan == "fixed":
        commission = np.maximum(
            shares * float(ibkr_cfg.get("fixed_commission_per_share_usd", 0.005)),
            float(ibkr_cfg.get("fixed_min_commission_per_order_usd", 1.0)),
        )
    elif plan == "tiered":
        commission = np.maximum(
            shares * float(ibkr_cfg.get("tiered_commission_per_share_usd", 0.0035)),
            float(ibkr_cfg.get("tiered_min_commission_per_order_usd", 0.35)),
        )
        commission = commission + shares * float(ibkr_cfg.get("tiered_clearing_per_share_per_side_usd", 0.00020))
    else:
        raise ValueError(f"Unsupported IBKR plan: {plan}")

    commission = np.minimum(commission, trade_value * float(ibkr_cfg.get("max_commission_pct_trade_value", 0.01)))
    sell = delta.loc[active].lt(0.0)
    sec_fee = trade_value * float(ibkr_cfg.get("sec_fee_rate_on_sell", 0.0000206))
    taf_fee = np.minimum(
        shares * float(ibkr_cfg.get("finra_taf_per_share_on_sell_usd", 0.000195)),
        float(ibkr_cfg.get("finra_taf_cap_usd", 9.79)),
    )
    sell_fees = pd.Series(0.0, index=trade_value.index, dtype=float)
    sell_fees.loc[sell] = sec_fee.loc[sell] + taf_fee.loc[sell]
    half_spread_slippage_bps = float(ibkr_cfg.get("spread_slippage_bps_round_trip", 0.0)) / 2.0
    execution_cost = trade_value * half_spread_slippage_bps / 10_000.0
    costs.loc[active] = (commission + sell_fees + execution_cost) / full_notional
    return costs


def cost_return(
    frame: pd.DataFrame,
    position: pd.Series,
    entry_turnover: pd.Series,
    exit_turnover: pd.Series,
    scenario: dict[str, Any],
) -> pd.Series:
    if scenario["cost_kind"] == "bps":
        one_way_cost = float(scenario["round_trip_bps"]) / 2.0 / 10_000.0
        return (entry_turnover + exit_turnover) * one_way_cost
    if scenario["cost_kind"] == "ibkr":
        previous = position.groupby(frame["session"], sort=False).shift(1).fillna(0.0)
        entry_delta = position.astype(float) - previous.astype(float)
        exit_delta = pd.Series(0.0, index=position.index, dtype=float)
        exit_delta.loc[exit_turnover.gt(0.0)] = -position.loc[exit_turnover.gt(0.0)].astype(float)
        entry_cost = _one_way_ibkr_cost_return(frame["entry_px"].astype(float), entry_delta, scenario)
        exit_cost = _one_way_ibkr_cost_return(frame["exit_px"].astype(float), exit_delta, scenario)
        return entry_cost + exit_cost
    raise ValueError(f"Unsupported cost kind: {scenario['cost_kind']}")


def evaluate_allocation_frame(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> dict[str, Any]:
    entry_turnover, exit_turnover = turnover_series(position, frame["session"])
    turnover = entry_turnover + exit_turnover
    gross = position.astype(float) * frame["fwd_ret"].astype(float)
    cost = cost_return(frame, position, entry_turnover, exit_turnover, scenario)
    net = gross - cost
    active = position.abs() > 1e-12
    active_net = net[active]
    sessions = int(frame["session"].nunique())
    turnover_sum = float(turnover.sum())
    daily = net.groupby(frame["session"]).sum()
    top_session_pct = np.nan
    abs_daily = daily.abs()
    if abs_daily.sum() > 0.0:
        top_session_pct = float(abs_daily.max() / abs_daily.sum())
    effective_round_trip_cost_bps = float(cost.sum() / turnover_sum * 2.0 * 10_000.0) if turnover_sum > 0.0 else np.nan
    return {
        "rows": int(len(frame)),
        "sessions": sessions,
        "active_bars": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "avg_abs_position": float(position.abs().mean()) if len(position) else 0.0,
        "avg_long_position": float(position[position > 0.0].mean()) if (position > 0.0).any() else 0.0,
        "avg_short_position": float(position[position < 0.0].mean()) if (position < 0.0).any() else 0.0,
        "long_bars": int((position > 0.0).sum()),
        "short_bars": int((position < 0.0).sum()),
        "rebalance_count": int(entry_turnover.gt(1e-12).sum() + exit_turnover.gt(1e-12).sum()),
        "turnover": turnover_sum,
        "turnover_per_session": turnover_sum / sessions if sessions else np.nan,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "effective_round_trip_cost_bps": effective_round_trip_cost_bps,
        "net_return": float(net.sum()),
        "net_per_turnover": float(net.sum() / turnover_sum) if turnover_sum > 0.0 else 0.0,
        "avg_active_bar_net": float(active_net.mean()) if len(active_net) else 0.0,
        "hit_rate_active_bar": float((active_net > 0.0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": _max_drawdown(net),
        "top_session_abs_net_share": top_session_pct,
    }


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0.0].sum()
    gross_loss = -active_net[active_net < 0.0].sum()
    if gross_loss == 0.0:
        return np.inf if gross_profit > 0.0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252.0) * daily.mean() / std)


def _max_drawdown(net: pd.Series) -> float:
    equity = net.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def prepare_allocation_dataset(features: pd.DataFrame, posteriors: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _alloc_cfg(config)
    horizons = [int(value) for value in cfg.get("holding_horizon_bars", [1])]
    forward_returns = build_forward_returns(features, horizons)
    merged = posteriors.merge(
        forward_returns,
        on=MERGE_KEYS,
        how="inner",
        validate="many_to_many",
    )
    merged["target_open_next"] = merged["entry_px"].astype(float)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged.sort_values(["variant", "fold", "split", "timestamp"], kind="stable").reset_index(drop=True)


def evaluate_probability_allocations(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _alloc_cfg(config)
    methods = [str(value) for value in cfg.get("allocation_methods", ["edge", "confidence_edge", "dominant_probability"])]
    min_abs_values = [float(value) for value in cfg.get("min_abs_positions", [0.0, 0.1, 0.2])]
    entropy_values = cfg.get("max_entropy_values", [None, 0.75])
    smoothing_values = [float(value) for value in cfg.get("smoothing_alphas", [1.0, 0.5])]
    rebalance_values = [float(value) for value in cfg.get("rebalance_thresholds", [0.0, 0.05])]
    eval_splits = {str(value) for value in cfg.get("eval_splits", ["validation", "test"])}
    scenario_names = cfg.get("grid_cost_scenarios", [str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))])
    scenarios = cost_scenarios(config, [str(value) for value in scenario_names])
    if not scenarios:
        scenarios = cost_scenarios(config)
    rows: list[dict[str, Any]] = []
    for keys, group in dataset.groupby(["variant", "fold", "split", "horizon_bars"], sort=False):
        variant, fold, split, horizon = keys
        if str(split) not in eval_splits:
            continue
        for method in methods:
            for min_abs in min_abs_values:
                for max_entropy in entropy_values:
                    target = target_position(group, method=method, min_abs_position=min_abs, max_entropy=max_entropy)
                    for smoothing_alpha in smoothing_values:
                        for rebalance_threshold in rebalance_values:
                            position = executable_position(
                                target,
                                group["session"],
                                smoothing_alpha=smoothing_alpha,
                                rebalance_threshold=rebalance_threshold,
                            )
                            for scenario in scenarios:
                                rows.append(
                                    {
                                        "variant": str(variant),
                                        "fold": int(fold),
                                        "split": str(split),
                                        "horizon_bars": int(horizon),
                                        "allocation_method": method,
                                        "min_abs_position": float(min_abs),
                                        "max_entropy": np.nan if max_entropy is None else float(max_entropy),
                                        "smoothing_alpha": float(smoothing_alpha),
                                        "rebalance_threshold": float(rebalance_threshold),
                                        "cost_scenario": str(scenario["cost_scenario"]),
                                        "cost_kind": str(scenario["cost_kind"]),
                                        "configured_round_trip_bps": float(scenario["round_trip_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                                        "ibkr_plan": str(scenario.get("ibkr_plan", "")),
                                        "notional_usd": float(scenario.get("notional_usd", np.nan)),
                                        "spread_slippage_bps_round_trip": float(scenario.get("ibkr", {}).get("spread_slippage_bps_round_trip", np.nan)),
                                        **evaluate_allocation_frame(group, position, scenario),
                                    }
                                )
    return pd.DataFrame(rows)


def parameter_columns() -> list[str]:
    return [
        "variant",
        "horizon_bars",
        "allocation_method",
        "min_abs_position",
        "max_entropy",
        "smoothing_alpha",
        "rebalance_threshold",
    ]


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    group_cols = [
        *parameter_columns(),
        "split",
        "cost_scenario",
        "cost_kind",
        "configured_round_trip_bps",
        "ibkr_plan",
        "notional_usd",
        "spread_slippage_bps_round_trip",
    ]
    grouped = (
        metrics.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            rows=("rows", "sum"),
            sessions=("sessions", "sum"),
            active_bars=("active_bars", "sum"),
            long_bars=("long_bars", "sum"),
            short_bars=("short_bars", "sum"),
            rebalance_count=("rebalance_count", "sum"),
            turnover=("turnover", "sum"),
            gross_return=("gross_return", "sum"),
            total_cost=("total_cost", "sum"),
            net_return=("net_return", "sum"),
            avg_abs_position_mean=("avg_abs_position", "mean"),
            turnover_per_session_mean=("turnover_per_session", "mean"),
            net_per_turnover_mean=("net_per_turnover", "mean"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_session_abs_net_share_max=("top_session_abs_net_share", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["net_per_turnover_pooled"] = grouped["net_return"] / grouped["turnover"].replace(0.0, np.nan)
    grouped["effective_round_trip_cost_bps"] = grouped["total_cost"] / grouped["turnover"].replace(0.0, np.nan) * 2.0 * 10_000.0
    return grouped.sort_values(["split", "cost_scenario", "daily_sharpe_mean", "net_return"], ascending=[True, True, False, False], kind="stable").reset_index(drop=True)


def _equal_or_both_nan(series: pd.Series, value: Any) -> pd.Series:
    if pd.isna(value):
        return series.isna()
    return series.eq(value)


def select_validation_allocation(aggregate: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _alloc_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    min_turnover = float(cfg.get("min_validation_turnover", 5.0))
    validation = aggregate[aggregate["split"].eq("validation") & aggregate["cost_scenario"].eq(primary)].copy()
    if validation.empty:
        validation = aggregate[aggregate["split"].eq("validation")].copy()
    eligible = validation[validation["turnover"].ge(min_turnover)].copy()
    if eligible.empty:
        eligible = validation.copy()
    if eligible.empty:
        return pd.DataFrame()
    selected = eligible.sort_values(
        ["daily_sharpe_mean", "net_return", "net_per_turnover_pooled", "turnover"],
        ascending=[False, False, False, False],
        kind="stable",
    ).head(1).copy()
    selected["selected_on"] = "validation"
    selected["selection_cost_scenario"] = primary
    selected["selection_rank_metric"] = "daily_sharpe_mean"
    return selected.reset_index(drop=True)


def selected_sensitivity(aggregate: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty or selected.empty:
        return pd.DataFrame()
    row = selected.iloc[0]
    mask = aggregate["split"].isin(["validation", "test"])
    for column in parameter_columns():
        mask &= _equal_or_both_nan(aggregate[column], row[column])
    return aggregate.loc[mask].sort_values(["split", "cost_kind", "effective_round_trip_cost_bps", "cost_scenario"], kind="stable").reset_index(drop=True)


def selected_cost_sensitivity(dataset: pd.DataFrame, selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or selected.empty:
        return pd.DataFrame()
    row = selected.iloc[0]
    mask = dataset["variant"].eq(str(row["variant"]))
    mask &= dataset["horizon_bars"].eq(int(row["horizon_bars"]))
    mask &= dataset["split"].isin(["validation", "test"])
    selected_dataset = dataset.loc[mask].copy()
    scenarios = cost_scenarios(config)
    rows: list[dict[str, Any]] = []
    for keys, group in selected_dataset.groupby(["variant", "fold", "split", "horizon_bars"], sort=False):
        variant, fold, split, horizon = keys
        target = target_position(
            group,
            method=str(row["allocation_method"]),
            min_abs_position=float(row["min_abs_position"]),
            max_entropy=None if pd.isna(row["max_entropy"]) else float(row["max_entropy"]),
        )
        position = executable_position(
            target,
            group["session"],
            smoothing_alpha=float(row["smoothing_alpha"]),
            rebalance_threshold=float(row["rebalance_threshold"]),
        )
        for scenario in scenarios:
            rows.append(
                {
                    "variant": str(variant),
                    "fold": int(fold),
                    "split": str(split),
                    "horizon_bars": int(horizon),
                    "allocation_method": str(row["allocation_method"]),
                    "min_abs_position": float(row["min_abs_position"]),
                    "max_entropy": np.nan if pd.isna(row["max_entropy"]) else float(row["max_entropy"]),
                    "smoothing_alpha": float(row["smoothing_alpha"]),
                    "rebalance_threshold": float(row["rebalance_threshold"]),
                    "cost_scenario": str(scenario["cost_scenario"]),
                    "cost_kind": str(scenario["cost_kind"]),
                    "configured_round_trip_bps": float(scenario["round_trip_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                    "ibkr_plan": str(scenario.get("ibkr_plan", "")),
                    "notional_usd": float(scenario.get("notional_usd", np.nan)),
                    "spread_slippage_bps_round_trip": float(scenario.get("ibkr", {}).get("spread_slippage_bps_round_trip", np.nan)),
                    **evaluate_allocation_frame(group, position, scenario),
                }
            )
    return aggregate_metrics(pd.DataFrame(rows))


def promotion_decision(selected: pd.DataFrame, sensitivity: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _alloc_cfg(config)
    if selected.empty or sensitivity.empty:
        return pd.DataFrame([{"decision": "rejected", "reason": "no selected validation allocation"}])

    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    min_positive_fold_share = float(cfg.get("min_positive_fold_share", 0.67))
    min_daily_sharpe = float(cfg.get("min_daily_sharpe", 0.75))
    min_net_per_turnover_bps = float(cfg.get("min_net_per_turnover_bps", 1.0))

    def row(split: str, scenario: str) -> pd.Series | None:
        frame = sensitivity[sensitivity["split"].eq(split) & sensitivity["cost_scenario"].eq(scenario)]
        return None if frame.empty else frame.iloc[0]

    checks: list[str] = []

    def pass_core(split: str, scenario: str) -> bool:
        current = row(split, scenario)
        if current is None:
            checks.append(f"missing_{split}_{scenario}")
            return False
        fold_share = float(current["positive_folds"]) / max(float(current["folds"]), 1.0)
        net_per_turnover_bps = float(current["net_per_turnover_pooled"]) * 10_000.0
        ok = True
        if float(current["net_return"]) <= 0.0:
            checks.append(f"{split}_{scenario}_net_not_positive")
            ok = False
        if fold_share < min_positive_fold_share:
            checks.append(f"{split}_{scenario}_positive_folds_below_min")
            ok = False
        if float(current["daily_sharpe_mean"]) < min_daily_sharpe:
            checks.append(f"{split}_{scenario}_sharpe_below_min")
            ok = False
        if net_per_turnover_bps < min_net_per_turnover_bps:
            checks.append(f"{split}_{scenario}_net_per_turnover_below_min")
            ok = False
        return ok

    validation_primary_ok = pass_core("validation", primary)
    test_primary_ok = pass_core("test", primary)
    test_conservative_ok = pass_core("test", conservative)
    test_stress = row("test", stress)
    stress_ok = bool(test_stress is not None and float(test_stress["net_return"]) > 0.0)
    if not stress_ok:
        checks.append(f"test_{stress}_net_not_positive")

    if validation_primary_ok and test_primary_ok and test_conservative_ok and stress_ok:
        decision = "promotion_candidate"
    elif validation_primary_ok and test_primary_ok and test_conservative_ok:
        decision = "research_candidate_cost_fragile"
    else:
        decision = "rejected"

    selected_row = selected.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "decision": decision,
                "primary_cost_scenario": primary,
                "conservative_cost_scenario": conservative,
                "stress_cost_scenario": stress,
                "failed_checks": ",".join(checks),
                "selected_variant": selected_row.get("variant"),
                "selected_method": selected_row.get("allocation_method"),
                "selected_min_abs_position": selected_row.get("min_abs_position"),
                "selected_max_entropy": selected_row.get("max_entropy"),
                "selected_smoothing_alpha": selected_row.get("smoothing_alpha"),
                "selected_rebalance_threshold": selected_row.get("rebalance_threshold"),
            }
        ]
    )


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    folds: list[LabFold],
    selected: pd.DataFrame,
    sensitivity: pd.DataFrame,
    decision: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _alloc_cfg(config)
    selected_cols = [
        *parameter_columns(),
        "cost_scenario",
        "folds",
        "positive_folds",
        "active_bars",
        "turnover",
        "gross_return",
        "total_cost",
        "net_return",
        "net_per_turnover_pooled",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    sensitivity_cols = [
        "split",
        "cost_scenario",
        "effective_round_trip_cost_bps",
        "notional_usd",
        "folds",
        "positive_folds",
        "active_bars",
        "turnover",
        "gross_return",
        "total_cost",
        "net_return",
        "net_per_turnover_pooled",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    decision_cols = [
        "decision",
        "primary_cost_scenario",
        "conservative_cost_scenario",
        "stress_cost_scenario",
        "failed_checks",
        "selected_variant",
        "selected_method",
    ]
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    fold_text = f"{len(folds)} (`{folds[0].train_months[0]}` to `{folds[-1].test_months[-1]}`)" if folds else "0"
    return f"""# H8d Probability Allocation - {target_symbol.upper()}

## Scope

- Feature file: `{bayesian_regime_h8.features_path(config, target_symbol)}`
- H8 variants: `{config.get("bayesian_regime_h8", {}).get("variants", ["manual_h8a"])}`
- Holding horizon bars: `{cfg.get("holding_horizon_bars", [1])}`
- Allocation methods: `{cfg.get("allocation_methods", ["edge", "confidence_edge", "dominant_probability"])}`
- Primary cost scenario: `{cfg.get("primary_cost_scenario", "ibkr_tiered_10000")}`
- Walk-forward folds: {fold_text}
- Test selection used: `no`

## Promotion Decision

{_markdown_table(decision.loc[:, [column for column in decision_cols if column in decision.columns]], max_rows=5)}

## Selected Validation Allocation

{_markdown_table(selected.loc[:, [column for column in selected_cols if column in selected.columns]], max_rows=10)}

## Selected Cost Sensitivity

{_markdown_table(sensitivity.loc[:, [column for column in sensitivity_cols if column in sensitivity.columns]], max_rows=80)}

## Outputs

{outputs_text}

## Notes

- Position target is continuous in `[-1, +1]`; default `edge` means `P(bull_trend) - P(bear_stress)`.
- PnL is applied from next open to the following open for `horizon_bars=1`.
- Costs are charged on turnover: entry rebalance plus forced end-of-session flatten.
- Fixed bps scenarios are interpreted as round-trip bps; each one-way turnover leg pays half.
- IBKR scenarios charge one-way commissions, SEC/TAF on sells, and half of the configured round-trip spread/slippage per turnover leg.
"""


def run_config(config: dict[str, Any], target_symbol: str | None = None) -> tuple[Path, Path]:
    target = _target_symbol(config, target_symbol)
    raw_features = pd.read_parquet(bayesian_regime_h8.features_path(config, target))
    h8_frame = bayesian_regime_h8.prepare_h8_frame(raw_features, config)
    folds = bayesian_regime_h8.build_h8_folds(h8_frame, config)
    posteriors, model_registry = bayesian_regime_h8.run_h8_posteriors(h8_frame, folds, config, target)
    dataset = prepare_allocation_dataset(raw_features, posteriors, config)
    metrics = evaluate_probability_allocations(dataset, config)
    aggregate = aggregate_metrics(metrics)
    selected = select_validation_allocation(aggregate, config)
    sensitivity = selected_cost_sensitivity(dataset, selected, config)
    decision = promotion_decision(selected, sensitivity, config)

    out_dir = results_dir(config, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "posteriors": out_dir / "h8d_posteriors.parquet",
        "model_registry": out_dir / "h8d_model_registry.parquet",
        "allocation_dataset": out_dir / "h8d_allocation_dataset.parquet",
        "metrics": out_dir / "h8d_probability_allocation_metrics.parquet",
        "aggregate": out_dir / "h8d_probability_allocation_aggregate.parquet",
        "selected": out_dir / "h8d_selected_allocation.parquet",
        "selected_sensitivity": out_dir / "h8d_selected_sensitivity.parquet",
        "promotion_decision": out_dir / "h8d_promotion_decision.parquet",
    }
    posteriors.to_parquet(outputs["posteriors"], index=False)
    model_registry.to_parquet(outputs["model_registry"], index=False)
    dataset.to_parquet(outputs["allocation_dataset"], index=False)
    metrics.to_parquet(outputs["metrics"], index=False)
    aggregate.to_parquet(outputs["aggregate"], index=False)
    selected.to_parquet(outputs["selected"], index=False)
    sensitivity.to_parquet(outputs["selected_sensitivity"], index=False)
    decision.to_parquet(outputs["promotion_decision"], index=False)

    report = report_path(config, target)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(config, target, folds, selected, sensitivity, decision, outputs), encoding="utf-8")
    return report, outputs["selected_sensitivity"]


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    return run_config(load_yaml(config_path), target_symbol=target_symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H8d continuous probability allocation.")
    parser.add_argument("--config", default="configs/hmm_bayesian_regime_h8d_qqq_15min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report, selected_sensitivity = run(args.config, args.target)
    print(f"H8d report written to: {report}")
    print(f"H8d selected sensitivity written to: {selected_sensitivity}")


if __name__ == "__main__":
    main()

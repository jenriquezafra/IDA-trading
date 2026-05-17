from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bayesian_regime_h8_allocation import cost_return, cost_scenarios, turnover_series
from src.hmm_lab import _target_symbol, build_lab_folds, load_yaml
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.setup_signal_search import (
    _base_mask,
    _configured_signal_columns,
    _json_dumps,
    _signal_column_map,
    load_feature_frame,
    signal_mask,
)


PARAM_COLUMNS = [
    "rule_name",
    "family",
    "direction",
    "max_hold_bars",
    "min_hold_bars",
    "stop_loss_bps",
    "take_profit_bps",
    "exit_on_signal_loss",
    "cooldown_bars",
]


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_portfolio_lifecycle", {})


def _targets(config: dict[str, Any]) -> list[str]:
    configured = _cfg(config).get("targets")
    if configured:
        return [str(value).upper() for value in configured]
    return [_target_symbol(config)]


def _rule_spec(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _cfg(config)
    raw = cfg.get("rule")
    if raw is None:
        rules = cfg.get("rules") or config.get("setup_signal_fixed_rules", {}).get("rules", [])
        if not rules:
            raise ValueError("setup_signal_portfolio_lifecycle requires a rule")
        raw = rules[0]
    params = {str(key): value for key, value in dict(raw.get("params", {})).items()}
    direction = str(raw.get("direction", params.get("direction", "long")))
    params["direction"] = direction
    return {
        "rule_name": str(raw["name"]),
        "family": str(raw["family"]),
        "direction": direction,
        "params": params,
        "column_map": {str(k): str(v) for k, v in dict(raw.get("column_map", {})).items()},
    }


def _candidate_meta(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config.get("candidate", {}))
    return {
        "candidate_id": str(raw.get("candidate_id", "")),
        "hypothesis_id": str(raw.get("hypothesis_id", "")),
        "candidate_role": str(raw.get("role", "")),
        "candidate_status": str(raw.get("status", "")),
    }


def _disabled_bps(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and np.isnan(value):
        return 0.0
    return max(0.0, float(value))


def _exit_parameter_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = _cfg(config).get("exit_grid", {})
    rows: list[dict[str, Any]] = []
    for max_hold in grid.get("max_hold_bars", [24]):
        for min_hold in grid.get("min_hold_bars", [1]):
            for stop in grid.get("stop_loss_bps", [0.0]):
                for take in grid.get("take_profit_bps", [0.0]):
                    for exit_on_loss in grid.get("exit_on_signal_loss", [False]):
                        for cooldown in grid.get("cooldown_bars", [0]):
                            rows.append(
                                {
                                    "max_hold_bars": int(max_hold),
                                    "min_hold_bars": int(min_hold),
                                    "stop_loss_bps": _disabled_bps(stop),
                                    "take_profit_bps": _disabled_bps(take),
                                    "exit_on_signal_loss": bool(exit_on_loss),
                                    "cooldown_bars": int(cooldown),
                                }
                            )
    return rows


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    cfg = _cfg(config)
    results_dir = Path(cfg.get("results_dir", Path(config.get("paths", {}).get("results_dir", "results")) / "_portfolio_lifecycle"))
    reports_dir = Path(cfg.get("reports_dir", Path(config.get("paths", {}).get("reports_dir", "reports")) / "_portfolio_lifecycle"))
    return {
        "fold_metrics": results_dir / "setup_signal_portfolio_lifecycle_fold_metrics.parquet",
        "summary": results_dir / "setup_signal_portfolio_lifecycle_summary.parquet",
        "promotion": results_dir / "setup_signal_portfolio_lifecycle_promotion.parquet",
        "report": reports_dir / "setup_signal_portfolio_lifecycle.md",
    }


def build_lifecycle_dataset(config: dict[str, Any], target: str) -> pd.DataFrame:
    features = load_feature_frame(config, target)
    forward = build_forward_returns(features, [1])
    if forward.empty:
        return pd.DataFrame()

    indexed = features.reset_index(names="source_index")
    columns = [
        "source_index",
        *[column for column in _configured_signal_columns(config) if column in indexed.columns and column not in forward.columns],
    ]
    merged = forward.merge(indexed.loc[:, columns], on="source_index", how="left", validate="many_to_one")
    merged["target"] = target.upper()
    merged["target_open_next"] = merged["entry_px"].astype(float)

    folds = build_lab_folds(features, config)
    parts: list[pd.DataFrame] = []
    for fold in folds:
        for split, sessions in {"validation": set(fold.validation_sessions), "test": set(fold.test_sessions)}.items():
            part = merged[merged["session"].isin(sessions)].copy()
            if part.empty:
                continue
            part.insert(0, "split", split)
            part.insert(0, "fold", int(fold.fold))
            parts.append(part)
    if not parts:
        return pd.DataFrame()

    output = pd.concat(parts, ignore_index=True)
    output["timestamp"] = pd.to_datetime(output["timestamp"])
    output["month"] = output["timestamp"].dt.strftime("%Y-%m")
    return output.sort_values(["target", "fold", "split", "session", "bar_index"], kind="stable").reset_index(drop=True)


def lifecycle_position_from_entries(
    frame: pd.DataFrame,
    entry_signal: pd.Series,
    *,
    direction: float,
    max_hold_bars: int,
    min_hold_bars: int = 1,
    stop_loss_bps: float = 0.0,
    take_profit_bps: float = 0.0,
    exit_on_signal_loss: bool = False,
    cooldown_bars: int = 0,
) -> pd.Series:
    signal = entry_signal.reindex(frame.index).fillna(False).astype(bool)
    output = pd.Series(0.0, index=frame.index, dtype=float)
    max_hold = max(0, int(max_hold_bars))
    min_hold = max(1, int(min_hold_bars))
    stop = _disabled_bps(stop_loss_bps) / 10_000.0
    take = _disabled_bps(take_profit_bps) / 10_000.0
    cooldown_cfg = max(0, int(cooldown_bars))
    signed_direction = 1.0 if float(direction) >= 0.0 else -1.0
    signal_values = signal.to_numpy(dtype=bool)
    returns = frame["fwd_ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    output_values = np.zeros(len(frame), dtype=float)

    for positions in frame.groupby("session", sort=False).indices.values():
        position = 0.0
        age = 0
        cumulative = 0.0
        cooldown = 0
        for loc in positions:
            current_entry = bool(signal_values[loc])
            if position == 0.0:
                if cooldown > 0:
                    cooldown -= 1
                elif current_entry:
                    position = signed_direction
                    age = 0
                    cumulative = 0.0

            output_values[loc] = position

            if position != 0.0:
                cumulative += position * float(returns[loc])
                age += 1
                can_exit = age >= min_hold
                exit_now = False
                if can_exit and stop > 0.0 and cumulative <= -stop:
                    exit_now = True
                if can_exit and take > 0.0 and cumulative >= take:
                    exit_now = True
                if can_exit and max_hold > 0 and age >= max_hold:
                    exit_now = True
                if can_exit and exit_on_signal_loss and not current_entry:
                    exit_now = True
                if exit_now:
                    position = 0.0
                    age = 0
                    cumulative = 0.0
                    cooldown = cooldown_cfg
    output.iloc[:] = output_values
    return output


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252.0) * daily.mean() / std)


def _max_drawdown_from_timestamp(frame: pd.DataFrame, net: pd.Series) -> float:
    path = net.groupby(pd.to_datetime(frame["timestamp"])).sum()
    if path.empty:
        return 0.0
    equity = path.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0.0].sum()
    gross_loss = -active_net[active_net < 0.0].sum()
    if gross_loss == 0.0:
        return np.inf if gross_profit > 0.0 else np.nan
    return float(gross_profit / gross_loss)


def _top_session_abs_share(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    denom = daily.abs().sum()
    if denom == 0.0 or np.isnan(denom):
        return np.nan
    return float(daily.abs().max() / denom)


def _turnover_group(frame: pd.DataFrame) -> pd.Series:
    if "target" not in frame.columns:
        return frame["session"].astype(str)
    return frame["target"].astype(str) + "|" + frame["session"].astype(str)


def evaluate_lifecycle_frame(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> dict[str, Any]:
    position = position.reindex(frame.index).astype(float).fillna(0.0)
    cost_frame = frame.copy()
    cost_frame["session"] = _turnover_group(frame)
    entry_turnover, exit_turnover = turnover_series(position, cost_frame["session"])
    turnover = entry_turnover + exit_turnover
    gross = position * frame["fwd_ret"].astype(float)
    cost = cost_return(cost_frame, position, entry_turnover, exit_turnover, scenario)
    net = gross - cost
    active = position.abs() > 1e-12
    active_net = net[active]
    turnover_sum = float(turnover.sum())
    sessions = int(frame["session"].nunique()) if "session" in frame else 0
    previous = position.groupby(cost_frame["session"], sort=False).shift(1).fillna(0.0)
    entries = int((previous.abs().le(1e-12) & position.abs().gt(1e-12)).sum())
    effective_cost = float(cost.sum() / turnover_sum * 2.0 * 10_000.0) if turnover_sum > 0.0 else np.nan
    return {
        "rows": int(len(frame)),
        "sessions": sessions,
        "entries": entries,
        "active_bars": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "avg_abs_position": float(position.abs().mean()) if len(position) else 0.0,
        "turnover": turnover_sum,
        "turnover_per_session": float(turnover_sum / sessions) if sessions else np.nan,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "effective_round_trip_cost_bps": effective_cost,
        "net_return": float(net.sum()),
        "net_per_turnover": float(net.sum() / turnover_sum) if turnover_sum > 0.0 else 0.0,
        "avg_active_bar_net": float(active_net.mean()) if len(active_net) else 0.0,
        "hit_rate_active_bar": float((active_net > 0.0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": _max_drawdown_from_timestamp(frame, net),
        "top_session_abs_net_share": _top_session_abs_share(frame, net),
    }


def _scenario_names(config: dict[str, Any]) -> list[str] | None:
    names = _cfg(config).get("cost_scenarios")
    return [str(value) for value in names] if names else None


def _scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    ibkr = scenario.get("ibkr", {})
    return {
        "cost_scenario": str(scenario["cost_scenario"]),
        "cost_kind": str(scenario["cost_kind"]),
        "configured_round_trip_bps": float(scenario["round_trip_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
        "ibkr_plan": str(scenario.get("ibkr_plan", "")),
        "notional_usd": float(scenario.get("notional_usd", np.nan)),
        "spread_slippage_bps_round_trip": float(ibkr.get("spread_slippage_bps_round_trip", np.nan)),
    }


def evaluate_lifecycle(config: dict[str, Any]) -> pd.DataFrame:
    targets = _targets(config)
    datasets = [build_lifecycle_dataset(config, target) for target in targets]
    dataset = pd.concat([part for part in datasets if not part.empty], ignore_index=True, sort=False) if datasets else pd.DataFrame()
    if dataset.empty:
        return pd.DataFrame()

    rule = _rule_spec(config)
    candidate_meta = _candidate_meta(config)
    columns = _signal_column_map(config, rule["column_map"])
    direction = 1.0 if rule["direction"] == "long" else -1.0
    scenarios = cost_scenarios(config, _scenario_names(config))
    if not scenarios:
        scenarios = cost_scenarios(config)
    include_base_control = bool(_cfg(config).get("include_base_control", True))

    rows: list[dict[str, Any]] = []
    target_count = max(1, len(targets))
    for exit_params in _exit_parameter_rows(config):
        positioned_parts: list[pd.DataFrame] = []
        for keys, group in dataset.groupby(["target", "fold", "split"], sort=False):
            target, fold, split = keys
            sorted_group = group.sort_values(["session", "bar_index"], kind="stable").copy()
            signals = {"lifecycle_rule": signal_mask(sorted_group, rule["family"], rule["params"], columns)}
            if include_base_control:
                signals["base_segment_control"] = _base_mask(sorted_group, rule["family"], rule["params"], columns)
            for bucket, entry in signals.items():
                position = lifecycle_position_from_entries(
                    sorted_group,
                    entry,
                    direction=direction,
                    **exit_params,
                )
                positioned = sorted_group.copy()
                positioned["bucket"] = bucket
                positioned["position"] = position
                positioned_parts.append(positioned)
                base = {
                    "scope": "target",
                    "target": str(target),
                    "portfolio_weight": "unit_target",
                    "bucket": bucket,
                    "fold": int(fold),
                    "split": str(split),
                    "rule_name": rule["rule_name"],
                    "family": rule["family"],
                    "direction": rule["direction"],
                    **candidate_meta,
                    "params_json": _json_dumps(rule["params"]),
                    "column_map_json": _json_dumps(columns),
                    "exit_params_json": json.dumps(exit_params, sort_keys=True),
                    **exit_params,
                }
                for scenario in scenarios:
                    rows.append({**base, **_scenario_metadata(scenario), **evaluate_lifecycle_frame(positioned, position, scenario)})

        if not positioned_parts:
            continue
        positioned_all = pd.concat(positioned_parts, ignore_index=True, sort=False)
        for keys, group in positioned_all.groupby(["bucket", "fold", "split"], sort=False):
            bucket, fold, split = keys
            sorted_group = group.sort_values(["timestamp", "target", "bar_index"], kind="stable").copy()
            position = sorted_group["position"].astype(float) / float(target_count)
            base = {
                "scope": "portfolio",
                "target": "PORTFOLIO",
                "portfolio_weight": "equal_target",
                "bucket": str(bucket),
                "fold": int(fold),
                "split": str(split),
                "rule_name": rule["rule_name"],
                "family": rule["family"],
                "direction": rule["direction"],
                **candidate_meta,
                "params_json": _json_dumps(rule["params"]),
                "column_map_json": _json_dumps(columns),
                "exit_params_json": json.dumps(exit_params, sort_keys=True),
                **exit_params,
            }
            for scenario in scenarios:
                rows.append({**base, **_scenario_metadata(scenario), **evaluate_lifecycle_frame(sorted_group, position, scenario)})

    return pd.DataFrame(rows)


def aggregate_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    group_cols = [
        "scope",
        "target",
        "portfolio_weight",
        "bucket",
        *PARAM_COLUMNS,
        "params_json",
        "column_map_json",
        "exit_params_json",
        "split",
        "cost_scenario",
        "cost_kind",
        "configured_round_trip_bps",
        "ibkr_plan",
        "notional_usd",
        "spread_slippage_bps_round_trip",
    ]
    grouped = (
        fold_metrics.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            rows=("rows", "sum"),
            sessions=("sessions", "sum"),
            entries=("entries", "sum"),
            active_bars=("active_bars", "sum"),
            exposure=("exposure", "mean"),
            avg_abs_position=("avg_abs_position", "mean"),
            turnover=("turnover", "sum"),
            gross_return=("gross_return", "sum"),
            total_cost=("total_cost", "sum"),
            net_return=("net_return", "sum"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_session_abs_net_share_max=("top_session_abs_net_share", "max"),
            hit_rate_active_bar_mean=("hit_rate_active_bar", "mean"),
            profit_factor_median=("profit_factor", "median"),
        )
        .reset_index(drop=True)
    )
    grouped["positive_fold_share"] = grouped["positive_folds"] / grouped["folds"].replace(0, np.nan)
    grouped["net_per_turnover_pooled"] = grouped["net_return"] / grouped["turnover"].replace(0.0, np.nan)
    grouped["avg_entry_net_pooled"] = grouped["net_return"] / grouped["entries"].replace(0, np.nan)
    grouped["effective_round_trip_cost_bps"] = grouped["total_cost"] / grouped["turnover"].replace(0.0, np.nan) * 2.0 * 10_000.0

    keys = [column for column in group_cols if column != "bucket"]
    control = grouped[grouped["bucket"].eq("base_segment_control")].loc[:, [*keys, "net_return", "daily_sharpe_mean", "max_drawdown_max"]]
    control = control.rename(
        columns={
            "net_return": "base_net_return",
            "daily_sharpe_mean": "base_daily_sharpe_mean",
            "max_drawdown_max": "base_max_drawdown_max",
        }
    )
    signal = grouped[grouped["bucket"].eq("lifecycle_rule")].merge(control, on=keys, how="left", validate="many_to_one")
    signal["net_delta_vs_base_segment"] = signal["net_return"] - signal["base_net_return"].fillna(0.0)
    signal["daily_sharpe_delta_vs_base_segment"] = signal["daily_sharpe_mean"] - signal["base_daily_sharpe_mean"]
    signal["drawdown_delta_vs_base_segment"] = signal["max_drawdown_max"] - signal["base_max_drawdown_max"]
    controls = grouped[~grouped["bucket"].eq("lifecycle_rule")].copy()
    for column in ["base_net_return", "base_daily_sharpe_mean", "base_max_drawdown_max", "net_delta_vs_base_segment", "daily_sharpe_delta_vs_base_segment", "drawdown_delta_vs_base_segment"]:
        controls[column] = np.nan
    return pd.concat([signal, controls], ignore_index=True, sort=False).sort_values(
        ["scope", "split", "cost_scenario", "net_return"],
        ascending=[True, True, True, False],
        kind="stable",
    )


def build_promotion_summary(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    cfg = _cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    min_targets = int(cfg.get("min_positive_targets", len(_targets(config))))
    min_stress_targets = int(cfg.get("min_stress_nonnegative_targets", min_targets))
    max_drawdown = float(cfg.get("max_primary_drawdown", np.inf))
    min_net_turnover_bps = float(cfg.get("min_primary_net_per_turnover_bps", 0.0))

    test = summary[
        summary["split"].eq("test")
        & summary["bucket"].eq("lifecycle_rule")
    ].copy()
    parameter_keys = PARAM_COLUMNS + ["params_json", "column_map_json", "exit_params_json"]
    portfolio = test[test["scope"].eq("portfolio")]
    rows: list[dict[str, Any]] = []
    for _, primary_row in portfolio[portfolio["cost_scenario"].eq(primary)].iterrows():
        key_mask = pd.Series(True, index=test.index)
        for column in parameter_keys:
            key_mask &= test[column].eq(primary_row[column])
        stress_rows = portfolio.loc[key_mask.loc[portfolio.index] & portfolio["cost_scenario"].eq(stress)]
        stress_row = stress_rows.iloc[0] if not stress_rows.empty else pd.Series(dtype=object)
        target_primary = test.loc[key_mask & test["scope"].eq("target") & test["cost_scenario"].eq(primary)]
        target_stress = test.loc[key_mask & test["scope"].eq("target") & test["cost_scenario"].eq(stress)]
        primary_positive_targets = int((target_primary["net_return"].gt(0.0) & target_primary["net_per_turnover_pooled"].gt(0.0)).sum())
        stress_nonnegative_targets = int((target_stress["net_return"].ge(0.0) & target_stress["net_per_turnover_pooled"].ge(0.0)).sum())
        net_turnover_bps = float(primary_row["net_per_turnover_pooled"] * 10_000.0)
        passes = bool(
            float(primary_row["net_return"]) > 0.0
            and float(stress_row.get("net_return", np.nan)) >= 0.0
            and net_turnover_bps >= min_net_turnover_bps
            and float(primary_row["max_drawdown_max"]) <= max_drawdown
            and primary_positive_targets >= min_targets
            and stress_nonnegative_targets >= min_stress_targets
        )
        rows.append(
            {
                **{column: primary_row[column] for column in PARAM_COLUMNS},
                "primary_cost_scenario": primary,
                "stress_cost_scenario": stress,
                "portfolio_primary_net_return": float(primary_row["net_return"]),
                "portfolio_stress_net_return": float(stress_row.get("net_return", np.nan)),
                "portfolio_primary_net_per_turnover_bps": net_turnover_bps,
                "portfolio_stress_net_per_turnover_bps": float(stress_row.get("net_per_turnover_pooled", np.nan) * 10_000.0),
                "portfolio_primary_sharpe": float(primary_row["daily_sharpe_mean"]),
                "portfolio_primary_max_drawdown": float(primary_row["max_drawdown_max"]),
                "portfolio_primary_entries": int(primary_row["entries"]),
                "portfolio_primary_turnover": float(primary_row["turnover"]),
                "primary_positive_targets": primary_positive_targets,
                "stress_nonnegative_targets": stress_nonnegative_targets,
                "target_count": int(target_primary["target"].nunique()),
                "passes_promotion_gate": passes,
            }
        )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["passes_promotion_gate", "portfolio_primary_net_return", "portfolio_primary_net_per_turnover_bps"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def render_report(config: dict[str, Any], summary: pd.DataFrame, promotion: pd.DataFrame, outputs: dict[str, Path]) -> str:
    cfg = _cfg(config)
    candidate_meta = _candidate_meta(config)
    targets = ", ".join(_targets(config))
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    accepted = int(promotion["passes_promotion_gate"].sum()) if not promotion.empty and "passes_promotion_gate" in promotion else 0
    conclusion = (
        f"{accepted} lifecycle variants pass the configured promotion gate."
        if accepted
        else "No lifecycle variant passes the configured promotion gate; keep as research until robustness improves."
    )
    primary_portfolio = summary[
        summary["scope"].eq("portfolio")
        & summary["bucket"].eq("lifecycle_rule")
        & summary["split"].eq("test")
        & summary["cost_scenario"].eq(primary)
    ].sort_values("net_return", ascending=False, kind="stable")
    primary_targets = summary[
        summary["scope"].eq("target")
        & summary["bucket"].eq("lifecycle_rule")
        & summary["split"].eq("test")
        & summary["cost_scenario"].eq(primary)
    ].sort_values(["target", "net_return"], ascending=[True, False], kind="stable")
    output_lines = "\n".join(f"- `{name}`: `{path}`" for name, path in outputs.items())
    return f"""# H9 Portfolio Lifecycle Evaluation

## Scope

- Targets: `{targets}`
- Candidate: `{candidate_meta["candidate_id"] or "unassigned"}`
- Hypothesis: `{candidate_meta["hypothesis_id"] or "unassigned"}`
- Candidate role/status: `{candidate_meta["candidate_role"] or "n/a"}` / `{candidate_meta["candidate_status"] or "n/a"}`
- Primary cost: `{primary}`
- Stress cost: `{stress}`
- Portfolio sizing: `equal_target`
- Stop/take-profit measurement: open-to-open 5 minute path. The current feature set does not include intrabar high/low, so stop/take exits are conservative close-of-bar approximations.

## Promotion Gate

{_markdown_table(promotion, max_rows=int(cfg.get("report_top_rows", 40)))}

## Portfolio Test Summary - Primary Cost

{_markdown_table(primary_portfolio, max_rows=int(cfg.get("report_top_rows", 40)))}

## Target Test Summary - Primary Cost

{_markdown_table(primary_targets, max_rows=int(cfg.get("report_top_rows", 80)))}

## Outputs

{output_lines}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    fold_metrics = evaluate_lifecycle(config)
    summary = aggregate_summary(fold_metrics)
    promotion = build_promotion_summary(summary, config)
    outputs = _output_paths(config)
    outputs["fold_metrics"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_parquet(outputs["fold_metrics"], index=False)
    summary.to_parquet(outputs["summary"], index=False)
    promotion.to_parquet(outputs["promotion"], index=False)
    outputs["report"].write_text(render_report(config, summary, promotion, outputs), encoding="utf-8")
    return outputs["report"], outputs["promotion"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a setup signal as an executable portfolio lifecycle.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report_path, promotion_path = run(args.config)
    print(f"H9 portfolio lifecycle report written to: {report_path}")
    print(f"H9 portfolio lifecycle promotion summary written to: {promotion_path}")


if __name__ == "__main__":
    main()

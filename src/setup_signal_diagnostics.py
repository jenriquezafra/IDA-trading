from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import scenario_cost_return
from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios
from src.setup_signal_search import (
    _base_mask,
    _combined_cfg,
    _json_dumps,
    _json_loads,
    _signal_column_map,
    build_signal_dataset,
    evaluate_candidate,
    signal_mask,
)


DEFAULT_FEATURE_COLUMNS = [
    "target_ret_1",
    "target_ret_2",
    "target_ret_4",
    "target_dist_vwap_atr",
    "target_dist_open",
    "target_range_ratio_2_8",
    "target_rv_4_rel_by_bar",
    "target_rel_volume_by_bar",
    "target_close_location_bar",
    "target_lower_wick_ratio",
    "target_upper_wick_ratio",
    "target_minutes_from_open",
    "target_minutes_to_close",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_2",
    "positive_sector_count_2",
    "intraday_stress_score",
    "chop_score",
]


def _diag_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_diagnostics", {})


def _path_from_template(template: str | Path, target_symbol: str) -> Path:
    return Path(str(template).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "setup_signal_diagnostics.md"


def default_input_paths(config: dict[str, Any], target_symbol: str) -> dict[str, Path]:
    cfg = _diag_cfg(config)
    results_dir = results_output_dir(config, target_symbol)
    return {
        "validation": _path_from_template(cfg.get("validation", results_dir / "setup_signal_validation.parquet"), target_symbol),
        "decisions": _path_from_template(cfg.get("decisions", results_dir / "setup_signal_decisions.parquet"), target_symbol),
    }


def configured_focus(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _diag_cfg(config)
    return {
        "family": str(cfg.get("family", "breakdown_short_risk_off")),
        "direction": str(cfg.get("direction", "short")),
        "horizon_bars": int(cfg.get("horizon_bars", 4)),
        "max_specs_per_fold": int(cfg.get("max_specs_per_fold", 12)),
    }


def cost_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    cfg = _combined_cfg(config)
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg}, names)
    if not scenarios:
        raise ValueError(f"Cost scenarios not found: {names or 'configured scenarios'}")
    return scenarios


def primary_cost_name(config: dict[str, Any]) -> str:
    return str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))


def stress_cost_name(config: dict[str, Any]) -> str:
    return str(_combined_cfg(config).get("stress_cost_scenario", "bps_5"))


def select_focus_specs(validation: pd.DataFrame, decisions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    focus = configured_focus(config)
    primary = primary_cost_name(config)
    frame = validation[
        validation["bucket"].eq("setup_signal")
        & validation["cost_scenario"].eq(primary)
        & validation["family"].eq(focus["family"])
        & validation["direction"].eq(focus["direction"])
        & validation["horizon_bars"].astype(int).eq(focus["horizon_bars"])
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["validation_positive"] = frame["net_return"].gt(0.0) & frame["avg_trade_net"].gt(0.0)
    frame["was_selected"] = frame["candidate_id"].isin(set(decisions.get("candidate_id", pd.Series(dtype=object))))
    frame["status_rank"] = np.select(
        [
            frame["candidate_status"].eq("setup_validation_candidate"),
            frame["candidate_status"].eq("rejected_month_concentration"),
            frame["candidate_status"].eq("rejected_day_concentration"),
            frame["validation_positive"],
        ],
        [0, 1, 2, 3],
        default=4,
    )
    frame = frame.sort_values(
        ["fold", "was_selected", "status_rank", "net_return", "avg_trade_net"],
        ascending=[True, False, True, False, False],
        kind="stable",
    )
    frame = frame.groupby("fold", group_keys=False, sort=False).head(focus["max_specs_per_fold"]).reset_index(drop=True)
    cols = [
        "candidate_id",
        "fold",
        "family",
        "direction",
        "horizon_bars",
        "params_json",
        "column_map_json",
        "candidate_status",
        "was_selected",
        "validation_positive",
        "trades",
        "net_return",
        "avg_trade_net",
        "profit_factor",
        "daily_sharpe",
        "top_day_abs_net_share",
        "top_month_abs_net_share",
    ]
    return frame.loc[:, [column for column in cols if column in frame.columns]]


def evaluate_focus_specs(dataset: pd.DataFrame, specs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    scenarios = cost_scenarios(config)
    rows = []
    for _, spec in specs.iterrows():
        fold = int(spec["fold"])
        horizon = int(spec["horizon_bars"])
        for split in ["validation", "test"]:
            frame = dataset[
                dataset["split"].eq(split)
                & dataset["fold"].eq(fold)
                & dataset["horizon_bars"].astype(int).eq(horizon)
            ].copy()
            if frame.empty:
                continue
            for scenario in scenarios:
                rows.append(evaluate_candidate(frame, spec, split, scenario))
    if not rows:
        return pd.DataFrame()
    evaluated = pd.concat(rows, ignore_index=True)
    meta = specs.loc[:, ["candidate_id", "candidate_status", "was_selected", "validation_positive"]].drop_duplicates("candidate_id")
    return evaluated.merge(meta, on="candidate_id", how="left", validate="many_to_one")


def reconstruct_bar_returns(
    dataset: pd.DataFrame,
    specs: pd.DataFrame,
    scenario: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if dataset.empty or specs.empty:
        return pd.DataFrame()
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    rows = []
    for _, spec in specs.iterrows():
        params = _json_loads(spec["params_json"])
        columns = _signal_column_map(override=_json_loads(spec.get("column_map_json")))
        direction = 1.0 if str(spec["direction"]) == "long" else -1.0
        for split in splits:
            frame = dataset[
                dataset["split"].eq(split)
                & dataset["fold"].eq(int(spec["fold"]))
                & dataset["horizon_bars"].astype(int).eq(int(spec["horizon_bars"]))
            ].copy()
            if frame.empty:
                continue
            signal = signal_mask(frame, str(spec["family"]), params, columns)
            base = _base_mask(frame, str(spec["family"]), params, columns)
            position = pd.Series(0.0, index=frame.index)
            position.loc[signal] = direction
            base_position = pd.Series(0.0, index=frame.index)
            base_position.loc[base] = direction
            active = position.ne(0.0)
            if not active.any():
                continue
            cost = scenario_cost_return(frame, position, scenario)
            base_cost = scenario_cost_return(frame, base_position, scenario)
            fwd = frame["fwd_ret"].astype(float)
            out = pd.DataFrame(
                {
                    "candidate_id": spec["candidate_id"],
                    "validation_status": spec.get("candidate_status", ""),
                    "was_selected": bool(spec.get("was_selected", False)),
                    "split": split,
                    "fold": int(spec["fold"]),
                    "family": spec["family"],
                    "direction": spec["direction"],
                    "horizon_bars": int(spec["horizon_bars"]),
                    "timestamp": pd.to_datetime(frame["timestamp"]),
                    "session": frame["session"].values,
                    "bar_index": frame["bar_index"].astype(int).values,
                    "hour": frame["hour"].astype(int).values,
                    "position": position.values,
                    "fwd_ret": fwd.values,
                    "gross_return": (position * fwd).values,
                    "cost_return": cost.values,
                    "net_return": (position * fwd - cost).values,
                    "base_position": base_position.values,
                    "base_net_return": (base_position * fwd - base_cost).values,
                    "base_active": base_position.ne(0.0).values,
                },
                index=frame.index,
            )
            out = out.loc[active].copy()
            out["month"] = out["timestamp"].dt.strftime("%Y-%m")
            out["date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
            out["params_json"] = spec["params_json"]
            out["column_map_json"] = _json_dumps(columns)
            for column in feature_columns:
                if column in frame.columns:
                    out[column] = frame.loc[out.index, column].astype(float).values
            rows.append(out.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_returns(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, as_index=False, sort=False).agg(
        candidates=("candidate_id", "nunique"),
        trades=("net_return", "size"),
        net_return=("net_return", "sum"),
        gross_return=("gross_return", "sum"),
        cost_return=("cost_return", "sum"),
        avg_trade_net=("net_return", "mean"),
        mean_fwd_ret=("fwd_ret", "mean"),
        hit_rate=("net_return", lambda values: float((values > 0.0).mean())),
    )
    grouped["net_per_candidate"] = grouped["net_return"] / grouped["candidates"].replace(0, np.nan)
    return grouped.sort_values([*group_cols], kind="stable").reset_index(drop=True)


def feature_shift_summary(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    feature_columns = [column for column in (feature_columns or DEFAULT_FEATURE_COLUMNS) if column in frame.columns]
    rows = []
    for (split, fold), group in frame.groupby(["split", "fold"], sort=False):
        for column in feature_columns:
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "feature": column,
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "rows": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _best_setup_eval(evaluation: pd.DataFrame, fold: int, split: str, cost_scenario: str) -> pd.Series:
    if evaluation.empty:
        return pd.Series(dtype=object)
    frame = evaluation[
        evaluation["bucket"].eq("setup_signal")
        & evaluation["fold"].astype(int).eq(int(fold))
        & evaluation["split"].eq(split)
        & evaluation["cost_scenario"].eq(cost_scenario)
    ].copy()
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.sort_values(["net_return", "avg_trade_net"], ascending=[False, False], kind="stable").iloc[0]


def _candidate_setup_eval(evaluation: pd.DataFrame, candidate_id: str, split: str, cost_scenario: str) -> pd.Series:
    if evaluation.empty:
        return pd.Series(dtype=object)
    frame = evaluation[
        evaluation["bucket"].eq("setup_signal")
        & evaluation["candidate_id"].eq(candidate_id)
        & evaluation["split"].eq(split)
        & evaluation["cost_scenario"].eq(cost_scenario)
    ].copy()
    return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)


def _metric(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _accepted_like_candidates(evaluation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(cfg.get("conservative_cost_scenario", "bps_2"))
    stress = str(cfg.get("stress_cost_scenario", "bps_5"))
    setup = evaluation[evaluation["bucket"].eq("setup_signal") & evaluation["split"].eq("test")].copy()
    if "was_selected" in setup.columns:
        setup = setup[setup["was_selected"].fillna(False).astype(bool)]
    if setup.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for candidate_id, group in setup.groupby("candidate_id", sort=False):
        primary_row = group[group["cost_scenario"].eq(primary)]
        conservative_row = group[group["cost_scenario"].eq(conservative)]
        stress_row = group[group["cost_scenario"].eq(stress)]
        if primary_row.empty:
            continue
        test_row = primary_row.iloc[0]
        cons_row = conservative_row.iloc[0] if not conservative_row.empty else pd.Series(dtype=object)
        stress_item = stress_row.iloc[0] if not stress_row.empty else pd.Series(dtype=object)

        primary_ok = bool(
            _metric(test_row, "trades") >= int(cfg.get("min_trades", 40))
            and _metric(test_row, "net_return") > 0.0
            and _metric(test_row, "avg_trade_net") > 0.0
            and _metric(test_row, "profit_factor") >= float(cfg.get("min_profit_factor", 1.05))
            and _metric(test_row, "max_drawdown") <= float(cfg.get("max_drawdown", 0.30))
            and _metric(test_row, "top_day_abs_net_share") <= float(cfg.get("max_top_day_abs_net_share", 0.30))
            and _metric(test_row, "top_month_abs_net_share") <= float(cfg.get("max_top_month_abs_net_share", 0.50))
        )
        conservative_ok = bool(
            not cons_row.empty and _metric(cons_row, "net_return") > 0.0 and _metric(cons_row, "avg_trade_net") > 0.0
        )
        stress_ok = bool(
            not stress_item.empty and _metric(stress_item, "net_return") >= 0.0 and _metric(stress_item, "avg_trade_net") >= 0.0
        )
        if primary_ok and conservative_ok and stress_ok:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold": int(test_row.get("fold", -1)),
                    "net_return": _metric(test_row, "net_return"),
                    "avg_trade_net": _metric(test_row, "avg_trade_net"),
                    "trades": int(_metric(test_row, "trades")),
                    "profit_factor": _metric(test_row, "profit_factor"),
                    "top_month_abs_net_share": _metric(test_row, "top_month_abs_net_share"),
                    "stress_net_return": _metric(stress_item, "net_return"),
                }
            )
    return pd.DataFrame(rows)


def _max_month_share(monthly: pd.DataFrame, fold: int, split: str) -> tuple[str, float, float]:
    if monthly.empty:
        return "", np.nan, np.nan
    frame = monthly[monthly["fold"].astype(int).eq(int(fold)) & monthly["split"].eq(split)].copy()
    if frame.empty:
        return "", np.nan, np.nan
    frame["abs_net_return"] = frame["net_return"].abs()
    total_abs = float(frame["abs_net_return"].sum())
    if total_abs <= 0.0:
        return "", np.nan, np.nan
    best = frame.sort_values("abs_net_return", ascending=False, kind="stable").iloc[0]
    return str(best["month"]), float(best["net_return"]), float(best["abs_net_return"] / total_abs)


def diagnostic_findings(evaluation: pd.DataFrame, monthly: pd.DataFrame, feature_shift: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    primary = primary_cost_name(config)
    stress = stress_cost_name(config)
    f0_val = _best_setup_eval(evaluation, 0, "validation", primary)
    f0_test = _best_setup_eval(evaluation, 0, "test", primary)
    f1_val = _best_setup_eval(evaluation, 1, "validation", primary)
    f1_best_forced_test = _best_setup_eval(evaluation, 1, "test", primary)
    f0_stress = _candidate_setup_eval(evaluation, str(f0_test.get("candidate_id", "")), "test", stress) if not f0_test.empty else pd.Series(dtype=object)
    f1_same_test = _candidate_setup_eval(evaluation, str(f1_val.get("candidate_id", "")), "test", primary) if not f1_val.empty else pd.Series(dtype=object)
    f1_same_stress = _candidate_setup_eval(evaluation, str(f1_val.get("candidate_id", "")), "test", stress) if not f1_val.empty else pd.Series(dtype=object)

    if not f1_val.empty:
        rows.append(
            {
                "finding": "fold1_validation_candidates_exist",
                "evidence": f"best fold1 validation net={f1_val['net_return']:.6f}; status={f1_val.get('candidate_status', '')}; top_month_share={f1_val.get('top_month_abs_net_share', np.nan):.2%}",
                "implication": "La familia no desaparece en fold1 validation; requiere revisar estabilidad temporal, costes y fold-by-fold.",
            }
        )
    if not f1_val.empty and not f1_same_test.empty:
        rows.append(
            {
                "finding": "fold1_validation_to_test_decay",
                "evidence": f"best fold1 validation candidate net={f1_val['net_return']:.6f}; same candidate forced test net={f1_same_test['net_return']:.6f}; avg trade test={f1_same_test['avg_trade_net']:.6f}",
                "implication": "La fuerza baja de validation a test; si sigue positiva, debe tratarse como candidato bajo revision, no como estrategia final.",
            }
        )
    if not f1_best_forced_test.empty:
        rows.append(
            {
                "finding": "fold1_best_forced_test_still_not_acceptable",
                "evidence": f"best fold1 forced test net={f1_best_forced_test['net_return']:.6f}; trades={int(f1_best_forced_test.get('trades', 0))}; top_month_share={f1_best_forced_test.get('top_month_abs_net_share', np.nan):.2%}",
                "implication": "Incluso el mejor candidato observado a posteriori en fold1 test sigue siendo pequeno/concentrado y no sirve para seleccion.",
            }
        )
    if not f0_test.empty:
        rows.append(
            {
                "finding": "fold0_candidate_positive_but_concentrated",
                "evidence": f"best fold0 test net={f0_test['net_return']:.6f}; avg trade={f0_test['avg_trade_net']:.6f}; top_month_share={f0_test.get('top_month_abs_net_share', np.nan):.2%}",
                "implication": "El mejor resultado economico no es aceptable como estrategia porque depende demasiado de pocos meses.",
            }
        )
    if not f0_stress.empty and not f1_same_stress.empty:
        rows.append(
            {
                "finding": "stress_survival_not_stable",
                "evidence": f"best fold0 test candidate stress net={f0_stress['net_return']:.6f}; best fold1 validation candidate stress net={f1_same_stress['net_return']:.6f}",
                "implication": "La robustez a costes no es estable fold-by-fold; no hay margen suficiente para HMM overlay.",
            }
        )
    for fold in [0, 1]:
        month, net, share = _max_month_share(monthly, fold, "test")
        if month:
            rows.append(
                {
                    "finding": f"fold{fold}_test_month_concentration",
                    "evidence": f"largest absolute month={month}; net={net:.6f}; abs_share={share:.2%}",
                    "implication": "Debe penalizarse o bloquearse la familia si el edge depende de un solo tramo temporal.",
                }
            )
    for feature in ["risk_off_score", "positive_index_count_2", "positive_sector_count_2", "target_rel_volume_by_bar"]:
        if feature_shift.empty:
            continue
        f0 = feature_shift[
            feature_shift["split"].eq("test") & feature_shift["fold"].astype(int).eq(0) & feature_shift["feature"].eq(feature)
        ]
        f1 = feature_shift[
            feature_shift["split"].eq("test") & feature_shift["fold"].astype(int).eq(1) & feature_shift["feature"].eq(feature)
        ]
        if not f0.empty and not f1.empty:
            f0_mean = float(f0.iloc[0]["mean"])
            f1_mean = float(f1.iloc[0]["mean"])
            if pd.isna(f0_mean) or pd.isna(f1_mean):
                continue
            rows.append(
                {
                    "finding": f"active_feature_shift_{feature}",
                    "evidence": f"fold0 active mean={f0_mean:.6f}; fold1 active mean={f1_mean:.6f}",
                    "implication": "Comparar si los mismos filtros capturan un regimen economicamente distinto entre folds.",
                }
            )
    accepted_like = _accepted_like_candidates(evaluation, config)
    if accepted_like.empty:
        rows.append(
            {
                "finding": "hmm_still_blocked",
                "evidence": "No hay candidato aceptado: falla estabilidad fold-by-fold y concentracion temporal.",
                "implication": "No usar HMM para rescatar esta familia; antes hay que resolver concentracion o cambiar hipotesis.",
            }
        )
    else:
        best = accepted_like.sort_values(["net_return", "avg_trade_net"], ascending=[False, False], kind="stable").iloc[0]
        rows.append(
            {
                "finding": "focused_candidate_accepted_but_family_unstable",
                "evidence": f"selected candidate={best['candidate_id']}; fold={int(best['fold'])}; test net={best['net_return']:.6f}; avg trade={best['avg_trade_net']:.6f}; trades={int(best['trades'])}; stress net={best['stress_net_return']:.6f}",
                "implication": "Primer candidato operable bajo reglas actuales, pero no es estrategia final hasta resolver estabilidad por folds y controles no-HMM/HMM.",
            }
        )
    return pd.DataFrame(rows)


def render_report(
    target_symbol: str,
    focus_specs: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    hourly_summary: pd.DataFrame,
    feature_shift: pd.DataFrame,
    findings: pd.DataFrame,
    outputs: dict[str, Path],
    config: dict[str, Any],
    *,
    max_rows: int = 80,
) -> str:
    focus = configured_focus(config)
    primary = primary_cost_name(config)
    setup_eval = evaluation[evaluation["bucket"].eq("setup_signal") & evaluation["cost_scenario"].isin([primary, stress_cost_name(config)])].copy()
    setup_eval = setup_eval.sort_values(["split", "fold", "cost_scenario", "net_return"], ascending=[True, True, True, False], kind="stable")
    month_top = monthly_summary.sort_values(["split", "fold", "net_return"], ascending=[True, True, False], kind="stable")
    day_top = daily_summary.sort_values(["split", "fold", "net_return"], ascending=[True, True, False], kind="stable")
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    accepted_like = _accepted_like_candidates(evaluation, config)
    if accepted_like.empty:
        conclusion = (
            f"The focused `{focus['family']}` {focus['direction']} h{focus['horizon_bars']} branch is not accepted. "
            "The next move should be a new hypothesis/timeframe/target; HMM remains blocked for this family."
        )
    else:
        accepted_folds = ", ".join(str(int(value)) for value in sorted(accepted_like["fold"].unique()))
        conclusion = (
            f"The focused `{focus['family']}` {focus['direction']} h{focus['horizon_bars']} branch has an accepted-like "
            f"selected candidate under the configured primary/conservative/stress costs, currently only in fold(s) {accepted_folds}. "
            "Treat it as a candidate under stability review, not as a final strategy: the broader family still lacks fold-by-fold "
            "stress survival and must be compared against no-HMM controls before any HMM overlay."
        )
    return f"""# Setup Signal Diagnostics - {target_symbol.upper()}

## Scope

- Focus family: `{focus["family"]}`.
- Direction: `{focus["direction"]}`.
- Horizon bars: `{focus["horizon_bars"]}`.
- Primary cost: `{primary}`.
- Purpose: explain whether the focused setup family deserves more work as a candidate.

## Findings

{_markdown_table(findings, max_rows=max_rows)}

## Focus Specs

{_markdown_table(focus_specs, max_rows=max_rows)}

## Candidate Evaluation

{_markdown_table(setup_eval, max_rows=max_rows)}

## Candidate Summary

{_markdown_table(candidate_summary, max_rows=max_rows)}

## Monthly Summary

{_markdown_table(month_top, max_rows=max_rows)}

## Daily Summary

{_markdown_table(day_top, max_rows=max_rows)}

## Hourly Summary

{_markdown_table(hourly_summary, max_rows=max_rows)}

## Active Feature Shift

{_markdown_table(feature_shift, max_rows=max_rows)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    cfg = _diag_cfg(config)
    paths = default_input_paths(config, target)
    validation = pd.read_parquet(paths["validation"])
    decisions = pd.read_parquet(paths["decisions"])
    dataset = build_signal_dataset(config, target)
    focus_specs = select_focus_specs(validation, decisions, config)
    evaluation = evaluate_focus_specs(dataset, focus_specs, config)
    feature_columns = [str(value) for value in cfg.get("feature_columns", DEFAULT_FEATURE_COLUMNS)]
    primary = cost_scenarios(config, [primary_cost_name(config)])[0]
    bar_returns = reconstruct_bar_returns(dataset, focus_specs, primary, feature_columns=feature_columns)
    candidate_summary = summarize_returns(bar_returns, ["split", "fold", "candidate_id", "validation_status", "was_selected"])
    monthly_summary = summarize_returns(bar_returns, ["split", "fold", "month"])
    daily_summary = summarize_returns(bar_returns, ["split", "fold", "date"])
    hourly_summary = summarize_returns(bar_returns, ["split", "fold", "hour"])
    features = feature_shift_summary(bar_returns, feature_columns)
    findings = diagnostic_findings(evaluation, monthly_summary, features, config)

    results_dir = results_output_dir(config, target)
    outputs = {
        "setup_signal_focus_specs": results_dir / "setup_signal_focus_specs.parquet",
        "setup_signal_focus_evaluation": results_dir / "setup_signal_focus_evaluation.parquet",
        "setup_signal_focus_bar_returns": results_dir / "setup_signal_focus_bar_returns.parquet",
        "setup_signal_focus_candidate_summary": results_dir / "setup_signal_focus_candidate_summary.parquet",
        "setup_signal_focus_monthly": results_dir / "setup_signal_focus_monthly.parquet",
        "setup_signal_focus_daily": results_dir / "setup_signal_focus_daily.parquet",
        "setup_signal_focus_hourly": results_dir / "setup_signal_focus_hourly.parquet",
        "setup_signal_focus_feature_shift": results_dir / "setup_signal_focus_feature_shift.parquet",
        "setup_signal_focus_findings": results_dir / "setup_signal_focus_findings.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    focus_specs.to_parquet(outputs["setup_signal_focus_specs"], index=False)
    evaluation.to_parquet(outputs["setup_signal_focus_evaluation"], index=False)
    bar_returns.to_parquet(outputs["setup_signal_focus_bar_returns"], index=False)
    candidate_summary.to_parquet(outputs["setup_signal_focus_candidate_summary"], index=False)
    monthly_summary.to_parquet(outputs["setup_signal_focus_monthly"], index=False)
    daily_summary.to_parquet(outputs["setup_signal_focus_daily"], index=False)
    hourly_summary.to_parquet(outputs["setup_signal_focus_hourly"], index=False)
    features.to_parquet(outputs["setup_signal_focus_feature_shift"], index=False)
    findings.to_parquet(outputs["setup_signal_focus_findings"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            target,
            focus_specs,
            evaluation,
            candidate_summary,
            monthly_summary,
            daily_summary,
            hourly_summary,
            features,
            findings,
            outputs,
            config,
            max_rows=int(cfg.get("report_top_rows", 80)),
        ),
        encoding="utf-8",
    )
    return report_path, outputs["setup_signal_focus_findings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose focused setup signal candidates across folds.")
    parser.add_argument("--config", default="configs/hmm_lab_15min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, findings_path = run(args.config, args.target)
    print(f"Setup signal diagnostics report written to: {report_path}")
    print(f"Setup signal diagnostics findings written to: {findings_path}")


if __name__ == "__main__":
    main()

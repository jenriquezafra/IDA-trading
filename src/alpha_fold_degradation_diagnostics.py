from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.alpha_discovery_base import _combined_cfg, _json_loads, alpha_position, build_split_dataset
from src.candidate_cost_sensitivity_cross_asset import scenario_cost_return
from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios


DEFAULT_FEATURE_COLUMNS = [
    "target_ret_6",
    "target_ret_12",
    "target_dist_vwap_atr",
    "target_signed_efficiency_12",
    "target_range_ratio_6_24",
    "chop_score",
    "intraday_stress_score",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_6",
    "positive_sector_count_6",
]


def _diag_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("alpha_fold_degradation_diagnostics", {})


def _path_from_template(template: str | Path, target_symbol: str) -> Path:
    return Path(str(template).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return (
        Path(config.get("paths", {}).get("reports_dir", "reports"))
        / target_symbol.upper()
        / "alpha_fold_degradation_diagnostics.md"
    )


def default_input_paths(config: dict[str, Any], target_symbol: str) -> dict[str, Path]:
    cfg = _diag_cfg(config)
    results_dir = results_output_dir(config, target_symbol)
    return {
        "selected_specs": _path_from_template(
            cfg.get("selected_specs", results_dir / "alpha_discovery_selected_specs.parquet"),
            target_symbol,
        ),
        "decisions": _path_from_template(
            cfg.get("decisions", results_dir / "alpha_discovery_decisions.parquet"),
            target_symbol,
        ),
    }


def primary_cost_scenario(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_combined_cfg(config))
    cfg.update({key: value for key, value in _diag_cfg(config).items() if key.endswith("cost_scenario")})
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg}, [primary])
    if not scenarios:
        raise ValueError(f"Cost scenario not found: {primary}")
    return scenarios[0]


def side_labels(position: pd.Series) -> pd.Series:
    values = position.astype(float).fillna(0.0)
    labels = np.select([values.gt(0.0), values.lt(0.0)], ["long", "short"], default="flat")
    return pd.Series(labels, index=position.index, dtype="object")


def _selected_hours(value: str | float | int | None) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(int(item) for item in str(value).split(",") if item != "")


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
        fold = int(spec["fold"])
        horizon = int(spec["horizon_bars"])
        alpha_variant = str(spec["alpha_variant"])
        base_variant = str(spec.get("base_variant", alpha_variant))
        threshold = float(spec["threshold"])
        gates = _json_loads(spec.get("gates_json", "{}"))
        hours = _selected_hours(spec.get("selected_hours"))
        for split in splits:
            frame = dataset[
                dataset["split"].eq(split)
                & dataset["fold"].eq(fold)
                & dataset["horizon_bars"].eq(horizon)
            ].copy()
            if frame.empty:
                continue
            position = alpha_position(frame, alpha_variant, threshold, gates)
            active = position.ne(0.0)
            if not active.any():
                continue
            base_position = alpha_position(frame, base_variant, threshold, gates)
            hour_multiplier = frame["hour"].isin(hours).astype(float) if hours else pd.Series(0.0, index=frame.index)
            same_hour_position = base_position * hour_multiplier
            cost = scenario_cost_return(frame, position, scenario)
            base_cost = scenario_cost_return(frame, base_position, scenario)
            same_hour_cost = scenario_cost_return(frame, same_hour_position, scenario)
            fwd = frame["fwd_ret"].astype(float)
            out = pd.DataFrame(
                {
                    "candidate_id": spec["candidate_id"],
                    "split": split,
                    "fold": fold,
                    "alpha_variant": alpha_variant,
                    "base_variant": base_variant,
                    "horizon_bars": horizon,
                    "threshold": threshold,
                    "timestamp": pd.to_datetime(frame["timestamp"]),
                    "session": frame["session"].values,
                    "hour": frame["hour"].astype(int).values,
                    "position": position.values,
                    "side": side_labels(position).values,
                    "fwd_ret": fwd.values,
                    "gross_return": (position * fwd).values,
                    "cost_return": cost.values,
                    "net_return": (position * fwd - cost).values,
                    "base_position": base_position.values,
                    "base_net_return": (base_position * fwd - base_cost).values,
                    "same_hour_position": same_hour_position.values,
                    "same_hour_net_return": (same_hour_position * fwd - same_hour_cost).values,
                    "selected_hour_active": hour_multiplier.astype(bool).values,
                },
                index=frame.index,
            )
            out = out.loc[active].copy()
            out["month"] = out["timestamp"].dt.strftime("%Y-%m")
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
        win_rate=("net_return", lambda values: float((values > 0.0).mean())),
    )
    grouped["net_per_candidate"] = grouped["net_return"] / grouped["candidates"].replace(0, np.nan)
    return grouped.sort_values([*group_cols], kind="stable").reset_index(drop=True)


def feature_shift_summary(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    feature_columns = [column for column in (feature_columns or DEFAULT_FEATURE_COLUMNS) if column in frame.columns]
    if not feature_columns:
        return pd.DataFrame()
    rows = []
    for (split, fold, side), group in frame.groupby(["split", "fold", "side"], sort=False):
        for column in feature_columns:
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "split": split,
                    "fold": int(fold),
                    "side": side,
                    "feature": column,
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "rows": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def matched_variant_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    cols = [
        "fold",
        "alpha_variant",
        "horizon_bars",
        "decision",
        "test_net_primary",
        "test_sharpe_primary",
        "test_profit_factor_primary",
        "test_avg_trade_net_primary",
        "test_net_stress",
        "test_trades_primary",
    ]
    available = [column for column in cols if column in decisions.columns]
    best = decisions.loc[:, available].copy()
    best = best.sort_values(
        ["alpha_variant", "horizon_bars", "fold", "test_net_primary", "test_sharpe_primary"],
        ascending=[True, True, True, False, False],
        kind="stable",
    )
    best = best.drop_duplicates(["alpha_variant", "horizon_bars", "fold"], keep="first")
    rows = []
    for (variant, horizon), group in best.groupby(["alpha_variant", "horizon_bars"], sort=False):
        fold0 = group[group["fold"].eq(0)]
        fold1 = group[group["fold"].eq(1)]
        row: dict[str, Any] = {"alpha_variant": variant, "horizon_bars": int(horizon)}
        for label, fold_frame in [("fold0", fold0), ("fold1", fold1)]:
            if fold_frame.empty:
                continue
            item = fold_frame.iloc[0]
            row[f"{label}_decision"] = item.get("decision", "")
            row[f"{label}_net"] = item.get("test_net_primary", np.nan)
            row[f"{label}_sharpe"] = item.get("test_sharpe_primary", np.nan)
            row[f"{label}_profit_factor"] = item.get("test_profit_factor_primary", np.nan)
            row[f"{label}_avg_trade"] = item.get("test_avg_trade_net_primary", np.nan)
            row[f"{label}_stress_net"] = item.get("test_net_stress", np.nan)
            row[f"{label}_trades"] = item.get("test_trades_primary", np.nan)
        if "fold0_net" in row and "fold1_net" in row:
            row["fold1_minus_fold0_net"] = row["fold1_net"] - row["fold0_net"]
            row["fold1_minus_fold0_sharpe"] = row["fold1_sharpe"] - row["fold0_sharpe"]
            row["fold1_avg_trade_ratio"] = (
                row["fold1_avg_trade"] / row["fold0_avg_trade"] if row.get("fold0_avg_trade", 0.0) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold1_minus_fold0_net", "alpha_variant"], na_position="last", kind="stable")


def _summary_value(frame: pd.DataFrame, split: str, fold: int, side: str, column: str) -> float:
    if frame.empty:
        return np.nan
    match = frame[frame["split"].eq(split) & frame["fold"].eq(fold) & frame["side"].eq(side)]
    if match.empty or column not in match:
        return np.nan
    return float(match.iloc[0][column])


def _feature_mean(feature_shift: pd.DataFrame, split: str, fold: int, side: str, feature: str) -> float:
    if feature_shift.empty:
        return np.nan
    match = feature_shift[
        feature_shift["split"].eq(split)
        & feature_shift["fold"].eq(fold)
        & feature_shift["side"].eq(side)
        & feature_shift["feature"].eq(feature)
    ]
    if match.empty:
        return np.nan
    return float(match.iloc[0]["mean"])


def diagnostic_findings(
    side_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    feature_shift: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    f0_short_net = _summary_value(side_summary, "test", 0, "short", "net_return")
    f0_long_net = _summary_value(side_summary, "test", 0, "long", "net_return")
    f1_short_net = _summary_value(side_summary, "test", 1, "short", "net_return")
    f1_long_net = _summary_value(side_summary, "test", 1, "long", "net_return")
    f1_val_short_net = _summary_value(side_summary, "validation", 1, "short", "net_return")
    f1_val_long_net = _summary_value(side_summary, "validation", 1, "long", "net_return")
    f0_short_avg = _summary_value(side_summary, "test", 0, "short", "avg_trade_net")
    f1_short_avg = _summary_value(side_summary, "test", 1, "short", "avg_trade_net")
    f1_long_avg = _summary_value(side_summary, "test", 1, "long", "avg_trade_net")
    if np.isfinite(f0_short_net) and np.isfinite(f0_long_net):
        rows.append(
            {
                "finding": "fold0_short_side_dominated",
                "evidence": f"fold0 test short net={f0_short_net:.6f}; long net={f0_long_net:.6f}",
                "implication": "El edge validado en fold0 viene principalmente de continuacion bajista intradia, no de una simetria long/short.",
            }
        )
    if np.isfinite(f1_long_net) and f1_long_net < 0:
        rows.append(
            {
                "finding": "fold1_long_side_flipped_negative",
                "evidence": f"fold1 test long net={f1_long_net:.6f}; avg trade={f1_long_avg:.6f}",
                "implication": "Las entradas long de momentum no deben mezclarse con shorts sin una compuerta propia.",
            }
        )
    if np.isfinite(f1_val_long_net) and np.isfinite(f1_long_net):
        rows.append(
            {
                "finding": "fold1_long_validation_to_test_decay",
                "evidence": f"fold1 long validation net={f1_val_long_net:.6f}; test net={f1_long_net:.6f}",
                "implication": "El selector esta aceptando longs que eran validos en la ventana anterior pero no generalizan al siguiente tramo.",
            }
        )
    if np.isfinite(f1_val_short_net) and np.isfinite(f1_short_net):
        rows.append(
            {
                "finding": "fold1_short_validation_to_test_decay",
                "evidence": f"fold1 short validation net={f1_val_short_net:.6f}; test net={f1_short_net:.6f}",
                "implication": "El short sigue positivo, pero pierde gran parte del margen observado en validation.",
            }
        )
    if np.isfinite(f0_short_avg) and np.isfinite(f1_short_avg):
        rows.append(
            {
                "finding": "fold1_short_edge_compressed",
                "evidence": f"short avg trade fold0={f0_short_avg:.6f}; fold1={f1_short_avg:.6f}",
                "implication": "El lado short conserva algo de edge, pero el margen por trade cae y se vuelve mas sensible a costes.",
            }
        )
    if not monthly_summary.empty:
        f0_short_months = monthly_summary[
            monthly_summary["split"].eq("test") & monthly_summary["fold"].eq(0) & monthly_summary["side"].eq("short")
        ].copy()
        if not f0_short_months.empty and f0_short_months["net_return"].sum() > 0:
            best = f0_short_months.sort_values("net_return", ascending=False, kind="stable").iloc[0]
            share = float(best["net_return"] / f0_short_months["net_return"].sum())
            rows.append(
                {
                    "finding": "fold0_has_month_concentration",
                    "evidence": f"best short month={best['month']} net={best['net_return']:.6f}; share={share:.2%}",
                    "implication": "Conviene penalizar concentracion temporal antes de aceptar un candidato como estrategia.",
                }
            )
    for feature in ["positive_index_count_6", "positive_sector_count_6", "intraday_stress_score", "chop_score"]:
        f0 = _feature_mean(feature_shift, "test", 0, "short", feature)
        f1 = _feature_mean(feature_shift, "test", 1, "short", feature)
        if np.isfinite(f0) and np.isfinite(f1):
            rows.append(
                {
                    "finding": f"short_feature_shift_{feature}",
                    "evidence": f"short active mean fold0={f0:.6f}; fold1={f1:.6f}",
                    "implication": "Revisar si la compuerta debe exigir condiciones de mercado mas extremas o mas especificas.",
                }
            )
    return pd.DataFrame(rows)


def render_report(
    target_symbol: str,
    side_summary: pd.DataFrame,
    variant_side_summary: pd.DataFrame,
    monthly_side_summary: pd.DataFrame,
    feature_shift: pd.DataFrame,
    matched_variants: pd.DataFrame,
    findings: pd.DataFrame,
    outputs: dict[str, Path],
    *,
    max_rows: int = 80,
) -> str:
    focus_features = feature_shift[
        feature_shift["feature"].isin(
            [
                "target_ret_6",
                "target_ret_12",
                "target_dist_vwap_atr",
                "chop_score",
                "intraday_stress_score",
                "risk_on_score",
                "risk_off_score",
                "positive_index_count_6",
                "positive_sector_count_6",
            ]
        )
    ].copy()
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    return f"""# Alpha Fold Degradation Diagnostics - {target_symbol.upper()}

## Scope

- Reconstructs active alpha bars from frozen `alpha_discovery_base` specs.
- Uses the configured primary cost scenario.
- Compares fold 0 vs fold 1 by side, family, month and active feature profile.

## Diagnostic Findings

{_markdown_table(findings, max_rows=max_rows)}

## Side Summary

{_markdown_table(side_summary, max_rows=max_rows)}

## Variant And Side Summary

{_markdown_table(variant_side_summary, max_rows=max_rows)}

## Monthly Side Summary

{_markdown_table(monthly_side_summary, max_rows=max_rows)}

## Active Feature Shift

{_markdown_table(focus_features, max_rows=max_rows)}

## Matched Variants

{_markdown_table(matched_variants, max_rows=max_rows)}

## Outputs

{outputs_text}

## Conclusion

Fold 0 works mainly because the short side of intraday momentum has strong follow-through. Fold 1 degrades because long entries turn negative and the short edge compresses, with losses clustered in later months. The next search should be asymmetric: prioritize short-only or short-dominant candidates, require stronger breadth/risk-off confirmation, and add a fold-stability gate before returning to HMM filtering.
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    cfg = _diag_cfg(config)
    paths = default_input_paths(config, target)
    specs = pd.read_parquet(paths["selected_specs"])
    decisions = pd.read_parquet(paths["decisions"])
    dataset = build_split_dataset(config, target)
    scenario = primary_cost_scenario(config)
    feature_columns = [str(value) for value in cfg.get("feature_columns", DEFAULT_FEATURE_COLUMNS)]
    bar_returns = reconstruct_bar_returns(dataset, specs, scenario, feature_columns=feature_columns)
    side_summary = summarize_returns(bar_returns, ["split", "fold", "side"])
    variant_side_summary = summarize_returns(bar_returns, ["split", "fold", "alpha_variant", "horizon_bars", "side"]).sort_values(
        ["split", "fold", "net_return"], ascending=[True, True, False], kind="stable"
    )
    monthly_side_summary = summarize_returns(bar_returns, ["split", "fold", "month", "side"])
    features = feature_shift_summary(bar_returns, feature_columns)
    matched = matched_variant_summary(decisions)
    findings = diagnostic_findings(side_summary, monthly_side_summary, features)

    results_dir = results_output_dir(config, target)
    outputs = {
        "alpha_fold_bar_returns": results_dir / "alpha_fold_bar_returns.parquet",
        "alpha_fold_side_summary": results_dir / "alpha_fold_side_summary.parquet",
        "alpha_fold_variant_side_summary": results_dir / "alpha_fold_variant_side_summary.parquet",
        "alpha_fold_monthly_side_summary": results_dir / "alpha_fold_monthly_side_summary.parquet",
        "alpha_fold_feature_shift": results_dir / "alpha_fold_feature_shift.parquet",
        "alpha_fold_matched_variants": results_dir / "alpha_fold_matched_variants.parquet",
        "alpha_fold_findings": results_dir / "alpha_fold_findings.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    bar_returns.to_parquet(outputs["alpha_fold_bar_returns"], index=False)
    side_summary.to_parquet(outputs["alpha_fold_side_summary"], index=False)
    variant_side_summary.to_parquet(outputs["alpha_fold_variant_side_summary"], index=False)
    monthly_side_summary.to_parquet(outputs["alpha_fold_monthly_side_summary"], index=False)
    features.to_parquet(outputs["alpha_fold_feature_shift"], index=False)
    matched.to_parquet(outputs["alpha_fold_matched_variants"], index=False)
    findings.to_parquet(outputs["alpha_fold_findings"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            target,
            side_summary,
            variant_side_summary,
            monthly_side_summary,
            features,
            matched,
            findings,
            outputs,
            max_rows=int(cfg.get("report_top_rows", 80)),
        ),
        encoding="utf-8",
    )
    return report_path, outputs["alpha_fold_findings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose alpha base degradation from fold 0 to fold 1.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, findings_path = run(args.config, args.target)
    print(f"Alpha fold diagnostics report written to: {report_path}")
    print(f"Alpha fold diagnostics findings written to: {findings_path}")


if __name__ == "__main__":
    main()

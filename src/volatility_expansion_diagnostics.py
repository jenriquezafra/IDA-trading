from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import scenario_cost_return
from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.volatility_expansion_search import (
    _control_positions,
    _cost_scenarios,
    _json_loads,
    _search_cfg,
    build_split_dataset,
)


DEFAULT_FEATURE_COLUMNS = [
    "prior_target_range_ratio_2_8",
    "prior_target_rv_4_rel_by_bar",
    "target_breakout_margin_roll_high_4_atr",
    "target_close_location_bar",
    "target_rel_volume_by_bar",
    "target_rel_volume_accel_2",
    "positive_index_count_2",
    "positive_sector_count_2",
    "index_above_vwap_count",
    "sector_above_vwap_count",
    "spread_credit_12",
    "risk_on_score",
    "risk_off_score",
    "intraday_stress_score",
    "cross_asset_vol_expansion_score",
]


def _diag_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("volatility_expansion_diagnostics", {})


def _path_from_template(template: str | Path, target_symbol: str) -> Path:
    return Path(str(template).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return (
        Path(config.get("paths", {}).get("reports_dir", "reports"))
        / target_symbol.upper()
        / "volatility_expansion_diagnostics.md"
    )


def default_input_paths(config: dict[str, Any], target_symbol: str) -> dict[str, Path]:
    cfg = _diag_cfg(config)
    results_dir = results_output_dir(config, target_symbol)
    return {
        "selected_specs": _path_from_template(
            cfg.get("selected_specs", results_dir / "volatility_expansion_selected_specs.parquet"),
            target_symbol,
        ),
        "decisions": _path_from_template(
            cfg.get("decisions", results_dir / "volatility_expansion_decisions.parquet"),
            target_symbol,
        ),
    }


def _scenario_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    scenarios = _cost_scenarios(config, [name])
    if not scenarios:
        raise ValueError(f"Cost scenario not found: {name}")
    return scenarios[0]


def diagnostic_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    search_cfg = _search_cfg(config)
    diag_cfg = _diag_cfg(config)
    names = [
        str(diag_cfg.get("primary_cost_scenario", search_cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))),
        str(diag_cfg.get("stress_cost_scenario", search_cfg.get("stress_cost_scenario", "bps_5"))),
    ]
    unique_names = list(dict.fromkeys(names))
    return [_scenario_by_name(config, name) for name in unique_names]


def side_labels(position: pd.Series) -> pd.Series:
    values = position.astype(float).fillna(0.0)
    labels = np.select([values.gt(0.0), values.lt(0.0)], ["long", "short"], default="flat")
    return pd.Series(labels, index=position.index, dtype="object")


def select_focus_specs(specs: pd.DataFrame, decisions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if specs.empty or decisions.empty:
        return pd.DataFrame()
    cfg = _diag_cfg(config)
    candidate_status = str(cfg.get("validation_status", "volatility_expansion_validation_candidate"))
    focus_decisions = [str(value) for value in cfg.get("focus_decisions", ["accepted_candidate", "cost_fragile", "research_candidate", "rejected"])]
    allowed_decisions = set(focus_decisions)
    decision_rank = {decision: rank for rank, decision in enumerate(focus_decisions)}
    decision_rank.setdefault("rejected_validation_failed", len(decision_rank))
    eligible = decisions[decisions["validation_status"].eq(candidate_status)].copy()
    if allowed_decisions:
        eligible = eligible[eligible["decision"].isin(allowed_decisions)].copy()
    if eligible.empty:
        eligible = decisions[decisions["validation_status"].eq(candidate_status)].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible["decision_rank"] = eligible["decision"].map(decision_rank).fillna(99).astype(int)
    eligible = eligible.sort_values(
        ["decision_rank", "test_net_primary", "test_sharpe_primary"],
        ascending=[True, False, False],
        kind="stable",
    )
    max_candidates = int(cfg.get("max_candidates", 12))
    focus_ids = eligible["candidate_id"].drop_duplicates().head(max_candidates)
    rank_by_id = {candidate_id: rank for rank, candidate_id in enumerate(focus_ids)}
    focused = specs[specs["candidate_id"].isin(focus_ids)].copy()
    focused["_focus_rank"] = focused["candidate_id"].map(rank_by_id).fillna(len(rank_by_id)).astype(int)
    return focused.sort_values("_focus_rank", kind="stable").drop(columns=["_focus_rank"]).reset_index(drop=True)


def reconstruct_bucket_trades(
    dataset: pd.DataFrame,
    specs: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if dataset.empty or specs.empty or not scenarios:
        return pd.DataFrame()
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    rows: list[pd.DataFrame] = []
    for _, spec in specs.iterrows():
        spec_dict = dict(spec)
        thresholds = _json_loads(spec_dict["thresholds_json"])
        candidate_id = str(spec_dict["candidate_id"])
        for split in splits:
            frame = dataset[
                dataset["split"].eq(split)
                & dataset["fold"].eq(int(spec_dict["fold"]))
                & dataset["horizon_bars"].eq(int(spec_dict["horizon_bars"]))
            ].copy()
            if frame.empty:
                continue
            positions = _control_positions(frame, spec_dict, thresholds, config, candidate_id, split)
            for scenario in scenarios:
                for bucket, position in positions.items():
                    active = position.abs().gt(0.0)
                    if not active.any():
                        continue
                    cost = scenario_cost_return(frame, position, scenario)
                    fwd = frame["fwd_ret"].astype(float)
                    out = pd.DataFrame(
                        {
                            "candidate_id": candidate_id,
                            "split": split,
                            "fold": int(spec_dict["fold"]),
                            "variant": str(spec_dict["variant"]),
                            "side": str(spec_dict["side"]),
                            "horizon_bars": int(spec_dict["horizon_bars"]),
                            "hour_filter_name": str(spec_dict.get("hour_filter_name", "all")),
                            "bucket": bucket,
                            "cost_scenario": str(scenario["cost_scenario"]),
                            "timestamp": pd.to_datetime(frame["timestamp"]),
                            "session": frame["session"].values,
                            "bar_index": frame["bar_index"].astype(int).values,
                            "hour": frame["hour"].astype(int).values,
                            "position": position.astype(float).values,
                            "position_side": side_labels(position).values,
                            "fwd_ret": fwd.values,
                            "gross_return": (position * fwd).values,
                            "cost_return": cost.values,
                            "net_return": (position * fwd - cost).values,
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


def _top_abs_share(values: pd.Series) -> float:
    total = float(values.abs().sum())
    if total <= 0.0:
        return 0.0
    return float(values.abs().max() / total)


def summarize_trades(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(group_cols, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        daily = group.groupby("session", sort=False)["net_return"].sum()
        trades = int(len(group))
        row = {column: value for column, value in zip(group_cols, key_values, strict=True)}
        row.update(
            {
                "candidates": int(group["candidate_id"].nunique()),
                "trades": trades,
                "net_return": float(group["net_return"].sum()),
                "gross_return": float(group["gross_return"].sum()),
                "cost_return": float(group["cost_return"].sum()),
                "avg_trade_net": float(group["net_return"].mean()) if trades else 0.0,
                "win_rate": float(group["net_return"].gt(0.0).mean()) if trades else 0.0,
                "effective_cost_bps": float(group["cost_return"].sum() / trades * 10_000.0) if trades else np.nan,
                "top_day_abs_net_share": _top_abs_share(daily),
                "best_day_net": float(daily.max()) if not daily.empty else 0.0,
                "worst_day_net": float(daily.min()) if not daily.empty else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols, kind="stable").reset_index(drop=True)


def control_delta_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keys = ["candidate_id", "split", "cost_scenario"]
    required = [*keys, "bucket", "trades", "net_return", "avg_trade_net", "win_rate", "effective_cost_bps"]
    available = [column for column in required if column in summary.columns]
    source = summary.loc[:, available].copy()
    alpha = source[source["bucket"].eq("alpha_signal")].copy()
    if alpha.empty:
        return pd.DataFrame()
    controls = [
        ("breakout_only_control", "breakout"),
        ("compression_only_control", "compression"),
        ("same_hour_random_control", "random"),
        ("inverted_signal", "inverted"),
    ]
    for bucket, suffix in controls:
        control = source[source["bucket"].eq(bucket)].copy()
        if control.empty:
            continue
        rename = {
            "trades": f"{suffix}_trades",
            "net_return": f"{suffix}_net_return",
            "avg_trade_net": f"{suffix}_avg_trade_net",
            "win_rate": f"{suffix}_win_rate",
            "effective_cost_bps": f"{suffix}_effective_cost_bps",
        }
        alpha = alpha.merge(
            control.drop(columns=["bucket"]).rename(columns=rename),
            on=keys,
            how="left",
            validate="one_to_one",
        )
        alpha[f"net_delta_vs_{suffix}"] = alpha["net_return"] - alpha[f"{suffix}_net_return"]
        alpha[f"avg_trade_delta_vs_{suffix}"] = alpha["avg_trade_net"] - alpha[f"{suffix}_avg_trade_net"]
    return alpha.sort_values(["split", "cost_scenario", "net_return"], ascending=[True, True, False], kind="stable").reset_index(drop=True)


def feature_profile_summary(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    feature_columns = [column for column in (feature_columns or DEFAULT_FEATURE_COLUMNS) if column in frame.columns]
    if not feature_columns:
        return pd.DataFrame()
    active = frame[frame["bucket"].eq("alpha_signal")].copy()
    rows: list[dict[str, Any]] = []
    for (candidate_id, split, cost_scenario), group in active.groupby(["candidate_id", "split", "cost_scenario"], sort=False):
        for column in feature_columns:
            values = group[column].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "split": split,
                    "cost_scenario": cost_scenario,
                    "feature": column,
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "rows": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def candidate_failure_attribution(decisions: pd.DataFrame, deltas: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    search_cfg = _search_cfg(config)
    primary = str(search_cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    stress = str(search_cfg.get("stress_cost_scenario", "bps_5"))
    min_trades = int(search_cfg.get("min_trades", 35))
    max_top_day_share = float(search_cfg.get("max_top_day_abs_net_share", 0.35))
    rows: list[dict[str, Any]] = []
    for _, row in decisions.iterrows():
        candidate_id = str(row["candidate_id"])
        primary_delta = deltas[
            deltas["candidate_id"].eq(candidate_id)
            & deltas["split"].eq("test")
            & deltas["cost_scenario"].eq(primary)
        ]
        stress_delta = deltas[
            deltas["candidate_id"].eq(candidate_id)
            & deltas["split"].eq("test")
            & deltas["cost_scenario"].eq(stress)
        ]
        reasons = []
        if float(row.get("test_net_primary", np.nan)) <= 0.0:
            reasons.append("negative_primary_edge")
        if int(row.get("test_trades_primary", 0)) < min_trades:
            reasons.append("insufficient_test_trades")
        if float(row.get("test_avg_trade_net_primary", np.nan)) < 0.0005:
            reasons.append("thin_avg_trade_vs_5bps")
        if float(row.get("test_net_primary", np.nan)) > 0.0 and float(row.get("test_net_stress", np.nan)) <= 0.0:
            reasons.append("stress_cost_fragility")
        if float(row.get("test_net_delta_vs_random_primary", np.nan)) <= 0.0:
            reasons.append("random_control_stronger")
        if float(row.get("test_top_day_abs_net_share_primary", np.nan)) > max_top_day_share:
            reasons.append("day_concentration")
        if not primary_delta.empty:
            delta_row = primary_delta.iloc[0]
            if float(delta_row.get("net_delta_vs_breakout", np.nan)) <= 0.0:
                reasons.append("does_not_beat_breakout_control")
        if not reasons:
            reasons.append("passes_diagnostic_gates")
        rows.append(
            {
                "candidate_id": candidate_id,
                "decision": row.get("decision", ""),
                "fold": int(row.get("fold", -1)),
                "variant": row.get("variant", ""),
                "side": row.get("side", ""),
                "horizon_bars": int(row.get("horizon_bars", -1)),
                "hour_filter_name": row.get("hour_filter_name", "all"),
                "primary_net": row.get("test_net_primary", np.nan),
                "primary_avg_trade": row.get("test_avg_trade_net_primary", np.nan),
                "primary_trades": row.get("test_trades_primary", np.nan),
                "stress_net": row.get("test_net_stress", np.nan),
                "net_delta_vs_random": row.get("test_net_delta_vs_random_primary", np.nan),
                "net_delta_vs_breakout": row.get("test_net_delta_vs_breakout_primary", np.nan),
                "top_day_abs_net_share": row.get("test_top_day_abs_net_share_primary", np.nan),
                "diagnostic_reasons": ", ".join(reasons),
                "primary_effective_cost_bps": primary_delta.iloc[0].get("effective_cost_bps", np.nan) if not primary_delta.empty else np.nan,
                "stress_effective_cost_bps": stress_delta.iloc[0].get("effective_cost_bps", np.nan) if not stress_delta.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "primary_net"], ascending=[True, False], kind="stable")


def diagnostic_findings(attribution: pd.DataFrame, monthly: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not attribution.empty:
        cost_fragile = attribution[attribution["diagnostic_reasons"].str.contains("stress_cost_fragility", na=False)]
        if not cost_fragile.empty:
            item = cost_fragile.sort_values("primary_net", ascending=False, kind="stable").iloc[0]
            rows.append(
                {
                    "finding": "stress_cost_is_binding",
                    "evidence": f"{item['candidate_id']} primary={item['primary_net']:.6f}; stress={item['stress_net']:.6f}; avg_trade={item['primary_avg_trade']:.6f}",
                    "implication": "El margen por trade es demasiado pequeno para una estrategia que pueda absorber 5 bps.",
                }
            )
        random_stronger = attribution[attribution["diagnostic_reasons"].str.contains("random_control_stronger", na=False)]
        if not random_stronger.empty:
            item = random_stronger.sort_values("primary_net", ascending=False, kind="stable").iloc[0]
            rows.append(
                {
                    "finding": "random_matched_competes_with_alpha",
                    "evidence": f"{item['candidate_id']} delta_vs_random={item['net_delta_vs_random']:.6f}",
                    "implication": "Parte del efecto puede ser hora/regimen general, no el gatillo de compresion-breakout.",
                }
            )
        low_sample = attribution[attribution["diagnostic_reasons"].str.contains("insufficient_test_trades", na=False)]
        if not low_sample.empty:
            item = low_sample.sort_values("primary_net", ascending=False, kind="stable").iloc[0]
            rows.append(
                {
                    "finding": "sample_size_is_binding",
                    "evidence": f"{item['candidate_id']} trades={int(item['primary_trades'])}",
                    "implication": "El candidato positivo no tiene frecuencia suficiente para congelarlo sin ampliar muestra o relajar compuertas.",
                }
            )
    alpha_monthly = monthly[
        monthly["bucket"].eq("alpha_signal")
        & monthly["split"].eq("test")
        & monthly["cost_scenario"].astype(str).str.contains("ibkr", case=False, na=False)
    ].copy()
    if not alpha_monthly.empty:
        grouped = alpha_monthly.groupby("candidate_id", sort=False)
        for candidate_id, group in grouped:
            positive_total = float(group.loc[group["net_return"].gt(0.0), "net_return"].sum())
            if positive_total > 0:
                best = group.sort_values("net_return", ascending=False, kind="stable").iloc[0]
                share = float(best["net_return"] / positive_total)
                if share > 0.5:
                    rows.append(
                        {
                            "finding": "positive_edge_is_month_concentrated",
                            "evidence": f"{candidate_id} best_month={best['month']} net={best['net_return']:.6f}; share={share:.2%}",
                            "implication": "La rentabilidad positiva puede depender de una ventana corta.",
                        }
                    )
                    break
    alpha_hourly = hourly[
        hourly["bucket"].eq("alpha_signal")
        & hourly["split"].eq("test")
        & hourly["cost_scenario"].astype(str).str.contains("ibkr", case=False, na=False)
    ].copy()
    if not alpha_hourly.empty:
        by_hour = alpha_hourly.groupby("hour", as_index=False)["net_return"].sum().sort_values("net_return", ascending=False, kind="stable")
        if len(by_hour) > 1 and float(by_hour.iloc[0]["net_return"]) > 0.0:
            rows.append(
                {
                    "finding": "hour_filter_may_matter",
                    "evidence": f"best_hour={int(by_hour.iloc[0]['hour'])} net={by_hour.iloc[0]['net_return']:.6f}; worst_hour={int(by_hour.iloc[-1]['hour'])} net={by_hour.iloc[-1]['net_return']:.6f}",
                    "implication": "Antes de otro grid amplio, probar una compuerta horaria definida en validation.",
                }
            )
    return pd.DataFrame(rows)


def render_report(
    target_symbol: str,
    focus_specs: pd.DataFrame,
    attribution: pd.DataFrame,
    control_deltas: pd.DataFrame,
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    features: pd.DataFrame,
    findings: pd.DataFrame,
    outputs: dict[str, Path],
    *,
    max_rows: int = 80,
) -> str:
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    focus_cols = [
        "candidate_id",
        "fold",
        "variant",
        "side",
        "horizon_bars",
        "hour_filter_name",
        "compression_quantile",
        "candidate_status",
        "utility_score",
    ]
    focus_table = focus_specs.loc[:, [column for column in focus_cols if column in focus_specs.columns]] if not focus_specs.empty else pd.DataFrame()
    feature_focus = features[
        features["feature"].isin(
            [
                "prior_target_range_ratio_2_8",
                "prior_target_rv_4_rel_by_bar",
                "target_breakout_margin_roll_high_4_atr",
                "target_rel_volume_by_bar",
                "target_rel_volume_accel_2",
                "intraday_stress_score",
            ]
        )
    ].copy() if not features.empty else pd.DataFrame()
    accepted = attribution[attribution["decision"].eq("accepted_candidate")] if not attribution.empty else pd.DataFrame()
    if not accepted.empty:
        best = accepted.sort_values("primary_net", ascending=False, kind="stable").iloc[0]
        conclusion = (
            "At least one candidate passes the current diagnostic gates: "
            f"{best['candidate_id']} with primary_net={best['primary_net']:.6f}, "
            f"stress_net={best['stress_net']:.6f}, trades={int(best['primary_trades'])}. "
            "Treat it as a provisional accepted candidate pending additional robustness and stability checks."
        )
    else:
        conclusion = (
            "The current QQQ compression-breakout clue is useful for research, but not robust enough to promote. "
            "Next action should be a targeted repair: validation-defined hour/cost/sample gates around QQQ long h4, not a broad new search."
        )
    return f"""# Volatility Expansion Diagnostics - {target_symbol.upper()}

## Scope

- Reconstructs active trades from frozen `volatility_expansion_search` specs.
- Compares alpha against breakout-only, compression-only, same-hour random and inverted controls.
- Uses primary and stress cost scenarios.
- Focuses on validation-passing candidates selected by the previous block.

## Diagnostic Findings

{_markdown_table(findings, max_rows=max_rows)}

## Failure Attribution

{_markdown_table(attribution, max_rows=max_rows)}

## Focus Specs

{_markdown_table(focus_table, max_rows=max_rows)}

## Control Delta Summary

{_markdown_table(control_deltas, max_rows=max_rows)}

## Monthly Alpha Summary

{_markdown_table(monthly[monthly["bucket"].eq("alpha_signal")] if not monthly.empty else monthly, max_rows=max_rows)}

## Hourly Alpha Summary

{_markdown_table(hourly[hourly["bucket"].eq("alpha_signal")] if not hourly.empty else hourly, max_rows=max_rows)}

## Active Feature Profile

{_markdown_table(feature_focus, max_rows=max_rows)}

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
    specs = pd.read_parquet(paths["selected_specs"])
    decisions = pd.read_parquet(paths["decisions"])
    focus_specs = select_focus_specs(specs, decisions, config)
    dataset = build_split_dataset(config, target)
    scenarios = diagnostic_scenarios(config)
    feature_columns = [str(value) for value in cfg.get("feature_columns", DEFAULT_FEATURE_COLUMNS)]
    trades = reconstruct_bucket_trades(dataset, focus_specs, scenarios, config, feature_columns=feature_columns)
    bucket_summary = summarize_trades(trades, ["candidate_id", "split", "cost_scenario", "bucket"])
    control_deltas = control_delta_summary(bucket_summary)
    monthly = summarize_trades(trades, ["candidate_id", "split", "cost_scenario", "bucket", "month"])
    hourly = summarize_trades(trades, ["candidate_id", "split", "cost_scenario", "bucket", "hour"])
    features = feature_profile_summary(trades, feature_columns)
    focus_decisions = decisions[decisions["candidate_id"].isin(focus_specs["candidate_id"])] if not focus_specs.empty else pd.DataFrame()
    attribution = candidate_failure_attribution(focus_decisions, control_deltas, config)
    findings = diagnostic_findings(attribution, monthly, hourly)

    results_dir = results_output_dir(config, target)
    outputs = {
        "volatility_expansion_focus_specs": results_dir / "volatility_expansion_focus_specs.parquet",
        "volatility_expansion_diagnostic_trades": results_dir / "volatility_expansion_diagnostic_trades.parquet",
        "volatility_expansion_bucket_summary": results_dir / "volatility_expansion_bucket_summary.parquet",
        "volatility_expansion_control_deltas": results_dir / "volatility_expansion_control_deltas.parquet",
        "volatility_expansion_monthly_summary": results_dir / "volatility_expansion_monthly_summary.parquet",
        "volatility_expansion_hourly_summary": results_dir / "volatility_expansion_hourly_summary.parquet",
        "volatility_expansion_feature_profile": results_dir / "volatility_expansion_feature_profile.parquet",
        "volatility_expansion_failure_attribution": results_dir / "volatility_expansion_failure_attribution.parquet",
        "volatility_expansion_diagnostic_findings": results_dir / "volatility_expansion_diagnostic_findings.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    focus_specs.to_parquet(outputs["volatility_expansion_focus_specs"], index=False)
    trades.to_parquet(outputs["volatility_expansion_diagnostic_trades"], index=False)
    bucket_summary.to_parquet(outputs["volatility_expansion_bucket_summary"], index=False)
    control_deltas.to_parquet(outputs["volatility_expansion_control_deltas"], index=False)
    monthly.to_parquet(outputs["volatility_expansion_monthly_summary"], index=False)
    hourly.to_parquet(outputs["volatility_expansion_hourly_summary"], index=False)
    features.to_parquet(outputs["volatility_expansion_feature_profile"], index=False)
    attribution.to_parquet(outputs["volatility_expansion_failure_attribution"], index=False)
    findings.to_parquet(outputs["volatility_expansion_diagnostic_findings"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            target,
            focus_specs,
            attribution,
            control_deltas,
            monthly,
            hourly,
            features,
            findings,
            outputs,
            max_rows=int(cfg.get("report_top_rows", 80)),
        ),
        encoding="utf-8",
    )
    return report_path, outputs["volatility_expansion_failure_attribution"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose validation-passing volatility expansion candidates.")
    parser.add_argument("--config", default="configs/hmm_lab_15min_expansion.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, attribution_path = run(args.config, args.target)
    print(f"Volatility expansion diagnostics report written to: {report_path}")
    print(f"Volatility expansion failure attribution written to: {attribution_path}")


if __name__ == "__main__":
    main()

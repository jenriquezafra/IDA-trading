from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_risk_filter import (
    base_position,
    build_filter_dataset,
    evaluate_position,
    filter_multiplier,
    load_candidate_combos,
    split_combo_frame,
)
from src.hmm_state_interpretability_cross_asset import _markdown_table


METRIC_COLS = ["net_return", "daily_sharpe", "profit_factor", "avg_trade_net", "max_drawdown", "turnover", "trades"]
CONTROL_BUCKETS = ["base", "same_hour_control", "always_flat", "shuffled_state_control"]


def _comparison_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("cross_asset_vs_baselines", {})


def _path_from_template(template: str, target_symbol: str) -> Path:
    return Path(template.format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "cross_asset_vs_baselines.md"


def feature_group(feature_set: str) -> str:
    if feature_set == "target_only_frozen":
        return "target_only"
    if feature_set.startswith("cross_asset"):
        return "cross_asset"
    return "spy_only_frozen"


def ablation_proxy(feature_set: str) -> str:
    mapping = {
        "target_only_frozen": "no_cross_asset",
        "cross_asset_minimal": "compact_scores_only",
        "cross_asset_sectors": "no_bonds_credit_macro",
        "cross_asset_macro": "no_sector_leadership_raw",
        "cross_asset_full_core": "full_core",
    }
    return mapping.get(feature_set, "other")


def model_class_for_risk_row(row: pd.Series) -> str:
    bucket = str(row["bucket"])
    strategy = str(row.get("strategy", ""))
    group = feature_group(str(row.get("feature_set", "")))
    if bucket == "hmm_filter":
        return "target_only_hmm_filter" if group == "target_only" else "cross_asset_hmm_filter"
    if bucket == "base":
        if strategy == "supervised_simple":
            return "supervised_no_hmm"
        if strategy in {"momentum_simple", "reversion_simple", "vwap_location"}:
            return f"{strategy}_no_hmm"
        return "base_no_hmm"
    if bucket == "same_hour_control":
        return "same_hour_control"
    if bucket == "always_flat":
        return "always_flat"
    if bucket == "shuffled_state_control":
        return "shuffled_state_control"
    return bucket


def model_class_for_rule_row(row: pd.Series) -> str:
    bucket = str(row["bucket"])
    group = feature_group(str(row.get("feature_set", "")))
    if bucket == "hmm_state_rule":
        return "target_only_hmm_direct_rule" if group == "target_only" else "cross_asset_hmm_direct_rule"
    if bucket == "no_hmm_equivalent":
        return "target_only_rules_no_hmm" if group == "target_only" else "cross_asset_rules_no_hmm"
    if bucket == "same_hour_control":
        return "same_hour_control"
    return bucket


def _base_output_cols(extra: list[str] | None = None) -> list[str]:
    cols = [
        "source_family",
        "source_artifact",
        "comparison_id",
        "filter_id",
        "candidate_id",
        "feature_set",
        "feature_group",
        "ablation_proxy",
        "n_states",
        "seed",
        "fold",
        "split",
        "strategy",
        "filter_name",
        "bucket",
        "model_class",
        "horizon_bars",
        "cost_bps",
        "threshold",
        "selected_hours",
        "uses_hmm",
        "uses_cross_asset",
        "uses_time_control",
        "is_shuffled_control",
        *METRIC_COLS,
        "filter_status",
    ]
    return [*cols, *(extra or [])]


def _candidate_id_from_row(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['feature_set']}__k{int(row['n_states'])}__seed{int(row['seed'])}__fold{int(row['fold'])}"
        f"__{row['strategy']}__{row['filter_name']}__h{int(row['horizon_bars'])}__thr{float(row['threshold']):g}"
    )


def risk_filter_rows(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _comparison_cfg(config)
    results_dir = results_output_dir(config, target_symbol)
    decisions_path = _path_from_template(str(cfg.get("candidate_decisions", "results/{target_symbol}/candidate_decisions.parquet")), target_symbol)
    decisions = pd.read_parquet(decisions_path)
    ids = set(decisions["filter_id"].astype(str))
    frames = []
    for name in ["risk_filter_validation.parquet", "risk_filter_test.parquet"]:
        path = results_dir / name
        frame = pd.read_parquet(path)
        frame = frame[frame["filter_id"].astype(str).isin(ids)].copy()
        frame["source_family"] = "risk_filter"
        frame["source_artifact"] = str(path)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if rows.empty:
        return rows
    rows["candidate_id"] = rows.apply(_candidate_id_from_row, axis=1)
    rows["feature_group"] = rows["feature_set"].map(feature_group)
    rows["ablation_proxy"] = rows["feature_set"].map(ablation_proxy)
    rows["model_class"] = rows.apply(model_class_for_risk_row, axis=1)
    rows["comparison_id"] = rows["filter_id"]
    rows["uses_hmm"] = rows["bucket"].eq("hmm_filter")
    rows["uses_cross_asset"] = rows["feature_group"].eq("cross_asset")
    rows["uses_time_control"] = rows["bucket"].eq("same_hour_control")
    rows["is_shuffled_control"] = False
    return rows.loc[:, [col for col in _base_output_cols() if col in rows.columns]].reset_index(drop=True)


def state_rule_rows(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    results_dir = results_output_dir(config, target_symbol)
    frames = []
    for name in ["state_rules_validation.parquet", "state_rules_test.parquet"]:
        path = results_dir / name
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["source_family"] = "state_rules"
        frame["source_artifact"] = str(path)
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if rows.empty:
        return rows
    rows = rows.rename(columns={"rule_id": "filter_id", "rule_type": "strategy", "signal_threshold": "threshold"})
    rows["candidate_id"] = rows["candidate_state_id"]
    rows["feature_group"] = rows["feature_set"].map(feature_group)
    rows["ablation_proxy"] = rows["feature_set"].map(ablation_proxy)
    rows["filter_name"] = rows["proposed_label"]
    rows["model_class"] = rows.apply(model_class_for_rule_row, axis=1)
    rows["comparison_id"] = rows["filter_id"]
    rows["uses_hmm"] = rows["bucket"].eq("hmm_state_rule")
    rows["uses_cross_asset"] = rows["feature_group"].eq("cross_asset")
    rows["uses_time_control"] = rows["bucket"].eq("same_hour_control")
    rows["is_shuffled_control"] = False
    rows["filter_status"] = rows.get("rule_status", "")
    return rows.loc[:, [col for col in _base_output_cols(["proposed_label"]) if col in rows.columns]].reset_index(drop=True)


def spy_only_frozen_rows(config: dict[str, Any]) -> pd.DataFrame:
    cfg = _comparison_cfg(config)
    path = Path(cfg.get("spy_only_frozen_results", "baselines/spy_only_hmm/results.parquet"))
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path).copy()
    rows = pd.DataFrame(
        {
            "source_family": "spy_only_frozen",
            "source_artifact": str(path),
            "comparison_id": frame["id"].astype(str),
            "filter_id": frame["id"].astype(str),
            "candidate_id": frame["candidate_id"].astype(str).where(frame["candidate_id"].notna(), ""),
            "feature_set": frame.get("feature_set", pd.Series("spy_only_frozen", index=frame.index)).fillna("spy_only_frozen"),
            "feature_group": "spy_only_frozen",
            "ablation_proxy": "legacy_spy_only",
            "n_states": np.nan,
            "seed": np.nan,
            "fold": np.nan,
            "split": "test",
            "strategy": frame["result_group"].astype(str),
            "filter_name": frame["status"].astype(str),
            "bucket": frame["result_group"].astype(str),
            "model_class": np.where(frame["result_group"].eq("hmm_candidate_threshold"), "spy_only_hmm_frozen", frame["id"].astype(str)),
            "horizon_bars": np.nan,
            "cost_bps": frame["cost_bps"].astype(float),
            "threshold": np.nan,
            "selected_hours": "",
            "uses_hmm": frame["result_group"].eq("hmm_candidate_threshold"),
            "uses_cross_asset": False,
            "uses_time_control": False,
            "is_shuffled_control": False,
            "net_return": frame["net_return"],
            "daily_sharpe": frame["daily_sharpe_net"],
            "profit_factor": frame["profit_factor_net"],
            "avg_trade_net": frame["avg_trade_net"],
            "max_drawdown": frame["max_drawdown"],
            "turnover": np.nan,
            "trades": frame["trades"],
            "filter_status": frame["status"].astype(str),
        }
    )
    return rows.reset_index(drop=True)


def shuffled_state_rows(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    cfg = _comparison_cfg(config)
    n_shuffles = int(cfg.get("shuffled_state_samples", 3))
    if n_shuffles <= 0:
        return pd.DataFrame()
    decisions = pd.read_parquet(_path_from_template(str(cfg.get("candidate_decisions", "results/{target_symbol}/candidate_decisions.parquet")), target_symbol))
    if decisions.empty:
        return pd.DataFrame()
    selected_specs_path = results_output_dir(config, target_symbol) / "risk_filter_selected_specs.parquet"
    selected_specs = pd.read_parquet(selected_specs_path)
    specs = selected_specs[selected_specs["filter_id"].isin(decisions["filter_id"])].copy()
    specs["candidate_id"] = specs.apply(_candidate_id_from_row, axis=1)
    specs = specs.loc[
        :,
        ["filter_id", "candidate_id", "feature_set", "n_states", "seed", "fold", "strategy", "filter_name", "horizon_bars", "cost_bps", "threshold"],
    ].drop_duplicates("filter_id")
    combos = specs.loc[:, ["feature_set", "n_states", "seed", "fold"]].drop_duplicates().reset_index(drop=True)
    available = load_candidate_combos(config, target_symbol)
    combos = combos.merge(available, on=["feature_set", "n_states", "seed", "fold"], how="inner")
    if combos.empty:
        return pd.DataFrame()
    merged = build_filter_dataset(config, target_symbol, combos)
    rng = np.random.default_rng(int(cfg.get("shuffled_state_seed", 1729)))
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        for split in [str(cfg.get("candidate_split", "validation")), str(cfg.get("test_split", "test"))]:
            frame = split_combo_frame(merged, spec, split, int(spec["horizon_bars"]))
            if frame.empty:
                continue
            base = base_position(frame, str(spec["strategy"]), float(spec["threshold"]))
            labels = frame["proposed_label"].astype(str).to_numpy()
            for sample in range(n_shuffles):
                shuffled = frame.copy()
                shuffled["proposed_label"] = rng.permutation(labels)
                position = base * filter_multiplier(shuffled, str(spec["filter_name"]))
                metrics = evaluate_position(frame, position, float(spec["cost_bps"]))
                row = {
                    "source_family": "shuffled_state",
                    "source_artifact": "computed_from_hmm_posteriors",
                    "comparison_id": f"{spec['filter_id']}__shuffle{sample}",
                    "filter_id": spec["filter_id"],
                    "candidate_id": spec["candidate_id"],
                    "feature_set": spec["feature_set"],
                    "feature_group": feature_group(str(spec["feature_set"])),
                    "ablation_proxy": ablation_proxy(str(spec["feature_set"])),
                    "n_states": int(spec["n_states"]),
                    "seed": int(spec["seed"]),
                    "fold": int(spec["fold"]),
                    "split": split,
                    "strategy": spec["strategy"],
                    "filter_name": spec["filter_name"],
                    "bucket": "shuffled_state_control",
                    "model_class": "shuffled_state_control",
                    "horizon_bars": int(spec["horizon_bars"]),
                    "cost_bps": float(spec["cost_bps"]),
                    "threshold": float(spec["threshold"]),
                    "selected_hours": "",
                    "uses_hmm": False,
                    "uses_cross_asset": feature_group(str(spec["feature_set"])) == "cross_asset",
                    "uses_time_control": False,
                    "is_shuffled_control": True,
                    "filter_status": "control",
                    "shuffle_sample": sample,
                }
                rows.append({**row, **metrics})
    return pd.DataFrame(rows)


def add_incrementality(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    key_cols = ["filter_id", "split"]
    hmm_mask = rows["source_family"].eq("risk_filter") & rows["bucket"].eq("hmm_filter")
    hmm = rows[hmm_mask].copy()
    if hmm.empty:
        for metric in METRIC_COLS:
            rows[f"{metric}_delta_vs_base"] = np.nan
            rows[f"{metric}_delta_vs_same_hour"] = np.nan
            rows[f"{metric}_delta_vs_shuffled"] = np.nan
        return rows
    delta_cols = []
    for bucket in CONTROL_BUCKETS:
        control = rows[rows["bucket"].eq(bucket)].copy()
        if bucket == "shuffled_state_control":
            control = control.groupby(key_cols, as_index=False)[METRIC_COLS].mean(numeric_only=True)
        else:
            control = control.loc[:, [*key_cols, *METRIC_COLS]].drop_duplicates(key_cols)
        suffix = "shuffled" if bucket == "shuffled_state_control" else ("same_hour" if bucket == "same_hour_control" else bucket)
        control = control.rename(columns={metric: f"{metric}_{suffix}" for metric in METRIC_COLS})
        hmm = hmm.merge(control, on=key_cols, how="left", validate="many_to_one")
        for metric in METRIC_COLS:
            delta = f"{metric}_delta_vs_{suffix}"
            hmm[delta] = hmm[metric] - hmm[f"{metric}_{suffix}"]
            delta_cols.append(delta)
    rows = rows.merge(hmm.loc[:, [*key_cols, *sorted(set(delta_cols))]], on=key_cols, how="left", validate="many_to_one")
    return rows


def baseline_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    test = rows[rows["split"].eq("test")].copy()
    return (
        test.groupby(["source_family", "model_class", "feature_group"], as_index=False)
        .agg(
            rows=("comparison_id", "nunique"),
            median_net_return=("net_return", "median"),
            best_net_return=("net_return", "max"),
            positive_net_rate=("net_return", lambda values: float((values > 0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            median_turnover=("turnover", "median"),
            median_trades=("trades", "median"),
        )
        .sort_values(["median_net_return", "median_daily_sharpe"], ascending=[False, False], kind="stable")
    )


def incrementality_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    hmm = rows[(rows["split"].eq("test")) & (rows["source_family"].eq("risk_filter")) & (rows["bucket"].eq("hmm_filter"))].copy()
    if hmm.empty:
        return pd.DataFrame()
    group_cols = ["feature_group", "feature_set", "strategy", "filter_name"]
    return (
        hmm.groupby(group_cols, as_index=False)
        .agg(
            candidate_rows=("filter_id", "nunique"),
            median_net_return=("net_return", "median"),
            median_net_delta_vs_base=("net_return_delta_vs_base", "median"),
            median_net_delta_vs_same_hour=("net_return_delta_vs_same_hour", "median"),
            median_net_delta_vs_shuffled=("net_return_delta_vs_shuffled", "median"),
            positive_vs_base_rate=("net_return_delta_vs_base", lambda values: float((values > 0).mean())),
            positive_vs_same_hour_rate=("net_return_delta_vs_same_hour", lambda values: float((values > 0).mean())),
            positive_vs_shuffled_rate=("net_return_delta_vs_shuffled", lambda values: float((values > 0).mean())),
            median_sharpe_delta_vs_base=("daily_sharpe_delta_vs_base", "median"),
            median_sharpe_delta_vs_same_hour=("daily_sharpe_delta_vs_same_hour", "median"),
            median_sharpe_delta_vs_shuffled=("daily_sharpe_delta_vs_shuffled", "median"),
        )
        .sort_values(["median_net_delta_vs_shuffled", "median_net_delta_vs_base"], ascending=[False, False], kind="stable")
    )


def ablation_study(rows: pd.DataFrame, config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    hmm = rows[(rows["split"].eq("test")) & (rows["source_family"].eq("risk_filter")) & (rows["bucket"].eq("hmm_filter"))].copy()
    decisions_path = _path_from_template(str(_comparison_cfg(config).get("candidate_decisions", "results/{target_symbol}/candidate_decisions.parquet")), target_symbol)
    decisions = pd.read_parquet(decisions_path) if decisions_path.exists() else pd.DataFrame()
    grouped = (
        hmm.groupby(["feature_set", "feature_group", "ablation_proxy"], as_index=False)
        .agg(
            candidate_rows=("filter_id", "nunique"),
            median_net_return=("net_return", "median"),
            best_net_return=("net_return", "max"),
            positive_net_rate=("net_return", lambda values: float((values > 0).mean())),
            median_daily_sharpe=("daily_sharpe", "median"),
            median_profit_factor=("profit_factor", "median"),
            median_avg_trade_net=("avg_trade_net", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            median_turnover=("turnover", "median"),
        )
        if not hmm.empty
        else pd.DataFrame()
    )
    if not decisions.empty:
        decision_counts = decisions.pivot_table(index="feature_set", columns="decision_label", values="filter_id", aggfunc="count", fill_value=0).reset_index()
        grouped = grouped.merge(decision_counts, on="feature_set", how="left")
    dep_path = results_output_dir(config, target_symbol) / "state_ticker_dependency.parquet"
    if dep_path.exists() and not grouped.empty:
        dep = pd.read_parquet(dep_path)
        dep_summary = dep.groupby("feature_set", as_index=False).agg(
            median_top_ticker_abs_z_share=("top_ticker_abs_z_share", "median"),
            max_top_ticker_abs_z_share=("top_ticker_abs_z_share", "max"),
            top_ticker_mode=("top_ticker", lambda values: values.mode().iloc[0] if not values.mode().empty else ""),
        )
        grouped = grouped.merge(dep_summary, on="feature_set", how="left")
    loo_path = results_output_dir(config, target_symbol) / "state_leave_one_ticker_out.parquet"
    if loo_path.exists() and not grouped.empty:
        loo = pd.read_parquet(loo_path)
        loo = loo.replace([np.inf, -np.inf], np.nan)
        loo_summary = loo.groupby("feature_set", as_index=False).agg(
            median_profile_cosine_after_ticker_removal=("profile_cosine_after_removal", "median"),
            min_profile_cosine_after_ticker_removal=("profile_cosine_after_removal", "min"),
        )
        grouped = grouped.merge(loo_summary, on="feature_set", how="left")
    target = grouped[grouped["feature_set"].eq("target_only_frozen")]
    target_median = float(target["median_net_return"].iloc[0]) if not target.empty else np.nan
    grouped["median_net_delta_vs_target_only"] = grouped["median_net_return"] - target_median
    return grouped.sort_values(["median_net_return", "positive_net_rate"], ascending=[False, False], kind="stable")


def incremental_decision(summary: pd.DataFrame, ablation: pd.DataFrame) -> str:
    if summary.empty:
        return "insufficient_data"
    cross = summary[summary["feature_group"].eq("cross_asset")]
    if cross.empty:
        return "insufficient_data"
    cross_best = float(cross["median_net_delta_vs_shuffled"].max())
    cross_vs_base = float(cross["median_net_delta_vs_base"].max())
    target_median = float(ablation.loc[ablation["feature_set"].eq("target_only_frozen"), "median_net_return"].iloc[0]) if not ablation.empty and ablation["feature_set"].eq("target_only_frozen").any() else np.nan
    cross_median = float(ablation.loc[ablation["feature_group"].eq("cross_asset"), "median_net_return"].max()) if not ablation.empty and ablation["feature_group"].eq("cross_asset").any() else np.nan
    if cross_best > 0 and cross_vs_base > 0 and np.isfinite(target_median) and cross_median > target_median:
        return "cross_asset_incremental_candidate"
    if cross_best > 0 or cross_vs_base > 0:
        return "partial_incremental_but_not_superior_to_target_only"
    return "no_incremental_edge"


def render_report(
    config: dict[str, Any],
    target_symbol: str,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    incremental: pd.DataFrame,
    ablation: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _comparison_cfg(config)
    decision = incremental_decision(incremental, ablation)
    test_rows = rows[rows["split"].eq("test")] if not rows.empty else pd.DataFrame()
    source_counts = test_rows.groupby(["source_family", "model_class"], as_index=False).size().rename(columns={"size": "rows"}) if not test_rows.empty else pd.DataFrame()
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    conclusion = {
        "cross_asset_incremental_candidate": "Cross-asset HMM shows incremental evidence versus controls; still combine with cost gates before promotion.",
        "partial_incremental_but_not_superior_to_target_only": "Cross-asset HMM has partial incremental evidence, but it does not clearly beat target-only/frozen alternatives.",
        "no_incremental_edge": "Cross-asset HMM does not show robust incremental edge versus no-HMM, same-hour and shuffled controls.",
        "insufficient_data": "Insufficient comparable rows to decide incrementality.",
    }.get(decision, decision)
    return f"""# Cross-Asset HMM vs Baselines - {target_symbol.upper()}

## Scope

- Risk-filter candidates: validation-selected only.
- Test is diagnostic only; no threshold or candidate is selected on test.
- Same walk-forward artifacts: `{cfg.get("risk_filter_test", "results/{target_symbol}/risk_filter_test.parquet")}` and state-rule outputs.
- Shuffled-state control samples: `{cfg.get("shuffled_state_samples", 3)}` with fixed seed `{cfg.get("shuffled_state_seed", 1729)}`.
- Ablations use existing trained feature-set proxies plus ticker-dependency diagnostics; they are not new leave-one-group retrains.
- Incrementality decision: `{decision}`.

## Source Counts

{_markdown_table(source_counts, max_rows=60)}

## Baseline Summary

{_markdown_table(summary, max_rows=80)}

## Incrementality Summary

{_markdown_table(incremental, max_rows=80)}

## Ablation Study

{_markdown_table(ablation, max_rows=80)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    results_dir = results_output_dir(config, target)
    risk_rows = risk_filter_rows(config, target)
    rule_rows = state_rule_rows(config, target)
    spy_rows = spy_only_frozen_rows(config)
    shuffled_rows = shuffled_state_rows(config, target)
    rows = pd.concat([risk_rows, rule_rows, spy_rows, shuffled_rows], ignore_index=True, sort=False)
    rows = add_incrementality(rows)
    summary = baseline_summary(rows)
    incremental = incrementality_summary(rows)
    ablation = ablation_study(rows, config, target)
    outputs = {
        "baseline_comparison": results_dir / "baseline_comparison.parquet",
        "ablation_study": results_dir / "ablation_study.parquet",
        "incrementality_summary": results_dir / "incrementality_summary.parquet",
        "shuffled_state_control": results_dir / "shuffled_state_control.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(outputs["baseline_comparison"], index=False)
    ablation.to_parquet(outputs["ablation_study"], index=False)
    incremental.to_parquet(outputs["incrementality_summary"], index=False)
    shuffled_rows.to_parquet(outputs["shuffled_state_control"], index=False)
    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(config, target, rows, summary, incremental, ablation, outputs), encoding="utf-8")
    return report_path, outputs["baseline_comparison"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare cross-asset HMM candidates against SPY-only and no-HMM baselines.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, comparison_path = run(args.config, args.target)
    print(f"Cross-asset baseline comparison report written to: {report_path}")
    print(f"Baseline comparison written to: {comparison_path}")


if __name__ == "__main__":
    main()

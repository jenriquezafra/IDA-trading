from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.candidate_cost_sensitivity_cross_asset import max_drawdown, scenario_cost_return
from src.hmm_lab import _lab_cfg, _target_symbol, build_lab_folds, features_input_path, load_yaml, results_output_dir
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.operable_candidate_search import available_cost_scenarios


DEFAULT_SETUP_COLUMNS = [
    "target_overnight_ret",
    "target_abs_overnight_ret",
    "target_gap_fill_progress",
    "target_dist_open",
    "target_dist_vwap_atr",
    "target_range_ratio_6_24",
    "target_rv_12",
    "target_rv_12_rel_by_bar",
    "target_rv_ratio_6_24",
    "target_rel_volume_by_bar",
    "target_rel_cum_volume_by_bar",
    "target_close_location_bar",
    "target_bar_efficiency",
    "target_upper_wick_ratio",
    "target_lower_wick_ratio",
    "target_consecutive_up_bars",
    "target_consecutive_down_bars",
    "target_above_or_6_high",
    "target_below_or_6_low",
    "target_failed_breakout_high_12",
    "target_failed_breakout_low_12",
    "target_breaks_roll_high_12",
    "target_breaks_roll_low_12",
    "target_first_60m",
    "target_lunch",
    "target_last_60m",
    "target_minutes_to_close",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_6",
    "positive_sector_count_6",
]

DEFAULT_TERCILE_COLUMNS = {
    "gap": "target_overnight_ret",
    "abs_gap": "target_abs_overnight_ret",
    "rv12_rel": "target_rv_12_rel_by_bar",
    "dist_vwap": "target_dist_vwap_atr",
    "rel_volume": "target_rel_volume_by_bar",
    "range_ratio": "target_range_ratio_6_24",
}

DEFAULT_BOOLEAN_SEGMENTS = [
    "target_above_or_6_high",
    "target_below_or_6_low",
    "target_failed_breakout_high_12",
    "target_failed_breakout_low_12",
    "target_breaks_roll_high_12",
    "target_breaks_roll_low_12",
    "target_first_60m",
    "target_lunch",
    "target_last_60m",
]


def _feasibility_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("spy_setup_feasibility", {})


def _combined_cfg(config: dict[str, Any]) -> dict[str, Any]:
    combined = dict(config.get("operable_candidate_search", {}))
    combined.update(_feasibility_cfg(config))
    return combined


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "spy_setup_feasibility.md"


def _feature_config(config: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(Path(_lab_cfg(config).get("features_config", "configs/features/cross_asset_v1.yaml")))


def load_feature_frame(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    feature_config = _feature_config(config)
    return pd.read_parquet(features_input_path(config, target_symbol, feature_config)).sort_values(["session", "bar_index"], kind="stable")


def _configured_horizons(config: dict[str, Any]) -> list[int]:
    return [int(value) for value in _feasibility_cfg(config).get("horizons", [12, 24])]


def _configured_setup_columns(config: dict[str, Any]) -> list[str]:
    configured = _feasibility_cfg(config).get("setup_columns")
    return [str(value) for value in configured] if configured else DEFAULT_SETUP_COLUMNS


def build_feasibility_dataset(config: dict[str, Any], target_symbol: str) -> pd.DataFrame:
    features = load_feature_frame(config, target_symbol)
    horizons = _configured_horizons(config)
    forward = build_forward_returns(features, horizons)
    if forward.empty:
        return pd.DataFrame()
    indexed = features.reset_index(names="source_index")
    extra_columns = [
        "source_index",
        *[column for column in _configured_setup_columns(config) if column in indexed.columns and column not in forward.columns],
    ]
    merged = forward.merge(indexed.loc[:, extra_columns], on="source_index", how="left", validate="many_to_one")
    merged["target_open_next"] = merged["entry_px"].astype(float)
    folds = build_lab_folds(features, config)
    parts = []
    for fold in folds:
        split_sessions = {
            "validation": set(fold.validation_sessions),
            "test": set(fold.test_sessions),
        }
        for split, sessions in split_sessions.items():
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
    return output


def _profit_factor(active_net: pd.Series) -> float:
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
    return float(np.sqrt(252) * daily.mean() / std)


def _top_session_abs_share(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    denom = daily.abs().sum()
    if denom == 0.0 or np.isnan(denom):
        return np.nan
    return float(daily.abs().max() / denom)


def evaluate_direction(frame: pd.DataFrame, direction: str, scenario: dict[str, Any]) -> dict[str, float | int | str]:
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported direction: {direction}")
    sign = 1.0 if direction == "long" else -1.0
    position = pd.Series(sign, index=frame.index, dtype=float)
    gross = position * frame["fwd_ret"].astype(float)
    cost = scenario_cost_return(frame, position, scenario)
    net = gross - cost
    return {
        "rows": int(len(frame)),
        "trades": int(len(frame)),
        "gross_return": float(gross.sum()),
        "cost_return": float(cost.sum()),
        "net_return": float(net.sum()),
        "mean_fwd_ret": float(frame["fwd_ret"].mean()) if len(frame) else np.nan,
        "avg_trade_gross": float(gross.mean()) if len(gross) else np.nan,
        "avg_trade_net": float(net.mean()) if len(net) else np.nan,
        "median_trade_net": float(net.median()) if len(net) else np.nan,
        "hit_rate": float((net > 0.0).mean()) if len(net) else np.nan,
        "profit_factor": _profit_factor(net),
        "daily_sharpe": _daily_sharpe(frame, net),
        "max_drawdown": max_drawdown(net),
        "top_session_abs_net_share": _top_session_abs_share(frame, net),
    }


def _day_part(frame: pd.DataFrame) -> pd.Series:
    first = frame.get("target_first_60m", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    lunch = frame.get("target_lunch", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    last = frame.get("target_last_60m", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    labels = np.select([first, lunch, last], ["first_60m", "lunch", "last_60m"], default="core")
    return pd.Series(labels, index=frame.index, dtype="object")


def _tercile_thresholds(reference: pd.Series) -> tuple[float, float] | None:
    clean = reference.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(clean) < 10 or clean.nunique() < 3:
        return None
    low = float(clean.quantile(1.0 / 3.0))
    high = float(clean.quantile(2.0 / 3.0))
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return None
    return low, high


def iter_segments(frame: pd.DataFrame, reference: pd.DataFrame, config: dict[str, Any]) -> Iterable[tuple[str, str, pd.Series]]:
    yield "all", "all", pd.Series(True, index=frame.index)

    for hour in sorted(frame["hour"].dropna().astype(int).unique().tolist()):
        yield "hour", str(hour), frame["hour"].astype(int).eq(hour)

    parts = _day_part(frame)
    for part in ["first_60m", "core", "lunch", "last_60m"]:
        yield "day_part", part, parts.eq(part)

    terciles = _feasibility_cfg(config).get("tercile_columns", DEFAULT_TERCILE_COLUMNS)
    for label, column in terciles.items():
        column = str(column)
        if column not in frame.columns or column not in reference.columns:
            continue
        thresholds = _tercile_thresholds(reference[column])
        if thresholds is None:
            continue
        low, high = thresholds
        values = frame[column].replace([np.inf, -np.inf], np.nan).astype(float)
        yield f"{label}_tercile", "low", values.le(low)
        yield f"{label}_tercile", "mid", values.gt(low) & values.lt(high)
        yield f"{label}_tercile", "high", values.ge(high)

    for column in [str(value) for value in _feasibility_cfg(config).get("boolean_segments", DEFAULT_BOOLEAN_SEGMENTS)]:
        if column in frame.columns:
            yield "setup_flag", column, frame[column].fillna(False).astype(bool)


def feasibility_grid(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    cfg = _combined_cfg(config)
    scenarios = available_cost_scenarios({**config, "operable_candidate_search": cfg}, [str(value) for value in cfg.get("cost_scenarios", ["ibkr_tiered_10000", "bps_2", "bps_5"])])
    min_rows = int(_feasibility_cfg(config).get("min_segment_rows", 30))
    rows: list[dict[str, Any]] = []
    for (fold, horizon), fold_horizon in dataset.groupby(["fold", "horizon_bars"], sort=False):
        reference = fold_horizon[fold_horizon["split"].eq("validation")].copy()
        if reference.empty:
            continue
        for split, frame in fold_horizon.groupby("split", sort=False):
            for segment_name, segment_value, mask in iter_segments(frame, reference, config):
                segment = frame.loc[mask].copy()
                if len(segment) < min_rows:
                    continue
                for scenario in scenarios:
                    for direction in ["long", "short"]:
                        metric = evaluate_direction(segment, direction, scenario)
                        rows.append(
                            {
                                "fold": int(fold),
                                "split": str(split),
                                "horizon_bars": int(horizon),
                                "segment_name": segment_name,
                                "segment_value": segment_value,
                                "direction": direction,
                                "cost_scenario": scenario["cost_scenario"],
                                "cost_kind": scenario["cost_kind"],
                                **metric,
                            }
                        )
    return pd.DataFrame(rows)


def stability_table(grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    primary = str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))
    base = grid[grid["cost_scenario"].eq(primary)].copy()
    keys = ["fold", "horizon_bars", "segment_name", "segment_value", "direction"]
    validation = base[base["split"].eq("validation")].loc[
        :, [*keys, "rows", "net_return", "avg_trade_net", "daily_sharpe", "profit_factor", "top_session_abs_net_share"]
    ].rename(
        columns={
            "rows": "validation_rows",
            "net_return": "validation_net_return",
            "avg_trade_net": "validation_avg_trade_net",
            "daily_sharpe": "validation_daily_sharpe",
            "profit_factor": "validation_profit_factor",
            "top_session_abs_net_share": "validation_top_session_abs_net_share",
        }
    )
    test = base[base["split"].eq("test")].loc[
        :, [*keys, "rows", "net_return", "avg_trade_net", "daily_sharpe", "profit_factor", "top_session_abs_net_share"]
    ].rename(
        columns={
            "rows": "test_rows",
            "net_return": "test_net_return",
            "avg_trade_net": "test_avg_trade_net",
            "daily_sharpe": "test_daily_sharpe",
            "profit_factor": "test_profit_factor",
            "top_session_abs_net_share": "test_top_session_abs_net_share",
        }
    )
    merged = validation.merge(test, on=keys, how="inner", validate="one_to_one")
    merged["avg_trade_decay"] = merged["test_avg_trade_net"] - merged["validation_avg_trade_net"]
    merged["stable_positive"] = (
        merged["validation_avg_trade_net"].gt(0.0)
        & merged["test_avg_trade_net"].gt(0.0)
        & merged["validation_net_return"].gt(0.0)
        & merged["test_net_return"].gt(0.0)
    )
    return merged.sort_values(["stable_positive", "test_avg_trade_net", "test_net_return"], ascending=[False, False, False], kind="stable")


def findings_table(stability: pd.DataFrame, grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stability.empty:
        return pd.DataFrame(rows)
    primary = str(_combined_cfg(config).get("primary_cost_scenario", "ibkr_tiered_10000"))
    stable = stability[stability["stable_positive"]].copy()
    rows.append(
        {
            "finding": "stable_positive_segments_primary_cost",
            "evidence": f"{len(stable)} segments with positive validation and test avg trade under {primary}",
            "implication": "Use these as feasibility zones, not final strategies.",
        }
    )
    for direction in ["long", "short"]:
        direction_rows = stable[stable["direction"].eq(direction)].sort_values("test_avg_trade_net", ascending=False, kind="stable")
        if direction_rows.empty:
            rows.append(
                {
                    "finding": f"no_stable_{direction}_segment",
                    "evidence": f"No {direction} segment is positive in both validation and test under {primary}",
                    "implication": f"Do not search {direction} signals until setup features create a clearer conditional edge.",
                }
            )
            continue
        top = direction_rows.iloc[0]
        rows.append(
            {
                "finding": f"top_stable_{direction}_segment",
                "evidence": (
                    f"fold={int(top['fold'])}, h={int(top['horizon_bars'])}, "
                    f"{top['segment_name']}={top['segment_value']}, test avg={top['test_avg_trade_net']:.6f}, "
                    f"test net={top['test_net_return']:.6f}"
                ),
                "implication": "Promote this zone to the next interpretable signal-family search.",
            }
        )

    stress = grid[grid["cost_scenario"].eq(str(_combined_cfg(config).get("stress_cost_scenario", "bps_5")))]
    if not stress.empty:
        positives = stress[stress["split"].eq("test") & stress["avg_trade_net"].gt(0.0)]
        rows.append(
            {
                "finding": "stress_cost_positive_test_segments",
                "evidence": f"{positives[['fold', 'horizon_bars', 'segment_name', 'segment_value', 'direction']].drop_duplicates().shape[0]} unique test segments positive under stress cost",
                "implication": "Segments that disappear here are cost-fragile and should not drive selection.",
            }
        )
    return pd.DataFrame(rows)


def render_report(
    target_symbol: str,
    grid: pd.DataFrame,
    stability: pd.DataFrame,
    findings: pd.DataFrame,
    outputs: dict[str, Path],
    config: dict[str, Any],
) -> str:
    cfg = _combined_cfg(config)
    primary = str(cfg.get("primary_cost_scenario", "ibkr_tiered_10000"))
    primary_grid = grid[grid["cost_scenario"].eq(primary)].copy() if not grid.empty else pd.DataFrame()
    top_test = (
        primary_grid[primary_grid["split"].eq("test")]
        .sort_values(["avg_trade_net", "net_return"], ascending=[False, False], kind="stable")
        .head(int(_feasibility_cfg(config).get("report_top_rows", 80)))
        if not primary_grid.empty
        else pd.DataFrame()
    )
    stable_top = stability.head(int(_feasibility_cfg(config).get("report_top_rows", 80))) if not stability.empty else pd.DataFrame()
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    return f"""# SPY Setup Feasibility - {target_symbol.upper()}

## Scope

- Uses setup-oriented SPY intraday features.
- Evaluates unconditional long and short expectancy by segment.
- Direction is not selected by HMM.
- Primary cost scenario: `{primary}`.
- Horizons: `{_configured_horizons(config)}`.

## Findings

{_markdown_table(findings)}

## Top Test Segments

{_markdown_table(top_test)}

## Validation/Test Stability

{_markdown_table(stable_top)}

## Outputs

{outputs_text}

## Conclusion

This is a feasibility map, not a strategy. Promote only stable positive segments into the next short/long-separated signal search, and keep HMM as a later risk overlay.
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    dataset = build_feasibility_dataset(config, target)
    grid = feasibility_grid(dataset, config)
    stability = stability_table(grid, config)
    findings = findings_table(stability, grid, config)

    results_dir = results_output_dir(config, target)
    outputs = {
        "spy_setup_feasibility": results_dir / "spy_setup_feasibility.parquet",
        "spy_setup_feasibility_stability": results_dir / "spy_setup_feasibility_stability.parquet",
        "spy_setup_feasibility_findings": results_dir / "spy_setup_feasibility_findings.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(outputs["spy_setup_feasibility"], index=False)
    stability.to_parquet(outputs["spy_setup_feasibility_stability"], index=False)
    findings.to_parquet(outputs["spy_setup_feasibility_findings"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(target, grid, stability, findings, outputs, config), encoding="utf-8")
    return report_path, outputs["spy_setup_feasibility_findings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map SPY intraday setup feasibility before signal search.")
    parser.add_argument("--config", default="configs/hmm_lab.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, findings_path = run(args.config, args.target)
    print(f"SPY setup feasibility report written to: {report_path}")
    print(f"SPY setup feasibility findings written to: {findings_path}")


if __name__ == "__main__":
    main()

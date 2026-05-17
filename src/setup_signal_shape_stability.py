from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_lab import _target_symbol, load_yaml, results_output_dir
from src.hmm_state_interpretability_cross_asset import _markdown_table
from src.setup_signal_anti_concentration import (
    configured_focus as anti_configured_focus,
    focused_search_config,
    primary_cost_name,
    rank_validation_candidates,
    stress_cost_name,
)
from src.setup_signal_diagnostics import reconstruct_bar_returns
from src.setup_signal_search import evaluate_selected_on_split, validation_grid
from src.setup_signal_search import build_signal_dataset
from src.operable_candidate_search import available_cost_scenarios


SPEC_COLUMNS = [
    "candidate_id",
    "fold",
    "family",
    "direction",
    "horizon_bars",
    "params_json",
    "column_map_json",
]

BASE_RULE_SHAPE_KEYS = (
    "close_location_min",
    "rel_volume_q",
    "vwap_min",
)

OPTIONAL_RULE_SHAPE_KEYS = (
    "filter_set",
    "bias_attempts_min",
    "risk_off_q",
    "range_ratio_q",
    "wick_q",
    "breakout_margin_q",
    "rel_cum_volume_q",
    "rel_volume_accel_q",
    "rv_rel_q",
    "positive_index_min",
    "positive_sector_min",
    "positive_index_open_min",
    "positive_sector_open_min",
    "index_above_vwap_min",
    "sector_above_vwap_min",
    "relopen_qqq_spy_min",
    "relopen_iwm_spy_min",
    "risk_on_open_min",
    "risk_on_min",
    "risk_off_max",
    "dist_open_min",
    "vwap_floor",
    "vwap_ceiling",
    "opening_persistence_min",
    "opening_attempts_max",
    "max_upper_wick",
    "max_lower_wick",
    "min_minutes_from_open",
    "min_minutes_to_close",
)


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("setup_signal_shape_stability", {})


def configured_focus(config: dict[str, Any]) -> dict[str, Any]:
    focus = anti_configured_focus(config)
    cfg = _cfg(config)
    return {
        "family": str(cfg.get("family", focus["family"])),
        "direction": str(cfg.get("direction", focus["direction"])),
        "horizon_bars": int(cfg.get("horizon_bars", focus["horizon_bars"])),
    }


def report_output_path(config: dict[str, Any], target_symbol: str) -> Path:
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "setup_signal_shape_stability.md"


def parse_rule_shape(params_json: str) -> dict[str, Any]:
    params = json.loads(str(params_json))
    shape = {key: params.get(key) for key in BASE_RULE_SHAPE_KEYS if key in params}
    shape.update({key: params[key] for key in OPTIONAL_RULE_SHAPE_KEYS if key in params})
    return shape


def rule_shape_key(params_json: str) -> str:
    shape = parse_rule_shape(params_json)
    return "|".join(f"{key}={shape[key]}" for key in sorted(shape))


def add_rule_shape(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    parsed = output["params_json"].map(parse_rule_shape)
    output["rule_shape"] = output["params_json"].map(rule_shape_key)
    for key in sorted({key for item in parsed for key in item}):
        output[key] = parsed.map(lambda item, column=key: item.get(column))
    return output


def validation_candidates(config: dict[str, Any], target_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focused = focused_search_config(config)
    dataset = build_signal_dataset(focused, target_symbol)
    if dataset.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    validation = validation_grid(dataset, focused)
    primary = primary_cost_name(focused)
    primary_rows = validation[validation["bucket"].eq("setup_signal") & validation["cost_scenario"].eq(primary)]
    scenarios = available_cost_scenarios({**focused, "operable_candidate_search": focused["setup_signal_search"]}, [primary])
    bar_returns = (
        reconstruct_bar_returns(dataset, primary_rows, scenarios[0], splits=("validation",))
        if not primary_rows.empty and scenarios
        else pd.DataFrame()
    )
    ranked = rank_validation_candidates(validation, bar_returns, config)
    focus = configured_focus(config)
    ranked = ranked[
        ranked["family"].eq(focus["family"])
        & ranked["direction"].eq(focus["direction"])
        & ranked["horizon_bars"].astype(int).eq(focus["horizon_bars"])
    ].copy()
    return dataset, validation, add_rule_shape(ranked)


def evaluate_all_test(dataset: pd.DataFrame, ranked_validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if dataset.empty or ranked_validation.empty:
        return pd.DataFrame()
    focused = focused_search_config(config)
    specs = ranked_validation.loc[:, [column for column in SPEC_COLUMNS if column in ranked_validation.columns]].drop_duplicates("candidate_id")
    test = evaluate_selected_on_split(dataset, specs, "test", focused)
    return add_rule_shape(test)


def _first_value(frame: pd.DataFrame, column: str, default: Any = np.nan) -> Any:
    return frame[column].iloc[0] if column in frame and not frame.empty else default


def shape_stability_table(ranked_validation: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if ranked_validation.empty:
        return pd.DataFrame()
    cfg = config.get("setup_signal_anti_concentration", {})
    primary = primary_cost_name(config)
    stress = stress_cost_name(config)
    setup_test = test[test["bucket"].eq("setup_signal")].copy() if not test.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for shape, validation_group in ranked_validation.groupby("rule_shape", sort=False):
        folds_present = int(validation_group["fold"].nunique())
        primary_test = setup_test[setup_test["rule_shape"].eq(shape) & setup_test["cost_scenario"].eq(primary)]
        stress_test = setup_test[setup_test["rule_shape"].eq(shape) & setup_test["cost_scenario"].eq(stress)]
        validation_positive = validation_group["net_return"].gt(0.0) & validation_group["avg_trade_net"].gt(0.0)
        primary_positive = primary_test["net_return"].gt(0.0) & primary_test["avg_trade_net"].gt(0.0) if not primary_test.empty else pd.Series(dtype=bool)
        stress_nonnegative = stress_test["net_return"].ge(0.0) & stress_test["avg_trade_net"].ge(0.0) if not stress_test.empty else pd.Series(dtype=bool)
        statuses = (
            validation_group.sort_values("fold", kind="stable")
            .assign(status_text=lambda frame: frame["fold"].astype(str) + ":" + frame["anti_status"].astype(str))
            ["status_text"]
            .tolist()
        )
        rows.append(
            {
                "rule_shape": shape,
                "folds_present": folds_present,
                "validation_positive_folds": int(validation_positive.sum()),
                "validation_anti_candidate_folds": int(validation_group["anti_status"].eq("anti_concentration_candidate").sum()),
                "validation_min_trades": int(validation_group["trades"].min()),
                "validation_min_net": float(validation_group["net_return"].min()),
                "validation_min_avg_trade": float(validation_group["avg_trade_net"].min()),
                "validation_min_leave_one_month_net": float(validation_group["leave_one_month_min_net"].min(skipna=True))
                if validation_group["leave_one_month_min_net"].notna().any()
                else np.nan,
                "validation_max_top_month_abs_share": float(validation_group["top_month_abs_net_share_rebuilt"].max(skipna=True))
                if validation_group["top_month_abs_net_share_rebuilt"].notna().any()
                else np.nan,
                "validation_statuses": ", ".join(statuses),
                "test_primary_positive_folds": int(primary_positive.sum()) if not primary_test.empty else 0,
                "test_stress_nonnegative_folds": int(stress_nonnegative.sum()) if not stress_test.empty else 0,
                "test_min_trades_primary": int(primary_test["trades"].min()) if not primary_test.empty else 0,
                "test_min_net_primary": float(primary_test["net_return"].min()) if not primary_test.empty else np.nan,
                "test_min_avg_trade_primary": float(primary_test["avg_trade_net"].min()) if not primary_test.empty else np.nan,
                "test_min_net_stress": float(stress_test["net_return"].min()) if not stress_test.empty else np.nan,
                "stable_validation_shape": bool(
                    folds_present >= 2
                    and validation_group["anti_status"].eq("anti_concentration_candidate").all()
                    and int(validation_group["trades"].min()) >= int(cfg.get("min_trades", 40))
                ),
                "stable_test_shape": bool(
                    folds_present >= 2
                    and primary_positive.sum() == folds_present
                    and stress_nonnegative.sum() == folds_present
                    and (not primary_test.empty and int(primary_test["trades"].min()) >= int(cfg.get("min_trades", 40)))
                ),
                "close_location_min": _first_value(validation_group, "close_location_min"),
                "rel_volume_q": _first_value(validation_group, "rel_volume_q"),
                "vwap_min": _first_value(validation_group, "vwap_min"),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(
        [
            "stable_validation_shape",
            "stable_test_shape",
            "validation_anti_candidate_folds",
            "validation_positive_folds",
            "test_primary_positive_folds",
            "test_stress_nonnegative_folds",
            "test_min_net_primary",
        ],
        ascending=[False, False, False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def render_report(
    target_symbol: str,
    ranked_validation: pd.DataFrame,
    test: pd.DataFrame,
    stability: pd.DataFrame,
    outputs: dict[str, Path],
    config: dict[str, Any],
) -> str:
    cfg = _cfg(config)
    focus = configured_focus(config)
    max_rows = int(cfg.get("report_top_rows", 80))
    validation_counts = (
        ranked_validation["anti_status"].value_counts().rename_axis("anti_status").reset_index(name="rows")
        if not ranked_validation.empty
        else pd.DataFrame()
    )
    stable_validation = int(stability["stable_validation_shape"].sum()) if not stability.empty else 0
    stable_test = int(stability["stable_test_shape"].sum()) if not stability.empty else 0
    near_misses = stability.head(max_rows) if not stability.empty else stability
    top_validation = (
        ranked_validation.sort_values(["fold", "anti_status", "anti_score"], ascending=[True, True, False], kind="stable").head(max_rows)
        if not ranked_validation.empty
        else ranked_validation
    )
    setup_test = (
        test[test["bucket"].eq("setup_signal")]
        .sort_values(["rule_shape", "fold", "cost_scenario"], kind="stable")
        .head(max_rows)
        if not test.empty
        else test
    )
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    if stable_validation:
        conclusion = (
            f"{stable_validation} rule shape(s) pass validation in every fold. Review test stability before allowing HMM overlay."
        )
    else:
        conclusion = (
            "No rule shape passes anti-concentration validation in both folds. The current setup family cannot be repaired "
            "inside this parameterization without relaxing gates or adding a genuinely new feature/filter hypothesis."
        )
    if stable_test and not stable_validation:
        conclusion += (
            " Some shapes look positive in forced test diagnostics, but they were not validation-stable and must not be promoted."
        )

    summary = pd.DataFrame(
        [
            {
                "metric": "stable_validation_shapes",
                "value": stable_validation,
            },
            {
                "metric": "stable_test_shapes_forced_diagnostic",
                "value": stable_test,
            },
        ]
    )
    return f"""# Setup Signal Shape Stability - {target_symbol.upper()}

## Scope

- Family: `{focus["family"]}`.
- Direction: `{focus["direction"]}`.
- Horizon bars: `{focus["horizon_bars"]}`.
- Purpose: check whether the same rule shape is validation-stable across folds before any HMM overlay.

## Summary

{_markdown_table(summary)}

## Validation Status Counts

{_markdown_table(validation_counts)}

## Rule Shape Stability

{_markdown_table(near_misses, max_rows=max_rows)}

## Top Validation Rows

{_markdown_table(top_validation, max_rows=max_rows)}

## Forced Test Diagnostics

{_markdown_table(setup_test, max_rows=max_rows)}

## Outputs

{outputs_text}

## Conclusion

{conclusion}
"""


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    target = _target_symbol(config, target_symbol)
    dataset, _, ranked_validation = validation_candidates(config, target)
    test = evaluate_all_test(dataset, ranked_validation, config)
    stability = shape_stability_table(ranked_validation, test, config)

    results_dir = results_output_dir(config, target)
    outputs = {
        "setup_signal_shape_validation": results_dir / "setup_signal_shape_validation.parquet",
        "setup_signal_shape_test": results_dir / "setup_signal_shape_test.parquet",
        "setup_signal_shape_stability": results_dir / "setup_signal_shape_stability.parquet",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    ranked_validation.to_parquet(outputs["setup_signal_shape_validation"], index=False)
    test.to_parquet(outputs["setup_signal_shape_test"], index=False)
    stability.to_parquet(outputs["setup_signal_shape_stability"], index=False)

    report_path = report_output_path(config, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(target, ranked_validation, test, stability, outputs, config), encoding="utf-8")
    return report_path, outputs["setup_signal_shape_stability"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check same-rule-shape stability for a focused setup family.")
    parser.add_argument("--config", default="configs/hmm_lab_30min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report_path, stability_path = run(args.config, args.target)
    print(f"Setup signal shape stability report written to: {report_path}")
    print(f"Setup signal shape stability table written to: {stability_path}")


if __name__ == "__main__":
    main()

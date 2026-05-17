from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from src.cross_asset_data import load_yaml, write_yaml


DEFAULT_HORIZON_SECTIONS = [
    "hmm_state_economics_cross_asset",
    "state_rules_cross_asset",
    "hmm_risk_filter",
    "candidate_cost_sensitivity_cross_asset",
    "operable_candidate_search",
    "operable_alpha_refinement",
    "alpha_discovery_base",
    "spy_setup_feasibility",
    "setup_signal_search",
]

SETUP_15MIN_COLUMN_MAP = {
    "range_ratio": "target_range_ratio_2_8",
    "rv_rel": "target_rv_4_rel_by_bar",
    "opening_high": "target_above_or_2_high",
    "opening_low": "target_below_or_2_low",
    "failed_breakout_high": "target_failed_breakout_high_4",
    "failed_breakout_low": "target_failed_breakout_low_4",
    "breaks_roll_high": "target_breaks_roll_high_4",
    "breaks_roll_low": "target_breaks_roll_low_4",
    "positive_index_count": "positive_index_count_2",
    "positive_sector_count": "positive_sector_count_2",
}

SETUP_15MIN_COLUMNS = [
    "target_overnight_ret",
    "target_abs_overnight_ret",
    "target_gap_fill_progress",
    "target_dist_open",
    "target_dist_vwap_atr",
    "target_range_ratio_2_8",
    "target_rv_4",
    "target_rv_4_rel_by_bar",
    "target_rel_volume_by_bar",
    "target_rel_cum_volume_by_bar",
    "target_close_location_bar",
    "target_bar_efficiency",
    "target_upper_wick_ratio",
    "target_lower_wick_ratio",
    "target_consecutive_up_bars",
    "target_consecutive_down_bars",
    "target_above_or_2_high",
    "target_below_or_2_low",
    "target_failed_breakout_high_4",
    "target_failed_breakout_low_4",
    "target_breaks_roll_high_4",
    "target_breaks_roll_low_4",
    "target_first_60m",
    "target_lunch",
    "target_last_60m",
    "target_minutes_to_close",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_2",
    "positive_sector_count_2",
]

SETUP_30MIN_COLUMN_MAP = {
    "range_ratio": "target_range_ratio_1_4",
    "rv_rel": "target_rv_2_rel_by_bar",
    "opening_high": "target_above_or_1_high",
    "opening_low": "target_below_or_1_low",
    "failed_breakout_high": "target_failed_breakout_high_2",
    "failed_breakout_low": "target_failed_breakout_low_2",
    "breaks_roll_high": "target_breaks_roll_high_2",
    "breaks_roll_low": "target_breaks_roll_low_2",
    "positive_index_count": "positive_index_count_2",
    "positive_sector_count": "positive_sector_count_2",
}

SETUP_30MIN_COLUMNS = [
    "target_overnight_ret",
    "target_abs_overnight_ret",
    "target_gap_fill_progress",
    "target_dist_open",
    "target_dist_vwap_atr",
    "target_range_ratio_1_4",
    "target_rv_2",
    "target_rv_2_rel_by_bar",
    "target_rel_volume_by_bar",
    "target_rel_cum_volume_by_bar",
    "target_close_location_bar",
    "target_bar_efficiency",
    "target_upper_wick_ratio",
    "target_lower_wick_ratio",
    "target_consecutive_up_bars",
    "target_consecutive_down_bars",
    "target_above_or_1_high",
    "target_below_or_1_low",
    "target_failed_breakout_high_2",
    "target_failed_breakout_low_2",
    "target_breaks_roll_high_2",
    "target_breaks_roll_low_2",
    "target_first_60m",
    "target_lunch",
    "target_last_60m",
    "target_minutes_to_close",
    "risk_on_score",
    "risk_off_score",
    "positive_index_count_2",
    "positive_sector_count_2",
]


def _replace_results_template(value: Any, timeframe: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_results_template(item, timeframe) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_results_template(item, timeframe) for item in value]
    if isinstance(value, str):
        return value.replace("results/{target_symbol}/", f"results/{timeframe}/{{target_symbol}}/")
    return value


def make_timeframe_config(
    base_config: dict[str, Any],
    timeframe: str,
    expected_bars_per_session: int,
    features_config: str,
    feature_set_version: str,
    horizons: list[int],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("project", {})["frequency"] = timeframe
    config.setdefault("lab", {})["timeframe"] = timeframe
    config.setdefault("session", {})["expected_bars_per_session"] = int(expected_bars_per_session)
    config.setdefault("hmm_lab", {})["features_config"] = features_config
    config.setdefault("hmm_lab", {})["feature_set_version"] = feature_set_version

    paths = config.setdefault("paths", {})
    paths["models_dir"] = f"models/{timeframe}"
    paths["results_dir"] = f"results/{timeframe}"
    paths["reports_dir"] = f"reports/{timeframe}"
    paths["data_coverage_dir"] = f"reports/{timeframe}/data_coverage"
    paths["alignment_dir"] = f"reports/{timeframe}/alignment"

    for section in DEFAULT_HORIZON_SECTIONS:
        if section in config and "horizons" in config[section]:
            config[section]["horizons"] = [int(value) for value in horizons]

    setup_columns: list[str] | None = None
    setup_column_map: dict[str, str] | None = None
    setup_diagnostic_horizon = 4
    setup_feature_columns = [
        "target_dist_vwap_atr",
        "target_dist_open",
        "target_range_ratio_2_8",
        "target_rv_4_rel_by_bar",
        "target_rel_volume_by_bar",
        "target_close_location_bar",
        "target_lower_wick_ratio",
        "target_upper_wick_ratio",
        "target_minutes_to_close",
        "risk_on_score",
        "risk_off_score",
        "positive_index_count_2",
        "positive_sector_count_2",
        "intraday_stress_score",
        "chop_score",
    ]
    feasibility_terciles = {
        "gap": "target_overnight_ret",
        "abs_gap": "target_abs_overnight_ret",
        "rv_rel": "target_rv_4_rel_by_bar",
        "dist_vwap": "target_dist_vwap_atr",
        "rel_volume": "target_rel_volume_by_bar",
        "range_ratio": "target_range_ratio_2_8",
    }
    boolean_segments = [
        "target_above_or_2_high",
        "target_below_or_2_low",
        "target_failed_breakout_high_4",
        "target_failed_breakout_low_4",
        "target_breaks_roll_high_4",
        "target_breaks_roll_low_4",
        "target_first_60m",
        "target_lunch",
        "target_last_60m",
    ]
    if timeframe == "15min":
        setup_columns = list(SETUP_15MIN_COLUMNS)
        setup_column_map = dict(SETUP_15MIN_COLUMN_MAP)
    elif timeframe == "30min":
        setup_columns = list(SETUP_30MIN_COLUMNS)
        setup_column_map = dict(SETUP_30MIN_COLUMN_MAP)
        setup_diagnostic_horizon = 2
        setup_feature_columns = [
            "target_dist_vwap_atr",
            "target_dist_open",
            "target_range_ratio_1_4",
            "target_rv_2_rel_by_bar",
            "target_rel_volume_by_bar",
            "target_close_location_bar",
            "target_lower_wick_ratio",
            "target_upper_wick_ratio",
            "target_minutes_to_close",
            "risk_on_score",
            "risk_off_score",
            "positive_index_count_2",
            "positive_sector_count_2",
            "intraday_stress_score",
            "chop_score",
        ]
        feasibility_terciles = {
            "gap": "target_overnight_ret",
            "abs_gap": "target_abs_overnight_ret",
            "rv_rel": "target_rv_2_rel_by_bar",
            "dist_vwap": "target_dist_vwap_atr",
            "rel_volume": "target_rel_volume_by_bar",
            "range_ratio": "target_range_ratio_1_4",
        }
        boolean_segments = [
            "target_above_or_1_high",
            "target_below_or_1_low",
            "target_failed_breakout_high_2",
            "target_failed_breakout_low_2",
            "target_breaks_roll_high_2",
            "target_breaks_roll_low_2",
            "target_first_60m",
            "target_lunch",
            "target_last_60m",
        ]

    if setup_columns is not None and setup_column_map is not None:
        feasibility = config.setdefault("spy_setup_feasibility", {})
        feasibility["setup_columns"] = list(setup_columns)
        feasibility["tercile_columns"] = dict(feasibility_terciles)
        feasibility["boolean_segments"] = list(boolean_segments)

        search = config.setdefault("setup_signal_search", {})
        search["signal_columns"] = list(setup_columns)
        search["column_map"] = dict(setup_column_map)
        config["setup_signal_diagnostics"] = {
            "family": "breakdown_short_risk_off",
            "direction": "short",
            "horizon_bars": setup_diagnostic_horizon,
            "max_specs_per_fold": 12,
            "report_top_rows": 80,
            "feature_columns": setup_feature_columns,
        }
        config["setup_signal_anti_concentration"] = {
            "family": "breakdown_short_risk_off",
            "direction": "short",
            "horizon_bars": setup_diagnostic_horizon,
            "rel_volume_quantiles": [0.67, 0.75, 0.85, 0.90],
            "risk_off_quantiles": [0.60, 0.75, 0.85],
            "vwap_abs_mins": [0.0, 0.25, 0.50, 0.75],
            "min_trades": 40,
            "min_profit_factor": 1.05,
            "min_daily_sharpe": 0.30,
            "max_top_day_abs_net_share": 0.30,
            "min_months": 4,
            "min_positive_months": 3,
            "min_positive_month_rate": 0.50,
            "max_top_month_abs_net_share": 0.45,
            "min_leave_one_month_net": 0.0,
            "max_selected_per_fold": 5,
            "report_top_rows": 80,
        }

    config = _replace_results_template(config, timeframe)
    return config


def run(
    base_path: str | Path,
    output_path: str | Path,
    timeframe: str,
    expected_bars_per_session: int,
    features_config: str,
    feature_set_version: str,
    horizons: list[int],
) -> Path:
    config = make_timeframe_config(
        load_yaml(base_path),
        timeframe=timeframe,
        expected_bars_per_session=expected_bars_per_session,
        features_config=features_config,
        feature_set_version=feature_set_version,
        horizons=horizons,
    )
    output = Path(output_path)
    write_yaml(output, config)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated HMM lab config for a larger timeframe.")
    parser.add_argument("--base", default="configs/hmm_lab.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--expected-bars-per-session", type=int, required=True)
    parser.add_argument("--features-config", required=True)
    parser.add_argument("--feature-set-version", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, required=True)
    args = parser.parse_args()

    output = run(
        args.base,
        args.output,
        timeframe=args.timeframe,
        expected_bars_per_session=args.expected_bars_per_session,
        features_config=args.features_config,
        feature_set_version=args.feature_set_version,
        horizons=args.horizons,
    )
    print(f"Timeframe config written to: {output}")


if __name__ == "__main__":
    main()

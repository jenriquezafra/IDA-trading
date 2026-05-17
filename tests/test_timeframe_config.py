from __future__ import annotations

from src.timeframe_config import make_timeframe_config


def test_make_timeframe_config_isolates_paths_horizons_and_setup_columns() -> None:
    base = {
        "project": {"frequency": "5min"},
        "lab": {"timeframe": "5min"},
        "session": {"expected_bars_per_session": 78},
        "paths": {
            "models_dir": "models",
            "results_dir": "results",
            "reports_dir": "reports",
            "data_coverage_dir": "reports/data_coverage",
            "alignment_dir": "reports/alignment",
        },
        "hmm_lab": {"features_config": "configs/features/cross_asset_v1.yaml", "feature_set_version": "cross_asset_v1"},
        "spy_setup_feasibility": {"horizons": [12, 24]},
        "setup_signal_search": {"horizons": [12, 24]},
        "alpha_fold_degradation_diagnostics": {"selected_specs": "results/{target_symbol}/alpha_discovery_selected_specs.parquet"},
    }

    config = make_timeframe_config(
        base,
        timeframe="15min",
        expected_bars_per_session=26,
        features_config="configs/features/cross_asset_15min.yaml",
        feature_set_version="cross_asset_v1_15min",
        horizons=[4, 8, 12],
    )

    assert config["project"]["frequency"] == "15min"
    assert config["lab"]["timeframe"] == "15min"
    assert config["session"]["expected_bars_per_session"] == 26
    assert config["paths"]["results_dir"] == "results/15min"
    assert config["paths"]["reports_dir"] == "reports/15min"
    assert config["hmm_lab"]["features_config"] == "configs/features/cross_asset_15min.yaml"
    assert config["spy_setup_feasibility"]["horizons"] == [4, 8, 12]
    assert config["setup_signal_search"]["column_map"]["range_ratio"] == "target_range_ratio_2_8"
    assert config["alpha_fold_degradation_diagnostics"]["selected_specs"] == "results/15min/{target_symbol}/alpha_discovery_selected_specs.parquet"


def test_make_timeframe_config_maps_30min_setup_columns() -> None:
    base = {
        "project": {"frequency": "5min"},
        "lab": {"timeframe": "5min"},
        "session": {"expected_bars_per_session": 78},
        "paths": {
            "models_dir": "models",
            "results_dir": "results",
            "reports_dir": "reports",
            "data_coverage_dir": "reports/data_coverage",
            "alignment_dir": "reports/alignment",
        },
        "hmm_lab": {"features_config": "configs/features/cross_asset_v1.yaml", "feature_set_version": "cross_asset_v1"},
        "spy_setup_feasibility": {"horizons": [12, 24]},
        "setup_signal_search": {"horizons": [12, 24]},
    }

    config = make_timeframe_config(
        base,
        timeframe="30min",
        expected_bars_per_session=13,
        features_config="configs/features/cross_asset_30min.yaml",
        feature_set_version="cross_asset_v1_30min",
        horizons=[2, 4, 6],
    )

    assert config["paths"]["results_dir"] == "results/30min"
    assert config["session"]["expected_bars_per_session"] == 13
    assert config["setup_signal_search"]["horizons"] == [2, 4, 6]
    assert config["setup_signal_search"]["column_map"]["range_ratio"] == "target_range_ratio_1_4"
    assert config["setup_signal_search"]["column_map"]["opening_low"] == "target_below_or_1_low"
    assert config["setup_signal_search"]["column_map"]["breaks_roll_low"] == "target_breaks_roll_low_2"
    assert config["spy_setup_feasibility"]["tercile_columns"]["rv_rel"] == "target_rv_2_rel_by_bar"
    assert config["setup_signal_diagnostics"]["horizon_bars"] == 2

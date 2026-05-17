from __future__ import annotations

import pandas as pd
import pytest

from src.cross_asset_vs_baselines import add_incrementality, ablation_proxy, feature_group, incremental_decision, model_class_for_risk_row


def test_feature_group_and_ablation_proxy_are_stable() -> None:
    assert feature_group("target_only_frozen") == "target_only"
    assert feature_group("cross_asset_full_core") == "cross_asset"
    assert feature_group("minimal_vwap_location") == "spy_only_frozen"

    assert ablation_proxy("cross_asset_macro") == "no_sector_leadership_raw"
    assert ablation_proxy("cross_asset_sectors") == "no_bonds_credit_macro"


def test_model_class_for_risk_row_separates_hmm_and_no_hmm_controls() -> None:
    hmm = pd.Series({"bucket": "hmm_filter", "strategy": "momentum_simple", "feature_set": "cross_asset_full_core"})
    base = pd.Series({"bucket": "base", "strategy": "supervised_simple", "feature_set": "cross_asset_full_core"})
    target = pd.Series({"bucket": "hmm_filter", "strategy": "momentum_simple", "feature_set": "target_only_frozen"})

    assert model_class_for_risk_row(hmm) == "cross_asset_hmm_filter"
    assert model_class_for_risk_row(base) == "supervised_no_hmm"
    assert model_class_for_risk_row(target) == "target_only_hmm_filter"


def test_add_incrementality_merges_base_same_hour_and_shuffled_controls() -> None:
    rows = pd.DataFrame(
        [
            {
                "source_family": "risk_filter",
                "filter_id": "f1",
                "split": "test",
                "bucket": "hmm_filter",
                "net_return": 0.03,
                "daily_sharpe": 1.0,
                "profit_factor": 1.2,
                "avg_trade_net": 0.01,
                "max_drawdown": 0.05,
                "turnover": 2.0,
                "trades": 20,
            },
            {
                "source_family": "risk_filter",
                "filter_id": "f1",
                "split": "test",
                "bucket": "base",
                "net_return": 0.01,
                "daily_sharpe": 0.5,
                "profit_factor": 1.0,
                "avg_trade_net": 0.0,
                "max_drawdown": 0.08,
                "turnover": 3.0,
                "trades": 30,
            },
            {
                "source_family": "risk_filter",
                "filter_id": "f1",
                "split": "test",
                "bucket": "same_hour_control",
                "net_return": 0.02,
                "daily_sharpe": 0.7,
                "profit_factor": 1.1,
                "avg_trade_net": 0.005,
                "max_drawdown": 0.06,
                "turnover": 2.5,
                "trades": 25,
            },
            {
                "source_family": "shuffled_state",
                "filter_id": "f1",
                "split": "test",
                "bucket": "shuffled_state_control",
                "net_return": 0.015,
                "daily_sharpe": 0.6,
                "profit_factor": 1.0,
                "avg_trade_net": 0.002,
                "max_drawdown": 0.07,
                "turnover": 2.0,
                "trades": 20,
            },
        ]
    )

    output = add_incrementality(rows)
    hmm = output[output["bucket"].eq("hmm_filter")].iloc[0]

    assert hmm["net_return_delta_vs_base"] == pytest.approx(0.02)
    assert hmm["net_return_delta_vs_same_hour"] == pytest.approx(0.01)
    assert hmm["net_return_delta_vs_shuffled"] == pytest.approx(0.015)


def test_incremental_decision_rejects_when_target_only_is_better() -> None:
    summary = pd.DataFrame(
        {
            "feature_group": ["cross_asset"],
            "median_net_delta_vs_shuffled": [0.01],
            "median_net_delta_vs_base": [0.01],
        }
    )
    ablation = pd.DataFrame(
        {
            "feature_group": ["target_only", "cross_asset"],
            "feature_set": ["target_only_frozen", "cross_asset_full_core"],
            "median_net_return": [0.02, 0.01],
        }
    )

    assert incremental_decision(summary, ablation) == "partial_incremental_but_not_superior_to_target_only"

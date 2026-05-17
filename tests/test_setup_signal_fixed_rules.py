from __future__ import annotations

import pandas as pd

from src.setup_signal_fixed_rules import aggregate_rule_summary, aggregate_target_summary


def test_fixed_rule_summary_requires_positive_targets_under_all_costs() -> None:
    rows = []
    for target in ["SPY", "QQQ", "KO"]:
        for scenario, net in [("ibkr_tiered_10000", 0.02), ("bps_2", 0.01), ("bps_5", 0.001)]:
            rows.append(
                {
                    "target": target,
                    "rule_name": "ob_long",
                    "family": "opening_bias_followthrough",
                    "direction": "long",
                    "horizon_bars": 4,
                    "params_json": "{}",
                    "split": "test",
                    "cost_scenario": scenario,
                    "cost_kind": "bps",
                    "folds": 2,
                    "positive_folds": 2,
                    "trades": 10,
                    "exposure": 0.1,
                    "gross_return": net + 0.001,
                    "cost_return": 0.001,
                    "net_return": net,
                    "avg_trade_net_mean": net / 10.0,
                    "min_fold_avg_trade_net": net / 20.0,
                    "hit_rate_mean": 0.6,
                    "profit_factor_median": 1.2,
                    "daily_sharpe_mean": 0.5,
                    "max_drawdown_max": 0.02,
                    "top_day_abs_net_share_max": 0.2,
                    "top_month_abs_net_share_max": 0.3,
                    "net_delta_vs_base_segment": 0.01,
                    "avg_trade_net_pooled": net / 10.0,
                    "positive_fold_share": 1.0,
                }
            )
    target_summary = pd.DataFrame(rows)
    config = {"setup_signal_fixed_rules": {"targets": ["SPY", "QQQ", "KO"], "min_positive_targets": 3}}

    summary = aggregate_rule_summary(target_summary, config)

    assert bool(summary.iloc[0]["promotable_family"]) is True
    assert summary.iloc[0]["stress_nonnegative_targets"] == 3


def test_target_summary_pools_fold_returns() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "target": "SPY",
                "rule_name": "ob_long",
                "family": "opening_bias_followthrough",
                "direction": "long",
                "horizon_bars": 4,
                "params_json": "{}",
                "fold": 0,
                "split": "test",
                "bucket": "fixed_rule",
                "cost_scenario": "ibkr_tiered_10000",
                "cost_kind": "ibkr",
                "trades": 2,
                "exposure": 0.1,
                "gross_return": 0.02,
                "cost_return": 0.002,
                "net_return": 0.018,
                "avg_trade_net": 0.009,
                "hit_rate": 0.5,
                "profit_factor": 1.2,
                "daily_sharpe": 0.3,
                "max_drawdown": 0.01,
                "top_day_abs_net_share": 0.2,
                "top_month_abs_net_share": 0.2,
                "net_delta_vs_base_segment": 0.01,
            },
            {
                "target": "SPY",
                "rule_name": "ob_long",
                "family": "opening_bias_followthrough",
                "direction": "long",
                "horizon_bars": 4,
                "params_json": "{}",
                "fold": 1,
                "split": "test",
                "bucket": "fixed_rule",
                "cost_scenario": "ibkr_tiered_10000",
                "cost_kind": "ibkr",
                "trades": 3,
                "exposure": 0.1,
                "gross_return": -0.01,
                "cost_return": 0.003,
                "net_return": -0.013,
                "avg_trade_net": -0.004333,
                "hit_rate": 0.4,
                "profit_factor": 0.8,
                "daily_sharpe": -0.2,
                "max_drawdown": 0.02,
                "top_day_abs_net_share": 0.3,
                "top_month_abs_net_share": 0.3,
                "net_delta_vs_base_segment": -0.02,
            },
        ]
    )

    summary = aggregate_target_summary(fold_metrics)

    assert summary.iloc[0]["folds"] == 2
    assert summary.iloc[0]["positive_folds"] == 1
    assert summary.iloc[0]["trades"] == 5
    assert abs(summary.iloc[0]["avg_trade_net_pooled"] - 0.001) < 1e-12

from __future__ import annotations

import pandas as pd
import pytest

from src.setup_signal_anti_concentration import (
    candidate_monthly_metrics,
    classify_anti_concentration,
    select_specs,
)


def _config() -> dict:
    return {
        "setup_signal_anti_concentration": {
            "min_trades": 20,
            "min_profit_factor": 1.05,
            "min_daily_sharpe": 0.30,
            "max_top_day_abs_net_share": 0.30,
            "min_months": 3,
            "min_positive_months": 2,
            "min_positive_month_rate": 0.50,
            "max_top_month_abs_net_share": 0.45,
            "min_leave_one_month_net": 0.0,
            "max_selected_per_fold": 1,
        }
    }


def test_candidate_monthly_metrics_flags_top_month_dependency() -> None:
    bars = pd.DataFrame(
        {
            "split": ["validation"] * 6,
            "candidate_id": ["c1"] * 6,
            "fold": [0] * 6,
            "month": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"],
            "net_return": [0.04, 0.02, 0.01, -0.005, -0.004, -0.001],
        }
    )

    metrics = candidate_monthly_metrics(bars).iloc[0]

    assert metrics["validation_months"] == 3
    assert metrics["positive_months"] == 2
    assert metrics["top_month_abs_net_share_rebuilt"] == pytest.approx(0.06 / 0.07)
    assert metrics["leave_one_month_min_net"] == pytest.approx(0.0)


def test_classify_anti_concentration_accepts_distributed_validation_edge() -> None:
    row = pd.Series(
        {
            "trades": 60,
            "net_return": 0.04,
            "avg_trade_net": 0.00067,
            "profit_factor": 1.4,
            "daily_sharpe": 1.0,
            "top_day_abs_net_share": 0.20,
            "validation_months": 4,
            "positive_months": 3,
            "positive_month_rate": 0.75,
            "top_month_abs_net_share_rebuilt": 0.35,
            "leave_one_month_min_net": 0.01,
            "median_month_net": 0.005,
        }
    )

    assert classify_anti_concentration(row, _config()) == "anti_concentration_candidate"


def test_select_specs_keeps_best_candidate_per_fold() -> None:
    ranked = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "fold": 0,
                "family": "breakdown_short_risk_off",
                "direction": "short",
                "horizon_bars": 4,
                "params_json": "{}",
                "column_map_json": "{}",
                "anti_status": "anti_concentration_candidate",
                "anti_score": 1.0,
                "net_return": 0.01,
            },
            {
                "candidate_id": "c2",
                "fold": 0,
                "family": "breakdown_short_risk_off",
                "direction": "short",
                "horizon_bars": 4,
                "params_json": "{}",
                "column_map_json": "{}",
                "anti_status": "anti_concentration_candidate",
                "anti_score": 2.0,
                "net_return": 0.02,
            },
            {
                "candidate_id": "c3",
                "fold": 1,
                "family": "breakdown_short_risk_off",
                "direction": "short",
                "horizon_bars": 4,
                "params_json": "{}",
                "column_map_json": "{}",
                "anti_status": "rejected_month_concentration",
                "anti_score": 9.0,
                "net_return": 0.09,
            },
        ]
    )

    selected = select_specs(ranked, _config())

    assert selected["candidate_id"].tolist() == ["c2"]

from __future__ import annotations

import pandas as pd
import pytest

from src.ablation import compare_test_variants, experiment_plan, run_signal_variant
from src.robustness import data_sufficiency


def _config() -> dict:
    return {
        "labeling": {"round_trip_cost_bps": 1.0},
        "walkforward": {
            "train_months": 5,
            "validation_months": 1,
            "test_months": 1,
            "step_months": 1,
            "purge_bars": 2,
            "embargo_bars": 0,
        },
        "signal": {
            "theta_prob_grid": [0.55],
            "theta_score_grid": [0.10],
            "max_neutral_grid": [0.55],
            "max_hmm_entropy_grid": [0.90],
            "allowed_hmm_states": [],
        },
    }


def _predictions() -> pd.DataFrame:
    rows = []
    for split in ["validation", "test"]:
        for idx, (p_up, p_down, fwd_ret) in enumerate([(0.70, 0.10, 0.002), (0.10, 0.70, -0.002), (0.40, 0.20, 0.001)]):
            rows.append(
                {
                    "timestamp": pd.Timestamp(f"2024-01-02 10:{30 + len(rows):02d}", tz="America/New_York"),
                    "session": "2024-01-02" if split == "validation" else "2024-01-03",
                    "bar_index": idx,
                    "target": 1 if fwd_ret > 0 else -1,
                    "fwd_ret": fwd_ret,
                    "neutral_zone": 0.001,
                    "p_down": p_down,
                    "p_neutral": 1.0 - p_up - p_down,
                    "p_up": p_up,
                    "predicted_class": 1 if p_up > p_down else -1,
                    "score": p_up - p_down,
                    "split": split,
                    "calibrated": split == "test",
                }
            )
    return pd.DataFrame(rows)


def _labels(months: int = 3) -> pd.DataFrame:
    rows = []
    start = pd.Period("2024-01", freq="M")
    for month_idx in range(months):
        month = start + month_idx
        rows.append({"timestamp": pd.Timestamp(f"{month}-01 10:30", tz="America/New_York"), "session": f"{month}-01", "bar_index": 0})
    return pd.DataFrame(rows)


def test_run_signal_variant_selects_thresholds_on_validation_and_scores_test() -> None:
    signals, selected = run_signal_variant("base_no_hmm", _predictions(), _config())

    test = signals[signals["split"] == "test"]
    assert selected == {"theta_prob": 0.55, "theta_score": 0.10, "max_neutral": 0.55, "max_hmm_entropy": 999.0}
    assert test["signal"].tolist() == [1, -1, 0]
    assert test["signal_net_ret"].sum() == pytest.approx(0.0038)


def test_compare_test_variants_reports_delta_vs_base() -> None:
    signal_metrics = pd.DataFrame(
        {
            "variant": ["base_no_hmm", "hmm_all_features_with_filter"],
            "split": ["test", "test"],
            "net_return": [0.001, 0.003],
            "trades": [2, 3],
            "avg_trade_net": [0.0005, 0.001],
            "hit_ratio": [0.5, 2 / 3],
        }
    )
    quality = pd.DataFrame(
        {
            "variant": ["base_no_hmm", "hmm_all_features_with_filter"],
            "split": ["test", "test"],
            "log_loss": [1.0, 0.9],
            "accuracy": [0.5, 0.6],
            "expected_calibration_error": [0.2, 0.1],
        }
    )

    comparison = compare_test_variants(signal_metrics, quality)

    hmm = comparison[comparison["variant"] == "hmm_all_features_with_filter"].iloc[0]
    assert hmm["delta_net_return_vs_base"] == pytest.approx(0.002)
    assert hmm["test_log_loss"] == pytest.approx(0.9)


def test_experiment_plan_blocks_xgboost_until_next_block() -> None:
    sufficiency = data_sufficiency(_labels(months=3), _config())
    plan = experiment_plan(sufficiency)

    assert "pending_long_intraday_history" in set(plan["status"])
    assert plan.loc[plan["variant"] == "xgboost_no_hmm", "status"].iloc[0] == "blocked_until_block_17"

from __future__ import annotations

import json

import pandas as pd

from src.freeze_spy_only_baseline import build_frozen_results, freeze_spy_only_baseline


def test_build_frozen_results_combines_baselines_and_candidates() -> None:
    baseline = pd.DataFrame(
        [
            {
                "strategy": "always_flat",
                "source": "reports/baseline_trades.parquet",
                "status": "benchmark",
                "cost_bps": 1.0,
                "trades": 0,
                "net_return": 0.0,
                "daily_sharpe_net": None,
                "profit_factor_net": None,
                "avg_trade_net": 0.0,
                "max_drawdown": 0.0,
                "folds_positive": 0,
                "folds_negative": 0,
            }
        ]
    )
    selected_tests = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "candidate_status": "candidate",
                "cost_bps": 1.0,
                "total_trades": 10,
                "total_net_return": 0.2,
                "median_daily_sharpe": 2.0,
                "median_profit_factor": 1.2,
                "avg_trade_net": 0.01,
                "max_drawdown_abs": 0.03,
                "positive_folds": 3,
                "negative_folds": 1,
            }
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "accepted": False,
                "cost_fragile": True,
            }
        ]
    )

    results = build_frozen_results(baseline, selected_tests, decisions)

    assert results["result_group"].tolist() == ["baseline_status", "hmm_candidate_threshold"]
    assert bool(results.loc[1, "cost_fragile"]) is True
    assert bool(results.loc[1, "accepted"]) is False


def test_freeze_spy_only_baseline_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports/hmm_candidate_thresholds").mkdir(parents=True)
    (tmp_path / "data/cleaned").mkdir(parents=True)

    (tmp_path / "configs/base.yaml").write_text(
        """
project:
  frequency: 5min
data:
  provider: polygon
  symbol: SPY
  cleaned_file: data/cleaned/spy_5min_clean.parquet
backtest:
  base_round_trip_cost_bps: 1.0
  conservative_round_trip_cost_bps: 2.0
  stress_round_trip_cost_bps: 5.0
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=2, freq="5min", tz="America/New_York"),
            "session": ["2024-01-02", "2024-01-02"],
        }
    ).to_parquet("data/cleaned/spy_5min_clean.parquet", index=False)
    pd.DataFrame(
        [
            {
                "strategy": "hmm_lr_walkforward_oos",
                "source": "reports/walkforward/fold_*/signals.parquet",
                "status": "rejected_economic",
                "cost_bps": 1.0,
                "trades": 4,
                "net_return": -0.01,
                "daily_sharpe_net": -0.5,
                "profit_factor_net": 0.8,
                "avg_trade_net": -0.0025,
                "max_drawdown": 0.02,
                "folds_positive": 1,
                "folds_negative": 2,
            }
        ]
    ).to_parquet("reports/baseline_status.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "source_rank": 1,
                "feature_set": "minimal",
                "status_1bps": "candidate",
                "status_2bps": "weak_sharpe",
                "accepted": False,
                "cost_fragile": True,
                "test_net_1bps": 0.1,
                "test_net_2bps": 0.02,
                "test_drawdown_1bps": 0.03,
                "test_drawdown_2bps": 0.04,
            }
        ]
    ).to_parquet("reports/hmm_candidate_thresholds/candidate_decisions.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "candidate_status": "candidate",
                "cost_bps": 1.0,
                "total_trades": 10,
                "total_net_return": 0.1,
                "median_daily_sharpe": 2.0,
                "median_profit_factor": 1.2,
                "avg_trade_net": 0.01,
                "max_drawdown_abs": 0.03,
                "positive_folds": 3,
                "negative_folds": 1,
            }
        ]
    ).to_parquet("reports/hmm_candidate_thresholds/selected_test_results.parquet", index=False)
    pd.DataFrame([{"status": "PASS"}]).to_parquet("reports/leakage_audit.parquet", index=False)

    report_path = freeze_spy_only_baseline()

    assert report_path == tmp_path / "reports/baseline_spy_only_frozen.md"
    assert (tmp_path / "baselines/spy_only_hmm/config.yaml").exists()
    assert (tmp_path / "baselines/spy_only_hmm/results.parquet").exists()
    summary = json.loads((tmp_path / "baselines/spy_only_hmm/summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "rejected_cost_fragile"
    assert summary["best_fallback_candidate"]["candidate_id"] == "candidate_a"

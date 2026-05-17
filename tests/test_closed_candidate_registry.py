from __future__ import annotations

import pandas as pd
import pytest

from src.closed_candidate_registry import build_registry, close_reason, infer_closed_status, normalize_decisions, reopen_requires


def test_infer_closed_status_maps_cost_fragility_from_decision() -> None:
    row = {"decision": "cost_fragile", "test_net_primary": 0.01, "test_net_stress": -0.02}

    assert infer_closed_status(row) == "rejected_cost_fragile"


def test_infer_closed_status_maps_positive_random_failure_to_unstable() -> None:
    row = {
        "decision": "research_candidate",
        "test_net_primary": 0.01,
        "test_net_stress": 0.005,
        "test_net_delta_vs_random_primary": -0.001,
    }

    assert infer_closed_status(row) == "accepted_research_only"


def test_close_reason_prefers_diagnostic_reasons() -> None:
    row = {"diagnostic_reasons": "thin_avg_trade_vs_5bps", "decision": "cost_fragile"}

    assert close_reason(row) == "thin_avg_trade_vs_5bps"


def test_reopen_requires_documents_status() -> None:
    assert "avg_trade" in reopen_requires("rejected_cost_fragile")
    assert "Nueva hipotesis" in reopen_requires("rejected_no_edge")


def test_normalize_decisions_merges_failure_attribution(tmp_path) -> None:
    result_dir = tmp_path / "results" / "15min_expansion" / "QQQ"
    result_dir.mkdir(parents=True)
    decisions_path = result_dir / "volatility_expansion_decisions.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "decision": "cost_fragile",
                "validation_status": "volatility_expansion_validation_candidate",
                "variant": "compression_breakout",
                "side": "long",
                "horizon_bars": 4,
                "fold": 1,
                "test_net_primary": 0.01,
                "test_avg_trade_net_primary": 0.0002,
                "test_trades_primary": 40,
                "test_net_stress": -0.01,
            }
        ]
    ).to_parquet(decisions_path, index=False)
    pd.DataFrame([{"candidate_id": "c1", "diagnostic_reasons": "stress_cost_fragility"}]).to_parquet(
        result_dir / "volatility_expansion_failure_attribution.parquet",
        index=False,
    )

    registry = normalize_decisions(decisions_path)

    assert registry.loc[0, "target_symbol"] == "QQQ"
    assert registry.loc[0, "family"] == "volatility_expansion"
    assert registry.loc[0, "hour_filter_name"] == "all"
    assert registry.loc[0, "closed_status"] == "rejected_cost_fragile"
    assert registry.loc[0, "close_reason"] == "stress_cost_fragility"


def test_normalize_decisions_maps_holdout_failure_to_closed_candidate(tmp_path) -> None:
    result_dir = tmp_path / "results" / "15min_expansion_frequency_repair" / "QQQ"
    result_dir.mkdir(parents=True)
    decisions_path = result_dir / "volatility_expansion_holdout_decisions.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "holdout_status": "holdout_failed",
                "failed_checks": "primary_net_positive, stress_net_positive",
                "holdout_net_primary": -0.006,
                "holdout_avg_trade_primary": -0.0001,
                "holdout_trades_primary": 43,
                "holdout_net_stress": -0.019,
            }
        ]
    ).to_parquet(decisions_path, index=False)

    registry = normalize_decisions(decisions_path)

    assert registry.loc[0, "family"] == "volatility_expansion"
    assert registry.loc[0, "decision"] == "holdout_failed"
    assert registry.loc[0, "validation_status"] == "posterior_holdout"
    assert registry.loc[0, "closed_status"] == "rejected_no_edge"
    assert registry.loc[0, "close_reason"] == "primary_net_positive, stress_net_positive"
    assert registry.loc[0, "test_net_primary"] == pytest.approx(-0.006)
    assert registry.loc[0, "test_avg_trade_net_primary"] == pytest.approx(-0.0001)


def test_build_registry_prefers_holdout_over_initial_acceptance(tmp_path) -> None:
    result_dir = tmp_path / "results" / "15min_expansion_frequency_repair" / "QQQ"
    result_dir.mkdir(parents=True)
    accepted_path = result_dir / "volatility_expansion_decisions.parquet"
    holdout_path = result_dir / "volatility_expansion_holdout_decisions.parquet"
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "decision": "accepted_candidate",
                "test_net_primary": 0.01,
                "test_avg_trade_net_primary": 0.0005,
                "test_trades_primary": 21,
                "test_net_stress": 0.005,
            }
        ]
    ).to_parquet(accepted_path, index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "holdout_status": "holdout_failed",
                "failed_checks": "primary_net_positive",
                "holdout_net_primary": -0.006,
                "holdout_avg_trade_primary": -0.0001,
                "holdout_trades_primary": 43,
                "holdout_net_stress": -0.019,
            }
        ]
    ).to_parquet(holdout_path, index=False)

    registry = build_registry([accepted_path, holdout_path])

    assert len(registry) == 1
    assert registry.loc[0, "decision"] == "holdout_failed"
    assert registry.loc[0, "closed_status"] == "rejected_no_edge"


def test_build_registry_deduplicates_candidates(tmp_path) -> None:
    result_dir = tmp_path / "results" / "15min_expansion" / "QQQ"
    result_dir.mkdir(parents=True)
    decisions_path = result_dir / "volatility_expansion_decisions.parquet"
    pd.DataFrame(
        [
            {"candidate_id": "c1", "decision": "rejected", "test_net_primary": -0.01},
            {"candidate_id": "c1", "decision": "rejected", "test_net_primary": -0.01},
        ]
    ).to_parquet(decisions_path, index=False)

    registry = build_registry([decisions_path])

    assert len(registry) == 1
    assert registry.loc[0, "closed_status"] == "rejected_no_edge"

from __future__ import annotations

from pathlib import Path

from src.research import manifest_from_strategy
from src.strategy import StrategySpec


def test_strategy_spec_validates_operable_contract() -> None:
    spec = StrategySpec.from_mapping(
        {
            "strategy_id": "qqq_15min_m6_v1",
            "target_symbol": "qqq",
            "timeframe": "15min",
            "feature_set_id": "cross_asset_liquid_15min",
            "alpha_id": "m6_base",
            "entry_rule": "next_open",
            "exit_rule": {"horizon_bars": 4},
            "position": {"side": "long_short", "max_gross_exposure": 1.0},
            "risk": {"no_new_trades_after": "15:45", "force_flat_before": "15:55", "max_turnover": 4.0},
            "cost_profile_id": "ibkr_tiered_10000",
            "split_policy_id": "wf_24m_6m_6m_step6m",
        }
    )

    assert spec.target_symbol == "QQQ"
    assert spec.entry_rule == "next_open"
    assert spec.exit_rule.horizon_bars == 4


def test_manifest_from_strategy_writes_reproducible_contract(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    feature_path.write_text("placeholder", encoding="utf-8")
    spec = StrategySpec.from_mapping(
        {
            "strategy_id": "qqq_15min_m6_v1",
            "target_symbol": "QQQ",
            "timeframe": "15min",
            "feature_set_id": "cross_asset_liquid_15min",
            "alpha_id": "m6_base",
            "entry_rule": "next_open",
            "exit_rule": {"horizon_bars": 4},
            "position": {"side": "long_short", "max_gross_exposure": 1.0},
            "risk": {"no_new_trades_after": "15:45", "force_flat_before": "15:55", "max_turnover": 4.0},
            "cost_profile_id": "ibkr_tiered_10000",
            "split_policy_id": "wf_24m_6m_6m_step6m",
        }
    )

    manifest = manifest_from_strategy(spec, run_type="alpha_research", feature_path=feature_path)
    output = manifest.write(tmp_path / "manifest.yaml")

    assert output.exists()
    data = manifest.to_dict()
    assert data["run"]["run_id"].startswith("alpha_research_qqq_15min_m6_v1")
    assert data["data"]["feature_fingerprint"].startswith("statsha1:")

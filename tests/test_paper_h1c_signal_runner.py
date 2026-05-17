from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.execution.paper_h1c_signal_runner import (
    H1COperationalThresholds,
    evaluate_h1c_signal,
    run_h1c_signal_runner,
    select_latest_frozen_thresholds,
)


def test_select_latest_frozen_thresholds_uses_highest_fold() -> None:
    thresholds = pd.DataFrame(
        [
            {"fold": 0, "risk_off_min": 1.0, "vix_z20_min": 2.0, "spread_credit_12_max_threshold": 0.0},
            {"fold": 4, "risk_off_min": 3.0, "vix_z20_min": 4.0, "spread_credit_12_max_threshold": -0.1},
        ]
    )

    selected = select_latest_frozen_thresholds(thresholds)

    assert selected.source_fold == 4
    assert selected.risk_off_min == 3.0
    assert selected.vix_z20_min == 4.0
    assert selected.spread_credit_12_max == -0.1


def test_evaluate_h1c_signal_requires_all_conditions_and_entry_available() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 14:00:00-04:00",
                "session": "2026-05-01",
                "bar_index": 10,
                "target_open_next": 100.0,
                "target_next_open_timestamp": "2026-05-01 14:15:00-04:00",
                "target_can_open_trade": True,
                "target_ret_6": -0.01,
                "target_ret_12": -0.02,
                "risk_off_score": 0.5,
                "prev_vix_z20": 1.0,
                "spread_credit_12": -0.001,
            },
            {
                "timestamp": "2026-05-01 15:45:00-04:00",
                "session": "2026-05-01",
                "bar_index": 25,
                "target_open_next": 101.0,
                "target_next_open_timestamp": "2026-05-01 16:00:00-04:00",
                "target_can_open_trade": False,
                "target_ret_6": -0.01,
                "target_ret_12": -0.02,
                "risk_off_score": 0.5,
                "prev_vix_z20": 1.0,
                "spread_credit_12": -0.001,
            },
        ]
    )
    thresholds = H1COperationalThresholds(source_fold=4, risk_off_min=0.1, vix_z20_min=0.2, spread_credit_12_max=0.0)

    signals = evaluate_h1c_signal(frame, thresholds, horizon_bars=1)

    assert bool(signals.iloc[0]["h1c_signal_short"]) is True
    assert bool(signals.iloc[0]["h1c_exit_available"]) is True
    assert signals.iloc[0]["desired_position_unit"] == -1.0
    assert bool(signals.iloc[1]["h1c_signal_short"]) is False
    assert signals.iloc[1]["desired_position_unit"] == 0.0


def _write_strategy_spec(path: Path, features_path: Path, risk_context_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "qqq_15min_risk_off_short_h1c_v1",
                "target_symbol": "QQQ",
                "timeframe": "15min",
                "feature_set_id": "test",
                "alpha_id": "risk_off_short_h1c_credit_spread_v1",
                "entry_rule": "next_open",
                "exit_rule": {"type": "fixed_horizon_open", "horizon_bars": 6},
                "position": {"side": "short_only", "max_gross_exposure": 1.0, "sizing": "fixed_unit"},
                "risk": {"no_new_trades_after": "15:45", "force_flat_before": "15:55", "max_turnover": 4.0},
                "cost_profile_id": "test",
                "split_policy_id": "test",
                "alpha": {"hypothesis_id": "H1c"},
                "data": {"feature_path": features_path.as_posix(), "risk_context_path": risk_context_path.as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_run_h1c_signal_runner_writes_signal_artifacts(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    risk_context_path = tmp_path / "risk_context.parquet"
    thresholds_path = tmp_path / "thresholds.parquet"
    strategy_path = tmp_path / "strategy.yaml"
    freeze_manifest_path = tmp_path / "freeze_manifest.yaml"
    config_path = tmp_path / "runner.yaml"

    features = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-01 14:00:00", tz="America/New_York") + pd.Timedelta(minutes=15 * idx),
                "session": "2026-05-01",
                "bar_index": 10 + idx,
                "target_open_next": 100.0 + idx,
                "target_next_open_timestamp": pd.Timestamp("2026-05-01 14:15:00", tz="America/New_York") + pd.Timedelta(minutes=15 * idx),
                "target_can_open_trade": idx == 0,
                "target_ret_6": -0.01,
                "target_ret_12": -0.02,
                "risk_off_score": 0.5,
                "spread_credit_12": -0.001,
            }
            for idx in range(7)
        ]
    )
    features.to_parquet(features_path, index=False)
    risk_context = pd.DataFrame(
        [
            {
                "source_date": "2026-04-30",
                "available_session": pd.Timestamp("2026-05-01"),
                "prev_vix_z20": 1.0,
            }
        ]
    )
    risk_context.to_parquet(risk_context_path, index=False)
    pd.DataFrame([{"fold": 4, "risk_off_min": 0.1, "vix_z20_min": 0.2, "spread_credit_12_max_threshold": 0.0}]).to_parquet(
        thresholds_path, index=False
    )
    _write_strategy_spec(strategy_path, features_path, risk_context_path)
    freeze_manifest_path.write_text("run:\n  status: freeze_review\n", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "runner": {"mode": "signal_only", "threshold_policy": "latest_frozen_fold", "max_data_staleness_days": 9999},
                "strategy": {
                    "strategy_spec_path": strategy_path.as_posix(),
                    "freeze_manifest_path": freeze_manifest_path.as_posix(),
                    "fold_thresholds_path": thresholds_path.as_posix(),
                },
                "data": {"features_path": features_path.as_posix(), "risk_context_path": risk_context_path.as_posix()},
                "paper": {
                    "target_account": "DU123",
                    "target_symbol": "QQQ",
                    "unit_size": 1.0,
                    "order_type": "MKT",
                    "time_in_force": "DAY",
                    "execution_timing": "next_bar_open_simulated",
                    "send_orders": False,
                },
                "outputs": {"output_dir": (tmp_path / "runs").as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    paths, summary = run_h1c_signal_runner(config_path=config_path, as_of="2026-05-01 14:00:00-04:00")

    assert paths.signals_path.exists()
    assert paths.ticket_path.exists()
    assert paths.report_path.exists()
    assert summary["ticket"]["action"] == "SELL"
    assert summary["ticket"]["theoretical_exit_timestamp"] == "2026-05-01 15:45:00-04:00"
    assert summary["ticket"]["send_orders"] is False


def test_run_h1c_signal_runner_rejects_send_orders_true(tmp_path: Path) -> None:
    config_path = tmp_path / "runner.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runner": {"mode": "signal_only"},
                "strategy": {},
                "data": {},
                "paper": {"send_orders": True},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="send_orders=false"):
        run_h1c_signal_runner(config_path=config_path)

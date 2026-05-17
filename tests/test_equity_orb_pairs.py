from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.strategy.equity_orb_pairs import (
    OrbWindow,
    PairSpec,
    add_orb_range,
    build_pair_frame,
    failed_reversion_directions,
    run_strategy,
    simulate_pair_base,
    spread_directions,
)
from src.strategy.equity_orb_failed_pairs import run_strategy as run_failed_strategy
from src.strategy.equity_orb_range_quality import (
    RangeQualityFilter,
    apply_width_filter,
    fit_width_thresholds,
    run_strategy as run_range_quality_strategy,
)


def _panel(sessions: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for session in sessions or ["2026-01-02", "2026-01-05"]:
        timestamps = pd.date_range(f"{session} 09:30", periods=8, freq="5min", tz="America/New_York")
        for i, timestamp in enumerate(timestamps):
            a = [100.0, 100.2, 101.2, 101.6, 102.0, 102.4, 102.8, 103.0][i]
            b = [100.0, 100.0, 100.0, 99.9, 99.9, 99.8, 99.8, 99.8][i]
            rows.append(
                {
                    "timestamp": timestamp,
                    "session": session,
                    "bar_index": i,
                    "AAA__open": a,
                    "AAA__close": a,
                    "BBB__open": b,
                    "BBB__close": b,
                    "SPY__open": 100.0 + i * 0.1,
                    "SPY__close": 100.0 + i * 0.1,
                    "is_available_AAA": True,
                    "is_available_BBB": True,
                    "is_available_SPY": True,
                }
            )
    return pd.DataFrame(rows)


def test_spread_orb_generates_long_pair_trade() -> None:
    panel = _panel()
    pair = PairSpec(pair_id="AAA_BBB", asset_a="AAA", asset_b="BBB")
    window = OrbWindow(window_id="orb_10m", label="09:30-09:40", range_bars=2)
    pair_frame = add_orb_range(build_pair_frame(panel, pair), window)

    directions = spread_directions(pair_frame)
    trades = simulate_pair_base(
        pair_frame,
        directions,
        label="orb_spread_breakout",
        fold=0,
        split="validation",
        horizon=2,
        strategy_id="test_orb",
    )

    assert not trades.empty
    assert set(trades["side"]) == {"long_spread"}
    assert (trades["gross_return"] > 0).all()


def test_failed_orb_reverses_after_reentry_inside_range() -> None:
    panel = _panel()
    pair = PairSpec(pair_id="AAA_BBB", asset_a="AAA", asset_b="BBB")
    window = OrbWindow(window_id="orb_10m", label="09:30-09:40", range_bars=2)
    pair_frame = add_orb_range(build_pair_frame(panel, pair), window)

    directions = failed_reversion_directions(pair_frame)
    trades = simulate_pair_base(
        pair_frame,
        directions,
        label="failed_orb_reversion",
        fold=0,
        split="validation",
        horizon=2,
        strategy_id="test_failed_orb",
    )

    # The synthetic panel breaks upward and never re-enters, so no failed ORB.
    assert trades.empty


def test_run_strategy_writes_standard_artifacts(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    _panel(["2026-01-02", "2026-02-02", "2026-03-02"]).to_parquet(panel_path, index=False)
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "results"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "test_equity_orb_pairs",
                "hypothesis_id": "H2.2",
                "timeframe": "5min",
                "data": {"panel_path": panel_path.as_posix()},
                "pairs": [{"pair_id": "AAA_BBB", "asset_a": "AAA", "asset_b": "BBB"}],
                "orb": {"windows": [{"window_id": "orb_10m", "label": "09:30-09:40", "range_bars": 2}]},
                "exit": {"horizon_bars": [2]},
                "costs": {"round_trip_bps_per_leg": [0.0, 2.0]},
                "controls": {"directional_orb_baseline": True, "random_same_frequency": True, "same_hour": True, "market_beta": True},
                "split_policy": {"train_months": 1, "validation_months": 1, "test_months": 1, "step_months": 1, "embargo_sessions": 0},
                "outputs": {"output_dir": output_dir.as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    outputs = run_strategy(config_path=config_path)

    assert outputs.trades_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.report_path.exists()


def test_run_failed_strategy_writes_standard_artifacts(tmp_path: Path) -> None:
    rows = []
    for session in ["2026-01-02", "2026-02-02", "2026-03-02"]:
        timestamps = pd.date_range(f"{session} 09:30", periods=8, freq="5min", tz="America/New_York")
        a_values = [100.0, 100.1, 101.5, 100.05, 99.8, 99.7, 99.6, 99.5]
        b_values = [100.0] * 8
        for i, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "session": session,
                    "bar_index": i,
                    "AAA__open": a_values[i],
                    "AAA__close": a_values[i],
                    "BBB__open": b_values[i],
                    "BBB__close": b_values[i],
                    "SPY__open": 100.0,
                    "SPY__close": 100.0,
                    "is_available_AAA": True,
                    "is_available_BBB": True,
                    "is_available_SPY": True,
                }
            )
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    config_path = tmp_path / "failed_config.yaml"
    output_dir = tmp_path / "failed_results"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "test_failed_orb_pairs",
                "hypothesis_id": "H2.5",
                "timeframe": "5min",
                "data": {"panel_path": panel_path.as_posix()},
                "pairs": [{"pair_id": "AAA_BBB", "asset_a": "AAA", "asset_b": "BBB"}],
                "orb": {"windows": [{"window_id": "orb_10m", "label": "09:30-09:40", "range_bars": 2}]},
                "exit": {"horizon_bars": [2]},
                "costs": {"round_trip_bps_per_leg": [0.0, 2.0]},
                "controls": {"directional_failed_orb_baseline": True, "random_same_frequency": True, "same_hour": True, "market_beta": True, "continuation_reference": True},
                "split_policy": {"train_months": 1, "validation_months": 1, "test_months": 1, "step_months": 1, "embargo_sessions": 0},
                "outputs": {"output_dir": output_dir.as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    outputs = run_failed_strategy(config_path=config_path)

    assert outputs.trades_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.report_path.exists()


def test_range_quality_thresholds_use_train_sessions_only() -> None:
    frame = pd.DataFrame(
        {
            "session": ["s1", "s2", "s3", "s4", "s5"],
            "orb_width": [1.0, 2.0, 3.0, 100.0, 200.0],
            "orb_observations": [2, 2, 2, 2, 2],
            "orb_range_bars": [2, 2, 2, 2, 2],
        }
    )
    quality_filter = RangeQualityFilter(
        filter_id="width_mid_25_75",
        label="train width percentile 25-75",
        min_percentile=0.25,
        max_percentile=0.75,
    )

    thresholds = fit_width_thresholds(
        frame,
        ("s1", "s2", "s3", "s4"),
        (quality_filter,),
        fold=0,
        pair_id="AAA_BBB",
        orb_window="orb_10m",
    )
    filtered = apply_width_filter(frame, pd.Series([1, 1, 1, 1, 1], index=frame.index), thresholds.iloc[0])

    train_widths = pd.Series([1.0, 2.0, 3.0, 100.0])
    assert thresholds.loc[0, "lower_width"] == train_widths.quantile(0.25)
    assert thresholds.loc[0, "upper_width"] == train_widths.quantile(0.75)
    assert filtered.tolist() == [0, 1, 1, 0, 0]


def test_run_range_quality_strategy_writes_standard_artifacts(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    _panel(["2026-01-02", "2026-02-02", "2026-03-02"]).to_parquet(panel_path, index=False)
    config_path = tmp_path / "range_quality_config.yaml"
    output_dir = tmp_path / "range_quality_results"
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategy_id": "test_equity_orb_range_quality",
                "hypothesis_id": "H2.4",
                "timeframe": "5min",
                "data": {"panel_path": panel_path.as_posix()},
                "pairs": [{"pair_id": "AAA_BBB", "asset_a": "AAA", "asset_b": "BBB"}],
                "orb": {"windows": [{"window_id": "orb_10m", "label": "09:30-09:40", "range_bars": 2}]},
                "range_quality": {
                    "filters": [
                        {
                            "filter_id": "width_mid_20_80",
                            "label": "train width percentile 20-80",
                            "min_percentile": 0.20,
                            "max_percentile": 0.80,
                        }
                    ]
                },
                "exit": {"horizon_bars": [2]},
                "costs": {"round_trip_bps_per_leg": [0.0, 2.0]},
                "controls": {"unfiltered_reference": True, "random_same_frequency": True, "same_hour": True, "market_beta": True},
                "split_policy": {"train_months": 1, "validation_months": 1, "test_months": 1, "step_months": 1, "embargo_sessions": 0},
                "outputs": {"output_dir": output_dir.as_posix()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    outputs = run_range_quality_strategy(config_path=config_path)

    assert outputs.trades_path.exists()
    assert outputs.thresholds_path.exists()
    assert outputs.summary_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.report_path.exists()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering_cross_asset import build_cross_asset_features, run


def _lab_config(tmp_path: Path) -> dict:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
context_universe_core:
  - SPY
  - QQQ
  - IWM
  - DIA
  - XLK
  - XLF
  - XLE
  - XLV
  - XLY
  - XLP
  - XLU
  - TLT
  - IEF
  - HYG
  - LQD
  - GLD
  - USO
""".strip(),
        encoding="utf-8",
    )
    return {
        "project": {"timezone": "America/New_York", "frequency": "5min"},
        "lab": {
            "target_symbol": "SPY",
            "timeframe": "5min",
            "universe_config": str(universe_path),
            "universe_id": "test_universe",
            "context_key": "context_universe_core",
        },
        "paths": {"aligned_dir": str(tmp_path / "data/aligned"), "reports_dir": str(tmp_path / "reports")},
    }


def _feature_config() -> dict:
    return {
        "feature_set_version": "cross_asset_v1",
        "return_windows": [1, 3, 6, 12, 24],
        "range_ratio_short_window": 6,
        "range_ratio_long_window": 24,
        "efficiency_window": 12,
        "vwap_slope_window": 12,
        "setup_features": {
            "opening_range_bars": [3, 6],
            "rolling_breakout_windows": [12],
            "realized_vol_windows": [3, 6, 12],
            "expected_by_bar_min_periods": 1,
        },
        "groups": {
            "indices": ["SPY", "QQQ", "IWM", "DIA"],
            "sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU"],
            "growth": ["QQQ", "XLK", "XLY"],
            "cyclicals": ["IWM", "XLY", "XLF", "XLE"],
            "defensives": ["XLP", "XLV", "XLU"],
            "bonds": ["TLT", "IEF"],
        },
        "hmm_feature_columns": ["target_ret_3", "relret_QQQ_SPY_6", "risk_on_score"],
    }


def _panel(symbols: list[str], sessions: int = 2, bars: int = 30) -> pd.DataFrame:
    frames = []
    for session_idx in range(sessions):
        session = f"2024-01-{session_idx + 2:02d}"
        timestamps = pd.date_range(f"{session} 09:30", periods=bars, freq="5min", tz="America/New_York")
        columns: dict[str, object] = {"timestamp": timestamps, "session": [session] * bars, "bar_index": np.arange(bars)}
        for symbol_idx, symbol in enumerate(symbols):
            base = 100.0 + symbol_idx * 10 + session_idx * 2
            close = base + np.arange(bars, dtype=float) * (1.0 + symbol_idx / 20.0)
            columns[f"{symbol}__open"] = close - 0.1
            columns[f"{symbol}__high"] = close + 0.5
            columns[f"{symbol}__low"] = close - 0.5
            columns[f"{symbol}__close"] = close
            columns[f"{symbol}__volume"] = np.full(bars, 1000 + symbol_idx)
            columns[f"is_available_{symbol}"] = np.full(bars, True)
        frame = pd.DataFrame(columns)
        frame["SPY__target_open_next"] = frame.groupby("session")["SPY__open"].shift(-1)
        frame["SPY__next_open_timestamp"] = frame.groupby("session")["timestamp"].shift(-1)
        frame["SPY__target_crosses_session_close"] = False
        frame["SPY__can_open_trade"] = True
        frame["SPY__force_flat_bar"] = False
        frame["SPY__trade_could_remain_open_past_close"] = False
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_cross_asset_features_create_target_and_relative_columns(tmp_path) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    features = build_cross_asset_features(_panel(symbols), _lab_config(tmp_path), _feature_config(), symbols=symbols)

    expected = {
        "target_ret_3",
        "target_range_ratio_6_24",
        "target_signed_efficiency_12",
        "target_dist_vwap_atr",
        "target_overnight_ret",
        "target_or_6_high",
        "target_rv_12_rel_by_bar",
        "target_failed_breakout_high_12",
        "target_rel_volume_by_bar",
        "target_rel_volume_accel_2",
        "target_breakout_margin_or_6_high_atr",
        "target_breakout_attempt_count_or_6_high",
        "target_dist_prev_high_atr",
        "relopen_QQQ_SPY",
        "relret_QQQ_SPY_6",
        "relret_HYG_LQD_12",
        "spread_growth_defensive_12",
        "positive_index_count_open",
        "index_above_vwap_count",
        "risk_on_open_confirmation",
        "positive_index_count_6",
        "sector_above_vwap_count",
        "cross_asset_vol_expansion_score",
        "risk_on_score",
        "risk_off_score",
        "chop_score",
    }
    assert expected.issubset(features.columns)
    assert "target_open_next" in features.columns


def test_cross_asset_features_can_emit_additional_clock_windows(tmp_path) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    config = _feature_config()
    config["range_ratio_windows"] = [[2, 8], [6, 24]]
    config["efficiency_windows"] = [4, 12]
    config["vwap_slope_windows"] = [4, 12]
    config["atr_windows"] = [2, 4, 12]
    config["breadth_windows"] = [2, 6, 12, 24]
    config["setup_features"]["atr_column"] = "target_atr_2"
    config["setup_features"]["breakout_persistence_windows"] = [2]
    config["setup_features"]["volume_accel_windows"] = [2]

    features = build_cross_asset_features(_panel(symbols), _lab_config(tmp_path), config, symbols=symbols)

    assert "target_range_ratio_2_8" in features.columns
    assert "market_range_ratio_2_8" in features.columns
    assert "target_signed_efficiency_4" in features.columns
    assert "target_vwap_slope_4" in features.columns
    assert "target_atr_2" in features.columns
    assert "target_dist_vwap_atr_2" in features.columns
    assert "positive_index_count_2" in features.columns


def test_returns_and_relative_returns_are_same_session_and_same_timestamp(tmp_path) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    features = build_cross_asset_features(_panel(symbols), _lab_config(tmp_path), _feature_config(), symbols=symbols)
    second_session = features[features["session"] == "2024-01-03"].reset_index(drop=True)

    assert np.isnan(second_session.loc[0, "ret_SPY_1"])
    row = features[(features["session"] == "2024-01-02") & (features["bar_index"] == 6)].iloc[0]
    assert row["relret_QQQ_SPY_6"] == row["ret_QQQ_6"] - row["ret_SPY_6"]


def test_target_intraday_features_are_causal(tmp_path) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    panel = _panel(symbols)
    features = build_cross_asset_features(panel, _lab_config(tmp_path), _feature_config(), symbols=symbols)

    row = features[(features["session"] == "2024-01-02") & (features["bar_index"] == 12)].iloc[0]
    expected_dist_open = np.log(panel.loc[12, "SPY__close"] / panel.loc[0, "SPY__open"])
    assert row["target_dist_open"] == expected_dist_open
    assert row["target_signed_efficiency_12"] == 1.0
    assert 0.0 <= row["target_pos_session_range"] <= 1.0


def test_target_setup_features_use_completed_intraday_information(tmp_path) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    panel = _panel(symbols, sessions=3, bars=30)
    features = build_cross_asset_features(panel, _lab_config(tmp_path), _feature_config(), symbols=symbols)

    first_session = features[features["session"] == "2024-01-02"].reset_index(drop=True)
    assert pd.isna(first_session.loc[4, "target_or_6_high"])
    assert first_session.loc[5, "target_or_6_high"] == panel.loc[:5, "SPY__high"].max()
    assert "target_breakout_margin_or_6_high_atr" in features.columns
    assert first_session.loc[5, "target_breakout_attempt_count_or_6_high"] >= 0

    third_session = features[features["session"] == "2024-01-04"].reset_index(drop=True)
    assert third_session.loc[0, "target_prev_session_close"] == panel[panel["session"] == "2024-01-03"]["SPY__close"].iloc[-1]
    assert third_session.loc[0, "target_prev_session_high"] == panel[panel["session"] == "2024-01-03"]["SPY__high"].max()
    assert pd.notna(third_session.loc[12, "target_rel_volume_by_bar"])
    assert "target_rel_volume_accel_2" in features.columns
    assert "target_failed_breakout_high_12" in features.columns


def test_run_writes_features_and_report(tmp_path, monkeypatch) -> None:
    symbols = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "TLT", "IEF", "HYG", "LQD", "GLD", "USO"]
    lab_config = _lab_config(tmp_path)
    panel_path = tmp_path / "data/aligned/SPY/5min/test_universe/panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel(symbols).to_parquet(panel_path, index=False)

    lab_config_path = tmp_path / "hmm_lab.yaml"
    feature_config_path = tmp_path / "cross_asset_v1.yaml"
    import yaml

    lab_config_path.write_text(yaml.safe_dump(lab_config), encoding="utf-8")
    feature_config_path.write_text(yaml.safe_dump(_feature_config()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    output = run(lab_config_path, feature_config_path, "SPY")

    assert output.exists()
    assert output.as_posix().endswith("data/features/SPY/5min/test_universe/cross_asset_v1/features.parquet")
    assert (tmp_path / "reports/SPY/cross_asset_features_cross_asset_v1.md").exists()

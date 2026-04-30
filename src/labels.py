from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _grouped_shift(df: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return df.groupby("session", sort=False)[column].shift(periods)


def build_labels(features: pd.DataFrame, config: dict[str, Any], drop_invalid: bool = True) -> pd.DataFrame:
    required = {
        "timestamp",
        "open",
        "session",
        "bar_index",
        "bars_in_session",
        "rv_12",
        "target_crosses_session_close",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Features data is missing required columns: {missing}")

    labeling_cfg = config["labeling"]
    horizon = int(labeling_cfg["horizon_bars"])
    round_trip_cost = float(labeling_cfg["round_trip_cost_bps"]) / 10_000.0
    buffer = float(labeling_cfg["buffer_bps"]) / 10_000.0
    lambda_vol = float(labeling_cfg["lambda_vol"])

    labeled = features.sort_values(["session", "bar_index"]).reset_index(drop=True).copy()
    labeled["horizon_bars"] = horizon
    labeled["entry_px"] = _grouped_shift(labeled, "open", -1)
    labeled["exit_px"] = _grouped_shift(labeled, "open", -(horizon + 1))
    labeled["entry_timestamp"] = _grouped_shift(labeled, "timestamp", -1)
    labeled["exit_timestamp"] = _grouped_shift(labeled, "timestamp", -(horizon + 1))
    labeled["fwd_ret"] = np.log(labeled["exit_px"] / labeled["entry_px"])
    labeled["sigma_h"] = labeled["rv_12"] * np.sqrt(horizon)
    labeled["neutral_zone"] = np.maximum(round_trip_cost + buffer, lambda_vol * labeled["sigma_h"])

    labeled["target"] = 0
    labeled.loc[labeled["fwd_ret"] > labeled["neutral_zone"], "target"] = 1
    labeled.loc[labeled["fwd_ret"] < -labeled["neutral_zone"], "target"] = -1

    target_crosses = labeled["bar_index"] + horizon + 1 >= labeled["bars_in_session"]
    labeled["target_crosses_session_close"] = target_crosses

    if drop_invalid:
        valid_mask = (
            labeled["entry_px"].notna()
            & labeled["exit_px"].notna()
            & labeled["fwd_ret"].notna()
            & labeled["sigma_h"].notna()
            & labeled["neutral_zone"].notna()
            & ~labeled["target_crosses_session_close"]
        )
        labeled = labeled.loc[valid_mask].reset_index(drop=True)

    return labeled


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    input_path = Path(config["data"]["features_file"])
    output_path = Path(config["data"]["labels_file"])

    features = pd.read_parquet(input_path)
    labels = build_labels(features, config, drop_invalid=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ternary labels for next-open intraday targets.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    output_path = run(args.config)
    print(f"Labels written to: {output_path}")


if __name__ == "__main__":
    main()

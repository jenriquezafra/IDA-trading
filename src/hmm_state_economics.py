from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.labels import build_labels
from src.walkforward import _fold_hmm_features, build_monthly_folds


ACTIONS = ("long", "short", "momentum_ret_3", "reversion_ret_3", "random_symmetric", "flat")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _config_for_horizon(config: dict[str, Any], horizon: int) -> dict[str, Any]:
    copied = deepcopy(config)
    copied["labeling"]["horizon_bars"] = int(horizon)
    return copied


def _position_for_action(frame: pd.DataFrame, action: str) -> pd.Series:
    if action == "long":
        return pd.Series(1, index=frame.index, dtype="int64")
    if action == "short":
        return pd.Series(-1, index=frame.index, dtype="int64")
    if action == "flat":
        return pd.Series(0, index=frame.index, dtype="int64")
    if action == "random_symmetric":
        hashed = pd.util.hash_pandas_object(frame[["session", "bar_index"]], index=False)
        return pd.Series(np.where(hashed.to_numpy() % 2 == 0, 1, -1), index=frame.index, dtype="int64")
    if action in {"momentum_ret_3", "reversion_ret_3"}:
        signal = pd.Series(0, index=frame.index, dtype="int64")
        signal.loc[frame["ret_3"] > frame["neutral_zone"]] = 1
        signal.loc[frame["ret_3"] < -frame["neutral_zone"]] = -1
        return signal if action == "momentum_ret_3" else -signal
    raise ValueError(f"Unsupported action: {action}")


def _daily_sharpe(frame: pd.DataFrame, net_ret: pd.Series) -> float:
    if frame.empty:
        return np.nan
    daily = net_ret.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0].sum()
    gross_loss = -active_net[active_net < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def _state_duration_stats(split_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    ordered = split_frame.sort_values(["session", "bar_index"])
    for session, group in ordered.groupby("session", sort=False):
        states = group["hmm_state"].astype(int).tolist()
        if not states:
            continue
        run_state = states[0]
        run_length = 1
        for state in states[1:]:
            if state == run_state:
                run_length += 1
            else:
                rows.append({"hmm_state": run_state, "duration": run_length})
                run_state = state
                run_length = 1
        rows.append({"hmm_state": run_state, "duration": run_length})

    if not rows:
        return pd.DataFrame(columns=["hmm_state", "mean_duration", "runs"])
    return (
        pd.DataFrame(rows)
        .groupby("hmm_state", as_index=False)
        .agg(mean_duration=("duration", "mean"), runs=("duration", "size"))
    )


def _state_persistence(split_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = split_frame.sort_values(["session", "bar_index"]).copy()
    ordered["next_state"] = ordered.groupby("session", sort=False)["hmm_state"].shift(-1)
    valid = ordered.dropna(subset=["next_state"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["hmm_state", "persistence"])
    valid["same_next_state"] = valid["hmm_state"].astype(int) == valid["next_state"].astype(int)
    return valid.groupby("hmm_state", as_index=False).agg(persistence=("same_next_state", "mean"))


def state_structure(split_frame: pd.DataFrame, n_states: int) -> pd.DataFrame:
    total = len(split_frame)
    counts = split_frame["hmm_state"].value_counts().reindex(range(n_states), fill_value=0).rename_axis("hmm_state").reset_index(name="state_rows")
    counts["frequency"] = counts["state_rows"] / total if total else 0.0
    durations = _state_duration_stats(split_frame)
    persistence = _state_persistence(split_frame)
    return (
        counts.merge(durations, on="hmm_state", how="left")
        .merge(persistence, on="hmm_state", how="left")
        .fillna({"mean_duration": 0.0, "runs": 0, "persistence": 0.0})
    )


def evaluate_state_action(
    frame: pd.DataFrame,
    action: str,
    cost_bps: float,
) -> dict[str, float | int]:
    position = _position_for_action(frame, action)
    active = position != 0
    gross = position * frame["fwd_ret"]
    cost = position.abs() * (float(cost_bps) / 10_000.0)
    net = gross - cost
    active_net = net[active]
    return {
        "rows": int(len(frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(frame) else 0.0,
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "net_return": float(net.sum()),
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "median_trade_net": float(active_net.median()) if len(active_net) else 0.0,
        "hit_ratio": float((active_net > 0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(frame, net),
    }


def _prepare_horizon_labels(features: pd.DataFrame, config: dict[str, Any], horizons: list[int]) -> dict[int, pd.DataFrame]:
    labels_by_horizon = {}
    for horizon in horizons:
        labels_by_horizon[int(horizon)] = build_labels(features, _config_for_horizon(config, int(horizon)), drop_invalid=True)
    return labels_by_horizon


def _hour_distribution(frame: pd.DataFrame, fold: int, split: str, horizon: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["fold", "split", "horizon_bars", "hmm_state", "hour", "rows", "row_pct"])
    output = frame.copy()
    output["hour"] = output["timestamp"].dt.hour
    counts = output.groupby(["hmm_state", "hour"], as_index=False).size().rename(columns={"size": "rows"})
    totals = output.groupby("hmm_state").size().rename("state_total").reset_index()
    counts = counts.merge(totals, on="hmm_state", how="left")
    counts["row_pct"] = counts["rows"] / counts["state_total"]
    counts.insert(0, "horizon_bars", int(horizon))
    counts.insert(0, "split", split)
    counts.insert(0, "fold", int(fold))
    return counts.drop(columns=["state_total"])


def run_fold_state_economics(
    features: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
    fold,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    n_states = int(config["hmm"]["n_states"])
    cost_grid = [float(value) for value in config.get("robustness", {}).get("cost_bps", [config["labeling"]["round_trip_cost_bps"]])]
    filtered_hmm, _, _ = _fold_hmm_features(features, fold, config)
    hmm_cols = ["timestamp", "session", "bar_index", "hmm_state", "hmm_entropy", "hmm_max_prob"]

    rows: list[dict[str, Any]] = []
    hour_frames: list[pd.DataFrame] = []
    fold_sessions = fold.train_sessions + fold.validation_sessions + fold.test_sessions

    for horizon, labels in labels_by_horizon.items():
        fold_labels = labels[labels["session"].isin(fold_sessions)].copy()
        merged = fold_labels.merge(filtered_hmm[hmm_cols], on=["timestamp", "session", "bar_index"], how="inner", validate="one_to_one")

        for split, sessions in [("validation", fold.validation_sessions), ("test", fold.test_sessions)]:
            split_frame = merged[merged["session"].isin(sessions)].copy()
            if split_frame.empty:
                continue
            structure = state_structure(split_frame, n_states)
            hour_frames.append(_hour_distribution(split_frame, fold.fold, split, horizon))

            for state in range(n_states):
                state_frame = split_frame[split_frame["hmm_state"] == state].copy()
                state_info = structure[structure["hmm_state"] == state].iloc[0].to_dict()
                for cost_bps in cost_grid:
                    for action in ACTIONS:
                        metrics = evaluate_state_action(state_frame, action, cost_bps)
                        rows.append(
                            {
                                "fold": int(fold.fold),
                                "train_months": ",".join(fold.train_months),
                                "validation_months": ",".join(fold.validation_months),
                                "test_months": ",".join(fold.test_months),
                                "split": split,
                                "horizon_bars": int(horizon),
                                "cost_bps": float(cost_bps),
                                "hmm_state": int(state),
                                "action": action,
                                "state_rows": int(state_info["state_rows"]),
                                "state_frequency": float(state_info["frequency"]),
                                "mean_duration": float(state_info["mean_duration"]),
                                "persistence": float(state_info["persistence"]),
                                **metrics,
                            }
                        )
    return rows, hour_frames


def aggregate_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics[(metrics["split"] == "validation") & (metrics["action"] != "flat")].copy()
    if active.empty:
        return pd.DataFrame()

    grouped = (
        active.groupby(["horizon_bars", "cost_bps", "hmm_state", "action"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            total_trades=("trades", "sum"),
            total_net_return=("net_return", "sum"),
            avg_trade_net=("avg_trade_net", "mean"),
            median_profit_factor=("profit_factor", "median"),
            median_daily_sharpe=("daily_sharpe", "median"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            negative_folds=("net_return", lambda values: int((values < 0).sum())),
            avg_state_frequency=("state_frequency", "mean"),
            avg_persistence=("persistence", "mean"),
        )
        .sort_values(["total_net_return", "avg_trade_net", "total_trades"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    grouped["candidate_status"] = grouped.apply(classify_candidate, axis=1)
    return grouped


def classify_candidate(row: pd.Series) -> str:
    if row.get("action") == "random_symmetric":
        return "random_benchmark"
    min_trades = 50
    if int(row["total_trades"]) < min_trades:
        return "insufficient_trades"
    if row["total_net_return"] <= 0 or row["avg_trade_net"] <= 0:
        return "negative_economic"
    if row["median_profit_factor"] <= 1.10:
        return "weak_profit_factor"
    if row["median_daily_sharpe"] <= 1.0:
        return "weak_sharpe"
    if int(row["positive_folds"]) <= int(row["negative_folds"]):
        return "not_stable_across_folds"
    return "candidate"


def classify_state_roles(metrics: pd.DataFrame) -> pd.DataFrame:
    validation = metrics[metrics["split"] == "validation"].copy()
    if validation.empty:
        return pd.DataFrame()

    grouped = (
        validation.groupby(["horizon_bars", "cost_bps", "hmm_state", "action"], as_index=False)
        .agg(total_net_return=("net_return", "sum"), total_trades=("trades", "sum"), avg_trade_net=("avg_trade_net", "mean"))
        .sort_values(["horizon_bars", "cost_bps", "hmm_state", "total_net_return"], ascending=[True, True, True, False])
    )
    best = grouped.groupby(["horizon_bars", "cost_bps", "hmm_state"], as_index=False).head(1).reset_index(drop=True)
    best["state_role"] = best["action"].map(
        {
            "long": "drift_positive",
            "short": "drift_negative",
            "momentum_ret_3": "momentum_bias",
            "reversion_ret_3": "reversion_bias",
            "random_symmetric": "random_like",
            "flat": "no_trade",
        }
    )
    best["economic_label"] = np.where(
        (best["total_net_return"] > 0) & (best["avg_trade_net"] > 0) & (best["action"] != "random_symmetric"),
        "positive_validation_bias",
        "not_exploitable",
    )
    return best


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if value == np.inf:
        return "inf"
    if value == -np.inf:
        return "-inf"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    headers = display.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_report(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    state_roles: pd.DataFrame,
    hour_distribution: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    candidates = ranking[ranking["candidate_status"] == "candidate"] if not ranking.empty else pd.DataFrame()
    top_cols = [
        "horizon_bars",
        "cost_bps",
        "hmm_state",
        "action",
        "total_trades",
        "total_net_return",
        "avg_trade_net",
        "median_profit_factor",
        "median_daily_sharpe",
        "positive_folds",
        "negative_folds",
        "candidate_status",
    ]
    status_counts = ranking["candidate_status"].value_counts().rename_axis("candidate_status").reset_index(name="count") if not ranking.empty else pd.DataFrame()
    best_by_state = (
        ranking.sort_values(["horizon_bars", "cost_bps", "hmm_state", "total_net_return"], ascending=[True, True, True, False])
        .groupby(["horizon_bars", "cost_bps", "hmm_state"], as_index=False)
        .head(1)
        .loc[:, top_cols]
    )

    conclusion = (
        "At least one validation candidate passed the configured economic filters."
        if not candidates.empty
        else "No HMM state/action candidate passed the validation economic filters under this diagnostic."
    )

    return f"""# HMM State Economics

## Scope

- Features: `{config["data"]["features_file"]}`
- HMM states: {config["hmm"]["n_states"]}
- HMM fit: train sessions only per walk-forward fold
- HMM inference: online filtered probabilities on validation/test
- Horizons: `{config.get("robustness", {}).get("horizons", [1, 2, 3]) + [6] if 6 not in config.get("robustness", {}).get("horizons", []) else config.get("robustness", {}).get("horizons", [])}`
- Costs bps: `{config.get("robustness", {}).get("cost_bps", [config["labeling"]["round_trip_cost_bps"]])}`
- Actions: `{", ".join(ACTIONS)}`

## Candidate Status Counts

{_markdown_table(status_counts)}

## Top Validation Rankings

{_markdown_table(ranking.loc[:, top_cols], max_rows=25)}

## Best Action By State

{_markdown_table(best_by_state, max_rows=40)}

## State Role Classification

{_markdown_table(state_roles, max_rows=40)}

## Hour Distribution Sample

{_markdown_table(hour_distribution.head(30))}

## Outputs

- `reports/hmm_state_economics/state_fold_metrics.parquet`
- `reports/hmm_state_economics/state_ranking.parquet`
- `reports/hmm_state_economics/state_roles.parquet`
- `reports/hmm_state_economics/state_hour_distribution.parquet`

## Conclusion

{conclusion}

This report is diagnostic only. It does not optimize on test and does not accept HMM as an edge unless validation candidates survive the explicit economic filters and are later confirmed by frozen walk-forward tests.
"""


def run(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    features = pd.read_parquet(config["data"]["features_file"])
    base_labels = pd.read_parquet(config["data"]["labels_file"])
    horizons = sorted(set(int(value) for value in config.get("robustness", {}).get("horizons", [1, 2, 3])) | {6})
    labels_by_horizon = _prepare_horizon_labels(features, config, horizons)
    folds = build_monthly_folds(base_labels, config)

    rows: list[dict[str, Any]] = []
    hour_frames: list[pd.DataFrame] = []
    for fold in folds:
        fold_rows, fold_hour_frames = run_fold_state_economics(features, labels_by_horizon, fold, config)
        rows.extend(fold_rows)
        hour_frames.extend(fold_hour_frames)

    metrics = pd.DataFrame(rows)
    hour_distribution = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    ranking = aggregate_ranking(metrics)
    state_roles = classify_state_roles(metrics)

    output_dir = Path(config["paths"]["reports"]) / "hmm_state_economics"
    report_path = Path(config["paths"]["reports"]) / "hmm_state_economics.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_dir / "state_fold_metrics.parquet", index=False)
    ranking.to_parquet(output_dir / "state_ranking.parquet", index=False)
    state_roles.to_parquet(output_dir / "state_roles.parquet", index=False)
    hour_distribution.to_parquet(output_dir / "state_hour_distribution.parquet", index=False)
    report_path.write_text(render_report(metrics, ranking, state_roles, hour_distribution, config), encoding="utf-8")
    return output_dir / "state_fold_metrics.parquet", report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pure economic behavior of HMM states.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    metrics_path, report_path = run(args.config)
    print(f"HMM state metrics written to: {metrics_path}")
    print(f"HMM state economics report written to: {report_path}")


if __name__ == "__main__":
    main()

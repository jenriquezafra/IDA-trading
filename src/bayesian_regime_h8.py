from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from src.bayesian_regime_hmm import (
    RegimeSpec,
    filtered_regime_probabilities,
    h8_regime_specs,
    h8_transition_matrix,
    tradingview_regime_specs,
    tradingview_transition_matrix,
)
from src.hmm_filter import filtered_probabilities as trained_filtered_probabilities
from src.hmm_lab import LabFold, build_lab_folds
from src.hmm_state_economics_cross_asset import build_forward_returns


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
OBS_COLUMNS = ["mom_z", "vol_z", "eff_z"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _h8_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("bayesian_regime_h8", {})


def _target_symbol(config: dict[str, Any], target_symbol: str | None = None) -> str:
    return (target_symbol or config.get("lab", {}).get("target_symbol") or _h8_cfg(config).get("target_symbol") or "SPY").upper()


def _feature_set_version(config: dict[str, Any]) -> str:
    return str(_h8_cfg(config).get("feature_set_version", config.get("hmm_lab", {}).get("feature_set_version", "cross_asset_v1")))


def features_path(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8_cfg(config)
    if cfg.get("features_file"):
        return Path(str(cfg["features_file"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    features_dir = Path(config.get("paths", {}).get("features_dir", "data/features"))
    timeframe = config.get("lab", {}).get("timeframe", config.get("project", {}).get("frequency", "15min"))
    universe_id = config.get("lab", {}).get("universe_id", "core_cross_asset_v1")
    return features_dir / target_symbol.upper() / str(timeframe) / str(universe_id) / _feature_set_version(config) / "features.parquet"


def results_dir(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8_cfg(config)
    if cfg.get("results_dir"):
        return Path(str(cfg["results_dir"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("results_dir", "results")) / target_symbol.upper() / "h8_bayesian_regime"


def report_path(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8_cfg(config)
    if cfg.get("report_file"):
        return Path(str(cfg["report_file"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "h8_bayesian_regime.md"


def models_dir(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8_cfg(config)
    if cfg.get("models_dir"):
        return Path(str(cfg["models_dir"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("models_dir", "models")) / target_symbol.upper() / "h8_bayesian_regime"


def _source_columns(config: dict[str, Any]) -> dict[str, str]:
    cfg = _h8_cfg(config)
    return {
        "mom_z": str(cfg.get("momentum_column", "target_ret_4")),
        "vol_z": str(cfg.get("volatility_column", "target_rv_12_rel_by_bar")),
        "eff_z": str(cfg.get("efficiency_column", "target_signed_efficiency_12")),
    }


def _transform_source_features(frame: pd.DataFrame, source_columns: dict[str, str]) -> pd.DataFrame:
    missing = sorted(set(source_columns.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"H8 source feature columns are missing: {missing}")
    transformed = pd.DataFrame(index=frame.index)
    transformed["mom_z"] = frame[source_columns["mom_z"]].astype(float)
    vol = frame[source_columns["vol_z"]].astype(float)
    transformed["vol_z"] = np.log(vol.clip(lower=1e-12))
    transformed["eff_z"] = frame[source_columns["eff_z"]].astype(float)
    return transformed


def prepare_h8_frame(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    source_columns = _source_columns(config)
    required = set(INDEX_COLUMNS) | set(source_columns.values())
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Features data is missing required H8 columns: {missing}")

    frame = features.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=False).rename(columns={"index": "source_index"})
    raw_obs = _transform_source_features(frame, source_columns).replace([np.inf, -np.inf], np.nan)
    valid = raw_obs.notna().all(axis=1)
    prepared = frame.loc[valid, ["source_index", *INDEX_COLUMNS]].copy()
    for column in OBS_COLUMNS:
        prepared[f"raw_{column}"] = raw_obs.loc[valid, column].to_numpy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    return prepared.reset_index(drop=True)


def build_h8_folds(frame: pd.DataFrame, config: dict[str, Any]) -> list[LabFold]:
    cfg = _h8_cfg(config)
    fold_config = {
        "hmm_lab": {
            "max_folds": cfg.get("max_folds"),
            "walk_forward": cfg.get(
                "walk_forward",
                {
                    "train_months": 24,
                    "validation_months": 6,
                    "test_months": 6,
                    "step_months": 6,
                },
            ),
        }
    }
    folds = build_lab_folds(frame, fold_config)
    if not folds:
        raise ValueError("No H8 walk-forward folds could be built")
    return folds


def _split_frames(frame: pd.DataFrame, fold: LabFold) -> dict[str, pd.DataFrame]:
    sessions = {
        "train": fold.train_sessions,
        "validation": fold.validation_sessions,
        "test": fold.test_sessions,
    }
    return {split: frame[frame["session"].isin(split_sessions)].copy() for split, split_sessions in sessions.items()}


def _lengths_by_session(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("session", sort=False).size().astype(int).tolist()


def _fit_observation_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[[f"raw_{column}" for column in OBS_COLUMNS]].to_numpy())
    return scaler


def _scaled_observations(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(frame[[f"raw_{column}" for column in OBS_COLUMNS]].to_numpy())
    return pd.DataFrame(scaled, columns=OBS_COLUMNS, index=frame.index)


def _posterior_frame(
    split_frame: pd.DataFrame,
    probabilities: np.ndarray,
    state_names_by_id: list[str],
    fold: LabFold,
    split: str,
    variant: str,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    out = split_frame[["source_index", *INDEX_COLUMNS]].copy()
    out.insert(0, "split", split)
    out.insert(0, "fold", int(fold.fold))
    out.insert(0, "variant", variant)
    for state_id, state_name in enumerate(state_names_by_id):
        out[f"p_{state_name}"] = probabilities[:, state_id]
    out["state_id"] = probabilities.argmax(axis=1).astype(int)
    out["regime"] = out["state_id"].map(dict(enumerate(state_names_by_id)))
    out["max_prob"] = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))).sum(axis=1)
    out["entropy"] = entropy / np.log(probabilities.shape[1])
    for column in OBS_COLUMNS:
        if column in observations:
            out[column] = observations[column].to_numpy()
    return out


def _run_manual_variant(
    frame: pd.DataFrame,
    fold: LabFold,
    split_frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    variant: str,
    specs: list[RegimeSpec],
    transition: np.ndarray,
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    state_names = [state.name for state in specs]
    feature_columns = list(specs[0].means)
    for split, split_frame in split_frames.items():
        if split_frame.empty:
            continue
        observations = _scaled_observations(split_frame, scaler)
        probabilities = filtered_regime_probabilities(
            observations.loc[:, feature_columns],
            specs,
            transition,
            sessions=split_frame["session"],
        )
        outputs.append(_posterior_frame(split_frame, probabilities, state_names, fold, split, variant, observations))
    return outputs


def _assign_trained_state_names(means: np.ndarray, specs: list[RegimeSpec]) -> list[str]:
    if means.shape[0] != len(specs):
        return [f"trained_state_{idx}" for idx in range(means.shape[0])]
    feature_columns = list(specs[0].means)
    prototypes = np.array([[state.means[column] for column in feature_columns] for state in specs], dtype=float)
    distances = np.linalg.norm(means[:, None, :] - prototypes[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(distances)
    assignment = {int(row): specs[int(col)].name for row, col in zip(row_ind, col_ind, strict=True)}
    return [assignment[idx] for idx in range(means.shape[0])]


def _run_trained_variant(
    fold: LabFold,
    split_frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
    config: dict[str, Any],
    target_symbol: str,
) -> tuple[list[pd.DataFrame], Path]:
    cfg = _h8_cfg(config)
    train_frame = split_frames["train"]
    train_obs = _scaled_observations(train_frame, scaler)
    model = GaussianHMM(
        n_components=int(cfg.get("trained_n_states", 4)),
        covariance_type=str(cfg.get("trained_covariance_type", "diag")),
        n_iter=int(cfg.get("trained_n_iter", 80)),
        random_state=int(cfg.get("trained_random_state", 42)),
        min_covar=float(cfg.get("trained_min_covar", 1e-4)),
    )
    model.fit(train_obs.to_numpy(), lengths=_lengths_by_session(train_frame))

    state_names = _assign_trained_state_names(model.means_, h8_regime_specs())
    outputs: list[pd.DataFrame] = []
    for split, split_frame in split_frames.items():
        if split_frame.empty:
            continue
        observations = _scaled_observations(split_frame, scaler)
        probabilities = trained_filtered_probabilities(model, observations.to_numpy(), split_frame["session"])
        outputs.append(_posterior_frame(split_frame, probabilities, state_names, fold, split, "trained_h8b", observations))

    model_path = models_dir(config, target_symbol) / f"fold_{int(fold.fold)}" / "trained_h8b.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "observation_scaler": scaler,
            "observation_columns": OBS_COLUMNS,
            "state_names_by_id": state_names,
            "fold": int(fold.fold),
            "train_sessions": fold.train_sessions,
        },
        model_path,
    )
    return outputs, model_path


def run_h8_posteriors(frame: pd.DataFrame, folds: list[LabFold], config: dict[str, Any], target_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _h8_cfg(config)
    variants = set(str(value) for value in cfg.get("variants", ["manual_tv3", "manual_h8a", "trained_h8b"]))
    posterior_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []

    for fold in folds:
        split_frames = _split_frames(frame, fold)
        train_frame = split_frames["train"]
        if train_frame.empty:
            continue
        scaler = _fit_observation_scaler(train_frame)
        if "manual_tv3" in variants:
            posterior_frames.extend(
                _run_manual_variant(
                    frame,
                    fold,
                    split_frames,
                    scaler,
                    "manual_tv3",
                    tradingview_regime_specs(),
                    tradingview_transition_matrix(),
                )
            )
        if "manual_h8a" in variants:
            posterior_frames.extend(
                _run_manual_variant(
                    frame,
                    fold,
                    split_frames,
                    scaler,
                    "manual_h8a",
                    h8_regime_specs(),
                    h8_transition_matrix(),
                )
            )
        if "trained_h8b" in variants:
            trained_frames, model_path = _run_trained_variant(fold, split_frames, scaler, config, target_symbol)
            posterior_frames.extend(trained_frames)
            model_rows.append({"variant": "trained_h8b", "fold": int(fold.fold), "model_path": str(model_path)})

    if not posterior_frames:
        raise ValueError("No H8 posterior frames were generated")
    return pd.concat(posterior_frames, ignore_index=True), pd.DataFrame(model_rows)


def _state_runs(states: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in pd.DataFrame({"state": states.astype(str), "session": sessions}).groupby("session", sort=False):
        run_state: str | None = None
        run_length = 0
        for state in group["state"]:
            if run_state is None or state != run_state:
                if run_state is not None:
                    rows.append({"regime": run_state, "duration": run_length})
                run_state = state
                run_length = 1
            else:
                run_length += 1
        if run_state is not None:
            rows.append({"regime": run_state, "duration": run_length})
    return pd.DataFrame(rows)


def summarize_regimes(posteriors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in posteriors.groupby(["variant", "fold", "split"], sort=False):
        variant, fold, split = keys
        durations = _state_runs(group["regime"], group["session"])
        duration_lookup = (
            durations.groupby("regime")["duration"].mean().to_dict() if not durations.empty else {}
        )
        for regime, state_group in group.groupby("regime", sort=False):
            row = {
                "variant": variant,
                "fold": int(fold),
                "split": split,
                "regime": regime,
                "rows": int(len(state_group)),
                "frequency": float(len(state_group) / len(group)) if len(group) else 0.0,
                "mean_duration": float(duration_lookup.get(regime, 0.0)),
                "mean_max_prob": float(state_group["max_prob"].mean()),
                "mean_entropy": float(state_group["entropy"].mean()),
                "top_hour_pct": float(pd.to_datetime(state_group["timestamp"]).dt.hour.value_counts(normalize=True).iloc[0]),
            }
            for column in OBS_COLUMNS:
                if column in state_group:
                    row[f"mean_{column}"] = float(state_group[column].mean())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "fold", "split", "regime"], kind="stable").reset_index(drop=True)


def _position_from_probabilities(frame: pd.DataFrame, threshold: float, max_entropy: float | None) -> pd.Series:
    bull = frame.get("p_bull_trend", pd.Series(0.0, index=frame.index)).fillna(0.0)
    bear = frame.get("p_bear_stress", pd.Series(0.0, index=frame.index)).fillna(0.0)
    allowed = pd.Series(True, index=frame.index)
    if max_entropy is not None:
        allowed = frame["entropy"].astype(float) <= float(max_entropy)
    position = pd.Series(0.0, index=frame.index)
    position.loc[allowed & (bull >= float(threshold)) & (bull > bear)] = 1.0
    position.loc[allowed & (bear >= float(threshold)) & (bear > bull)] = -1.0
    return position


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0].sum()
    gross_loss = -active_net[active_net < 0].sum()
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252) * daily.mean() / std)


def _max_drawdown(net: pd.Series) -> float:
    equity = net.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def evaluate_directional_gate(posteriors: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _h8_cfg(config)
    horizons = [int(value) for value in cfg.get("horizons", [1, 2, 4])]
    costs = [float(value) for value in cfg.get("cost_bps", [1.0, 2.0, 5.0])]
    thresholds = [float(value) for value in cfg.get("probability_thresholds", [0.55, 0.65, 0.75])]
    entropy_values = cfg.get("max_entropy_values", [None, 0.75])
    forward_returns = build_forward_returns(features, horizons)
    merged = posteriors.merge(
        forward_returns,
        on=["source_index", "timestamp", "session", "bar_index"],
        how="inner",
        validate="many_to_many",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(["variant", "fold", "split", "horizon_bars"], sort=False):
        variant, fold, split, horizon = keys
        for threshold in thresholds:
            for max_entropy in entropy_values:
                position = _position_from_probabilities(group, threshold=threshold, max_entropy=max_entropy)
                for cost_bps in costs:
                    gross = position * group["fwd_ret"].astype(float)
                    cost = position.abs() * (cost_bps / 10_000.0)
                    net = gross - cost
                    active = position.abs() > 0
                    active_net = net[active]
                    rows.append(
                        {
                            "variant": variant,
                            "fold": int(fold),
                            "split": split,
                            "horizon_bars": int(horizon),
                            "probability_threshold": float(threshold),
                            "max_entropy": np.nan if max_entropy is None else float(max_entropy),
                            "cost_bps": float(cost_bps),
                            "rows": int(len(group)),
                            "trades": int(active.sum()),
                            "exposure": float(active.mean()) if len(group) else 0.0,
                            "long_trades": int((position > 0).sum()),
                            "short_trades": int((position < 0).sum()),
                            "gross_return": float(gross.sum()),
                            "total_cost": float(cost.sum()),
                            "net_return": float(net.sum()),
                            "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
                            "hit_rate": float((active_net > 0).mean()) if len(active_net) else np.nan,
                            "profit_factor": _profit_factor(active_net),
                            "daily_sharpe": _daily_sharpe(group, net),
                            "max_drawdown": _max_drawdown(net),
                            "top_session_pct": float(group.loc[active, "session"].value_counts(normalize=True).iloc[0]) if active.any() else np.nan,
                        }
                    )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    sort_cols = ["split", "avg_trade_net", "profit_factor", "daily_sharpe"]
    return diagnostics.sort_values(sort_cols, ascending=[True, False, False, False], kind="stable").reset_index(drop=True)


def aggregate_gate_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    group_cols = ["variant", "split", "horizon_bars", "probability_threshold", "max_entropy", "cost_bps"]
    grouped = (
        diagnostics.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0).sum())),
            trades=("trades", "sum"),
            long_trades=("long_trades", "sum"),
            short_trades=("short_trades", "sum"),
            net_return=("net_return", "sum"),
            gross_return=("gross_return", "sum"),
            total_cost=("total_cost", "sum"),
            avg_trade_net_mean=("avg_trade_net", "mean"),
            avg_trade_net_min=("avg_trade_net", "min"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_session_pct_max=("top_session_pct", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["avg_trade_net_pooled"] = grouped["net_return"] / grouped["trades"].replace(0, np.nan)
    return grouped.sort_values(
        ["split", "avg_trade_net_pooled", "net_return"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
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
    config: dict[str, Any],
    target_symbol: str,
    folds: list[LabFold],
    profiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    aggregate_diagnostics: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _h8_cfg(config)
    validation = diagnostics[diagnostics["split"].eq("validation")].copy() if not diagnostics.empty else pd.DataFrame()
    test = diagnostics[diagnostics["split"].eq("test")].copy() if not diagnostics.empty else pd.DataFrame()
    top_cols = [
        "variant",
        "fold",
        "split",
        "horizon_bars",
        "probability_threshold",
        "max_entropy",
        "cost_bps",
        "trades",
        "long_trades",
        "short_trades",
        "net_return",
        "avg_trade_net",
        "hit_rate",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
        "top_session_pct",
    ]
    profile_cols = [
        "variant",
        "fold",
        "split",
        "regime",
        "rows",
        "frequency",
        "mean_duration",
        "mean_max_prob",
        "mean_entropy",
        "top_hour_pct",
        "mean_mom_z",
        "mean_vol_z",
        "mean_eff_z",
    ]
    aggregate_cols = [
        "variant",
        "split",
        "horizon_bars",
        "probability_threshold",
        "max_entropy",
        "cost_bps",
        "folds",
        "positive_folds",
        "trades",
        "long_trades",
        "short_trades",
        "net_return",
        "avg_trade_net_pooled",
        "avg_trade_net_mean",
        "avg_trade_net_min",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    aggregate_validation = aggregate_diagnostics[aggregate_diagnostics["split"].eq("validation")].copy() if not aggregate_diagnostics.empty else pd.DataFrame()
    aggregate_test = aggregate_diagnostics[aggregate_diagnostics["split"].eq("test")].copy() if not aggregate_diagnostics.empty else pd.DataFrame()
    outputs_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    fold_text = f"{len(folds)} (`{folds[0].train_months[0]}` to `{folds[-1].test_months[-1]}`)" if folds else "0"
    return f"""# H8 Bayesian Regime Probe - {target_symbol.upper()}

## Scope

- Feature file: `{features_path(config, target_symbol)}`
- Feature version: `{_feature_set_version(config)}`
- Source columns: `{_source_columns(config)}`
- Variants: `{cfg.get("variants", ["manual_tv3", "manual_h8a", "trained_h8b"])}`
- Walk-forward folds: {fold_text}
- Horizons: `{cfg.get("horizons", [1, 2, 4])}`
- Probability thresholds: `{cfg.get("probability_thresholds", [0.55, 0.65, 0.75])}`
- Costs bps: `{cfg.get("cost_bps", [1.0, 2.0, 5.0])}`
- Test selection used: `no`

## Regime Profiles

{_markdown_table(profiles.loc[:, [column for column in profile_cols if column in profiles.columns]], max_rows=80)}

## Top Validation Directional Gates

{_markdown_table(validation.sort_values(["avg_trade_net", "profit_factor"], ascending=[False, False]).loc[:, [column for column in top_cols if column in validation.columns]], max_rows=30)}

## Aggregate Validation Gates

{_markdown_table(aggregate_validation.sort_values(["avg_trade_net_pooled", "net_return"], ascending=[False, False]).loc[:, [column for column in aggregate_cols if column in aggregate_validation.columns]], max_rows=30)}

## Aggregate Test Gates

{_markdown_table(aggregate_test.sort_values(["avg_trade_net_pooled", "net_return"], ascending=[False, False]).loc[:, [column for column in aggregate_cols if column in aggregate_test.columns]], max_rows=30)}

## Test Sanity Rows

{_markdown_table(test.sort_values(["avg_trade_net", "profit_factor"], ascending=[False, False]).loc[:, [column for column in top_cols if column in test.columns]], max_rows=30)}

## Outputs

{outputs_text}

## Notes

- H8a/manual variants use fixed emission prototypes after train-only scaling.
- H8b trains GaussianHMM only on the train split of each fold.
- The directional gate is a first probe: long when `P(bull_trend)` clears the threshold, short when `P(bear_stress)` clears it.
- This report is not a promotion decision; it is the first pass to see whether regime probability can drive direction and position confidence.
"""


def run_config(config: dict[str, Any], target_symbol: str | None = None) -> tuple[Path, Path]:
    target = _target_symbol(config, target_symbol)
    raw_features = pd.read_parquet(features_path(config, target))
    h8_frame = prepare_h8_frame(raw_features, config)
    folds = build_h8_folds(h8_frame, config)
    posteriors, model_registry = run_h8_posteriors(h8_frame, folds, config, target)
    profiles = summarize_regimes(posteriors)
    diagnostics = evaluate_directional_gate(posteriors, raw_features, config)
    aggregate_diagnostics = aggregate_gate_diagnostics(diagnostics)

    out_dir = results_dir(config, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "posteriors": out_dir / "h8_posteriors.parquet",
        "regime_profiles": out_dir / "h8_regime_profiles.parquet",
        "directional_gate_diagnostics": out_dir / "h8_directional_gate_diagnostics.parquet",
        "directional_gate_aggregate": out_dir / "h8_directional_gate_aggregate.parquet",
        "model_registry": out_dir / "h8_model_registry.parquet",
    }
    posteriors.to_parquet(outputs["posteriors"], index=False)
    profiles.to_parquet(outputs["regime_profiles"], index=False)
    diagnostics.to_parquet(outputs["directional_gate_diagnostics"], index=False)
    aggregate_diagnostics.to_parquet(outputs["directional_gate_aggregate"], index=False)
    model_registry.to_parquet(outputs["model_registry"], index=False)
    report = report_path(config, target)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(config, target, folds, profiles, diagnostics, aggregate_diagnostics, outputs), encoding="utf-8")
    return report, outputs["directional_gate_diagnostics"]


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    return run_config(load_yaml(config_path), target_symbol=target_symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H8 Bayesian regime HMM probe.")
    parser.add_argument("--config", default="configs/hmm_bayesian_regime_h8_spy_15min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report, diagnostics = run(args.config, args.target)
    print(f"H8 report written to: {report}")
    print(f"H8 diagnostics written to: {diagnostics}")


if __name__ == "__main__":
    main()

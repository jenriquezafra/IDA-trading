from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import bayesian_regime_h8
from src.bayesian_regime_h8_allocation import cost_scenarios as allocation_cost_scenarios
from src.bayesian_regime_h8_allocation import evaluate_allocation_frame
from src.candidate_cost_sensitivity_cross_asset import cost_scenarios, scenario_cost_return
from src.hmm_lab import LabFold
from src.hmm_state_economics_cross_asset import build_forward_returns
from src.setup_signal_search import SIGNAL_COLUMNS, signal_mask


INDEX_COLUMNS = ["timestamp", "session", "bar_index"]
MERGE_KEYS = ["source_index", "timestamp", "session", "bar_index"]

DEFAULT_MODEL_FEATURE_COLUMNS = [
    "target_ret_1",
    "target_ret_2",
    "target_ret_4",
    "target_ret_8",
    "target_signed_efficiency_4",
    "target_signed_efficiency_12",
    "target_dir_persistence_4",
    "target_dir_persistence_12",
    "target_dist_vwap_atr",
    "target_dist_open",
    "target_pos_session_range",
    "target_intraday_runup",
    "target_intraday_drawdown",
    "target_rel_volume_by_bar",
    "target_rel_cum_volume_by_bar",
    "target_rv_12_rel_by_bar",
    "target_absret_12_to_rv_12",
    "target_breakout_margin_roll_high_8_atr",
    "target_breakdown_margin_roll_low_8_atr",
    "target_breakout_margin_or_2_high_atr",
    "target_breakdown_margin_or_2_low_atr",
    "target_close_location_bar",
    "target_clv",
    "positive_index_count_6",
    "positive_sector_count_6",
    "sector_above_vwap_count",
    "sector_rel_strength_count_6",
    "leadership_concentration_score_6",
    "spread_growth_defensive_12",
    "spread_tech_broad_12",
    "spread_credit_12",
    "relret_QQQ_SPY_6",
    "relret_IWM_SPY_6",
    "relret_HYG_LQD_12",
    "risk_on_score",
    "risk_off_score",
    "defensive_rotation_score",
    "narrow_rally_score",
    "cross_asset_vol_expansion_score",
    "intraday_stress_score",
    "chop_score",
]


@dataclass(frozen=True)
class BinarySideModel:
    side: str
    feature_columns: list[str]
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant_probability: float | None


@dataclass(frozen=True)
class NetSideModel:
    side: str
    feature_columns: list[str]
    scaler: StandardScaler | None
    model: Ridge | None
    constant_prediction: float


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _h8c_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("h8_position_model", {})


def _lifecycle_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("h8_position_lifecycle", {})


def _target_symbol(config: dict[str, Any], target_symbol: str | None = None) -> str:
    return (target_symbol or config.get("lab", {}).get("target_symbol") or _h8c_cfg(config).get("target_symbol") or "QQQ").upper()


def _selected_variant(config: dict[str, Any]) -> str:
    return str(_h8c_cfg(config).get("selected_h8_variant", "manual_h8a"))


def results_dir(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8c_cfg(config)
    if cfg.get("results_dir"):
        return Path(str(cfg["results_dir"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("results_dir", "results")) / target_symbol.upper() / "h8_position_model"


def report_path(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8c_cfg(config)
    if cfg.get("report_file"):
        return Path(str(cfg["report_file"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("reports_dir", "reports")) / target_symbol.upper() / "h8_position_model.md"


def models_dir(config: dict[str, Any], target_symbol: str) -> Path:
    cfg = _h8c_cfg(config)
    if cfg.get("models_dir"):
        return Path(str(cfg["models_dir"]).format(target_symbol=target_symbol.upper(), target=target_symbol.upper()))
    return Path(config.get("paths", {}).get("models_dir", "models")) / target_symbol.upper() / "h8_position_model"


def _configured_model_features(config: dict[str, Any]) -> list[str]:
    return [str(column) for column in _h8c_cfg(config).get("model_feature_columns", DEFAULT_MODEL_FEATURE_COLUMNS)]


def _posterior_feature_columns(posteriors: pd.DataFrame, selected_variant: str) -> list[str]:
    selected = posteriors[posteriors["variant"].eq(selected_variant)]
    probability_columns = [column for column in selected.columns if column.startswith("p_")]
    meta_columns = [column for column in ["max_prob", "entropy", "mom_z", "vol_z", "eff_z"] if column in selected.columns]
    return probability_columns + meta_columns


def _finite_feature_mask(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    features = frame.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    return features.notna().all(axis=1)


def _consecutive_age(values: pd.Series) -> pd.Series:
    groups = values.ne(values.shift()).cumsum()
    return values.groupby(groups, sort=False).cumcount().astype(float) + 1.0


def _rolling_flip_count(values: pd.Series, window: int = 8) -> pd.Series:
    changed = values.ne(values.shift()).astype(float)
    if len(changed):
        changed.iloc[0] = 0.0
    return changed.rolling(window, min_periods=1).sum().astype(float)


def _add_regime_dynamics_features(frame: pd.DataFrame, probability_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    prob_cols = [column for column in probability_columns if column in frame.columns and column.startswith("p_")]
    if not prob_cols:
        return frame, []

    output = frame.copy()
    probs = output.loc[:, prob_cols].astype(float)
    sorted_probs = np.sort(probs.to_numpy(), axis=1)
    output["posterior_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2] if len(prob_cols) > 1 else sorted_probs[:, -1]
    output["dominant_state"] = probs.idxmax(axis=1)
    output["dominant_is_bull"] = output["dominant_state"].eq("p_bull_trend").astype(float)
    output["dominant_is_bear"] = output["dominant_state"].eq("p_bear_stress").astype(float)
    output["dominant_is_chop"] = output["dominant_state"].eq("p_chop_noise").astype(float)
    if {"p_bull_trend", "p_bear_stress"}.issubset(output.columns):
        output["bull_bear_edge"] = output["p_bull_trend"].astype(float) - output["p_bear_stress"].astype(float)
    if {"p_bull_trend", "p_chop_noise"}.issubset(output.columns):
        output["bull_chop_edge"] = output["p_bull_trend"].astype(float) - output["p_chop_noise"].astype(float)
    if {"p_bear_stress", "p_chop_noise"}.issubset(output.columns):
        output["bear_chop_edge"] = output["p_bear_stress"].astype(float) - output["p_chop_noise"].astype(float)

    group_cols = ["fold", "split", "session"]
    ordered = output.sort_values([*group_cols, "bar_index"], kind="stable").copy()
    dynamic_base = [*prob_cols, "max_prob", "entropy", "posterior_margin", "bull_bear_edge"]
    dynamic_base = [column for column in dynamic_base if column in ordered.columns]
    for column in dynamic_base:
        grouped = ordered.groupby(group_cols, sort=False)[column]
        ordered[f"{column}_delta_1"] = grouped.diff(1).fillna(0.0)
        ordered[f"{column}_delta_3"] = grouped.diff(3).fillna(0.0)

    ordered["dominant_state_age_bars"] = (
        ordered.groupby(group_cols, sort=False)["dominant_state"].transform(_consecutive_age).astype(float)
    )
    changed = ordered.groupby(group_cols, sort=False)["dominant_state"].transform(lambda values: values.ne(values.shift()).astype(float))
    first_in_group = ordered.groupby(group_cols, sort=False).cumcount().eq(0)
    ordered["dominant_state_changed"] = changed.mask(first_in_group, 0.0).astype(float)
    ordered["regime_flip_count_8"] = ordered.groupby(group_cols, sort=False)["dominant_state"].transform(_rolling_flip_count).astype(float)

    feature_columns = [
        "posterior_margin",
        "dominant_is_bull",
        "dominant_is_bear",
        "dominant_is_chop",
        "bull_bear_edge",
        "bull_chop_edge",
        "bear_chop_edge",
        *[f"{column}_delta_1" for column in dynamic_base],
        *[f"{column}_delta_3" for column in dynamic_base],
        "dominant_state_age_bars",
        "dominant_state_changed",
        "regime_flip_count_8",
    ]
    feature_columns = [column for column in feature_columns if column in ordered.columns]
    output = ordered.sort_index(kind="stable")
    output.loc[:, feature_columns] = output.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output, feature_columns


def _scenario_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    scenarios = {str(scenario["cost_scenario"]): scenario for scenario in _cost_scenarios(config)}
    if name not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise ValueError(f"Configured label_cost_scenario is not available: {name}. Available scenarios: {available}")
    return scenarios[name]


def _label_cost_returns(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, str, float]:
    cfg = _h8c_cfg(config)
    scenario_name = cfg.get("label_cost_scenario")
    if scenario_name:
        scenario = _scenario_by_name(config, str(scenario_name))
        cost_frame = frame.copy()
        cost_frame["target_open_next"] = cost_frame["entry_px"].astype(float)
        full_position = pd.Series(1.0, index=cost_frame.index, dtype=float)
        cost = scenario_cost_return(cost_frame, full_position, scenario).astype(float)
        configured_bps = float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan
        return cost, str(scenario["cost_scenario"]), configured_bps

    label_cost_bps = float(cfg.get("label_cost_bps", cfg.get("primary_cost_bps", 1.0)))
    cost = pd.Series(label_cost_bps / 10_000.0, index=frame.index, dtype=float)
    return cost, f"bps_{label_cost_bps:g}", label_cost_bps


def prepare_supervised_frame(
    features: pd.DataFrame,
    posteriors: pd.DataFrame,
    config: dict[str, Any],
    *,
    selected_variant: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = _h8c_cfg(config)
    variant = selected_variant or _selected_variant(config)
    horizon = int(cfg.get("horizon_bars", 4))

    forward_returns = build_forward_returns(features, [horizon])
    posterior = posteriors[posteriors["variant"].eq(variant)].copy()
    if posterior.empty:
        raise ValueError(f"No H8 posterior rows found for variant: {variant}")

    posterior_feature_columns = _posterior_feature_columns(posteriors, variant)
    missing_posterior = sorted(set(posterior_feature_columns) - set(posterior.columns))
    if missing_posterior:
        raise ValueError(f"H8 posterior columns are missing: {missing_posterior}")

    raw_features = features.sort_values(["session", "bar_index"], kind="stable").reset_index(names="source_index")
    configured_features = _configured_model_features(config)
    available_features = [column for column in configured_features if column in raw_features.columns]
    missing_features = [column for column in configured_features if column not in raw_features.columns]
    if missing_features and bool(cfg.get("strict_feature_columns", False)):
        raise ValueError(f"H8c model feature columns are missing: {missing_features}")
    if not available_features:
        raise ValueError("No configured H8c model feature columns are available")

    raw_slice = raw_features.loc[:, ["source_index", *available_features]].copy()
    merged = posterior.merge(
        forward_returns,
        on=MERGE_KEYS,
        how="inner",
        validate="many_to_one",
    )
    merged = merged.merge(raw_slice, on="source_index", how="left", validate="many_to_one")
    regime_feature_columns: list[str] = []
    if bool(cfg.get("include_regime_dynamics_features", True)):
        merged, regime_feature_columns = _add_regime_dynamics_features(merged, posterior_feature_columns)
    model_columns = list(dict.fromkeys([*posterior_feature_columns, *regime_feature_columns, *available_features]))

    finite = _finite_feature_mask(merged, model_columns)
    output = merged.loc[finite].copy().reset_index(drop=True)
    label_cost, label_cost_scenario, label_cost_bps = _label_cost_returns(output, config)
    output["horizon_bars"] = int(horizon)
    output["label_cost_bps"] = float(label_cost_bps)
    output["label_cost_scenario"] = label_cost_scenario
    output["label_cost_return"] = label_cost.to_numpy()
    output["label_cost_effective_bps"] = label_cost.astype(float).to_numpy() * 10_000.0
    output["long_net_return"] = output["fwd_ret"].astype(float) - output["label_cost_return"].astype(float)
    output["short_net_return"] = -output["fwd_ret"].astype(float) - output["label_cost_return"].astype(float)
    output["label_long_profit"] = (output["long_net_return"] > 0.0).astype(int)
    output["label_short_profit"] = (output["short_net_return"] > 0.0).astype(int)
    output["timestamp"] = pd.to_datetime(output["timestamp"])
    return output, model_columns


def _fit_side_model(train: pd.DataFrame, side: str, feature_columns: list[str], config: dict[str, Any]) -> BinarySideModel:
    cfg = _h8c_cfg(config)
    label_col = f"label_{side}_profit"
    y = train[label_col].astype(int).to_numpy()
    positive_rate = float(y.mean()) if len(y) else 0.0
    if len(np.unique(y)) < 2:
        return BinarySideModel(side=side, feature_columns=feature_columns, scaler=None, model=None, constant_probability=positive_rate)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.loc[:, feature_columns].to_numpy())
    model = LogisticRegression(
        solver=str(cfg.get("solver", "lbfgs")),
        C=float(cfg.get("C", 0.5)),
        max_iter=int(cfg.get("max_iter", 2000)),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=int(cfg.get("random_state", 42)),
    )
    model.fit(x_train, y)
    return BinarySideModel(side=side, feature_columns=feature_columns, scaler=scaler, model=model, constant_probability=None)


def _fit_side_net_model(train: pd.DataFrame, side: str, feature_columns: list[str], config: dict[str, Any]) -> NetSideModel:
    cfg = _h8c_cfg(config)
    target_col = f"{side}_net_return"
    y = train[target_col].astype(float).to_numpy()
    y = np.where(np.isfinite(y), y, 0.0)
    clip_bps = cfg.get("net_target_clip_bps", 75.0)
    if clip_bps is not None:
        clip = float(clip_bps) / 10_000.0
        y = np.clip(y, -clip, clip)
    constant = float(np.mean(y)) if len(y) else 0.0
    if len(y) < 2 or float(np.std(y)) < 1e-12:
        return NetSideModel(side=side, feature_columns=feature_columns, scaler=None, model=None, constant_prediction=constant)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.loc[:, feature_columns].to_numpy())
    model = Ridge(alpha=float(cfg.get("net_ridge_alpha", 10.0)))
    model.fit(x_train, y)
    return NetSideModel(side=side, feature_columns=feature_columns, scaler=scaler, model=model, constant_prediction=constant)


def _predict_side_probability(model: BinarySideModel, frame: pd.DataFrame) -> np.ndarray:
    if model.model is None or model.scaler is None:
        return np.full(len(frame), float(model.constant_probability or 0.0), dtype=float)
    x = model.scaler.transform(frame.loc[:, model.feature_columns].to_numpy())
    classes = list(model.model.classes_)
    probabilities = model.model.predict_proba(x)
    if 1 not in classes:
        return np.zeros(len(frame), dtype=float)
    return probabilities[:, classes.index(1)]


def _predict_side_net(model: NetSideModel, frame: pd.DataFrame) -> np.ndarray:
    if model.model is None or model.scaler is None:
        return np.full(len(frame), float(model.constant_prediction), dtype=float)
    x = model.scaler.transform(frame.loc[:, model.feature_columns].to_numpy())
    return model.model.predict(x).astype(float)


def _safe_auc(y_true: pd.Series, probability: pd.Series) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true.astype(int), probability.astype(float)))


def _model_quality_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["fold", "split"], sort=False):
        fold, split = keys
        for side in ["long", "short"]:
            y = group[f"label_{side}_profit"].astype(int)
            p = group[f"p_{side}_profit"].astype(float)
            actual_net = group[f"{side}_net_return"].astype(float)
            expected_net = group.get(f"expected_{side}_net", pd.Series(np.nan, index=group.index)).astype(float)
            error_bps = (expected_net - actual_net) * 10_000.0
            valid_error = error_bps.replace([np.inf, -np.inf], np.nan).dropna()
            if expected_net.notna().sum() > 1 and actual_net.nunique() > 1:
                net_corr = float(expected_net.corr(actual_net))
            else:
                net_corr = np.nan
            rows.append(
                {
                    "fold": int(fold),
                    "split": split,
                    "side": side,
                    "rows": int(len(group)),
                    "positive_rate": float(y.mean()) if len(y) else np.nan,
                    "avg_probability": float(p.mean()) if len(p) else np.nan,
                    "auc": _safe_auc(y, p),
                    "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
                    "actual_net_mean_bps": float(actual_net.mean() * 10_000.0) if len(actual_net) else np.nan,
                    "expected_net_mean_bps": float(expected_net.mean() * 10_000.0) if len(expected_net) else np.nan,
                    "net_mae_bps": float(valid_error.abs().mean()) if len(valid_error) else np.nan,
                    "net_rmse_bps": float(np.sqrt((valid_error**2).mean())) if len(valid_error) else np.nan,
                    "net_corr": net_corr,
                }
            )
    return pd.DataFrame(rows)


def _profit_factor(active_net: pd.Series) -> float:
    if active_net.empty:
        return np.nan
    gross_profit = active_net[active_net > 0.0].sum()
    gross_loss = -active_net[active_net < 0.0].sum()
    if gross_loss == 0.0:
        return np.inf if gross_profit > 0.0 else np.nan
    return float(gross_profit / gross_loss)


def _daily_sharpe(frame: pd.DataFrame, net: pd.Series) -> float:
    daily = net.groupby(frame["session"]).sum()
    if len(daily) < 2:
        return np.nan
    std = daily.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(252.0) * daily.mean() / std)


def _max_drawdown(net: pd.Series) -> float:
    if net.empty:
        return 0.0
    equity = net.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def _position_from_probabilities(
    frame: pd.DataFrame,
    threshold: float,
    min_probability_gap: float,
    *,
    gate_mode: str = "position_only",
    regime_threshold: float | None = None,
    max_entropy: float | None = None,
    min_expected_net_bps: float | None = None,
    min_expected_net_gap_bps: float | None = None,
) -> pd.Series:
    long_probability = frame["p_long_profit"].astype(float)
    short_probability = frame["p_short_profit"].astype(float)
    position = pd.Series(0.0, index=frame.index)
    long_mask = (long_probability >= float(threshold)) & ((long_probability - short_probability) >= float(min_probability_gap))
    short_mask = (short_probability >= float(threshold)) & ((short_probability - long_probability) >= float(min_probability_gap))
    if min_expected_net_bps is not None:
        if "expected_long_net" not in frame.columns or "expected_short_net" not in frame.columns:
            raise ValueError("Expected-net gate requires expected_long_net and expected_short_net columns")
        expected_long_bps = frame["expected_long_net"].astype(float) * 10_000.0
        expected_short_bps = frame["expected_short_net"].astype(float) * 10_000.0
        expected_gap = 0.0 if min_expected_net_gap_bps is None else float(min_expected_net_gap_bps)
        long_mask &= (expected_long_bps >= float(min_expected_net_bps)) & ((expected_long_bps - expected_short_bps) >= expected_gap)
        short_mask &= (expected_short_bps >= float(min_expected_net_bps)) & ((expected_short_bps - expected_long_bps) >= expected_gap)
    if gate_mode == "regime_confirmed":
        if "p_bull_trend" not in frame.columns or "p_bear_stress" not in frame.columns:
            raise ValueError("regime_confirmed gate requires p_bull_trend and p_bear_stress columns")
        floor = float(0.0 if regime_threshold is None else regime_threshold)
        bull = frame["p_bull_trend"].astype(float)
        bear = frame["p_bear_stress"].astype(float)
        long_mask &= (bull >= floor) & (bull > bear)
        short_mask &= (bear >= floor) & (bear > bull)
        if max_entropy is not None:
            long_mask &= frame["entropy"].astype(float) <= float(max_entropy)
            short_mask &= frame["entropy"].astype(float) <= float(max_entropy)
    elif gate_mode != "position_only":
        raise ValueError(f"Unsupported H8c gate_mode: {gate_mode}")
    position.loc[long_mask] = 1.0
    position.loc[short_mask] = -1.0
    return position


def _expected_net_thresholds(cfg: dict[str, Any]) -> list[float | None]:
    if not bool(cfg.get("use_expected_net_gate", False)):
        return [None]
    return [None if value is None else float(value) for value in cfg.get("expected_net_threshold_bps", [0.0])]


def _expected_net_gaps(cfg: dict[str, Any]) -> list[float | None]:
    if not bool(cfg.get("use_expected_net_gate", False)):
        return [None]
    return [None if value is None else float(value) for value in cfg.get("min_expected_net_gaps_bps", [0.0])]


def evaluate_probability_gate(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _h8c_cfg(config)
    gate_modes = [str(value) for value in cfg.get("gate_modes", ["position_only", "regime_confirmed"])]
    thresholds = [float(value) for value in cfg.get("probability_thresholds", [0.50, 0.55, 0.60, 0.65, 0.70])]
    gaps = [float(value) for value in cfg.get("min_probability_gaps", [0.0, 0.05, 0.10])]
    regime_thresholds = [float(value) for value in cfg.get("regime_thresholds", [0.65, 0.75])]
    entropy_values = cfg.get("max_entropy_values", [None])
    expected_thresholds = _expected_net_thresholds(cfg)
    expected_gaps = _expected_net_gaps(cfg)
    costs = [float(value) for value in cfg.get("cost_bps", [1.0, 2.0, 5.0])]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["fold", "split"], sort=False):
        fold, split = keys
        for gate_mode in gate_modes:
            mode_regime_thresholds = [None] if gate_mode == "position_only" else regime_thresholds
            mode_entropy_values = [None] if gate_mode == "position_only" else entropy_values
            for threshold in thresholds:
                for gap in gaps:
                    for regime_threshold in mode_regime_thresholds:
                        for max_entropy in mode_entropy_values:
                            for expected_threshold in expected_thresholds:
                                expected_gap_values = [None] if expected_threshold is None else expected_gaps
                                for expected_gap in expected_gap_values:
                                    position = _position_from_probabilities(
                                        group,
                                        threshold,
                                        gap,
                                        gate_mode=gate_mode,
                                        regime_threshold=regime_threshold,
                                        max_entropy=max_entropy,
                                        min_expected_net_bps=expected_threshold,
                                        min_expected_net_gap_bps=expected_gap,
                                    )
                                    for cost_bps in costs:
                                        gross = position * group["fwd_ret"].astype(float)
                                        cost = position.abs() * (cost_bps / 10_000.0)
                                        net = gross - cost
                                        active = position.abs() > 0.0
                                        active_net = net[active]
                                        rows.append(
                                            {
                                                "fold": int(fold),
                                                "split": split,
                                                "gate_mode": gate_mode,
                                                "horizon_bars": int(group["horizon_bars"].iloc[0]) if len(group) else int(cfg.get("horizon_bars", 4)),
                                                "threshold": float(threshold),
                                                "min_probability_gap": float(gap),
                                                "expected_net_threshold_bps": np.nan if expected_threshold is None else float(expected_threshold),
                                                "min_expected_net_gap_bps": np.nan if expected_gap is None else float(expected_gap),
                                                "regime_threshold": np.nan if regime_threshold is None else float(regime_threshold),
                                                "max_entropy": np.nan if max_entropy is None else float(max_entropy),
                                                "cost_bps": float(cost_bps),
                                                "rows": int(len(group)),
                                                "trades": int(active.sum()),
                                                "exposure": float(active.mean()) if len(group) else 0.0,
                                                "long_trades": int((position > 0.0).sum()),
                                                "short_trades": int((position < 0.0).sum()),
                                                "gross_return": float(gross.sum()),
                                                "total_cost": float(cost.sum()),
                                                "net_return": float(net.sum()),
                                                "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
                                                "hit_rate": float((active_net > 0.0).mean()) if len(active_net) else np.nan,
                                                "profit_factor": _profit_factor(active_net),
                                                "daily_sharpe": _daily_sharpe(group, net),
                                                "max_drawdown": _max_drawdown(net),
                                                "top_session_pct": float(group.loc[active, "session"].value_counts(normalize=True).iloc[0]) if active.any() else np.nan,
                                            }
                                )
    return pd.DataFrame(rows)


def _parameter_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "gate_mode",
            "threshold",
            "min_probability_gap",
            "expected_net_threshold_bps",
            "min_expected_net_gap_bps",
            "regime_threshold",
            "max_entropy",
            "cost_bps",
        ]
        if column in frame.columns
    ]


def select_validation_gates(grid: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = _h8c_cfg(config)
    primary_cost = float(cfg.get("primary_cost_bps", cfg.get("label_cost_bps", 1.0)))
    min_validation_trades = int(cfg.get("min_validation_trades", 30))
    selection_scope = str(cfg.get("selection_scope", "global"))
    parameter_columns = _parameter_columns(grid)
    if selection_scope == "global":
        validation = grid[(grid["split"].eq("validation")) & (grid["cost_bps"].eq(primary_cost))].copy()
        grouped = (
            validation.groupby(parameter_columns, as_index=False, dropna=False)
            .agg(
                folds=("fold", "nunique"),
                positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
                rows=("rows", "sum"),
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
        eligible = grouped[grouped["trades"].ge(min_validation_trades)].copy()
        if eligible.empty:
            eligible = grouped.copy()
        selected = eligible.sort_values(
            ["avg_trade_net_pooled", "net_return", "daily_sharpe_mean", "trades"],
            ascending=[False, False, False, False],
            kind="stable",
        ).iloc[[0]].copy()
        selected.insert(0, "fold", -1)
        selected.insert(1, "split", "validation")
        selected["selection_scope"] = "global"
        selected["selection_rank_metric"] = "avg_trade_net_pooled"
        return selected.reset_index(drop=True)

    if selection_scope != "per_fold":
        raise ValueError(f"Unsupported H8c selection_scope: {selection_scope}")

    rows: list[pd.Series] = []
    validation = grid[(grid["split"].eq("validation")) & (grid["cost_bps"].eq(primary_cost))].copy()
    for fold, group in validation.groupby("fold", sort=False):
        eligible = group[group["trades"].ge(min_validation_trades)].copy()
        if eligible.empty:
            eligible = group.copy()
        eligible = eligible.sort_values(
            ["avg_trade_net", "net_return", "daily_sharpe", "trades"],
            ascending=[False, False, False, False],
            kind="stable",
        )
        selected = eligible.iloc[0].copy()
        selected["selection_split"] = "validation"
        selected["selection_scope"] = "per_fold"
        selected["selection_rank_metric"] = "avg_trade_net"
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _equal_or_both_nan(series: pd.Series, value: Any) -> pd.Series:
    if pd.isna(value):
        return series.isna()
    return series.eq(value)


def selected_gate_metrics(grid: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    parameter_columns = _parameter_columns(grid)
    rows: list[pd.Series] = []
    for _, row in selected.iterrows():
        mask = grid["split"].isin(["validation", "test"])
        if int(row.get("fold", -1)) >= 0:
            mask &= grid["fold"].eq(int(row["fold"]))
        for column in parameter_columns:
            mask &= _equal_or_both_nan(grid[column], row[column])
        rows.extend([candidate.copy() for _, candidate in grid.loc[mask].iterrows()])
    output = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    if not output.empty:
        output["selected_on"] = "validation"
    return output


def aggregate_selected_metrics(selected_metrics: pd.DataFrame) -> pd.DataFrame:
    if selected_metrics.empty:
        return selected_metrics
    grouped = (
        selected_metrics.groupby(["split", "cost_bps"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
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
    return grouped.sort_values(["split", "cost_bps"], kind="stable").reset_index(drop=True)


def _scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _h8c_cfg(config)
    if "candidate_cost_sensitivity_cross_asset" in config:
        return config
    sensitivity_cfg = cfg.get(
        "cost_sensitivity",
        {
            "cost_bps": cfg.get("cost_bps", [1.0, 2.0, 5.0]),
            "ibkr": {"enabled": False},
        },
    )
    return {**config, "candidate_cost_sensitivity_cross_asset": sensitivity_cfg}


def _cost_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    return cost_scenarios(_scenario_config(config))


def _selected_row_float(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _selected_position(frame: pd.DataFrame, selected_row: pd.Series) -> pd.Series:
    return _position_from_probabilities(
        frame,
        float(selected_row["threshold"]),
        float(selected_row["min_probability_gap"]),
        gate_mode=str(selected_row.get("gate_mode", "position_only")),
        regime_threshold=_selected_row_float(selected_row, "regime_threshold"),
        max_entropy=_selected_row_float(selected_row, "max_entropy"),
        min_expected_net_bps=_selected_row_float(selected_row, "expected_net_threshold_bps"),
        min_expected_net_gap_bps=_selected_row_float(selected_row, "min_expected_net_gap_bps"),
    )


def _param_value(params: pd.Series | dict[str, Any], column: str, default: Any = None) -> Any:
    if isinstance(params, pd.Series):
        return params.get(column, default)
    return params.get(column, default)


def _param_float(params: pd.Series | dict[str, Any], column: str, default: float | None = None) -> float | None:
    value = _param_value(params, column, default)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _param_bool(params: pd.Series | dict[str, Any], column: str, default: bool = False) -> bool:
    value = _param_value(params, column, default)
    if value is None or pd.isna(value):
        return bool(default)
    return bool(value)


def _lifecycle_signal_from_params(frame: pd.DataFrame, params: pd.Series | dict[str, Any], prefix: str) -> pd.Series:
    gate_mode = str(_param_value(params, f"{prefix}_gate_mode", _param_value(params, "gate_mode", "position_only")))
    threshold = _param_float(params, f"{prefix}_threshold", _param_float(params, "threshold", 0.50))
    gap = _param_float(params, f"{prefix}_min_probability_gap", _param_float(params, "min_probability_gap", 0.0))
    if threshold is None or gap is None:
        raise ValueError(f"Lifecycle {prefix} signal requires threshold and min_probability_gap")
    return _position_from_probabilities(
        frame,
        threshold,
        gap,
        gate_mode=gate_mode,
        regime_threshold=_param_float(params, f"{prefix}_regime_threshold", _param_float(params, "regime_threshold")),
        max_entropy=_param_float(params, f"{prefix}_max_entropy", _param_float(params, "max_entropy")),
        min_expected_net_bps=_param_float(params, f"{prefix}_expected_net_threshold_bps", _param_float(params, "expected_net_threshold_bps")),
        min_expected_net_gap_bps=_param_float(params, f"{prefix}_min_expected_net_gap_bps", _param_float(params, "min_expected_net_gap_bps")),
    )


def _json_dumps(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _lifecycle_setup_specs(lifecycle: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(lifecycle.get("setup_entry_mode", False)):
        return [{}]
    specs: list[dict[str, Any]] = []
    for index, raw in enumerate(lifecycle.get("setup_specs", [])):
        if not isinstance(raw, dict):
            raise ValueError("Each H8e setup spec must be a mapping")
        params = dict(raw.get("params", {}))
        direction = str(raw.get("direction", params.get("direction", "both"))).lower()
        if direction not in {"long", "short", "both"}:
            raise ValueError(f"Unsupported H8e setup direction: {direction}")
        if direction != "both":
            params["direction"] = direction
        specs.append(
            {
                "setup_entry_mode": True,
                "setup_name": str(raw.get("name", f"setup_{index}")),
                "setup_family": str(raw["family"]),
                "setup_direction": direction,
                "setup_params_json": _json_dumps(params),
                "setup_column_map_json": _json_dumps(dict(raw.get("column_map", {}))),
            }
        )
    if not specs:
        raise ValueError("H8e setup_entry_mode requires at least one setup_specs item")
    return specs


def _setup_signal_mask(frame: pd.DataFrame, params: pd.Series | dict[str, Any]) -> pd.Series:
    if not _param_bool(params, "setup_entry_mode", False):
        return pd.Series(True, index=frame.index, dtype=bool)
    family = str(_param_value(params, "setup_family"))
    setup_params = _json_loads(_param_value(params, "setup_params_json"))
    column_map = _json_loads(_param_value(params, "setup_column_map_json"))
    return signal_mask(frame, family, setup_params, column_map)


def _apply_setup_to_entry_signal(entry_signal: pd.Series, setup: pd.Series, params: pd.Series | dict[str, Any]) -> pd.Series:
    if not _param_bool(params, "setup_entry_mode", False):
        return entry_signal
    direction = str(_param_value(params, "setup_direction", "both")).lower()
    output = pd.Series(0.0, index=entry_signal.index, dtype=float)
    if direction == "long":
        output.loc[setup & entry_signal.gt(0.0)] = 1.0
    elif direction == "short":
        output.loc[setup & entry_signal.lt(0.0)] = -1.0
    elif direction == "both":
        output.loc[setup] = entry_signal.loc[setup].astype(float)
    else:
        raise ValueError(f"Unsupported H8e setup direction: {direction}")
    return output


def _scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    ibkr = scenario.get("ibkr", {})
    return {
        "cost_scenario": str(scenario["cost_scenario"]),
        "cost_kind": str(scenario["cost_kind"]),
        "configured_cost_bps": float(scenario["cost_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
        "ibkr_plan": str(scenario.get("ibkr_plan", "")),
        "notional_usd": float(scenario.get("notional_usd", np.nan)),
        "spread_slippage_bps_round_trip": float(ibkr.get("spread_slippage_bps_round_trip", np.nan)),
    }


def _evaluate_position_cost(frame: pd.DataFrame, position: pd.Series, scenario: dict[str, Any]) -> dict[str, float | int]:
    eval_frame = frame.copy()
    if "target_open_next" not in eval_frame.columns:
        eval_frame["target_open_next"] = eval_frame["entry_px"] if "entry_px" in eval_frame.columns else np.nan
    eval_frame["timestamp"] = pd.to_datetime(eval_frame["timestamp"])

    position = position.astype(float).fillna(0.0)
    active = position.abs() > 0.0
    gross = position * eval_frame["fwd_ret"].astype(float)
    cost = scenario_cost_return(eval_frame, position, scenario).astype(float)
    net = gross - cost
    active_net = net[active]
    abs_position_sum = float(position.abs().sum())
    effective_cost_bps = float(cost.sum() / abs_position_sum * 10_000.0) if abs_position_sum > 0.0 else np.nan
    return {
        "rows": int(len(eval_frame)),
        "trades": int(active.sum()),
        "exposure": float(active.mean()) if len(eval_frame) else 0.0,
        "long_trades": int((position > 0.0).sum()),
        "short_trades": int((position < 0.0).sum()),
        "gross_return": float(gross.sum()),
        "total_cost": float(cost.sum()),
        "effective_cost_bps": effective_cost_bps,
        "net_return": float(net.sum()),
        "avg_trade_net": float(active_net.mean()) if len(active_net) else 0.0,
        "hit_rate": float((active_net > 0.0).mean()) if len(active_net) else np.nan,
        "profit_factor": _profit_factor(active_net),
        "daily_sharpe": _daily_sharpe(eval_frame, net),
        "max_drawdown": _max_drawdown(net),
        "top_session_pct": float(eval_frame.loc[active, "session"].value_counts(normalize=True).iloc[0]) if active.any() else np.nan,
    }


def selected_cost_sensitivity(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty or selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    scenarios = _cost_scenarios(config)
    rows: list[dict[str, Any]] = []
    for _, selected_row in selected.iterrows():
        mask = predictions["split"].isin(["validation", "test"])
        selected_fold = int(selected_row.get("fold", -1))
        if selected_fold >= 0:
            mask &= predictions["fold"].eq(selected_fold)
        selected_predictions = predictions.loc[mask].copy()
        for keys, group in selected_predictions.groupby(["fold", "split"], sort=False):
            fold, split = keys
            position = _selected_position(group, selected_row)
            for scenario in scenarios:
                rows.append(
                    {
                        "selected_fold": selected_fold,
                        "fold": int(fold),
                        "split": str(split),
                        "gate_mode": str(selected_row.get("gate_mode", "position_only")),
                        "threshold": float(selected_row["threshold"]),
                        "min_probability_gap": float(selected_row["min_probability_gap"]),
                        "regime_threshold": _selected_row_float(selected_row, "regime_threshold"),
                        "max_entropy": _selected_row_float(selected_row, "max_entropy"),
                        **_scenario_metadata(scenario),
                        **_evaluate_position_cost(group, position, scenario),
                    }
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    group_columns = [
        "split",
        "cost_scenario",
        "cost_kind",
        "configured_cost_bps",
        "ibkr_plan",
        "notional_usd",
        "spread_slippage_bps_round_trip",
    ]
    aggregate = (
        detail.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            trades=("trades", "sum"),
            long_trades=("long_trades", "sum"),
            short_trades=("short_trades", "sum"),
            gross_return=("gross_return", "sum"),
            total_cost=("total_cost", "sum"),
            net_return=("net_return", "sum"),
            avg_trade_net_mean=("avg_trade_net", "mean"),
            avg_trade_net_min=("avg_trade_net", "min"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_session_pct_max=("top_session_pct", "max"),
        )
        .reset_index(drop=True)
    )
    aggregate["avg_trade_net_pooled"] = aggregate["net_return"] / aggregate["trades"].replace(0, np.nan)
    aggregate["effective_cost_bps"] = aggregate["total_cost"] / aggregate["trades"].replace(0, np.nan) * 10_000.0
    split_order = {"validation": 0, "test": 1}
    aggregate["_split_order"] = aggregate["split"].map(split_order).fillna(99)
    aggregate = aggregate.sort_values(
        ["_split_order", "cost_kind", "configured_cost_bps", "ibkr_plan", "notional_usd", "cost_scenario"],
        kind="stable",
    ).drop(columns=["_split_order"])
    return detail.reset_index(drop=True), aggregate.reset_index(drop=True)


def _lifecycle_parameter_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = _h8c_cfg(config)
    lifecycle = _lifecycle_cfg(config)
    if bool(lifecycle.get("entry_exit_mode", lifecycle.get("separate_entry_exit", False))):
        entry_gate_modes = [str(value) for value in lifecycle.get("entry_gate_modes", lifecycle.get("gate_modes", ["regime_confirmed"]))]
        entry_thresholds = [float(value) for value in lifecycle.get("entry_probability_thresholds", lifecycle.get("probability_thresholds", [0.55, 0.60, 0.65]))]
        entry_gaps = [float(value) for value in lifecycle.get("entry_min_probability_gaps", lifecycle.get("min_probability_gaps", [0.05, 0.10]))]
        entry_expected_cfg = {
            "use_expected_net_gate": lifecycle.get("entry_use_expected_net_gate", lifecycle.get("use_expected_net_gate", True)),
            "expected_net_threshold_bps": lifecycle.get("entry_expected_net_threshold_bps", lifecycle.get("expected_net_threshold_bps", [0.0, 1.0])),
            "min_expected_net_gaps_bps": lifecycle.get("entry_min_expected_net_gaps_bps", lifecycle.get("min_expected_net_gaps_bps", [0.0])),
        }
        entry_expected_thresholds = _expected_net_thresholds(entry_expected_cfg)
        entry_expected_gaps = _expected_net_gaps(entry_expected_cfg)
        entry_regime_thresholds = [float(value) for value in lifecycle.get("entry_regime_thresholds", lifecycle.get("regime_thresholds", [0.65, 0.75]))]
        entry_entropy_values = lifecycle.get("entry_max_entropy_values", lifecycle.get("max_entropy_values", [0.75]))

        exit_gate_modes = [str(value) for value in lifecycle.get("exit_gate_modes", entry_gate_modes)]
        exit_thresholds = [float(value) for value in lifecycle.get("exit_probability_thresholds", [0.45, 0.50, 0.55])]
        exit_gaps = [float(value) for value in lifecycle.get("exit_min_probability_gaps", [0.0, 0.05])]
        exit_expected_cfg = {
            "use_expected_net_gate": lifecycle.get("exit_use_expected_net_gate", False),
            "expected_net_threshold_bps": lifecycle.get("exit_expected_net_threshold_bps", [None]),
            "min_expected_net_gaps_bps": lifecycle.get("exit_min_expected_net_gaps_bps", [0.0]),
        }
        exit_expected_thresholds = _expected_net_thresholds(exit_expected_cfg)
        exit_expected_gaps = _expected_net_gaps(exit_expected_cfg)
        exit_regime_thresholds = [float(value) for value in lifecycle.get("exit_regime_thresholds", [0.45, 0.55])]
        exit_entropy_values = lifecycle.get("exit_max_entropy_values", [None])

        setup_specs = _lifecycle_setup_specs(lifecycle)
        max_hold_values = [int(value) for value in lifecycle.get("max_hold_bars", [0, 8, 16])]
        min_hold_values = [int(value) for value in lifecycle.get("min_hold_bars", [0])]
        cooldown_values = [int(value) for value in lifecycle.get("cooldown_bars", [0])]
        exit_on_loss_values = [bool(value) for value in lifecycle.get("exit_on_signal_loss", [True])]
        rows: list[dict[str, Any]] = []
        for entry_gate_mode in entry_gate_modes:
            entry_mode_regime_thresholds = [None] if entry_gate_mode == "position_only" else entry_regime_thresholds
            entry_mode_entropy_values = [None] if entry_gate_mode == "position_only" else entry_entropy_values
            for exit_gate_mode in exit_gate_modes:
                exit_mode_regime_thresholds = [None] if exit_gate_mode == "position_only" else exit_regime_thresholds
                exit_mode_entropy_values = [None] if exit_gate_mode == "position_only" else exit_entropy_values
                for entry_threshold in entry_thresholds:
                    for entry_gap in entry_gaps:
                        for entry_expected_threshold in entry_expected_thresholds:
                            entry_gap_values = [None] if entry_expected_threshold is None else entry_expected_gaps
                            for entry_expected_gap in entry_gap_values:
                                for entry_regime_threshold in entry_mode_regime_thresholds:
                                    for entry_max_entropy in entry_mode_entropy_values:
                                        for exit_threshold in exit_thresholds:
                                            for exit_gap in exit_gaps:
                                                for exit_expected_threshold in exit_expected_thresholds:
                                                    exit_gap_values = [None] if exit_expected_threshold is None else exit_expected_gaps
                                                    for exit_expected_gap in exit_gap_values:
                                                        for exit_regime_threshold in exit_mode_regime_thresholds:
                                                            for exit_max_entropy in exit_mode_entropy_values:
                                                                for max_hold in max_hold_values:
                                                                    for min_hold in min_hold_values:
                                                                        for cooldown in cooldown_values:
                                                                            for exit_on_loss in exit_on_loss_values:
                                                                                base = {
                                                                                    "entry_exit_mode": True,
                                                                                    "entry_gate_mode": entry_gate_mode,
                                                                                    "entry_threshold": float(entry_threshold),
                                                                                    "entry_min_probability_gap": float(entry_gap),
                                                                                    "entry_expected_net_threshold_bps": (
                                                                                        np.nan if entry_expected_threshold is None else float(entry_expected_threshold)
                                                                                    ),
                                                                                    "entry_min_expected_net_gap_bps": (
                                                                                        np.nan if entry_expected_gap is None else float(entry_expected_gap)
                                                                                    ),
                                                                                    "entry_regime_threshold": (
                                                                                        np.nan if entry_regime_threshold is None else float(entry_regime_threshold)
                                                                                    ),
                                                                                    "entry_max_entropy": np.nan if entry_max_entropy is None else float(entry_max_entropy),
                                                                                    "exit_gate_mode": exit_gate_mode,
                                                                                    "exit_threshold": float(exit_threshold),
                                                                                    "exit_min_probability_gap": float(exit_gap),
                                                                                    "exit_expected_net_threshold_bps": (
                                                                                        np.nan if exit_expected_threshold is None else float(exit_expected_threshold)
                                                                                    ),
                                                                                    "exit_min_expected_net_gap_bps": (
                                                                                        np.nan if exit_expected_gap is None else float(exit_expected_gap)
                                                                                    ),
                                                                                    "exit_regime_threshold": (
                                                                                        np.nan if exit_regime_threshold is None else float(exit_regime_threshold)
                                                                                    ),
                                                                                    "exit_max_entropy": np.nan if exit_max_entropy is None else float(exit_max_entropy),
                                                                                    "max_hold_bars": int(max_hold),
                                                                                    "min_hold_bars": int(min_hold),
                                                                                    "cooldown_bars": int(cooldown),
                                                                                    "exit_on_signal_loss": bool(exit_on_loss),
                                                                                }
                                                                                rows.extend({**base, **setup} for setup in setup_specs)
        return rows

    gate_modes = [str(value) for value in lifecycle.get("gate_modes", cfg.get("gate_modes", ["position_only", "regime_confirmed"]))]
    thresholds = [float(value) for value in lifecycle.get("probability_thresholds", cfg.get("probability_thresholds", [0.50, 0.55, 0.60, 0.65, 0.70]))]
    gaps = [float(value) for value in lifecycle.get("min_probability_gaps", cfg.get("min_probability_gaps", [0.0, 0.05, 0.10]))]
    expected_cfg = {
        "use_expected_net_gate": lifecycle.get("use_expected_net_gate", cfg.get("use_expected_net_gate", False)),
        "expected_net_threshold_bps": lifecycle.get("expected_net_threshold_bps", cfg.get("expected_net_threshold_bps", [0.0])),
        "min_expected_net_gaps_bps": lifecycle.get("min_expected_net_gaps_bps", cfg.get("min_expected_net_gaps_bps", [0.0])),
    }
    expected_thresholds = _expected_net_thresholds(expected_cfg)
    expected_gaps = _expected_net_gaps(expected_cfg)
    regime_thresholds = [float(value) for value in lifecycle.get("regime_thresholds", cfg.get("regime_thresholds", [0.65, 0.75]))]
    entropy_values = lifecycle.get("max_entropy_values", cfg.get("max_entropy_values", [None]))
    max_hold_values = [int(value) for value in lifecycle.get("max_hold_bars", [2, 4, 8])]
    exit_on_loss_values = [bool(value) for value in lifecycle.get("exit_on_signal_loss", [True, False])]
    rows: list[dict[str, Any]] = []
    for gate_mode in gate_modes:
        mode_regime_thresholds = [None] if gate_mode == "position_only" else regime_thresholds
        mode_entropy_values = [None] if gate_mode == "position_only" else entropy_values
        for threshold in thresholds:
            for gap in gaps:
                for expected_threshold in expected_thresholds:
                    expected_gap_values = [None] if expected_threshold is None else expected_gaps
                    for expected_gap in expected_gap_values:
                        for regime_threshold in mode_regime_thresholds:
                            for max_entropy in mode_entropy_values:
                                for max_hold in max_hold_values:
                                    for exit_on_loss in exit_on_loss_values:
                                        rows.append(
                                            {
                                                "gate_mode": gate_mode,
                                                "threshold": float(threshold),
                                                "min_probability_gap": float(gap),
                                                "expected_net_threshold_bps": np.nan if expected_threshold is None else float(expected_threshold),
                                                "min_expected_net_gap_bps": np.nan if expected_gap is None else float(expected_gap),
                                                "regime_threshold": np.nan if regime_threshold is None else float(regime_threshold),
                                                "max_entropy": np.nan if max_entropy is None else float(max_entropy),
                                                "max_hold_bars": int(max_hold),
                                                "exit_on_signal_loss": bool(exit_on_loss),
                                            }
                                        )
    return rows


def lifecycle_position_from_signal(
    signal: pd.Series,
    sessions: pd.Series,
    *,
    max_hold_bars: int,
    exit_on_signal_loss: bool,
) -> pd.Series:
    output = pd.Series(0.0, index=signal.index, dtype=float)
    for _, idx in sessions.groupby(sessions, sort=False).groups.items():
        position = 0.0
        age = 0
        for label in idx:
            current_signal = float(signal.loc[label])
            if position == 0.0:
                if current_signal != 0.0:
                    position = current_signal
                    age = 1
            elif current_signal == -position:
                position = current_signal
                age = 1
            elif current_signal == position:
                age = 1
            elif exit_on_signal_loss:
                position = 0.0
                age = 0
            elif int(max_hold_bars) > 0 and age >= int(max_hold_bars):
                position = 0.0
                age = 0
            else:
                age += 1
            output.loc[label] = position
    return output


def lifecycle_position_from_entry_exit(
    entry_signal: pd.Series,
    hold_signal: pd.Series,
    sessions: pd.Series,
    *,
    max_hold_bars: int = 0,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
    exit_on_signal_loss: bool = True,
) -> pd.Series:
    output = pd.Series(0.0, index=entry_signal.index, dtype=float)
    max_hold = int(max_hold_bars)
    min_hold = max(0, int(min_hold_bars))
    cooldown_cfg = max(0, int(cooldown_bars))
    for _, idx in sessions.groupby(sessions, sort=False).groups.items():
        position = 0.0
        age = 0
        cooldown = 0
        for label in idx:
            current_entry = float(entry_signal.loc[label])
            current_hold = float(hold_signal.loc[label])
            if position == 0.0:
                if cooldown > 0:
                    cooldown -= 1
                elif current_entry != 0.0:
                    position = current_entry
                    age = 1
            else:
                can_exit = age >= min_hold
                if can_exit and current_entry == -position:
                    position = current_entry
                    age = 1
                elif can_exit and max_hold > 0 and age >= max_hold:
                    position = 0.0
                    age = 0
                    cooldown = cooldown_cfg
                elif can_exit and exit_on_signal_loss and current_hold != position:
                    position = 0.0
                    age = 0
                    cooldown = cooldown_cfg
                else:
                    age += 1
            output.loc[label] = position
    return output


def _lifecycle_dataset(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    forward = build_forward_returns(features, [1]).rename(
        columns={"horizon_bars": "lifecycle_horizon_bars", "fwd_ret": "bar_fwd_ret", "entry_px": "bar_entry_px", "exit_px": "bar_exit_px"}
    )
    drop_columns = [column for column in ["fwd_ret", "entry_px", "exit_px"] if column in predictions.columns]
    merged = predictions.drop(columns=drop_columns).merge(forward, on=MERGE_KEYS, how="inner", validate="many_to_one")
    feature_slice = features.reset_index(names="source_index")
    setup_columns = [column for column in SIGNAL_COLUMNS if column in feature_slice.columns and column not in merged.columns]
    if setup_columns:
        merged = merged.merge(feature_slice.loc[:, ["source_index", *setup_columns]], on="source_index", how="left", validate="many_to_one")
    merged["fwd_ret"] = merged["bar_fwd_ret"].astype(float)
    merged["entry_px"] = merged["bar_entry_px"].astype(float)
    merged["exit_px"] = merged["bar_exit_px"].astype(float)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return merged.sort_values(["fold", "split", "timestamp"], kind="stable").reset_index(drop=True)


def _lifecycle_scenarios(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    scenarios = allocation_cost_scenarios(_scenario_config(config), names)
    if not scenarios and names is not None:
        scenarios = allocation_cost_scenarios(_scenario_config(config))
    return scenarios


def _lifecycle_position_for_params(group: pd.DataFrame, params: pd.Series | dict[str, Any]) -> pd.Series:
    if _param_bool(params, "entry_exit_mode", False):
        entry_signal = _lifecycle_signal_from_params(group, params, "entry")
        entry_signal = _apply_setup_to_entry_signal(entry_signal, _setup_signal_mask(group, params), params)
        hold_signal = _lifecycle_signal_from_params(group, params, "exit")
        return lifecycle_position_from_entry_exit(
            entry_signal,
            hold_signal,
            group["session"],
            max_hold_bars=int(_param_value(params, "max_hold_bars", 0)),
            min_hold_bars=int(_param_value(params, "min_hold_bars", 0)),
            cooldown_bars=int(_param_value(params, "cooldown_bars", 0)),
            exit_on_signal_loss=_param_bool(params, "exit_on_signal_loss", True),
        )

    signal = _position_from_probabilities(
        group,
        float(_param_value(params, "threshold")),
        float(_param_value(params, "min_probability_gap")),
        gate_mode=str(_param_value(params, "gate_mode", "position_only")),
        regime_threshold=_param_float(params, "regime_threshold"),
        max_entropy=_param_float(params, "max_entropy"),
        min_expected_net_bps=_param_float(params, "expected_net_threshold_bps"),
        min_expected_net_gap_bps=_param_float(params, "min_expected_net_gap_bps"),
    )
    return lifecycle_position_from_signal(
        signal,
        group["session"],
        max_hold_bars=int(_param_value(params, "max_hold_bars", 0)),
        exit_on_signal_loss=_param_bool(params, "exit_on_signal_loss", True),
    )


def evaluate_lifecycle_grid(predictions: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lifecycle = _lifecycle_cfg(config)
    eval_splits = {str(value) for value in lifecycle.get("eval_splits", ["validation", "test"])}
    scenario_names = lifecycle.get("grid_cost_scenarios", [str(lifecycle.get("primary_cost_scenario", "ibkr_tiered_10000"))])
    scenarios = _lifecycle_scenarios(config, [str(value) for value in scenario_names])
    dataset = _lifecycle_dataset(predictions, features)
    rows: list[dict[str, Any]] = []
    for keys, group in dataset.groupby(["fold", "split"], sort=False):
        fold, split = keys
        if str(split) not in eval_splits:
            continue
        for params in _lifecycle_parameter_rows(config):
            position = _lifecycle_position_for_params(group, params)
            for scenario in scenarios:
                rows.append(
                    {
                        "fold": int(fold),
                        "split": str(split),
                        **params,
                        "cost_scenario": str(scenario["cost_scenario"]),
                        "cost_kind": str(scenario["cost_kind"]),
                        "configured_round_trip_bps": float(scenario["round_trip_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                        "ibkr_plan": str(scenario.get("ibkr_plan", "")),
                        "notional_usd": float(scenario.get("notional_usd", np.nan)),
                        "spread_slippage_bps_round_trip": float(scenario.get("ibkr", {}).get("spread_slippage_bps_round_trip", np.nan)),
                        **evaluate_allocation_frame(group, position, scenario),
                    }
                )
    return pd.DataFrame(rows)


def _lifecycle_param_columns() -> list[str]:
    return [
        "entry_exit_mode",
        "setup_entry_mode",
        "setup_name",
        "setup_family",
        "setup_direction",
        "setup_params_json",
        "setup_column_map_json",
        "gate_mode",
        "threshold",
        "min_probability_gap",
        "expected_net_threshold_bps",
        "min_expected_net_gap_bps",
        "regime_threshold",
        "max_entropy",
        "entry_gate_mode",
        "entry_threshold",
        "entry_min_probability_gap",
        "entry_expected_net_threshold_bps",
        "entry_min_expected_net_gap_bps",
        "entry_regime_threshold",
        "entry_max_entropy",
        "exit_gate_mode",
        "exit_threshold",
        "exit_min_probability_gap",
        "exit_expected_net_threshold_bps",
        "exit_min_expected_net_gap_bps",
        "exit_regime_threshold",
        "exit_max_entropy",
        "max_hold_bars",
        "min_hold_bars",
        "cooldown_bars",
        "exit_on_signal_loss",
    ]


def aggregate_lifecycle_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    param_columns = [column for column in _lifecycle_param_columns() if column in metrics.columns]
    group_cols = [
        *param_columns,
        "split",
        "cost_scenario",
        "cost_kind",
        "configured_round_trip_bps",
        "ibkr_plan",
        "notional_usd",
        "spread_slippage_bps_round_trip",
    ]
    grouped = (
        metrics.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            folds=("fold", "nunique"),
            positive_folds=("net_return", lambda values: int((values > 0.0).sum())),
            rows=("rows", "sum"),
            sessions=("sessions", "sum"),
            active_bars=("active_bars", "sum"),
            long_bars=("long_bars", "sum"),
            short_bars=("short_bars", "sum"),
            rebalance_count=("rebalance_count", "sum"),
            turnover=("turnover", "sum"),
            gross_return=("gross_return", "sum"),
            total_cost=("total_cost", "sum"),
            net_return=("net_return", "sum"),
            avg_abs_position_mean=("avg_abs_position", "mean"),
            turnover_per_session_mean=("turnover_per_session", "mean"),
            net_per_turnover_mean=("net_per_turnover", "mean"),
            daily_sharpe_mean=("daily_sharpe", "mean"),
            max_drawdown_max=("max_drawdown", "max"),
            top_session_abs_net_share_max=("top_session_abs_net_share", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["net_per_turnover_pooled"] = grouped["net_return"] / grouped["turnover"].replace(0.0, np.nan)
    grouped["effective_round_trip_cost_bps"] = grouped["total_cost"] / grouped["turnover"].replace(0.0, np.nan) * 2.0 * 10_000.0
    return grouped.sort_values(["split", "cost_scenario", "daily_sharpe_mean", "net_return"], ascending=[True, True, False, False], kind="stable").reset_index(drop=True)


def select_lifecycle_candidate(aggregate: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lifecycle = _lifecycle_cfg(config)
    primary = str(lifecycle.get("primary_cost_scenario", "ibkr_tiered_10000"))
    selection_cost = str(lifecycle.get("selection_cost_scenario", primary))
    min_turnover = float(lifecycle.get("min_validation_turnover", 10.0))
    validation = aggregate[aggregate["split"].eq("validation") & aggregate["cost_scenario"].eq(selection_cost)].copy()
    if validation.empty:
        validation = aggregate[aggregate["split"].eq("validation")].copy()
    eligible = validation[validation["turnover"].ge(min_turnover)].copy()
    if eligible.empty:
        return pd.DataFrame()
    selected = eligible.sort_values(
        ["daily_sharpe_mean", "net_return", "net_per_turnover_pooled", "turnover"],
        ascending=[False, False, False, False],
        kind="stable",
    ).head(1).copy()
    selected["selected_on"] = "validation"
    selected["selection_cost_scenario"] = selection_cost
    selected["selection_rank_metric"] = "daily_sharpe_mean"
    return selected.reset_index(drop=True)


def lifecycle_selected_sensitivity(predictions: pd.DataFrame, features: pd.DataFrame, selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    dataset = _lifecycle_dataset(predictions, features)
    row = selected.iloc[0]
    scenarios = _lifecycle_scenarios(config)
    rows: list[dict[str, Any]] = []
    for keys, group in dataset[dataset["split"].isin(["validation", "test"])].groupby(["fold", "split"], sort=False):
        fold, split = keys
        position = _lifecycle_position_for_params(group, row)
        for scenario in scenarios:
            rows.append(
                {
                    "fold": int(fold),
                    "split": str(split),
                    **{column: row[column] for column in _lifecycle_param_columns() if column in row.index},
                    "cost_scenario": str(scenario["cost_scenario"]),
                    "cost_kind": str(scenario["cost_kind"]),
                    "configured_round_trip_bps": float(scenario["round_trip_bps"]) if scenario["cost_kind"] == "bps" else np.nan,
                    "ibkr_plan": str(scenario.get("ibkr_plan", "")),
                    "notional_usd": float(scenario.get("notional_usd", np.nan)),
                    "spread_slippage_bps_round_trip": float(scenario.get("ibkr", {}).get("spread_slippage_bps_round_trip", np.nan)),
                    **evaluate_allocation_frame(group, position, scenario),
                }
            )
    return aggregate_lifecycle_metrics(pd.DataFrame(rows))


def lifecycle_promotion_decision(selected: pd.DataFrame, sensitivity: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lifecycle = _lifecycle_cfg(config)
    if selected.empty or sensitivity.empty:
        return pd.DataFrame([{"decision": "rejected", "reason": "no selected lifecycle candidate"}])
    primary = str(lifecycle.get("primary_cost_scenario", "ibkr_tiered_10000"))
    conservative = str(lifecycle.get("conservative_cost_scenario", "bps_2"))
    stress = str(lifecycle.get("stress_cost_scenario", "bps_5"))
    min_positive_fold_share = float(lifecycle.get("min_positive_fold_share", 0.67))
    min_daily_sharpe = float(lifecycle.get("min_daily_sharpe", 0.75))
    min_net_per_turnover_bps = float(lifecycle.get("min_net_per_turnover_bps", 1.0))
    failed: list[str] = []

    def row(split: str, scenario: str) -> pd.Series | None:
        frame = sensitivity[sensitivity["split"].eq(split) & sensitivity["cost_scenario"].eq(scenario)]
        return None if frame.empty else frame.iloc[0]

    def pass_core(split: str, scenario: str) -> bool:
        current = row(split, scenario)
        if current is None:
            failed.append(f"missing_{split}_{scenario}")
            return False
        ok = True
        fold_share = float(current["positive_folds"]) / max(float(current["folds"]), 1.0)
        net_per_turnover_bps = float(current["net_per_turnover_pooled"]) * 10_000.0
        if float(current["net_return"]) <= 0.0:
            failed.append(f"{split}_{scenario}_net_not_positive")
            ok = False
        if fold_share < min_positive_fold_share:
            failed.append(f"{split}_{scenario}_positive_folds_below_min")
            ok = False
        if float(current["daily_sharpe_mean"]) < min_daily_sharpe:
            failed.append(f"{split}_{scenario}_sharpe_below_min")
            ok = False
        if net_per_turnover_bps < min_net_per_turnover_bps:
            failed.append(f"{split}_{scenario}_net_per_turnover_below_min")
            ok = False
        return ok

    validation_primary_ok = pass_core("validation", primary)
    test_primary_ok = pass_core("test", primary)
    test_conservative_ok = pass_core("test", conservative)
    test_stress = row("test", stress)
    stress_ok = bool(test_stress is not None and float(test_stress["net_return"]) > 0.0)
    if not stress_ok:
        failed.append(f"test_{stress}_net_not_positive")

    if validation_primary_ok and test_primary_ok and test_conservative_ok and stress_ok:
        decision = "promotion_candidate"
    elif validation_primary_ok and test_primary_ok and test_conservative_ok:
        decision = "research_candidate_cost_fragile"
    else:
        decision = "rejected"

    selected_row = selected.iloc[0]
    return pd.DataFrame(
        [
            {
                "decision": decision,
                "primary_cost_scenario": primary,
                "conservative_cost_scenario": conservative,
                "stress_cost_scenario": stress,
                "failed_checks": ",".join(failed),
                "gate_mode": selected_row.get("gate_mode"),
                "threshold": selected_row.get("threshold"),
                "min_probability_gap": selected_row.get("min_probability_gap"),
                "expected_net_threshold_bps": selected_row.get("expected_net_threshold_bps"),
                "min_expected_net_gap_bps": selected_row.get("min_expected_net_gap_bps"),
                "regime_threshold": selected_row.get("regime_threshold"),
                "max_entropy": selected_row.get("max_entropy"),
                "entry_exit_mode": selected_row.get("entry_exit_mode"),
                "setup_entry_mode": selected_row.get("setup_entry_mode"),
                "setup_name": selected_row.get("setup_name"),
                "setup_family": selected_row.get("setup_family"),
                "setup_direction": selected_row.get("setup_direction"),
                "setup_params_json": selected_row.get("setup_params_json"),
                "setup_column_map_json": selected_row.get("setup_column_map_json"),
                "entry_gate_mode": selected_row.get("entry_gate_mode"),
                "entry_threshold": selected_row.get("entry_threshold"),
                "entry_min_probability_gap": selected_row.get("entry_min_probability_gap"),
                "entry_expected_net_threshold_bps": selected_row.get("entry_expected_net_threshold_bps"),
                "entry_min_expected_net_gap_bps": selected_row.get("entry_min_expected_net_gap_bps"),
                "entry_regime_threshold": selected_row.get("entry_regime_threshold"),
                "entry_max_entropy": selected_row.get("entry_max_entropy"),
                "exit_gate_mode": selected_row.get("exit_gate_mode"),
                "exit_threshold": selected_row.get("exit_threshold"),
                "exit_min_probability_gap": selected_row.get("exit_min_probability_gap"),
                "exit_expected_net_threshold_bps": selected_row.get("exit_expected_net_threshold_bps"),
                "exit_min_expected_net_gap_bps": selected_row.get("exit_min_expected_net_gap_bps"),
                "exit_regime_threshold": selected_row.get("exit_regime_threshold"),
                "exit_max_entropy": selected_row.get("exit_max_entropy"),
                "max_hold_bars": selected_row.get("max_hold_bars"),
                "min_hold_bars": selected_row.get("min_hold_bars"),
                "cooldown_bars": selected_row.get("cooldown_bars"),
                "exit_on_signal_loss": selected_row.get("exit_on_signal_loss"),
            }
        ]
    )


def _split_frame(frame: pd.DataFrame, fold: LabFold, split: str) -> pd.DataFrame:
    sessions = {
        "train": fold.train_sessions,
        "validation": fold.validation_sessions,
        "test": fold.test_sessions,
    }[split]
    return frame[(frame["fold"].eq(int(fold.fold))) & (frame["split"].eq(split)) & frame["session"].isin(sessions)].copy()


def fit_predict_walk_forward(
    supervised: pd.DataFrame,
    feature_columns: list[str],
    folds: list[LabFold],
    config: dict[str, Any],
    target_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    model_root = models_dir(config, target_symbol)
    model_root.mkdir(parents=True, exist_ok=True)

    h8_columns = [
        column
        for column in supervised.columns
        if column.startswith("p_") and column not in {"p_long_profit", "p_short_profit"}
    ]
    h8_columns.extend([column for column in ["max_prob", "entropy"] if column in supervised.columns])
    base_columns = [
        "variant",
        "fold",
        "split",
        "source_index",
        "timestamp",
        "session",
        "bar_index",
        "horizon_bars",
        "fwd_ret",
        "entry_px",
        "exit_px",
        "label_cost_bps",
        "label_cost_scenario",
        "label_cost_return",
        "label_cost_effective_bps",
        "label_long_profit",
        "label_short_profit",
        "long_net_return",
        "short_net_return",
    ] + h8_columns
    for fold in folds:
        train = _split_frame(supervised, fold, "train")
        if train.empty:
            continue
        side_models = {side: _fit_side_model(train, side, feature_columns, config) for side in ["long", "short"]}
        net_models = {side: _fit_side_net_model(train, side, feature_columns, config) for side in ["long", "short"]}
        fold_dir = model_root / f"fold_{int(fold.fold)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"feature_columns": feature_columns, "side_models": side_models, "net_models": net_models, "fold": int(fold.fold)},
            fold_dir / "h8c_position_model.joblib",
        )

        for side, side_model in side_models.items():
            if side_model.model is None:
                coefficient_rows.append(
                    {
                        "fold": int(fold.fold),
                        "side": side,
                        "feature": "__constant_probability__",
                        "coefficient": float(side_model.constant_probability or 0.0),
                    }
                )
                continue
            for feature, coefficient in zip(feature_columns, side_model.model.coef_[0], strict=True):
                coefficient_rows.append({"fold": int(fold.fold), "side": side, "feature": feature, "coefficient": float(coefficient)})
        for side, net_model in net_models.items():
            if net_model.model is None:
                coefficient_rows.append(
                    {
                        "fold": int(fold.fold),
                        "side": f"{side}_expected_net",
                        "feature": "__constant_prediction__",
                        "coefficient": float(net_model.constant_prediction),
                    }
                )
                continue
            for feature, coefficient in zip(feature_columns, net_model.model.coef_, strict=True):
                coefficient_rows.append(
                    {"fold": int(fold.fold), "side": f"{side}_expected_net", "feature": feature, "coefficient": float(coefficient)}
                )

        for split in ["train", "validation", "test"]:
            split_frame = _split_frame(supervised, fold, split)
            if split_frame.empty:
                continue
            out = split_frame.loc[:, [column for column in base_columns if column in split_frame.columns]].copy()
            out["p_long_profit"] = _predict_side_probability(side_models["long"], split_frame)
            out["p_short_profit"] = _predict_side_probability(side_models["short"], split_frame)
            out["expected_long_net"] = _predict_side_net(net_models["long"], split_frame)
            out["expected_short_net"] = _predict_side_net(net_models["short"], split_frame)
            out["expected_long_net_bps"] = out["expected_long_net"] * 10_000.0
            out["expected_short_net_bps"] = out["expected_short_net"] * 10_000.0
            out["expected_net_edge_bps"] = out["expected_long_net_bps"] - out["expected_short_net_bps"]
            out["probability_edge"] = out["p_long_profit"] - out["p_short_profit"]
            out["predicted_side"] = np.where(out["probability_edge"] >= 0.0, "long", "short")
            prediction_frames.append(out)

    if not prediction_frames:
        raise ValueError("No H8c prediction frames were generated")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients


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
    feature_columns: list[str],
    folds: list[LabFold],
    model_quality: pd.DataFrame,
    grid: pd.DataFrame,
    selected: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    cost_sensitivity_aggregate: pd.DataFrame,
    lifecycle_selected: pd.DataFrame,
    lifecycle_sensitivity: pd.DataFrame,
    lifecycle_decision: pd.DataFrame,
    coefficients: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    cfg = _h8c_cfg(config)
    selected_cols = [
        "fold",
        "split",
        "gate_mode",
        "threshold",
        "min_probability_gap",
        "expected_net_threshold_bps",
        "min_expected_net_gap_bps",
        "regime_threshold",
        "max_entropy",
        "cost_bps",
        "trades",
        "long_trades",
        "short_trades",
        "net_return",
        "avg_trade_net_pooled",
        "avg_trade_net",
        "hit_rate",
        "profit_factor",
        "daily_sharpe",
        "max_drawdown",
    ]
    aggregate_cols = [
        "split",
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
    cost_cols = [
        "split",
        "cost_scenario",
        "cost_kind",
        "effective_cost_bps",
        "notional_usd",
        "spread_slippage_bps_round_trip",
        "positive_folds",
        "trades",
        "gross_return",
        "total_cost",
        "net_return",
        "avg_trade_net_pooled",
        "avg_trade_net_min",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    lifecycle_selected_cols = [
        "entry_exit_mode",
        "setup_entry_mode",
        "setup_name",
        "setup_family",
        "setup_direction",
        "gate_mode",
        "threshold",
        "min_probability_gap",
        "expected_net_threshold_bps",
        "min_expected_net_gap_bps",
        "regime_threshold",
        "max_entropy",
        "entry_gate_mode",
        "entry_threshold",
        "entry_min_probability_gap",
        "entry_expected_net_threshold_bps",
        "entry_min_expected_net_gap_bps",
        "entry_regime_threshold",
        "entry_max_entropy",
        "exit_gate_mode",
        "exit_threshold",
        "exit_min_probability_gap",
        "exit_expected_net_threshold_bps",
        "exit_min_expected_net_gap_bps",
        "exit_regime_threshold",
        "exit_max_entropy",
        "max_hold_bars",
        "min_hold_bars",
        "cooldown_bars",
        "exit_on_signal_loss",
        "cost_scenario",
        "positive_folds",
        "active_bars",
        "rebalance_count",
        "turnover",
        "gross_return",
        "total_cost",
        "net_return",
        "net_per_turnover_pooled",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    lifecycle_sensitivity_cols = [
        "split",
        "cost_scenario",
        "effective_round_trip_cost_bps",
        "notional_usd",
        "positive_folds",
        "active_bars",
        "rebalance_count",
        "turnover",
        "gross_return",
        "total_cost",
        "net_return",
        "net_per_turnover_pooled",
        "daily_sharpe_mean",
        "max_drawdown_max",
    ]
    lifecycle_decision_cols = [
        "decision",
        "primary_cost_scenario",
        "conservative_cost_scenario",
        "stress_cost_scenario",
        "failed_checks",
        "entry_exit_mode",
        "setup_entry_mode",
        "setup_name",
        "setup_family",
        "setup_direction",
        "gate_mode",
        "threshold",
        "entry_threshold",
        "exit_threshold",
        "max_hold_bars",
        "min_hold_bars",
        "cooldown_bars",
        "expected_net_threshold_bps",
        "entry_expected_net_threshold_bps",
        "exit_expected_net_threshold_bps",
        "exit_on_signal_loss",
    ]
    quality_cols = [
        "fold",
        "split",
        "side",
        "rows",
        "positive_rate",
        "avg_probability",
        "auc",
        "brier",
        "actual_net_mean_bps",
        "expected_net_mean_bps",
        "net_mae_bps",
        "net_corr",
    ]
    top_coef = (
        coefficients.assign(abs_coefficient=coefficients["coefficient"].abs())
        .sort_values(["fold", "side", "abs_coefficient"], ascending=[True, True, False], kind="stable")
        .groupby(["fold", "side"], as_index=False)
        .head(12)
    )
    output_text = "\n".join(f"- {name}: `{path}`" for name, path in outputs.items())
    fold_text = f"{len(folds)} (`{folds[0].train_months[0]}` to `{folds[-1].test_months[-1]}`)" if folds else "0"
    return f"""# H8c Position Probability Model - {target_symbol.upper()}

## Scope

- Feature file: `{bayesian_regime_h8.features_path(config, target_symbol)}`
- H8 variant used as input: `{_selected_variant(config)}`
- Horizon bars: `{cfg.get("horizon_bars", 4)}`
- Label cost: `{cfg.get("label_cost_scenario", f"bps_{float(cfg.get('label_cost_bps', cfg.get('primary_cost_bps', 1.0))):g}")}`
- Selection cost bps: `{cfg.get("primary_cost_bps", cfg.get("label_cost_bps", 1.0))}`
- Walk-forward folds: {fold_text}
- Model: two binary logistic regressions, one for profitable long and one for profitable short.
- Net edge model: two ridge regressions for expected long/short net return after label cost.
- Test selection used: `no`

## Selected Validation Gates

{_markdown_table(selected.loc[:, [column for column in selected_cols if column in selected.columns]], max_rows=30)}

## Selected Validation/Test Metrics

{_markdown_table(selected_metrics.loc[:, [column for column in selected_cols if column in selected_metrics.columns]], max_rows=60)}

## Aggregate Selected Metrics

{_markdown_table(aggregate.loc[:, [column for column in aggregate_cols if column in aggregate.columns]], max_rows=20)}

## Realistic Cost Sensitivity

{_markdown_table(cost_sensitivity_aggregate.loc[:, [column for column in cost_cols if column in cost_sensitivity_aggregate.columns]], max_rows=80)}

## Executable Lifecycle Decision

{_markdown_table(lifecycle_decision.loc[:, [column for column in lifecycle_decision_cols if column in lifecycle_decision.columns]], max_rows=5)}

## Executable Lifecycle Selected Candidate

{_markdown_table(lifecycle_selected.loc[:, [column for column in lifecycle_selected_cols if column in lifecycle_selected.columns]], max_rows=10)}

## Executable Lifecycle Cost Sensitivity

{_markdown_table(lifecycle_sensitivity.loc[:, [column for column in lifecycle_sensitivity_cols if column in lifecycle_sensitivity.columns]], max_rows=80)}

## Probability Model Quality

{_markdown_table(model_quality.loc[:, [column for column in quality_cols if column in model_quality.columns]], max_rows=80)}

## Top Coefficients

{_markdown_table(top_coef.loc[:, ["fold", "side", "feature", "coefficient", "abs_coefficient"]], max_rows=80)}

## Feature Columns

{chr(10).join(f"- `{column}`" for column in feature_columns)}

## Outputs

{output_text}

## Notes

- Labels are net of the configured label cost scenario: long is profitable when forward return exceeds cost; short is profitable when negative forward return exceeds cost.
- Optional expected-net gates require the predicted net return to clear the configured bps threshold before a signal is tradable.
- The probability models are fit only on each fold's train split.
- Threshold and probability-gap selection is done only on validation.
- Realistic cost sensitivity replays the selected validation gate under fixed bps and IBKR-style scenarios. IBKR rows include two commissions, SEC/TAF sell fees, configured notional, and configured spread/slippage round-trip cost.
- Executable lifecycle rows convert the probability gate into one non-overlapping position, with exits on flip, signal loss, max hold, and session flatten. Costs are charged on turnover.
- Test rows are reported after validation selection, without choosing parameters on test.
"""


def run_config(config: dict[str, Any], target_symbol: str | None = None) -> tuple[Path, Path]:
    target = _target_symbol(config, target_symbol)
    raw_features = pd.read_parquet(bayesian_regime_h8.features_path(config, target))
    h8_frame = bayesian_regime_h8.prepare_h8_frame(raw_features, config)
    folds = bayesian_regime_h8.build_h8_folds(h8_frame, config)
    posteriors, _model_registry = bayesian_regime_h8.run_h8_posteriors(h8_frame, folds, config, target)
    supervised, feature_columns = prepare_supervised_frame(raw_features, posteriors, config, selected_variant=_selected_variant(config))
    predictions, coefficients = fit_predict_walk_forward(supervised, feature_columns, folds, config, target)
    model_quality = _model_quality_rows(predictions)
    grid = evaluate_probability_gate(predictions, config)
    selected = select_validation_gates(grid, config)
    selected_metrics = selected_gate_metrics(grid, selected)
    aggregate = aggregate_selected_metrics(selected_metrics)
    cost_sensitivity, cost_sensitivity_aggregate = selected_cost_sensitivity(predictions, selected, config)
    lifecycle_grid = evaluate_lifecycle_grid(predictions, raw_features, config)
    lifecycle_aggregate = aggregate_lifecycle_metrics(lifecycle_grid)
    lifecycle_selected = select_lifecycle_candidate(lifecycle_aggregate, config)
    lifecycle_sensitivity = lifecycle_selected_sensitivity(predictions, raw_features, lifecycle_selected, config)
    lifecycle_decision = lifecycle_promotion_decision(lifecycle_selected, lifecycle_sensitivity, config)

    out_dir = results_dir(config, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": out_dir / "h8c_position_predictions.parquet",
        "model_quality": out_dir / "h8c_model_quality.parquet",
        "threshold_grid": out_dir / "h8c_threshold_grid.parquet",
        "selected_gates": out_dir / "h8c_selected_gates.parquet",
        "selected_metrics": out_dir / "h8c_selected_metrics.parquet",
        "aggregate": out_dir / "h8c_selected_aggregate.parquet",
        "cost_sensitivity": out_dir / "h8c_cost_sensitivity.parquet",
        "cost_sensitivity_aggregate": out_dir / "h8c_cost_sensitivity_aggregate.parquet",
        "lifecycle_grid": out_dir / "h8c_lifecycle_grid.parquet",
        "lifecycle_aggregate": out_dir / "h8c_lifecycle_aggregate.parquet",
        "lifecycle_selected": out_dir / "h8c_lifecycle_selected.parquet",
        "lifecycle_sensitivity": out_dir / "h8c_lifecycle_sensitivity.parquet",
        "lifecycle_decision": out_dir / "h8c_lifecycle_decision.parquet",
        "coefficients": out_dir / "h8c_coefficients.parquet",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    model_quality.to_parquet(outputs["model_quality"], index=False)
    grid.to_parquet(outputs["threshold_grid"], index=False)
    selected.to_parquet(outputs["selected_gates"], index=False)
    selected_metrics.to_parquet(outputs["selected_metrics"], index=False)
    aggregate.to_parquet(outputs["aggregate"], index=False)
    cost_sensitivity.to_parquet(outputs["cost_sensitivity"], index=False)
    cost_sensitivity_aggregate.to_parquet(outputs["cost_sensitivity_aggregate"], index=False)
    lifecycle_grid.to_parquet(outputs["lifecycle_grid"], index=False)
    lifecycle_aggregate.to_parquet(outputs["lifecycle_aggregate"], index=False)
    lifecycle_selected.to_parquet(outputs["lifecycle_selected"], index=False)
    lifecycle_sensitivity.to_parquet(outputs["lifecycle_sensitivity"], index=False)
    lifecycle_decision.to_parquet(outputs["lifecycle_decision"], index=False)
    coefficients.to_parquet(outputs["coefficients"], index=False)

    report = report_path(config, target)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(
            config,
            target,
            feature_columns,
            folds,
            model_quality,
            grid,
            selected,
            selected_metrics,
            aggregate,
            cost_sensitivity_aggregate,
            lifecycle_selected,
            lifecycle_sensitivity,
            lifecycle_decision,
            coefficients,
            outputs,
        ),
        encoding="utf-8",
    )
    return report, outputs["selected_metrics"]


def run(config_path: str | Path, target_symbol: str | None = None) -> tuple[Path, Path]:
    return run_config(load_yaml(config_path), target_symbol=target_symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H8c supervised position probability model.")
    parser.add_argument("--config", default="configs/hmm_bayesian_regime_h8c_qqq_15min.yaml")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    report, selected_metrics = run(args.config, args.target)
    print(f"H8c report written to: {report}")
    print(f"H8c selected metrics written to: {selected_metrics}")


if __name__ == "__main__":
    main()

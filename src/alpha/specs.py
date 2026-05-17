from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


SUPPORTED_ALPHA_MODES = {
    "signed",
    "inverse",
    "long_positive",
    "short_positive",
    "short_negative",
}

SUPPORTED_CONFIRMATIONS = {
    "same_sign",
    "opposite_sign",
    "min_value",
    "max_value",
    "min_abs_value",
    "max_abs_value",
    "min_quantile",
    "max_quantile",
    "min_abs_quantile",
    "max_abs_quantile",
}


@dataclass(frozen=True)
class ConfirmationRule:
    """Declarative filter applied after an alpha proposes a direction."""

    rule_type: str
    column: str
    value: float | None = None
    quantile: float | None = None
    description: str = ""

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ConfirmationRule":
        rule_type = str(raw.get("type", "")).strip()
        if rule_type not in SUPPORTED_CONFIRMATIONS:
            raise ValueError(f"unsupported confirmation type: {rule_type}")
        column = str(raw.get("column", "")).strip()
        if not column:
            raise ValueError("confirmation rules require a column")
        value = raw.get("value")
        quantile = raw.get("quantile")
        if rule_type.endswith("_quantile") and quantile is None:
            raise ValueError(f"{rule_type} confirmation requires quantile")
        return cls(
            rule_type=rule_type,
            column=column,
            value=None if value is None else float(value),
            quantile=None if quantile is None else float(quantile),
            description=str(raw.get("description", "")),
        )

    def gate_key(self, alpha_id: str) -> str:
        suffix = f"q{self.quantile:g}" if self.quantile is not None else "value"
        return f"{alpha_id}:{self.rule_type}:{self.column}:{suffix}"


@dataclass(frozen=True)
class AlphaSpec:
    """Alpha family member used by the research runner."""

    alpha_id: str
    family: str
    signal_column: str
    mode: str
    horizons: tuple[int, ...]
    threshold_quantiles: tuple[float, ...]
    confirmations: tuple[ConfirmationRule, ...] = ()
    description: str = ""
    min_trades: int = 30

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AlphaSpec":
        alpha_id = str(raw.get("alpha_id", "")).strip()
        family = str(raw.get("family", "")).strip()
        signal_column = str(raw.get("signal_column", "")).strip()
        mode = str(raw.get("mode", "")).strip()
        if not alpha_id:
            raise ValueError("alpha specs require alpha_id")
        if not family:
            raise ValueError(f"{alpha_id}: family is required")
        if not signal_column:
            raise ValueError(f"{alpha_id}: signal_column is required")
        if mode not in SUPPORTED_ALPHA_MODES:
            raise ValueError(f"{alpha_id}: unsupported mode {mode}")

        horizons = tuple(int(value) for value in raw.get("horizons", ()))
        if not horizons:
            raise ValueError(f"{alpha_id}: at least one horizon is required")
        threshold_quantiles = tuple(float(value) for value in raw.get("threshold_quantiles", ()))
        if not threshold_quantiles:
            raise ValueError(f"{alpha_id}: at least one threshold_quantile is required")
        confirmations = tuple(ConfirmationRule.from_mapping(item) for item in raw.get("confirmations", ()))
        return cls(
            alpha_id=alpha_id,
            family=family,
            signal_column=signal_column,
            mode=mode,
            horizons=horizons,
            threshold_quantiles=threshold_quantiles,
            confirmations=confirmations,
            description=str(raw.get("description", "")),
            min_trades=int(raw.get("min_trades", 30)),
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.signal_column, *(rule.column for rule in self.confirmations)]))


@dataclass(frozen=True)
class AlphaResearchPlan:
    """Top-level alpha research config.

    This object is intentionally small and declarative. Runners can evolve
    without changing the spec format that defines what is being researched.
    """

    research_id: str
    target_symbol: str
    timeframe: str
    feature_set_id: str
    feature_path_template: str
    split_policy_id: str
    primary_cost_profile: str
    conservative_cost_profile: str
    stress_cost_profile: str
    split_policy: dict[str, Any]
    output_dir_template: str
    alphas: tuple[AlphaSpec, ...]
    strategy_defaults: dict[str, Any]
    promotion_gates: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AlphaResearchPlan":
        research = raw.get("research", {})
        dataset = raw.get("dataset", {})
        costs = raw.get("costs", {})
        alpha_items = raw.get("alphas", ())
        if not isinstance(research, dict) or not isinstance(dataset, dict) or not isinstance(costs, dict):
            raise ValueError("research, dataset and costs sections must be mappings")
        alphas = tuple(AlphaSpec.from_mapping(item) for item in alpha_items)
        if not alphas:
            raise ValueError("alpha research plan requires at least one alpha")
        research_id = str(research.get("research_id", "")).strip()
        target_symbol = str(research.get("target_symbol", "")).strip().upper()
        timeframe = str(research.get("timeframe", "")).strip()
        feature_set_id = str(dataset.get("feature_set_id", "")).strip()
        feature_path_template = str(dataset.get("feature_path_template", "")).strip()
        split_policy_id = str(research.get("split_policy_id", "")).strip()
        if not all([research_id, target_symbol, timeframe, feature_set_id, feature_path_template, split_policy_id]):
            raise ValueError("research_id, target_symbol, timeframe, feature_set_id, feature_path_template and split_policy_id are required")
        return cls(
            research_id=research_id,
            target_symbol=target_symbol,
            timeframe=timeframe,
            feature_set_id=feature_set_id,
            feature_path_template=feature_path_template,
            split_policy_id=split_policy_id,
            primary_cost_profile=str(costs.get("primary", "")).strip(),
            conservative_cost_profile=str(costs.get("conservative", "")).strip(),
            stress_cost_profile=str(costs.get("stress", "")).strip(),
            split_policy=dict(research.get("split_policy", {})),
            output_dir_template=str(research.get("output_dir_template", "results/alpha/{research_id}/{target_symbol}/{timeframe}")).strip(),
            alphas=alphas,
            strategy_defaults=dict(raw.get("strategy_defaults", {})),
            promotion_gates=dict(raw.get("promotion_gates", {})),
            raw=dict(raw),
        )

    @property
    def required_feature_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for alpha in self.alphas:
            columns.extend(alpha.required_columns)
        return tuple(dict.fromkeys(columns))

    def feature_path(self) -> Path:
        return Path(
            self.feature_path_template.format(
                target_symbol=self.target_symbol,
                target=self.target_symbol,
                timeframe=self.timeframe,
                feature_set_id=self.feature_set_id,
            )
        )

    def output_dir(self) -> Path:
        return Path(
            self.output_dir_template.format(
                research_id=self.research_id,
                target_symbol=self.target_symbol,
                target=self.target_symbol,
                timeframe=self.timeframe,
                feature_set_id=self.feature_set_id,
            )
        )


def load_alpha_research_plan(path: str | Path) -> AlphaResearchPlan:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"alpha research config must be a mapping: {path}")
    return AlphaResearchPlan.from_mapping(raw)


def _clean_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        raise KeyError(f"missing required alpha column: {column}")
    return frame[column].replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _threshold_source(values: pd.Series, mode: str) -> pd.Series:
    if mode in {"signed", "inverse", "short_negative"}:
        source = values.abs()
    else:
        source = values[values > 0]
    return source[source > 0].dropna()


def thresholds_for_spec(frame: pd.DataFrame, spec: AlphaSpec) -> tuple[float, ...]:
    values = _clean_series(frame, spec.signal_column, np.nan)
    source = _threshold_source(values, spec.mode)
    if source.empty:
        return ()
    thresholds = [float(source.quantile(q)) for q in spec.threshold_quantiles]
    return tuple(sorted({round(value, 12) for value in thresholds if np.isfinite(value) and value >= 0.0}))


def fit_confirmation_gates(frame: pd.DataFrame, specs: Iterable[AlphaSpec]) -> dict[str, float]:
    gates: dict[str, float] = {}
    for spec in specs:
        for rule in spec.confirmations:
            if not rule.rule_type.endswith("_quantile"):
                continue
            values = _clean_series(frame, rule.column, np.nan).dropna()
            if values.empty:
                gates[rule.gate_key(spec.alpha_id)] = np.nan
                continue
            source = values.abs() if "_abs_" in rule.rule_type else values
            gates[rule.gate_key(spec.alpha_id)] = float(source.quantile(float(rule.quantile)))
    return gates


def _initial_direction(values: pd.Series, mode: str, threshold: float) -> pd.Series:
    direction = pd.Series(0.0, index=values.index)
    if mode == "signed":
        direction.loc[values > threshold] = 1.0
        direction.loc[values < -threshold] = -1.0
    elif mode == "inverse":
        direction.loc[values > threshold] = -1.0
        direction.loc[values < -threshold] = 1.0
    elif mode == "long_positive":
        direction.loc[values > threshold] = 1.0
    elif mode == "short_positive":
        direction.loc[values > threshold] = -1.0
    elif mode == "short_negative":
        direction.loc[values < -threshold] = -1.0
    else:
        raise ValueError(f"unsupported alpha mode: {mode}")
    return direction


def _rule_threshold(rule: ConfirmationRule, spec: AlphaSpec, gates: dict[str, float]) -> float:
    if rule.rule_type.endswith("_quantile"):
        return float(gates.get(rule.gate_key(spec.alpha_id), np.nan))
    if rule.value is None:
        raise ValueError(f"{rule.rule_type} confirmation for {rule.column} requires value")
    return float(rule.value)


def _apply_confirmation(
    active: pd.Series,
    direction: pd.Series,
    frame: pd.DataFrame,
    spec: AlphaSpec,
    rule: ConfirmationRule,
    gates: dict[str, float],
) -> pd.Series:
    values = _clean_series(frame, rule.column)
    rule_type = rule.rule_type
    if rule_type == "same_sign":
        return active & np.sign(values).eq(direction) & values.ne(0.0)
    if rule_type == "opposite_sign":
        return active & np.sign(values).eq(-direction) & values.ne(0.0)

    threshold = _rule_threshold(rule, spec, gates)
    if not np.isfinite(threshold):
        return active & False
    if rule_type in {"min_value", "min_quantile"}:
        return active & values.ge(threshold)
    if rule_type in {"max_value", "max_quantile"}:
        return active & values.le(threshold)
    if rule_type in {"min_abs_value", "min_abs_quantile"}:
        return active & values.abs().ge(threshold)
    if rule_type in {"max_abs_value", "max_abs_quantile"}:
        return active & values.abs().le(threshold)
    raise ValueError(f"unsupported confirmation type: {rule_type}")


def alpha_position(frame: pd.DataFrame, spec: AlphaSpec, threshold: float, gates: dict[str, float] | None = None) -> pd.Series:
    values = _clean_series(frame, spec.signal_column)
    direction = _initial_direction(values, spec.mode, float(threshold))
    active = direction.ne(0.0)
    gate_values = gates or {}
    for rule in spec.confirmations:
        active = _apply_confirmation(active, direction, frame, spec, rule, gate_values)
    return direction.where(active, 0.0).astype(float)

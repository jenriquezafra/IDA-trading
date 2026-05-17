from __future__ import annotations

"""Declarative strategy contracts."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExitRule:
    horizon_bars: int
    force_flat_before: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExitRule":
        horizon = int(raw.get("horizon_bars", 0))
        if horizon <= 0:
            raise ValueError("exit_rule.horizon_bars must be positive")
        force_flat = raw.get("force_flat_before")
        return cls(horizon_bars=horizon, force_flat_before=None if force_flat is None else str(force_flat))


@dataclass(frozen=True)
class PositionRule:
    side: str
    max_gross_exposure: float = 1.0
    sizing: str = "fixed_unit"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PositionRule":
        side = str(raw.get("side", "")).strip()
        if side not in {"long_only", "short_only", "long_short"}:
            raise ValueError(f"unsupported position side: {side}")
        exposure = float(raw.get("max_gross_exposure", 1.0))
        if exposure <= 0.0:
            raise ValueError("position.max_gross_exposure must be positive")
        return cls(side=side, max_gross_exposure=exposure, sizing=str(raw.get("sizing", "fixed_unit")))


@dataclass(frozen=True)
class RiskRule:
    no_new_trades_after: str
    force_flat_before: str
    max_turnover: float
    max_daily_loss: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "RiskRule":
        no_new = str(raw.get("no_new_trades_after", "")).strip()
        flat = str(raw.get("force_flat_before", "")).strip()
        if not no_new or not flat:
            raise ValueError("risk requires no_new_trades_after and force_flat_before")
        max_turnover = float(raw.get("max_turnover", 0.0))
        if max_turnover <= 0.0:
            raise ValueError("risk.max_turnover must be positive")
        max_daily_loss = raw.get("max_daily_loss")
        return cls(
            no_new_trades_after=no_new,
            force_flat_before=flat,
            max_turnover=max_turnover,
            max_daily_loss=None if max_daily_loss is None else float(max_daily_loss),
        )


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    target_symbol: str
    timeframe: str
    feature_set_id: str
    alpha_id: str
    entry_rule: str
    exit_rule: ExitRule
    position: PositionRule
    risk: RiskRule
    cost_profile_id: str
    split_policy_id: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "StrategySpec":
        required = ["strategy_id", "target_symbol", "timeframe", "feature_set_id", "alpha_id", "entry_rule", "cost_profile_id", "split_policy_id"]
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"missing strategy fields: {', '.join(missing)}")
        entry_rule = str(raw["entry_rule"]).strip()
        if entry_rule != "next_open":
            raise ValueError(f"unsupported entry_rule: {entry_rule}")
        return cls(
            strategy_id=str(raw["strategy_id"]).strip(),
            target_symbol=str(raw["target_symbol"]).strip().upper(),
            timeframe=str(raw["timeframe"]).strip(),
            feature_set_id=str(raw["feature_set_id"]).strip(),
            alpha_id=str(raw["alpha_id"]).strip(),
            entry_rule=entry_rule,
            exit_rule=ExitRule.from_mapping(dict(raw.get("exit_rule", {}))),
            position=PositionRule.from_mapping(dict(raw.get("position", {}))),
            risk=RiskRule.from_mapping(dict(raw.get("risk", {}))),
            cost_profile_id=str(raw["cost_profile_id"]).strip(),
            split_policy_id=str(raw["split_policy_id"]).strip(),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategySpec":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"strategy spec must be a mapping: {path}")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostBreakdown:
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps + self.impact_bps

    @property
    def total_return(self) -> float:
        return self.total_bps / 10_000.0


def scenario_cost_config(config: dict[str, Any], scenario: str | None = None) -> dict[str, float]:
    scenario_name = scenario or config.get("backtest", {}).get("cost_scenario", "base")
    try:
        return config["costs"][scenario_name]
    except KeyError as exc:
        raise ValueError(f"Unknown cost scenario: {scenario_name}") from exc


def per_side_cost(
    config: dict[str, Any],
    scenario: str | None = None,
    participation_rate: float = 0.0,
) -> CostBreakdown:
    cost_cfg = scenario_cost_config(config, scenario)
    impact_bps = float(cost_cfg.get("impact_bps_per_1pct_participation", 0.0)) * (participation_rate / 0.01)
    return CostBreakdown(
        commission_bps=float(cost_cfg.get("commission_bps_per_side", 0.0)),
        spread_bps=float(cost_cfg.get("spread_bps_per_side", 0.0)),
        slippage_bps=float(cost_cfg.get("slippage_bps_per_side", 0.0)),
        impact_bps=impact_bps,
    )


def round_trip_cost(
    config: dict[str, Any],
    scenario: str | None = None,
    entry_participation_rate: float = 0.0,
    exit_participation_rate: float = 0.0,
) -> CostBreakdown:
    entry = per_side_cost(config, scenario, entry_participation_rate)
    exit_ = per_side_cost(config, scenario, exit_participation_rate)
    return CostBreakdown(
        commission_bps=entry.commission_bps + exit_.commission_bps,
        spread_bps=entry.spread_bps + exit_.spread_bps,
        slippage_bps=entry.slippage_bps + exit_.slippage_bps,
        impact_bps=entry.impact_bps + exit_.impact_bps,
    )


def gross_return(position: float, entry_px: float, exit_px: float) -> float:
    if entry_px <= 0 or exit_px <= 0:
        raise ValueError("entry_px and exit_px must be positive")
    return position * ((exit_px / entry_px) - 1.0)


def net_return(
    position: float,
    entry_px: float,
    exit_px: float,
    config: dict[str, Any],
    scenario: str | None = None,
    entry_participation_rate: float = 0.0,
    exit_participation_rate: float = 0.0,
) -> tuple[float, float, CostBreakdown]:
    gross = gross_return(position, entry_px, exit_px)
    cost = round_trip_cost(config, scenario, entry_participation_rate, exit_participation_rate)
    net = gross - abs(position) * cost.total_return
    return gross, net, cost

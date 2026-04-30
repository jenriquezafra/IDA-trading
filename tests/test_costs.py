from __future__ import annotations

import pytest

from src.costs import gross_return, net_return, per_side_cost, round_trip_cost


def _config() -> dict:
    return {
        "backtest": {"cost_scenario": "base"},
        "costs": {
            "base": {
                "commission_bps_per_side": 0.1,
                "spread_bps_per_side": 0.2,
                "slippage_bps_per_side": 0.3,
                "impact_bps_per_1pct_participation": 0.4,
            },
            "stress": {
                "commission_bps_per_side": 1.0,
                "spread_bps_per_side": 1.0,
                "slippage_bps_per_side": 1.0,
                "impact_bps_per_1pct_participation": 0.0,
            },
        },
    }


def test_per_side_and_round_trip_cost_breakdown() -> None:
    per_side = per_side_cost(_config(), participation_rate=0.02)
    round_trip = round_trip_cost(_config(), entry_participation_rate=0.02, exit_participation_rate=0.0)

    assert per_side.total_bps == pytest.approx(1.4)
    assert round_trip.commission_bps == pytest.approx(0.2)
    assert round_trip.spread_bps == pytest.approx(0.4)
    assert round_trip.slippage_bps == pytest.approx(0.6)
    assert round_trip.impact_bps == pytest.approx(0.8)
    assert round_trip.total_bps == pytest.approx(2.0)


def test_gross_and_net_return_for_long_and_short() -> None:
    config = _config()

    assert gross_return(1.0, 100.0, 101.0) == pytest.approx(0.01)
    assert gross_return(-1.0, 100.0, 101.0) == pytest.approx(-0.01)

    gross, net, cost = net_return(1.0, 100.0, 101.0, config)

    assert gross == pytest.approx(0.01)
    assert cost.total_bps == pytest.approx(1.2)
    assert net == pytest.approx(0.01 - 0.00012)

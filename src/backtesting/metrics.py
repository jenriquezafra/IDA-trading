from __future__ import annotations

"""Backtesting metrics shared by alpha research and strategy validation."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    rows: int
    trades: int
    exposure: float
    turnover: float
    gross_return: float
    total_cost: float
    net_return: float
    avg_trade_net: float
    profit_factor: float
    daily_sharpe: float
    max_drawdown: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_sharpe(daily_net: pd.Series) -> float:
    clean = daily_net.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(252.0) * clean.mean() / std)


def _max_drawdown(net_returns: pd.Series) -> float:
    equity = net_returns.fillna(0.0).cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max()) if not drawdown.empty else 0.0


def evaluate_positions(
    frame: pd.DataFrame,
    position: pd.Series,
    *,
    return_column: str,
    cost_bps: float,
    session_column: str = "session",
) -> BacktestMetrics:
    """Evaluate a simple one-bar vector signal.

    This is deliberately small. It is the common metric contract for alpha
    research; richer execution simulation should live in `ida.execution`.
    """

    if return_column not in frame:
        raise KeyError(f"missing return column: {return_column}")
    aligned_position = position.reindex(frame.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
    returns = frame[return_column].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    turnover_series = aligned_position.diff().abs().fillna(aligned_position.abs())
    gross = aligned_position * returns
    costs = turnover_series * (float(cost_bps) / 10_000.0)
    net = gross - costs
    active_trades = int(aligned_position.ne(0.0).sum())
    positive = float(net[net > 0.0].sum())
    negative = float(-net[net < 0.0].sum())
    profit_factor = positive / negative if negative > 0.0 else (np.inf if positive > 0.0 else 0.0)
    if session_column in frame:
        daily_net = net.groupby(frame[session_column]).sum()
    else:
        daily_net = net
    return BacktestMetrics(
        rows=int(len(frame)),
        trades=active_trades,
        exposure=float(aligned_position.abs().mean()) if len(aligned_position) else 0.0,
        turnover=float(turnover_series.sum()),
        gross_return=float(gross.sum()),
        total_cost=float(costs.sum()),
        net_return=float(net.sum()),
        avg_trade_net=float(net[aligned_position.ne(0.0)].mean()) if active_trades else 0.0,
        profit_factor=float(profit_factor),
        daily_sharpe=_safe_sharpe(daily_net),
        max_drawdown=_max_drawdown(net),
    )

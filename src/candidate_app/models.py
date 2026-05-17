from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CandidateStatus = str

VALID_STATUSES: tuple[CandidateStatus, ...] = (
    "candidate",
    "paper_trading",
    "rejected",
    "archived",
)

VALID_LEDGER_EVENT_TYPES: tuple[str, ...] = (
    "signal",
    "order_planned",
    "order_submitted",
    "fill",
    "mark",
    "fee",
    "adjustment",
    "note",
)

REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "cagr",
    "annualized_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "volatility",
    "win_rate",
    "profit_factor",
    "trade_count",
    "turnover",
    "estimated_costs_bps",
    "estimated_slippage_bps",
    "backtest_period_start",
    "backtest_period_end",
    "last_evaluated_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_status(status: str) -> CandidateStatus:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid candidate status: {status}")
    return status


def validate_ledger_event_type(event_type: str) -> str:
    if event_type not in VALID_LEDGER_EVENT_TYPES:
        raise ValueError(f"invalid paper ledger event type: {event_type}")
    return event_type


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(inner) for inner in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    normalized = {key: None for key in REQUIRED_METRIC_KEYS}
    normalized.update(json_safe(metrics or {}))
    return normalized


def metric_value(candidate: dict[str, Any], key: str, default: Any = None) -> Any:
    metrics = candidate.get("metrics") or {}
    return metrics.get(key, default)


@dataclass(frozen=True)
class CandidateStrategy:
    id: str
    name: str
    strategy_type: str
    asset_universe: list[str]
    status: CandidateStatus = "candidate"
    created_at: str = field(default_factory=utc_now)
    promoted_at: str | None = None
    description: str = ""
    backtest_summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    paper_trading_metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    drawdown_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_status(self.status)
        object.__setattr__(self, "asset_universe", [str(asset).upper() for asset in self.asset_universe])
        object.__setattr__(self, "metrics", normalize_metrics(self.metrics))
        object.__setattr__(self, "paper_trading_metrics", json_safe(self.paper_trading_metrics or {}))
        object.__setattr__(self, "backtest_summary", json_safe(self.backtest_summary or {}))
        object.__setattr__(self, "notes", list(self.notes or []))
        object.__setattr__(self, "equity_curve", json_safe(self.equity_curve or []))
        object.__setattr__(self, "drawdown_curve", json_safe(self.drawdown_curve or []))
        object.__setattr__(self, "trades", json_safe(self.trades or []))

    def to_record(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class AuditLogEntry:
    event_id: str
    candidate_id: str
    changed_at: str
    actor: str
    from_status: str | None
    to_status: str
    reason: str = ""
    note: str = ""

    def to_record(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class PaperLedgerEntry:
    entry_id: str
    candidate_id: str
    event_at: str
    event_type: str
    strategy_run_id: str | None = None
    symbol: str | None = None
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    gross_pnl: float | None = None
    fees: float | None = None
    slippage_bps: float | None = None
    net_pnl: float | None = None
    exposure: float | None = None
    currency: str = "USD"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        validate_ledger_event_type(self.event_type)
        object.__setattr__(self, "symbol", self.symbol.upper() if self.symbol else None)
        object.__setattr__(self, "side", self.side.lower() if self.side else None)
        object.__setattr__(self, "currency", (self.currency or "USD").upper())
        object.__setattr__(self, "metadata", json_safe(self.metadata or {}))

    def to_record(self) -> dict[str, Any]:
        return json_safe(asdict(self))

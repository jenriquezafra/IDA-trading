from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.execution.ibkr_read_only import (
    IBKRReadOnlyClient,
    IBKRReadOnlyConfig,
    account_summary_frame,
    load_ibkr_read_only_config,
    positions_frame,
    write_snapshot,
)


def _config(**overrides: object) -> IBKRReadOnlyConfig:
    raw = {
        "connection": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 71,
            "timeout_seconds": 1,
            "trading_mode": "paper",
            "expected_account": "DU123",
        },
        "safety": {
            "read_only": True,
            "allow_orders": False,
            "require_paper_account": True,
            "require_paper_port": True,
        },
        "outputs": {"output_dir": "results/paper/ibkr_read_only"},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        raw[section][key] = value
    return IBKRReadOnlyConfig.from_mapping(raw)


class FakeIB:
    def __init__(self) -> None:
        self.connected = False
        self.connect_kwargs = {}

    def connect(self, host: str, port: int, *, clientId: int, timeout: float, readonly: bool) -> None:
        self.connected = True
        self.connect_kwargs = {
            "host": host,
            "port": port,
            "clientId": clientId,
            "timeout": timeout,
            "readonly": readonly,
        }

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def managedAccounts(self) -> list[str]:
        return ["DU123"]

    def reqCurrentTime(self) -> datetime:
        return datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)

    def accountSummary(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(account="DU123", tag="NetLiquidation", value="1000000", currency="USD", modelCode="")]

    def positions(self) -> list[SimpleNamespace]:
        contract = SimpleNamespace(
            conId=320227571,
            symbol="QQQ",
            secType="STK",
            exchange="SMART",
            primaryExchange="NASDAQ",
            currency="USD",
            localSymbol="QQQ",
            tradingClass="QQQ",
        )
        return [SimpleNamespace(account="DU123", contract=contract, position=-10, avgCost=450.25)]

    def openTrades(self) -> list[SimpleNamespace]:
        return []


def test_read_only_config_rejects_live_or_order_enabled() -> None:
    with pytest.raises(ValueError, match="trading_mode=paper"):
        _config(connection__trading_mode="live")
    with pytest.raises(ValueError, match="allow_orders=false"):
        _config(safety__allow_orders=True)
    with pytest.raises(ValueError, match="paper mode"):
        _config(connection__port=7496)


def test_load_default_config_is_paper_gateway_read_only() -> None:
    config = load_ibkr_read_only_config()

    assert config.port == 4002
    assert config.trading_mode == "paper"
    assert config.read_only is True
    assert config.allow_orders is False


def test_serializers_keep_account_and_position_fields() -> None:
    account = account_summary_frame([SimpleNamespace(account="DU123", tag="NetLiquidation", value="1", currency="USD", modelCode="")])
    positions = positions_frame(
        [
            SimpleNamespace(
                account="DU123",
                contract=SimpleNamespace(symbol="QQQ", secType="STK", exchange="SMART", currency="USD", conId=1),
                position=-3,
                avgCost=100.5,
            )
        ]
    )

    assert account.iloc[0]["tag"] == "NetLiquidation"
    assert positions.iloc[0]["symbol"] == "QQQ"
    assert positions.iloc[0]["position"] == -3.0


def test_client_uses_readonly_connection_and_writes_snapshot(tmp_path) -> None:
    fake = FakeIB()
    client = IBKRReadOnlyClient(_config(), ib_factory=lambda: fake)

    client.connect()
    snapshot = client.snapshot()
    paths = write_snapshot(snapshot, client.config, output_dir=tmp_path)

    assert fake.connect_kwargs["readonly"] is True
    assert snapshot["health"]["managed_accounts"] == ["DU123"]
    assert paths.manifest_path.exists()
    assert paths.positions_path.exists()
    assert paths.report_path.exists()

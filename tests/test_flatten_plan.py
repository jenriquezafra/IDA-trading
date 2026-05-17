from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.execution.flatten_plan import ExecutionPolicy, build_flatten_plan, create_flatten_plan, find_latest_snapshot


def _positions(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "account",
        "con_id",
        "symbol",
        "sec_type",
        "exchange",
        "primary_exchange",
        "currency",
        "local_symbol",
        "trading_class",
        "position",
        "avg_cost",
    ]
    return pd.DataFrame(rows, columns=columns)


def _open_trades(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=["symbol", "status"])


def _write_config(path: Path, expected_account: str = "DU123") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "connection": {
                    "host": "127.0.0.1",
                    "port": 4002,
                    "client_id": 71,
                    "timeout_seconds": 1,
                    "trading_mode": "paper",
                    "expected_account": expected_account,
                },
                "safety": {
                    "read_only": True,
                    "allow_orders": False,
                    "require_paper_account": True,
                    "require_paper_port": True,
                },
                "outputs": {"output_dir": "unused"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_snapshot(root: Path, name: str, positions: pd.DataFrame, open_trades: pd.DataFrame, account: str = "DU123") -> Path:
    snapshot_dir = root / name
    snapshot_dir.mkdir(parents=True)
    positions.to_parquet(snapshot_dir / "positions.parquet", index=False)
    open_trades.to_parquet(snapshot_dir / "open_trades.parquet", index=False)
    (snapshot_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"created_at_utc": "2026-05-10T12:00:00Z"},
                "health": {"managed_accounts": [account]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return snapshot_dir


def test_build_flatten_plan_reverses_long_and_short_stock_positions() -> None:
    positions = _positions(
        [
            {
                "account": "DU123",
                "con_id": 1,
                "symbol": "AAPL",
                "sec_type": "STK",
                "exchange": "NASDAQ",
                "primary_exchange": "NASDAQ",
                "currency": "USD",
                "local_symbol": "AAPL",
                "trading_class": "NMS",
                "position": 3,
                "avg_cost": 100,
            },
            {
                "account": "DU123",
                "con_id": 2,
                "symbol": "QQQ",
                "sec_type": "STK",
                "exchange": "NASDAQ",
                "primary_exchange": "NASDAQ",
                "currency": "USD",
                "local_symbol": "QQQ",
                "trading_class": "NMS",
                "position": -2,
                "avg_cost": 400,
            },
        ]
    )

    orders, skipped, summary = build_flatten_plan(positions, _open_trades(), expected_account="DU123")

    assert summary["decision"] == "ready_for_review"
    assert skipped.empty
    assert orders[["symbol", "action", "quantity", "transmit", "dry_run"]].to_dict("records") == [
        {"symbol": "AAPL", "action": "SELL", "quantity": 3.0, "transmit": False, "dry_run": True},
        {"symbol": "QQQ", "action": "BUY", "quantity": 2.0, "transmit": False, "dry_run": True},
    ]
    assert set(orders["tif"]) == {"DAY"}
    assert set(orders["outside_rth"]) == {False}


def test_build_flatten_plan_accepts_market_on_open_policy() -> None:
    positions = _positions(
        [
            {
                "account": "DU123",
                "con_id": 1,
                "symbol": "AAPL",
                "sec_type": "STK",
                "exchange": "NASDAQ",
                "primary_exchange": "NASDAQ",
                "currency": "USD",
                "local_symbol": "AAPL",
                "trading_class": "NMS",
                "position": 3,
                "avg_cost": 100,
            }
        ]
    )

    orders, _, summary = build_flatten_plan(
        positions,
        _open_trades(),
        expected_account="DU123",
        execution_policy=ExecutionPolicy(order_type="MKT", tif="OPG", outside_rth=False, intent="flatten_at_next_open"),
    )

    assert summary["execution_policy"]["tif"] == "OPG"
    assert orders.iloc[0]["order_type"] == "MKT"
    assert orders.iloc[0]["tif"] == "OPG"
    assert bool(orders.iloc[0]["outside_rth"]) is False


def test_build_flatten_plan_blocks_when_open_trades_exist() -> None:
    positions = _positions(
        [
            {
                "account": "DU123",
                "con_id": 1,
                "symbol": "AAPL",
                "sec_type": "STK",
                "exchange": "NASDAQ",
                "primary_exchange": "NASDAQ",
                "currency": "USD",
                "local_symbol": "AAPL",
                "trading_class": "NMS",
                "position": 3,
                "avg_cost": 100,
            }
        ]
    )

    orders, _, summary = build_flatten_plan(positions, _open_trades([{"symbol": "AAPL", "status": "Submitted"}]))

    assert summary["decision"] == "blocked_open_trades"
    assert orders.iloc[0]["status"] == "blocked_open_trades"


def test_build_flatten_plan_skips_unsupported_instruments() -> None:
    positions = _positions(
        [
            {
                "account": "DU123",
                "con_id": 1,
                "symbol": "AAPL  260515C00100000",
                "sec_type": "OPT",
                "exchange": "SMART",
                "primary_exchange": "",
                "currency": "USD",
                "local_symbol": "AAPL  260515C00100000",
                "trading_class": "AAPL",
                "position": 1,
                "avg_cost": 2.5,
            }
        ]
    )

    orders, skipped, summary = build_flatten_plan(positions, _open_trades())

    assert orders.empty
    assert summary["decision"] == "blocked_unsupported_instruments"
    assert skipped.iloc[0]["skip_reason"] == "unsupported_sec_type"


def test_create_flatten_plan_uses_latest_snapshot_and_writes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "ibkr.yaml"
    _write_config(config_path)
    snapshot_root = tmp_path / "snapshots"
    _write_snapshot(snapshot_root, "20260510T120000Z", _positions([]), _open_trades())
    latest = _write_snapshot(
        snapshot_root,
        "20260510T130000Z",
        _positions(
            [
                {
                    "account": "DU123",
                    "con_id": 1,
                    "symbol": "MSFT",
                    "sec_type": "STK",
                    "exchange": "NASDAQ",
                    "primary_exchange": "NASDAQ",
                    "currency": "USD",
                    "local_symbol": "MSFT",
                    "trading_class": "NMS",
                    "position": 5,
                    "avg_cost": 300,
                }
            ]
        ),
        _open_trades(),
    )

    assert find_latest_snapshot(snapshot_root) == latest

    paths, summary = create_flatten_plan(snapshot_root=snapshot_root, config_path=config_path, output_dir=tmp_path / "plans")

    assert summary["decision"] == "ready_for_review"
    assert paths.manifest_path.exists()
    assert paths.orders_path.exists()
    assert paths.orders_csv_path.exists()
    assert pd.read_parquet(paths.orders_path).iloc[0]["symbol"] == "MSFT"


def test_create_flatten_plan_rejects_snapshot_for_wrong_account(tmp_path: Path) -> None:
    config_path = tmp_path / "ibkr.yaml"
    _write_config(config_path, expected_account="DU999")
    snapshot_root = tmp_path / "snapshots"
    _write_snapshot(snapshot_root, "20260510T130000Z", _positions([]), _open_trades(), account="DU123")

    with pytest.raises(ValueError, match="expected account"):
        create_flatten_plan(snapshot_root=snapshot_root, config_path=config_path, output_dir=tmp_path / "plans")

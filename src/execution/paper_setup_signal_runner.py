from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.hmm_lab import features_input_path, load_yaml
from src.research.manifest import fingerprint_path
from src.setup_signal_search import signal_mask


DEFAULT_CONFIG_PATH = Path("configs/execution/c2_setup_signal_runner.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/c2_signal_runner")


@dataclass(frozen=True)
class PaperSetupSignalConfig:
    mode: str
    candidate_id: str
    strategy_id: str
    max_data_staleness_days: int
    lifecycle_config_path: Path
    features_path: Path | None
    target_account: str
    target_symbol: str
    unit_size: float
    order_type: str
    time_in_force: str
    execution_timing: str
    send_orders: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PaperSetupSignalConfig":
        runner = dict(raw.get("runner", {}))
        strategy = dict(raw.get("strategy", {}))
        data = dict(raw.get("data", {}))
        paper = dict(raw.get("paper", {}))
        outputs = dict(raw.get("outputs", {}))
        features_path = data.get("features_path")
        config = cls(
            mode=str(runner.get("mode", "signal_only")).strip(),
            candidate_id=str(runner.get("candidate_id", "")).strip(),
            strategy_id=str(strategy.get("strategy_id", "")).strip(),
            max_data_staleness_days=int(runner.get("max_data_staleness_days", 3)),
            lifecycle_config_path=Path(strategy.get("lifecycle_config_path", "configs/setup_signal_portfolio_lifecycle_c2_googl_5min_monthly.yaml")),
            features_path=None if features_path in {None, ""} else Path(features_path),
            target_account=str(paper.get("target_account", "")).strip(),
            target_symbol=str(paper.get("target_symbol", "GOOGL")).strip().upper(),
            unit_size=float(paper.get("unit_size", 1.0)),
            order_type=str(paper.get("order_type", "MKT")).strip().upper(),
            time_in_force=str(paper.get("time_in_force", "DAY")).strip().upper(),
            execution_timing=str(paper.get("execution_timing", "next_bar_open_simulated")).strip(),
            send_orders=bool(paper.get("send_orders", False)),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode != "signal_only":
            raise ValueError("setup-signal paper runner currently supports mode=signal_only only")
        if not self.candidate_id:
            raise ValueError("runner.candidate_id is required")
        if not self.strategy_id:
            raise ValueError("strategy.strategy_id is required")
        if self.max_data_staleness_days < 0:
            raise ValueError("max_data_staleness_days must be non-negative")
        if self.unit_size <= 0.0:
            raise ValueError("unit_size must be positive")
        if self.send_orders:
            raise ValueError("signal-only runner requires paper.send_orders=false")
        if self.order_type != "MKT":
            raise ValueError("signal-only runner currently supports paper.order_type=MKT only")
        if self.time_in_force != "DAY":
            raise ValueError("signal-only runner currently supports paper.time_in_force=DAY only")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lifecycle_config_path"] = self.lifecycle_config_path.as_posix()
        data["features_path"] = None if self.features_path is None else self.features_path.as_posix()
        data["output_dir"] = self.output_dir.as_posix()
        return data


@dataclass(frozen=True)
class PaperSetupSignalRunnerPaths:
    output_dir: Path
    signals_path: Path
    latest_signal_path: Path
    ticket_path: Path
    manifest_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_runner_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperSetupSignalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return PaperSetupSignalConfig.from_mapping(raw)


def lifecycle_rule(config: dict[str, Any]) -> dict[str, Any]:
    lifecycle = dict(config.get("setup_signal_portfolio_lifecycle", {}) or {})
    raw = lifecycle.get("rule")
    if raw is None:
        rules = lifecycle.get("rules") or config.get("setup_signal_fixed_rules", {}).get("rules", [])
        if not rules:
            raise ValueError("setup_signal_portfolio_lifecycle requires a rule")
        raw = rules[0]
    params = {str(key): value for key, value in dict(raw.get("params", {}) or {}).items()}
    direction = str(raw.get("direction", params.get("direction", "long")))
    params["direction"] = direction
    return {
        "rule_name": str(raw["name"]),
        "family": str(raw["family"]),
        "direction": direction,
        "params": params,
        "column_map": {str(key): str(value) for key, value in dict(raw.get("column_map", {}) or {}).items()},
    }


def lifecycle_exit_params(config: dict[str, Any]) -> dict[str, Any]:
    grid = dict(config.get("setup_signal_portfolio_lifecycle", {}).get("exit_grid", {}) or {})
    return {
        "max_hold_bars": int((grid.get("max_hold_bars") or [24])[0]),
        "min_hold_bars": int((grid.get("min_hold_bars") or [1])[0]),
        "stop_loss_bps": float((grid.get("stop_loss_bps") or [0.0])[0] or 0.0),
        "take_profit_bps": float((grid.get("take_profit_bps") or [0.0])[0] or 0.0),
        "exit_on_signal_loss": bool((grid.get("exit_on_signal_loss") or [False])[0]),
        "cooldown_bars": int((grid.get("cooldown_bars") or [0])[0]),
    }


def feature_path_from_lifecycle(config: dict[str, Any], target_symbol: str) -> Path:
    feature_config = load_yaml(Path(config.get("hmm_lab", {}).get("features_config", "configs/features/cross_asset_v1.yaml")))
    return features_input_path(config, target_symbol, feature_config)


def load_features_for_runner(config: PaperSetupSignalConfig, lifecycle: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    path = config.features_path or feature_path_from_lifecycle(lifecycle, config.target_symbol)
    frame = pd.read_parquet(path).sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)
    return frame, path


def _exit_row_for_signal(session_frame: pd.DataFrame, loc: int, max_hold_bars: int) -> pd.Series | None:
    eligible = session_frame.loc[
        session_frame.index > loc,
        ["target_next_open_timestamp", "target_open_next", "target_crosses_session_close"],
    ].copy()
    eligible = eligible[eligible["target_next_open_timestamp"].notna() & eligible["target_open_next"].notna()]
    eligible = eligible[~eligible.get("target_crosses_session_close", False).fillna(False).astype(bool)]
    if eligible.empty:
        return None
    target_index = min(loc + max(1, int(max_hold_bars)), int(eligible.index.max()))
    future = eligible.loc[eligible.index <= target_index]
    if future.empty:
        return None
    return future.iloc[-1]


def evaluate_setup_signal(frame: pd.DataFrame, rule: dict[str, Any], exit_params: dict[str, Any]) -> pd.DataFrame:
    required = {
        "timestamp",
        "session",
        "bar_index",
        "target_open_next",
        "target_next_open_timestamp",
        "target_can_open_trade",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"setup-signal input frame missing columns: {missing}")

    out = frame.copy()
    base_signal = signal_mask(out, rule["family"], rule["params"], rule.get("column_map") or {})
    out["setup_rule_signal"] = base_signal.fillna(False).astype(bool)
    out["setup_can_enter"] = out["target_can_open_trade"].fillna(False).astype(bool) & out["target_open_next"].notna()
    out["setup_signal_active"] = out["setup_rule_signal"] & out["setup_can_enter"]
    out["setup_theoretical_exit_timestamp"] = None
    out["setup_theoretical_exit_price"] = math.nan
    out["setup_exit_available"] = False

    for _, session_frame in out.groupby("session", sort=False):
        for loc in session_frame.index:
            if not bool(out.at[loc, "setup_signal_active"]):
                continue
            exit_row = _exit_row_for_signal(session_frame, int(loc), int(exit_params["max_hold_bars"]))
            if exit_row is None:
                continue
            out.at[loc, "setup_theoretical_exit_timestamp"] = exit_row["target_next_open_timestamp"]
            out.at[loc, "setup_theoretical_exit_price"] = float(exit_row["target_open_next"])
            out.at[loc, "setup_exit_available"] = True

    out["setup_signal_active"] = out["setup_signal_active"] & out["setup_exit_available"]
    direction = str(rule["direction"])
    out["desired_position_unit"] = out["setup_signal_active"].map({True: 1.0 if direction == "long" else -1.0, False: 0.0}).astype(float)
    columns = [
        "timestamp",
        "session",
        "bar_index",
        "target_next_open_timestamp",
        "target_open_next",
        "target_can_open_trade",
        "target_rel_volume_by_bar",
        "target_close_location_bar",
        "target_dist_vwap_atr",
        "target_breakout_attempt_count_or_6_high",
        "target_breakout_attempt_count_or_6_low",
        "target_minutes_from_open",
        "target_minutes_to_close",
        "setup_rule_signal",
        "setup_can_enter",
        "setup_exit_available",
        "setup_signal_active",
        "desired_position_unit",
        "setup_theoretical_exit_timestamp",
        "setup_theoretical_exit_price",
    ]
    return out[[column for column in columns if column in out.columns]].copy()


def select_asof_signal(signals: pd.DataFrame, as_of: str | None = None) -> pd.Series:
    if signals.empty:
        raise ValueError("no setup-signal rows available")
    ordered = signals.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)
    if as_of:
        asof_ts = pd.Timestamp(as_of)
        timestamps = pd.to_datetime(ordered["timestamp"])
        if timestamps.dt.tz is not None and asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize(timestamps.dt.tz)
        elif timestamps.dt.tz is not None and asof_ts.tzinfo is not None:
            asof_ts = asof_ts.tz_convert(timestamps.dt.tz)
        eligible = ordered[timestamps <= asof_ts]
        if eligible.empty:
            raise ValueError(f"no setup-signal rows at or before as_of={as_of}")
        return eligible.iloc[-1]
    return ordered.iloc[-1]


def build_signal_ticket(latest: pd.Series, config: PaperSetupSignalConfig, rule: dict[str, Any], exit_params: dict[str, Any]) -> dict[str, Any]:
    direction = str(rule["direction"])
    signal = bool(latest["setup_signal_active"])
    action = "BUY" if signal and direction == "long" else ("SELL" if signal else "NONE")
    entry_timestamp = latest["target_next_open_timestamp"]
    entry_price = latest["target_open_next"]
    exit_timestamp = latest["setup_theoretical_exit_timestamp"]
    exit_price = latest["setup_theoretical_exit_price"]
    return {
        "mode": config.mode,
        "send_orders": False,
        "strategy_id": config.strategy_id,
        "candidate_id": config.candidate_id,
        "account": config.target_account,
        "symbol": config.target_symbol,
        "signal_timestamp": str(latest["timestamp"]),
        "session": str(latest["session"]),
        "bar_index": int(latest["bar_index"]),
        "entry_rule": rule["rule_name"],
        "exit_rule": "setup_signal_lifecycle",
        "horizon_bars": int(exit_params["max_hold_bars"]),
        "min_hold_bars": int(exit_params["min_hold_bars"]),
        "stop_loss_bps": float(exit_params["stop_loss_bps"]),
        "take_profit_bps": float(exit_params["take_profit_bps"]),
        "execution_timing": config.execution_timing,
        "theoretical_entry_timestamp": None if pd.isna(entry_timestamp) else str(entry_timestamp),
        "theoretical_entry_price": None if pd.isna(entry_price) else float(entry_price),
        "theoretical_exit_timestamp": None if pd.isna(exit_timestamp) else str(exit_timestamp),
        "theoretical_exit_price": None if pd.isna(exit_price) else float(exit_price),
        "desired_position_unit": float(latest["desired_position_unit"]),
        "action": action,
        "quantity": float(config.unit_size) if signal else 0.0,
        "order_type": config.order_type,
        "time_in_force": config.time_in_force,
        "status": "paper_ticket_only" if signal else "no_signal",
        "reason": f"{config.candidate_id} {direction} setup signal active" if signal else f"{config.candidate_id} setup conditions not all true or entry unavailable",
    }


def data_staleness_warning(latest: pd.Series, max_days: int) -> str:
    latest_session = pd.Timestamp(str(latest["session"]))
    now_session = pd.Timestamp(datetime.now(timezone.utc).date())
    age_days = int((now_session - latest_session).days)
    if age_days > max_days:
        return f"latest feature session {latest_session.date()} is {age_days} days old; refresh intraday data before paper decisions"
    return ""


def _yaml_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(path: Path, *, config: PaperSetupSignalConfig, rule: dict[str, Any], exit_params: dict[str, Any], latest: pd.Series, ticket: dict[str, Any], warnings: list[str]) -> None:
    lines = [
        f"# {config.candidate_id} paper setup-signal runner",
        "",
        f"- Mode: `{config.mode}`",
        f"- Strategy: `{ticket['strategy_id']}`",
        f"- Account: `{config.target_account}`",
        f"- Symbol: `{config.target_symbol}`",
        f"- Rule: `{rule['rule_name']}`",
        f"- Direction: `{rule['direction']}`",
        f"- Latest signal timestamp: `{ticket['signal_timestamp']}`",
        f"- Theoretical entry timestamp: `{ticket['theoretical_entry_timestamp']}`",
        f"- Theoretical exit timestamp: `{ticket['theoretical_exit_timestamp']}`",
        f"- Signal active: `{bool(latest['setup_signal_active'])}`",
        f"- Action: `{ticket['action']}`",
        f"- Quantity: `{ticket['quantity']}`",
        f"- Send orders: `{ticket['send_orders']}`",
        f"- Max hold bars: `{exit_params['max_hold_bars']}`",
        f"- Take profit bps: `{exit_params['take_profit_bps']}`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    lines.append("This runner is signal-only. It does not connect to IBKR and does not submit orders.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_setup_signal_runner(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    as_of: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PaperSetupSignalRunnerPaths, dict[str, Any]]:
    config = load_runner_config(config_path)
    lifecycle = load_yaml(config.lifecycle_config_path)
    candidate = dict(lifecycle.get("candidate", {}) or {})
    if candidate and str(candidate.get("candidate_id", "")) != config.candidate_id:
        raise ValueError(f"lifecycle candidate_id {candidate.get('candidate_id')} does not match runner candidate_id {config.candidate_id}")
    rule = lifecycle_rule(lifecycle)
    if rule["direction"] != "long":
        raise ValueError("paper setup-signal runner currently expects a long C2-style rule")
    exit_params = lifecycle_exit_params(lifecycle)
    features, features_path = load_features_for_runner(config, lifecycle)
    signals = evaluate_setup_signal(features, rule, exit_params)
    latest = select_asof_signal(signals, as_of=as_of)
    ticket = build_signal_ticket(latest, config, rule, exit_params)
    warnings = [warning for warning in [data_staleness_warning(latest, config.max_data_staleness_days)] if warning]

    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = PaperSetupSignalRunnerPaths(
        output_dir=root,
        signals_path=root / "signals.parquet",
        latest_signal_path=root / "latest_signal.yaml",
        ticket_path=root / "paper_ticket.yaml",
        manifest_path=root / "manifest.yaml",
        report_path=root / "report.md",
    )
    root.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(paths.signals_path, index=False)
    latest_payload = {key: _yaml_scalar(value) for key, value in latest.to_dict().items()}
    paths.latest_signal_path.write_text(yaml.safe_dump(latest_payload, sort_keys=False), encoding="utf-8")
    paths.ticket_path.write_text(yaml.safe_dump(ticket, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "setup_signal_paper_runner",
            "candidate_id": config.candidate_id,
            "created_at_utc": created,
            "status": "signal_only",
        },
        "config": config.to_dict(),
        "candidate": candidate,
        "rule": rule,
        "exit_params": exit_params,
        "latest": {
            "timestamp": ticket["signal_timestamp"],
            "session": ticket["session"],
            "signal_active": bool(latest["setup_signal_active"]),
            "action": ticket["action"],
        },
        "warnings": warnings,
        "data": {
            "features_path": features_path.as_posix(),
            "features_fingerprint": fingerprint_path(features_path) if features_path.exists() else "MISSING",
            "lifecycle_config_path": config.lifecycle_config_path.as_posix(),
            "lifecycle_config_fingerprint": fingerprint_path(config.lifecycle_config_path) if config.lifecycle_config_path.exists() else "MISSING",
        },
        "outputs": {
            "signals": paths.signals_path.as_posix(),
            "latest_signal": paths.latest_signal_path.as_posix(),
            "paper_ticket": paths.ticket_path.as_posix(),
            "report": paths.report_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, config=config, rule=rule, exit_params=exit_params, latest=latest, ticket=ticket, warnings=warnings)
    return paths, {"ticket": ticket, "rule": rule, "exit_params": exit_params, "warnings": warnings}


def latest_operational_price(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    as_of: str | None = None,
) -> dict[str, Any]:
    config = load_runner_config(config_path)
    lifecycle = load_yaml(config.lifecycle_config_path)
    features, features_path = load_features_for_runner(config, lifecycle)
    latest = select_asof_signal(features.loc[:, ["timestamp", "session", "bar_index", "target_open_next", "target_next_open_timestamp"]].copy(), as_of=as_of)
    price = latest.get("target_open_next")
    return {
        "features_path": features_path.as_posix(),
        "timestamp": str(latest.get("timestamp")),
        "session": str(latest.get("session")),
        "bar_index": int(latest.get("bar_index")),
        "price": None if pd.isna(price) else float(price),
        "next_open_timestamp": None if pd.isna(latest.get("target_next_open_timestamp")) else str(latest.get("target_next_open_timestamp")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a C2/setup-signal paper signal runner without submitting orders")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--as-of", default=None, help="optional timestamp cutoff for selecting latest signal row")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, summary = run_setup_signal_runner(config_path=args.config, as_of=args.as_of, output_dir=args.output_dir)
    print(json.dumps({"summary": summary, "paths": {key: str(value) for key, value in asdict(paths).items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

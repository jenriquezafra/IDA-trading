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

from src.alpha.risk_off_eda import load_eda_frame
from src.research.manifest import fingerprint_path
from src.strategy import StrategySpec


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_runner_h1c_signal_only.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_signal_runner")
DEFAULT_HORIZON = 6


@dataclass(frozen=True)
class PaperH1CConfig:
    mode: str
    threshold_policy: str
    max_data_staleness_days: int
    strategy_spec_path: Path
    freeze_manifest_path: Path
    fold_thresholds_path: Path
    features_path: Path
    risk_context_path: Path
    target_account: str
    target_symbol: str
    unit_size: float
    order_type: str
    time_in_force: str
    execution_timing: str
    send_orders: bool
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PaperH1CConfig":
        runner = dict(raw.get("runner", {}))
        strategy = dict(raw.get("strategy", {}))
        data = dict(raw.get("data", {}))
        paper = dict(raw.get("paper", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            mode=str(runner.get("mode", "signal_only")).strip(),
            threshold_policy=str(runner.get("threshold_policy", "latest_frozen_fold")).strip(),
            max_data_staleness_days=int(runner.get("max_data_staleness_days", 3)),
            strategy_spec_path=Path(strategy.get("strategy_spec_path", "configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml")),
            freeze_manifest_path=Path(strategy.get("freeze_manifest_path", "")),
            fold_thresholds_path=Path(strategy.get("fold_thresholds_path", "")),
            features_path=Path(data.get("features_path", "")),
            risk_context_path=Path(data.get("risk_context_path", "")),
            target_account=str(paper.get("target_account", "")).strip(),
            target_symbol=str(paper.get("target_symbol", "QQQ")).strip().upper(),
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
            raise ValueError("H1c paper runner currently supports mode=signal_only only")
        if self.threshold_policy != "latest_frozen_fold":
            raise ValueError("threshold_policy must be latest_frozen_fold")
        if self.max_data_staleness_days < 0:
            raise ValueError("max_data_staleness_days must be non-negative")
        if self.unit_size <= 0.0:
            raise ValueError("unit_size must be positive")
        if self.send_orders:
            raise ValueError("signal-only runner requires paper.send_orders=false")
        if self.order_type != "MKT":
            raise ValueError("signal-only runner currently supports paper.order_type=MKT only")
        if self.time_in_force not in {"DAY"}:
            raise ValueError("signal-only runner currently supports paper.time_in_force=DAY only")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in [
            "strategy_spec_path",
            "freeze_manifest_path",
            "fold_thresholds_path",
            "features_path",
            "risk_context_path",
            "output_dir",
        ]:
            data[key] = data[key].as_posix()
        return data


@dataclass(frozen=True)
class H1COperationalThresholds:
    source_fold: int
    risk_off_min: float
    vix_z20_min: float
    spread_credit_12_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperH1CRunnerPaths:
    output_dir: Path
    signals_path: Path
    latest_signal_path: Path
    ticket_path: Path
    manifest_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_runner_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperH1CConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return PaperH1CConfig.from_mapping(raw)


def load_strategy_spec_raw(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return raw


def select_latest_frozen_thresholds(thresholds: pd.DataFrame) -> H1COperationalThresholds:
    if thresholds.empty:
        raise ValueError("fold thresholds are empty")
    required = {"fold", "risk_off_min", "vix_z20_min"}
    missing = sorted(required.difference(thresholds.columns))
    if missing:
        raise ValueError(f"fold thresholds missing columns: {missing}")
    row = thresholds.sort_values("fold", kind="stable").iloc[-1]
    credit_column = "spread_credit_12_max_threshold"
    credit_max = float(row[credit_column]) if credit_column in row and pd.notna(row[credit_column]) else 0.0
    return H1COperationalThresholds(
        source_fold=int(row["fold"]),
        risk_off_min=float(row["risk_off_min"]),
        vix_z20_min=float(row["vix_z20_min"]),
        spread_credit_12_max=credit_max,
    )


def evaluate_h1c_signal(frame: pd.DataFrame, thresholds: H1COperationalThresholds, *, horizon_bars: int = DEFAULT_HORIZON) -> pd.DataFrame:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    required = {
        "timestamp",
        "session",
        "bar_index",
        "target_open_next",
        "target_next_open_timestamp",
        "target_can_open_trade",
        "target_ret_6",
        "target_ret_12",
        "risk_off_score",
        "prev_vix_z20",
        "spread_credit_12",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"H1c input frame missing columns: {missing}")
    out = frame.copy()
    out["h1c_target_breakdown"] = out["target_ret_6"].lt(0.0) & out["target_ret_12"].lt(0.0)
    out["h1c_risk_off_pass"] = out["risk_off_score"].ge(thresholds.risk_off_min)
    out["h1c_vix_pass"] = out["prev_vix_z20"].ge(thresholds.vix_z20_min)
    out["h1c_credit_pass"] = out["spread_credit_12"].le(thresholds.spread_credit_12_max)
    out["h1c_can_enter"] = out["target_can_open_trade"].fillna(False).astype(bool) & out["target_open_next"].notna()
    out["h1c_theoretical_exit_timestamp"] = out.groupby("session", sort=False)["target_next_open_timestamp"].shift(-int(horizon_bars))
    out["h1c_theoretical_exit_price"] = out.groupby("session", sort=False)["target_open_next"].shift(-int(horizon_bars))
    out["h1c_exit_available"] = out["h1c_theoretical_exit_timestamp"].notna() & out["h1c_theoretical_exit_price"].notna()
    out["h1c_signal_short"] = (
        out["h1c_target_breakdown"]
        & out["h1c_risk_off_pass"]
        & out["h1c_vix_pass"]
        & out["h1c_credit_pass"]
        & out["h1c_can_enter"]
        & out["h1c_exit_available"]
    ).fillna(False)
    out["desired_position_unit"] = out["h1c_signal_short"].map({True: -1.0, False: 0.0}).astype(float)
    out["h1c_horizon_bars"] = int(horizon_bars)
    out["threshold_source_fold"] = thresholds.source_fold
    out["risk_off_min"] = thresholds.risk_off_min
    out["vix_z20_min"] = thresholds.vix_z20_min
    out["spread_credit_12_max"] = thresholds.spread_credit_12_max
    columns = [
        "timestamp",
        "session",
        "bar_index",
        "target_next_open_timestamp",
        "target_open_next",
        "target_can_open_trade",
        "target_ret_6",
        "target_ret_12",
        "risk_off_score",
        "prev_vix_z20",
        "spread_credit_12",
        "h1c_target_breakdown",
        "h1c_risk_off_pass",
        "h1c_vix_pass",
        "h1c_credit_pass",
        "h1c_can_enter",
        "h1c_exit_available",
        "h1c_signal_short",
        "desired_position_unit",
        "h1c_horizon_bars",
        "h1c_theoretical_exit_timestamp",
        "h1c_theoretical_exit_price",
        "threshold_source_fold",
        "risk_off_min",
        "vix_z20_min",
        "spread_credit_12_max",
    ]
    return out[columns].copy()


def select_asof_signal(signals: pd.DataFrame, as_of: str | None = None) -> pd.Series:
    if signals.empty:
        raise ValueError("no signals available")
    ordered = signals.sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)
    if as_of:
        asof_ts = pd.Timestamp(as_of)
        timestamps = pd.to_datetime(ordered["timestamp"])
        if timestamps.dt.tz is not None and asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize(timestamps.dt.tz)
        eligible = ordered[timestamps <= asof_ts]
        if eligible.empty:
            raise ValueError(f"no signal rows at or before as_of={as_of}")
        return eligible.iloc[-1]
    return ordered.iloc[-1]


def build_signal_ticket(latest: pd.Series, config: PaperH1CConfig, strategy: StrategySpec) -> dict[str, Any]:
    signal = bool(latest["h1c_signal_short"])
    entry_timestamp = latest["target_next_open_timestamp"]
    exit_timestamp = latest["h1c_theoretical_exit_timestamp"]
    return {
        "mode": config.mode,
        "send_orders": False,
        "strategy_id": strategy.strategy_id,
        "account": config.target_account,
        "symbol": config.target_symbol,
        "signal_timestamp": str(latest["timestamp"]),
        "session": str(latest["session"]),
        "bar_index": int(latest["bar_index"]),
        "entry_rule": strategy.entry_rule,
        "exit_rule": "fixed_horizon_open",
        "horizon_bars": int(strategy.exit_rule.horizon_bars),
        "execution_timing": config.execution_timing,
        "theoretical_entry_timestamp": None if pd.isna(entry_timestamp) else str(entry_timestamp),
        "theoretical_entry_price": None if pd.isna(latest["target_open_next"]) else float(latest["target_open_next"]),
        "theoretical_exit_timestamp": None if pd.isna(exit_timestamp) else str(exit_timestamp),
        "theoretical_exit_price": None if pd.isna(latest["h1c_theoretical_exit_price"]) else float(latest["h1c_theoretical_exit_price"]),
        "desired_position_unit": float(latest["desired_position_unit"]),
        "action": "SELL" if signal else "NONE",
        "quantity": float(config.unit_size) if signal else 0.0,
        "order_type": config.order_type,
        "time_in_force": config.time_in_force,
        "status": "paper_ticket_only" if signal else "no_signal",
        "reason": "H1c short signal active" if signal else "H1c conditions not all true or entry unavailable",
    }


def data_staleness_warning(latest: pd.Series, max_days: int) -> str:
    latest_session = pd.Timestamp(str(latest["session"]))
    now_session = pd.Timestamp(datetime.now(timezone.utc).date())
    age_days = int((now_session - latest_session).days)
    if age_days > max_days:
        return f"latest feature session {latest_session.date()} is {age_days} days old; refresh intraday data before live paper decisions"
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


def _write_report(path: Path, *, config: PaperH1CConfig, thresholds: H1COperationalThresholds, latest: pd.Series, ticket: dict[str, Any], warnings: list[str]) -> None:
    lines = [
        "# H1c paper signal runner",
        "",
        f"- Mode: `{config.mode}`",
        f"- Strategy: `{ticket['strategy_id']}`",
        f"- Account: `{config.target_account}`",
        f"- Symbol: `{config.target_symbol}`",
        f"- Latest signal timestamp: `{ticket['signal_timestamp']}`",
        f"- Theoretical entry timestamp: `{ticket['theoretical_entry_timestamp']}`",
        f"- Theoretical exit timestamp: `{ticket['theoretical_exit_timestamp']}`",
        f"- Signal short: `{bool(latest['h1c_signal_short'])}`",
        f"- Action: `{ticket['action']}`",
        f"- Quantity: `{ticket['quantity']}`",
        f"- Send orders: `{ticket['send_orders']}`",
        f"- Threshold source fold: `{thresholds.source_fold}`",
        f"- risk_off_min: `{thresholds.risk_off_min:.8f}`",
        f"- vix_z20_min: `{thresholds.vix_z20_min:.8f}`",
        f"- spread_credit_12_max: `{thresholds.spread_credit_12_max:.8f}`",
        "",
        "## Condition State",
        "",
        f"- target breakdown: `{bool(latest['h1c_target_breakdown'])}`",
        f"- risk-off pass: `{bool(latest['h1c_risk_off_pass'])}`",
        f"- VIX pass: `{bool(latest['h1c_vix_pass'])}`",
        f"- credit pass: `{bool(latest['h1c_credit_pass'])}`",
        f"- can enter: `{bool(latest['h1c_can_enter'])}`",
        f"- exit available: `{bool(latest['h1c_exit_available'])}`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    lines.append("This runner is signal-only. It does not connect to IBKR and does not submit orders.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_h1c_signal_runner(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    as_of: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PaperH1CRunnerPaths, dict[str, Any]]:
    config = load_runner_config(config_path)
    strategy = StrategySpec.from_yaml(config.strategy_spec_path)
    raw_spec = load_strategy_spec_raw(config.strategy_spec_path)
    if strategy.strategy_id != "qqq_15min_risk_off_short_h1c_v1":
        raise ValueError(f"unexpected strategy_id for H1c runner: {strategy.strategy_id}")
    if strategy.target_symbol != config.target_symbol:
        raise ValueError(f"strategy target {strategy.target_symbol} does not match runner target {config.target_symbol}")

    horizon_bars = int(strategy.exit_rule.horizon_bars)
    thresholds_df = pd.read_parquet(config.fold_thresholds_path)
    thresholds = select_latest_frozen_thresholds(thresholds_df)
    frame = load_eda_frame(config.features_path, config.risk_context_path, (horizon_bars,))
    signals = evaluate_h1c_signal(frame, thresholds, horizon_bars=horizon_bars)
    latest = select_asof_signal(signals, as_of=as_of)
    ticket = build_signal_ticket(latest, config, strategy)
    warnings = [warning for warning in [data_staleness_warning(latest, config.max_data_staleness_days)] if warning]

    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = PaperH1CRunnerPaths(
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
            "run_type": "h1c_paper_signal_runner",
            "created_at_utc": created,
            "status": "signal_only",
        },
        "config": config.to_dict(),
        "strategy": strategy.to_dict(),
        "alpha": raw_spec.get("alpha", {}),
        "thresholds": thresholds.to_dict(),
        "latest": {
            "timestamp": ticket["signal_timestamp"],
            "session": ticket["session"],
            "signal_short": bool(latest["h1c_signal_short"]),
            "action": ticket["action"],
        },
        "warnings": warnings,
        "data": {
            "features_path": config.features_path.as_posix(),
            "features_fingerprint": fingerprint_path(config.features_path) if config.features_path.exists() else "MISSING",
            "risk_context_path": config.risk_context_path.as_posix(),
            "risk_context_fingerprint": fingerprint_path(config.risk_context_path) if config.risk_context_path.exists() else "MISSING",
            "fold_thresholds_path": config.fold_thresholds_path.as_posix(),
            "fold_thresholds_fingerprint": fingerprint_path(config.fold_thresholds_path) if config.fold_thresholds_path.exists() else "MISSING",
            "freeze_manifest_path": config.freeze_manifest_path.as_posix(),
            "freeze_manifest_fingerprint": fingerprint_path(config.freeze_manifest_path) if config.freeze_manifest_path.exists() else "MISSING",
        },
        "outputs": {
            "signals": paths.signals_path.as_posix(),
            "latest_signal": paths.latest_signal_path.as_posix(),
            "paper_ticket": paths.ticket_path.as_posix(),
            "report": paths.report_path.as_posix(),
        },
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, config=config, thresholds=thresholds, latest=latest, ticket=ticket, warnings=warnings)
    return paths, {"ticket": ticket, "thresholds": thresholds.to_dict(), "warnings": warnings}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run H1c paper signal runner without submitting orders")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--as-of", default=None, help="optional timestamp cutoff for selecting latest signal row")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    paths, summary = run_h1c_signal_runner(config_path=args.config, as_of=args.as_of, output_dir=args.output_dir)
    print(json.dumps({"summary": summary, "paths": {key: str(value) for key, value in asdict(paths).items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

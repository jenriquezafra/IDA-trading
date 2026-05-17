from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.execution.paper_data_refresh import PaperDataRefreshPaths, run_paper_data_refresh
from src.execution.paper_h1c_signal_runner import PaperH1CRunnerPaths, run_h1c_signal_runner
from src.execution.paper_reconcile_h1c import H1CReconciliationPaths, run_h1c_reconciliation
from src.execution.paper_state_store import PaperStatePaths, apply_ticket


DEFAULT_CONFIG_PATH = Path("configs/execution/paper_cycle_h1c.yaml")
DEFAULT_OUTPUT_DIR = Path("results/paper/h1c_cycle")


@dataclass(frozen=True)
class PaperCycleConfig:
    strategy_id: str
    account: str
    symbol: str
    refresh_data: bool
    apply_state: bool
    reconcile_after_state: bool
    signal_only: bool
    default_skip_download: bool
    default_skip_cboe: bool
    default_refresh_dry_run: bool
    data_refresh_config_path: Path
    signal_runner_config_path: Path
    state_config_path: Path
    reconciliation_config_path: Path
    output_dir: Path

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PaperCycleConfig":
        cycle = dict(raw.get("cycle", {}))
        components = dict(raw.get("components", {}))
        outputs = dict(raw.get("outputs", {}))
        config = cls(
            strategy_id=str(cycle.get("strategy_id", "")).strip(),
            account=str(cycle.get("account", "")).strip(),
            symbol=str(cycle.get("symbol", "QQQ")).strip().upper(),
            refresh_data=bool(cycle.get("refresh_data", True)),
            apply_state=bool(cycle.get("apply_state", True)),
            reconcile_after_state=bool(cycle.get("reconcile_after_state", False)),
            signal_only=bool(cycle.get("signal_only", True)),
            default_skip_download=bool(cycle.get("default_skip_download", False)),
            default_skip_cboe=bool(cycle.get("default_skip_cboe", True)),
            default_refresh_dry_run=bool(cycle.get("default_refresh_dry_run", False)),
            data_refresh_config_path=Path(components.get("data_refresh_config_path", "configs/execution/paper_data_refresh.yaml")),
            signal_runner_config_path=Path(components.get("signal_runner_config_path", "configs/execution/paper_runner_h1c_signal_only.yaml")),
            state_config_path=Path(components.get("state_config_path", "configs/execution/paper_state_h1c.yaml")),
            reconciliation_config_path=Path(components.get("reconciliation_config_path", "configs/execution/paper_reconcile_h1c.yaml")),
            output_dir=Path(outputs.get("output_dir", DEFAULT_OUTPUT_DIR)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("cycle.strategy_id is required")
        if not self.account:
            raise ValueError("cycle.account is required")
        if not self.symbol:
            raise ValueError("cycle.symbol is required")
        if not self.signal_only:
            raise ValueError("paper cycle currently supports signal_only=true only")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ["data_refresh_config_path", "signal_runner_config_path", "state_config_path", "reconciliation_config_path", "output_dir"]:
            data[key] = data[key].as_posix()
        return data


@dataclass(frozen=True)
class PaperCyclePaths:
    output_dir: Path
    manifest_path: Path
    report_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_cycle_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperCycleConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping: {config_path}")
    return PaperCycleConfig.from_mapping(raw)


def _paths_to_dict(paths: Any | None) -> dict[str, str]:
    if paths is None:
        return {}
    return {key: str(value) for key, value in asdict(paths).items()}


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    latest = manifest.get("signal", {}).get("latest", {})
    event = manifest.get("state", {}).get("event", {})
    warnings = manifest.get("warnings", [])
    lines = [
        "# H1c paper cycle",
        "",
        f"- Created UTC: `{manifest['run']['created_at_utc']}`",
        f"- Status: `{manifest['run']['status']}`",
        f"- Strategy: `{manifest['strategy_id']}`",
        f"- Account: `{manifest['account']}`",
        f"- Symbol: `{manifest['symbol']}`",
        f"- Refresh data: `{manifest['steps']['refresh_data']}`",
        f"- Apply state: `{manifest['steps']['apply_state']}`",
        f"- Reconcile: `{manifest['steps']['reconcile_after_state']}`",
        "",
        "## Decision",
        "",
        f"- Signal timestamp: `{latest.get('timestamp', '')}`",
        f"- Signal short: `{latest.get('signal_short', '')}`",
        f"- Ticket action: `{latest.get('action', '')}`",
        f"- State event: `{event.get('event_type', '')}`",
        f"- New state: `{event.get('new_status', '')}`",
        f"- Reconciliation decision: `{manifest.get('reconciliation', {}).get('decision', '')}`",
        f"- Reconciliation severity: `{manifest.get('reconciliation', {}).get('severity', '')}`",
        "",
        "## Outputs",
        "",
        f"- Data refresh report: `{manifest['outputs'].get('data_refresh_report', '')}`",
        f"- Signal report: `{manifest['outputs'].get('signal_report', '')}`",
        f"- Paper ticket: `{manifest['outputs'].get('paper_ticket', '')}`",
        f"- State report: `{manifest['outputs'].get('state_report', '')}`",
        f"- State path: `{manifest['outputs'].get('state_path', '')}`",
        f"- Reconciliation report: `{manifest['outputs'].get('reconciliation_report', '')}`",
    ]
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    lines.append("")
    lines.append("This cycle is signal-only. Reconciliation may connect to IBKR read-only; it never submits orders.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_paper_cycle(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    as_of: str | None = None,
    refresh_data: bool | None = None,
    apply_state_update: bool | None = None,
    reconcile_after_state: bool | None = None,
    skip_download: bool | None = None,
    skip_cboe: bool | None = None,
    refresh_dry_run: bool | None = None,
    output_dir: str | Path | None = None,
) -> tuple[PaperCyclePaths, dict[str, Any]]:
    config = load_cycle_config(config_path)
    created = utc_now()
    root = (Path(output_dir) if output_dir is not None else config.output_dir) / created.replace(":", "").replace("-", "")
    paths = PaperCyclePaths(output_dir=root, manifest_path=root / "manifest.yaml", report_path=root / "report.md")
    root.mkdir(parents=True, exist_ok=True)

    do_refresh = config.refresh_data if refresh_data is None else refresh_data
    do_state = config.apply_state if apply_state_update is None else apply_state_update
    do_reconcile = config.reconcile_after_state if reconcile_after_state is None else reconcile_after_state
    effective_skip_download = config.default_skip_download if skip_download is None else skip_download
    effective_skip_cboe = config.default_skip_cboe if skip_cboe is None else skip_cboe
    effective_refresh_dry_run = config.default_refresh_dry_run if refresh_dry_run is None else refresh_dry_run

    refresh_paths: PaperDataRefreshPaths | None = None
    refresh_manifest: dict[str, Any] | None = None
    if do_refresh:
        refresh_paths, refresh_manifest = run_paper_data_refresh(
            config_path=config.data_refresh_config_path,
            dry_run=effective_refresh_dry_run,
            skip_download=effective_skip_download,
            skip_cboe=effective_skip_cboe,
            output_dir=root / "data_refresh",
        )

    signal_paths, signal_summary = run_h1c_signal_runner(
        config_path=config.signal_runner_config_path,
        as_of=as_of,
        output_dir=root / "signal",
    )

    state_paths: PaperStatePaths | None = None
    state_summary: dict[str, Any] | None = None
    if do_state:
        state_paths, state_summary = apply_ticket(
            ticket_path=signal_paths.ticket_path,
            config_path=config.state_config_path,
            output_dir=root / "state",
        )

    reconciliation_paths: H1CReconciliationPaths | None = None
    reconciliation_manifest: dict[str, Any] | None = None
    if do_reconcile:
        reconciliation_paths, reconciliation_manifest = run_h1c_reconciliation(
            config_path=config.reconciliation_config_path,
            output_dir=root / "reconciliation",
        )

    refresh_warnings = list((refresh_manifest or {}).get("warnings", []))
    signal_warnings = list(signal_summary.get("warnings", []))
    manifest = {
        "schema_version": 1,
        "run": {
            "run_type": "h1c_paper_cycle",
            "created_at_utc": created,
            "status": "complete",
        },
        "strategy_id": config.strategy_id,
        "account": config.account,
        "symbol": config.symbol,
        "config": config.to_dict(),
        "steps": {
            "refresh_data": do_refresh,
            "apply_state": do_state,
            "reconcile_after_state": do_reconcile,
            "skip_download": effective_skip_download,
            "skip_cboe": effective_skip_cboe,
            "refresh_dry_run": effective_refresh_dry_run,
        },
        "data_refresh": {
            "paths": _paths_to_dict(refresh_paths),
            "date_window": (refresh_manifest or {}).get("date_window", {}),
            "status": (refresh_manifest or {}).get("run", {}).get("status", "skipped"),
        },
        "signal": {
            "paths": _paths_to_dict(signal_paths),
            "latest": {
                "timestamp": signal_summary["ticket"]["signal_timestamp"],
                "session": signal_summary["ticket"]["session"],
                "signal_short": signal_summary["ticket"]["action"] == "SELL",
                "action": signal_summary["ticket"]["action"],
                "quantity": signal_summary["ticket"]["quantity"],
                "status": signal_summary["ticket"]["status"],
            },
            "thresholds": signal_summary["thresholds"],
        },
        "state": {
            "paths": _paths_to_dict(state_paths),
            "event": {} if state_summary is None else state_summary["event"],
            "state": {} if state_summary is None else state_summary["state"],
        },
        "reconciliation": {} if reconciliation_manifest is None else reconciliation_manifest["reconciliation"],
        "outputs": {
            "manifest": paths.manifest_path.as_posix(),
            "report": paths.report_path.as_posix(),
            "data_refresh_report": "" if refresh_paths is None else refresh_paths.report_path.as_posix(),
            "signal_report": signal_paths.report_path.as_posix(),
            "paper_ticket": signal_paths.ticket_path.as_posix(),
            "state_report": "" if state_paths is None else state_paths.report_path.as_posix(),
            "state_path": "" if state_paths is None else state_paths.state_path.as_posix(),
            "event_log_path": "" if state_paths is None else state_paths.event_log_path.as_posix(),
            "reconciliation_report": "" if reconciliation_paths is None else reconciliation_paths.report_path.as_posix(),
        },
        "warnings": refresh_warnings + signal_warnings,
    }
    paths.manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_report(paths.report_path, manifest)
    return paths, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one signal-only H1c paper cycle: refresh data, generate ticket, update state")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--as-of", default=None, help="optional timestamp cutoff for selecting latest signal row")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-refresh", action="store_true", help="skip the data refresh stage")
    parser.add_argument("--no-state", action="store_true", help="skip applying the generated ticket to the local state store")
    parser.add_argument("--no-reconcile", action="store_true", help="skip read-only IBKR reconciliation")
    parser.add_argument("--skip-download", action="store_true", help="do not call Polygon download during refresh")
    parser.add_argument("--skip-cboe", action="store_true", help="do not refresh Cboe risk context during refresh")
    parser.add_argument("--refresh-dry-run", action="store_true", help="write refresh manifest without modifying data files")
    args = parser.parse_args(argv)
    paths, manifest = run_paper_cycle(
        config_path=args.config,
        as_of=args.as_of,
        refresh_data=False if args.no_refresh else None,
        apply_state_update=False if args.no_state else None,
        reconcile_after_state=False if args.no_reconcile else None,
        skip_download=True if args.skip_download else None,
        skip_cboe=True if args.skip_cboe else None,
        refresh_dry_run=True if args.refresh_dry_run else None,
        output_dir=args.output_dir,
    )
    print(json.dumps({"paths": _paths_to_dict(paths), "summary": manifest}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

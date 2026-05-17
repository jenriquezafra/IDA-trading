from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.alpha.research import add_forward_returns
from src.backtesting import evaluate_positions


DEFAULT_FEATURES_PATH = Path("data/features/QQQ/15min/core_cross_asset_v1/cross_asset_liquid_15min/features.parquet")
DEFAULT_RISK_CONTEXT_PATH = Path("data/external/cboe/risk_context_daily.parquet")
DEFAULT_OUTPUT_DIR = Path("reports/eda/risk_off_short")
DEFAULT_HORIZONS = (2, 3, 4, 6)


@dataclass(frozen=True)
class RiskOffEdaOutputs:
    output_dir: Path
    report_path: Path
    bucket_summary_path: Path
    condition_summary_path: Path
    yearly_summary_path: Path
    control_pnl_path: Path


def _date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d")


def load_eda_frame(features_path: str | Path, risk_context_path: str | Path, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    features = pd.read_parquet(features_path).sort_values(["session", "bar_index"], kind="stable").reset_index(drop=True)
    context = pd.read_parquet(risk_context_path).copy()
    context["session"] = _date_key(context["available_session"])
    context = context.sort_values(["session", "source_date"], kind="stable").drop_duplicates("session", keep="last")
    merged = features.merge(context.drop(columns=["available_session"]), on="session", how="left", validate="many_to_one")
    enriched = add_forward_returns(merged, set(horizons))
    enriched["hour"] = pd.to_datetime(enriched["timestamp"]).dt.hour
    enriched["year"] = pd.to_datetime(enriched["session"]).dt.year
    return enriched


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    if column not in frame:
        return np.nan
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def _metric_row(frame: pd.DataFrame, label: str, horizon: int, extra: dict[str, object] | None = None) -> dict[str, object]:
    column = f"fwd_ret_{horizon}"
    values = frame[column].replace([np.inf, -np.inf], np.nan).dropna() if column in frame else pd.Series(dtype=float)
    row: dict[str, object] = {
        "label": label,
        "horizon_bars": int(horizon),
        "rows": int(len(frame)),
        "valid_returns": int(values.shape[0]),
        "mean_fwd_bps": float(values.mean() * 10_000.0) if not values.empty else np.nan,
        "median_fwd_bps": float(values.median() * 10_000.0) if not values.empty else np.nan,
        "mean_short_bps": float(-values.mean() * 10_000.0) if not values.empty else np.nan,
        "short_win_rate": float(values.lt(0.0).mean()) if not values.empty else np.nan,
        "p10_fwd_bps": float(values.quantile(0.10) * 10_000.0) if not values.empty else np.nan,
        "p90_fwd_bps": float(values.quantile(0.90) * 10_000.0) if not values.empty else np.nan,
    }
    if extra:
        row.update(extra)
    return row


def condition_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    risk_off_high = _quantile(frame, "risk_off_score", 0.70)
    risk_on_low = _quantile(frame, "risk_on_score", 0.30)
    credit_weak = _quantile(frame, "spread_credit_12", 0.30)
    defensive_high = _quantile(frame, "defensive_rotation_score", 0.70)
    vix_high = _quantile(frame, "prev_vix_z20", 0.70)

    target_breakdown = frame["target_ret_6"].lt(0.0) & frame["target_ret_12"].lt(0.0)
    risk_off_context = frame["risk_off_score"].ge(risk_off_high)
    low_risk_on = frame["risk_on_score"].le(risk_on_low)
    weak_credit = frame["spread_credit_12"].le(credit_weak)
    defensive_rotation = frame["defensive_rotation_score"].ge(defensive_high)
    vix_pressure = frame["prev_vix_z20"].ge(vix_high)

    return {
        "unconditional": pd.Series(True, index=frame.index),
        "target_breakdown": target_breakdown,
        "risk_off_top30": risk_off_context,
        "target_breakdown__risk_off": target_breakdown & risk_off_context,
        "target_breakdown__risk_off__risk_on_low": target_breakdown & risk_off_context & low_risk_on,
        "target_breakdown__risk_off__credit_weak": target_breakdown & risk_off_context & weak_credit,
        "target_breakdown__risk_off__defensive": target_breakdown & risk_off_context & defensive_rotation,
        "target_breakdown__risk_off__vix_pressure": target_breakdown & risk_off_context & vix_pressure,
        "h1_core": target_breakdown & risk_off_context & low_risk_on & weak_credit,
    }


def build_condition_summary(frame: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = condition_masks(frame)
    for label, mask in masks.items():
        subset = frame[mask.fillna(False)].copy()
        for horizon in horizons:
            rows.append(_metric_row(subset, label, horizon))
    return pd.DataFrame(rows)


def build_yearly_summary(frame: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS, condition: str = "h1_core") -> pd.DataFrame:
    masks = condition_masks(frame)
    subset = frame[masks[condition].fillna(False)].copy()
    rows: list[dict[str, object]] = []
    for year, year_frame in subset.groupby("year", sort=True):
        for horizon in horizons:
            rows.append(_metric_row(year_frame, condition, horizon, {"year": int(year)}))
    return pd.DataFrame(rows)


def build_control_pnl(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cost_bps_values: tuple[float, ...] = (1.0, 2.0, 5.0),
    candidate_label: str = "target_breakdown__risk_off__vix_pressure",
    random_seed: int = 42,
) -> pd.DataFrame:
    masks = condition_masks(frame)
    candidate = masks[candidate_label].fillna(False)
    active_hours = sorted(int(value) for value in frame.loc[candidate, "hour"].dropna().unique().tolist())
    same_hour = frame["hour"].isin(active_hours)
    base_masks = {
        candidate_label: candidate,
        "h1_core": masks["h1_core"].fillna(False),
        "target_breakdown": masks["target_breakdown"].fillna(False),
        "risk_off_top30": masks["risk_off_top30"].fillna(False),
        "same_hour_short_control": same_hour,
        "always_flat": pd.Series(False, index=frame.index),
    }

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        return_column = f"fwd_ret_{horizon}"
        valid = frame[return_column].notna() if return_column in frame else pd.Series(False, index=frame.index)
        rng = np.random.default_rng(random_seed + int(horizon))
        candidate_count = int((candidate & valid).sum())
        random_mask = pd.Series(False, index=frame.index)
        valid_indices = np.flatnonzero(valid.to_numpy())
        if candidate_count > 0 and len(valid_indices) >= candidate_count:
            sampled = rng.choice(valid_indices, size=candidate_count, replace=False)
            random_mask.iloc[sampled] = True
        masks_for_horizon = {**base_masks, "random_same_count_control": random_mask}

        for label, mask in masks_for_horizon.items():
            position = pd.Series(0.0, index=frame.index)
            position.loc[mask.fillna(False)] = -1.0
            for cost_bps in cost_bps_values:
                metrics = evaluate_positions(frame, position, return_column=return_column, cost_bps=cost_bps)
                rows.append(
                    {
                        "label": label,
                        "horizon_bars": int(horizon),
                        "cost_bps": float(cost_bps),
                        "active_hours": ",".join(str(hour) for hour in active_hours) if label == "same_hour_short_control" else "",
                        **metrics.to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def build_bucket_summary(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    buckets: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        valid = frame[values.notna()].copy()
        if valid.empty or valid[column].nunique(dropna=True) < 2:
            continue
        try:
            valid["_bucket"] = pd.qcut(valid[column], q=buckets, duplicates="drop")
        except ValueError:
            continue
        for bucket, bucket_frame in valid.groupby("_bucket", observed=True, sort=True):
            for horizon in horizons:
                rows.append(_metric_row(bucket_frame, str(bucket), horizon, {"feature": column}))
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> list[str]:
    if frame.empty:
        return ["No rows."]
    visible = frame.loc[:, [col for col in columns if col in frame.columns]].head(limit)
    lines = ["| " + " | ".join(visible.columns) + " |", "| " + " | ".join(["---"] * len(visible.columns)) + " |"]
    for _, row in visible.iterrows():
        values = []
        for col in visible.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    frame: pd.DataFrame,
    condition_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    control_pnl: pd.DataFrame,
    features_path: Path,
    risk_context_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h1 = condition_summary[condition_summary["label"].eq("h1_core")].sort_values("horizon_bars", kind="stable")
    top_conditions = condition_summary[condition_summary["horizon_bars"].eq(4)].sort_values("mean_short_bps", ascending=False, kind="stable")
    risk_off_buckets = bucket_summary[
        bucket_summary["feature"].eq("risk_off_score") & bucket_summary["horizon_bars"].eq(4)
    ].sort_values("label", kind="stable")
    control_h4 = control_pnl[
        control_pnl["horizon_bars"].eq(4) & control_pnl["cost_bps"].eq(2.0)
    ].sort_values("net_return", ascending=False, kind="stable")
    lines = [
        "# Risk-off short EDA",
        "",
        "EDA significa Exploratory Data Analysis: analisis exploratorio de datos antes de definir una estrategia.",
        "Este reporte no selecciona thresholds finales ni valida una estrategia; busca evidencia de que la hipotesis merece pasar a una regla operable.",
        "",
        "## Inputs",
        "",
        f"- Features: `{features_path}`",
        f"- Risk context: `{risk_context_path}`",
        f"- Rows intradia: `{len(frame)}`",
        f"- Sessions: `{frame['session'].nunique()}`",
        f"- Context coverage: `{frame['source_date'].notna().mean():.2%}`",
        "",
        "## H1 core",
        "",
        "`h1_core = target_ret_6 < 0 + target_ret_12 < 0 + risk_off_score top 30% + risk_on_score bottom 30% + weak credit`",
        "",
        *_markdown_table(
            h1,
            ["label", "horizon_bars", "valid_returns", "mean_fwd_bps", "mean_short_bps", "short_win_rate", "p10_fwd_bps", "p90_fwd_bps"],
        ),
        "",
        "## Best condition summaries at h=4",
        "",
        *_markdown_table(
            top_conditions,
            ["label", "valid_returns", "mean_fwd_bps", "mean_short_bps", "short_win_rate", "p10_fwd_bps", "p90_fwd_bps"],
            limit=12,
        ),
        "",
        "## Risk-off score buckets at h=4",
        "",
        *_markdown_table(
            risk_off_buckets,
            ["feature", "label", "valid_returns", "mean_fwd_bps", "mean_short_bps", "short_win_rate"],
        ),
        "",
        "## Pre-strategy PnL controls at h=4, cost=2 bps",
        "",
        "This is still not an execution backtest. It is a bar-signal diagnostic with simple short positions and turnover costs.",
        "",
        *_markdown_table(
            control_h4,
            ["label", "trades", "exposure", "turnover", "net_return", "avg_trade_net", "profit_factor", "daily_sharpe", "max_drawdown"],
            limit=12,
        ),
        "",
        "## H1 core yearly stability",
        "",
        *_markdown_table(
            yearly_summary[yearly_summary["horizon_bars"].eq(4)],
            ["year", "label", "valid_returns", "mean_fwd_bps", "mean_short_bps", "short_win_rate"],
        ),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    risk_context_path: str | Path = DEFAULT_RISK_CONTEXT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> RiskOffEdaOutputs:
    features_path = Path(features_path)
    risk_context_path = Path(risk_context_path)
    output_dir = Path(output_dir)
    frame = load_eda_frame(features_path, risk_context_path, horizons)
    condition_summary = build_condition_summary(frame, horizons)
    yearly_summary = build_yearly_summary(frame, horizons)
    control_pnl = build_control_pnl(frame, horizons)
    bucket_summary = build_bucket_summary(
        frame,
        (
            "risk_off_score",
            "risk_on_score",
            "spread_credit_12",
            "defensive_rotation_score",
            "intraday_stress_score",
            "prev_vix_z20",
            "prev_vix9d_vix_ratio",
            "prev_total_put_call_ratio_z20",
            "prev_index_put_call_ratio_z20",
        ),
        horizons,
    )
    outputs = RiskOffEdaOutputs(
        output_dir=output_dir,
        report_path=output_dir / "risk_off_short_eda.md",
        bucket_summary_path=output_dir / "bucket_summary.parquet",
        condition_summary_path=output_dir / "condition_summary.parquet",
        yearly_summary_path=output_dir / "yearly_summary.parquet",
        control_pnl_path=output_dir / "control_pnl.parquet",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_summary.to_parquet(outputs.bucket_summary_path, index=False)
    condition_summary.to_parquet(outputs.condition_summary_path, index=False)
    yearly_summary.to_parquet(outputs.yearly_summary_path, index=False)
    control_pnl.to_parquet(outputs.control_pnl_path, index=False)
    write_report(outputs.report_path, frame, condition_summary, yearly_summary, bucket_summary, control_pnl, features_path, risk_context_path)
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build risk-off short continuation EDA")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--risk-context", default=str(DEFAULT_RISK_CONTEXT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    args = parser.parse_args(argv)
    outputs = run_eda(
        features_path=args.features,
        risk_context_path=args.risk_context,
        output_dir=args.output_dir,
        horizons=tuple(args.horizons),
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

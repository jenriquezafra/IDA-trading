from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROMOTION_GATES = {
    "min_validation_trades": 40,
    "min_test_trades": 30,
    "min_validation_positive_folds": 4,
    "min_test_positive_folds": 4,
    "min_validation_net_return": 0.0,
    "min_test_net_return": 0.0,
    "min_avg_trade_net_bps": 5.0,
    "stress_cost_bps": 5.0,
    "min_validation_stress_net_return": 0.0,
    "min_test_stress_net_return": 0.0,
    "min_sessions_per_fold": 8,
    "max_top5_abs_share": 0.70,
    "require_beats_best_control": True,
}


def _gate_row(gate_id: str, observed: float | int | str | bool, threshold: float | int | str | bool, passed: bool, rationale: str) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "pass" if passed else "fail",
        "observed": str(observed),
        "threshold": str(threshold),
        "severity": "block",
        "rationale": rationale,
    }


def rollup_by_cost(summary: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    filtered = summary[summary["cost_bps"].eq(float(cost_bps)) & summary["split"].isin(["validation", "test"])]
    rows: list[dict[str, Any]] = []
    for (split, label), group in filtered.groupby(["split", "label"], sort=False):
        trades = int(group["trades"].sum())
        net_return = float(group["net_return"].sum())
        rows.append(
            {
                "split": str(split),
                "label": str(label),
                "folds": int(group["fold"].nunique()),
                "trades": trades,
                "net_return": net_return,
                "avg_trade_net": net_return / trades if trades else 0.0,
                "positive_folds": int(group["net_return"].gt(0.0).sum()),
                "min_fold_return": float(group["net_return"].min()),
                "max_fold_return": float(group["net_return"].max()),
                "mean_daily_sharpe": float(group["daily_sharpe"].mean()),
                "max_fold_drawdown": float(group["max_drawdown"].max()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_split_order"] = result["split"].map({"validation": 0, "test": 1}).fillna(9).astype(int)
    return result.sort_values(["_split_order", "net_return"], ascending=[True, False], kind="stable").drop(columns="_split_order")


def evaluate_promotion_gates(
    selected_summary: pd.DataFrame,
    selected_concentration: pd.DataFrame,
    gates: dict[str, Any] | None = None,
    *,
    candidate_label: str,
    primary_cost_bps: float = 2.0,
    splits: tuple[str, ...] = ("validation", "test"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate_cfg = dict(gates or DEFAULT_PROMOTION_GATES)
    if selected_summary.empty:
        decision = {
            "status": "not_evaluated",
            "summary": "No selected-threshold summary was available.",
            "failed_gates": [],
            "gate_config": gate_cfg,
        }
        return pd.DataFrame(columns=["gate_id", "status", "observed", "threshold", "severity", "rationale"]), decision

    primary = rollup_by_cost(selected_summary, cost_bps=primary_cost_bps)
    stress = rollup_by_cost(selected_summary, cost_bps=float(gate_cfg["stress_cost_bps"]))
    rows: list[dict[str, Any]] = []

    def candidate_row(frame: pd.DataFrame, split: str) -> pd.Series:
        match = frame[frame["split"].eq(split) & frame["label"].eq(candidate_label)]
        if match.empty:
            return pd.Series(dtype=object)
        return match.iloc[0]

    def best_control_net(frame: pd.DataFrame, split: str) -> float:
        controls = frame[frame["split"].eq(split) & ~frame["label"].eq(candidate_label)]
        if controls.empty:
            return -np.inf
        return float(controls["net_return"].max())

    split_gate_keys = {
        "validation": ("min_validation_trades", "min_validation_positive_folds", "min_validation_net_return"),
        "test": ("min_test_trades", "min_test_positive_folds", "min_test_net_return"),
    }
    for split in splits:
        min_trades_key, min_folds_key, min_net_key = split_gate_keys[split]
        candidate = candidate_row(primary, split)
        trades = int(candidate.get("trades", 0)) if not candidate.empty else 0
        positive_folds = int(candidate.get("positive_folds", 0)) if not candidate.empty else 0
        net_return = float(candidate.get("net_return", 0.0)) if not candidate.empty else 0.0
        avg_trade_bps = float(candidate.get("avg_trade_net", 0.0)) * 10_000.0 if not candidate.empty else 0.0
        rows.append(_gate_row(f"{split}_min_trades", trades, int(gate_cfg[min_trades_key]), trades >= int(gate_cfg[min_trades_key]), "Enough trades to reduce small-sample noise."))
        rows.append(
            _gate_row(
                f"{split}_positive_folds",
                positive_folds,
                int(gate_cfg[min_folds_key]),
                positive_folds >= int(gate_cfg[min_folds_key]),
                "Edge should appear across folds, not only one window.",
            )
        )
        rows.append(
            _gate_row(
                f"{split}_net_return_positive",
                round(net_return, 6),
                float(gate_cfg[min_net_key]),
                net_return > float(gate_cfg[min_net_key]),
                "Net return after primary cost must be positive.",
            )
        )
        rows.append(
            _gate_row(
                f"{split}_avg_trade_net_bps",
                round(avg_trade_bps, 3),
                float(gate_cfg["min_avg_trade_net_bps"]),
                avg_trade_bps >= float(gate_cfg["min_avg_trade_net_bps"]),
                "Average trade should leave room for slippage/model error.",
            )
        )
        if bool(gate_cfg["require_beats_best_control"]):
            control_net = best_control_net(primary, split)
            rows.append(
                _gate_row(
                    f"{split}_beats_best_control",
                    round(net_return - control_net, 6),
                    "> 0",
                    net_return > control_net,
                    "Candidate must beat the best simple control in the same split.",
                )
            )

    stress_gate_keys = {"validation": "min_validation_stress_net_return", "test": "min_test_stress_net_return"}
    for split in splits:
        min_net_key = stress_gate_keys[split]
        candidate = candidate_row(stress, split)
        stress_net = float(candidate.get("net_return", 0.0)) if not candidate.empty else 0.0
        rows.append(
            _gate_row(
                f"{split}_stress_cost_positive",
                round(stress_net, 6),
                float(gate_cfg[min_net_key]),
                stress_net > float(gate_cfg[min_net_key]),
                "Candidate must survive stress transaction cost.",
            )
        )

    for split in splits:
        split_concentration = selected_concentration[selected_concentration["split"].eq(split)] if not selected_concentration.empty else pd.DataFrame()
        min_sessions = int(split_concentration["sessions_with_trades"].min()) if not split_concentration.empty else 0
        max_top5 = float(split_concentration["top5_abs_share"].max()) if not split_concentration.empty else 1.0
        rows.append(
            _gate_row(
                f"{split}_min_sessions_per_fold",
                min_sessions,
                int(gate_cfg["min_sessions_per_fold"]),
                min_sessions >= int(gate_cfg["min_sessions_per_fold"]),
                "Each fold needs enough sessions with trades to avoid single-event dependence.",
            )
        )
        rows.append(
            _gate_row(
                f"{split}_top5_abs_share",
                round(max_top5, 4),
                float(gate_cfg["max_top5_abs_share"]),
                max_top5 <= float(gate_cfg["max_top5_abs_share"]),
                "Top sessions should not dominate absolute PnL contribution.",
            )
        )

    gates_frame = pd.DataFrame(rows)
    failed = gates_frame.loc[gates_frame["status"].eq("fail"), "gate_id"].astype(str).tolist()
    status = "freeze_review" if not failed else "continue_research"
    decision = {
        "status": status,
        "summary": "All promotion gates passed." if not failed else "Promotion blocked by failing gates; keep candidate in research.",
        "failed_gates": failed,
        "gate_config": gate_cfg,
        "splits": list(splits),
    }
    return gates_frame, decision

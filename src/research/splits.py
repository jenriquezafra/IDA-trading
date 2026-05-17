from __future__ import annotations

"""Walk-forward split builders for research modules."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ResearchFold:
    fold: int
    train_months: tuple[str, ...]
    validation_months: tuple[str, ...]
    test_months: tuple[str, ...]
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def session_months(frame: pd.DataFrame, session_column: str = "session") -> pd.DataFrame:
    if session_column not in frame:
        raise KeyError(f"missing session column: {session_column}")
    sessions = frame[[session_column]].drop_duplicates().copy()
    sessions["session_date"] = pd.to_datetime(sessions[session_column])
    sessions["month"] = sessions["session_date"].dt.to_period("M").astype(str)
    return sessions.rename(columns={session_column: "session"})


def build_monthly_folds(frame: pd.DataFrame, policy: dict[str, Any]) -> tuple[ResearchFold, ...]:
    train_months = int(policy.get("train_months", 24))
    validation_months = int(policy.get("validation_months", 6))
    test_months = int(policy.get("test_months", 6))
    step_months = int(policy.get("step_months", test_months))
    embargo_sessions = int(policy.get("embargo_sessions", 0))
    max_folds = policy.get("max_folds")
    if min(train_months, validation_months, test_months, step_months) <= 0:
        raise ValueError("split policy month values must be positive")
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be non-negative")

    indexed = session_months(frame)
    months = tuple(sorted(indexed["month"].unique().tolist()))
    window = train_months + validation_months + test_months
    folds: list[ResearchFold] = []
    for fold_id, start in enumerate(range(0, max(len(months) - window + 1, 0), step_months)):
        train = months[start : start + train_months]
        validation = months[start + train_months : start + train_months + validation_months]
        test = months[start + train_months + validation_months : start + window]
        if len(train) != train_months or len(validation) != validation_months or len(test) != test_months:
            continue
        train_sessions = tuple(indexed.loc[indexed["month"].isin(train), "session"].astype(str).tolist())
        validation_sessions = tuple(indexed.loc[indexed["month"].isin(validation), "session"].astype(str).tolist())
        test_sessions = tuple(indexed.loc[indexed["month"].isin(test), "session"].astype(str).tolist())
        if embargo_sessions:
            train_sessions = train_sessions[:-embargo_sessions] if len(train_sessions) > embargo_sessions else tuple()
        folds.append(
            ResearchFold(
                fold=fold_id,
                train_months=train,
                validation_months=validation,
                test_months=test,
                train_sessions=train_sessions,
                validation_sessions=validation_sessions,
                test_sessions=test_sessions,
            )
        )
        if max_folds is not None and len(folds) >= int(max_folds):
            break
    return tuple(folds)

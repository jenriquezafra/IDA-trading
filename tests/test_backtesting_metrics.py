from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting import evaluate_positions


def test_evaluate_positions_uses_turnover_costs() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"],
            "forward_return": [0.01, -0.01, 0.02, -0.01],
        }
    )
    position = pd.Series([1.0, 1.0, -1.0, 0.0], index=frame.index)

    metrics = evaluate_positions(frame, position, return_column="forward_return", cost_bps=1.0)

    assert metrics.rows == 4
    assert metrics.trades == 3
    assert metrics.turnover == 4.0
    assert metrics.total_cost == pytest.approx(0.0004)
    assert metrics.gross_return == pytest.approx(-0.02)
    assert metrics.net_return == pytest.approx(-0.0204)

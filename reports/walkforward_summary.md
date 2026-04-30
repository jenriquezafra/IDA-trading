# Walk-Forward Evaluation Summary

## Scope

- Trades: `reports/backtest_trades.parquet`
- Daily PnL: `reports/daily_pnl.parquet`
- Equity curve: `reports/equity_curve.parquet`
- Predictions: `data/features/predictive_hmm_predictions.parquet`
- Cost scenario: `base`

## Net Metrics

| metric | value |
| --- | --- |
| trades | 160 |
| net_return | -0.006774 |
| gross_return | 0.009226 |
| total_cost | 0.016000 |
| daily_sharpe_net | -1.289799 |
| max_drawdown | 0.023907 |
| profit_factor | 0.906848 |
| hit_ratio | 0.493750 |
| avg_trade_net | -0.000042 |
| median_trade_net | -0.000018 |
| turnover_trades_per_day | 4.571429 |
| exposure | 0.088462 |

## Calibration

| metric | value |
| --- | --- |
| rows | 663 |
| log_loss | 0.926168 |
| brier_multiclass | 0.538099 |
| expected_calibration_error | 0.036347 |
| avg_confidence | 0.594121 |
| accuracy | 0.630468 |

## PnL Long Vs Short

| side | trades | net_return | avg_trade_net | hit_ratio |
| --- | --- | --- | --- | --- |
| long | 81 | -0.006303 | -0.000078 | 0.506173 |
| short | 79 | -0.000471 | -0.000006 | 0.481013 |

## PnL By Regime

| hmm_state | trades | net_return | avg_trade_net | hit_ratio |
| --- | --- | --- | --- | --- |
| 0 | 65 | -0.011582 | -0.000178 | 0.415385 |
| 1 | 27 | 0.000430 | 0.000016 | 0.629630 |
| 2 | 30 | 0.007241 | 0.000241 | 0.633333 |
| 3 | 38 | -0.002863 | -0.000075 | 0.421053 |

## PnL By Hour

| entry_hour | trades | net_return | avg_trade_net | hit_ratio |
| --- | --- | --- | --- | --- |
| 11 | 12 | -0.004159 | -0.000347 | 0.250000 |
| 12 | 32 | 0.004766 | 0.000149 | 0.625000 |
| 13 | 38 | -0.006437 | -0.000169 | 0.447368 |
| 14 | 22 | 0.014091 | 0.000640 | 0.727273 |
| 15 | 56 | -0.015035 | -0.000268 | 0.410714 |

## PnL By Fold

_No rows._


No real walk-forward folds are available with the current dataset. The configured schema needs 5 train month(s), 1 validation month(s), and 1 test month(s).

## Notes

- Daily Sharpe is computed from net daily PnL and annualized with 252 sessions.
- Max drawdown is computed from the additive net equity curve.
- Turnover is reported as completed trades per day.
- Exposure is estimated as held bars divided by cleaned regular-session bars.

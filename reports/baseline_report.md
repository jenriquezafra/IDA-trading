# Baseline Report

## Scope

- Input labels: `data/features/labels.parquet`
- Rows: 3780
- Sessions: 60
- Period: `2026-02-03 10:30:00-05:00` to `2026-04-29 15:40:00-04:00`
- Horizon: 2 bars
- Execution: signal at close `t`, entry at `open_(t+1)`, exit at `open_(t+h+1)`
- Round-trip cost: 1.00 bps

## Results

| strategy | rows | trades | exposure | net_return | gross_return | total_cost | avg_trade_net | median_trade_net | hit_ratio | profit_factor | daily_sharpe | max_daily_loss | downside_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_flat | 3780 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  | 0.000000 | 0 |
| random | 3780 | 2549 | 0.674339 | -0.268919 | -0.014019 | 0.254900 | -0.000105 | -0.000136 | 0.430757 | 0.759631 | -9.325839 | -0.028901 | 49 |
| intraday_buy_hold | 3780 | 3780 | 1.000000 | -0.352394 | 0.025606 | 0.378000 | -0.000093 | -0.000085 | 0.457407 | 0.785447 | -7.636584 | -0.034720 | 45 |
| momentum | 3780 | 1788 | 0.473016 | -0.197385 | -0.018585 | 0.178800 | -0.000110 | -0.000100 | 0.451902 | 0.758336 | -7.134885 | -0.019429 | 44 |
| reversion | 3780 | 1788 | 0.473016 | -0.160215 | 0.018585 | 0.178800 | -0.000090 | -0.000100 | 0.453579 | 0.799345 | -5.598611 | -0.026786 | 39 |

## Notes

- `always_flat` is the zero-risk benchmark.
- `intraday_buy_hold` is long on every eligible label row over the same next-open horizon.
- `momentum` follows `ret_3` when it exceeds the neutral-zone threshold.
- `reversion` takes the opposite side of the same threshold rule.
- `random` uses a fixed seed for reproducibility.

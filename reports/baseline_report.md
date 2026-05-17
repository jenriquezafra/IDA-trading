# Baseline Report

## Scope

- Input labels: `data/features/labels.parquet`
- Rows: 78498
- Sessions: 1246
- Period: `2021-05-03 10:30:00-04:00` to `2026-05-01 15:40:00-04:00`
- Horizon: 2 bars
- Execution: signal at close `t`, entry at `open_(t+1)`, exit at `open_(t+h+1)`
- Round-trip cost: 1.00 bps

## Results

| strategy | rows | trades | exposure | net_return | gross_return | total_cost | avg_trade_net | median_trade_net | hit_ratio | profit_factor | daily_sharpe | max_daily_loss | downside_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_flat | 78498 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |  | 0.000000 | 0 |
| random | 78498 | 52307 | 0.666348 | -4.856032 | 0.374668 | 5.230700 | -0.000093 | -0.000100 | 0.442866 | 0.791295 | -7.779481 | -0.037340 | 949 |
| intraday_buy_hold | 78498 | 78498 | 1.000000 | -7.538451 | 0.311349 | 7.849800 | -0.000096 | -0.000074 | 0.458636 | 0.783280 | -6.586682 | -0.128889 | 901 |
| momentum | 78498 | 37119 | 0.472866 | -3.784921 | -0.073021 | 3.711900 | -0.000102 | -0.000100 | 0.441445 | 0.773577 | -6.115493 | -0.051942 | 909 |
| reversion | 78498 | 37119 | 0.472866 | -3.638879 | 0.073021 | 3.711900 | -0.000098 | -0.000100 | 0.444220 | 0.781299 | -5.712170 | -0.068777 | 850 |

## Notes

- `always_flat` is the zero-risk benchmark.
- `intraday_buy_hold` is long on every eligible label row over the same next-open horizon.
- `momentum` follows `ret_3` when it exceeds the neutral-zone threshold.
- `reversion` takes the opposite side of the same threshold rule.
- `random` uses a fixed seed for reproducibility.

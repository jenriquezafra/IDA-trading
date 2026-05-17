# Backtest Report

## Scope

- Signals: `data/features/signals.parquet`
- Cost scenario: `base`
- Entry: next open only
- Time stop bars: 2
- Stop loss: 10.00 bps

## Summary

| metric | value |
| --- | ---: |
| trades | 102 |
| gross_return | 0.000622 |
| net_return | -0.009578 |
| total_cost | 0.010200 |
| hit_ratio | 0.274510 |
| avg_trade_net | -0.000094 |
| max_daily_loss | -0.008800 |

## Exit Reasons

| exit_reason | trades |
| --- | ---: |
| stop | 69 |
| time | 32 |
| force_flat | 1 |

## Notes

- PnL metrics are net of entry and exit costs unless explicitly labelled gross.
- Entries are scheduled from signal bar `t` and executed at `open_(t+1)`.
- Trades are forced flat before the configured intraday cutoff and no overnight positions are allowed.

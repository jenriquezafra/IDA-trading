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
| trades | 160 |
| gross_return | 0.009226 |
| net_return | -0.006774 |
| total_cost | 0.016000 |
| hit_ratio | 0.493750 |
| avg_trade_net | -0.000042 |
| max_daily_loss | -0.007465 |

## Exit Reasons

| exit_reason | trades |
| --- | ---: |
| time | 96 |
| stop | 61 |
| force_flat | 3 |

## Notes

- PnL metrics are net of entry and exit costs unless explicitly labelled gross.
- Entries are scheduled from signal bar `t` and executed at `open_(t+1)`.
- Trades are forced flat before the configured intraday cutoff and no overnight positions are allowed.

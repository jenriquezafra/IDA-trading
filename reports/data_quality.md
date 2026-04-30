# Data Quality Report

## Inputs

- Input: `data/raw/spy_5min.parquet`
- Output: `data/cleaned/spy_5min_clean.parquet`
- Start timestamp: `2026-02-03T09:30:00-05:00`
- End timestamp: `2026-04-29T15:55:00-04:00`

## Row Counts

| Metric | Value |
| --- | ---: |
| Raw rows | 4680 |
| Clean rows | 4680 |
| Dropped rows | 0 |

## Checks

| Check | Count |
| --- | ---: |
| Duplicate timestamps | 0 |
| Critical NaN rows | 0 |
| Invalid price rows | 0 |
| Negative volume rows | 0 |
| Extreme range rows | 0 |
| Out-of-session rows | 0 |
| Rows on non-trading sessions | 0 |
| Dropped half-day rows | 0 |
| Dropped incomplete-session rows | 0 |
| Rows whose target would cross session close | 180 |
| Rows where a new trade cannot be opened | 180 |
| Force-flat bars | 60 |

## Non-Trading Sessions

- Ninguna

## Half-Day Sessions

- Ninguna

## Incomplete Sessions

- Ninguna

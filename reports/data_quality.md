# Data Quality Report

## Inputs

- Input: `data/raw/spy_5min.parquet`
- Output: `data/cleaned/spy_5min_clean.parquet`
- Start timestamp: `2021-05-03T09:30:00-04:00`
- End timestamp: `2026-05-01T15:55:00-04:00`

## Row Counts

| Metric | Value |
| --- | ---: |
| Raw rows | 237156 |
| Clean rows | 97188 |
| Dropped rows | 139968 |

## Checks

| Check | Count |
| --- | ---: |
| Duplicate timestamps | 0 |
| Critical NaN rows | 0 |
| Invalid price rows | 0 |
| Negative volume rows | 0 |
| Extreme range rows | 0 |
| Out-of-session rows | 139548 |
| Rows on non-trading sessions | 0 |
| Dropped half-day rows | 420 |
| Dropped incomplete-session rows | 0 |
| Rows whose target would cross session close | 3738 |
| Rows where a new trade cannot be opened | 3738 |
| Force-flat bars | 1246 |

## Non-Trading Sessions

- Ninguna

## Half-Day Sessions

- 2021-11-26
- 2022-11-25
- 2023-07-03
- 2023-11-24
- 2024-07-03
- 2024-11-29
- 2024-12-24
- 2025-07-03
- 2025-11-28
- 2025-12-24

## Incomplete Sessions

- Ninguna

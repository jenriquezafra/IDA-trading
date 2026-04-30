# Regime Diagnostics

## Scope

- Input features: `data/features/features_base.parquet`
- Output features: `data/features/hmm_features.parquet`
- Filtered rows: 3894
- HMM columns: `ret_1, ret_3, rv_6, rv_12, range, rel_volume, trend_12, intraday_drawdown`
- Main model: K=4, covariance `diag`, seed 42
- Train sessions: 41 (`2026-02-04` to `2026-04-02`)
- Test sessions: 18 (`2026-04-06` to `2026-04-29`)

## State Diagnostics

| state | count | occupancy | mean_ret_1 | mean_rv_12 | mean_entropy | mean_duration | runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 772 | 0.198254 | -0.000154 | 0.002175 | 0.114893 | 6.031250 | 128 |
| 1 | 1432 | 0.367745 | 0.000033 | 0.001420 | 0.050477 | 10.302158 | 139 |
| 2 | 674 | 0.173087 | 0.000031 | 0.004246 | 0.091236 | 4.585034 | 147 |
| 3 | 1016 | 0.260914 | 0.000086 | 0.003046 | 0.125636 | 5.183673 | 196 |

## Transition Matrix

| from_state | to_0 | to_1 | to_2 | to_3 |
| --- | --- | --- | --- | --- |
| 0 | 0.907707 | 0.047714 | 0.029306 | 0.015272 |
| 1 | 0.058529 | 0.915954 | 0.011916 | 0.013601 |
| 2 | 0.025457 | 0.000000 | 0.830977 | 0.143566 |
| 3 | 0.021033 | 0.042722 | 0.066607 | 0.869638 |

## Stability

| n_states | seed | train_avg_loglik | test_avg_loglik | converged | iterations |
| --- | --- | --- | --- | --- | --- |
| 2 | 42 | -9.476880 | -7.727001 | True | 21 |
| 2 | 7 | -9.476880 | -7.726713 | True | 13 |
| 2 | 123 | -9.476881 | -7.727565 | True | 17 |
| 3 | 42 | -8.639331 | -6.014862 | True | 33 |
| 3 | 7 | -8.639331 | -6.014899 | True | 36 |
| 3 | 123 | -8.639331 | -6.014971 | True | 31 |
| 4 | 42 | -8.176030 | -5.807321 | True | 39 |
| 4 | 7 | -8.176030 | -5.807125 | True | 37 |
| 4 | 123 | -8.176029 | -5.808052 | True | 36 |
| 5 | 42 | -7.836396 | -5.365963 | True | 72 |
| 5 | 7 | -7.829970 | -5.655824 | True | 31 |
| 5 | 123 | -7.829971 | -5.655204 | True | 32 |
| 6 | 42 | -7.571996 | -5.160067 | True | 100 |
| 6 | 7 | -7.562357 | -5.207100 | True | 100 |
| 6 | 123 | -7.578519 | -5.432729 | True | 40 |

## Notes

- Scaling is fit only on train sessions.
- HMM training uses session `lengths`.
- Probabilities are causal forward-filtered and reset at each session.
- The report uses a chronological train/test split for diagnostics; walk-forward validation is implemented later.

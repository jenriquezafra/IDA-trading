# Regime Diagnostics

## Scope

- Input features: `data/features/features_base.parquet`
- Output features: `data/features/hmm_features.parquet`
- Filtered rows: 82170
- HMM columns: `ret_1, ret_3, rv_6, rv_12, range, rel_volume, trend_12, intraday_drawdown`
- Main model: K=4, covariance `diag`, seed 42
- Train sessions: 871 (`2021-05-04` to `2024-10-24`)
- Test sessions: 374 (`2024-10-25` to `2026-05-01`)

## State Diagnostics

| state | count | occupancy | mean_ret_1 | mean_rv_12 | mean_entropy | mean_duration | runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 27692 | 0.337009 | 0.000022 | 0.001864 | 0.085552 | 6.808950 | 4067 |
| 1 | 24144 | 0.293830 | -0.000016 | 0.003183 | 0.082854 | 6.732850 | 3586 |
| 2 | 9544 | 0.116149 | -0.000040 | 0.005835 | 0.065633 | 5.456832 | 1749 |
| 3 | 20790 | 0.253012 | 0.000018 | 0.001060 | 0.049344 | 10.590932 | 1963 |

## Transition Matrix

| from_state | to_0 | to_1 | to_2 | to_3 |
| --- | --- | --- | --- | --- |
| 0 | 0.905423 | 0.044251 | 0.005913 | 0.044414 |
| 1 | 0.061759 | 0.903192 | 0.035049 | 0.000000 |
| 2 | 0.006182 | 0.101688 | 0.885692 | 0.006438 |
| 3 | 0.059554 | 0.002964 | 0.004660 | 0.932822 |

## Stability

| n_states | seed | train_avg_loglik | test_avg_loglik | converged | iterations |
| --- | --- | --- | --- | --- | --- |
| 2 | 42 | -8.189647 | -8.562631 | True | 12 |
| 2 | 7 | -8.189647 | -8.562632 | True | 25 |
| 2 | 123 | -8.189647 | -8.562633 | True | 25 |
| 3 | 42 | -7.081879 | -7.071205 | True | 46 |
| 3 | 7 | -7.081879 | -7.071194 | True | 52 |
| 3 | 123 | -7.081879 | -7.071219 | True | 53 |
| 4 | 42 | -6.529088 | -6.334623 | True | 83 |
| 4 | 7 | -6.529088 | -6.334640 | True | 94 |
| 4 | 123 | -6.529088 | -6.334633 | True | 86 |
| 5 | 42 | -6.196655 | -5.868031 | True | 93 |
| 5 | 7 | -6.219607 | -5.986012 | True | 78 |
| 5 | 123 | -6.243623 | -6.079279 | True | 92 |
| 6 | 42 | -5.926703 | -5.612517 | True | 100 |
| 6 | 7 | -5.925813 | -5.607804 | True | 100 |
| 6 | 123 | -5.936306 | -5.563783 | True | 100 |

## Notes

- Scaling is fit only on train sessions.
- HMM training uses session `lengths`.
- Probabilities are causal forward-filtered and reset at each session.
- The report uses a chronological train/test split for diagnostics; walk-forward validation is implemented later.

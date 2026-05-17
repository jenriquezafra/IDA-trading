# Robustness Report

## Evidence Status

- Available months: 61 (`2021-05,2021-06,2021-07,2021-08,2021-09,2021-10,2021-11,2021-12,2022-01,2022-02,2022-03,2022-04,2022-05,2022-06,2022-07,2022-08,2022-09,2022-10,2022-11,2022-12,2023-01,2023-02,2023-03,2023-04,2023-05,2023-06,2023-07,2023-08,2023-09,2023-10,2023-11,2023-12,2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10,2024-11,2024-12,2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05`)
- Required months for one configured fold: 7
- Generated walk-forward folds: 55
- Interpretation: Full walk-forward robustness can be interpreted.

## Horizon Sensitivity

| horizon_bars | rows | sessions | target_down_pct | target_neutral_pct | target_up_pct | avg_abs_fwd_ret | median_neutral_zone | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 79744 | 1246 | 0.182747 | 0.628298 | 0.188955 | 0.000562 | 0.000514 | exploratory_current_data |
| 2 | 78498 | 1246 | 0.180157 | 0.631099 | 0.188744 | 0.000788 | 0.000728 | exploratory_current_data |
| 3 | 77252 | 1246 | 0.180112 | 0.633628 | 0.186261 | 0.000961 | 0.000892 | exploratory_current_data |

## Cost Stress

| split | cost_bps | rows | trades | exposure | gross_return | total_cost | net_return | avg_trade_net | hit_ratio | daily_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.000000 | 38097 | 170 | 0.004462 | 0.076813 | 0.017000 | 0.059813 | 0.000352 | 0.517647 | 0.791637 |
| train | 2.000000 | 38097 | 170 | 0.004462 | 0.076813 | 0.034000 | 0.042813 | 0.000252 | 0.505882 | 0.579952 |
| train | 5.000000 | 38097 | 170 | 0.004462 | 0.076813 | 0.085000 | -0.008187 | -0.000048 | 0.429412 | -0.115237 |
| validation | 1.000000 | 12699 | 5 | 0.000394 | 0.001583 | 0.000500 | 0.001083 | 0.000217 | 0.400000 | 1.006006 |
| validation | 2.000000 | 12699 | 5 | 0.000394 | 0.001583 | 0.001000 | 0.000583 | 0.000117 | 0.400000 | 1.006006 |
| validation | 5.000000 | 12699 | 5 | 0.000394 | 0.001583 | 0.002500 | -0.000917 | -0.000183 | 0.200000 | -1.006006 |
| test | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| test | 2.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| test | 5.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |

## Threshold Sensitivity Top 20

| theta_prob | theta_score | max_neutral | max_hmm_entropy | rows | trades | exposure | gross_return | total_cost | net_return | avg_trade_net | hit_ratio | daily_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.450000 | 0.000000 | 0.550000 | 0.500000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.550000 | 0.700000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.550000 | 0.900000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.550000 | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.650000 | 0.500000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.650000 | 0.700000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.650000 | 0.900000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.650000 | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.750000 | 0.500000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.750000 | 0.700000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.750000 | 0.900000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.000000 | 0.750000 | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.550000 | 0.500000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.550000 | 0.700000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.550000 | 0.900000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.550000 | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.650000 | 0.500000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.650000 | 0.700000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.650000 | 0.900000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| 0.450000 | 0.050000 | 0.650000 | 1.000000 | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |

## PnL By Regime

| hmm_state | trades | net_return | avg_trade_net | hit_ratio |
| --- | --- | --- | --- | --- |
| 0 | 5 | -0.004083 | -0.000817 | 0.200000 |
| 1 | 14 | 0.003628 | 0.000259 | 0.571429 |
| 2 | 82 | -0.008023 | -0.000098 | 0.231707 |
| 3 | 1 | -0.001100 | -0.001100 | 0.000000 |

## PnL By Hour

| entry_hour | trades | net_return | avg_trade_net | hit_ratio |
| --- | --- | --- | --- | --- |
| 11 | 3 | 0.003719 | 0.001240 | 0.666667 |
| 12 | 20 | -0.015508 | -0.000775 | 0.150000 |
| 13 | 11 | -0.006473 | -0.000588 | 0.272727 |
| 14 | 17 | 0.019865 | 0.001169 | 0.411765 |
| 15 | 51 | -0.011181 | -0.000219 | 0.254902 |

## Experiment Plan

| family | parameter | value | status |
| --- | --- | --- | --- |
| horizon | horizon_bars | 1 | profiled_current_data |
| horizon | horizon_bars | 2 | profiled_current_data |
| horizon | horizon_bars | 3 | profiled_current_data |
| hmm_k | n_states | 2 | ready_for_full_rerun |
| hmm_k | n_states | 3 | ready_for_full_rerun |
| hmm_k | n_states | 4 | ready_for_full_rerun |
| hmm_k | n_states | 5 | ready_for_full_rerun |
| hmm_k | n_states | 6 | ready_for_full_rerun |
| cost | round_trip_cost_bps | 1.0 | stress_replayed_current_signals |
| cost | round_trip_cost_bps | 2.0 | stress_replayed_current_signals |
| cost | round_trip_cost_bps | 5.0 | stress_replayed_current_signals |
| seed | random_state | 42 | ready_for_full_rerun |
| seed | random_state | 7 | ready_for_full_rerun |
| seed | random_state | 123 | ready_for_full_rerun |
| training_window | train_months | 3 | ready_for_full_rerun |
| training_window | train_months | 5 | ready_for_full_rerun |
| training_window | train_months | 9 | ready_for_full_rerun |
| period | period | all | ready_for_full_rerun |
| regime | hmm_state | all_states | profiled_current_trades |
| hour | entry_hour | all_hours | profiled_current_trades |

## Future Rerun Checklist

- Load enough intraday history for the configured walk-forward schema before treating robustness as evidence.
- Rerun HMM K, seed, training-window, and period experiments end to end after loading long history.
- Keep threshold and cost stress separate from threshold selection; do not optimize on test.
- Recreate this report with `python -m src.robustness --config configs/base.yaml` after replacing the dataset.

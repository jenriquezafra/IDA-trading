# Ablation Report

## Evidence Status

- Available months: 61 (`2021-05,2021-06,2021-07,2021-08,2021-09,2021-10,2021-11,2021-12,2022-01,2022-02,2022-03,2022-04,2022-05,2022-06,2022-07,2022-08,2022-09,2022-10,2022-11,2022-12,2023-01,2023-02,2023-03,2023-04,2023-05,2023-06,2023-07,2023-08,2023-09,2023-10,2023-11,2023-12,2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10,2024-11,2024-12,2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05`)
- Required months for one configured fold: 7
- Generated walk-forward folds: 55
- Interpretation: Full walk-forward ablation can be interpreted.

## Current Test Comparison

| variant | test_net_return | delta_net_return_vs_base | test_trades | test_avg_trade_net | test_hit_ratio | test_log_loss | test_accuracy | test_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_no_hmm | 0.000000 | 0.000000 | 0 | 0.000000 |  | 0.907742 | 0.630995 | 0.008315 |
| hmm_all_features_no_filter | 0.000000 | 0.000000 | 0 | 0.000000 |  | 0.907797 | 0.630995 | 0.009785 |
| hmm_all_features_with_filter | 0.000000 | 0.000000 | 0 | 0.000000 |  | 0.907797 | 0.630995 | 0.009785 |

## Decision Summary

HMM with filters matches base current test net return; this is only walk-forward evidence.

## Selected Signal Thresholds

| variant | theta_prob | theta_score | max_neutral | max_hmm_entropy |
| --- | --- | --- | --- | --- |
| base_no_hmm | 0.450000 | 0.100000 | 0.550000 | 999.000000 |
| hmm_all_features_no_filter | 0.500000 | 0.000000 | 0.550000 | 999.000000 |
| hmm_all_features_with_filter | 0.500000 | 0.000000 | 0.550000 | 0.500000 |

## Signal Metrics

| variant | split | rows | trades | exposure | gross_return | total_cost | net_return | avg_trade_net | hit_ratio | daily_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_no_hmm | train | 38097 | 389 | 0.010211 | 0.134443 | 0.038900 | 0.095543 | 0.000246 | 0.550129 | 0.908782 |
| base_no_hmm | validation | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| base_no_hmm | test | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| hmm_all_features_no_filter | train | 38097 | 170 | 0.004462 | 0.076813 | 0.017000 | 0.059813 | 0.000352 | 0.517647 | 0.791637 |
| hmm_all_features_no_filter | validation | 12699 | 5 | 0.000394 | 0.001583 | 0.000500 | 0.001083 | 0.000217 | 0.400000 | 1.006006 |
| hmm_all_features_no_filter | test | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |
| hmm_all_features_with_filter | train | 38097 | 170 | 0.004462 | 0.076813 | 0.017000 | 0.059813 | 0.000352 | 0.517647 | 0.791637 |
| hmm_all_features_with_filter | validation | 12699 | 5 | 0.000394 | 0.001583 | 0.000500 | 0.001083 | 0.000217 | 0.400000 | 1.006006 |
| hmm_all_features_with_filter | test | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  |

## Prediction Quality

| variant | split | rows | log_loss | brier_multiclass | expected_calibration_error | avg_confidence | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_no_hmm | train | 38097 | 1.075003 | 0.646993 | 0.054286 | 0.393134 | 0.447148 |
| base_no_hmm | validation | 12699 | 0.905682 | 0.527213 | 0.009770 | 0.636387 | 0.633357 |
| base_no_hmm | test | 12699 | 0.907742 | 0.529452 | 0.008315 | 0.637090 | 0.630995 |
| hmm_all_features_no_filter | train | 38097 | 1.074111 | 0.646164 | 0.048856 | 0.396077 | 0.444864 |
| hmm_all_features_no_filter | validation | 12699 | 0.905728 | 0.527243 | 0.011086 | 0.636237 | 0.633436 |
| hmm_all_features_no_filter | test | 12699 | 0.907797 | 0.529475 | 0.009785 | 0.638333 | 0.630995 |
| hmm_all_features_with_filter | train | 38097 | 1.074111 | 0.646164 | 0.048856 | 0.396077 | 0.444864 |
| hmm_all_features_with_filter | validation | 12699 | 0.905728 | 0.527243 | 0.011086 | 0.636237 | 0.633436 |
| hmm_all_features_with_filter | test | 12699 | 0.907797 | 0.529475 | 0.009785 | 0.638333 | 0.630995 |

## Experiment Plan

| variant | status | notes |
| --- | --- | --- |
| base_no_hmm | available_current_artifact | Uses predictive_base_predictions. |
| hmm_all_features_no_filter | available_current_artifact | Uses current HMM-feature model without entropy/regime signal filters. |
| hmm_all_features_with_filter | available_current_artifact | Uses current HMM-feature model with configured signal filters. |
| hard_hmm_state_only | ready_for_full_rerun | Requires retraining model with hard state one-hot and without HMM probabilities. |
| hmm_probabilities_only | ready_for_full_rerun | Requires retraining model with HMM probabilities and without hard state one-hot. |
| hmm_filters_only | ready_for_full_rerun | Requires base model plus HMM filters without HMM features in the model matrix. |
| separate_models_by_regime | ready_for_full_rerun | Requires enough per-regime data inside each train fold. |
| xgboost_no_hmm | blocked_until_block_17 | XGBoost challenger is the next roadmap block. |
| xgboost_with_hmm | blocked_until_block_17 | XGBoost challenger is the next roadmap block. |

## Future Rerun Checklist

- Retrain hard-state-only and probability-only HMM variants after loading long intraday history.
- Run separate per-regime models only when each train fold has enough observations by regime.
- Add XGBoost variants in block 17, then rerun this ablation report including those artifacts.
- Treat HMM as useful only if it improves the base model OOS after costs and across folds.

# Leakage Audit

## Summary

| status | count |
| --- | --- |
| PASS | 41 |

## Checks

| check_id | module | description | status | evidence |
| --- | --- | --- | --- | --- |
| features_no_future_label_columns | features | Feature file excludes target/future execution columns. | PASS | No forbidden future columns found. |
| feature_ret_1_past_close_only | features | ret_1 uses only close at t and prior same-session closes. | PASS | Recomputed grouped shift(1); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_ret_2_past_close_only | features | ret_2 uses only close at t and prior same-session closes. | PASS | Recomputed grouped shift(2); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_ret_3_past_close_only | features | ret_3 uses only close at t and prior same-session closes. | PASS | Recomputed grouped shift(3); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_ret_6_past_close_only | features | ret_6 uses only close at t and prior same-session closes. | PASS | Recomputed grouped shift(6); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_ret_12_past_close_only | features | ret_12 uses only close at t and prior same-session closes. | PASS | Recomputed grouped shift(12); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_rv_3_rolling_past_only | features | rv_3 uses rolling returns available through t only. | PASS | Recomputed same-session rolling window 3; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_rv_6_rolling_past_only | features | rv_6 uses rolling returns available through t only. | PASS | Recomputed same-session rolling window 6; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_rv_12_rolling_past_only | features | rv_12 uses rolling returns available through t only. | PASS | Recomputed same-session rolling window 12; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_rv_24_rolling_past_only | features | rv_24 uses rolling returns available through t only. | PASS | Recomputed same-session rolling window 24; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_vol_ratio_3_12_past_vol_only | features | vol_ratio_3_12 uses realized volatility windows available through t only. | PASS | Recomputed rv_3 / rv_12; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_vol_ratio_6_24_past_vol_only | features | vol_ratio_6_24 uses realized volatility windows available through t only. | PASS | Recomputed rv_6 / rv_24; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_signed_efficiency_12_rolling_past_only | features | signed_efficiency_12 uses same-session returns available through t only. | PASS | Recomputed rolling efficiency window 12; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_dir_persistence_12_rolling_past_only | features | dir_persistence_12 uses same-session return signs available through t only. | PASS | Recomputed rolling sign mean window 12; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_range_ratio_6_24_rolling_past_only | features | range_ratio_6_24 uses rolling same-session bar ranges available through t only. | PASS | Recomputed rolling mean range(6) / rolling mean range(24); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_dist_open_intraday_past_only | features | dist_open uses current bar and same-session cumulative information through t only. | PASS | Recomputed with session open and cumulative high/low; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_pos_session_range_intraday_past_only | features | pos_session_range uses current bar and same-session cumulative information through t only. | PASS | Recomputed with session open and cumulative high/low; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_dist_session_high_atr_intraday_past_only | features | dist_session_high_atr uses current bar and same-session cumulative information through t only. | PASS | Recomputed with session open and cumulative high/low; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_dist_session_low_atr_intraday_past_only | features | dist_session_low_atr uses current bar and same-session cumulative information through t only. | PASS | Recomputed with session open and cumulative high/low; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_intraday_runup_intraday_past_only | features | intraday_runup uses current bar and same-session cumulative information through t only. | PASS | Recomputed with session open and cumulative high/low; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_dist_vwap_atr_current_vwap_only | features | dist_vwap_atr uses cumulative VWAP through t and ATR through t. | PASS | Recomputed log(close / vwap) / atr_12; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_vwap_slope_12_past_vwap_only | features | vwap_slope_12 uses cumulative VWAP at t and t-12. | PASS | Recomputed grouped vwap shift(12); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| feature_rel_volume_prior_sessions_only | features | rel_volume uses only prior sessions for the same bar_index. | PASS | Recomputed expanding mean shifted by one session; max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| label_entry_next_open | labels | Labels enter at open_{t+1}. | PASS | Recomputed entry_px from grouped open shift(-1); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| label_exit_horizon_open | labels | Labels exit at open_{t+h+1}. | PASS | Recomputed exit_px from grouped open shift(-3); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| label_forward_return_alignment | labels | fwd_ret is computed from configured entry and exit opens. | PASS | Recomputed log(exit_px / entry_px); max_abs_diff=0.000e+00; tolerance=1.000e-10 |
| label_neutral_zone_ex_ante_vol | labels | neutral_zone uses ex-ante rv_12 at t plus cost/buffer floor. | PASS | Recomputed max(cost+buffer, lambda_vol * rv_12 * sqrt(h)); max_abs_diff=2.711e-20; tolerance=1.000e-10 |
| label_no_session_close_cross | labels | Dropped labels whose target would cross session close. | PASS | No target_crosses_session_close rows remain. |
| execution_next_open_only | backtest | Trades enter on the next bar open after signal t. | PASS | Checked 102 trades. |
| execution_no_close_t_fill | backtest | No trade uses close_t as execution fill. | PASS | Checked 102 trades. |
| execution_no_overnight | backtest | Trades open and close within the same session. | PASS | Checked 102 trades. |
| execution_costs_applied | backtest | Round-trip costs are applied to every executed trade. | PASS | Checked 102 trades at 1.0 bps. |
| fold_sessions_disjoint | walkforward | Train, validation and test sessions are disjoint per fold. | PASS | Checked 55 folds. |
| fold_sessions_chronological | walkforward | Folds are chronological: train < validation < test. | PASS | Checked 55 folds. |
| calendar_no_non_trading_sessions | calendar | Cleaned data contains only NYSE trading sessions. | PASS | Checked 1246 sessions. |
| calendar_half_days_dropped | calendar | Configured half-days are not present in cleaned full-session dataset. | PASS | Dropped half-days in range=10. |
| calendar_expected_bar_counts | calendar | Cleaned sessions have expected bar counts. | PASS | Checked 1246 sessions. |
| hmm_walkforward_fit_train_only | hmm | Walk-forward HMM fit uses fold train sessions only. | PASS | Source contains train_frame filtered by fold.train_sessions before fit_hmm_model. |
| model_walkforward_fit_train_only | model | Walk-forward predictive model fit uses train split only. | PASS | Source calls fit_base_model(train, ...) after split_frames. |
| thresholds_validation_only | signal | Threshold selection is delegated to validation-only selector. | PASS | Source calls select_thresholds_on_validation. |
| hmm_filtered_not_smoothed | hmm | HMM probabilities use causal forward filtering, not smoothing. | PASS | hmm_filter.filtered_probabilities implements forward recursion and session reset. |

## Violations

- None

## Conclusion

No material leakage violations detected by automated audit.

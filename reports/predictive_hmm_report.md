# Predictive Model With HMM Report

## Scope

- Input labels: `data/features/labels.parquet`
- Model: multinomial Logistic Regression with elastic-net regularization
- Calibration: `sigmoid` on validation split
- Train sessions: 747 (`2021-05-04` to `2024-04-26`)
- Validation sessions: 249 (`2024-04-29` to `2025-04-30`)
- Test sessions: 249 (`2025-05-01` to `2026-05-01`)

## Metrics

| split | rows | accuracy | balanced_accuracy | macro_f1 | log_loss | avg_p_down | avg_p_neutral | avg_p_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_uncalibrated | 38097 | 0.444864 | 0.401887 | 0.385531 | 1.074111 | 0.328792 | 0.342854 | 0.328354 |
| validation_calibrated | 12699 | 0.633436 | 0.333415 | 0.261158 | 0.905728 | 0.175868 | 0.635114 | 0.189018 |
| test_calibrated | 12699 | 0.630995 | 0.333333 | 0.257918 | 0.907797 | 0.176604 | 0.638333 | 0.185063 |


## Feature Columns

- `ret_1`
- `ret_2`
- `ret_3`
- `ret_6`
- `ret_12`
- `rv_3`
- `rv_6`
- `rv_12`
- `rv_24`
- `range`
- `atr_6`
- `atr_12`
- `sma_6`
- `sma_12`
- `sma_24`
- `trend_6`
- `trend_12`
- `trend_24`
- `vwap`
- `dist_vwap`
- `intraday_drawdown`
- `rel_volume`
- `sin_time`
- `cos_time`
- `minutes_to_close`
- `open_window`
- `close_window`
- `midday`
- `hmm_p0`
- `hmm_p1`
- `hmm_p2`
- `hmm_p3`
- `hmm_entropy`
- `hmm_max_prob`
- `hmm_state_0`
- `hmm_state_1`
- `hmm_state_2`
- `hmm_state_3`

## Notes

- Base engineered features plus HMM probabilities, entropy, max probability, and one-hot state are included.
- The scaler is fit only on train sessions.
- Calibration is fit only on validation sessions.
- Test metrics are reported once after calibration.

## OOS Metric Comparison vs Base

| metric | base | with_hmm | delta | improved |
| --- | --- | --- | --- | --- |
| accuracy | 0.630995 | 0.630995 | 0.000000 | False |
| balanced_accuracy | 0.333333 | 0.333333 | 0.000000 | False |
| macro_f1 | 0.257918 | 0.257918 | 0.000000 | False |
| log_loss | 0.907742 | 0.907797 | 0.000054 | False |

## Simple Test Score Rule Net Return

| model | test_rule_net_return | test_rule_trades |
| --- | --- | --- |
| base | -1.179687 | 12699 |
| with_hmm | -1.239787 | 12699 |

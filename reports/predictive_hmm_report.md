# Predictive Model With HMM Report

## Scope

- Input labels: `data/features/labels.parquet`
- Model: multinomial Logistic Regression with elastic-net regularization
- Calibration: `sigmoid` on validation split
- Train sessions: 35 (`2026-02-04` to `2026-03-25`)
- Validation sessions: 11 (`2026-03-26` to `2026-04-10`)
- Test sessions: 13 (`2026-04-13` to `2026-04-29`)

## Metrics

| split | rows | accuracy | balanced_accuracy | macro_f1 | log_loss | avg_p_down | avg_p_neutral | avg_p_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_uncalibrated | 1785 | 0.417367 | 0.438420 | 0.389906 | 1.056659 | 0.327155 | 0.345211 | 0.327634 |
| validation_calibrated | 561 | 0.645276 | 0.333333 | 0.261466 | 0.886193 | 0.182729 | 0.643598 | 0.173673 |
| test_calibrated | 663 | 0.630468 | 0.333333 | 0.257786 | 0.926168 | 0.237262 | 0.594121 | 0.168618 |


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
| accuracy | 0.630468 | 0.630468 | 0.000000 | False |
| balanced_accuracy | 0.333333 | 0.333333 | 0.000000 | False |
| macro_f1 | 0.257786 | 0.257786 | 0.000000 | False |
| log_loss | 0.923610 | 0.926168 | 0.002558 | False |

## Simple Test Score Rule Net Return

| model | test_rule_net_return | test_rule_trades |
| --- | --- | --- |
| base | -0.092243 | 663 |
| with_hmm | -0.092243 | 663 |

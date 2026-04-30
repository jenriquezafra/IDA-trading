# Predictive Base Model Report

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
| train_uncalibrated | 1785 | 0.410084 | 0.419327 | 0.377933 | 1.060096 | 0.326964 | 0.345070 | 0.327966 |
| validation_calibrated | 561 | 0.645276 | 0.333333 | 0.261466 | 0.887553 | 0.182717 | 0.643586 | 0.173696 |
| test_calibrated | 663 | 0.630468 | 0.333333 | 0.257786 | 0.923610 | 0.231516 | 0.597041 | 0.171443 |

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

## Notes

- HMM features are excluded.
- The scaler is fit only on train sessions.
- Calibration is fit only on validation sessions.
- Test metrics are reported once after calibration.

# Predictive Base Model Report

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
| train_uncalibrated | 38097 | 0.447148 | 0.400129 | 0.385162 | 1.075003 | 0.328989 | 0.342465 | 0.328545 |
| validation_calibrated | 12699 | 0.633357 | 0.333308 | 0.260906 | 0.905682 | 0.175768 | 0.635288 | 0.188944 |
| test_calibrated | 12699 | 0.630995 | 0.333333 | 0.257918 | 0.907742 | 0.176878 | 0.637090 | 0.186031 |


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

# Predictive XGBoost Report

## Scope

- Input labels: `data/features/labels.parquet`
- Model: XGBoost multiclass challenger
- Calibration: `sigmoid` on validation split
- Train sessions: 747 (`2021-05-04` to `2024-04-26`)
- Validation sessions: 249 (`2024-04-29` to `2025-04-30`)
- Test sessions: 249 (`2025-05-01` to `2026-05-01`)

## Regularization

- max_depth: `3`
- n_estimators: `250`
- learning_rate: `0.03`
- min_child_weight: `25.0`
- subsample: `0.8`
- colsample_bytree: `0.8`
- reg_alpha: `0.1`
- reg_lambda: `5.0`

## Probability Metrics

| split | rows | accuracy | balanced_accuracy | macro_f1 | log_loss | avg_p_down | avg_p_neutral | avg_p_up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_uncalibrated | 38097 | 0.613382 | 0.334526 | 0.256124 | 0.901981 | 0.189842 | 0.612953 | 0.197205 |
| validation_calibrated | 12699 | 0.634696 | 0.334487 | 0.262689 | 0.895093 | 0.176359 | 0.634092 | 0.189549 |
| test_calibrated | 12699 | 0.631231 | 0.335497 | 0.263813 | 0.896575 | 0.184668 | 0.614527 | 0.200805 |

## Logistic Regression Comparison

| metric | logistic_regression | xgboost | delta | improved |
| --- | --- | --- | --- | --- |
| accuracy | 0.630995 | 0.631231 | 0.000236 | True |
| balanced_accuracy | 0.333333 | 0.335497 | 0.002164 | True |
| macro_f1 | 0.257918 | 0.263813 | 0.005895 | True |
| log_loss | 0.907742 | 0.896575 | -0.011167 | True |

## Selected Signal Thresholds

- theta_prob: 0.4500
- theta_score: 0.0000
- max_neutral: 0.5500
- max_hmm_entropy: 0.5000

## Signal Metrics

| split | rows | trades | exposure | gross_return | total_cost | net_return | avg_trade_net | hit_ratio | daily_sharpe | turnover_trades_per_day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 38097 | 1 | 0.000026 | 0.003529 | 0.000100 | 0.003429 | 0.003429 | 1.000000 | 0.580818 | 0.001339 |
| validation | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  | 0.000000 |
| test | 12699 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  | 0.000000 |

## Validation Grid Top 10

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

## Decision Guardrails

- Treat XGBoost as useful only if it improves validation and test, not train only.
- Probability calibration and threshold selection are fit only on validation rows.
- The configured model limits depth and uses shrinkage, subsampling, column sampling, L1, and L2 regularization.

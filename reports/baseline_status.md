# Baseline Status

## Scope

- Config: `configs/base.yaml`
- Coste base: 1.00 bps round-trip
- Objetivo: congelar el baseline negativo actual como referencia reproducible antes de explorar nuevas hipotesis HMM-first.

## Unified Metrics

| strategy | cost_bps | trades | net_return | daily_sharpe_net | profit_factor_net | avg_trade_net | max_drawdown | folds_positive | folds_negative | beats_always_flat | beats_random | beats_momentum | beats_reversion | beats_model_without_hmm | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_flat | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | no | yes | yes | yes | no | benchmark |
| random | 1.000000 | 52307 | -4.856032 | -7.779481 | 0.791295 | -0.000093 | 4.859001 | 4 | 57 | no | no | no | no | no | rejected_economic |
| intraday_buy_hold | 1.000000 | 78498 | -7.538451 | -6.586682 | 0.783280 | -0.000096 | 7.546438 | 1 | 60 | no | no | no | no | no | rejected_economic |
| momentum | 1.000000 | 37119 | -3.784921 | -6.115493 | 0.773577 | -0.000102 | 3.790514 | 4 | 57 | no | yes | no | no | no | rejected_economic |
| reversion | 1.000000 | 37119 | -3.638879 | -5.712170 | 0.781299 | -0.000098 | 3.643728 | 3 | 58 | no | yes | yes | no | no | rejected_economic |
| hmm_lr_static_backtest | 1.000000 | 102 | -0.009578 | -0.831806 | 0.877254 | -0.000094 | 0.025515 | 5 | 13 | no | yes | yes | yes | no | rejected_economic |
| base_no_hmm_static_signal | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | no | yes | yes | yes | no | rejected_no_oos_trades |
| xgboost_static_signal | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | no | yes | yes | yes | no | rejected_no_oos_trades |
| hmm_lr_walkforward_oos | 1.000000 | 46 | -0.031415 | -0.485786 | 0.443821 | -0.000683 | 0.036760 | 5 | 8 | no | yes | yes | yes | no | rejected_economic |

## Sources

| strategy | source |
| --- | --- |
| always_flat | reports/baseline_trades.parquet |
| random | reports/baseline_trades.parquet |
| intraday_buy_hold | reports/baseline_trades.parquet |
| momentum | reports/baseline_trades.parquet |
| reversion | reports/baseline_trades.parquet |
| hmm_lr_static_backtest | reports/backtest_trades.parquet |
| base_no_hmm_static_signal | data/features/predictive_base_predictions.parquet |
| xgboost_static_signal | data/features/predictive_xgboost_predictions.parquet |
| hmm_lr_walkforward_oos | reports/walkforward/fold_*/signals.parquet |

## Closed Branches

- `hmm_lr_static_backtest`: rechazado economicamente con la configuracion actual porque el PnL neto y el avg trade net son negativos.
- `hmm_lr_walkforward_oos`: rechazado economicamente con la configuracion actual porque el PnL neto OOS agregado es negativo.
- `Logistic Regression con HMM como feature plana`: no queda aceptada como edge; solo queda como referencia negativa.
- `XGBoost estatico`: mejora metricas predictivas estaticas, pero no ha generado senales utiles con el grid actual.

## Conclusion

Baseline actual rechazado economicamente salvo nueva evidencia HMM-first.

Este reporte no rechaza definitivamente la hipotesis HMM-first. Rechaza el baseline actual y obliga a que cualquier nueva rama demuestre mejora neta frente a esta tabla, siempre con costes, walk-forward y sin optimizar en test.

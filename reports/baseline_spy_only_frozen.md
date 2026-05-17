# Frozen SPY-only HMM Baseline

## Status

- Baseline id: `spy_only_hmm`
- Target: `SPY`
- Timeframe: `5min`
- Provider: `polygon`
- Final status: `rejected_cost_fragile`
- Generated UTC: `2026-05-02T21:06:00.990738+00:00`

## Conclusion

SPY-only direct directional edge is not accepted. Legacy predictive baselines are rejected; best HMM fallback is valid at 1 bps but cost-fragile at 2 bps.

The frozen baseline is a comparison target for the cross-asset HMM branch. It is not an accepted strategy.

## Data Snapshot

- Cleaned file: `data/cleaned/spy_5min_clean.parquet`
- Rows: `97188`
- Sessions: `1246`
- Start: `2021-05-03T09:30:00-04:00`
- End: `2026-05-01T15:55:00-04:00`

## Legacy Baselines

| strategy | cost_bps | trades | net_return | daily_sharpe_net | profit_factor_net | avg_trade_net | max_drawdown | folds_positive | folds_negative | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_flat | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | benchmark |
| random | 1.000000 | 52307 | -4.856032 | -7.779481 | 0.791295 | -0.000093 | 4.859001 | 4 | 57 | rejected_economic |
| intraday_buy_hold | 1.000000 | 78498 | -7.538451 | -6.586682 | 0.783280 | -0.000096 | 7.546438 | 1 | 60 | rejected_economic |
| momentum | 1.000000 | 37119 | -3.784921 | -6.115493 | 0.773577 | -0.000102 | 3.790514 | 4 | 57 | rejected_economic |
| reversion | 1.000000 | 37119 | -3.638879 | -5.712170 | 0.781299 | -0.000098 | 3.643728 | 3 | 58 | rejected_economic |
| hmm_lr_static_backtest | 1.000000 | 102 | -0.009578 | -0.831806 | 0.877254 | -0.000094 | 0.025515 | 5 | 13 | rejected_economic |
| base_no_hmm_static_signal | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | rejected_no_oos_trades |
| xgboost_static_signal | 1.000000 | 0 | 0.000000 |  |  | 0.000000 | 0.000000 | 0 | 0 | rejected_no_oos_trades |
| hmm_lr_walkforward_oos | 1.000000 | 46 | -0.031415 | -0.485786 | 0.443821 | -0.000683 | 0.036760 | 5 | 8 | rejected_economic |

## HMM Candidate Decisions

| source_rank | candidate_id | feature_set | status_1bps | status_2bps | accepted | cost_fragile | test_net_1bps | test_net_2bps | test_drawdown_1bps | test_drawdown_2bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | minimal_vwap_location | candidate | weak_sharpe | no | yes | 0.311068 | 0.074447 | 0.063029 | 0.016915 |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | candidate | weak_profit_factor | no | yes | 0.118942 | 0.053342 | 0.073622 | 0.075322 |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | minimal_vwap_location | candidate | insufficient_trades | no | yes | 0.048779 | 0.014290 | 0.089986 | 0.006476 |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | candidate | insufficient_trades | no | yes | 0.202882 | 0.000000 | 0.063418 | 0.000000 |

## Leakage Audit

- Available: `True`
- Rows: `41`
- Critical failures: `0`
- Status counts: `{'PASS': 41}`

## Frozen Outputs

- `baselines/spy_only_hmm/config.yaml`
- `baselines/spy_only_hmm/results.parquet`
- `baselines/spy_only_hmm/summary.json`
- `baselines/spy_only_hmm/source_artifacts/`
- `reports/baseline_spy_only_frozen.md`

Result row groups: `{'hmm_candidate_threshold': 16, 'baseline_status': 9}`

## Freeze Policy

- Allowed future changes: Only verified bug fixes or report regeneration with identical methodology.
- Not allowed: Further parameter/feature/threshold optimization on SPY-only without a new written hypothesis.

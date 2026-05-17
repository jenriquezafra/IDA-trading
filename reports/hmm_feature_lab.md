# HMM Feature Lab

## Scope

- Feature sets: 12
- Max folds: `12`
- Horizons: `[2, 6]`
- Costs bps: `[1.0, 2.0]`
- HMM K: `4`
- HMM n_iter: `100`

## Feature Set Summary

| feature_set | n_features | columns | missing_columns | status | best_horizon_bars | best_cost_bps | best_hmm_state | best_action | best_total_trades | best_total_net_return | best_avg_trade_net | best_median_profit_factor | best_median_daily_sharpe | best_candidate_status | candidate_count | avg_state_frequency | avg_persistence | avg_mean_duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_default | 8 | ret_1,ret_3,rv_6,rv_12,range,rel_volume,trend_12,intraday_drawdown |  | ready | 6 | 1.000000 | 3 | momentum_ret_3 | 774 | 0.165299 | 0.000159 | 1.280337 | 2.523524 | candidate | 2 | 0.250000 | 0.839980 | 6.508889 |
| volatility_liquidity | 8 | rv_3,rv_6,rv_12,rv_24,range,atr_6,atr_12,rel_volume |  | ready | 6 | 1.000000 | 2 | momentum_ret_3 | 576 | 0.164995 | 0.000277 | 1.310077 | 2.645613 | candidate | 3 | 0.250000 | 0.877059 | 9.324942 |
| trend_structure | 7 | ret_1,ret_3,ret_6,trend_6,trend_12,trend_24,intraday_drawdown |  | ready | 6 | 1.000000 | 3 | short | 2820 | 0.408423 | 0.000159 | 1.097483 | 0.837027 | weak_profit_factor | 0 | 0.250000 | 0.736364 | 3.846195 |
| vwap_reversion | 6 | ret_1,ret_3,dist_vwap,intraday_drawdown,range,rel_volume |  | ready | 6 | 1.000000 | 0 | momentum_ret_3 | 851 | 0.129305 | 0.000036 | 0.969561 | -0.091072 | weak_profit_factor | 0 | 0.250000 | 0.815073 | 6.547646 |
| time_volatility | 8 | sin_time,cos_time,minutes_to_close,open_window,close_window,rv_6,rv_12,rel_volume |  | ready | 6 | 1.000000 | 2 | long | 2334 | 0.130190 | 0.000153 | 1.000651 | -0.141815 | weak_profit_factor | 0 | 0.250000 | 0.825082 | 12.521927 |
| minimal_interpretable | 5 | rv_12,range,rel_volume,trend_12,dist_vwap |  | ready | 6 | 1.000000 | 3 | momentum_ret_3 | 701 | 0.017005 | -0.000031 | 0.896240 | -1.143650 | negative_economic | 0 | 0.250000 | 0.858765 | 7.357131 |
| minimal_trend_efficiency | 5 | ret_3,trend_12,signed_efficiency_12,dir_persistence_12,rv_12 |  | ready | 6 | 1.000000 | 2 | momentum_ret_3 | 784 | 0.073751 | 0.000138 | 1.123745 | 0.681139 | weak_sharpe | 0 | 0.250000 | 0.836052 | 6.106466 |
| minimal_vwap_location | 6 | dist_vwap_atr,dist_open,pos_session_range,intraday_drawdown,rv_12,rel_volume |  | ready | 6 | 1.000000 | 3 | momentum_ret_3 | 737 | 0.180227 | 0.000209 | 1.315821 | 2.187013 | candidate | 2 | 0.250000 | 0.910469 | 10.692538 |
| minimal_compression_expansion | 6 | vol_ratio_6_24,range_ratio_6_24,signed_efficiency_12,pos_session_range,rv_12,range |  | ready | 6 | 1.000000 | 2 | momentum_ret_3 | 757 | 0.086120 | -0.000041 | 0.804265 | -3.284554 | negative_economic | 0 | 0.250000 | 0.893537 | 9.057855 |
| ablation_price_structure_only | 7 | ret_3,rv_12,vol_ratio_6_24,signed_efficiency_12,dist_vwap_atr,pos_session_range,intraday_drawdown |  | ready | 6 | 1.000000 | 3 | momentum_ret_3 | 685 | 0.008536 | -0.000039 | 0.879294 | -1.217141 | negative_economic | 0 | 0.250000 | 0.891670 | 8.765121 |
| rich_trend_vwap_structure | 9 | ret_3,ret_6,trend_12,trend_24,signed_efficiency_12,dir_persistence_12,dist_vwap_atr,vwap_slope_12,pos_session_range |  | ready | 6 | 1.000000 | 2 | momentum_ret_3 | 709 | 0.041416 | 0.000232 | 1.053095 | 0.651164 | weak_profit_factor | 0 | 0.250000 | 0.845059 | 6.380231 |
| rich_extreme_reversion | 9 | dist_session_high_atr,dist_session_low_atr,pos_session_range,intraday_drawdown,intraday_runup,dist_vwap_atr,rv_12,ret_3,rel_volume |  | ready | 6 | 1.000000 | 3 | momentum_ret_3 | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | candidate | 1 | 0.250000 | 0.882075 | 8.459195 |

## Top Validation Rankings

| feature_set | horizon_bars | cost_bps | hmm_state | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trend_structure | 6 | 1.000000 | 3 | short | 2820 | 0.408423 | 0.000159 | 1.097483 | 0.837027 | 8 | 4 | weak_profit_factor |
| minimal_vwap_location | 6 | 1.000000 | 3 | momentum_ret_3 | 737 | 0.180227 | 0.000209 | 1.315821 | 2.187013 | 7 | 5 | candidate |
| current_default | 6 | 1.000000 | 3 | momentum_ret_3 | 774 | 0.165299 | 0.000159 | 1.280337 | 2.523524 | 9 | 3 | candidate |
| volatility_liquidity | 6 | 1.000000 | 2 | momentum_ret_3 | 576 | 0.164995 | 0.000277 | 1.310077 | 2.645613 | 8 | 4 | candidate |
| minimal_vwap_location | 6 | 1.000000 | 1 | short | 3877 | 0.135371 | -0.000044 | 0.991041 | -0.068302 | 6 | 6 | negative_economic |
| time_volatility | 6 | 1.000000 | 2 | long | 2334 | 0.130190 | 0.000153 | 1.000651 | -0.141815 | 4 | 4 | weak_profit_factor |
| vwap_reversion | 6 | 1.000000 | 0 | momentum_ret_3 | 851 | 0.129305 | 0.000036 | 0.969561 | -0.091072 | 6 | 6 | weak_profit_factor |
| trend_structure | 6 | 2.000000 | 3 | short | 2820 | 0.126423 | 0.000059 | 1.003892 | 0.034721 | 7 | 5 | weak_profit_factor |
| volatility_liquidity | 6 | 2.000000 | 2 | momentum_ret_3 | 576 | 0.107395 | 0.000177 | 1.170670 | 1.598395 | 7 | 5 | candidate |
| minimal_vwap_location | 6 | 2.000000 | 3 | momentum_ret_3 | 737 | 0.106527 | 0.000109 | 1.196194 | 1.510023 | 7 | 5 | candidate |
| current_default | 6 | 2.000000 | 3 | momentum_ret_3 | 774 | 0.087899 | 0.000059 | 1.163650 | 1.491915 | 7 | 5 | candidate |
| minimal_compression_expansion | 6 | 1.000000 | 2 | momentum_ret_3 | 757 | 0.086120 | -0.000041 | 0.804265 | -3.284554 | 3 | 9 | negative_economic |
| trend_structure | 6 | 1.000000 | 3 | momentum_ret_3 | 758 | 0.084132 | -0.000058 | 1.038099 | 0.140687 | 6 | 6 | negative_economic |
| minimal_vwap_location | 6 | 1.000000 | 3 | long | 3882 | 0.076100 | 0.000082 | 0.994465 | -0.293430 | 6 | 6 | weak_profit_factor |
| minimal_trend_efficiency | 6 | 1.000000 | 2 | momentum_ret_3 | 784 | 0.073751 | 0.000138 | 1.123745 | 0.681139 | 7 | 5 | weak_sharpe |
| trend_structure | 6 | 1.000000 | 1 | long | 3408 | 0.064696 | -0.000005 | 1.000288 | 0.004266 | 6 | 6 | negative_economic |
| minimal_vwap_location | 6 | 1.000000 | 0 | momentum_ret_3 | 777 | 0.060074 | 0.000012 | 1.053058 | 0.390181 | 6 | 6 | weak_profit_factor |
| time_volatility | 6 | 1.000000 | 1 | momentum_ret_3 | 1160 | 0.059200 | -0.000030 | 0.727672 | -3.587087 | 4 | 6 | negative_economic |
| rich_extreme_reversion | 6 | 1.000000 | 3 | momentum_ret_3 | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| minimal_trend_efficiency | 6 | 1.000000 | 0 | momentum_ret_3 | 711 | 0.057107 | 0.000071 | 1.103850 | 0.898711 | 7 | 5 | weak_sharpe |
| ablation_price_structure_only | 6 | 1.000000 | 2 | random_symmetric | 3104 | 0.051589 | 0.000019 | 1.010469 | 0.468313 | 6 | 6 | random_benchmark |
| vwap_reversion | 6 | 2.000000 | 0 | momentum_ret_3 | 851 | 0.044205 | -0.000064 | 0.886171 | -0.892994 | 6 | 6 | negative_economic |
| rich_extreme_reversion | 6 | 1.000000 | 1 | momentum_ret_3 | 634 | 0.042639 | 0.000034 | 1.007870 | -0.039820 | 6 | 6 | weak_profit_factor |
| rich_trend_vwap_structure | 6 | 1.000000 | 2 | momentum_ret_3 | 709 | 0.041416 | 0.000232 | 1.053095 | 0.651164 | 6 | 6 | weak_profit_factor |
| rich_trend_vwap_structure | 6 | 1.000000 | 0 | long | 2755 | 0.031611 | -0.000014 | 1.111315 | 1.140806 | 6 | 6 | negative_economic |
| current_default | 6 | 1.000000 | 2 | momentum_ret_3 | 779 | 0.027186 | 0.000049 | 0.887978 | -1.227935 | 5 | 7 | weak_profit_factor |
| minimal_interpretable | 6 | 1.000000 | 3 | momentum_ret_3 | 701 | 0.017005 | -0.000031 | 0.896240 | -1.143650 | 5 | 7 | negative_economic |
| time_volatility | 2 | 1.000000 | 3 | momentum_ret_3 | 1183 | 0.016105 | 0.000048 | 0.972996 | -0.076530 | 6 | 6 | weak_profit_factor |
| time_volatility | 6 | 1.000000 | 0 | momentum_ret_3 | 1003 | 0.015736 | 0.000019 | 0.928175 | -0.769953 | 4 | 7 | weak_profit_factor |
| volatility_liquidity | 6 | 1.000000 | 0 | reversion_ret_3 | 720 | 0.015222 | 0.000272 | 1.157701 | 2.342871 | 6 | 5 | candidate |

## Validation Candidate Holdout Sanity

| feature_set | horizon_bars | cost_bps | hmm_state | action | split | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_default | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 774 | 0.165299 | 0.000159 | 1.280337 | 2.523524 | 9 | 3 | candidate |
| current_default | 6 | 1.000000 | 3 | momentum_ret_3 | test | 1025 | 0.064625 | -0.000017 | 0.925087 | -1.109374 | 5 | 7 | negative_economic |
| current_default | 6 | 2.000000 | 3 | momentum_ret_3 | validation | 774 | 0.087899 | 0.000059 | 1.163650 | 1.491915 | 7 | 5 | candidate |
| current_default | 6 | 2.000000 | 3 | momentum_ret_3 | test | 1025 | -0.037875 | -0.000117 | 0.859127 | -1.962852 | 4 | 8 | negative_economic |
| minimal_vwap_location | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 737 | 0.180227 | 0.000209 | 1.315821 | 2.187013 | 7 | 5 | candidate |
| minimal_vwap_location | 6 | 1.000000 | 3 | momentum_ret_3 | test | 827 | 0.130668 | 0.000092 | 1.030768 | 0.323806 | 7 | 5 | weak_profit_factor |
| minimal_vwap_location | 6 | 2.000000 | 3 | momentum_ret_3 | validation | 737 | 0.106527 | 0.000109 | 1.196194 | 1.510023 | 7 | 5 | candidate |
| minimal_vwap_location | 6 | 2.000000 | 3 | momentum_ret_3 | test | 827 | 0.047968 | -0.000008 | 0.934505 | -0.706388 | 5 | 7 | negative_economic |
| rich_extreme_reversion | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| rich_extreme_reversion | 6 | 1.000000 | 3 | momentum_ret_3 | test | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | candidate |
| volatility_liquidity | 6 | 1.000000 | 0 | reversion_ret_3 | validation | 720 | 0.015222 | 0.000272 | 1.157701 | 2.342871 | 6 | 5 | candidate |
| volatility_liquidity | 6 | 1.000000 | 0 | reversion_ret_3 | test | 518 | -0.102201 | 0.000020 | 1.016438 | 0.337023 | 5 | 5 | negative_economic |
| volatility_liquidity | 6 | 1.000000 | 2 | momentum_ret_3 | validation | 576 | 0.164995 | 0.000277 | 1.310077 | 2.645613 | 8 | 4 | candidate |
| volatility_liquidity | 6 | 1.000000 | 2 | momentum_ret_3 | test | 586 | -0.012188 | -0.000172 | 0.861896 | -1.859082 | 5 | 7 | negative_economic |
| volatility_liquidity | 6 | 2.000000 | 2 | momentum_ret_3 | validation | 576 | 0.107395 | 0.000177 | 1.170670 | 1.598395 | 7 | 5 | candidate |
| volatility_liquidity | 6 | 2.000000 | 2 | momentum_ret_3 | test | 586 | -0.070788 | -0.000272 | 0.732625 | -2.573404 | 4 | 8 | negative_economic |

## Feature Set Validation

| feature_set | n_features | columns | missing_columns | status |
| --- | --- | --- | --- | --- |
| current_default | 8 | ret_1,ret_3,rv_6,rv_12,range,rel_volume,trend_12,intraday_drawdown |  | ready |
| volatility_liquidity | 8 | rv_3,rv_6,rv_12,rv_24,range,atr_6,atr_12,rel_volume |  | ready |
| trend_structure | 7 | ret_1,ret_3,ret_6,trend_6,trend_12,trend_24,intraday_drawdown |  | ready |
| vwap_reversion | 6 | ret_1,ret_3,dist_vwap,intraday_drawdown,range,rel_volume |  | ready |
| time_volatility | 8 | sin_time,cos_time,minutes_to_close,open_window,close_window,rv_6,rv_12,rel_volume |  | ready |
| minimal_interpretable | 5 | rv_12,range,rel_volume,trend_12,dist_vwap |  | ready |
| minimal_trend_efficiency | 5 | ret_3,trend_12,signed_efficiency_12,dir_persistence_12,rv_12 |  | ready |
| minimal_vwap_location | 6 | dist_vwap_atr,dist_open,pos_session_range,intraday_drawdown,rv_12,rel_volume |  | ready |
| minimal_compression_expansion | 6 | vol_ratio_6_24,range_ratio_6_24,signed_efficiency_12,pos_session_range,rv_12,range |  | ready |
| ablation_price_structure_only | 7 | ret_3,rv_12,vol_ratio_6_24,signed_efficiency_12,dist_vwap_atr,pos_session_range,intraday_drawdown |  | ready |
| rich_trend_vwap_structure | 9 | ret_3,ret_6,trend_12,trend_24,signed_efficiency_12,dir_persistence_12,dist_vwap_atr,vwap_slope_12,pos_session_range |  | ready |
| rich_extreme_reversion | 9 | dist_session_high_atr,dist_session_low_atr,pos_session_range,intraday_drawdown,intraday_runup,dist_vwap_atr,rv_12,ret_3,rel_volume |  | ready |

## Outputs

- `reports/hmm_feature_lab/feature_set_metrics.parquet`
- `reports/hmm_feature_lab/feature_set_ranking.parquet`
- `reports/hmm_feature_lab/feature_set_summary.parquet`
- `reports/hmm_feature_lab/feature_set_holdout.parquet`
- `reports/hmm_feature_lab/feature_set_hour_distribution.parquet`

## Conclusion

At least one validation candidate also passed the same filters on test. It still needs full seed/K stability and a final frozen OOS evaluation.

This lab is for iterative feature-set screening. Do not accept a feature set from this report alone; promote only interpretable sets to full seed/K/fold stability tests and frozen OOS evaluation.

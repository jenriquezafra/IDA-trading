# HMM Candidate Diagnostics

## Scope

- Candidate source: `reports/hmm_stability/stability_holdout.parquet`
- Candidates inspected: 2
- Feature sets: `['rich_extreme_reversion']`
- Splits reconstructed: validation and test

## Candidate Rows

| candidate_id | feature_set | n_states | seed | horizon_bars | cost_bps | hmm_state | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 3 | momentum_ret_3 | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | candidate |

## Feature Profile Top Z-Scores

| candidate_id | split | feature | avg_state_z | median_state_z | std_state_z | positive_z_folds | negative_z_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | dist_vwap_atr | 0.335683 | 0.562929 | 0.910603 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | dist_session_low_atr | 0.325739 | 0.337287 | 0.952362 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | intraday_runup | 0.325739 | 0.337287 | 0.952362 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | pos_session_range | 0.304991 | 0.859223 | 0.933092 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | dist_session_high_atr | 0.244573 | 0.710571 | 0.693492 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | intraday_drawdown | 0.222151 | 0.653688 | 0.697549 | 8 | 4 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | rv_12 | -0.145921 | -0.084015 | 0.378393 | 6 | 6 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | ret_3 | 0.089373 | 0.150102 | 0.290567 | 9 | 3 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | rel_volume | -0.029790 | -0.031027 | 0.248959 | 4 | 8 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | dist_session_low_atr | 0.362014 | 0.040366 | 1.143118 | 6 | 6 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | intraday_runup | 0.362014 | 0.040366 | 1.143118 | 6 | 6 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | dist_vwap_atr | 0.306436 | 0.186266 | 1.062048 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | pos_session_range | 0.266858 | 0.548940 | 1.044384 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | dist_session_high_atr | 0.193991 | 0.425630 | 0.805015 | 7 | 5 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | intraday_drawdown | 0.167898 | 0.470559 | 0.795157 | 8 | 4 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | rv_12 | -0.080131 | -0.016764 | 0.386158 | 5 | 7 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | ret_3 | 0.052332 | 0.152024 | 0.305608 | 8 | 4 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | rel_volume | -0.026875 | -0.038341 | 0.271831 | 4 | 8 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | dist_session_low_atr | 0.311285 | 0.476415 | 0.942522 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | intraday_runup | 0.311285 | 0.476415 | 0.942522 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | pos_session_range | 0.301042 | 0.794038 | 1.105499 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | dist_vwap_atr | 0.273468 | 0.541118 | 1.089445 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | dist_session_high_atr | 0.151766 | 0.581511 | 0.977256 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | intraday_drawdown | 0.123348 | 0.654461 | 0.962186 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | ret_3 | 0.076735 | 0.193409 | 0.423070 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | rel_volume | 0.049070 | 0.058818 | 0.350902 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | rv_12 | 0.010934 | 0.114298 | 0.498065 | 7 | 5 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | pos_session_range | 0.315890 | 0.717994 | 0.958182 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | dist_session_low_atr | 0.277188 | 0.211294 | 0.990639 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | intraday_runup | 0.277188 | 0.211294 | 0.990639 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | dist_vwap_atr | 0.272850 | 0.427242 | 0.967806 | 8 | 4 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | dist_session_high_atr | 0.244574 | 0.601125 | 0.806387 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | intraday_drawdown | 0.174352 | 0.580053 | 0.804733 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | ret_3 | 0.051905 | 0.151214 | 0.395654 | 9 | 3 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | rel_volume | 0.036822 | 0.041581 | 0.413181 | 6 | 6 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | rv_12 | 0.009374 | 0.069705 | 0.584936 | 8 | 4 |

## Time Concentration

| candidate_id | split | top_hour | top_hour_state_pct | top_hour_lift | normalized_hour_entropy |
| --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 14 | 0.208455 | 1.024904 | 0.967966 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 14 | 0.215119 | 1.057670 | 0.971381 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 12 | 0.209948 | 1.032245 | 0.967293 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 12 | 0.219111 | 1.077298 | 0.972534 |

## Clock Control

| candidate_id | split | bucket | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | candidate_state | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | full_split_all_states | 3034 | 0.120767 | 0.000043 | 1.057709 | 0.920650 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | same_hours_all_states | 2999 | 0.108732 | 0.000036 | 1.057709 | 0.920650 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | same_hours_ex_state | 2030 | -0.094150 | -0.000014 | 1.013793 | 0.154312 | 6 | 6 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | candidate_state | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | full_split_all_states | 3031 | -0.011784 | -0.000002 | 0.914760 | -0.955682 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | same_hours_all_states | 3019 | -0.020621 | -0.000005 | 0.873017 | -1.573788 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | same_hours_ex_state | 2228 | -0.078045 | -0.000006 | 0.925652 | -1.218387 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | candidate_state | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | full_split_all_states | 3034 | 0.120767 | 0.000043 | 1.057709 | 0.920650 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | same_hours_all_states | 3003 | 0.082746 | 0.000031 | 1.018315 | 0.487398 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | same_hours_ex_state | 2347 | -0.036196 | -0.000006 | 1.007955 | 0.142306 | 6 | 6 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | candidate_state | 704 | 0.104032 | 0.000187 | 1.318801 | 2.205832 | 7 | 5 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | full_split_all_states | 3031 | -0.011784 | -0.000002 | 0.914760 | -0.955682 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | same_hours_all_states | 3000 | -0.020423 | -0.000007 | 0.908308 | -0.984668 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | same_hours_ex_state | 2296 | -0.124455 | -0.000053 | 0.925199 | -0.696417 | 6 | 6 | negative_economic |

## Target Fold Performance

| candidate_id | split | fold | trades | net_return | avg_trade_net | profit_factor | daily_sharpe | hit_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 0 | 86 | -0.015840 | -0.000184 | 0.723557 | -3.504976 | 0.430233 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0 | 82 | 0.025419 | 0.000310 | 1.771810 | 4.759722 | 0.475610 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 1 | 18 | 0.000282 | 0.000016 | 1.065849 | 0.696543 | 0.555556 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1 | 33 | -0.007442 | -0.000226 | 0.495095 | -6.587503 | 0.454545 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 2 | 90 | -0.019270 | -0.000214 | 0.777651 | -2.883949 | 0.466667 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 2 | 135 | 0.069620 | 0.000516 | 1.493105 | 5.669465 | 0.614815 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 3 | 72 | 0.055577 | 0.000772 | 1.741676 | 4.713990 | 0.583333 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 3 | 101 | 0.039151 | 0.000388 | 1.501945 | 3.870404 | 0.574257 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 4 | 90 | 0.013342 | 0.000148 | 1.138910 | 1.866894 | 0.500000 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 4 | 109 | 0.020626 | 0.000189 | 1.161182 | 3.511047 | 0.522936 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 5 | 109 | 0.020626 | 0.000189 | 1.161182 | 3.612088 | 0.522936 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 5 | 126 | 0.021484 | 0.000171 | 1.214137 | 2.632938 | 0.619048 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 6 | 59 | 0.003803 | 0.000064 | 1.064928 | 0.654045 | 0.542373 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 6 | 15 | -0.009355 | -0.000624 | 0.527753 | -2.847372 | 0.466667 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 7 | 45 | -0.046724 | -0.001038 | 0.528499 | -5.124103 | 0.400000 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 7 | 66 | 0.033899 | 0.000514 | 1.699364 | 4.569638 | 0.575758 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 8 | 38 | -0.008556 | -0.000225 | 0.726967 | -4.518887 | 0.526316 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 8 | 90 | 0.020086 | 0.000223 | 1.365195 | 3.651205 | 0.533333 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 9 | 89 | 0.023138 | 0.000260 | 1.445402 | 4.216053 | 0.539326 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 9 | 66 | 0.009091 | 0.000138 | 1.185781 | 1.707060 | 0.621212 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 10 | 72 | 0.023007 | 0.000320 | 1.580934 | 4.306902 | 0.611111 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 10 | 81 | -0.019264 | -0.000238 | 0.837458 | -1.256231 | 0.530864 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | 11 | 23 | 0.008038 | 0.000349 | 1.529142 | 3.268940 | 0.608696 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 11 | 65 | -0.000434 | -0.000007 | 0.992694 | -0.104945 | 0.615385 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 0 | 63 | -0.014941 | -0.000237 | 0.483744 | -9.222415 | 0.460317 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 0 | 22 | 0.001115 | 0.000051 | 1.220893 | 4.747685 | 0.454545 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 1 | 65 | -0.007038 | -0.000108 | 0.833234 | -1.887272 | 0.400000 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1 | 78 | -0.016373 | -0.000210 | 0.769256 | -2.885820 | 0.461538 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 2 | 74 | 0.022569 | 0.000305 | 1.492254 | 3.966441 | 0.581081 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 2 | 96 | 0.032430 | 0.000338 | 1.294185 | 2.676863 | 0.531250 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 3 | 82 | 0.057180 | 0.000697 | 1.690022 | 4.586560 | 0.585366 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 3 | 106 | 0.054823 | 0.000517 | 1.666300 | 4.760811 | 0.584906 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 4 | 13 | -0.000634 | -0.000049 | 0.943274 | -0.275761 | 0.307692 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 4 | 22 | -0.000975 | -0.000044 | 0.914590 | -0.605996 | 0.409091 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 5 | 58 | 0.030765 | 0.000530 | 1.497404 | 2.615689 | 0.448276 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 5 | 47 | 0.028717 | 0.000611 | 1.630980 | 3.774732 | 0.531915 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 6 | 102 | 0.011917 | 0.000117 | 1.145348 | 1.795976 | 0.598039 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 6 | 63 | 0.031732 | 0.000504 | 1.397091 | 3.511197 | 0.619048 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 7 | 45 | 0.023080 | 0.000513 | 1.593016 | 5.964525 | 0.622222 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 7 | 26 | -0.019303 | -0.000742 | 0.474946 | -3.976279 | 0.461538 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 8 | 71 | -0.054472 | -0.000767 | 0.530006 | -5.277862 | 0.323944 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 8 | 25 | 0.036174 | 0.001447 | 9.922946 | 9.067460 | 0.760000 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 9 | 47 | 0.035317 | 0.000751 | 3.119718 | 11.643897 | 0.659574 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 9 | 48 | 0.005470 | 0.000114 | 1.164720 | 1.552867 | 0.645833 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 10 | 69 | -0.009099 | -0.000132 | 0.822338 | -1.829247 | 0.434783 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 10 | 69 | -0.042454 | -0.000615 | 0.615102 | -2.450334 | 0.492754 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | 11 | 15 | 0.009389 | 0.000626 | 2.547236 | 7.745268 | 0.666667 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 11 | 54 | 0.007588 | 0.000141 | 1.197369 | 2.725371 | 0.629630 |

## Action Comparison Inside Candidate State

| candidate_id | split | target_action | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | flat | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | long | 4447 | -0.472863 | 0.000026 | 0.913198 | -1.227029 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | momentum_ret_3 | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | random_symmetric | 4447 | -0.638880 | -0.000152 | 0.824986 | -4.496106 | 1 | 11 | random_benchmark |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | reversion_ret_3 | 969 | -0.396682 | -0.000313 | 0.662542 | -5.537313 | 3 | 9 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | short | 4447 | -0.416537 | -0.000226 | 0.792480 | -2.416909 | 4 | 8 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | flat | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | long | 3770 | -0.183592 | -0.000033 | 0.885026 | -1.483909 | 4 | 8 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | momentum_ret_3 | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | random_symmetric | 3770 | -0.040225 | -0.000008 | 0.956654 | -1.651893 | 4 | 8 | random_benchmark |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | reversion_ret_3 | 791 | -0.215624 | -0.000238 | 0.735638 | -4.507058 | 3 | 9 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | short | 3770 | -0.570408 | -0.000167 | 0.873173 | -1.838943 | 3 | 9 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | flat | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | long | 3277 | -0.510996 | -0.000185 | 0.797484 | -3.021834 | 4 | 8 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | momentum_ret_3 | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | random_symmetric | 3277 | -0.532443 | -0.000155 | 0.808809 | -5.673248 | 3 | 9 | random_benchmark |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | reversion_ret_3 | 656 | -0.250142 | -0.000376 | 0.657768 | -4.475513 | 3 | 9 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | momentum_ret_3 | short | 3277 | -0.144404 | -0.000015 | 1.028261 | 0.452498 | 6 | 6 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | flat | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | long | 3286 | -0.122243 | 0.000026 | 0.953919 | -0.595814 | 5 | 7 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | momentum_ret_3 | 704 | 0.104032 | 0.000187 | 1.318801 | 2.205832 | 7 | 5 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | random_symmetric | 3286 | -0.102672 | -0.000035 | 0.947429 | -1.462143 | 5 | 7 | random_benchmark |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | reversion_ret_3 | 704 | -0.244832 | -0.000387 | 0.632254 | -3.970042 | 2 | 10 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | validation | momentum_ret_3 | short | 3286 | -0.534957 | -0.000226 | 0.794612 | -2.448740 | 2 | 10 | negative_economic |

## Hour Distribution

| candidate_id | feature_set | n_states | seed | hmm_state | target_action | split | hour | state_rows | split_rows | state_pct | split_pct | hour_lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 14 | 927 | 3012 | 0.208455 | 0.203390 | 1.024904 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 13 | 918 | 3012 | 0.206431 | 0.203390 | 1.014954 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 11 | 907 | 3012 | 0.203958 | 0.203390 | 1.002792 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 12 | 886 | 3012 | 0.199235 | 0.203390 | 0.979574 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 10 | 409 | 1506 | 0.091972 | 0.101695 | 0.904392 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | test | 15 | 400 | 1255 | 0.089948 | 0.084746 | 1.061390 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 14 | 811 | 3012 | 0.215119 | 0.203390 | 1.057670 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 13 | 784 | 3012 | 0.207958 | 0.203390 | 1.022458 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 11 | 747 | 3012 | 0.198143 | 0.203390 | 0.974204 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 12 | 708 | 3012 | 0.187798 | 0.203390 | 0.923342 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 10 | 366 | 1506 | 0.097082 | 0.101695 | 0.954642 |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | validation | 15 | 354 | 1255 | 0.093899 | 0.084746 | 1.108011 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 12 | 688 | 3012 | 0.209948 | 0.203390 | 1.032245 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 14 | 679 | 3012 | 0.207202 | 0.203390 | 1.018742 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 11 | 675 | 3012 | 0.205981 | 0.203390 | 1.012740 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 13 | 643 | 3012 | 0.196216 | 0.203390 | 0.964729 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 15 | 302 | 1255 | 0.092157 | 0.084746 | 1.087458 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | test | 10 | 290 | 1506 | 0.088496 | 0.101695 | 0.870206 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 12 | 720 | 3012 | 0.219111 | 0.203390 | 1.077298 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 13 | 686 | 3012 | 0.208764 | 0.203390 | 1.026425 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 14 | 632 | 3012 | 0.192331 | 0.203390 | 0.945628 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 11 | 608 | 3012 | 0.185027 | 0.203390 | 0.909718 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 10 | 325 | 1506 | 0.098904 | 0.101695 | 0.972560 |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | validation | 15 | 315 | 1255 | 0.095861 | 0.084746 | 1.131163 |

## Outputs

- `reports/hmm_candidate_diagnostics/candidates.parquet`
- `reports/hmm_candidate_diagnostics/feature_profile_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/feature_profile_summary.parquet`
- `reports/hmm_candidate_diagnostics/hour_distribution_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/hour_summary.parquet`
- `reports/hmm_candidate_diagnostics/time_concentration.parquet`
- `reports/hmm_candidate_diagnostics/state_action_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/state_action_summary.parquet`
- `reports/hmm_candidate_diagnostics/clock_control_by_fold.parquet`
- `reports/hmm_candidate_diagnostics/clock_control_summary.parquet`
- `reports/hmm_candidate_diagnostics/target_fold_metrics.parquet`

## Conclusion

The surviving rows remain candidates, and the same-hour ex-state control does not. This supports a regime-conditioned effect, but it is still cost-fragile.

State ids are inspected within each fold/model fit. The report does not assume that nominal state ids are comparable across different K/seed combinations.

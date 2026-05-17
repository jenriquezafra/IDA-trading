# HMM Candidate Thresholds

## Scope

- Candidate source: `reports/hmm_stability/stability_holdout.parquet`
- Feature sets: `['rich_extreme_reversion', 'minimal_vwap_location']`
- Threshold multipliers: `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]`
- Cost grid bps: `[0.5, 1.0, 1.5, 2.0]`
- Threshold selection: validation only, then reported on test.

## Candidate Source Rows

| source_rank | candidate_id | feature_set | n_states | seed | hmm_state | action | total_net_return | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | minimal_vwap_location | 5 | 42 | 1 | momentum_ret_3 | 0.204444 | 1.258477 | 2.593584 | 7 | 5 |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 4 | 42 | 3 | momentum_ret_3 | 0.202882 | 1.199959 | 3.071992 | 8 | 4 |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | rich_extreme_reversion | 5 | 7 | 3 | momentum_ret_3 | 0.118942 | 1.209131 | 2.701117 | 8 | 4 |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | minimal_vwap_location | 3 | 42 | 0 | momentum_ret_3 | 0.048779 | 1.161592 | 1.972969 | 7 | 5 |

## Candidate Decisions

| source_rank | candidate_id | status_1bps | status_2bps | accepted | cost_fragile | test_net_1bps | test_net_2bps | test_drawdown_1bps | test_drawdown_2bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | candidate | weak_sharpe | False | True | 0.311068 | 0.074447 | 0.063029 | 0.016915 |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | candidate | weak_profit_factor | False | True | 0.118942 | 0.053342 | 0.073622 | 0.075322 |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | candidate | insufficient_trades | False | True | 0.048779 | 0.014290 | 0.089986 | 0.006476 |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | candidate | insufficient_trades | False | True | 0.202882 | 0.000000 | 0.063418 | 0.000000 |

## Selected Threshold Test Results

| source_rank | candidate_id | cost_bps | validation_threshold_multiplier | validation_candidate_status | validation_total_net_return | validation_max_drawdown_abs | split | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | max_drawdown_abs | return_to_drawdown | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | 0.500000 | 0.750000 | candidate | 0.137627 | 0.090378 | test | 1771 | 0.133497 | 0.000064 | 1.089087 | 1.094771 | 8 | 4 | 0.125847 | 1.060790 | weak_profit_factor |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | 1.000000 | 1.000000 | candidate | 0.011583 | 0.063425 | test | 1026 | 0.048779 | 0.000041 | 1.161592 | 1.972969 | 7 | 5 | 0.089986 | 0.542081 | candidate |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | 1.500000 | 2.000000 | insufficient_trades | 0.018483 | 0.002943 | test | 25 | 0.015540 | 0.000415 | 1.425830 | 0.177545 | 5 | 5 | 0.006376 | 2.437322 | insufficient_trades |
| 4 | minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | 2.000000 | 2.000000 | insufficient_trades | 0.017233 | 0.002993 | test | 25 | 0.014290 | 0.000373 | 1.386410 | 0.053365 | 5 | 5 | 0.006476 | 2.206657 | insufficient_trades |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | 0.500000 | 0.750000 | candidate | 0.155333 | 0.055004 | test | 1240 | 0.373068 | 0.000267 | 1.336002 | 4.246557 | 9 | 3 | 0.062279 | 5.990294 | candidate |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | 1.000000 | 0.750000 | candidate | 0.110583 | 0.059404 | test | 1240 | 0.311068 | 0.000217 | 1.288041 | 3.757220 | 7 | 5 | 0.063029 | 4.935335 | candidate |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | 1.500000 | 1.500000 | candidate | 0.029270 | 0.021179 | test | 170 | 0.082947 | 0.000286 | 1.472036 | 1.272121 | 6 | 6 | 0.016065 | 5.163241 | not_stable_across_folds |
| 1 | minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | 2.000000 | 1.500000 | candidate | 0.022520 | 0.021929 | test | 170 | 0.074447 | 0.000236 | 1.361897 | 0.864828 | 6 | 6 | 0.016915 | 4.401263 | weak_sharpe |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | 0.500000 | 1.000000 | candidate | 0.096974 | 0.044474 | test | 969 | 0.251332 | 0.000163 | 1.271729 | 3.847489 | 9 | 3 | 0.062468 | 4.023378 | candidate |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | 1.000000 | 1.000000 | candidate | 0.057424 | 0.046724 | test | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | 0.063418 | 3.199128 | candidate |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | 1.500000 | 0.750000 | negative_economic | 0.020292 | 0.063047 | test | 1642 | 0.160952 | -0.000003 | 1.014654 | 0.220776 | 6 | 6 | 0.062225 | 2.586613 | negative_economic |
| 2 | rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | 2.000000 | 3.000000 | insufficient_trades | 0.000000 | 0.000000 | test | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | 0.500000 | 0.750000 | candidate | 0.144989 | 0.091712 | test | 1160 | 0.093121 | 0.000108 | 1.165000 | 1.627866 | 8 | 4 | 0.063322 | 1.470598 | candidate |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | 1.000000 | 1.000000 | candidate | 0.104032 | 0.067130 | test | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | 0.073622 | 1.615569 | candidate |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | 1.500000 | 1.000000 | candidate | 0.068832 | 0.069880 | test | 656 | 0.086142 | 0.000126 | 1.106870 | 1.323667 | 8 | 4 | 0.074472 | 1.156697 | candidate |
| 3 | rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | 2.000000 | 1.000000 | candidate | 0.033632 | 0.072630 | test | 656 | 0.053342 | 0.000076 | 1.036488 | 0.495772 | 7 | 5 | 0.075322 | 0.708182 | weak_profit_factor |

## Top Threshold Grid Rows

| candidate_id | split | threshold_multiplier | cost_bps | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | max_drawdown_abs | return_to_drawdown | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.750000 | 0.500000 | 1240 | 0.373068 | 0.000267 | 1.336002 | 4.246557 | 9 | 3 | 0.062279 | 5.990294 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.500000 | 0.500000 | 1829 | 0.347507 | 0.000190 | 1.116774 | 1.820892 | 10 | 2 | 0.081757 | 4.250470 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.500000 | 0.500000 | 2498 | 0.337689 | 0.000105 | 1.122036 | 2.329579 | 11 | 1 | 0.084304 | 4.005586 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.750000 | 0.500000 | 1642 | 0.325152 | 0.000097 | 1.151999 | 2.179966 | 8 | 4 | 0.058477 | 5.560383 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 0.500000 | 969 | 0.251332 | 0.000163 | 1.271729 | 3.847489 | 9 | 3 | 0.062468 | 4.023378 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.000000 | 0.500000 | 756 | 0.242244 | 0.000267 | 1.308624 | 3.105267 | 8 | 4 | 0.048651 | 4.979178 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.250000 | 0.500000 | 402 | 0.228405 | 0.000486 | 1.656477 | 3.078016 | 10 | 2 | 0.022099 | 10.335481 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 0.500000 | 0.500000 | 2645 | 0.199275 | 0.000068 | 1.049176 | 1.010737 | 9 | 3 | 0.169311 | 1.176978 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 0.500000 | 656 | 0.151742 | 0.000226 | 1.309745 | 3.360987 | 9 | 3 | 0.072772 | 2.085160 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 0.500000 | 504 | 0.134455 | 0.000121 | 1.170809 | 1.268672 | 7 | 5 | 0.035256 | 3.813733 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 0.750000 | 0.500000 | 1771 | 0.133497 | 0.000064 | 1.089087 | 1.094771 | 8 | 4 | 0.125847 | 1.060790 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 0.500000 | 352 | 0.114836 | 0.000334 | 1.326431 | 2.176152 | 10 | 2 | 0.038773 | 2.961754 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.000000 | 0.500000 | 1026 | 0.100079 | 0.000091 | 1.241466 | 2.984312 | 7 | 5 | 0.088507 | 1.130757 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.500000 | 0.500000 | 170 | 0.099947 | 0.000386 | 1.720471 | 1.917328 | 7 | 5 | 0.014653 | 6.820739 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.250000 | 0.500000 | 536 | 0.095829 | 0.000212 | 1.538863 | 3.242201 | 9 | 3 | 0.103109 | 0.929395 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 0.750000 | 0.500000 | 1160 | 0.093121 | 0.000108 | 1.165000 | 1.627866 | 8 | 4 | 0.063322 | 1.470598 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 0.500000 | 0.500000 | 1779 | 0.090255 | 0.000084 | 1.075886 | 1.418571 | 9 | 3 | 0.078250 | 1.153409 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 0.500000 | 224 | 0.085925 | 0.000097 | 1.334325 | 1.530777 | 6 | 6 | 0.015339 | 5.601611 | not_stable_across_folds |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 0.500000 | 171 | 0.076531 | 0.000402 | 1.907282 | 3.117207 | 10 | 2 | 0.020784 | 3.682275 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.500000 | 0.500000 | 247 | 0.046448 | 0.000244 | 1.274807 | 1.763695 | 9 | 3 | 0.041289 | 1.124947 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 2.000000 | 0.500000 | 25 | 0.018040 | 0.000498 | 1.516062 | 0.635290 | 5 | 5 | 0.006176 | 2.921062 | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 0.500000 | 18 | 0.013350 | 0.000420 | inf | 3.338063 | 5 | 3 | 0.006176 | 2.161597 | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2.000000 | 0.500000 | 15 | 0.003183 | 0.000079 | 0.946325 | -0.159393 | 3 | 4 | 0.009938 | 0.320242 | insufficient_trades |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 3.000000 | 0.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3.000000 | 0.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 0.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 0.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 0.500000 | 24 | -0.002677 | 0.000319 | 1.677419 | 1.233516 | 5 | 4 | 0.022188 | -0.120654 | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.750000 | 1.000000 | 1240 | 0.311068 | 0.000217 | 1.288041 | 3.757220 | 7 | 5 | 0.063029 | 4.935335 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.500000 | 1.000000 | 1829 | 0.256057 | 0.000140 | 1.071324 | 1.162520 | 8 | 4 | 0.082807 | 3.092202 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.750000 | 1.000000 | 1642 | 0.243052 | 0.000047 | 1.081178 | 1.216903 | 7 | 5 | 0.059199 | 4.105653 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.500000 | 1.000000 | 2498 | 0.212789 | 0.000055 | 1.069199 | 1.365657 | 8 | 4 | 0.090554 | 2.349844 | weak_profit_factor |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.250000 | 1.000000 | 402 | 0.208305 | 0.000436 | 1.539589 | 2.619986 | 10 | 2 | 0.022399 | 9.299699 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.000000 | 1.000000 | 756 | 0.204444 | 0.000217 | 1.258477 | 2.593584 | 7 | 5 | 0.049251 | 4.151028 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 1.000000 | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | 0.063418 | 3.199128 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 1.000000 | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | 0.073622 | 1.615569 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 1.000000 | 504 | 0.109255 | 0.000071 | 1.109326 | 0.835407 | 7 | 5 | 0.035856 | 3.047094 | weak_sharpe |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 1.000000 | 352 | 0.097236 | 0.000284 | 1.200057 | 1.517544 | 9 | 3 | 0.039580 | 2.456710 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.500000 | 1.000000 | 170 | 0.091447 | 0.000336 | 1.591235 | 1.628720 | 6 | 6 | 0.015353 | 5.956139 | not_stable_across_folds |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 1.000000 | 224 | 0.074725 | 0.000047 | 1.266400 | 1.116090 | 6 | 6 | 0.015439 | 4.839906 | not_stable_across_folds |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.250000 | 1.000000 | 536 | 0.069029 | 0.000162 | 1.472245 | 2.570407 | 8 | 4 | 0.105559 | 0.653938 | candidate |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 1.000000 | 171 | 0.067981 | 0.000352 | 1.447269 | 2.457155 | 10 | 2 | 0.021134 | 3.216725 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 0.500000 | 1.000000 | 2645 | 0.067025 | 0.000018 | 0.999663 | 0.003888 | 6 | 6 | 0.178861 | 0.374734 | weak_profit_factor |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.000000 | 1.000000 | 1026 | 0.048779 | 0.000041 | 1.161592 | 1.972969 | 7 | 5 | 0.089986 | 0.542081 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 0.750000 | 1.000000 | 1771 | 0.044947 | 0.000014 | 1.039418 | 0.435373 | 7 | 5 | 0.132897 | 0.338211 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 0.750000 | 1.000000 | 1160 | 0.035121 | 0.000058 | 1.105035 | 1.030883 | 7 | 5 | 0.063922 | 0.549439 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.500000 | 1.000000 | 247 | 0.034098 | 0.000194 | 1.212452 | 1.333796 | 9 | 3 | 0.041939 | 0.813036 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 2.000000 | 1.000000 | 25 | 0.016790 | 0.000456 | 1.466999 | 0.303282 | 5 | 5 | 0.006276 | 2.675338 | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 1.000000 | 18 | 0.012450 | 0.000387 | inf | 3.313446 | 5 | 3 | 0.006276 | 1.983744 | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2.000000 | 1.000000 | 15 | 0.002433 | 0.000050 | 0.768758 | -0.750411 | 3 | 4 | 0.010038 | 0.242338 | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 0.500000 | 1.000000 | 1779 | 0.001305 | 0.000034 | 1.027456 | 0.508035 | 7 | 5 | 0.083351 | 0.015653 | weak_profit_factor |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 3.000000 | 1.000000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3.000000 | 1.000000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 1.000000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 1.000000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 1.000000 | 24 | -0.003877 | 0.000282 | 1.636615 | 1.180012 | 5 | 4 | 0.022438 | -0.172791 | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.750000 | 1.500000 | 1240 | 0.249068 | 0.000167 | 1.241965 | 3.257076 | 7 | 5 | 0.063779 | 3.905186 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.250000 | 1.500000 | 402 | 0.188205 | 0.000386 | 1.416324 | 2.163837 | 9 | 3 | 0.022699 | 8.291296 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.000000 | 1.500000 | 756 | 0.166644 | 0.000167 | 1.140562 | 1.552809 | 7 | 5 | 0.049851 | 3.342812 | candidate |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0.500000 | 1.500000 | 1829 | 0.164607 | 0.000090 | 1.031826 | 0.500115 | 7 | 5 | 0.083857 | 1.962940 | weak_profit_factor |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.750000 | 1.500000 | 1642 | 0.160952 | -0.000003 | 1.014654 | 0.220776 | 6 | 6 | 0.062225 | 2.586613 | negative_economic |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 1.500000 | 969 | 0.154432 | 0.000063 | 1.131977 | 2.232410 | 8 | 4 | 0.064368 | 2.399208 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 0.500000 | 1.500000 | 2498 | 0.087889 | 0.000005 | 1.021373 | 0.395771 | 6 | 6 | 0.096819 | 0.907765 | weak_profit_factor |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.000000 | 1.500000 | 656 | 0.086142 | 0.000126 | 1.106870 | 1.323667 | 8 | 4 | 0.074472 | 1.156697 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 1.500000 | 504 | 0.084055 | 0.000021 | 1.050871 | 0.399971 | 6 | 6 | 0.036456 | 2.305690 | weak_profit_factor |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1.500000 | 1.500000 | 170 | 0.082947 | 0.000286 | 1.472036 | 1.272121 | 6 | 6 | 0.016065 | 5.163241 | not_stable_across_folds |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.250000 | 1.500000 | 352 | 0.079636 | 0.000234 | 1.153174 | 1.199206 | 7 | 5 | 0.040630 | 1.960041 | candidate |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 1.500000 | 224 | 0.063525 | -0.000003 | 1.201597 | 0.665983 | 6 | 6 | 0.015539 | 4.088004 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 1.500000 | 1.500000 | 171 | 0.059431 | 0.000302 | 1.374394 | 2.239673 | 9 | 3 | 0.021484 | 2.766344 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.250000 | 1.500000 | 536 | 0.042229 | 0.000112 | 1.394202 | 2.089226 | 8 | 4 | 0.108009 | 0.390977 | candidate |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.500000 | 1.500000 | 247 | 0.021748 | 0.000144 | 1.153144 | 0.991815 | 8 | 4 | 0.042589 | 0.510645 | weak_sharpe |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 2.000000 | 1.500000 | 25 | 0.015540 | 0.000415 | 1.425830 | 0.177545 | 5 | 5 | 0.006376 | 2.437322 | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 1.500000 | 18 | 0.011550 | 0.000354 | inf | 3.288316 | 5 | 3 | 0.006376 | 1.811471 | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2.000000 | 1.500000 | 15 | 0.001683 | 0.000021 | 0.620882 | -1.321696 | 3 | 4 | 0.010138 | 0.165970 | insufficient_trades |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 3.000000 | 1.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3.000000 | 1.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k4__seed42__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 1.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 3.000000 | 1.500000 | 0 | 0.000000 | 0.000000 |  |  | 0 | 0 | 0.000000 |  | insufficient_trades |
| minimal_vwap_location__k3__seed42__state0__momentum_ret_3__h6__c1 | test | 1.000000 | 1.500000 | 1026 | -0.002521 | -0.000009 | 1.079710 | 0.851758 | 7 | 5 | 0.094236 | -0.026747 | negative_economic |
| rich_extreme_reversion__k5__seed7__state3__momentum_ret_3__h6__c1 | test | 2.000000 | 1.500000 | 24 | -0.005077 | 0.000244 | 1.597035 | 1.126126 | 5 | 4 | 0.022688 | -0.223780 | insufficient_trades |

## Selected Threshold Fold Detail

| candidate_id | split | fold | threshold_multiplier | cost_bps | trades | net_return | avg_trade_net | profit_factor | daily_sharpe | max_drawdown_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 0 | 0.750000 | 0.500000 | 59 | 0.001427 | 0.000024 | 1.074398 | 0.789067 | 0.006757 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 0 | 0.750000 | 1.000000 | 59 | -0.001523 | -0.000026 | 0.927173 | -0.868215 | 0.009107 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 0 | 1.500000 | 1.500000 | 11 | -0.001487 | -0.000135 | 0.654539 | -2.463873 | 0.003507 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 0 | 1.500000 | 2.000000 | 11 | -0.002037 | -0.000185 | 0.557620 | -3.381632 | 0.003857 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0 | 0.750000 | 0.500000 | 80 | 0.001034 | 0.000013 | 1.028204 | 0.250531 | 0.014795 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0 | 0.750000 | 1.000000 | 80 | -0.002966 | -0.000037 | 0.924074 | -0.699229 | 0.017295 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0 | 1.500000 | 1.500000 | 10 | 0.003793 | 0.000379 | 2.039426 | 3.245216 | 0.001658 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 0 | 1.500000 | 2.000000 | 10 | 0.003293 | 0.000329 | 1.866777 | 2.817972 | 0.001708 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 1 | 0.750000 | 0.500000 | 72 | 0.008611 | 0.000120 | 1.200957 | 2.561835 | 0.019518 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 1 | 0.750000 | 1.000000 | 72 | 0.005011 | 0.000070 | 1.112016 | 1.548211 | 0.021218 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 1 | 1.500000 | 1.500000 | 9 | 0.002854 | 0.000317 | 1.567399 | 2.235420 | 0.002952 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 1 | 1.500000 | 2.000000 | 9 | 0.002404 | 0.000267 | 1.459649 | 1.941189 | 0.003052 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1 | 0.750000 | 0.500000 | 126 | -0.014475 | -0.000115 | 0.878627 | -1.857015 | 0.037359 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1 | 0.750000 | 1.000000 | 126 | -0.020775 | -0.000165 | 0.830353 | -2.701702 | 0.042809 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1 | 1.500000 | 1.500000 | 20 | -0.010967 | -0.000548 | 0.489536 | -3.569548 | 0.016055 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 1 | 1.500000 | 2.000000 | 20 | -0.011967 | -0.000598 | 0.454443 | -3.885148 | 0.016905 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 2 | 0.750000 | 0.500000 | 105 | -0.032121 | -0.000306 | 0.716581 | -4.383672 | 0.055004 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 2 | 0.750000 | 1.000000 | 105 | -0.037371 | -0.000356 | 0.678208 | -5.143960 | 0.059404 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 2 | 1.500000 | 1.500000 | 19 | -0.011543 | -0.000608 | 0.462694 | -4.078161 | 0.016631 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 2 | 1.500000 | 2.000000 | 19 | -0.012493 | -0.000658 | 0.430432 | -4.403704 | 0.017431 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2 | 0.750000 | 0.500000 | 178 | 0.123489 | 0.000694 | 1.613202 | 6.101626 | 0.062279 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2 | 0.750000 | 1.000000 | 178 | 0.114589 | 0.000644 | 1.559116 | 5.678067 | 0.063029 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2 | 1.500000 | 1.500000 | 25 | 0.035036 | 0.001401 | 3.147304 | 5.993702 | 0.010138 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 2 | 1.500000 | 2.000000 | 25 | 0.033786 | 0.001351 | 3.033310 | 5.829941 | 0.010238 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 3 | 0.750000 | 0.500000 | 133 | 0.054043 | 0.000406 | 1.335285 | 3.336574 | 0.037571 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 3 | 0.750000 | 1.000000 | 133 | 0.047393 | 0.000356 | 1.288593 | 2.926930 | 0.038321 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 3 | 1.500000 | 1.500000 | 17 | 0.014016 | 0.000824 | 1.683052 | 2.929213 | 0.015248 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 3 | 1.500000 | 2.000000 | 17 | 0.013166 | 0.000774 | 1.632381 | 2.754340 | 0.015548 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3 | 0.750000 | 0.500000 | 178 | -0.002162 | -0.000012 | 0.988643 | -0.149961 | 0.050728 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3 | 0.750000 | 1.000000 | 178 | -0.011062 | -0.000062 | 0.943149 | -0.764691 | 0.052028 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3 | 1.500000 | 1.500000 | 23 | 0.021339 | 0.000928 | 2.549251 | 5.136728 | 0.007058 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 3 | 1.500000 | 2.000000 | 23 | 0.020189 | 0.000878 | 2.434512 | 4.885212 | 0.007208 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 4 | 0.750000 | 0.500000 | 121 | 0.074347 | 0.000614 | 1.620862 | 5.616316 | 0.025707 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 4 | 0.750000 | 1.000000 | 121 | 0.068297 | 0.000564 | 1.557991 | 5.222087 | 0.026457 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 4 | 1.500000 | 1.500000 | 24 | -0.008405 | -0.000350 | 0.733875 | -2.255007 | 0.021179 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 4 | 1.500000 | 2.000000 | 24 | -0.009605 | -0.000400 | 0.702473 | -2.571708 | 0.021929 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 4 | 0.750000 | 0.500000 | 126 | 0.054629 | 0.000434 | 1.343349 | 5.450998 | 0.036063 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 4 | 0.750000 | 1.000000 | 126 | 0.048329 | 0.000384 | 1.297947 | 4.843910 | 0.037613 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 4 | 1.500000 | 1.500000 | 20 | -0.016881 | -0.000844 | 0.442779 | -8.512113 | 0.016065 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 4 | 1.500000 | 2.000000 | 20 | -0.017881 | -0.000894 | 0.421233 | -8.861877 | 0.016915 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 5 | 0.750000 | 0.500000 | 59 | 0.034056 | 0.000577 | 2.801348 | 11.208463 | 0.005496 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 5 | 0.750000 | 1.000000 | 59 | 0.031106 | 0.000527 | 2.570544 | 10.432522 | 0.006546 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 5 | 1.500000 | 1.500000 | 7 | 0.009710 | 0.001387 | inf | 11.978663 | 0.000000 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 5 | 1.500000 | 2.000000 | 7 | 0.009360 | 0.001337 | 204.627534 | 11.976790 | 0.000046 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 5 | 0.750000 | 0.500000 | 35 | 0.008978 | 0.000257 | 2.104005 | 11.708494 | 0.003971 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 5 | 0.750000 | 1.000000 | 35 | 0.007228 | 0.000207 | 1.824523 | 10.501405 | 0.004242 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 5 | 1.500000 | 1.500000 | 5 | -0.001581 | -0.000316 | 0.370785 | -13.391073 | 0.002513 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 5 | 1.500000 | 2.000000 | 5 | -0.001831 | -0.000366 | 0.325024 | -13.107110 | 0.002713 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 6 | 0.750000 | 0.500000 | 52 | 0.015209 | 0.000292 | 1.311422 | 2.236272 | 0.016361 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 6 | 0.750000 | 1.000000 | 52 | 0.012609 | 0.000242 | 1.251490 | 1.840554 | 0.016961 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 6 | 1.500000 | 1.500000 | 6 | 0.001091 | 0.000182 | 1.195834 | 1.433785 | 0.005310 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 6 | 1.500000 | 2.000000 | 6 | 0.000791 | 0.000132 | 1.139492 | 1.032208 | 0.005360 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 6 | 0.750000 | 0.500000 | 43 | 0.019788 | 0.000460 | 1.458975 | 5.876526 | 0.021229 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 6 | 0.750000 | 1.000000 | 43 | 0.017638 | 0.000410 | 1.400285 | 5.259069 | 0.022129 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 6 | 1.500000 | 1.500000 | 7 | 0.011506 | 0.001644 | 6.717793 | 12.341529 | 0.001074 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 6 | 1.500000 | 2.000000 | 7 | 0.011156 | 0.001594 | 6.281397 | 12.174773 | 0.001124 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 7 | 0.750000 | 0.500000 | 70 | 0.030077 | 0.000430 | 1.467486 | 5.883064 | 0.025876 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 7 | 0.750000 | 1.000000 | 70 | 0.026577 | 0.000380 | 1.403980 | 5.235013 | 0.027026 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 7 | 1.500000 | 1.500000 | 13 | 0.016038 | 0.001234 | 6.245576 | 9.674010 | 0.001074 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 7 | 1.500000 | 2.000000 | 13 | 0.015388 | 0.001184 | 5.723955 | 9.459396 | 0.001124 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 7 | 0.750000 | 0.500000 | 59 | 0.002780 | 0.000047 | 1.046238 | 0.520412 | 0.014961 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 7 | 0.750000 | 1.000000 | 59 | -0.000170 | -0.000003 | 0.997231 | -0.032033 | 0.015611 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 7 | 1.500000 | 1.500000 | 3 | -0.004265 | -0.001422 | 0.022600 | -4.294119 | 0.004364 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 7 | 1.500000 | 2.000000 | 3 | -0.004415 | -0.001472 | 0.008804 | -4.404284 | 0.004415 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 8 | 0.750000 | 0.500000 | 73 | -0.030823 | -0.000422 | 0.689557 | -4.071560 | 0.052110 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 8 | 0.750000 | 1.000000 | 73 | -0.034473 | -0.000472 | 0.659146 | -4.594870 | 0.054810 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 8 | 1.500000 | 1.500000 | 3 | -0.004265 | -0.001422 | 0.022600 | -3.756860 | 0.004364 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 8 | 1.500000 | 2.000000 | 3 | -0.004415 | -0.001472 | 0.008804 | -3.851409 | 0.004415 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 8 | 0.750000 | 0.500000 | 112 | 0.074630 | 0.000666 | 2.030292 | 6.624097 | 0.017140 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 8 | 0.750000 | 1.000000 | 112 | 0.069030 | 0.000616 | 1.926132 | 6.163206 | 0.017790 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 8 | 1.500000 | 1.500000 | 13 | 0.009794 | 0.000753 | 2.289684 | 3.375080 | 0.006823 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 8 | 1.500000 | 2.000000 | 13 | 0.009144 | 0.000703 | 2.158337 | 3.173390 | 0.007173 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 9 | 0.750000 | 0.500000 | 26 | 0.023301 | 0.000896 | 2.689892 | 13.513549 | 0.008227 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 9 | 0.750000 | 1.000000 | 26 | 0.022001 | 0.000846 | 2.550630 | 12.932045 | 0.008427 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 9 | 1.500000 | 1.500000 | 3 | 0.010415 | 0.003472 | inf | 11.814421 | 0.000000 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 9 | 1.500000 | 2.000000 | 3 | 0.010265 | 0.003422 | inf | 11.805039 | 0.000000 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 9 | 0.750000 | 0.500000 | 58 | -0.008317 | -0.000143 | 0.824873 | -1.605025 | 0.026720 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 9 | 0.750000 | 1.000000 | 58 | -0.011217 | -0.000193 | 0.770570 | -2.173745 | 0.027620 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 9 | 1.500000 | 1.500000 | 8 | -0.009485 | -0.001186 | 0.278804 | -8.138988 | 0.013151 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 9 | 1.500000 | 2.000000 | 8 | -0.009885 | -0.001236 | 0.265152 | -8.425975 | 0.013451 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 10 | 0.750000 | 0.500000 | 53 | -0.011555 | -0.000218 | 0.746676 | -2.403696 | 0.027508 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 10 | 0.750000 | 1.000000 | 53 | -0.014205 | -0.000268 | 0.697852 | -2.960795 | 0.028608 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 10 | 1.500000 | 1.500000 | 8 | -0.008244 | -0.001030 | 0.307849 | -9.010245 | 0.011911 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 10 | 1.500000 | 2.000000 | 8 | -0.008644 | -0.001080 | 0.292096 | -9.461486 | 0.012211 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 10 | 0.750000 | 0.500000 | 112 | 0.041395 | 0.000370 | 1.328655 | 3.042116 | 0.039508 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 10 | 0.750000 | 1.000000 | 112 | 0.035795 | 0.000320 | 1.278135 | 2.670530 | 0.041258 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 10 | 1.500000 | 1.500000 | 19 | -0.001732 | -0.000091 | 0.904645 | -0.700974 | 0.011661 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 10 | 1.500000 | 2.000000 | 19 | -0.002682 | -0.000141 | 0.857017 | -1.088316 | 0.012011 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 11 | 0.750000 | 0.500000 | 72 | -0.011239 | -0.000156 | 0.845710 | -2.107367 | 0.033226 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 11 | 0.750000 | 1.000000 | 72 | -0.014839 | -0.000206 | 0.800666 | -2.842787 | 0.034926 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 11 | 1.500000 | 1.500000 | 15 | 0.009091 | 0.000606 | 2.352701 | 5.134481 | 0.003735 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | validation | 11 | 1.500000 | 2.000000 | 15 | 0.008341 | 0.000556 | 2.188068 | 4.777346 | 0.003835 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 11 | 0.750000 | 0.500000 | 133 | 0.071299 | 0.000536 | 1.691866 | 5.649930 | 0.018707 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 11 | 0.750000 | 1.000000 | 133 | 0.064649 | 0.000486 | 1.609608 | 5.143468 | 0.019357 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 11 | 1.500000 | 1.500000 | 17 | 0.046390 | 0.002729 | 6.335154 | 5.758699 | 0.003527 |
| minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1 | test | 11 | 1.500000 | 2.000000 | 17 | 0.045540 | 0.002679 | 6.148580 | 5.677659 | 0.003577 |

## Outputs

- `reports/hmm_candidate_thresholds/candidates.parquet`
- `reports/hmm_candidate_thresholds/threshold_fold_metrics.parquet`
- `reports/hmm_candidate_thresholds/threshold_summary.parquet`
- `reports/hmm_candidate_thresholds/selected_validation_thresholds.parquet`
- `reports/hmm_candidate_thresholds/selected_test_results.parquet`
- `reports/hmm_candidate_thresholds/candidate_decisions.parquet`

## Conclusion

No candidate survives 2 bps. Best fallback is `minimal_vwap_location__k5__seed42__state1__momentum_ret_3__h6__c1`, but it remains cost-fragile.

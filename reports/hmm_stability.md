# HMM Stability

## Scope

- Feature sets: 2
- K grid: `[3, 4, 5]`
- Seeds: `[42, 7, 123]`
- Max folds: `12`
- Horizons: `[6]`
- Costs bps: `[1.0, 2.0]`

## Feature Set Stability Summary

| feature_set | cost_bps | combos | combos_with_validation_candidate | combos_with_test_candidate | total_validation_candidates | total_test_candidates | validation_combo_rate | test_combo_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_vwap_location | 1.000000 | 9 | 6 | 2 | 7 | 2 | 0.666667 | 0.222222 |
| rich_extreme_reversion | 1.000000 | 9 | 6 | 2 | 9 | 2 | 0.666667 | 0.222222 |
| rich_extreme_reversion | 2.000000 | 9 | 4 | 0 | 5 | 0 | 0.444444 | 0.000000 |
| minimal_vwap_location | 2.000000 | 9 | 1 | 0 | 1 | 0 | 0.111111 | 0.000000 |

## Combo Summary

| feature_set | n_states | seed | cost_bps | validation_candidates | test_candidates | best_validation_action | best_validation_total_net_return | best_validation_candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion | 3 | 42 | 1.000000 | 0 | 0 | momentum_ret_3 | 0.031254 | weak_profit_factor |
| rich_extreme_reversion | 3 | 42 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.070346 | negative_economic |
| rich_extreme_reversion | 3 | 7 | 1.000000 | 0 | 0 | momentum_ret_3 | 0.118160 | weak_profit_factor |
| rich_extreme_reversion | 3 | 7 | 2.000000 | 0 | 0 | momentum_ret_3 | 0.009560 | weak_profit_factor |
| rich_extreme_reversion | 3 | 123 | 1.000000 | 1 | 0 | momentum_ret_3 | 0.168178 | candidate |
| rich_extreme_reversion | 3 | 123 | 2.000000 | 1 | 0 | momentum_ret_3 | 0.076078 | candidate |
| rich_extreme_reversion | 4 | 42 | 1.000000 | 1 | 1 | momentum_ret_3 | 0.057424 | candidate |
| rich_extreme_reversion | 4 | 42 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.020761 | negative_economic |
| rich_extreme_reversion | 4 | 7 | 1.000000 | 0 | 0 | long | 0.033060 | weak_profit_factor |
| rich_extreme_reversion | 4 | 7 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.030944 | negative_economic |
| rich_extreme_reversion | 4 | 123 | 1.000000 | 2 | 0 | momentum_ret_3 | 0.143613 | candidate |
| rich_extreme_reversion | 4 | 123 | 2.000000 | 0 | 0 | momentum_ret_3 | 0.061613 | negative_economic |
| rich_extreme_reversion | 5 | 42 | 1.000000 | 1 | 0 | long | 0.387302 | candidate |
| rich_extreme_reversion | 5 | 42 | 2.000000 | 1 | 0 | long | 0.172002 | candidate |
| rich_extreme_reversion | 5 | 7 | 1.000000 | 3 | 1 | momentum_ret_3 | 0.208807 | candidate |
| rich_extreme_reversion | 5 | 7 | 2.000000 | 2 | 0 | momentum_ret_3 | 0.131407 | candidate |
| rich_extreme_reversion | 5 | 123 | 1.000000 | 1 | 0 | momentum_ret_3 | 0.259055 | candidate |
| rich_extreme_reversion | 5 | 123 | 2.000000 | 1 | 0 | momentum_ret_3 | 0.187555 | candidate |
| minimal_vwap_location | 3 | 42 | 1.000000 | 1 | 1 | momentum_ret_3 | 0.011583 | candidate |
| minimal_vwap_location | 3 | 42 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.078586 | negative_economic |
| minimal_vwap_location | 3 | 7 | 1.000000 | 1 | 0 | long | 0.208781 | candidate |
| minimal_vwap_location | 3 | 7 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.019357 | negative_economic |
| minimal_vwap_location | 3 | 123 | 1.000000 | 0 | 0 | short | 0.117318 | weak_profit_factor |
| minimal_vwap_location | 3 | 123 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.079588 | negative_economic |
| minimal_vwap_location | 4 | 42 | 1.000000 | 1 | 0 | momentum_ret_3 | 0.180227 | candidate |
| minimal_vwap_location | 4 | 42 | 2.000000 | 1 | 0 | momentum_ret_3 | 0.106527 | candidate |
| minimal_vwap_location | 4 | 7 | 1.000000 | 0 | 0 | momentum_ret_3 | 0.074188 | weak_profit_factor |
| minimal_vwap_location | 4 | 7 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.005312 | negative_economic |
| minimal_vwap_location | 4 | 123 | 1.000000 | 1 | 0 | short | 0.252566 | candidate |
| minimal_vwap_location | 4 | 123 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.004618 | negative_economic |
| minimal_vwap_location | 5 | 42 | 1.000000 | 1 | 1 | momentum_ret_3 | 0.054347 | candidate |
| minimal_vwap_location | 5 | 42 | 2.000000 | 0 | 0 | momentum_ret_3 | 0.002247 | weak_profit_factor |
| minimal_vwap_location | 5 | 7 | 1.000000 | 0 | 0 | momentum_ret_3 | 0.054912 | weak_profit_factor |
| minimal_vwap_location | 5 | 7 | 2.000000 | 0 | 0 | momentum_ret_3 | -0.015188 | negative_economic |
| minimal_vwap_location | 5 | 123 | 1.000000 | 2 | 0 | short | 0.139043 | candidate |
| minimal_vwap_location | 5 | 123 | 2.000000 | 0 | 0 | momentum_ret_3 | 0.037780 | weak_profit_factor |

## Top Validation Rankings

| feature_set | n_states | seed | horizon_bars | cost_bps | hmm_state | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_extreme_reversion | 5 | 42 | 6 | 1.000000 | 1 | long | 2153 | 0.387302 | 0.000275 | 1.286169 | 2.777946 | 9 | 3 | candidate |
| rich_extreme_reversion | 5 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | 715 | 0.259055 | 0.000281 | 1.423997 | 2.721032 | 9 | 3 | candidate |
| minimal_vwap_location | 4 | 123 | 6 | 1.000000 | 1 | short | 3522 | 0.252566 | 0.000117 | 1.108695 | 1.544273 | 8 | 4 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 4 | momentum_ret_3 | 774 | 0.208807 | 0.000231 | 1.381436 | 2.755441 | 8 | 4 | candidate |
| minimal_vwap_location | 3 | 7 | 6 | 1.000000 | 0 | long | 4450 | 0.208781 | 0.000104 | 1.134372 | 1.732383 | 8 | 4 | candidate |
| minimal_vwap_location | 5 | 42 | 6 | 1.000000 | 1 | short | 2467 | 0.187566 | 0.000097 | 1.023468 | 0.144322 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 5 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | 715 | 0.187555 | 0.000181 | 1.234898 | 1.574077 | 8 | 4 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | 737 | 0.180227 | 0.000209 | 1.315821 | 2.187013 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 42 | 6 | 2.000000 | 1 | long | 2153 | 0.172002 | 0.000175 | 1.102307 | 1.137813 | 7 | 5 | candidate |
| rich_extreme_reversion | 3 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | 921 | 0.168178 | 0.000144 | 1.203225 | 2.178409 | 7 | 5 | candidate |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 2 | momentum_ret_3 | 820 | 0.143613 | 0.000091 | 1.143973 | 1.277913 | 7 | 5 | candidate |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 1 | short | 2571 | 0.139043 | 0.000001 | 1.145009 | 1.427925 | 8 | 4 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 1 | short | 3877 | 0.135371 | -0.000044 | 0.991041 | -0.068302 | 6 | 6 | negative_economic |
| rich_extreme_reversion | 5 | 7 | 6 | 2.000000 | 4 | momentum_ret_3 | 774 | 0.131407 | 0.000131 | 1.169187 | 1.365414 | 7 | 5 | candidate |
| minimal_vwap_location | 3 | 7 | 6 | 1.000000 | 1 | short | 5229 | 0.128265 | 0.000022 | 1.006882 | 0.112058 | 7 | 5 | weak_profit_factor |
| rich_extreme_reversion | 3 | 7 | 6 | 1.000000 | 0 | momentum_ret_3 | 1086 | 0.118160 | 0.000113 | 1.081593 | 0.997497 | 7 | 5 | weak_profit_factor |
| minimal_vwap_location | 3 | 123 | 6 | 1.000000 | 1 | short | 4317 | 0.117318 | 0.000057 | 0.918526 | -0.953890 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 5 | 42 | 6 | 1.000000 | 4 | momentum_ret_3 | 611 | 0.113697 | -0.000063 | 1.225022 | 1.411572 | 7 | 5 | negative_economic |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 0 | momentum_ret_3 | 713 | 0.109080 | 0.000152 | 1.116923 | 1.249011 | 8 | 4 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 2.000000 | 3 | momentum_ret_3 | 737 | 0.106527 | 0.000109 | 1.196194 | 1.510023 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 3 | momentum_ret_3 | 704 | 0.104032 | 0.000187 | 1.318801 | 2.205832 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 0 | long | 2642 | 0.085810 | 0.000089 | 0.873585 | -1.376334 | 5 | 7 | weak_profit_factor |
| minimal_vwap_location | 4 | 123 | 6 | 1.000000 | 2 | momentum_ret_3 | 850 | 0.080382 | 0.000068 | 0.942806 | -0.596325 | 6 | 6 | weak_profit_factor |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 3 | long | 3882 | 0.076100 | 0.000082 | 0.994465 | -0.293430 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 3 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | 921 | 0.076078 | 0.000044 | 1.101498 | 1.194231 | 7 | 5 | candidate |
| minimal_vwap_location | 4 | 7 | 6 | 1.000000 | 2 | momentum_ret_3 | 795 | 0.074188 | 0.000088 | 1.090759 | 0.964680 | 7 | 5 | weak_profit_factor |
| minimal_vwap_location | 3 | 7 | 6 | 1.000000 | 0 | momentum_ret_3 | 906 | 0.071243 | 0.000088 | 1.015636 | 0.226382 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 4 | 123 | 6 | 2.000000 | 2 | momentum_ret_3 | 820 | 0.061613 | -0.000009 | 1.028048 | 0.292974 | 6 | 6 | negative_economic |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 0 | momentum_ret_3 | 777 | 0.060074 | 0.000012 | 1.053058 | 0.390181 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| minimal_vwap_location | 5 | 7 | 6 | 1.000000 | 1 | momentum_ret_3 | 701 | 0.054912 | 0.000014 | 1.084540 | 0.892301 | 7 | 5 | weak_profit_factor |
| minimal_vwap_location | 5 | 42 | 6 | 1.000000 | 1 | momentum_ret_3 | 521 | 0.054347 | 0.000148 | 1.115162 | 1.179104 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 42 | 6 | 2.000000 | 4 | momentum_ret_3 | 611 | 0.052597 | -0.000163 | 1.078094 | 0.535020 | 7 | 5 | negative_economic |
| minimal_vwap_location | 5 | 42 | 6 | 1.000000 | 1 | random_symmetric | 2467 | 0.048297 | 0.000016 | 1.053104 | 1.058109 | 7 | 5 | random_benchmark |
| rich_extreme_reversion | 3 | 7 | 6 | 1.000000 | 2 | short | 4172 | 0.043955 | 0.000012 | 0.994032 | -0.084471 | 4 | 8 | weak_profit_factor |
| rich_extreme_reversion | 4 | 42 | 6 | 1.000000 | 1 | momentum_ret_3 | 634 | 0.042639 | 0.000034 | 1.007870 | -0.039820 | 6 | 6 | weak_profit_factor |
| minimal_vwap_location | 5 | 123 | 6 | 2.000000 | 0 | momentum_ret_3 | 713 | 0.037780 | 0.000052 | 0.996200 | -0.006136 | 6 | 6 | weak_profit_factor |
| minimal_vwap_location | 5 | 7 | 6 | 1.000000 | 2 | momentum_ret_3 | 808 | 0.037087 | 0.000031 | 0.918916 | -1.212488 | 5 | 7 | weak_profit_factor |
| minimal_vwap_location | 4 | 123 | 6 | 1.000000 | 3 | long | 3138 | 0.035305 | 0.000029 | 1.029967 | 0.445304 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | 831 | 0.034324 | 0.000079 | 1.118581 | 1.026174 | 7 | 5 | candidate |

## Validation Candidate Holdout Sanity

| feature_set | n_states | seed | horizon_bars | cost_bps | hmm_state | action | split | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_vwap_location | 3 | 7 | 6 | 1.000000 | 0 | long | validation | 4450 | 0.208781 | 0.000104 | 1.134372 | 1.732383 | 8 | 4 | candidate |
| minimal_vwap_location | 3 | 7 | 6 | 1.000000 | 0 | long | test | 4275 | -0.868694 | -0.000231 | 0.833274 | -2.559902 | 2 | 10 | negative_economic |
| minimal_vwap_location | 3 | 42 | 6 | 1.000000 | 0 | momentum_ret_3 | validation | 912 | 0.011583 | 0.000092 | 1.107221 | 1.786007 | 9 | 3 | candidate |
| minimal_vwap_location | 3 | 42 | 6 | 1.000000 | 0 | momentum_ret_3 | test | 1026 | 0.048779 | 0.000041 | 1.161592 | 1.972969 | 7 | 5 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 737 | 0.180227 | 0.000209 | 1.315821 | 2.187013 | 7 | 5 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | test | 827 | 0.130668 | 0.000092 | 1.030768 | 0.323806 | 7 | 5 | weak_profit_factor |
| minimal_vwap_location | 4 | 42 | 6 | 2.000000 | 3 | momentum_ret_3 | validation | 737 | 0.106527 | 0.000109 | 1.196194 | 1.510023 | 7 | 5 | candidate |
| minimal_vwap_location | 4 | 42 | 6 | 2.000000 | 3 | momentum_ret_3 | test | 827 | 0.047968 | -0.000008 | 0.934505 | -0.706388 | 5 | 7 | negative_economic |
| minimal_vwap_location | 4 | 123 | 6 | 1.000000 | 1 | short | validation | 3522 | 0.252566 | 0.000117 | 1.108695 | 1.544273 | 8 | 4 | candidate |
| minimal_vwap_location | 4 | 123 | 6 | 1.000000 | 1 | short | test | 3469 | -0.816114 | -0.000236 | 0.867930 | -1.921975 | 5 | 7 | negative_economic |
| minimal_vwap_location | 5 | 42 | 6 | 1.000000 | 1 | momentum_ret_3 | validation | 521 | 0.054347 | 0.000148 | 1.115162 | 1.179104 | 7 | 5 | candidate |
| minimal_vwap_location | 5 | 42 | 6 | 1.000000 | 1 | momentum_ret_3 | test | 756 | 0.204444 | 0.000217 | 1.258477 | 2.593584 | 7 | 5 | candidate |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 0 | momentum_ret_3 | validation | 713 | 0.109080 | 0.000152 | 1.116923 | 1.249011 | 8 | 4 | candidate |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 0 | momentum_ret_3 | test | 661 | -0.132957 | -0.000198 | 0.972574 | -0.277051 | 6 | 6 | negative_economic |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 1 | short | validation | 2571 | 0.139043 | 0.000001 | 1.145009 | 1.427925 | 8 | 4 | candidate |
| minimal_vwap_location | 5 | 123 | 6 | 1.000000 | 1 | short | test | 3058 | -0.051778 | -0.000069 | 1.011579 | 0.144517 | 7 | 5 | negative_economic |
| rich_extreme_reversion | 3 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | validation | 921 | 0.168178 | 0.000144 | 1.203225 | 2.178409 | 7 | 5 | candidate |
| rich_extreme_reversion | 3 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | test | 1008 | -0.041071 | -0.000042 | 0.930368 | -0.653417 | 6 | 6 | negative_economic |
| rich_extreme_reversion | 3 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | validation | 921 | 0.076078 | 0.000044 | 1.101498 | 1.194231 | 7 | 5 | candidate |
| rich_extreme_reversion | 3 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | test | 1008 | -0.141871 | -0.000142 | 0.833033 | -2.006420 | 4 | 8 | negative_economic |
| rich_extreme_reversion | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 791 | 0.057424 | 0.000038 | 1.102380 | 1.281719 | 8 | 4 | candidate |
| rich_extreme_reversion | 4 | 42 | 6 | 1.000000 | 3 | momentum_ret_3 | test | 969 | 0.202882 | 0.000113 | 1.199959 | 3.071992 | 8 | 4 | candidate |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | validation | 831 | 0.034324 | 0.000079 | 1.118581 | 1.026174 | 7 | 5 | candidate |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | test | 706 | 0.010573 | -0.000024 | 0.959471 | -0.378017 | 6 | 6 | negative_economic |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 2 | momentum_ret_3 | validation | 820 | 0.143613 | 0.000091 | 1.143973 | 1.277913 | 7 | 5 | candidate |
| rich_extreme_reversion | 4 | 123 | 6 | 1.000000 | 2 | momentum_ret_3 | test | 724 | -0.041244 | -0.000106 | 0.818203 | -1.928314 | 4 | 8 | negative_economic |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 0 | reversion_ret_3 | validation | 478 | 0.021436 | 0.000088 | 1.255316 | 2.439542 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 0 | reversion_ret_3 | test | 545 | -0.182172 | -0.000288 | 0.601767 | -6.587638 | 2 | 10 | negative_economic |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 3 | momentum_ret_3 | validation | 704 | 0.104032 | 0.000187 | 1.318801 | 2.205832 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 3 | momentum_ret_3 | test | 656 | 0.118942 | 0.000176 | 1.209131 | 2.701117 | 8 | 4 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 4 | momentum_ret_3 | validation | 774 | 0.208807 | 0.000231 | 1.381436 | 2.755441 | 8 | 4 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 1.000000 | 4 | momentum_ret_3 | test | 706 | 0.001087 | 0.000001 | 1.016617 | -0.486419 | 6 | 6 | weak_profit_factor |
| rich_extreme_reversion | 5 | 7 | 6 | 2.000000 | 3 | momentum_ret_3 | validation | 704 | 0.033632 | 0.000087 | 1.164194 | 1.203870 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 2.000000 | 3 | momentum_ret_3 | test | 656 | 0.053342 | 0.000076 | 1.036488 | 0.495772 | 7 | 5 | weak_profit_factor |
| rich_extreme_reversion | 5 | 7 | 6 | 2.000000 | 4 | momentum_ret_3 | validation | 774 | 0.131407 | 0.000131 | 1.169187 | 1.365414 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 7 | 6 | 2.000000 | 4 | momentum_ret_3 | test | 706 | -0.069513 | -0.000099 | 0.911624 | -1.604575 | 5 | 7 | negative_economic |
| rich_extreme_reversion | 5 | 42 | 6 | 1.000000 | 1 | long | validation | 2153 | 0.387302 | 0.000275 | 1.286169 | 2.777946 | 9 | 3 | candidate |
| rich_extreme_reversion | 5 | 42 | 6 | 1.000000 | 1 | long | test | 2810 | 0.187796 | 0.000025 | 0.966202 | -0.298214 | 5 | 7 | weak_profit_factor |
| rich_extreme_reversion | 5 | 42 | 6 | 2.000000 | 1 | long | validation | 2153 | 0.172002 | 0.000175 | 1.102307 | 1.137813 | 7 | 5 | candidate |
| rich_extreme_reversion | 5 | 42 | 6 | 2.000000 | 1 | long | test | 2810 | -0.093204 | -0.000075 | 0.870538 | -1.249196 | 5 | 7 | negative_economic |
| rich_extreme_reversion | 5 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | validation | 715 | 0.259055 | 0.000281 | 1.423997 | 2.721032 | 9 | 3 | candidate |
| rich_extreme_reversion | 5 | 123 | 6 | 1.000000 | 1 | momentum_ret_3 | test | 648 | 0.032523 | -0.000033 | 0.918435 | -0.813460 | 4 | 8 | negative_economic |
| rich_extreme_reversion | 5 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | validation | 715 | 0.187555 | 0.000181 | 1.234898 | 1.574077 | 8 | 4 | candidate |
| rich_extreme_reversion | 5 | 123 | 6 | 2.000000 | 1 | momentum_ret_3 | test | 648 | -0.032277 | -0.000133 | 0.823783 | -1.665927 | 4 | 8 | negative_economic |

## Feature Set Validation

| feature_set | n_features | columns | missing_columns | status |
| --- | --- | --- | --- | --- |
| rich_extreme_reversion | 9 | dist_session_high_atr,dist_session_low_atr,pos_session_range,intraday_drawdown,intraday_runup,dist_vwap_atr,rv_12,ret_3,rel_volume |  | ready |
| minimal_vwap_location | 6 | dist_vwap_atr,dist_open,pos_session_range,intraday_drawdown,rv_12,rel_volume |  | ready |

## Conclusion

At least one feature set has test candidates across multiple K/seed combinations. Promote only those rows to deeper interpretation and frozen evaluation.

State ids are not compared directly across K/seed because HMM state labels are permutation-dependent. This report asks whether an economically similar state/action candidate reappears across independent fits.

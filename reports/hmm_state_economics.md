# HMM State Economics

## Scope

- Features: `data/features/features_base.parquet`
- HMM states: 4
- HMM fit: train sessions only per walk-forward fold
- HMM inference: online filtered probabilities on validation/test
- Horizons: `[1, 2, 3, 6]`
- Costs bps: `[1.0, 2.0, 5.0]`
- Actions: `long, short, momentum_ret_3, reversion_ret_3, random_symmetric, flat`

## Candidate Status Counts

| candidate_status | count |
| --- | --- |
| negative_economic | 192 |
| random_benchmark | 48 |

## Top Validation Rankings

| horizon_bars | cost_bps | hmm_state | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1.000000 | 3 | momentum_ret_3 | 3076 | 0.076151 | -0.000036 | 0.999971 | -0.000227 | 27 | 28 | negative_economic |
| 6 | 1.000000 | 2 | momentum_ret_3 | 3988 | 0.007976 | -0.000074 | 0.907850 | -1.108084 | 25 | 30 | negative_economic |
| 6 | 1.000000 | 0 | momentum_ret_3 | 3349 | -0.006886 | 0.000043 | 1.012689 | 0.129596 | 28 | 25 | negative_economic |
| 6 | 1.000000 | 1 | reversion_ret_3 | 3478 | -0.043954 | -0.000003 | 0.973684 | -0.279162 | 25 | 29 | negative_economic |
| 6 | 2.000000 | 3 | momentum_ret_3 | 3076 | -0.231449 | -0.000136 | 0.865401 | -1.642862 | 20 | 35 | negative_economic |
| 3 | 1.000000 | 3 | momentum_ret_3 | 5864 | -0.325949 | -0.000070 | 0.853972 | -2.456421 | 23 | 32 | negative_economic |
| 6 | 2.000000 | 0 | momentum_ret_3 | 3349 | -0.341786 | -0.000053 | 0.819272 | -2.138866 | 17 | 36 | negative_economic |
| 6 | 2.000000 | 2 | momentum_ret_3 | 3988 | -0.390824 | -0.000174 | 0.791061 | -1.827299 | 20 | 35 | negative_economic |
| 6 | 2.000000 | 1 | reversion_ret_3 | 3478 | -0.391754 | -0.000101 | 0.818718 | -1.925489 | 16 | 38 | negative_economic |
| 3 | 1.000000 | 1 | reversion_ret_3 | 6713 | -0.394633 | -0.000050 | 0.836055 | -2.407331 | 15 | 40 | negative_economic |
| 3 | 1.000000 | 2 | momentum_ret_3 | 7449 | -0.415378 | -0.000104 | 0.787000 | -3.568011 | 14 | 41 | negative_economic |
| 3 | 1.000000 | 0 | momentum_ret_3 | 6667 | -0.539539 | -0.000055 | 0.824133 | -2.741716 | 11 | 42 | negative_economic |
| 2 | 1.000000 | 1 | reversion_ret_3 | 8566 | -0.625967 | -0.000034 | 0.791824 | -3.698973 | 13 | 42 | negative_economic |
| 6 | 1.000000 | 1 | momentum_ret_3 | 3478 | -0.651646 | -0.000194 | 0.798789 | -2.723732 | 19 | 35 | negative_economic |
| 6 | 1.000000 | 0 | reversion_ret_3 | 3349 | -0.662914 | -0.000236 | 0.670926 | -3.496455 | 12 | 41 | negative_economic |
| 6 | 1.000000 | 3 | reversion_ret_3 | 3076 | -0.691351 | -0.000164 | 0.744980 | -2.885156 | 15 | 40 | negative_economic |
| 2 | 1.000000 | 3 | momentum_ret_3 | 7341 | -0.712610 | -0.000104 | 0.807956 | -3.798939 | 12 | 43 | negative_economic |
| 2 | 1.000000 | 2 | momentum_ret_3 | 9379 | -0.717044 | -0.000118 | 0.748029 | -4.633657 | 13 | 42 | negative_economic |
| 6 | 1.000000 | 3 | long | 14322 | -0.729347 | -0.000081 | 0.914269 | -0.953910 | 22 | 33 | negative_economic |
| 2 | 1.000000 | 3 | reversion_ret_3 | 7341 | -0.755590 | -0.000096 | 0.713963 | -3.878510 | 13 | 42 | negative_economic |
| 3 | 1.000000 | 0 | reversion_ret_3 | 6667 | -0.793861 | -0.000138 | 0.718543 | -4.992862 | 8 | 45 | negative_economic |
| 6 | 1.000000 | 2 | reversion_ret_3 | 3988 | -0.805576 | -0.000126 | 0.776500 | -2.559064 | 18 | 37 | negative_economic |
| 1 | 1.000000 | 3 | reversion_ret_3 | 9601 | -0.808674 | -0.000082 | 0.706645 | -5.479343 | 7 | 48 | negative_economic |
| 2 | 1.000000 | 0 | reversion_ret_3 | 8640 | -0.845403 | -0.000103 | 0.712698 | -5.113720 | 6 | 47 | negative_economic |
| 3 | 1.000000 | 3 | reversion_ret_3 | 5864 | -0.846851 | -0.000130 | 0.716786 | -4.146057 | 11 | 44 | negative_economic |

## Best Action By State

| horizon_bars | cost_bps | hmm_state | action | total_trades | total_net_return | avg_trade_net | median_profit_factor | median_daily_sharpe | positive_folds | negative_folds | candidate_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000000 | 0 | reversion_ret_3 | 11571 | -1.137002 | -0.000097 | 0.635538 | -8.628567 | 2 | 51 | negative_economic |
| 1 | 1.000000 | 1 | reversion_ret_3 | 11322 | -0.994417 | -0.000088 | 0.703507 | -7.598931 | 5 | 50 | negative_economic |
| 1 | 1.000000 | 2 | reversion_ret_3 | 12215 | -1.188799 | -0.000084 | 0.683951 | -8.425342 | 5 | 50 | negative_economic |
| 1 | 1.000000 | 3 | reversion_ret_3 | 9601 | -0.808674 | -0.000082 | 0.706645 | -5.479343 | 7 | 48 | negative_economic |
| 1 | 2.000000 | 0 | reversion_ret_3 | 11571 | -2.294102 | -0.000194 | 0.401198 | -13.344402 | 0 | 53 | negative_economic |
| 1 | 2.000000 | 1 | reversion_ret_3 | 11322 | -2.126617 | -0.000188 | 0.472870 | -12.858703 | 0 | 55 | negative_economic |
| 1 | 2.000000 | 2 | reversion_ret_3 | 12215 | -2.410299 | -0.000184 | 0.488760 | -12.186908 | 2 | 53 | negative_economic |
| 1 | 2.000000 | 3 | reversion_ret_3 | 9601 | -1.768774 | -0.000182 | 0.488482 | -10.348269 | 3 | 52 | negative_economic |
| 1 | 5.000000 | 0 | reversion_ret_3 | 11571 | -5.765402 | -0.000483 | 0.109642 | -17.901716 | 0 | 53 | negative_economic |
| 1 | 5.000000 | 1 | reversion_ret_3 | 11322 | -5.523217 | -0.000488 | 0.153055 | -18.151344 | 0 | 55 | negative_economic |
| 1 | 5.000000 | 2 | reversion_ret_3 | 12215 | -6.074799 | -0.000484 | 0.200290 | -17.601491 | 1 | 54 | negative_economic |
| 1 | 5.000000 | 3 | reversion_ret_3 | 9601 | -4.649074 | -0.000482 | 0.150743 | -15.731073 | 0 | 55 | negative_economic |
| 2 | 1.000000 | 0 | reversion_ret_3 | 8640 | -0.845403 | -0.000103 | 0.712698 | -5.113720 | 6 | 47 | negative_economic |
| 2 | 1.000000 | 1 | reversion_ret_3 | 8566 | -0.625967 | -0.000034 | 0.791824 | -3.698973 | 13 | 42 | negative_economic |
| 2 | 1.000000 | 2 | momentum_ret_3 | 9379 | -0.717044 | -0.000118 | 0.748029 | -4.633657 | 13 | 42 | negative_economic |
| 2 | 1.000000 | 3 | momentum_ret_3 | 7341 | -0.712610 | -0.000104 | 0.807956 | -3.798939 | 12 | 43 | negative_economic |
| 2 | 2.000000 | 0 | reversion_ret_3 | 8640 | -1.709403 | -0.000199 | 0.523032 | -8.631684 | 1 | 52 | negative_economic |
| 2 | 2.000000 | 1 | reversion_ret_3 | 8566 | -1.482567 | -0.000134 | 0.591824 | -7.760014 | 8 | 47 | negative_economic |
| 2 | 2.000000 | 2 | momentum_ret_3 | 9379 | -1.654944 | -0.000218 | 0.577812 | -7.397033 | 6 | 49 | negative_economic |
| 2 | 2.000000 | 3 | momentum_ret_3 | 7341 | -1.446710 | -0.000204 | 0.606669 | -7.924157 | 6 | 49 | negative_economic |
| 2 | 5.000000 | 0 | reversion_ret_3 | 8640 | -4.301403 | -0.000488 | 0.193398 | -13.905650 | 0 | 53 | negative_economic |
| 2 | 5.000000 | 1 | reversion_ret_3 | 8566 | -4.052367 | -0.000434 | 0.262934 | -13.850407 | 2 | 53 | negative_economic |
| 2 | 5.000000 | 2 | momentum_ret_3 | 9379 | -4.468644 | -0.000518 | 0.284583 | -15.674250 | 1 | 54 | negative_economic |
| 2 | 5.000000 | 3 | momentum_ret_3 | 7341 | -3.649010 | -0.000504 | 0.286593 | -14.586596 | 0 | 55 | negative_economic |
| 3 | 1.000000 | 0 | momentum_ret_3 | 6667 | -0.539539 | -0.000055 | 0.824133 | -2.741716 | 11 | 42 | negative_economic |
| 3 | 1.000000 | 1 | reversion_ret_3 | 6713 | -0.394633 | -0.000050 | 0.836055 | -2.407331 | 15 | 40 | negative_economic |
| 3 | 1.000000 | 2 | momentum_ret_3 | 7449 | -0.415378 | -0.000104 | 0.787000 | -3.568011 | 14 | 41 | negative_economic |
| 3 | 1.000000 | 3 | momentum_ret_3 | 5864 | -0.325949 | -0.000070 | 0.853972 | -2.456421 | 23 | 32 | negative_economic |
| 3 | 2.000000 | 0 | momentum_ret_3 | 6667 | -1.206239 | -0.000151 | 0.605888 | -6.153604 | 5 | 48 | negative_economic |
| 3 | 2.000000 | 1 | reversion_ret_3 | 6713 | -1.065933 | -0.000150 | 0.686372 | -4.854774 | 8 | 47 | negative_economic |
| 3 | 2.000000 | 2 | momentum_ret_3 | 7449 | -1.160278 | -0.000204 | 0.603500 | -5.845244 | 10 | 45 | negative_economic |
| 3 | 2.000000 | 3 | momentum_ret_3 | 5864 | -0.912349 | -0.000170 | 0.704409 | -4.894498 | 13 | 42 | negative_economic |
| 3 | 5.000000 | 0 | momentum_ret_3 | 6667 | -3.206339 | -0.000440 | 0.289919 | -13.910439 | 2 | 51 | negative_economic |
| 3 | 5.000000 | 1 | reversion_ret_3 | 6713 | -3.079833 | -0.000450 | 0.349096 | -11.995218 | 1 | 54 | negative_economic |
| 3 | 5.000000 | 2 | momentum_ret_3 | 7449 | -3.394978 | -0.000504 | 0.342708 | -11.090286 | 2 | 53 | negative_economic |
| 3 | 5.000000 | 3 | momentum_ret_3 | 5864 | -2.671549 | -0.000470 | 0.416752 | -10.826256 | 3 | 52 | negative_economic |
| 6 | 1.000000 | 0 | momentum_ret_3 | 3349 | -0.006886 | 0.000043 | 1.012689 | 0.129596 | 28 | 25 | negative_economic |
| 6 | 1.000000 | 1 | reversion_ret_3 | 3478 | -0.043954 | -0.000003 | 0.973684 | -0.279162 | 25 | 29 | negative_economic |
| 6 | 1.000000 | 2 | momentum_ret_3 | 3988 | 0.007976 | -0.000074 | 0.907850 | -1.108084 | 25 | 30 | negative_economic |
| 6 | 1.000000 | 3 | momentum_ret_3 | 3076 | 0.076151 | -0.000036 | 0.999971 | -0.000227 | 27 | 28 | negative_economic |

## State Role Classification

| horizon_bars | cost_bps | hmm_state | action | total_net_return | total_trades | avg_trade_net | state_role | economic_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 1.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 1.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 1.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 2.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 2.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 2.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 2.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 5.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 5.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 5.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 1 | 5.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 1.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 1.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 1.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 1.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 2.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 2.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 2.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 2.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 5.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 5.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 5.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 2 | 5.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 1.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 1.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 1.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 1.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 2.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 2.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 2.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 2.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 5.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 5.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 5.000000 | 2 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 3 | 5.000000 | 3 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 6 | 1.000000 | 0 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 6 | 1.000000 | 1 | flat | 0.000000 | 0 | 0.000000 | no_trade | not_exploitable |
| 6 | 1.000000 | 2 | momentum_ret_3 | 0.007976 | 3988 | -0.000074 | momentum_bias | not_exploitable |
| 6 | 1.000000 | 3 | momentum_ret_3 | 0.076151 | 3076 | -0.000036 | momentum_bias | not_exploitable |

## Hour Distribution Sample

| fold | split | horizon_bars | hmm_state | hour | rows | row_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | validation | 1 | 0 | 10 | 32 | 0.068230 |
| 0 | validation | 1 | 0 | 11 | 101 | 0.215352 |
| 0 | validation | 1 | 0 | 12 | 81 | 0.172708 |
| 0 | validation | 1 | 0 | 13 | 88 | 0.187633 |
| 0 | validation | 1 | 0 | 14 | 93 | 0.198294 |
| 0 | validation | 1 | 0 | 15 | 74 | 0.157783 |
| 0 | validation | 1 | 1 | 10 | 58 | 0.121849 |
| 0 | validation | 1 | 1 | 11 | 89 | 0.186975 |
| 0 | validation | 1 | 1 | 12 | 87 | 0.182773 |
| 0 | validation | 1 | 1 | 13 | 81 | 0.170168 |
| 0 | validation | 1 | 1 | 14 | 74 | 0.155462 |
| 0 | validation | 1 | 1 | 15 | 87 | 0.182773 |
| 0 | validation | 1 | 2 | 10 | 36 | 0.163636 |
| 0 | validation | 1 | 2 | 11 | 60 | 0.272727 |
| 0 | validation | 1 | 2 | 12 | 37 | 0.168182 |
| 0 | validation | 1 | 2 | 13 | 35 | 0.159091 |
| 0 | validation | 1 | 2 | 14 | 29 | 0.131818 |
| 0 | validation | 1 | 2 | 15 | 23 | 0.104545 |
| 0 | validation | 1 | 3 | 11 | 2 | 0.011173 |
| 0 | validation | 1 | 3 | 12 | 47 | 0.262570 |
| 0 | validation | 1 | 3 | 13 | 48 | 0.268156 |
| 0 | validation | 1 | 3 | 14 | 56 | 0.312849 |
| 0 | validation | 1 | 3 | 15 | 26 | 0.145251 |
| 0 | test | 1 | 0 | 10 | 43 | 0.081749 |
| 0 | test | 1 | 0 | 11 | 87 | 0.165399 |
| 0 | test | 1 | 0 | 12 | 103 | 0.195817 |
| 0 | test | 1 | 0 | 13 | 102 | 0.193916 |
| 0 | test | 1 | 0 | 14 | 97 | 0.184411 |
| 0 | test | 1 | 0 | 15 | 94 | 0.178707 |
| 0 | test | 1 | 1 | 10 | 46 | 0.114144 |

## Outputs

- `reports/hmm_state_economics/state_fold_metrics.parquet`
- `reports/hmm_state_economics/state_ranking.parquet`
- `reports/hmm_state_economics/state_roles.parquet`
- `reports/hmm_state_economics/state_hour_distribution.parquet`

## Conclusion

No HMM state/action candidate passed the validation economic filters under this diagnostic.

This report is diagnostic only. It does not optimize on test and does not accept HMM as an edge unless validation candidates survive the explicit economic filters and are later confirmed by frozen walk-forward tests.

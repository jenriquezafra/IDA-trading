# Cross-Asset Feature Definitions

Feature set: `cross_asset_v1`

## Timing Rule

Every feature row at bar `t` uses only bars closed through `t`. The target execution helper `target_open_next` is carried for labels/execution checks only and is not part of `hmm_feature_columns`.

## Per-Ticker Returns

For symbol `S` and window `w`:

```text
ret_S_w = log(close_S_t / close_S_{t-w})
```

Returns reset by session. No overnight return is used.

## Per-Ticker Range

```text
range_S = log(high_S_t / low_S_t)
range_ratio_S_6_24 = mean(range_S, last 6 bars) / mean(range_S, last 24 bars)
```

All rolling windows are same-session and include the current closed bar.

## Target Structure

For target `T`:

```text
target_ret_w = ret_T_w
target_signed_efficiency_12 = sum(ret_T_1, 12) / sum(abs(ret_T_1), 12)
target_dir_persistence_12 = mean(sign(ret_T_1), 12)
target_dist_open = log(close_T_t / session_open_T)
target_pos_session_range = (close_T_t - low_so_far_T) / (high_so_far_T - low_so_far_T)
target_dist_vwap_atr = log(close_T_t / vwap_T_t) / atr_T_12
target_intraday_runup = log(close_T_t / low_so_far_T) / atr_T_12
```

`vwap_T_t`, highs/lows so far and ATR use only current or previous bars in the same session.

## Relative Returns

```text
relret_A_B_w = ret_A_w - ret_B_w
```

Both legs use the same timestamp `t` and same window `w`.

## Spreads

```text
spread_growth_defensive_w = mean(ret_QQQ_w, ret_XLK_w, ret_XLY_w) - mean(ret_XLP_w, ret_XLV_w, ret_XLU_w)
spread_cyclicals_defensive_w = mean(ret_IWM_w, ret_XLY_w, ret_XLF_w, ret_XLE_w) - mean(ret_XLP_w, ret_XLV_w, ret_XLU_w)
spread_tech_broad_w = mean(ret_QQQ_w, ret_XLK_w) - mean(ret_SPY_w, ret_IWM_w, ret_DIA_w)
spread_equity_bonds_w = mean(ret_SPY_w, ret_QQQ_w, ret_IWM_w, ret_DIA_w) - mean(ret_TLT_w, ret_IEF_w)
spread_equity_gold_w = ret_SPY_w - ret_GLD_w
spread_credit_w = ret_HYG_w - ret_LQD_w
```

## Breadth And Leadership

```text
positive_index_count_w = count(ret_index_w > 0)
positive_sector_count_w = count(ret_sector_w > 0)
sector_above_vwap_count = count(close_sector_t > vwap_sector_t)
sector_rel_strength_count_w = count(ret_sector_w > ret_SPY_w)
leadership_concentration_score_w = max(ret_sector_w) - median(ret_sector_w)
```

## Volatility And Stress

```text
market_range_ratio_6_24 = mean(range_ratio_index_6_24)
sector_range_dispersion_12 = stdev(range_sector_t)
cross_asset_vol_expansion_score = mean(range_ratio_index_and_sector_6_24)
intraday_stress_score = mean(risk_off_score, cross_asset_vol_expansion_score)
```

## Composite Scores

Composite scores are deliberately simple linear combinations. They are not optimized on PnL.

```text
risk_on_score =
  mean(index returns 12)
  + spread_growth_defensive_12
  + spread_credit_12
  + spread_equity_bonds_12
  - market_range_ratio_6_24 * 0.001

risk_off_score =
  -mean(index returns 12)
  - spread_credit_12
  + mean(relret_TLT_SPY_12, relret_IEF_SPY_12, relret_GLD_SPY_12)
  + market_range_ratio_6_24 * 0.001

defensive_rotation_score =
  mean(defensive returns 12) - mean(growth/cyclical returns 12)
  + mean(relret_TLT_SPY_12, relret_IEF_SPY_12, relret_GLD_SPY_12)

narrow_rally_score =
  relret_QQQ_SPY_12
  + relret_XLK_SPY_12
  - relret_IWM_SPY_12
  + leadership_concentration_score_12

chop_score =
  -abs(target_signed_efficiency_12)
  -abs(target_dir_persistence_12)
  + cross_asset_signal_conflict_score
```

`cross_asset_signal_conflict_score` is the fraction of index and sector returns whose sign differs from SPY over the same window.

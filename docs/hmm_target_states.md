# HMM Target States

## Purpose

This document defines the economic states the cross-asset HMM should try to separate before looking at PnL.

The process is:

1. Define target states and expected feature signs.
2. Build cross-asset features.
3. Fit HMM only on train folds.
4. Profile states by features, occupancy, persistence, hour and fold.
5. Assign economic names before evaluating forward returns or strategy PnL.
6. Evaluate economic usefulness only after the state naming is frozen.

The HMM is allowed to discover states that do not match these hypotheses. Those states must be labeled `uninterpretable/noise` until their feature profile has a stable economic explanation.

## Naming Rules

- Do not name a state by its PnL.
- Do not inspect forward returns before assigning the economic label.
- Do not rename a state after seeing test performance.
- Use feature-state profiles, occupancy, persistence, hour distribution and fold distribution first.
- If two states have equivalent profiles, merge conceptually or reject one as redundant.
- If a state is mostly time-of-day, missing-data structure or one ticker artifact, reject it as an economic regime.

## Interpretation Metrics

For every HMM state, compute these before PnL:

- feature mean, median and z-score by state;
- state occupancy;
- average duration;
- transition matrix;
- distribution by hour;
- distribution by fold;
- distribution by calendar period;
- stability across `K` and seed;
- dependence on individual tickers;
- leave-one-ticker-out profile sensitivity.

Minimum interpretability requirement:

- at least 2-3 states must have clear economic profiles;
- no accepted state can be explained only by hour of day;
- no accepted state can depend entirely on one non-target ticker;
- state profiles must be recognizable across several folds or seeds.

## State 1 - Risk-On Trend

### Hypothesis

Broad market risk appetite is positive and coherent. SPY is supported by indices, growth/cyclical sectors and credit.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Index returns | `SPY`, `QQQ`, `IWM`, `DIA` positive over short windows |
| Breadth proxies | many indices/sectors positive |
| Growth/cyclicals vs defensives | growth/cyclicals stronger |
| Credit | `HYG` stronger than `LQD` |
| Bonds | equities stronger than `TLT`/`IEF` |
| Vol/range | contained or non-expanding |

### Candidate Features

- `ret_SPY_3`, `ret_QQQ_3`, `ret_IWM_3`, `ret_DIA_3`;
- `relret_QQQ_SPY_6`;
- `relret_IWM_SPY_6`;
- `positive_index_count_6`;
- `positive_sector_count_6`;
- `spread_growth_defensive_12`;
- `spread_cyclicals_defensive_12`;
- `relret_HYG_LQD_12`;
- `relret_SPY_TLT_12`;
- `risk_on_score`;
- `cross_asset_momentum_score`.

### Relevant Tickers

`SPY`, `QQQ`, `IWM`, `DIA`, `XLK`, `XLY`, `XLF`, `HYG`, `LQD`, `TLT`, `IEF`.

### Possible Exploitation Hypotheses

- momentum continuation in target;
- allow long-only target rules;
- avoid mean-reversion shorts unless validation proves otherwise.

These are hypotheses only. They are not accepted until tested out of sample after costs.

### False Interpretation Risks

- state is just the first hour of trading;
- state is just low-vol drift;
- state is dominated by `QQQ`/`XLK` and not broad risk appetite;
- state appears only in bull-market periods.

## State 2 - Risk-Off / Stress

### Hypothesis

Selling pressure is broad. Defensive assets, bonds or gold hold up better than equities, while ranges expand.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Index returns | `SPY`, `QQQ`, `IWM`, `DIA` negative |
| Breadth proxies | many indices/sectors negative |
| Defensives vs growth/cyclicals | defensives relatively stronger |
| Credit | `LQD` stronger than `HYG` |
| Bonds/gold | `TLT`, `IEF` or `GLD` stronger than SPY |
| Vol/range | expansion |

### Candidate Features

- `negative_index_count_6`;
- `market_ret_breadth_6`;
- `relret_defensive_growth_12`;
- `relret_TLT_SPY_12`;
- `relret_IEF_SPY_12`;
- `relret_GLD_SPY_12`;
- `relret_LQD_HYG_12`;
- `market_range_ratio_6_24`;
- `cross_asset_vol_expansion_score`;
- `risk_off_score`.

### Relevant Tickers

`SPY`, `QQQ`, `IWM`, `DIA`, `XLP`, `XLV`, `XLU`, `TLT`, `IEF`, `HYG`, `LQD`, `GLD`.

### Possible Exploitation Hypotheses

- no-trade / risk-off filter;
- short momentum only if costs and drawdown are acceptable;
- reduce size or block long mean-reversion entries.

### False Interpretation Risks

- state is just high volatility without directional information;
- state is concentrated in crisis months;
- state is defined by one bond or credit ETF gap;
- state has high headline PnL but unacceptable drawdown.

## State 3 - Defensive Rotation

### Hypothesis

The market is not necessarily crashing, but leadership rotates away from growth/cyclicals into defensives, bonds or gold.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Target/index returns | mixed or mildly weak |
| Defensives | `XLP`, `XLV`, `XLU` stronger |
| Growth/cyclicals | `XLK`, `XLY`, `XLF`, `IWM` weaker |
| Bonds/gold | stable or relatively strong |
| Vol/range | moderate; not necessarily panic |

### Candidate Features

- `relret_XLP_XLY_12`;
- `relret_XLV_XLK_12`;
- `relret_XLU_XLK_12`;
- `defensive_leadership_score`;
- `cyclical_weakness_score`;
- `relret_TLT_SPY_24`;
- `relret_IEF_SPY_24`;
- `relret_GLD_SPY_24`;
- `sector_dispersion_12`.

### Relevant Tickers

`SPY`, `QQQ`, `IWM`, `XLK`, `XLY`, `XLF`, `XLP`, `XLV`, `XLU`, `TLT`, `IEF`, `GLD`.

### Possible Exploitation Hypotheses

- avoid aggressive long continuation;
- prefer no-trade or smaller size;
- test whether target mean-reversion is better than momentum.

### False Interpretation Risks

- state is just sector dispersion without defensive logic;
- state is a weak version of risk-off and should be merged;
- state is unstable across seeds.

## State 4 - Tech-Led Narrow Rally

### Hypothesis

SPY is supported by technology/mega-cap strength while broad participation is weaker.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Tech/growth | `QQQ`, `XLK` strong |
| Broad/small-cap | `IWM`, `DIA`, cyclicals lag |
| Breadth proxies | weaker than headline SPY/QQQ |
| Credit | not necessarily strong |
| Vol/range | can be contained |

### Candidate Features

- `relret_QQQ_SPY_6`;
- `relret_XLK_SPY_6`;
- `relret_QQQ_IWM_12`;
- `relret_XLK_XLF_12`;
- `relret_XLK_XLE_12`;
- `tech_broad_spread_12`;
- `index_divergence_score`;
- `leadership_concentration_score`;
- `narrow_rally_score`.

### Relevant Tickers

`SPY`, `QQQ`, `IWM`, `DIA`, `XLK`, `XLF`, `XLE`, `XLY`.

### Possible Exploitation Hypotheses

- SPY momentum may work only when concentration is not too extreme;
- no-trade if narrow rally is unstable or reverses intraday;
- compare SPY target behavior against QQQ target behavior later.

### False Interpretation Risks

- state is just `QQQ > SPY`;
- state disappears outside mega-cap-led regimes;
- state is actually risk-on trend and should be merged.

## State 5 - Chop / Neutral

### Hypothesis

Signals are mixed. The target and context lack directional efficiency, leadership is unstable and price remains near intraday anchors.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Target efficiency | low signed efficiency and persistence |
| Target location | near VWAP/open/session midpoint |
| Cross-asset signals | conflicting signs |
| Breadth/dispersion | low or unstable |
| Vol/range | compressed or normal |

### Candidate Features

- `target_signed_efficiency_12`;
- `target_dir_persistence_12`;
- `target_dist_vwap_atr`;
- `target_dist_open`;
- `target_pos_session_range`;
- `cross_asset_signal_conflict_score`;
- `sector_dispersion_12`;
- `market_absret_compression_score`;
- `range_ratio_6_24`;
- `chop_score`.

### Relevant Tickers

Target symbol, `SPY`, `QQQ`, `IWM`, sector basket.

### Possible Exploitation Hypotheses

- no-trade filter;
- mean-reversion only if average trade survives costs;
- reject directional momentum rules unless validation is strong.

### False Interpretation Risks

- state is just lunch hour;
- state is low volume artifact;
- state is residual/noise rather than a true regime.

## State 6 - High-Volatility Expansion

### Hypothesis

Range and realized volatility expand across the target and context. This may be directional stress, breakout, forced de-risking or simply a no-trade regime.

### Expected Signs

| Feature family | Expected sign/profile |
| --- | --- |
| Target range | high relative to recent session history |
| Market range | high across indices/sectors |
| Dispersion | elevated |
| Abs returns | high across `SPY`, `QQQ`, `IWM` |
| Haven/risk proxies | may diverge strongly |

### Candidate Features

- `target_range_ratio_6_24`;
- `market_range_ratio_6_24`;
- `sector_range_dispersion_12`;
- `absret_SPY_12`;
- `absret_QQQ_12`;
- `absret_IWM_12`;
- `cross_asset_vol_expansion_score`;
- `intraday_stress_score`;
- `risk_off_score`.

### Relevant Tickers

Target symbol, `SPY`, `QQQ`, `IWM`, `DIA`, sector basket, `TLT`, `HYG`, `LQD`, `GLD`.

### Possible Exploitation Hypotheses

- risk filter / reduce size;
- no-trade during unstable expansion;
- momentum continuation only if state profile and validation support it.

### False Interpretation Risks

- state captures high volume/opening auction behavior;
- state is too rare to be useful;
- state combines opposite economic regimes and needs splitting.

## Uninterpretable / Noise State

The HMM may produce a state with no clear economic profile.

Label it `uninterpretable/noise` when:

- feature z-scores are near zero across most families;
- profile changes materially by seed or fold;
- occupancy is too small for reliable interpretation;
- state is mostly missing-data structure or time-of-day;
- no state hypothesis explains it without looking at PnL.

This state can still be useful as a no-trade filter only if economic diagnostics later prove it. It cannot be named by its returns.

## Fusion And Rejection Criteria

Fuse or treat states as redundant when:

- feature profiles have the same signs and magnitudes;
- transition behavior is similar;
- both states map to the same target hypothesis;
- separation is not stable by seed or fold.

Reject a state as an economic regime when:

- it is mostly one hour of the day;
- it depends on a single non-target ticker;
- it vanishes in leave-one-ticker-out checks;
- it exists only for one `K`, seed or fold;
- it has no stable feature profile;
- it can only be described by PnL.

## Pre-PnL Freeze Rule

Before running state economics, save the state profile and proposed labels to a report under:

```text
reports/{target_symbol}/state_hypotheses_pre_pnl.md
```

After that point, changes to state names require a documented reason based on feature-profile evidence, not returns.

# Context Universe Rationale

## Core Universe

`core_cross_asset_v1` is intentionally compact. It is meant to give the HMM enough cross-asset context to separate economic regimes without turning the feature space into an uninterpretable ticker search.

| Bucket | Symbols | Regime role |
| --- | --- | --- |
| Indices | `SPY`, `QQQ`, `IWM`, `DIA` | Broad market, growth, small-cap breadth and Dow/value confirmation. |
| Sectors | `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLU` | Leadership, defensive rotation, cyclical confirmation and sector dispersion. |
| Rates/bonds | `TLT`, `IEF`, `SHY` | Duration, rates pressure and cash-like anchor. |
| Credit | `HYG`, `LQD` | Risk appetite and credit stress via high yield vs investment grade. |
| Gold/commodities | `GLD`, `USO` | Haven and oil/energy shock context. |

## Optional Symbols

`UUP` is optional because the first Polygon 5min coverage audit produced many incomplete sessions after strict cleaning. It can be revisited with an optional/masked missing policy, but it should not reduce the core panel.

`VXX` is optional until data quality is audited. It can be useful as a tradeable volatility proxy, but product mechanics can dominate long history. `SMH` and `SOXX` are optional for semiconductor single-name targets such as `AMD` or `NVDA`.

## Target Templates

The lab should not assume `SPY` is always the target. `configs/universes/target_templates.yaml` defines initial context templates for:

- broad-market ETFs;
- large technology single names;
- semiconductor single names;
- custom target configurations.

The target-specific rule is:

> `target_symbol` is the instrument traded; `context_universe` is the information used to infer regimes.

# Operable Candidate Registry

Fecha: 2026-05-17

Este registro separa los IDs de hipotesis (`H...`) de los candidatos
operables congelados (`C...`). Una hipotesis puede generar varios candidatos,
y cualquier cambio material de regla, universo, timeframe, coste o ejecucion
debe crear un nuevo `candidate_id` o una variante decimal documentada.

## Convencion

- `H...`: hipotesis de investigacion o familia economica.
- `C...`: candidato operable congelado con artefactos reproducibles.
- Variante decimal, por ejemplo `C2.5`: variante relacionada que no reemplaza al
  candidato principal.

## Registro

| Candidate | Hipotesis | Nombre corto | Universo | Timeframe | Estado | Relacion |
|---|---|---|---|---|---|---|
| C1 | H1c | Risk-off short continuation | QQQ | 15min | paper_candidate / freeze_review | Candidato paper existente. H1c deriva de H1/H1b y usa filtro de credito interpretable. |
| C2 | H9 | Opening bias followthrough refined | GOOGL | 5min | frozen_validation_candidate | Candidato principal nuevo. Sale de H9 setup-first y queda congelado como single-name. |
| C2.5 | H9 | C2 portfolio variant | GOOGL, QQQ | 5min | saved_watchlist_candidate | Variante de C2 para cartera. Guardada porque mejora cartera mensual, pero QQQ es menos robusto en validacion alternativa. |

## C1 - H1c Risk-Off Short Continuation

Config:

- `configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml`

Evidencia:

- `results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/report.md`
- `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/manifest.yaml`
- `results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/report.md`

Resumen:

- Short QQQ tras breakdown intradia en contexto risk-off.
- `hypothesis_id`: `H1c`.
- `candidate_id`: `C1`.
- Estado actual: mantener en observabilidad paper.

## C2 - H9 Opening Bias Followthrough Refined

Config:

- `configs/setup_signal_portfolio_lifecycle_c2_googl_5min_monthly.yaml`

Artefactos:

- `reports/candidates/C2/googl_5min_24_6_6_step1/lifecycle/setup_signal_portfolio_lifecycle.md`
- `results/candidates/C2/googl_5min_24_6_6_step1/lifecycle/setup_signal_portfolio_lifecycle_summary.parquet`
- `results/candidates/C2/googl_5min_24_6_6_step1/lifecycle/setup_signal_portfolio_lifecycle_promotion.parquet`

Regla congelada:

- Target: `GOOGL`.
- Timeframe: `5min`.
- Side: `long`.
- Entrada desde `60min` despues de apertura.
- `rel_volume_by_bar >= 1.25`.
- `close_location_bar >= 0.65`.
- `0 <= dist_vwap_atr <= 0.50`.
- Salida maxima: `24` barras.
- Take-profit: `100 bps`.
- Sin stop loss en esta version.
- Costes: IBKR y sensibilidad `1/2/5 bps`, cobrados por turnover real.

Resultado principal, `24/6/6`, rolling mensual, `26` folds:

| Coste | Net | Sharpe medio | Max DD | Entradas | Folds positivos |
|---|---:|---:|---:|---:|---:|
| `ibkr_tiered_10000` | `+49.76%` | `1.65` | `1.00%` | `167` | `25/26` |
| `bps_5` | `+45.48%` | `1.53` | `1.02%` | `167` | `24/26` |
| `ibkr_fixed_5000` | `+44.28%` | `1.50` | `1.03%` | `167` | `24/26` |

Decision:

- C2 queda como candidato principal para siguiente fase de validacion/paper
  design.
- No cambiar parametros sin crear otro candidato.

## C2.5 - H9 C2 Portfolio Variant

Configs:

- `configs/setup_signal_portfolio_lifecycle_h9_refined_5min_qqq_googl_monthly.yaml`
- `configs/setup_signal_portfolio_lifecycle_h9_refined_5min_qqq_googl_18_3_3_step3.yaml`

Artefactos:

- `reports/h9_lifecycle/googl_qqq_5min_24_6_6_step1/_portfolio_lifecycle_refined/setup_signal_portfolio_lifecycle.md`
- `reports/h9_lifecycle/googl_qqq_5min_18_3_3_step3/_portfolio_lifecycle_refined/setup_signal_portfolio_lifecycle.md`

Resultado `24/6/6`, rolling mensual, `26` folds:

| Coste | Net | Sharpe medio | Max DD | Entradas | Folds positivos |
|---|---:|---:|---:|---:|---:|
| `ibkr_tiered_10000` | `+34.66%` | `1.74` | `1.31%` | `414` | `26/26` |
| `bps_5` | `+30.78%` | `1.56` | `1.35%` | `414` | `26/26` |
| `ibkr_fixed_5000` | `+21.03%` | `1.06` | `1.48%` | `414` | `25/26` |

Validacion alternativa `18/3/3`, step trimestral, `13` folds:

| Coste | Net | Sharpe medio | Max DD | Entradas | Folds positivos |
|---|---:|---:|---:|---:|---:|
| `ibkr_tiered_10000` | `+4.19%` | `1.01` | `3.20%` | `113` | `10/13` |
| `bps_5` | `+3.13%` | `0.83` | `3.23%` | `113` | `10/13` |
| `ibkr_fixed_5000` | `+0.47%` | `0.37` | `3.30%` | `113` | `8/13` |

Decision:

- C2.5 queda guardado como variante de cartera, no como reemplazo de C2.
- La razon es que GOOGL soporta mejor la validacion alternativa; QQQ queda mas
  cerca del ruido bajo stress.
- KO queda fuera de C2/C2.5 y solo se conserva como evidencia negativa en
  `C2.research_ko_check`.

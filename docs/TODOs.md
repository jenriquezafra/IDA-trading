# Active TODO - IDA rebuild

Este archivo sustituye el TODO antiguo. La direccion activa ya no es seguir
apilando scripts sobre el laboratorio HMM/SPY-only, sino reconstruir el proyecto
alrededor de contratos modulares y usar lo existente solo como base reutilizable.

Registro actual de hipotesis y prioridades: `docs/hypotheses_roadmap.md`.
Ese documento reasigna `H3` a earnings continuation intradia condicionado; la
rama historica `Options ORB` queda diferida/legacy.

## Principios

- La arquitectura activa vive directamente en `src/alpha`, `src/strategy`,
  `src/backtesting` y `src/research`.
- Los scripts legacy en `src/*.py` quedan como cantera temporal, no como arquitectura.
- Alpha research entra antes que nuevos modelos complejos.
- Toda hipotesis debe estar en YAML o en un contrato versionado, no escondida en un
  diccionario largo dentro de un script.
- Validation selecciona; test confirma; paper/live no existe sin manifest.
- Una estrategia es un contrato reproducible: datos, alpha, entrada, salida, costes,
  riesgo, splits y artefactos.

## Hipotesis activa

### H1 - Risk-off short continuation en QQQ

Hipotesis economica:

Cuando QQQ cae en un contexto risk-off confirmado, la presion vendedora suele
tener continuidad intradia porque aparecen flujos de de-risking, hedging,
stop-outs y reduccion de exposicion. Esa continuacion deberia ser mas fuerte
cuando SPY/IWM/DIA confirman, credito se debilita y defensivos/havens lideran
relativamente.

Proxies iniciales:

- `target_ret_6` y `target_ret_12` negativos.
- `risk_off_score` alto.
- `risk_on_score` bajo.
- debilidad en indices amplios y sectores ciclicos/growth.
- `spread_credit_12` debil o HYG bajo vs LQD.
- liderazgo relativo de defensivos, bonos u oro.

Prediccion:

- `fwd_ret_2`, `fwd_ret_3`, `fwd_ret_4` y/o `fwd_ret_6` negativos netos
  despues de costes.

Implementacion inicial:

- operar short delta sobre QQQ como alpha base.
- validar contra `same-hour short control`, base sin filtro, always-flat y
  costes primario/conservador/stress.

Extension con opciones:

- estudiar puts o put debit spreads de QQQ/SPY 1-4 DTE.
- separar `0DTE intraday` de `1-4 DTE overnight`; 0DTE no debe mantener
  overnight.
- comparar contra short QQQ equivalente por delta.
- exigir datos reales de options chain, bid/ask, volumen/OI, greeks e IV.
- modelar fills conservadores: comprar cerca de ask y vender cerca de bid.
- medir PnL por delta/theta/vega y no aceptar si el edge direccional no
  sobrevive en subyacente.

Invalidacion:

- no mejora controles.
- no sobrevive a coste conservador.
- PnL concentrado en pocos dias, horas o eventos.
- opciones no superan short delta equivalente despues de spreads/slippage.

### H2 - Equity ORB por pares/spreads relativos

Spec: `docs/equity_orb_hypothesis.md`.

Estado: `screened_not_promoted`.

Lineas testadas:

- `H2.2 - ORB por pares/spreads relativos`.
- `H2.4 - ORB condicionado por calidad del opening range`.
- `H2.5 - Failed ORB / reversion`.

Hipotesis economica:

Las rupturas del opening range son mas informativas cuando ocurren sobre spreads
relativos que aislan liderazgo/rotacion intradia y reducen beta de mercado. ORB
puro queda como baseline, no como edge suficiente.

Pares iniciales:

- `QQQ/SPY`
- `XLK/SPY`
- `IWM/SPY`
- `XLY/XLP`
- `HYG/LQD`

Representacion inicial:

- `spread_log = log(asset_a) - log(asset_b)`.
- sizing dollar-neutral simple.
- beta-neutral queda para una segunda version si el spread simple muestra edge.

### H3 - Options ORB

Spec: `docs/options_orb_hypothesis.md`.

Estado: `research_spec_deferred`.

Decision actual:

- no pagar datos historicos de opciones para ORB: la rama equity no fue
  promovida.
- prioridad de fuente cuando toque: `IBKR > Databento > ThetaData`.
- usar opciones solo long premium: compra de calls/puts `0-2 DTE` y cierre
  vendiendo la opcion.
- el ejercicio no forma parte de la estrategia normal.

Dependencia:

- H3 queda aparcada salvo que se abra explicitamente una rama separada de
  data-probe con IBKR para una hipotesis distinta.

## Datos y proxies H1

Datos exogenos ya disponibles:

- indices liquidos: `SPY`, `QQQ`, `IWM`, `DIA`.
- sectores: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLU`.
- rates/bonds: `TLT`, `IEF`, `SHY`.
- credito: `HYG`, `LQD`.
- haven/commodities: `GLD`, `USO`.

Proxies risk-off iniciales:

- target breakdown: `target_ret_6`, `target_ret_12`, `target_dist_vwap_atr`.
- confirmacion indices: `positive_index_count_2/6/12`, `relret_IWM_SPY`,
  `relret_DIA_SPY`.
- debilidad growth/ciclicos: `spread_growth_defensive_12`,
  `spread_cyclicals_defensive_12`, `spread_tech_broad_12`.
- credito: `relret_HYG_LQD_12`, `spread_credit_12`.
- haven/rates: `relret_TLT_SPY_12`, `relret_IEF_SPY_12`, `relret_GLD_SPY_12`.
- stress: `intraday_stress_score`, `cross_asset_vol_expansion_score`,
  `market_range_ratio_2_8`.
- scores agregados: `risk_off_score`, `risk_on_score`,
  `defensive_rotation_score`.

Datos exogenos a evaluar, no integrar todavia:

- [x] volatilidad Cboe diaria: `VIX`, `VIX9D`, `VIX3M`, `VVIX`.
- [x] put/call ratios Cboe diarios: total, index, equity, ETP, SPX/SPXW, VIX.
- VXX intradia como proxy tradeable de volatilidad, solo si pasa auditoria de
  cobertura y leakage.
- options chain QQQ/SPY 0-4 DTE: solo despues de demostrar edge en subyacente.

Datasets Cboe tratados:

- `data/external/cboe/volatility_indices_daily.parquet`
- `data/external/cboe/put_call_ratios_daily.parquet`
- `data/external/cboe/risk_context_daily.parquet`
- `reports/data_external/cboe_risk_context.md`

Regla point-in-time:

- los datos diarios de cierre Cboe se asignan a `available_session`, la siguiente
  sesion NYSE disponible.
- al hacer EDA intradia solo se puede unir por `session == available_session`;
  nunca por `source_date == session`.

Tareas inmediatas:

- [x] Posponer `configs/alpha/risk_off_short_continuation.yaml` hasta tener
  strategy runner y controles normales; no crear config de continuation/options
  antes de que el subyacente aguante triage.
- [x] Crear reporte EDA H1 con buckets de `risk_off_score`, credito,
  defensivos/havens y target breakdown.
- [x] Medir distribucion de `fwd_ret_2/3/4/6` por bucket.
- [x] Medir distribucion por hora y por franja intradia; cubierto por el triage
  H1 h=6 y artefactos `hour_summary`/franja intradia.
- [x] Comparar H1 contra short momentum sin contexto, same-hour short control,
  always-flat y random same-frequency.
- [x] Evaluar si VIX/VIX9D/VVIX diarios aportan informacion incremental sin
  introducir leakage intradia; `prev_vix_z20` mejora el candidato en EDA, pero
  aun necesita triage en strategy runner.

Resultado EDA inicial:

- reporte: `reports/eda/risk_off_short/risk_off_short_eda.md`.
- `h1_core` puro tiene edge corto muy pequeno: aprox. `0.33 bps` a h=4 y
  `0.86 bps` a h=6, con win rate cercano a 50%.
- la condicion `target_breakdown + risk_off + vix_pressure` es mas prometedora:
  aprox. `3.83 bps` de edge short medio a h=4 y win rate `52.7%`.
- diagnostico pre-strategy con costes:
  - `target_breakdown + risk_off + vix_pressure`, h=4, `2 bps`: net `0.161`,
    profit factor `1.11`, Sharpe diario `0.30`, max drawdown `0.27`.
  - el mismo candidato falla a `5 bps`, pero queda positivo a `2 bps` en h=3,
    h=4 y h=6.
  - controles `target_breakdown`, `risk_off_top30`, `same_hour_short_control`
    y `random_same_count_control` salen negativos a h=4, `2 bps`.
- conclusion: no hace falta meter mas features ahora. La senal que merece seguir
  es el filtro de presion VIX; el siguiente paso es pasar de diagnostico de barras
  a strategy runner con trades/daily/monthly y controles formales.

Resultado strategy runner inicial:

- runner: `python -m src.strategy.risk_off_short`.
- reporte: `results/strategy/risk_off_short/QQQ/15min/report.md`.
- artefactos: `trades.parquet`, `daily.parquet`, `monthly.parquet`,
  `summary.parquet` y `manifest.yaml`.
- politica actual: 5 walk-forward folds, `24m train / 6m validation / 6m test`,
  `embargo_sessions: 1`, thresholds ajustados solo en train y evaluados en
  validation/test.
- es una estrategia rule-based, no ML.
- a `2 bps`, el candidato `target_breakdown + risk_off + vix_pressure`:
  - h=2: validation `-0.0279`, test `+0.0127`.
  - h=3: validation `-0.0150`, test `+0.0201`.
  - h=4: validation `-0.0002`, test `+0.0268`.
  - h=6: validation `+0.0214` con `3/5` folds positivos, test `+0.0667`
    con `4/5` folds positivos.
- conclusion: h=6 es el unico horizonte que merece seguir. No esta listo para
  `StrategySpec` formal ni paper porque validation tiene folds perdedores y
  algunos controles son competitivos en ventanas concretas.

Siguiente triage H1:

- [x] Triage de h=6 por hora/franja, dia de semana y concentracion por sesiones.
- [x] Comparar h=6 contra `target_breakdown`, `risk_off_top30`,
  `same_hour_short_control` y `random_same_count_control` en agregado y por fold.
- [x] Revisar sensibilidad a `risk_off_score` y `prev_vix_z20` quantiles sin mirar
  test para seleccionar.
- [x] Definir promotion gates antes de congelar candidato.
- [x] Solo si h=6 pasa promotion gates, crear `StrategySpec` YAML formal;
  superado por H1b/H1c con `configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml`.
- [x] No evaluar puts/put debit spreads 1-4 DTE dentro de H1 antes de robustez;
  la rama de opciones queda reabierta como H3 independiente.

Resultado triage H1:

- runner: `python -m src.strategy.risk_off_short_triage`.
- reporte: `results/strategy/risk_off_short/QQQ/15min/triage/report.md`.
- h=6 q70/q70 supera controles agregados:
  - validation `+0.0214`, `3/5` folds positivos, `74` trades.
  - test `+0.0667`, `4/5` folds positivos, `77` trades.
- sensibilidad validation-only selecciona q80/q80:
  - validation `+0.0476`, `4/5` folds positivos, `45` trades.
  - confirmacion unica en test `+0.0756`, `5/5` folds positivos, `38` trades.
- riesgos:
  - muestra pequena.
  - concentracion por sesiones alta; top-5 de contribucion absoluta llega a
    `0.88` en un fold de validation.
  - la franja de 14:00 NY aporta mucho con solo `5` trades; medio dia es mas
    debil en validation.
  - miercoles/jueves explican buena parte del resultado.
- decision: mantener h=6 en research. No crear `StrategySpec` formal ni paper
  hasta tener gates explicitos de trades minimos, controles, concentracion y
  costes stress.

Promotion gates H1:

- artefactos:
  - `results/strategy/risk_off_short/QQQ/15min/triage/promotion_gates.parquet`
  - `results/strategy/risk_off_short/QQQ/15min/triage/promotion_decision.yaml`
  - `results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_controls.parquet`
  - `results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_concentration.parquet`
- definicion: reglas duras para pasar de research a `freeze_review`; no optimizan
  el backtest, bloquean fragilidad antes de congelar una estrategia.
- implementacion comun: `src/research/promotion.py`; cada hipotesis debe pasar
  su `candidate_label`, summary de controles y diagnostico de concentracion.
- q80/q80 pasa:
  - trades minimos: validation `45 >= 40`, test `38 >= 30`.
  - folds positivos: validation `4/5`, test `5/5`.
  - retorno neto positivo a `2 bps`.
  - retorno neto positivo a coste stress `5 bps`.
  - mejora contra el mejor control simple en validation y test.
  - avg trade neto superior a `5 bps`.
- q80/q80 falla:
  - `validation_min_sessions_per_fold`: observado `4`, minimo `8`.
  - `test_min_sessions_per_fold`: observado `4`, minimo `8`.
  - `validation_top5_abs_share`: observado `1.0`, maximo `0.7`.
  - `test_top5_abs_share`: observado `1.0`, maximo `0.7`.
- decision: `continue_research`. Proximo paso: reparar o falsar la concentracion
  con una variante que aumente sesiones por fold sin seleccionar por test. No
  crear `StrategySpec` formal ni paper hasta que la senal tenga mas anchura por
  sesiones o se reformule para reducir concentracion.

Promotion-aware sweep H1:

- runner: `python -m src.strategy.risk_off_short_promotion_sweep`.
- reporte:
  `results/strategy/risk_off_short/QQQ/15min/promotion_sweep/report.md`.
- grid: `128` variantes de thresholds `risk_off_score`, `prev_vix_z20` y
  politica horaria, seleccionadas solo con validation.
- variantes que pasan todos los gates de validation: `0`.
- variante seleccionada por validation: `riskq80__vixq50__all`.
- resultado de la variante seleccionada:
  - validation: `100` trades, net `+0.0635`, coste stress `+0.0335`,
    avg trade `6.35 bps`, `4/5` folds positivos, minimo `10` sesiones/fold.
  - test: `98` trades, net `+0.0836`, coste stress `+0.0542`,
    avg trade `8.53 bps`, `4/5` folds positivos, minimo `11` sesiones/fold.
- mejora respecto a q80/q80: ya no falla por minimo de sesiones por fold.
- bloqueo actual:
  - `validation_top5_abs_share`: observado `0.8846`, maximo `0.7`.
  - `test_top5_abs_share`: observado `0.8802`, maximo `0.7`.
- lectura: el edge existe y bate controles, pero aun depende demasiado de las
  mejores sesiones dentro de algunos folds. No hay suficiente anchura para
  congelarlo como estrategia.
- decision: `continue_research`.

Siguiente research H1:

- [x] Formular H1b para reparar concentracion sin tocar test: buscar un filtro
  economico que mantenga avg trade `> 5 bps` y baje `top5_abs_share <= 0.7`.
- [x] Probar variantes de anchura/dispersion con selection validation-only:
  por ejemplo filtro de volatilidad extrema maxima, confirmacion de credito o
  defensivos, o reglas de salida alternativas sobre h=6.
- [x] H1b pasa gates; no aparcar H1. Pasar a `freeze_review`.

Resultado H1b concentration repair:

- runner: `python -m src.strategy.risk_off_short_h1b_sweep`.
- reporte:
  `results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/report.md`.
- grid: `840` variantes de thresholds y filtros economicos, seleccionadas solo
  con validation.
- filtros evaluados: credito debil, risk-on bajo, rotacion defensiva, breadth
  debil, VIX extremo capado, VIX9D/VIX y posicion bajo VWAP.
- variantes que pasan todos los gates de validation: `26`.
- variantes que reparan concentracion en validation: `366`.
- variante seleccionada por validation: `riskq55__vixq45__credit_weak_q50`.
- lectura economica: `credit_weak_q50` exige `spread_credit_12` por debajo de
  su mediana de train, es decir HYG debil frente a LQD. Esto confirma de-risking
  de credito y hace que el short de QQQ no dependa tanto de unos pocos dias de
  volatilidad extrema.
- resultado de la variante seleccionada:
  - validation: `142` trades, net `+0.0723`, coste stress `+0.0297`,
    avg trade `5.09 bps`, `4/5` folds positivos, minimo `20` sesiones/fold,
    max top-5 abs share `0.6087`.
  - test: `135` trades, net `+0.0708`, coste stress `+0.0303`,
    avg trade `5.25 bps`, `4/5` folds positivos, minimo `19` sesiones/fold,
    max top-5 abs share `0.6694`.
- promotion gates finales: todos pasan.
- decision: `freeze_review`.

Siguiente paso H1:

- [x] Crear `StrategySpec` YAML formal para
  `riskq55__vixq45__credit_weak_q50`.
- [x] Congelar los artefactos usados por la decision: features fingerprint,
  risk context fingerprint, split policy, thresholds, filtro de credito,
  costes y gates.
- [x] Ejecutar una revision de robustez pre-paper: sensibilidad alrededor de
  q55/q45/credit q50, costes mas altos, subperiodos y estabilidad por folds.
- [x] No evaluar puts/put debit spreads 1-4 DTE como extension directa de H1b;
  opciones pasan a H3 y dependen primero de H2.2 o de data-probe explicita.

Freeze review H1b:

- spec formal:
  `configs/strategy/qqq_15min_risk_off_short_h1b_v1.yaml`.
- freezer: `python -m src.strategy.freeze_risk_off_short_h1b`.
- artefacto canonico:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/manifest.yaml`.
- snapshot del spec:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/strategy_spec.yaml`.
- thresholds entrenados por fold:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/fold_thresholds.parquet`.
- decision congelada:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/freeze_review_decision.yaml`.
- status: `freeze_review`.
- regla congelada:
  - `target_ret_6 < 0`.
  - `target_ret_12 < 0`.
  - `risk_off_score >= q55(train)`.
  - `prev_vix_z20 >= q45(train)`.
  - `spread_credit_12 <= q50(train)`.
- siguiente bloqueo antes de paper: robustez pre-paper. No evaluar opciones ni
  pasar a paper hasta revisar sensibilidad local, costes mas altos, subperiodos
  y estabilidad por fold.

Robustez pre-paper H1b:

- runner: `python -m src.strategy.risk_off_short_h1b_robustness`.
- reporte:
  `results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/report.md`.
- artefactos:
  - `local_threshold_sweep.parquet`
  - `local_threshold_gates.parquet`
  - `cost_sensitivity.parquet`
  - `subperiod_summary.parquet`
  - `fold_stability.parquet`
  - `robustness_decision.yaml`
- decision: `needs_more_research`.
- resultado:
  - ancla `riskq55__vixq45__credit_weak_q50`: sigue en `freeze_review`.
  - sweep local: `6/27` variantes pasan todos los gates.
  - soporte local por dimensiones: risk quantiles `3`, VIX quantiles `2`,
    credit quantiles `1`.
  - bloqueo: los passes solo sobreviven con `credit_q50`; mover credito a
    `q45` o `q55` degrada el edge por trade.
  - coste stress `5 bps`: positivo en validation y test.
  - extra stress `7.5 bps` y `10 bps`: negativo en validation y test.
  - folds: concentracion ya no falla gates, pero validation fold 4 y test fold
    3 son negativos.
- lectura: H1b no esta rechazada; la hipotesis economica y la variante ancla
  siguen teniendo edge. Pero no debe pasar a paper hasta resolver o justificar
  la dependencia exacta de `credit_q50` y definir un coste maximo realista.

Siguiente research H1b:

- [x] Reparar/justificar fragilidad del filtro de credito: probar filtro de
  credito con banda o score continuo en vez de corte exacto `q50`.
- [x] Revisar si el filtro `credit_weak_q50` debe ser una mediana economica fija
  o si necesita otra proxy de credito mas robusta.
- [x] Definir coste maximo admisible para paper; H1c sobrevive `7.5 bps`, pero
  no `10 bps`.
- [x] No evaluar opciones 1-4 DTE dentro de H1b; H1c reemplaza esta rama y la
  investigacion de opciones queda separada en H3.

Resultado H1c credit repair:

- runner: `python -m src.strategy.risk_off_short_h1c_credit_repair`.
- reporte:
  `results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/report.md`.
- grid: `72` variantes de risk/VIX local y reglas de credito.
- variantes que pasan validation gates: `20`.
- variantes interpretables que pasan validation gates: `14`.
- politicas interpretables que pasan:
  - `credit_spread_lte_0`
  - `relret_hyg_lqd_lte_0`
  - `credit_spread_lte_0_and_defensive_high_q50`
- variante seleccionada por validation:
  `riskq50__vixq45__credit_spread_lte_0`.
- lectura economica: reemplaza el corte exacto `credit_q50` por
  `spread_credit_12 <= 0`, es decir HYG no lidera a LQD. Es mas interpretable y
  no depende de una mediana entrenada exacta.
- resultado seleccionado:
  - validation: `139` trades, net `+0.0790`, coste stress `+0.0373`,
    avg trade `5.68 bps`, `4/5` folds positivos, max top-5 share `0.6694`.
  - test: `133` trades, net `+0.0760`, coste stress `+0.0361`,
    avg trade `5.71 bps`, `4/5` folds positivos, max top-5 share `0.6694`.
  - `7.5 bps`: positivo en validation `+0.0026` y test `+0.0029`.
  - `10 bps`: negativo en validation `-0.0322` y test `-0.0304`.
- promotion gates finales: todos pasan.
- repair decision: `credit_repaired`.
- warning: no positivo a `10 bps`.

Siguiente paso H1c:

- [x] Formalizar `StrategySpec` H1c con regla `spread_credit_12 <= 0`.
- [x] Congelar manifest H1c con fingerprints y thresholds por fold.
- [x] Ejecutar robustez pre-paper sobre H1c; si queda `paper_candidate`, entonces
  preparar paper runner.
- [x] Mantener opciones 1-4 DTE fuera hasta que H1c supere robustez pre-paper;
  H1c ya es `paper_candidate` y opciones siguen fuera como H3 separada.

Freeze review H1c:

- spec formal:
  `configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml`.
- freezer: `python -m src.strategy.freeze_risk_off_short_h1c`.
- artefacto canonico:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/manifest.yaml`.
- thresholds entrenados por fold:
  `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/fold_thresholds.parquet`.
- regla congelada:
  - `target_ret_6 < 0`.
  - `target_ret_12 < 0`.
  - `risk_off_score >= q50(train)`.
  - `prev_vix_z20 >= q45(train)`.
  - `spread_credit_12 <= 0`.
- status: `freeze_review`.

Robustez pre-paper H1c:

- runner: `python -m src.strategy.risk_off_short_h1c_robustness`.
- reporte:
  `results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/report.md`.
- decision: `paper_candidate`.
- resultado:
  - ancla `riskq50__vixq45__credit_spread_lte_0`: pasa todos los gates.
  - sweep local: `6/9` variantes pasan todos los gates.
  - soporte local: risk quantiles `3`, VIX quantiles `2`.
  - coste `5 bps`: positivo en validation y test.
  - coste `7.5 bps`: positivo en validation `+0.0026` y test `+0.0029`, pero
    margen muy fino.
  - coste `10 bps`: negativo en validation y test; queda como warning.
  - warning activo: `extra_stress_10bps_not_positive_all_splits`.
- lectura: H1c puede pasar a paper candidate si el paper runner usa supuestos de
  coste realistas cercanos a `2-5 bps` y monitoriza deterioro; `10 bps` invalida
  la operativa del subyacente.

Siguiente paso H1c paper:

- [x] Implementar conexion IBKR Gateway paper read-only.
- [x] Implementar snapshot + plan de liquidacion paper sin ejecutar ordenes por
  defecto.
- [x] Implementar executor paper bloqueado por defecto, con doble confirmacion
  explicita antes de transmitir ordenes.
- [x] Implementar paper runner signal-only que lea `StrategySpec` H1c y
  thresholds congelados sin enviar ordenes.
- [x] Conectar paper runner H1c a datos intradia actualizados via pipeline de
  refresh; ultimo refresh llega hasta `2026-05-08`.
- [x] Implementar paper state store local para estado teorico signal-only.
- [x] Conectar paper runner H1c a ejecucion paper real solo despues de validar
  senales, estado, costes, fills y kill switches.
- [ ] Completar observabilidad paper H1c: senales y thresholds ya quedan en
  manifests; falta cerrar fills simulados/costes/PnL ex-post contra expectativa
  backtest en un reporte agregado.
- [x] Definir limite de coste/slippage: pausar nuevas entradas si el ultimo
  slippage registrado cruza el umbral configurado.
- [x] Abrir rama experimental documental de opciones como H3, sin implementacion
  ni compra de datos historicos todavia.

IBKR Gateway paper read-only:

- config: `configs/execution/ibkr_paper_readonly.yaml`.
- modulo: `src/execution/ibkr_read_only.py`.
- validar sin conectar:
  `python -m src.execution.ibkr_read_only --validate-only`.
- conectar health-check:
  `python -m src.execution.ibkr_read_only`.
- snapshot:
  `python -m src.execution.ibkr_read_only --snapshot`.
- outputs por snapshot:
  `results/paper/ibkr_read_only/<timestamp>/account_summary.parquet`,
  `positions.parquet`, `open_trades.parquet`, `manifest.yaml`, `report.md`.
- guardrails:
  - solo `trading_mode: paper`.
  - solo puertos paper `4002` IB Gateway o `7497` TWS.
  - `read_only: true`.
  - `allow_orders: false`.
  - el cliente no expone metodos de ordenes.
- cuenta paper confirmada: `DU9782002`.
- primer snapshot real:
  `results/paper/ibkr_read_only/20260510T132956Z/report.md`.
- resultado snapshot: `20` posiciones, `0` ordenes abiertas.

IBKR flatten plan offline:

- modulo: `src/execution/flatten_plan.py`.
- execution policy para apertura:
  `configs/execution/flatten_policy_mkt_opg.yaml`.
- comando:
  `python -m src.execution.flatten_plan`.
- entrada: ultimo snapshot read-only por defecto, o `--snapshot-dir`.
- outputs por plan:
  `results/paper/flatten_plan/<timestamp>/orders.parquet`, `orders.csv`,
  `skipped_positions.parquet`, `manifest.yaml`, `report.md`.
- primer plan real:
  `results/paper/flatten_plan/20260510T133632Z/report.md`.
- resultado plan real: `ready_for_review`, `20` tickets `SELL`, `0` posiciones
  omitidas, `dry_run=true`, `transmit=false`.
- plan MKT-on-open real para lunes 2026-05-11:
  `results/paper/flatten_plan/20260510T150135Z/report.md`.
- resultado plan MKT-on-open: `ready_for_review`, `20` tickets `SELL`,
  `order_type=MKT`, `tif=OPG`, `outside_rth=false`, fingerprint
  `81dc6326a7fa98f1`.
- guardrails:
  - no conecta a IBKR.
  - no envia ordenes.
  - todos los tickets salen con `dry_run=true` y `transmit=false`.
  - bloquea el plan si el snapshot tiene ordenes abiertas.
  - por ahora solo planifica cierre de `STK`; otros instrumentos quedan
    bloqueados como `unsupported_sec_type`.

IBKR flatten executor paper:

- config: `configs/execution/ibkr_paper_executor.yaml`.
- modulo: `src/execution/flatten_executor.py`.
- validacion offline del plan:
  `python -m src.execution.flatten_executor --plan-dir results/paper/flatten_plan/<timestamp>`.
- preflight con IBKR sin enviar ordenes:
  `python -m src.execution.flatten_executor --plan-dir results/paper/flatten_plan/<timestamp> --connect-preflight`.
- por defecto no ejecuta:
  - `execution_enabled: false`.
  - `allow_orders: false`.
  - sin `--execute` no conecta ni envia ordenes.
- para ejecutar en paper debe cumplirse todo a la vez:
  - config con `execution_enabled: true` y `allow_orders: true`.
  - `--execute`.
  - `--transmit-orders`.
  - `--confirm-account <DU...>`.
  - `--confirm-fingerprint <fingerprint_del_plan>`.
  - variable de entorno `IBKR_PAPER_EXECUTION_CONFIRM=<DU...>`.
  - cuenta esperada presente en IBKR.
  - cero ordenes abiertas en IBKR.
  - mercado NYSE en regular trading hours si `require_market_open: true`.
- primer preflight conectado:
  `results/paper/flatten_execution/20260510T141732Z/report.md`.
- resultado preflight: plan valido, cuenta `DU9782002`, `0` ordenes abiertas,
  `0` ordenes enviadas, bloqueado correctamente porque NYSE RTH no esta abierto.
- preflight conectado final del plan MKT-on-open:
  `results/paper/flatten_execution/20260510T150401Z/report.md`.
- resultado preflight MKT-on-open: cuenta `DU9782002`, `0` ordenes abiertas,
  `0` ordenes enviadas, `live_valid=true`, `all_orders_opg=true`,
  `opg_outside_rth_submission_allowed=true`; sigue bloqueado por defecto porque
  `execution_enabled=false`, `allow_orders=false` y no se pidio `--execute`.

H1c paper signal runner:

- config: `configs/execution/paper_runner_h1c_signal_only.yaml`.
- modulo: `src/execution/paper_h1c_signal_runner.py`.
- comando:
  `python -m src.execution.paper_h1c_signal_runner`.
- modo actual: `signal_only`, no conecta a IBKR y no envia ordenes.
- inputs:
  - `configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml`.
  - `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/fold_thresholds.parquet`.
  - features QQQ 15min y risk context Cboe point-in-time.
- politica operativa provisional: `latest_frozen_fold`, sin refit.
- outputs por run:
  `results/paper/h1c_signal_runner/<timestamp>/signals.parquet`,
  `latest_signal.yaml`, `paper_ticket.yaml`, `manifest.yaml`, `report.md`.
- primer run real:
  `results/paper/h1c_signal_runner/20260510T154956Z/report.md`.
- resultado primer run: `signal_short=false`, `action=NONE`, `send_orders=false`.
  Warning esperado: los datos de features llegan hasta `2026-05-01`, por lo que
  hay que refrescar datos intradia antes de tomar decisiones paper reales.
- guardrail activo: `paper.send_orders=false`; si se cambia a `true`, el runner
  falla porque esta version solo permite senales/tickets teoricos.

Paper data refresh:

- config: `configs/execution/paper_data_refresh.yaml`.
- modulo: `src/execution/paper_data_refresh.py`.
- dry-run:
  `python -m src.execution.paper_data_refresh --dry-run`.
- refresh real sin Cboe:
  `python -m src.execution.paper_data_refresh --skip-cboe`.
- funcion:
  - resuelve universo QQQ cross-asset desde config.
  - descarga ventana reciente Polygon con lookback.
  - mergea raw por timestamp sin duplicar.
  - limpia OHLCV.
  - alinea panel cross-asset.
  - reconstruye features `cross_asset_liquid_15min`.
  - conserva sesiones incompletas para paper si `keep_incomplete_sessions_for_paper=true`.
- primer refresh real:
  `results/paper/data_refresh/20260510T163242Z/report.md`.
- resultado: `18` simbolos actualizados, features regeneradas en
  `data/features/QQQ/15min/core_cross_asset_v1/cross_asset_liquid_15min/features.parquet`,
  ultima barra limpia `2026-05-08 15:45 ET`.
- run H1c posterior al refresh:
  `results/paper/h1c_signal_runner/20260510T163316Z/report.md`.
- resultado H1c post-refresh: `signal_short=false`, `action=NONE`,
  `send_orders=false`, sin warning de staleness.

H1c paper state store:

- config: `configs/execution/paper_state_h1c.yaml`.
- modulo: `src/execution/paper_state_store.py`.
- comando:
  `python -m src.execution.paper_state_store --ticket results/paper/h1c_signal_runner/<timestamp>/paper_ticket.yaml`.
- estado persistente:
  `results/paper/h1c_state/state.yaml`.
- eventos:
  `results/paper/h1c_state/events.parquet`.
- estados soportados:
  - `flat`
  - `pending_entry`
  - `open`
  - `pending_exit`
- version actual:
  - acepta solo tickets `signal_only` con `send_orders=false`.
  - `SELL` desde `flat` crea `pending_entry`.
  - `NONE` desde `flat` mantiene `flat`.
  - no marca fills ni envia ordenes; eso va en el siguiente bloque.
- primer estado real:
  `results/paper/h1c_state/state.yaml`.
- primer evento real:
  `results/paper/h1c_state/runs/20260510T163918Z/report.md`.
- resultado: `flat_no_signal`, estado `flat`, posicion esperada `0.0`.

H1c paper cycle runner:

- config: `configs/execution/paper_cycle_h1c.yaml`.
- modulo: `src/execution/paper_cycle_runner.py`.
- comando normal:
  `python -m src.execution.paper_cycle_runner`.
- comando sin descarga externa, util para repetir con raw ya actualizado:
  `python -m src.execution.paper_cycle_runner --skip-download --skip-cboe`.
- funcion:
  - ejecuta refresh de datos/features.
  - ejecuta paper signal runner H1c.
  - aplica el ticket al state store.
  - crea un manifest/reporte unico que enlaza los sub-runs.
- primer ciclo real:
  `results/paper/h1c_cycle/20260510T170523Z/report.md`.
- resultado primer ciclo: `action=NONE`, evento `flat_no_signal`, estado `flat`,
  sin warnings.
- guardrail activo: sigue siendo `signal_only`; reconciliation puede conectar a
  IBKR en modo read-only, pero el ciclo no envia ordenes.
- ciclo con reconciliation integrada:
  `results/paper/h1c_cycle/20260510T210859Z/report.md`.
- resultado: `action=NONE`, state `flat`, reconciliation
  `ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS`.

H1c paper reconciliation:

- config: `configs/execution/paper_reconcile_h1c.yaml`.
- modulo: `src/execution/paper_reconcile_h1c.py`.
- comando:
  `python -m src.execution.paper_reconcile_h1c`.
- comando usando snapshot existente:
  `python -m src.execution.paper_reconcile_h1c --snapshot-dir results/paper/ibkr_read_only/<timestamp>`.
- version actual:
  - read-only; no envia, cancela ni modifica ordenes.
  - carga `results/paper/h1c_state/state.yaml`.
  - toma snapshot IBKR read-only y fuerza `reqAllOpenOrders()` si esta disponible.
  - compara estado esperado contra posiciones y ordenes abiertas de IBKR.
  - bloquea si hay ordenes abiertas ajenas al simbolo objetivo salvo override.
- decisiones actuales:
  - `OK_FLAT`
  - `OK_PENDING_ENTRY`
  - `OK_OPEN`
  - `OK_PENDING_EXIT`
  - `ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS`
  - `OPEN_ORDER_WITHOUT_PENDING_TICKET`
  - `PENDING_TICKET_WITHOUT_OPEN_ORDER`
  - `FILL_DETECTED_PENDING_ENTRY`
  - `FILL_DETECTED_PENDING_EXIT`
  - `PENDING_EXIT_WITHOUT_OPEN_ORDER`
  - `DRIFT_POSITION_MISMATCH`
  - `UNKNOWN_IBKR_STATE`
- primer run real:
  `results/paper/h1c_reconciliation/20260510T174831Z/report.md`.
- resultado primer run: `ACCOUNT_NOT_CLEAN_UNRELATED_OPEN_ORDERS`, severity
  `block`; IBKR tiene `20` posiciones y `20` ordenes abiertas de flatten
  pendientes, ninguna de QQQ/H1c. La estrategia debe seguir pausada hasta que
  la cuenta paper quede limpia tras la apertura.

H1c fill accounting / PnL ex-post:

- config: `configs/execution/paper_accounting_h1c.yaml`.
- modulo: `src/execution/paper_accounting_h1c.py`.
- comando:
  `python -m src.execution.paper_accounting_h1c --reconciliation-manifest results/paper/h1c_reconciliation/<timestamp>/manifest.yaml`.
- version actual:
  - actualiza estado local solo si reconciliation detecta fill de entrada.
  - `pending_entry` + `FILL_DETECTED_PENDING_ENTRY` -> `open`.
  - registra evento de entrada en `results/paper/h1c_state/pnl_events.parquet`.
  - calcula slippage de entrada si existe precio teorico y precio de fill/avg
    cost.
  - marca salidas `pending_exit` como `flat` cuando reconciliation detecta fill
    de salida y registra PnL cerrado.

H1c order planner:

- config: `configs/execution/h1c_order_plan.yaml`.
- modulo: `src/execution/h1c_order_plan.py`.
- comando:
  `python -m src.execution.h1c_order_plan --ticket <paper_ticket.yaml> --reconciliation-manifest <reconciliation_manifest.yaml>`.
- version actual:
  - offline/reviewable.
  - crea maximo una orden `SELL QQQ` para entrada o `BUY QQQ` para salida.
  - exige reconciliation `OK_FLAT` para entrada y `OK_OPEN` para salida.
  - si hay `NONE` o reconciliation bloqueada, no crea orden.
  - todas las filas salen con `dry_run=true` y `transmit=false`.
- primer plan real desde ciclo integrado:
  `results/paper/h1c_order_plan/20260510T210908Z/report.md`.
- resultado: `no_order_no_signal`, `planned_orders=0`.

H1c order executor:

- config: `configs/execution/h1c_order_executor.yaml`.
- modulo: `src/execution/h1c_order_executor.py`.
- comando dry-run:
  `python -m src.execution.h1c_order_executor --plan-dir results/paper/h1c_order_plan/<timestamp>`.
- desbloqueo real, solo cuando se decida conscientemente:
  `IBKR_H1C_EXECUTION_CONFIRM=DU9782002 python -m src.execution.h1c_order_executor --plan-dir <plan_dir> --execute --transmit-orders --confirm-account DU9782002 --confirm-fingerprint <fingerprint>`.
- guardrails por defecto:
  - `execution_enabled=false`
  - `allow_orders=false`
  - `max_orders=1`
  - `require_no_open_trades=true`
  - `require_market_open=true`
  - `require_fingerprint_confirmation=true`

H1c auto runner:

- config: `configs/execution/h1c_auto_runner.yaml`.
- modulo: `src/execution/h1c_auto_runner.py`.
- executor auto paper: `configs/execution/h1c_order_executor_auto_paper.yaml`.
- comando de una pasada:
  `python -m src.execution.h1c_auto_runner --skip-cboe`.
- wrapper local:
  `scripts/run_h1c_auto_once.sh`.
- launchd template:
  `ops/launchd/com.ida-trading.h1c-auto.plist`.
- daemon adaptativo:
  `src/execution/h1c_auto_daemon.py`.
- daemon config:
  `configs/execution/h1c_auto_daemon.yaml`.
- launchd instalado en:
  `/Users/jenriquezafra/Library/LaunchAgents/com.ida-trading.h1c-auto.plist`.
- scheduler:
  - si NYSE esta abierto: escanea cada `900` segundos.
  - si esta fuera de mercado: duerme hasta `15` minutos antes de la proxima
    apertura NYSE.
  - en la ventana pre-open: revisa cada `900` segundos.
  - si hay posicion/orden target viva y reconciliation esta en estado OK, baja
    temporalmente a `active_reconciliation_interval_seconds`.
  - si hay actividad pero reconciliation esta bloqueada, conserva el intervalo
    normal para evitar loops agresivos sobre una cuenta no limpia.
- wrapper daemon usa `caffeinate -dimsu` para evitar sleep por inactividad
  mientras el daemon este activo. Esto no garantiza ejecucion con tapa cerrada
  si macOS fuerza sleep por lid-close.
- flujo:
  - reconciliation read-only para detectar mercado abierto y cuenta limpia.
  - si NYSE esta cerrado, sale en `market_closed`.
  - si reconciliation detecta fill de entrada/salida, corre accounting antes de
    buscar nueva senal.
  - si esta `OK_OPEN` y la salida fixed-horizon ya vencio, planifica/ejecuta
    `BUY` de cobertura.
  - si esta `OK_FLAT`, refresca datos, calcula senal H1c y revisa fondos.
  - solo ejecuta entrada si pre-trade reconciliation es `OK_FLAT`; la salida
    exige `OK_OPEN`.
  - no duplica ordenes: `OK_PENDING_ENTRY`, orden target abierta o posicion target
    bloquean nueva entrada.
  - sizing paper: `sizing_mode=buying_power_fraction`,
    `capital_fraction=1.0`, `reserve_cash_usd=0`.
  - calcula cantidad como `floor(BuyingPower * capital_fraction / precio)`.
  - mantiene una orden maxima y min `AvailableFunds`/`BuyingPower` `1000`.
  - pausa nuevas entradas si `auto.enabled=false`, si existe
    `ops/kill_switches/h1c_auto_paused`, si se alcanza
    `max_daily_entry_orders` o si el PnL realizado diario cruza
    `max_daily_realized_loss_usd`.
  - pausa nuevas entradas si el ultimo slippage de entrada/salida cruza
    `max_entry_slippage_bps` o `max_exit_slippage_bps`.
  - esos bloqueos no impiden reconciliation/accounting ni salidas `BUY` de
    cobertura.
  - cada manifest incluye `latency`, `drift` y `entry_safety`; el daemon status
    incluye latencia de iteracion y errores/streak.
  - usa lock `results/paper/h1c_auto_runner/auto.lock` para evitar pasadas
    concurrentes.
- primer run real:
  `results/paper/h1c_auto_runner/20260510T211722Z/report.md`.
- resultado: `market_closed`; solo hizo reconciliation read-only y no intento
  refrescar ni enviar ordenes.
- instalado y arrancado con launchd; verificacion:
  `launchctl print gui/501/com.ida-trading.h1c-auto`.
- estado daemon:
  `results/paper/h1c_auto_runner/daemon_status.yaml`.
- recarga adaptativa:
  `2026-05-10T21:28:31Z`, next NYSE open `2026-05-11T13:30:00Z`,
  pre-open `2026-05-11T13:15:00Z`, sleep `56788` segundos.
- run desde launchd:
  `results/paper/h1c_auto_runner/20260510T211908Z/report.md`.

## H2/H3 - ORB research

Objetivo inmediato: validar si ORB tiene edge relativo en ETFs liquidos antes de
llevar la idea a opciones.

Prioridad:

1. H2.2 equity ORB por pares/spreads relativos.
2. H2.4 calidad del opening range, solo como diagnostico final antes de aparcar
   ORB equity.
3. H3 options ORB, solo despues de validar subyacente/spreads o despues de una
   data-probe explicita con IBKR para una hipotesis distinta.

Specs:

- `docs/equity_orb_hypothesis.md`.
- `docs/options_orb_hypothesis.md`.

Tareas H2.2:

- [x] Escribir spec de hipotesis equity ORB.
- [x] Definir subhipotesis H2.1-H2.5.
- [x] Marcar H2.2 como primera linea activa.
- [x] Convertir H2.2 en contrato YAML versionado:
  `configs/strategy/equity_orb_pairs_v1.yaml`.
- [x] Validar cobertura intradia de los simbolos requeridos por H2.2:
  `SPY`, `QQQ`, `IWM`, `XLK`, `XLY`, `XLP`, `HYG`, `LQD`.
- [x] Construir spreads log iniciales: `QQQ/SPY`, `XLK/SPY`, `IWM/SPY`,
  `XLY/XLP`, `HYG/LQD`.
- [x] Calcular opening range de 15 y 30 minutos para cada spread.
- [x] Generar eventos ORB relativos por spread y lado.
- [x] Generar baseline ORB direccional simple para cada leg.
- [x] Implementar backtest dollar-neutral con costes por ambas patas.
- [x] Probar horizontes `2/3/4/6` barras y force-flat intradia.
- [x] Reportar resultados por par, lado, ventana, horizonte y fold.
- [x] Comparar contra ORB direccional simple, random same-frequency,
  same-hour y beta control.
- [x] Generar artefactos:
  `manifest.yaml`, `trades.parquet`, `daily.parquet`, `monthly.parquet`,
  `summary.parquet`, `report.md`.
- [x] Decision inicial H2.2 continuation ORB: rechazado a `2 bps` por leg
  round-trip; todos los pares/ventanas/horizontes salen negativos en validation
  y test. Reporte:
  `results/strategy/equity_orb_pairs/5min/report.md`.

Tareas H2.5:

- [x] Convertir failed ORB/reversion en contrato YAML versionado:
  `configs/strategy/equity_orb_failed_pairs_v1.yaml`.
- [x] Implementar runner H2.5:
  `python -m src.strategy.equity_orb_failed_pairs --config configs/strategy/equity_orb_failed_pairs_v1.yaml`.
- [x] Generar eventos de ruptura fallida: ruptura fuera del opening range y
  cierre posterior de vuelta dentro del rango.
- [x] Simular reversion dollar-neutral del spread con entrada en siguiente open.
- [x] Comparar contra continuation reference, random same-frequency, same-hour,
  market beta y failed ORB direccional por leg.
- [x] Generar artefactos:
  `manifest.yaml`, `events.parquet`, `trades.parquet`, `daily.parquet`,
  `monthly.parquet`, `summary.parquet`, `report.md`.
- [x] Decision inicial H2.5 failed ORB: rechazado a `2 bps` por leg round-trip;
  todos los pares/ventanas/horizontes son negativos en validation y test.
  Reporte: `results/strategy/equity_orb_failed_pairs/5min/report.md`.

Tareas H2.4:

- [x] Convertir calidad del opening range en contrato YAML versionado:
  `configs/strategy/equity_orb_range_quality_v1.yaml`.
- [x] Implementar runner H2.4:
  `python -m src.strategy.equity_orb_range_quality --config configs/strategy/equity_orb_range_quality_v1.yaml`.
- [x] Definir filtros pre-registrados por percentil de ancho del opening range:
  `20-80`, `30-70`, `0-20`, `80-100`.
- [x] Ajustar umbrales solo con train por fold/par/ventana y aplicar sin cambios
  a validation/test.
- [x] Comparar contra continuation sin filtrar, random same-frequency, same-hour
  y market beta.
- [x] Generar artefactos:
  `manifest.yaml`, `events.parquet`, `range_quality_thresholds.parquet`,
  `trades.parquet`, `daily.parquet`, `monthly.parquet`, `summary.parquet`,
  `report.md`.
- [x] Decision inicial H2.4 range-quality: no promovido a `2 bps` por leg
  round-trip. Hay un bolsillo positivo en `XLY/XLP`, `orb_15m`, rango ancho
  `80-100`, pero pierde contra market-beta y tiene concentracion excesiva.
  Reporte: `results/strategy/equity_orb_range_quality/5min/report.md`.

Decision ORB equity:

- [x] Aparcar ORB equity en esta familia: H2.2, H2.5 y H2.4 no justifican
  promotion.
- [x] No pasar a H2.1/H2.3 porque serian filtros sobre una base negativa.
- [x] No conectar con Options ORB desde estos resultados.

Tareas H3:

- [x] Escribir spec de hipotesis options ORB.
- [x] Dejar fuente de datos diferida con prioridad `IBKR > Databento > ThetaData`.
- [x] No comprar datos historicos de opciones para ORB tras la no-promocion de
  H2.2/H2.4/H2.5, salvo data-probe separada.
- [ ] Probar IBKR para chains actuales de `SPY/QQQ`.
- [ ] Probar IBKR para historico bid/ask de una opcion activa y una expirada.
- [ ] Estimar coste Databento para muestra acotada si IBKR no sirve para
  historico expirado.

Validacion estilo libro:

- [x] Datos point-in-time para Cboe via `available_session`.
- [x] Walk-forward cronologico train/validation/test.
- [x] Validation selecciona; test confirma.
- [x] Embargo conservador de `1` sesion entre train y validation.
- [x] Costes y controles contra alternativas simples.
- [ ] Purged CV/CPCV formal si pasamos a ML, labels que crucen sesiones o grid
  de modelos amplio.

## Fase 0 - Base nueva

- [x] Crear paquetes modulares directos bajo `src/`.
- [x] Crear contrato declarativo de alpha research.
- [x] Crear contrato de estrategia operable.
- [x] Crear contrato de manifest reproducible.
- [x] Crear metricas minimas de backtesting para research.
- [x] Crear config inicial `configs/alpha/alpha_research_v1.yaml`.
- [x] Crear CLI unico `python -m src alpha-research ...`.
- [ ] Decidir que scripts legacy se migran, se archivan o se borran.

## Fase 1 - Alpha research

- [ ] Migrar alpha specs desde `src/alpha_discovery_base.py` y
  `src/operable_alpha_refinement.py` a YAML.
- [x] Implementar runner que cargue features, fit gates en validation y evalue
  thresholds/horizons/costes.
- [ ] Generar artefactos estandar:
  - [x] `manifest.yaml`
  - [x] `candidate_decisions.parquet`
  - [x] `validation.parquet`
  - [x] `test.parquet`
  - [ ] `trades.parquet`
  - [ ] `daily.parquet`
  - [ ] `monthly.parquet`
  - [x] `report.md`
- [x] Hacer que el dashboard lea el nuevo manifest antes que heuristicas legacy.
- [ ] Cerrar o borrar TODOs/reportes que solo describen ramas degradadas.

## Fase 2 - Strategy runner

- [x] Crear `StrategySpec` YAML por candidato promovido activo; H1b/H1c ya
  tienen spec formal.
- [x] Implementar runner rule-based para H1 con entrada en siguiente open.
- [ ] Separar `strategy` de `backtesting`: la estrategia decide posicion; el
  backtester simula y mide.
- [x] Anadir controles base:
  - base no filtrada
  - same-hour
  - always-flat
  - random/control si aplica
- [x] Generar artefactos strategy:
  - `manifest.yaml`
  - `trades.parquet`
  - `daily.parquet`
  - `monthly.parquet`
  - `summary.parquet`
  - `report.md`
- [x] Implementar promotion gates comunes.

## Fase 3 - Limpieza legacy

- [ ] Mantener solo adaptadores finos para comandos antiguos que sigan aportando valor.
- [ ] Mover o borrar scripts que sean duplicados de `src/alpha`, `src/strategy` o
  `src/backtesting`.
- [ ] Reescribir tests legacy para apuntar a los paquetes nuevos bajo `src/`.
- [ ] Reducir `README.md` a comandos activos y mover historia a docs archivados.
- [ ] Eliminar configs antiguas que no puedan reproducirse con manifest.

## Fase 4 - Paper implementation

- [x] Crear `src/execution/paper`.
- [x] Crear data refresh pipeline para paper H1c.
- [x] Generar order tickets simulados/signal-only para H1c.
- [x] Crear state store local para senal esperada y estado paper.
- [x] Crear paper cycle runner: refresh -> signal -> state -> reporte.
- [x] Crear reconciliation read-only: state esperado vs IBKR positions/orders.
- [x] Crear accounting inicial: fill entrada -> estado open + evento PnL.
- [x] Crear planner H1c reviewable, bloqueado por reconciliation.
- [x] Crear executor H1c bloqueado por defecto con confirmaciones.
- [x] Crear auto runner paper con market-open, anti-duplicado y cash checks.
- [x] Crear wrapper y plantilla launchd cada 15 minutos.
- [x] Implementar salida H1c: pending_exit -> flat y PnL cerrado.
- [x] Registrar drift, latencia y errores.
- [x] Automatizar ciclo cada 15 min y reconciliacion activa solo cuando
  reconciliation devuelva OK.
- [x] Definir limits y kill switches para nuevas entradas antes de cualquier
  promocion live.
- [ ] Promocion live real: requiere decision manual, revision de limites y
  despliegue explicito; no activarla por defecto.

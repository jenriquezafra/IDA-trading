# Hypotheses Registry And Roadmap

Fecha: 2026-05-15

Este documento es el registro operativo de hipotesis. Su objetivo es separar
lo que ya esta implementado, lo que esta en WIP y lo que queda en backlog, con
un orden claro de importancia.

Los candidatos operables se registran aparte en
`docs/candidate_registry.md`. Los IDs `H...` nombran hipotesis; los IDs `C...`
nombran candidatos congelados o variantes operables.

Nota de nomenclatura: `H3` queda asignada desde ahora a **Earnings continuation
intradia condicionado**. La antigua idea `Options ORB` queda como rama
deferida/legacy y no debe ocupar el ID activo `H3`.

## Estado Resumido

| Prioridad | ID | Hipotesis | Estado | Decision actual | Siguiente accion |
|---:|---|---|---|---|---|
| 1 | H1c | Risk-off short continuation en QQQ con filtro de credito interpretable | Implementada / paper candidate | Mantener en observabilidad paper; no extender antes de medir deterioro real | Cerrar fills, slippage y PnL ex-post contra expectativa congelada |
| 2 | H3 | Earnings continuation intradia condicionado | Contract registered / nueva prioridad research | Abrir como nueva vertical event-driven sobre stocks liquidos | Ejecutar data audit point-in-time antes de screening |
| 3 | H8 | Bayesian regime HMM mejorado + confirmation gate | Spec registered / WIP experimental | Primero validar estados, despues anadir senal operable | Ejecutar H8a como filtro causal y comparar contra HMM original |
| 4 | H4 | Read-through de earnings de lider sectorial hacia peers | Backlog prioritario | Buena tesis, requiere mapping economico de peers | Activar solo despues del dataset/event engine de H3 |
| 5 | H5 | Cascade de revisiones de analistas post-earnings | Backlog condicionado | Potencial, pero depende de vendor point-in-time caro | Revaluar cuando haya fuente fiable de estimates/revisions |
| 6 | H6 | Reversion de gap fundamental no confirmado por sector/peers | Backlog experimental | Interesante pero con colas y riesgo de clasificacion | Usar como rama secundaria tras tener news/event classifier |
| 7 | H7 | Shocks FX/tipos sobre companias con exposicion identificable | Backlog avanzado | Tesis fuerte, implementacion compleja | Posponer hasta tener firm exposure database |
| 8 | Legacy O1 | Options ORB sobre SPY/QQQ 0-2 DTE | Deferred | No activar desde H2; opciones solo despues de edge robusto en subyacente | Mantener como data-probe opcional, sin compra de historico |
| 9 | H2 | Equity ORB por pares/spreads relativos | Cerrada / no promovida | No seguir con filtros sobre base negativa | Reabrir solo con mecanismo economico distinto |

## Implementadas

### H1c - Risk-off short continuation en QQQ

Estado: `paper_candidate`.

Spec:

- `configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml`

Evidencia principal:

- `results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/report.md`
- `results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/manifest.yaml`
- `results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/report.md`

Hipotesis economica:

Cuando QQQ cae en contexto risk-off, con presion de volatilidad y credito
debil, la presion vendedora puede continuar intradia por de-risking, hedging,
stop-outs y reduccion de exposicion. H1c usa `spread_credit_12 <= 0` para evitar
una dependencia fragil de un quantile exacto de credito.

Resultado documentado:

- Validation net: aprox. `+7.90%`.
- Test net: aprox. `+7.60%`.
- Avg trade: aprox. `5.7 bps`.
- Coste `5 bps`: positivo en validation/test.
- Coste `7.5 bps`: positivo pero con margen fino.
- Coste `10 bps`: negativo; warning activo.

Pendiente:

- [ ] Reporte agregado de observabilidad paper: senales, fills teoricos/reales,
  slippage, costes y PnL ex-post.
- [ ] Comparar paper contra expectativa congelada por ventana, hora, fold
  analogico y regimen.
- [ ] Pausar nuevas entradas si slippage real supera el limite definido.
- [ ] No usar H1c para justificar opciones hasta que haya evidencia paper
  estable.

### H1 / H1b - Lineage de H1c

Estado: implementadas como research lineage, no activas.

- `H1` encontro la primera senal en h=6, pero era demasiado concentrada.
- `H1b` reparo concentracion con filtro `credit_weak_q50`, pero quedo fragil
  por dependencia del quantile exacto.
- `H1c` reemplaza a H1b con `spread_credit_12 <= 0`.

Decision:

- Mantener como genealogia y evidencia de investigacion.
- No operar H1 ni H1b directamente.

### H2 - Equity ORB por pares/spreads relativos

Estado: implementada como research, no promovida.

Docs y configs:

- `docs/equity_orb_hypothesis.md`
- `configs/strategy/equity_orb_pairs_v1.yaml`
- `configs/strategy/equity_orb_range_quality_v1.yaml`
- `configs/strategy/equity_orb_failed_pairs_v1.yaml`

Lineas testadas:

- `H2.2`: ORB continuation sobre spreads relativos.
- `H2.4`: ORB condicionado por calidad del opening range.
- `H2.5`: failed ORB / reversion.

Decision:

- `H2.2` rechazada: negativa en validation y test a costes actuales.
- `H2.5` rechazada: negativa y no supera controles.
- `H2.4` no promovida: pequeno bolsillo positivo, pero falla controles y
  concentracion.

Regla:

- No construir Options ORB encima de H2.
- No anadir filtros H2.1/H2.3 sobre una base negativa.

## WIP

### H8 - Bayesian regime HMM mejorado + confirmation gate

Estado: `spec_registered`.

Spec:

- `docs/hmm_bayesian_regime_h8.md`

Artefacto inicial:

- `src/bayesian_regime_hmm.py`

Hipotesis economica:

Un filtro bayesiano de regimen, inspirado en el HMM manual de TradingView pero
mejorado con log-space, reset por sesion, estados separados de chop/ruido y
eficiencia direccional, puede actuar como gate causal para setups de
continuation intradia. El estado no es una senal por si solo: solo permite o
bloquea una senal posterior.

Version inicial:

```text
Primero validar si H8a separa bull_trend, bear_stress, chop_compression y
volatile_noise de forma estable. Solo si los estados son interpretables,
anadir H8c: continuation long en bull_trend y short en bear_stress con
confirmacion VWAP/breakout/cross-asset.
```

Decision:

- No optimizar contra PnL antes de nombrar estados.
- Comparar contra el HMM original de tres estados y contra filtros simples
  momentum/ATR.
- Mantener como diagnostico si los estados son utiles pero no mejoran una
  senal operable neta.

### H3 - Earnings continuation intradia condicionado

Estado: `contract_registered`.

Spec y contrato:

- `docs/earnings_continuation_h3.md`
- `configs/strategy/equity_earnings_continuation_h3_v1.yaml`
- `docs/earnings_continuation_h3_data_audit.md`

Hipotesis economica:

Despues de earnings con sorpresa fundamental positiva clara, gap moderado,
volumen alto y confirmacion del sector/peers, el mercado puede seguir
incorporando informacion durante la sesion. El edge no debe ser momentum puro:
debe ser retorno condicional a informacion nueva, absorcion inicial y
confirmacion economica.

Version inicial:

```text
Si una accion liquida reporta earnings pre-market con sorpresa positiva,
gap positivo moderado, revenue/guidance no contradictorio, volumen relativo
alto en los primeros 30 minutos y sector/peers confirmando, comprar tras los
primeros 30 minutos y cerrar al cierre o T+1.
```

Version short:

Queda fuera de la primera implementacion salvo que los datos demuestren simetria.
Los shorts post-earnings tienen mas riesgo de borrow, squeeze y colas.

Universo inicial candidato:

- EE. UU. large/liquid: S&P 500 + Nasdaq 100 liquido.
- Europa liquida queda para v2 por complejidad de horarios, spreads y vendors.
- Excluir small caps, biotech binario, SPACs, M&A, litigios/fraude y nombres
  con spreads amplios.

Instrumentos:

- V1: acciones cash, sin opciones.
- Hedge opcional: sector ETF o indice futuro, solo para medir retorno residual.
- Opciones quedan fuera hasta demostrar edge robusto en subyacente.

Senal candidata:

- Earnings pre-market o after-market con timestamp fiable.
- `EPS_surprise` positiva y normalizada.
- `Revenue_surprise` positiva o no negativa.
- Guidance no negativa, si el dato esta disponible point-in-time.
- Gap positivo moderado: no demasiado pequeno, no extremo.
- Volumen relativo en primeros 30 minutos alto frente a patron historico.
- Precio a los 30 minutos por encima de VWAP o midpoint del rango inicial.
- Sector/peer basket positivo o neutral.
- Excluir dias con macro/evento simultaneo que domine el movimiento.

Target primario:

- Retorno residual neto desde entrada `T 10:00 ET` hasta cierre.

Targets secundarios:

- `T+1 open`.
- `T+1 10:30 ET`.
- `T+1 close`.

Metricas obligatorias:

- `P(r_net > 0 | signal)`.
- Esperanza condicional.
- Media, mediana y percentiles `5/25/75/95`.
- Hit rate.
- Payoff ratio.
- MAE/MFE intradia.
- Drawdown por dia de evento.
- Intervalos de confianza por bootstrap clusterizado por fecha y ticker.
- Sensibilidad a costes, slippage y retraso de ejecucion.

## Backlog

### H4 - Read-through de earnings de lider sectorial hacia peers

Hipotesis:

Una compania lider revela informacion sobre demanda, precios, margenes,
inventarios o costes que afecta a peers. El lider ajusta primero; algunos peers
pueden ajustar con retraso.

Estado:

- Backlog prioritario despues de H3 porque reutiliza calendario de earnings,
  timestamps, consenso y motor event-driven.

Activacion:

- Solo cuando H3 tenga dataset de eventos y mapping sector/peer versionado.

### H5 - Cascade de revisiones de analistas post-earnings

Hipotesis:

Tras earnings/guidance, los analistas actualizan estimates en cascada. Las
primeras revisiones de EPS/revenue pueden anticipar revisiones posteriores y
drift de precio de 1-5 sesiones.

Bloqueo:

- Requiere estimates/revisions point-in-time con timestamps reales.
- Sin vendor fiable, no hacer backtest porque el look-ahead es facil.

### H6 - Reversion de gap fundamental no confirmado

Hipotesis:

Algunos gaps por titular fundamental son sobrerreaccion de apertura si sector,
peers y variables relacionadas no confirman el movimiento.

Riesgo:

- Colas negativas grandes si la noticia si es estructural.
- Depende mucho de clasificar noticias correctamente.

Uso:

- Rama secundaria para cuando exista event/news classifier.

### H7 - Shock FX/tipos por exposicion firm-level

Hipotesis:

Shocks en FX o tipos reprician cash flows de companias con exposicion economica
clara, pero el mercado puede discriminar lentamente entre ganadores y perdedores.

Bloqueo:

- Requiere exposiciones por revenue geography, coste, deuda, sensibilidad a
  yields y betas historicas point-in-time.

### Legacy O1 - Options ORB

Estado: deferred.

Doc historico:

- `docs/options_orb_hypothesis.md`

Decision:

- No activar como H3.
- No comprar datos historicos de opciones para una tesis ORB que no supero
  controles en subyacente.
- Reabrir solo como data-probe acotado si una hipotesis de subyacente ya tiene
  edge robusto y se quiere medir convexidad neta de spreads/theta/IV.

## Roadmap H3 - Earnings Continuation Intradia Condicionado

### Fase 0 - Spec y contrato

- [x] Crear spec dedicada `docs/earnings_continuation_h3.md`.
- [x] Crear contrato YAML `configs/strategy/equity_earnings_continuation_h3_v1.yaml`.
- [x] Definir universo inicial exacto: tickers, fechas, liquidez minima,
  market cap minima y exclusiones.
- [x] Definir convencion de eventos: pre-market, after-market y durante sesion.
- [x] Definir artefactos esperados: `manifest`, `events`, `trades`, `daily`,
  `monthly`, `summary`, `distribution`, `report`.
- [x] Definir promotion gates antes de mirar resultados.

### Fase 1 - Data audit

- [x] Confirmar fuente de calendario de earnings con timestamps reales.
- [x] Confirmar fuente de consenso pre-evento para EPS y revenue.
- [x] Confirmar si hay guidance point-in-time; si no, dejar guidance fuera de v1.
- [x] Confirmar intradia 1min/5min para universo y periodo.
- [x] Confirmar bid/ask o proxy conservador de spread/slippage.
- [x] Construir sector/industry mapping historico.
- [x] Construir peer baskets iniciales o usar sector ETF como proxy v1.
- [x] Auditar timezone, horario regular, halts, corporate actions y splits.
- [x] Rechazar cualquier fuente que entregue consenso revisado post-evento sin
  snapshot pre-evento.

### Fase 2 - Dataset de eventos

- [ ] Comprar/activar add-on `Benzinga Earnings` en Polygon/Massive.
  - Bloqueante para generar `earnings_events.parquet` real, auditar snapshots
    point-in-time y correr screening/backtest H3 v1.
  - No bloquea seguir implementando el pipeline con payload raw autorizado o
    tests.
- [ ] Generar `earnings_events.parquet` point-in-time.
  - Generador implementado: `src/data/earnings_events_h3.py`.
  - Bloqueado para generacion real: endpoint `/benzinga/v1/earnings` devuelve
    `403 Forbidden` con la key local, pendiente de entitlement o payload raw
    autorizado.
- [x] Separar eventos pre-market y after-market.
  - El generador escribe particiones explicitas:
    `earnings_events_pre_market.parquet`,
    `earnings_events_after_market.parquet` y
    `earnings_events_excluded_timing.parquet`.
  - Repaso pendiente con Benzinga real: validar buckets/timestamps exactos y
    confirmar que la particion no depende de campos revisados.
- [x] Para v1, operar solo eventos que permitan entrada regular a los 30 minutos
  de la sesion siguiente.
  - Implementado en `src/data/earnings_events_h3.py`: `entry_timestamp` se
    deriva de calendario XNYS como `market_open + 30min`, `exit_timestamp` del
    `market_close`, y sesiones no regulares completas quedan excluidas con
    `non_full_regular_session`.
  - Pendiente posterior: al construir el panel intradia, confirmar barra limpia
    de entrada entre `10:00` y `10:05`.
- [x] Calcular `eps_surprise_z`, `revenue_surprise_z` y flags de dato faltante.
  - Implementado con sorpresa porcentual contra consenso pre-evento y z-score
    expansivo sobre sesiones anteriores solamente.
  - Flags explicitos: `missing_eps_actual`, `missing_eps_consensus`,
    `missing_revenue_actual`, `missing_revenue_consensus`,
    `eps_consensus_abs_below_floor` y `revenue_consensus_abs_below_floor`.
  - Repaso pendiente con Benzinga real: auditar snapshots pre-evento y
    distribucion de `eps_surprise_z`/`revenue_surprise_z`.
- [x] Calcular gap normalizado por volatilidad reciente.
  - Implementado como enriquecimiento opcional con panel OHLCV intradia:
    `gap_return = event_session_open / previous_regular_close - 1` y
    `gap_atr = gap_return / recent_atr_return`.
  - `recent_atr_return` usa true range diario normalizado del mismo simbolo con
    solo sesiones anteriores; si faltan precios o historia ATR, el evento queda
    no tradeable con flags de gap.
  - Repaso pendiente con Benzinga real + panel intradia real: validar cobertura
    de eventos y coherencia de gaps.
- [x] Calcular volumen relativo de primeros 30 minutos.
  - Implementado como enriquecimiento opcional con panel OHLCV intradia:
    `volume_30m` suma `09:30-10:00 ET` y `expected_volume_30m` usa mediana
    rolling de sesiones anteriores del mismo simbolo.
  - `rel_volume_30m = volume_30m / expected_volume_30m`; si faltan barras,
    volumen o historia suficiente, el evento queda no tradeable con flags de
    volumen.
  - Repaso pendiente con Benzinga real + panel intradia real: revisar
    distribucion por ticker y sesiones con volumen incompleto.
- [x] Calcular VWAP/rango inicial de 30 minutos.
  - Implementado como enriquecimiento opcional con panel OHLCV intradia sobre
    `09:30-10:00 ET`: `vwap_30m`, `range_high_30m`, `range_low_30m` y
    `close_30m`.
  - Si falta VWAP, rango, cierre o barras suficientes, el evento queda no
    tradeable con flags de ventana inicial.
  - Repaso pendiente con Benzinga real + panel intradia real: validar barra
    limpia de entrada entre `10:00` y `10:05`.
- [x] Calcular retorno sectorial/peer en los primeros 30 minutos.
  - Implementado como enriquecimiento opcional con panel OHLCV intradia y mapas
    `sector_map`/`peer_proxy`: calcula `sector_id`, `sector_proxy`,
    `sector_return_30m`, `peer_proxy_symbol`, `peer_proxy_fallback_used` y
    `peer_proxy_return_30m`.
  - V1 usa ETF sectorial; si falta el ETF y SPY esta disponible, usa fallback
    marcado. Si falta proxy o mapping, el evento queda no tradeable con flags.
  - Repaso pendiente con Benzinga real + panel intradia real: medir fallback a
    SPY y cobertura de ETFs sectoriales por evento.
- [x] Marcar exclusiones: macro days, earnings simultaneos de peers, halts,
  spreads altos y noticias binarias.
  - Implementado como capa final en `src/data/earnings_events_h3.py`: marca
    `macro_day_flag`, `simultaneous_peer_earnings_flag`, `halt_flag`,
    `high_spread_flag` y `binary_news_flag`.
  - Entradas externas opcionales: macro calendar, halts, quote quality y binary
    news. Peers simultaneos se detectan con eventos del mismo sector en la
    misma `event_session`.
  - Repaso pendiente con Benzinga real: revisar simultaneidad de peers sobre el
    calendario completo y completar tablas externas antes de screening.

Repaso obligatorio cuando este activo `Benzinga Earnings`: regenerar Fase 2 con
payload real, panel intradia real y tablas externas de exclusiones; comparar
conteos/flags contra los tests sintéticos antes de pasar a screening.

### Fase 3 - Screening baseline

- [x] Baseline 1: comprar todos los beats positivos y cerrar al cierre.
  - Implementado en `src/strategy/equity_earnings_continuation_screening.py`
    como `earnings_beat_only`.
  - Repaso pendiente con Benzinga real: comprobar que EPS/revenue beats salen
    de snapshots pre-evento y no de consenso revisado.
- [x] Baseline 2: gap moderado sin filtros de earnings.
  - Implementado como `gap_moderate_only`, usando el rango `gap_atr` de config
    sin filtros EPS/revenue.
  - Repaso pendiente con Benzinga real + panel intradia real: auditar gaps por
    split y por ticker antes de interpretar edge.
- [x] Baseline 3: momentum intradia sin evento de earnings.
  - Implementado como `intraday_momentum_no_event`: muestrea sesiones sin evento
    del mismo ticker en el mismo split y exige momentum positivo hasta entrada.
  - Repaso pendiente con panel real: validar que el universo sin evento tiene
    cobertura comparable al universo de eventos.
- [x] Baseline 4: mismo horario/random same-frequency por ticker.
  - Implementado como `random_same_frequency_by_ticker` y
    `same_hour_by_ticker`, con seed reproducible y misma frecuencia del
    candidato H3 por ticker/split.
  - Repaso pendiente con Benzinga real: comprobar conteos por ticker y folds.
- [x] Baseline 5: retorno sectorial equivalente.
  - Implementado como `sector_proxy_equivalent`, operando el proxy sectorial
    (`peer_proxy_symbol`/`sector_proxy`, fallback `SPY`) en el mismo horizonte.
  - Repaso pendiente con panel real: medir fallback a `SPY` y separar sectores
    con cobertura incompleta.
- [x] Medir retorno bruto y neto para horizontes intradia, T+1 open y T+1 close.
  - El runner usa los horizontes de config: `same_session_close`,
    `t_plus_1_open`, `t_plus_1_10_30` y `t_plus_1_close` si estan definidos.
  - Aplica costes round-trip base/conservador/stress.
- [x] Medir retornos residualizados contra sector/indice.
  - Artefactos incluyen `sector_residual_return`,
    `sector_residual_net_return`, `index_residual_return` e
    `index_residual_net_return`.
- [x] Reportar distribucion completa, no solo media.
  - `distribution.parquet` reporta media, std, min, p05, p25, mediana, p75,
    p95 y max para retornos bruto/neto/residuales.

Repaso obligatorio cuando este activo `Benzinga Earnings`: ejecutar Fase 3 con
`earnings_events.parquet` real, panel intradia real y tablas externas de
exclusion; comparar `h3_candidate_screen` contra todos los baselines antes de
implementar la estrategia operable de Fase 4.

### Fase 4 - Estrategia H3 v1

- [x] Implementar runner `src/strategy/equity_earnings_continuation.py`.
  - Implementado como backtest H3 v1 single-name long-only.
  - Repaso pendiente con Benzinga real: ejecutar contra dataset real antes de
    leer resultados economicos.
- [x] Entrada base: 30 minutos despues de apertura regular.
  - El runner usa la primera barra limpia `>= 10:00 ET` y `<= 10:05 ET`.
- [x] Salida base: cierre de la misma sesion.
  - Label primario `h3_v1_primary` usa `same_session_close`.
- [x] Filtros iniciales:
  - [x] sorpresa EPS positiva;
  - [x] revenue no negativa;
  - [x] gap positivo moderado;
  - [x] volumen relativo alto;
  - [x] precio sobre VWAP/rango inicial;
  - [x] sector/peer no contradice.
  - `rel_volume_30m` se ajusta solo con train del fold y se registra en
    `thresholds.parquet`/manifest.
- [x] Costes base/conservador/stress.
  - Costes round-trip aplicados en `trades.parquet`, `summary.parquet` y
    `distribution.parquet`.
- [x] Salida alternativa T+1 solo como sensibilidad, no como seleccion inicial.
  - Label `h3_v1_sensitivity` contiene T+1; no participa en la seleccion
    primaria.
- [x] Generar artefactos reproducibles y manifest.
  - Output: `results/strategy/equity_earnings_continuation/5min/h3_v1`.
- [x] Anadir tests unitarios para timestamp, filtros point-in-time, costes y
  retornos residualizados.
  - Test: `tests/test_equity_earnings_continuation_strategy.py`.

Repaso obligatorio cuando este activo `Benzinga Earnings`: ejecutar Fase 4 con
dataset real y revisar `thresholds.parquet`, conteos por split, costes,
residuales y sensibilidad T+1 antes de pasar a validacion/robustez.

### Fase 5 - Validacion y robustez

- [ ] Walk-forward temporal con thresholds entrenados solo en train.
- [ ] Validation selecciona; test confirma una sola vez.
- [ ] Bootstrap por fecha y ticker.
- [ ] Sensibilidad de entrada: 15, 30, 45 y 60 minutos.
- [ ] Sensibilidad de salida: close, T+1 open, T+1 10:30, T+1 close.
- [ ] Sensibilidad de gap: moderado vs extremo.
- [ ] Sensibilidad de volumen relativo.
- [ ] Separar por sector, market cap, mega caps vs resto, pre-market vs
  after-market.
- [ ] Excluir dias FOMC, CPI, NFP y shocks macro grandes.
- [ ] Costes 1x, 2x, 3x y retrasos de ejecucion 5/15/30 minutos.
- [ ] Medir concentracion por earnings season, dia, ticker y sector.

### Fase 6 - Promotion gates iniciales

Gates candidatos antes de `freeze_review`:

- [ ] Minimo `200` trades en validation y `100` en test, o justificar power
  analysis si el universo queda mas pequeno.
- [ ] Retorno neto positivo en validation y test con coste conservador.
- [ ] Avg trade neto superior a coste stress minimo razonable.
- [ ] `P(r_net > 0 | signal)` superior al baseline same-hour/random con
  intervalo de confianza no trivial.
- [ ] Payoff ratio no dependiente de un unico outlier.
- [ ] Top 5 dias no explican mas del `50%` del PnL absoluto.
- [ ] Al menos 4 sectores con actividad y ningun sector explica mas del `50%`
  del PnL.
- [ ] Supera baseline de gap-only y earnings-beat-only.
- [ ] Supera retorno sectorial/indice equivalente despues de costes.
- [ ] Sobrevive a costes duplicados y retraso de ejecucion.

### Fase 7 - Decision

- [ ] Si falla por falta de edge: cerrar H3 y pasar a H4 solo si el dataset de
  eventos sigue siendo reusable.
- [ ] Si falla por datos/vendor: no inferir nada sobre la hipotesis; bloquear
  hasta conseguir datos point-in-time.
- [ ] Si pasa screening pero falla robustez: congelar decision `continue_research`
  y reformular filtros economicos sin tocar test.
- [ ] Si pasa gates: crear freeze review, manifest canonico y paper signal-only.

## Orden De Trabajo Recomendado

1. Cerrar observabilidad paper de H1c porque es la unica rama operativa.
2. Ejecutar H8a como filtro de regimen causal sin PnL-directed tuning:
   comparar HMM original, H8 cuatro estados y filtros momentum/ATR.
3. Ejecutar data audit H3, sin escribir runner hasta saber que los datos
   point-in-time existen.
4. Construir event dataset reusable para H3/H4/H5.
5. Ejecutar screening H3 con baselines simples antes de anadir filtros.
6. Solo si H3 muestra edge residual neto, avanzar a runner formal y gates.
7. Reutilizar infraestructura H3 para H4 read-through.
8. Mantener H5/H6/H7 en backlog hasta resolver datos especificos.
9. Mantener Options ORB en deferred salvo edge previo claro en subyacente.

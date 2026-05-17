# H3 - Earnings Continuation Intradia Condicionado

Estado: `contract_registered`.

Contrato: `configs/strategy/equity_earnings_continuation_h3_v1.yaml`.
Data audit: `docs/earnings_continuation_h3_data_audit.md`.

Este documento preregistra H3 antes de mirar resultados. La primera version no
opera opciones, no opera shorts y no permite screening si el calendario de
earnings o el consenso pre-evento no son point-in-time.

## Hipotesis

Despues de earnings con sorpresa fundamental positiva clara, gap positivo
moderado, volumen relativo alto en los primeros 30 minutos y confirmacion de
sector/peers, el mercado puede seguir incorporando informacion durante la
sesion regular.

El edge buscado no es momentum generico de apertura. Debe ser retorno residual
neto condicionado a informacion nueva, absorcion inicial y confirmacion
economica.

Version inicial:

```text
Si una accion liquida de EE. UU. reporta earnings pre-market o after-market con
sorpresa EPS positiva, revenue no negativa, gap positivo moderado, volumen
relativo alto y sector no contradictorio, comprar a los 30 minutos de la sesion
regular relevante y cerrar al cierre de esa sesion.
```

## Alcance V1

Instrumento:

- acciones cash de EE. UU.;
- long only;
- sin opciones;
- sin borrow ni shorts;
- hedge sectorial solo para medir retorno residual, no como pata operada en V1.

Timeframe:

- primario: `5min`;
- `1min` aceptable para reconstruir VWAP/rango y luego agregar a `5min`;
- cualquier comparacion debe usar la misma convencion de entrada y salida.

Target primario:

- retorno neto residual desde `10:00 ET` hasta cierre regular de la sesion de
  evento.

Targets secundarios:

- `T+1 open`;
- `T+1 10:30 ET`;
- `T+1 close`.

Los targets secundarios son sensibilidad. No seleccionan la primera version.

## Universo Inicial

Universo exacto de data audit y screening V1:

```text
AAPL, ABBV, ABT, ADBE, AMD, AMGN, AMT, AMZN, APD, AVGO, AXP, BA, BAC, BKNG,
BLK, C, CAT, CI, CL, CMCSA, COF, COP, COST, CRM, CSCO, CVS, CVX, DE, DHR, DIS,
DUK, ELV, ETN, FCX, GE, GOOGL, GS, HD, HON, IBM, INTC, ISRG, JNJ, JPM, KO, LIN,
LLY, LMT, LOW, MA, MAR, MCD, MDT, META, MDLZ, MRK, MS, MSFT, MU, NEE, NFLX,
NKE, NOW, NVDA, ORCL, PANW, PEP, PFE, PG, PLD, PM, QCOM, RTX, SBUX, SCHW, SLB,
SO, T, TGT, TJX, TMUS, TMO, TSLA, TXN, UNH, UNP, UPS, V, VZ, WFC, WMT, XOM
```

Periodo inicial:

- inicio: `2016-01-01`;
- fin: `2025-12-31`;
- no extender la ventana sin nuevo contrato o manifest de revision.

Eligibilidad en fecha de evento:

- market cap minima: `20bn USD`;
- mediana de dollar volume regular de 20 sesiones: `100mm USD`;
- precio minimo: `10 USD`;
- cobertura intradia regular minima por evento: `98%`;
- earnings timestamp real disponible;
- consenso EPS y revenue con snapshot anterior al timestamp del evento.

Exclusiones:

- ADRs, fondos, ETFs, preferreds y share classes duplicadas;
- eventos con M&A, fraude, litigio material o FDA/binario dominante;
- halts o sesiones con datos intradia incompletos;
- splits/corporate actions sin ajuste fiable;
- macro days predefinidos cuando el movimiento de mercado domina el evento;
- peers principales reportando simultaneamente si contaminan la confirmacion.

Nota de sesgo: este universo fijo es suficiente para data audit y primer
screening controlado, pero no autoriza promotion si no queda documentado como
se evita survivorship bias. Para `freeze_review`, el manifest debe explicar si
se uso una fuente de constituyentes point-in-time o justificar por que el
universo fijo no altera la decision.

## Convencion De Eventos

Calendario:

- calendario regular: `XNYS`;
- timezone operativa: `America/New_York`;
- todos los timestamps fuente se almacenan tambien en UTC;
- sesiones half-day quedan excluidas en V1.

Clasificacion:

- `pre_market`: reporte antes de `09:30 ET` y despues del cierre regular previo;
- `after_market`: reporte despues de `16:00 ET`;
- `during_session`: reporte entre `09:30 ET` y `16:00 ET`;
- `unknown_time`: fecha sin hora verificable.

Politica V1:

- operar `pre_market` en la misma sesion regular si el timestamp precede a la
  apertura;
- operar `after_market` en la siguiente sesion regular;
- excluir `during_session`;
- excluir `unknown_time`.

Entrada:

- ventana de absorcion: primeros `30` minutos regulares;
- entrada base: primer open disponible a las `10:00 ET` o despues;
- bloquear evento si no existe barra limpia de entrada entre `10:00` y `10:05`;
- no recalcular filtros con informacion posterior a la barra de entrada.

Salida:

- salida primaria: close regular de la misma sesion;
- si falta close fiable, bloquear el trade;
- salidas `T+1` solo se calculan como sensibilidad preregistrada.

## Datos Requeridos

Fuentes obligatorias antes de screening:

- calendario de earnings con timestamp real: fuente primaria candidata
  `polygon_benzinga_earnings`, confirmada documentalmente y pendiente de sample
  audit con credenciales;
- consenso EPS pre-evento: fuente primaria candidata
  `polygon_benzinga_earnings`, pendiente de snapshot audit;
- consenso revenue pre-evento: fuente primaria candidata
  `polygon_benzinga_earnings`, pendiente de snapshot audit;
- intradia OHLCV: fuente primaria candidata
  `polygon_massive_stocks`, raw `1min` y derived `5min`, pendiente de coverage
  audit y entitlement para `2016-2025`;
- precios ajustados por corporate actions;
- bid/ask o proxy conservador: fuente real candidata
  `polygon_massive_stocks` quotes, con proxy H3 v1 `5/10/20 bps` round-trip
  pendiente de sample audit;
- mapping sector/industry historico o proxy sectorial versionado:
  `configs/strategy/equity_earnings_continuation_h3_sector_map_v1.yaml`;
  es seed estatico para screening y requiere audit historico antes de promotion;
- peer proxy v1: `configs/strategy/equity_earnings_continuation_h3_peer_proxy_v1.yaml`,
  con ETF sectorial como proxy y firm-level peer baskets diferidos.

Guidance:

- existe fuente candidata (`polygon_benzinga_guidance`), pero queda fuera de V1;
- no se usa como filtro, exclusion ni clasificador de beat/miss;
- no se permite inferir guidance con datos publicados despues del evento;
- puede reabrirse en V2 solo con sample audit, join determinista contra el
  evento de earnings y regla preregistrada de clasificacion.

Campos minimos de `earnings_events.parquet`:

- `event_id`;
- `symbol`;
- `report_timestamp_utc`;
- `report_timestamp_et`;
- `report_timing`;
- `event_session`;
- `entry_timestamp`;
- `exit_timestamp`;
- `eps_actual`;
- `eps_consensus`;
- `consensus_snapshot_at_utc`;
- `consensus_source`;
- `consensus_raw_hash`;
- `eps_surprise`;
- `eps_surprise_pct`;
- `eps_surprise_z`;
- `revenue_actual`;
- `revenue_consensus`;
- `revenue_surprise`;
- `revenue_surprise_pct`;
- `revenue_surprise_z`;
- `missing_eps_actual`;
- `missing_eps_consensus`;
- `missing_revenue_actual`;
- `missing_revenue_consensus`;
- `eps_consensus_abs_below_floor`;
- `revenue_consensus_abs_below_floor`;
- `gap_open`;
- `gap_prev_close`;
- `gap_return`;
- `recent_atr_return`;
- `gap_atr`;
- `missing_gap_open`;
- `missing_gap_prev_close`;
- `missing_recent_atr`;
- `volume_30m`;
- `expected_volume_30m`;
- `opening_30m_bar_count`;
- `rel_volume_30m`;
- `missing_volume_30m`;
- `missing_expected_volume_30m`;
- `insufficient_opening_30m_bars`;
- `vwap_30m`;
- `range_high_30m`;
- `range_low_30m`;
- `close_30m`;
- `missing_vwap_30m`;
- `missing_range_30m`;
- `missing_close_30m`;
- `sector_id`;
- `sector_proxy`;
- `sector_return_30m`;
- `peer_proxy_symbol`;
- `peer_proxy_fallback_used`;
- `peer_proxy_return_30m`;
- `missing_sector_mapping`;
- `missing_sector_proxy_return_30m`;
- `missing_peer_proxy_return_30m`;
- `macro_day_flag`;
- `simultaneous_peer_earnings_flag`;
- `spread_bps_30m`;
- `high_spread_flag`;
- `binary_news_flag`;
- `is_full_regular_session`;
- `halt_flag`;
- `suspected_halt_or_bad_session`;
- `split_flag`;
- `split_factor`;
- `corporate_action_flag`;
- `exclusion_flags`;
- `is_tradeable_v1`.

Consenso:

- `eps_consensus` y `revenue_consensus` deben proceder de snapshot pre-evento;
- cualquier fuente que entregue consenso revisado post-evento sin snapshot
  anterior queda rechazada para H3 v1;
- la sorpresa vendor no se usa si no puede recomputarse desde actual y consenso
  pre-evento.

## Senal V1

Filtros preregistrados:

- `eps_surprise_z > 0`;
- `revenue_surprise_z >= 0`;
- `0.25 <= gap_atr <= 2.50`;
- `rel_volume_30m` por encima del percentil `60` calculado solo en train por
  ticker;
- `close_30m >= vwap_30m`;
- `sector_return_30m >= -0.10%`;
- ningun flag de exclusion activo.

Si un filtro requiere threshold estimado, se ajusta solo con train del fold y se
aplica sin cambios a validation/test.

## Costes

Costes round-trip por trade sobre notional:

- base: `5 bps`;
- conservador: `10 bps`;
- stress: `20 bps`.

Los costes incluyen comision, spread, slippage y retraso operativo. Cualquier
fuente real de bid/ask puede reemplazar el proxy, pero debe conservar una serie
stress comparable.

Antes del primer screening, el proxy debe auditarse contra una muestra de quotes
historicas alrededor de entrada y salida. Si el percentil `90` observado supera
el coste conservador, se eleva el coste conservador antes de mirar resultados.

## Baselines Y Controles

Controles obligatorios antes de aceptar H3:

- `earnings_beat_only`: comprar todos los EPS beats positivos;
- `gap_only`: comprar gaps positivos moderados sin usar sorpresa fundamental;
- `intraday_momentum_no_event`: mismo setup tecnico sin earnings;
- `random_same_frequency_by_ticker`: eventos aleatorios con misma frecuencia por
  ticker y calendario;
- `same_hour_by_ticker`: mismo horario en sesiones sin evento;
- `sector_proxy_equivalent`: retorno del ETF sectorial o `SPY` equivalente.

H3 debe superar controles en retorno neto, avg trade, hit rate y distribucion,
no solo en retorno total.

Runner implementado:

```bash
.venv/bin/python -m src.strategy.equity_earnings_continuation_screening \
  --config configs/strategy/equity_earnings_continuation_h3_v1.yaml
```

El runner de Fase 3 escribe artefactos en
`results/strategy/equity_earnings_continuation/5min/phase3_screening`. Calcula
`h3_candidate_screen`, `earnings_beat_only`, `gap_moderate_only`,
`intraday_momentum_no_event`, `random_same_frequency_by_ticker`,
`same_hour_by_ticker` y `sector_proxy_equivalent` para los horizontes definidos
en config. Los thresholds entrenables, como `rel_volume_30m` p60, se ajustan
solo con train del fold. Cuando `Benzinga Earnings` este activo, este screening
debe reejecutarse con `earnings_events.parquet` real y panel intradia real antes
de interpretar resultados.

## Runner H3 V1

Runner de estrategia implementado:

```bash
.venv/bin/python -m src.strategy.equity_earnings_continuation \
  --config configs/strategy/equity_earnings_continuation_h3_v1.yaml
```

La estrategia V1 opera solo `long` single-name, entra en la primera barra limpia
`>= 10:00 ET` y usa `same_session_close` como salida primaria. Las salidas
`T+1` se escriben como sensibilidad bajo `h3_v1_sensitivity`, no como seleccion
inicial. El label primario es `h3_v1_primary`.

El runner ajusta thresholds entrenables solo con train del fold, registra
`thresholds.parquet`, aplica costes base/conservador/stress y calcula retornos
residualizados contra sector proxy y `SPY`. Como el resto de H3, sus resultados
economicos quedan bloqueados hasta tener `Benzinga Earnings` y panel intradia
real point-in-time.

## Artefactos Esperados

Directorio esperado:

```text
results/strategy/equity_earnings_continuation/5min
results/strategy/equity_earnings_continuation/5min/phase3_screening
results/strategy/equity_earnings_continuation/5min/h3_v1
```

Artefactos obligatorios:

- `manifest.yaml`;
- `data_audit.md`;
- `coverage.parquet`;
- raw payload autorizado: `data/raw/polygon/benzinga/earnings/h3_v1/earnings_raw.json`;
- `earnings_events.parquet`;
- `earnings_events_pre_market.parquet`;
- `earnings_events_after_market.parquet`;
- `earnings_events_excluded_timing.parquet`;
- `events.parquet`;
- `trades.parquet`;
- `daily.parquet`;
- `monthly.parquet`;
- `summary.parquet`;
- `distribution.parquet`;
- `thresholds.parquet`;
- `report.md`.

El manifest debe incluir hashes/fingerprints de config, fuentes, calendario,
consenso, datos intradia y reglas de exclusion.

Generador inicial:

```bash
python -m src.data.earnings_events_h3 \
  --config configs/strategy/equity_earnings_continuation_h3_v1.yaml \
  --raw-json data/raw/polygon/benzinga/earnings/h3_v1/earnings_raw.json \
  --intraday-panel data/aligned/equities/5min/h3_v1/panel.parquet \
  --macro-calendar data/events/macro/h3_v1/macro_exclusion_sessions.parquet \
  --halt-events data/events/halts/h3_v1/halt_events.parquet \
  --quote-quality data/events/quality/h3_v1/quote_spread_30m.parquet \
  --binary-news data/events/news/h3_v1/binary_news_events.parquet \
  --output data/events/earnings/h3_v1/earnings_events.parquet
```

El generador no descarga datos. Consume payload raw autorizado para mantener la
frontera entre vendor access, sample audit y construccion de artefactos. Por
defecto tambien materializa particiones por timing junto al parquet principal:
`earnings_events_pre_market.parquet`, `earnings_events_after_market.parquet` y
`earnings_events_excluded_timing.parquet`. Si se pasa `--intraday-panel`, tambien
calcula `gap_return`, `recent_atr_return`, `gap_atr`, `volume_30m`,
`rel_volume_30m`, `vwap_30m`, `range_high_30m`, `range_low_30m` y
`close_30m`, ademas de `sector_return_30m` y `peer_proxy_return_30m` si el
panel incluye ETFs sectoriales/SPY. Sin panel OHLCV esos campos quedan
pendientes. Las tablas opcionales de macro, halts, quote quality y noticias
binarias activan flags de exclusion; sin esas tablas, el pipeline puede
desarrollarse, pero screening/backtest quedan pendientes de audit.

## Promotion Gates

Gates candidatos antes de `freeze_review`:

- minimo `200` trades en validation;
- minimo `100` trades en test;
- retorno neto positivo en validation y test con coste conservador;
- avg trade neto positivo con coste stress;
- `P(r_net > 0 | signal)` superior a random/same-hour con intervalo bootstrap
  no trivial;
- payoff ratio no dependiente de un unico outlier;
- top 5 dias no explican mas del `50%` del PnL absoluto;
- al menos `4` sectores con actividad;
- ningun sector explica mas del `50%` del PnL;
- ningun ticker explica mas del `25%` del PnL;
- superar `gap_only` y `earnings_beat_only`;
- superar retorno sectorial/indice equivalente despues de costes;
- sobrevivir a costes duplicados y retraso de entrada de `15` minutos;
- validation selecciona y test confirma una sola vez.

Si falla por datos/vendor, no se infiere nada sobre la hipotesis. Si falla por
falta de edge, H3 se cierra y el dataset de eventos puede reutilizarse para H4
solo si los artefactos point-in-time quedan limpios.

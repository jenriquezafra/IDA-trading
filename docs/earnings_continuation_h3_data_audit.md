# H3 Data Audit

Estado: `in_progress`.

Spec: `docs/earnings_continuation_h3.md`.
Contrato: `configs/strategy/equity_earnings_continuation_h3_v1.yaml`.

Este audit documenta fuentes antes de construir dataset o runner. El objetivo es
bloquear cualquier backtest que use calendario, consenso o precios con
look-ahead.

## Resumen

| Item | Estado | Decision |
|---|---|---|
| Calendario earnings con timestamp real | `confirmed_source_pending_sample` | Usar Benzinga Earnings via Polygon/Massive como fuente primaria candidata. |
| Consenso EPS/revenue pre-evento | `confirmed_source_pending_snapshot_audit` | Benzinga Earnings expone consenso EPS/revenue, pero screening queda bloqueado hasta demostrar snapshot pre-evento. |
| Guidance point-in-time | `source_exists_deferred_v1` | Benzinga Corporate Guidance existe y esta timestamped, pero queda fuera de H3 v1. |
| Intradia 1min/5min | `confirmed_source_pending_coverage_audit` | Polygon/Massive minute aggregates cubren 1min; 5min se resamplea o se consulta via REST. |
| Bid/ask o proxy slippage | `confirmed_proxy_with_quote_audit` | Usar proxy conservador 5/10/20 bps round-trip y auditar contra Polygon/Massive quotes sample. |
| Sector/industry mapping historico | `static_seed_pending_historical_audit` | Mapping H3 v1 versionado con sector ETF/industry seed; audit historico por SIC/as-of-date queda requerido antes de promotion. |
| Peer baskets o proxy v1 | `sector_etf_proxy_v1` | Usar ETF sectorial como peer proxy v1; firm-level peer baskets quedan diferidos. |
| Timezone, sesion, halts y corporate actions | `rules_registered_pending_sample_audit` | XNYS/America-New_York, excluir half-days/halts y exigir splits/corporate actions reconciliados. |
| Rechazo consenso revisado post-evento | `hard_reject_without_pre_event_snapshot` | Cualquier fuente sin snapshot pre-evento verificable queda rechazada para consenso. |

## 1. Calendario De Earnings

Decision:

- Fuente primaria candidata: Benzinga Earnings via Polygon/Massive.
- Endpoint: `/benzinga/v1/earnings`.
- Uso en H3: calendario de evento, timing pre-market/after-market, status de
  fecha confirmada y metadata de update.
- Estado: aceptada para la primera fase de calendario, pero no autoriza
  screening hasta ejecutar sample audit con credenciales.

Razon:

- El repo ya usa Polygon para datos intradia, por lo que reduce integracion y
  vendor sprawl.
- La documentacion del endpoint Polygon/Massive lista historia completa,
  actualizacion en real time, fecha de evento, `date_status`, `last_updated`,
  `actual_eps`, `estimated_eps`, `actual_revenue`, `estimated_revenue` y `time`.
- La documentacion directa de Benzinga muestra campos equivalentes:
  `date`, `date_confirmed`, `time`, `updated`, EPS/revenue real y estimado.
- El producto Benzinga Earnings declara cobertura de US equities y datos desde
  2012, suficiente para el periodo H3 `2016-01-01` a `2025-12-31`.

Fuentes oficiales:

- Polygon/Massive Benzinga Earnings:
  https://polygon.io/docs/rest/partners/benzinga/earnings
- Benzinga Calendar Earnings:
  https://docs.benzinga.com/api-reference/calendar_api/earnings/returns-the-earnings-data
- Benzinga Corporate Earnings product:
  https://www.benzinga.com/apis/cloud-product/corporate-earnings/

Campos minimos a capturar:

- `ticker`;
- `date`;
- `time`;
- `date_status` o `date_confirmed`;
- `last_updated` o `updated`;
- vendor event id;
- fiscal period/year;
- EPS/revenue actual y estimado, si se aprueba tambien para consenso.

Normalizacion H3:

- Convertir `date` + `time` a `report_timestamp_et` cuando `time` sea
  `HH:MM:SS`.
- Si el proveedor entrega bucket textual (`bmo`, `amc`, `pre-market`,
  `after-market`) en lugar de hora exacta, no fabricamos un timestamp puntual:
  se normaliza a `report_timing` y solo se opera si el bucket permite entrada
  sin look-ahead.
- `date_status=confirmed` o `date_confirmed=1` requerido para eventos usados
  en screening.
- `last_updated`/`updated` debe almacenarse para auditoria de revisiones.

Bloqueos antes de screening:

- Descargar muestra con credenciales para al menos `AAPL`, `MSFT`, `NVDA`,
  `JPM`, `XOM` en 2022-2025.
- Medir porcentaje de eventos con `time` usable.
- Verificar que `pre_market` y `after_market` se separan sin usar precios
  posteriores.
- Confirmar si el endpoint historico preserva el consenso que existia antes del
  evento o si devuelve una version revisada.
- Guardar payload raw versionado para sample audit.

Decision operativa:

- Calendario: `go_for_sample_audit`.
- Dataset/backtest: `blocked_until_sample_audit`.
- Consenso: ver seccion 2.

## 2. Consenso EPS/Revenue Pre-Evento

Decision:

- Fuente primaria candidata: Benzinga Earnings via Polygon/Massive.
- Endpoint: `/benzinga/v1/earnings`.
- Uso en H3: consenso EPS y revenue contra actual para calcular sorpresa
  fundamental.
- Estado: fuente aceptada documentalmente para campos de consenso; no autoriza
  screening hasta demostrar que los valores son snapshots disponibles antes del
  evento.

Razon:

- La documentacion Polygon/Massive del endpoint lista `estimated_eps` y
  `estimated_revenue` junto a EPS/revenue actual, sorpresa y `last_updated`.
- La documentacion directa de Benzinga lista `eps_est`, `revenue_est`,
  `eps_surprise`, `revenue_surprise` y `updated`.
- El endpoint soporta consulta por `updated`, util para ingestion incremental y
  auditoria de cambios, pero la documentacion no garantiza por si sola que una
  consulta historica posterior devuelva exactamente el consenso vigente antes
  del release.

Campos candidatos:

- Polygon/Massive:
  - `estimated_eps`;
  - `estimated_revenue`;
  - `actual_eps`;
  - `actual_revenue`;
  - `eps_surprise`;
  - `revenue_surprise`;
  - `last_updated`.
- Benzinga directo:
  - `eps_est`;
  - `revenue_est`;
  - `eps`;
  - `revenue`;
  - `eps_surprise`;
  - `revenue_surprise`;
  - `updated`.

Normalizacion H3:

- `eps_consensus` = `estimated_eps` o `eps_est`;
- `revenue_consensus` = `estimated_revenue` o `revenue_est`;
- `eps_actual` = `actual_eps` o `eps`;
- `revenue_actual` = `actual_revenue` o `revenue`;
- `consensus_updated_at_utc` = `last_updated` o `updated`;
- no usar `eps_surprise`/`revenue_surprise` como fuente primaria si podemos
  recomputar con actual y consenso.
- `eps_surprise` y `revenue_surprise` se recomputan como
  `actual - consensus`.
- `eps_surprise_pct` y `revenue_surprise_pct` se calculan como
  `(actual - consensus) / abs(consensus)`, siempre que el consenso supere el
  floor configurado.
- `eps_surprise_z` y `revenue_surprise_z` se calculan con z-score expansivo
  usando solo sesiones de eventos anteriores; eventos de la misma sesion y
  eventos futuros quedan fuera del fit.
- Flags explicitos de dato faltante: `missing_eps_actual`,
  `missing_eps_consensus`, `missing_revenue_actual` y
  `missing_revenue_consensus`.

Bloqueos antes de screening:

- Descargar payload raw para eventos ya reportados y futuros/proximos en los
  mismos tickers de sample audit.
- Confirmar que antes del release existen `eps_est`/`revenue_est` o
  `estimated_eps`/`estimated_revenue` sin `actual_eps`/`actual_revenue`.
- Confirmar que despues del release los campos de consenso no se reescriben de
  forma que cambie la sorpresa historica.
- Si el endpoint solo entrega estado final revisado, exigir archivo/delta
  historico de updates o rechazarlo para consenso pre-evento.
- Registrar missingness por campo; revenue consensus no puede imputarse con
  datos posteriores.

Decision operativa:

- Consenso: `go_for_snapshot_sample_audit`.
- Dataset/backtest: `blocked_until_consensus_snapshot_audit`.

## 3. Guidance Point-In-Time

Decision:

- Fuente candidata confirmada documentalmente: Benzinga Corporate Guidance via
  Polygon/Massive.
- Endpoint Polygon/Massive: `/benzinga/v1/guidance`.
- Endpoint Benzinga directo: `/api/v2.1/calendar/guidance`.
- Estado H3 v1: excluido.

Razon:

- La documentacion Polygon/Massive describe records de guidance timestamped con
  `date`, fiscal period, release type, prior guidance y `last_updated`.
- La guia de Benzinga indica que el guidance proviene de comunicaciones
  oficiales de management y preserva rangos, estimaciones y timing del anuncio.
- La documentacion directa de Benzinga expone `date`, `time`, `updated`,
  `eps_guidance_*`, `revenue_guidance_*`, `is_primary` y `prelim`.
- Aunque la fuente existe, incluir guidance en H3 v1 meteria otra capa de
  missingness, parsing de rangos y comparacion contra expectativas. El primer
  screening debe aislar sorpresa EPS/revenue, gap, volumen y confirmacion
  sectorial.

Fuentes oficiales:

- Polygon/Massive Corporate Guidance:
  https://polygon.io/docs/rest/partners/benzinga/corporate-guidance
- Benzinga Guidance API:
  https://docs.benzinga.com/api-reference/calendar-api/get-guidance
- Benzinga Guidance Process:
  https://docs.benzinga.com/api-reference/guides/guidance-process-explained
- Benzinga Corporate Guidance product:
  https://www.benzinga.com/apis/cloud-product/corporate-guidance/

Campos candidatos para V2:

- `ticker`;
- `date`;
- `time`;
- `updated` o `last_updated`;
- `is_primary`;
- `prelim`;
- `eps_guidance_min`;
- `eps_guidance_max`;
- `eps_guidance_est`;
- `revenue_guidance_min`;
- `revenue_guidance_max`;
- `revenue_guidance_est`;
- prior guidance fields;
- fiscal period/year.

Politica H3 v1:

- No usar guidance como filtro.
- No usar ausencia de guidance como exclusion.
- No usar guidance para clasificar beats/misses.
- Mantener campos de guidance fuera de `earnings_events.parquet` v1 salvo
  `guidance_available=false` si el dataset necesita compatibilidad futura.

Requisitos para reabrir en V2:

- Sample audit con payload raw y entitlement confirmado.
- Join determinista entre guidance announcement y earnings event.
- Prueba de que `date`/`time`/`updated` permiten usar solo guidance disponible
  antes de la entrada `10:00 ET`.
- Regla preregistrada para clasificar guidance positiva, neutral o negativa sin
  usar reaccion de precio posterior.

Decision operativa:

- Guidance: `exclude_v1_candidate_v2`.
- Dataset/backtest H3 v1: `not_blocked_by_guidance`.

## 4. Intradia 1min/5min

Decision:

- Fuente primaria bulk: Polygon/Massive Stocks Flat Files
  `us_stocks_sip/minute_aggs_v1`.
- Fuente secundaria/sample: Polygon/Massive Stocks REST Custom Bars
  `/v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}`.
- Granularidad canonica H3: almacenar `1min` raw y derivar `5min` regular
  session con resampling local.
- Estado: fuente aceptada documentalmente; screening bloqueado hasta coverage
  audit por tickers/eventos y confirmacion de entitlement.

Razon:

- Los Flat Files incluyen minute aggregates OHLCV para todas las acciones de EE.
  UU. en un fichero diario, lo que encaja mejor con 92 tickers durante
  `2016-01-01` a `2025-12-31` que miles de llamadas REST por simbolo.
- REST Custom Bars permite pedir OHLCV historico por ticker, rango, multiplier
  y timespan, y soporta `1/minute` o `5/minute`; es adecuado para sample audit,
  reconciliacion y descargas acotadas.
- El repo ya implementa `download_polygon_ohlcv` con endpoint REST aggregates,
  `source_interval`, `adjusted=true` y resampling local a `5m`.

Fuentes oficiales:

- Polygon/Massive Stocks REST Custom Bars:
  https://polygon.io/docs/rest/stocks/aggregates/custom-bars
- Polygon/Massive Stocks Flat Files overview:
  https://polygon.io/docs/flat-files/stocks/overview
- Polygon/Massive Stocks Minute Aggregates:
  https://polygon.io/docs/flat-files/stocks/minute-aggregates/2023

Entitlement minimo:

- Para cubrir todo H3 `2016-01-01` a `2025-12-31`, se requiere plan con `all
  history` o acceso business equivalente.
- Planes con `5 years` o `10 years` pueden no cubrir el inicio de 2016 en la
  fecha actual del proyecto.
- El audit debe registrar plan, fecha de consulta y respuesta real de acceso.

Normalizacion H3:

- raw canonical: `1min`, timestamp UTC original + conversion a
  `America/New_York`;
- derived canonical: `5min`, `label=left`, `closed=left`,
  `origin=start_day`;
- filtrar regular session `09:30-16:00`;
- excluir half-days en V1;
- conservar pre-market/after-hours solo para gap/open diagnostics si el pipeline
  los requiere, no para entrada/salida primaria;
- expected bars por sesion regular: `390` para `1min`, `78` para `5min`.

Corporate actions:

- Flat Files de stocks son unadjusted; si se usan para backtest se debe aplicar
  ajuste manual con splits/corporate actions.
- REST Custom Bars puede pedirse con `adjusted=true`; usarlo como reconciliacion
  de sample y para validar la logica de ajuste.
- Cualquier evento con split/corporate action no ajustado de forma fiable queda
  excluido.

Bloqueos antes de screening:

- Descargar muestra de `1min` para `AAPL`, `MSFT`, `NVDA`, `JPM`, `XOM` en al
  menos 20 sesiones con earnings entre 2022 y 2025.
- Calcular cobertura por ticker/event session: barras esperadas, barras
  presentes, duplicados, gaps y timezone.
- Confirmar que el resampling `1min -> 5min` reproduce entradas `10:00 ET` y
  closes regulares.
- Confirmar plan/historia suficiente para todo `2016-2025`.
- Definir ruta de almacenamiento raw y derived antes de generar
  `earnings_events.parquet`.

Decision operativa:

- Intradia: `go_for_coverage_sample_audit`.
- Dataset/backtest: `blocked_until_intraday_coverage_audit`.

## 5. Bid/Ask O Proxy Conservador

Decision:

- Fuente real de auditoria: Polygon/Massive Stocks Quotes.
- Fuente primaria bulk candidata: Flat Files `us_stocks_sip/quotes_v1`.
- Fuente secundaria/sample: REST NBBO Quotes.
- Coste H3 v1 preregistrado: proxy round-trip sobre notional:
  - base: `5 bps`;
  - conservador: `10 bps`;
  - stress: `20 bps`.
- Estado: proxy aceptado para screening solo despues de sample audit que
  demuestre que no es inferior a spreads observados en las ventanas de entrada
  y salida.

Razon:

- Full quote history para 92 tickers durante 2016-2025 puede ser mucho mas
  pesado que OHLCV minute aggregates.
- H3 v1 opera acciones large/liquid y no necesita microestructura completa para
  decidir si existe edge bruto/residual; si no sobrevive a `10-20 bps`
  round-trip, no merece pasar a implementacion mas cara.
- Polygon/Massive ofrece quotes historicas con bid/ask via REST y Flat Files,
  suficientes para calibrar una muestra de spreads por ticker/evento.
- El repo ya tiene convencion de costes por bps y `src/costs.py` separa
  comision, spread, slippage e impacto cuando haga falta modelar componentes.

Fuentes oficiales:

- Polygon/Massive Stocks Quotes REST:
  https://polygon.io/docs/rest/stocks/quotes
- Polygon/Massive Stocks Quotes Flat Files:
  https://polygon.io/docs/flat-files/stocks/quotes/2023
- Polygon/Massive Stocks Flat Files overview:
  https://polygon.io/docs/flat-files/stocks/overview

Normalizacion H3:

- `quoted_spread_bps = 10000 * (ask - bid) / mid`;
- `mid = (bid + ask) / 2`;
- capturar quotes alrededor de:
  - entrada `10:00 ET`;
  - cierre regular;
  - alternativa `T+1 open` si se calcula sensibilidad;
- medir spread por ticker/evento como mediana y percentiles `75/90/95` en una
  ventana corta alrededor de la decision;
- no usar quotes posteriores a la entrada para decidir si el trade existe.

Proxy preregistrado:

- base `5 bps` round-trip;
- conservador `10 bps` round-trip;
- stress `20 bps` round-trip;
- retrasos de ejecucion: `5`, `15`, `30` minutos;
- H3 solo puede promocionar si validation/test son positivos con coste
  conservador y sobreviven a stress/retraso segun gates.

Sample audit requerido:

- Para `AAPL`, `MSFT`, `NVDA`, `JPM`, `XOM`, descargar quotes alrededor de al
  menos 20 eventos de earnings entre 2022 y 2025.
- Comparar percentiles de spread efectivo contra proxy `5/10/20 bps`.
- Marcar tickers/eventos con spread anormalmente alto o quotes incompletas.
- Si el percentil `90` de spread+slippage estimado supera `10 bps`, elevar el
  coste conservador antes del primer screening.
- Si quotes historicas no estan disponibles por entitlement, mantener proxy
  stress `20 bps` y documentar que la promocion queda limitada hasta calibrar
  quotes reales.

Decision operativa:

- Costes H3 v1: `proxy_conservative_bps`.
- Quotes: `go_for_quote_sample_audit`.
- Dataset/backtest: `blocked_until_quote_or_proxy_sample_audit`.

## 6. Sector/Industry Mapping Historico

Decision:

- Mapping seed versionado: `configs/strategy/equity_earnings_continuation_h3_sector_map_v1.yaml`.
- Uso H3 v1: sector ETF proxy y coarse industry group para filtros,
  residualizacion y concentracion.
- Fuente historica candidata: Polygon/Massive Ticker Overview con parametro
  `date`, capturando `cik`, `sic_code`, `sic_description`, `active`,
  `delisted_utc`, `list_date` y market cap as-of-date.
- Cross-check oficial: SEC SIC code list y filing headers por CIK.
- Estado: construido para screening como seed estatico; no es suficiente por si
  solo para freeze/promotion.

Razon:

- H3 necesita sector proxy desde el primer dataset para medir confirmacion de
  los primeros 30 minutos y retorno residual.
- GICS point-in-time real suele depender de vendor de clasificacion historica.
  No conviene bloquear el primer sample audit por esto si el filtro usa ETFs
  sectoriales amplios y el mapping queda versionado.
- Polygon/Massive documenta Ticker Overview con `date` point-in-time y campos
  de SIC, CIK, list/delist y market cap. SIC no sustituye a GICS, pero sirve
  como ancla auditada para detectar cambios de industria/issuer.

Fuentes oficiales:

- Polygon/Massive Ticker Overview:
  https://massive.com/docs/rest/stocks/tickers/ticker-overview
- SEC SIC Code List:
  https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list

Politica H3 v1:

- Usar `sector_proxy` del mapping seed para:
  - `sector_return_30m`;
  - retorno residual vs sector;
  - concentracion por sector;
  - peer basket v1 si no hay peers firm-level.
- Usar `industry_group` solo para diagnostico y agrupacion, no como filtro de
  seleccion en el primer screening.
- Registrar en manifest hash del mapping seed.

Bloqueos antes de promotion:

- Ejecutar audit as-of-date por evento con Ticker Overview `date=event_session`
  para todos los tickers usados.
- Guardar `cik`, `sic_code`, `sic_description`, `active`, `delisted_utc` y
  `list_date`.
- Comparar el seed sector ETF contra SIC/as-of-date y marcar discrepancias.
- Si hay cambios materiales de clasificacion o issuer, repetir screening o
  justificar que no altera seleccion/resultado.

Decision operativa:

- Sector mapping: `static_seed_ok_for_screening`.
- Freeze/promotion: `blocked_until_historical_mapping_audit`.

## 7. Peer Baskets Iniciales O Sector ETF Proxy

Decision:

- Contrato: `configs/strategy/equity_earnings_continuation_h3_peer_proxy_v1.yaml`.
- V1 usa sector ETF como proxy economico de peers.
- Firm-level peer baskets quedan diferidos a V2.
- `SPY` es fallback si el ETF sectorial no tiene cobertura limpia en la fecha
  de evento.

Razon:

- H3 primero necesita saber si hay edge residual post-earnings en nombres
  liquidos. Construir peers firm-level point-in-time antes del dataset de
  eventos mete complejidad y riesgo de sesgo de clasificacion.
- El sector ETF da una proxy observable, tradeable y facil de auditar con los
  mismos datos intradia que el stock.
- La confirmacion inicial solo exige que el contexto sectorial no contradiga:
  `sector_return_30m >= -0.10%`.
- La residualizacion primaria puede medirse como retorno del stock menos retorno
  del ETF sectorial sin asumir membership exacta del ETF.

Politica H3 v1:

- `peer_proxy_return_30m` = retorno del ETF sectorial en los primeros 30
  minutos.
- `sector_return_30m` y `peer_proxy_return_30m` son equivalentes en V1.
- Si falta cobertura limpia del ETF sectorial, usar `SPY` como fallback y marcar
  `peer_proxy_fallback_used=true`.
- No construir baskets de tickers peers en V1.
- No excluir eventos por peers simultaneos salvo que exista flag claro en el
  calendario; con proxy ETF, ese riesgo se reporta como diagnostico.

Context symbols requeridos:

- `SPY`, `XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`, `XLU`,
  `XLV`, `XLY`.

Campos minimos en eventos:

- `sector_id`;
- `sector_proxy`;
- `peer_proxy_symbol`;
- `peer_proxy_fallback_used`;
- `peer_proxy_return_30m`.

Bloqueos antes de screening:

- Confirmar cobertura intradia de los sector ETFs requeridos en ventanas de
  evento.
- Medir fallback rate por sector y por year.
- Si un sector depende de fallback `SPY` en exceso, reportarlo separado y no
  usarlo para justificar edge sectorial.

Decision operativa:

- Peer proxy: `sector_etf_proxy_ok_for_screening`.
- Firm-level peer baskets: `deferred_v2`.

## 8. Timezone, Horario Regular, Halts, Corporate Actions Y Splits

Decision:

- Timezone operativa: `America/New_York`.
- Storage timezone: `UTC`.
- Calendario regular: `XNYS`, sesion `09:30-16:00`.
- Half-days: excluidos en H3 v1.
- Halts: eventos con halt oficial o gap intradia incompatible alrededor de
  entrada/salida quedan excluidos.
- Corporate actions: splits deben estar reconciliados antes de usar precios.
- Estado: reglas registradas; screening bloqueado hasta sample audit.

Fuentes y herramientas:

- Calendario historico local: `pandas_market_calendars` / XNYS, ya usado por
  el repo para validacion de sesiones.
- Cross-check forward-looking: Polygon/Massive Market Status y Market Holidays.
- Halts oficiales: Nasdaq Trader Trading Halt History/Search.
- Splits: Polygon/Massive `/v3/reference/splits`.
- Dividends/special cash diagnostics: Polygon/Massive `/v3/reference/dividends`.
- Ticker changes: Polygon/Massive Ticker Events, solo como diagnostico porque
  el endpoint documenta `ticker_change` como experimental.

Fuentes oficiales:

- Polygon/Massive Market Status:
  https://polygon.io/docs/rest/stocks/market-operations/market-status
- Polygon/Massive Market Holidays:
  https://polygon.io/docs/rest/stocks/market-operations/market-holidays
- Nasdaq Trader Halt History:
  https://nasdaqtrader.com/trader.aspx?id=TradingHaltHistory
- Nasdaq Trader Halt Codes:
  https://www.nasdaqtrader.com/Trader.aspx?id=TradeHaltCodes
- Polygon/Massive Splits:
  https://polygon.io/docs/rest/stocks/corporate-actions/splits
- Polygon/Massive Dividends:
  https://polygon.io/docs/rest/stocks/corporate-actions/dividends
- Polygon/Massive Ticker Events:
  https://polygon.io/docs/rest/stocks/corporate-actions/ticker-events

Timezone policy:

- Cada timestamp fuente se guarda como UTC y se deriva una columna ET.
- No se aceptan timestamps naive.
- `session` se define por fecha ET de la sesion regular.
- Eventos `after_market` se asignan a la siguiente sesion XNYS completa.
- Eventos `pre_market` se asignan a la misma sesion XNYS si el timestamp es
  anterior a `09:30 ET`.
- Validar explicitamente semanas con cambio DST en marzo y noviembre.

Regular session policy:

- Barras de entrada/salida solo pueden estar en `09:30-16:00 ET`.
- Entrada base: primer open limpio a `10:00 ET` o despues.
- Bloquear si la entrada limpia no existe entre `10:00` y `10:05`.
- Salida primaria: close regular de la misma sesion.
- Excluir half-days porque no tienen el mismo cierre ni la misma geometria de
  primera media hora/cierre.
- Expected bars: `390` en `1min`, `78` en `5min`.

Halt policy:

- Marcar `halt_flag=true` si Nasdaq Trader reporta halt/pause en el ticker y
  sesion de evento.
- Marcar `suspected_halt_or_bad_session=true` si hay missing streak, volumen
  cero anormal o ausencia de quotes/barras alrededor de entrada/salida.
- Excluir cualquier trade si el halt o la sospecha afecta la ventana
  `09:30-10:05` o la ventana de salida.
- Reportar halts como diagnostico aunque se excluyan de la muestra tradable.

Corporate actions y splits policy:

- Flat Files intradia se tratan como raw/unadjusted salvo prueba contraria.
- Para cada ticker/evento, consultar splits con ventana alrededor de la fecha de
  evento y construir factor de ajuste.
- REST adjusted aggregates pueden usarse como reconciliacion sample.
- Si existe split o reverse split no reconciliado cerca del evento, excluir.
- Dividends ordinarios no se usan para ajustar retornos intradia en V1, pero
  special cash dividends se marcan para diagnostico/exclusion si afectan la
  fecha de evento.
- Ticker changes se auditan con Ticker Events y Ticker Overview; si la identidad
  issuer/ticker no es estable, excluir el evento.

Campos minimos en eventos/audit:

- `timestamp_utc`;
- `timestamp_et`;
- `session`;
- `is_full_regular_session`;
- `expected_1m_bars`;
- `actual_1m_bars`;
- `expected_5m_bars`;
- `actual_5m_bars`;
- `halt_flag`;
- `halt_source`;
- `suspected_halt_or_bad_session`;
- `split_flag`;
- `split_factor`;
- `corporate_action_flag`;
- `corporate_action_source`;

Sample audit requerido:

- Validar 20 eventos de la muestra H3 (`AAPL`, `MSFT`, `NVDA`, `JPM`, `XOM`)
  contra calendario XNYS, entradas `10:00 ET`, close regular y DST.
- Confirmar que half-days se excluyen.
- Consultar halts Nasdaq Trader para esas fechas/tickers y comparar contra gaps
  en OHLCV/quotes.
- Consultar splits/dividends para esas fechas/tickers y reconciliar precios raw
  vs adjusted REST en una muestra.

Decision operativa:

- Timezone/session: `rules_registered_pending_sample_audit`.
- Halts: `exclude_if_halt_or_suspected_bad_session`.
- Corporate actions/splits: `exclude_if_unreconciled`.
- Dataset/backtest: `blocked_until_timezone_session_halt_ca_audit`.

## 9. Rechazo De Consenso Revisado Post-Evento

Decision:

- Regla: hard reject.
- Aplica a: consenso EPS, consenso revenue y cualquier sorpresa calculada por
  vendor.
- Una fuente solo es valida para H3 si demuestra que el consenso usado estaba
  disponible antes del timestamp de earnings.
- Si una consulta historica posterior devuelve consenso revisado post-evento sin
  snapshot pre-evento verificable, la fuente queda rechazada para H3 v1.

Razon:

- H3 depende de sorpresa fundamental. Usar consenso corregido despues del
  release introduce look-ahead directo.
- `eps_surprise` y `revenue_surprise` calculados por vendor pueden ser utiles
  como control, pero no sustituyen actual + consenso snapshot.
- La existencia de `updated`/`last_updated` no basta si no permite reconstruir
  el valor conocido antes del release.

Criterios de aceptacion:

- Para cada evento tradeable, debe existir `consensus_snapshot_at_utc` menor o
  igual que `report_timestamp_utc`.
- Debe guardarse payload raw o record hash del snapshot.
- `eps_consensus` y `revenue_consensus` deben venir del snapshot pre-evento.
- `actual_eps` y `actual_revenue` pueden llegar post-evento, pero no pueden
  cambiar el consenso almacenado.
- Si hay multiples updates antes del evento, usar el ultimo update con
  timestamp anterior al evento.

Criterios de rechazo:

- El proveedor solo entrega estado final o consenso actual sin historial de
  updates.
- El proveedor recalcula/revisa consenso historico y no expone version previa.
- El proveedor no permite distinguir updates pre-evento de updates post-evento.
- `consensus_updated_at_utc` es posterior a `report_timestamp_utc` y no existe
  snapshot anterior.
- La sorpresa vendor no puede recomputarse desde actual y consenso pre-evento.

Politica de datos faltantes:

- Si falta EPS consensus pre-evento, el evento no es tradeable.
- Si falta revenue consensus pre-evento, el evento no es tradeable en H3 v1.
- No se permite imputar revenue consensus con datos posteriores ni usar
  sorpresa calculada por vendor como reemplazo.

Campos minimos:

- `eps_consensus`;
- `revenue_consensus`;
- `consensus_snapshot_at_utc`;
- `consensus_source`;
- `consensus_raw_hash`;
- `consensus_revision_policy`;
- `consensus_snapshot_is_pre_event`;

Decision operativa:

- Fuente sin snapshot pre-evento: `reject_for_h3_v1`.
- Dataset/backtest: `blocked_until_pre_event_consensus_snapshot_verified`.

## Fase 2 Readiness

Generador implementado:

- Modulo: `src/data/earnings_events_h3.py`.
- Test: `tests/test_earnings_events_h3.py`.
- CLI:

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

- Salidas por timing generadas junto al parquet principal:
  `earnings_events_pre_market.parquet`, `earnings_events_after_market.parquet`
  y `earnings_events_excluded_timing.parquet`.
- Elegibilidad de entrada V1 a nivel calendario:
  `entry_timestamp = market_open + 30min` segun calendario XNYS,
  `exit_timestamp = market_close`, y sesiones que no sean regulares completas
  quedan `is_tradeable_v1=false` con flag `non_full_regular_session`.
- La verificacion de barra limpia entre `10:00` y `10:05` queda para el paso de
  panel intradia, porque requiere datos 1min/5min reales.
- Sorpresas fundamentales:
  `eps_surprise_z` y `revenue_surprise_z` se derivan de sorpresa porcentual
  contra consenso pre-evento y se estandarizan contra eventos de sesiones
  anteriores, sin usar eventos futuros ni eventos de la misma sesion.
- Gap normalizado:
  `gap_return = event_session_open / previous_regular_close - 1` y
  `gap_atr = gap_return / recent_atr_return`. `recent_atr_return` es la media
  del true range diario normalizado del mismo simbolo usando solo sesiones
  anteriores. Si falta open, close previo o ATR suficiente, se marcan
  `missing_gap_open`, `missing_gap_prev_close` o `missing_recent_atr` y el
  evento queda no tradeable.
- Volumen relativo:
  `volume_30m` suma el volumen entre `09:30` y `10:00 ET` en el panel intradia
  ajustado. `expected_volume_30m` es la mediana rolling del mismo simbolo usando
  solo sesiones anteriores, y `rel_volume_30m = volume_30m /
  expected_volume_30m`. Si faltan barras, volumen o historia suficiente, se
  marcan `missing_volume_30m`, `missing_expected_volume_30m` o
  `insufficient_opening_30m_bars`, y el evento queda no tradeable.
- VWAP/rango inicial:
  `vwap_30m`, `range_high_30m`, `range_low_30m` y `close_30m` se calculan sobre
  la misma ventana `09:30-10:00 ET`. El VWAP usa `bar_vwap` si existe en el
  panel; si no existe, usa cierre de barra ponderado por volumen como proxy. Si
  falta VWAP, rango, cierre o barras suficientes, se marcan
  `missing_vwap_30m`, `missing_range_30m`, `missing_close_30m` o
  `insufficient_opening_30m_bars`, y el evento queda no tradeable.
- Retorno sectorial/peer:
  `sector_return_30m` y `peer_proxy_return_30m` usan el retorno open-to-close de
  `09:30-10:00 ET` del ETF sectorial definido en
  `equity_earnings_continuation_h3_sector_map_v1.yaml`. V1 usa el ETF sectorial
  como peer proxy. Si el ETF sectorial no esta disponible y SPY si, se usa SPY
  como fallback y se marca `peer_proxy_fallback_used=true`. Si falta mapping o
  no hay retorno de proxy, se marcan `missing_sector_mapping`,
  `missing_sector_proxy_return_30m` o `missing_peer_proxy_return_30m`, y el
  evento queda no tradeable.
- Exclusiones:
  la capa final marca `macro_day_flag`,
  `simultaneous_peer_earnings_flag`, `halt_flag`, `high_spread_flag` y
  `binary_news_flag`. Las fuentes externas son tablas opcionales de sesiones
  macro, halts, quote quality y noticias binarias. Los peers simultaneos se
  detectan dentro del propio dataset cuando hay mas de un simbolo del mismo
  sector en la misma `event_session`. Cualquier flag activo actualiza
  `exclusion_flags` y deja `is_tradeable_v1=false`.

Estado de generacion real:

- Una llamada local minima a `/benzinga/v1/earnings` con `POLYGON_API_KEY`
  devolvio `403 Forbidden`, consistente con falta de entitlement Benzinga
  Earnings.
- Por tanto, `earnings_events.parquet` real queda bloqueado hasta disponer de
  payload raw autorizado o activar el entitlement correspondiente.
- Accion pendiente: comprar/activar `Benzinga Earnings` en Polygon/Massive
  antes de la generacion real. Precio publico observado el 2026-05-15:
  `99 USD/mes` para uso individual; business requiere pricing comercial.
- Bloqueo: si no hay entitlement ni payload raw autorizado, quedan bloqueados
  el dataset real, el sample audit de calendario/consenso y cualquier
  screening/backtest H3 v1. No bloquea el desarrollo del generador, contratos,
  tests ni transformaciones sobre payload autorizado.
- El generador aplica la regla hard reject: si `last_updated`/`updated` no
  demuestra snapshot de consenso anterior a `report_timestamp_utc`, el evento
  queda `is_tradeable_v1=false` y se marca
  `post_event_revised_consensus_without_snapshot`.

## Fase 3 Readiness

Screening implementado:

- Modulo: `src/strategy/equity_earnings_continuation_screening.py`.
- Test: `tests/test_equity_earnings_continuation_screening.py`.
- CLI:

```bash
python -m src.strategy.equity_earnings_continuation_screening \
  --config configs/strategy/equity_earnings_continuation_h3_v1.yaml
```

Artefactos:

- `coverage.parquet`;
- `events.parquet`;
- `trades.parquet`;
- `daily.parquet`;
- `monthly.parquet`;
- `summary.parquet`;
- `distribution.parquet`;
- `manifest.yaml`;
- `report.md`.

Contratos cubiertos:

- `h3_candidate_screen` aplica filtros fijos y ajusta `rel_volume_30m` solo con
  train del fold.
- `earnings_beat_only`, `gap_moderate_only`,
  `intraday_momentum_no_event`, `random_same_frequency_by_ticker`,
  `same_hour_by_ticker` y `sector_proxy_equivalent` quedan registrados como
  controles.
- Se calculan retornos bruto/neto para los horizontes de config y retornos
  residualizados contra proxy sectorial y `SPY`.
- `distribution.parquet` reporta percentiles completos para evitar decidir por
  media agregada.

Bloqueo de interpretacion:

- El screening puede ejecutarse sobre fixtures o payload autorizado, pero sus
  resultados no son evidencia economica hasta regenerar `earnings_events.parquet`
  con `Benzinga Earnings` real, panel intradia real y tablas externas de
  exclusion auditadas.

## Fase 4 Readiness

Runner de estrategia implementado:

- Modulo: `src/strategy/equity_earnings_continuation.py`.
- Test: `tests/test_equity_earnings_continuation_strategy.py`.
- CLI:

```bash
python -m src.strategy.equity_earnings_continuation \
  --config configs/strategy/equity_earnings_continuation_h3_v1.yaml
```

Contratos cubiertos:

- Selecciona solo eventos `h3_v1_primary` que cumplen los filtros H3 v1.
- Entrada: primera barra limpia `>= 10:00 ET` y `<= 10:05 ET`.
- Salida primaria: `same_session_close`.
- Salidas `T+1`: solo sensibilidad bajo `h3_v1_sensitivity`; no se usan para
  seleccion inicial.
- `rel_volume_30m` p60 se ajusta solo con train del fold y queda registrado en
  `thresholds.parquet` y `manifest.yaml`.
- Costes base/conservador/stress se aplican como round-trip bps.
- Retornos residuales contra proxy sectorial y `SPY` quedan en `trades.parquet`,
  `summary.parquet` y `distribution.parquet`.

Bloqueo de interpretacion:

- El backtest H3 v1 real queda bloqueado por el mismo motivo que Fase 2/Fase 3:
  falta activar `Benzinga Earnings` o disponer de payload raw autorizado y
  auditar panel intradia/exclusiones point-in-time.

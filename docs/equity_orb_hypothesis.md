# Equity ORB Hypothesis

Hipotesis hermana de `Options ORB`, pero aplicada primero a ETFs/acciones
liquidas porque ya hay datos intradia disponibles y porque sirve como control
natural antes de pagar datos historicos de opciones.

ORB puro no se trata como edge suficiente. En este research, ORB es el evento
disparador; la hipotesis real es que algunas rupturas del opening range solo
tienen valor cuando expresan liderazgo relativo, confirmacion cross-asset o fallo
claro de la ruptura.

## Estado Actual

Estado: `screened_not_promoted_equity_orb_v1`.

Lineas testadas:

- `H2.2 - ORB por pares/spreads relativos / continuation`.
- `H2.4 - ORB condicionado por calidad del rango`.
- `H2.5 - Failed ORB / reversion`.

Baseline obligatorio: ORB direccional simple sobre el subyacente.

Resultado inicial:

H2.2 continuation:

- Config: `configs/strategy/equity_orb_pairs_v1.yaml`.
- Runner: `python -m src.strategy.equity_orb_pairs --config configs/strategy/equity_orb_pairs_v1.yaml`.
- Reporte: `results/strategy/equity_orb_pairs/5min/report.md`.
- Decision: H2.2 continuation ORB queda rechazado en primera lectura a `2 bps`
  por leg round-trip; todos los pares/ventanas/horizontes son negativos en
  validation y test.

H2.5 failed ORB/reversion:

- Config: `configs/strategy/equity_orb_failed_pairs_v1.yaml`.
- Runner: `python -m src.strategy.equity_orb_failed_pairs --config configs/strategy/equity_orb_failed_pairs_v1.yaml`.
- Reporte: `results/strategy/equity_orb_failed_pairs/5min/report.md`.
- Decision: H2.5 failed ORB/reversion queda rechazado en primera lectura a
  `2 bps` por leg round-trip; todos los pares/ventanas/horizontes son negativos
  en validation y test.
- Mejor validation net a `2 bps`: `-0.0711`.
- Mejor test net a `2 bps`: `-0.0569`.

H2.4 calidad del rango:

- Config: `configs/strategy/equity_orb_range_quality_v1.yaml`.
- Runner: `python -m src.strategy.equity_orb_range_quality --config configs/strategy/equity_orb_range_quality_v1.yaml`.
- Reporte: `results/strategy/equity_orb_range_quality/5min/report.md`.
- Decision: no promovida. Aparece un bolsillo positivo en `XLY/XLP`,
  `orb_15m`, rango ancho `80-100`, pero no supera controles de market beta y
  queda demasiado concentrado.
- Mejor validation net a `2 bps`: `+0.0107`.
- Mejor test net a `2 bps`: `+0.0117`.

## H2 - Equity ORB Conditional Continuation

Hipotesis general:

Cuando un ETF liquido rompe su opening range, la continuacion intradia solo es
explotable si la ruptura representa flujo direccional real y no ruido de apertura.
Ese flujo deberia verse mejor en fuerza relativa, spreads entre activos y
confirmacion cross-asset que en el precio absoluto aislado.

Subyacentes iniciales:

- `SPY`
- `QQQ`
- `IWM`
- `DIA`
- sectores liquidos: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLU`
- credito/rates/havens para contexto: `HYG`, `LQD`, `TLT`, `IEF`, `GLD`

Ventanas ORB iniciales:

- `09:30-09:45`
- `09:30-10:00`

Timeframe inicial:

- `5min` si el dataset intradia disponible esta limpio.
- `15min` como fallback o comparacion con la vertical H1 existente.

## Subhipotesis

### H2.1 - ORB Con Fuerza Relativa

Una ruptura de `QQQ` o `SPY` tiene mas valor si el activo rompe en la misma
direccion en la que ya esta liderando frente a su benchmark.

Ejemplos:

- Long `QQQ` si `QQQ` rompe al alza y `QQQ/SPY` tambien esta fuerte.
- Short `QQQ` si `QQQ` rompe a la baja y `QQQ/SPY` esta debil.
- Long `IWM` si rompe al alza y `IWM/SPY` confirma risk-on breadth.

Uso: filtro direccional sobre ORB absoluto.

### H2.2 - ORB Por Pares/Spreads Relativos

Primera linea testada. Resultado actual: rechazada como continuation ORB despues
de costes.

La senal no se calcula sobre el precio absoluto del ETF, sino sobre un spread o
ratio relativo. La idea es aislar liderazgo/rotacion y reducir beta de mercado.

Pares iniciales:

- `QQQ` vs `SPY`: growth/tech vs mercado amplio.
- `XLK` vs `SPY`: tech sector vs mercado.
- `IWM` vs `SPY`: small caps vs large caps.
- `XLY` vs `XLP`: risk-on consumo discrecional vs defensivo.
- `HYG` vs `LQD`: credito high yield vs investment grade.

Representacion inicial del spread:

```text
spread_log = log(asset_a) - log(asset_b)
```

Versiones posteriores pueden probar beta-neutral:

```text
spread_beta = log(asset_a) - beta_rolling * log(asset_b)
```

pero la primera version debe empezar simple para evitar meter otro grado de
libertad antes de saber si hay senal.

Senales:

- Long relative pair si el spread rompe por encima del high de su opening range.
- Short relative pair si el spread rompe por debajo del low de su opening range.

Implementacion tradable candidata:

- Para `QQQ/SPY`: long `QQQ`, short `SPY` cuando el spread rompe al alza.
- Para `QQQ/SPY`: short `QQQ`, long `SPY` cuando el spread rompe a la baja.
- Misma logica para pares sectoriales.

Sizing inicial:

- Dollar-neutral simple: mismo notional long y short.
- Variante posterior: beta-neutral con beta rolling, solo si dollar-neutral
  muestra algo real.

Objetivo:

- Medir si el ORB relativo captura rotacion intradia mejor que ORB direccional
  simple.
- Separar alpha relativo de beta de mercado.
- Identificar pares donde la ruptura del spread tenga persistencia intradia.

### H2.3 - ORB Con Confirmacion Cross-Asset

La ruptura de un activo solo se opera si el contexto no contradice la direccion.

Confirmaciones candidatas:

- `SPY`, `QQQ`, `IWM`, `DIA` alineados.
- sectores risk-on (`XLK`, `XLY`, `XLF`) acompañan en rupturas alcistas.
- defensivos (`XLP`, `XLU`, `XLV`) no lideran en rupturas risk-on.
- `HYG/LQD` confirma apetito o aversion por riesgo.
- `TLT/SPY` o `GLD/SPY` no contradicen el regimen.

Uso: filtro posterior para H2.1/H2.2 si el baseline relativo tiene senal.

### H2.4 - ORB Condicionado Por Calidad Del Rango

No todos los opening ranges son igual de informativos.

Filtros candidatos:

- rango inicial dentro de percentiles razonables de ATR intradia.
- evitar rangos demasiado estrechos: breakouts faciles de barrer.
- evitar rangos demasiado amplios: movimiento ya consumido.
- exigir expansion de rango/volumen despues de la ruptura.
- medir distancia a VWAP.
- separar rupturas tempranas de rupturas tardias.

Uso: filtro de calidad de setup, no estrategia independiente.

Resultado H2.4:

- Version testada: filtros pre-registrados por percentil del ancho del opening
  range del spread, calculados solo con train por fold/par/ventana.
- Buckets: `20-80`, `30-70`, `0-20`, `80-100`.
- Resultado: no promovido. El unico bolsillo con validation y test positivos es
  `XLY/XLP`, `orb_15m`, bucket `80-100`, horizontes `3/6`; falla porque pierde
  contra market-beta control y su `top5_abs_share` supera ampliamente el gate
  del 50%.

### H2.5 - Failed ORB / Reversion

Algunas rupturas son falsas. La hipotesis alternativa es que la ventaja este en
operar el fallo, no la continuacion.

Senal candidata:

- El activo o spread rompe el rango.
- Vuelve dentro del opening range.
- Se opera reversion hacia VWAP, midpoint del rango o lado opuesto del rango.

Uso: rama alternativa si continuation ORB falla o si aparece edge claro en
rupturas fallidas.

Resultado H2.5:

- Fallo definido como ruptura del spread fuera del opening range seguida de
  cierre de vuelta dentro del rango.
- Implementacion inicial: operar reversion dollar-neutral del spread en el
  primer fallo por sesion.
- Resultado: rechazado a `2 bps` por leg round-trip. Es menos malo que la
  continuation ORB agregada, pero no supera random same-frequency ni market beta
  control y queda negativo en validation y test.

## Baselines Y Controles

Controles obligatorios antes de aceptar cualquier subhipotesis:

- ORB direccional simple sobre cada subyacente.
- Always-flat.
- Random same-frequency por activo/par.
- Same-hour control.
- Same-direction market beta control.
- Buy-and-hold intradia desde hora equivalente hasta salida.

Para H2.2, comparar siempre contra:

- leg `asset_a` direccional.
- leg `asset_b` direccional.
- spread sin filtro ORB.
- spread ORB con costes stress.

## Salidas Iniciales

Salidas candidatas para H2.2:

- cierre tras `N` barras: 2, 3, 4, 6.
- cierre al final de sesion.
- stop si el spread vuelve dentro del opening range.
- stop en el lado opuesto del opening range.
- take profit por multiple del riesgo inicial del spread.

La primera version debe empezar con salida por horizonte fijo y force-flat. Los
stops/take-profits entran despues, si la senal base existe.

## Costes

Costes por leg:

- base: `1-2 bps` por entrada/salida agregada segun convencion local.
- conservador: `3-5 bps`.
- stress: `8-10 bps`.

Para pares/spreads:

- aplicar costes a ambas patas.
- registrar turnover bruto.
- reportar PnL neto por notional total y por gross exposure.

## Metricas De Decision

Metricas primarias:

- PnL neto por trade.
- Avg trade neto.
- Profit factor.
- Win rate.
- Payoff ratio.
- Max drawdown.
- Trades por fold/trimestre.
- PnL por par.
- PnL por hora de entrada.
- Concentracion por dia y semana.
- Sensibilidad a costes.

Para H2.2, anadir:

- PnL long-spread vs short-spread.
- Exposicion neta aproximada a mercado.
- Contribucion por cada leg.
- Correlacion del PnL con retorno de `SPY`.

## Roadmap

### Fase 0 - Especificacion

- [x] Apuntar subhipotesis H2.1-H2.5.
- [x] Seleccionar H2.2 como primera linea activa.
- [x] Convertir H2.2 en contrato YAML versionado.
- [x] Definir pares iniciales exactos.
- [x] Definir artefactos esperados y paths.
- [x] Definir promotion gates iniciales antes de mirar resultados.

### Fase 1 - Dataset Y Eventos

- [x] Validar cobertura intradia de los simbolos requeridos por los pares
  iniciales.
- [x] Construir spreads log para pares iniciales.
- [x] Calcular opening range de cada spread para 15 y 30 minutos.
- [x] Generar eventos de ruptura alcista/bajista por spread.
- [x] Generar eventos ORB direccionales simples para baseline.
- [x] Verificar timezone y sesion regular NY via panel limpio/alineado.

### Fase 2 - Backtest H2.2 Basico

- [x] Implementar backtest dollar-neutral del spread.
- [x] Aplicar costes por ambas patas.
- [x] Probar horizontes `2/3/4/6` barras.
- [x] Reportar resultados por par, lado, ventana, horizonte y fold.
- [x] Comparar contra ORB direccional simple y controles.
- [x] Generar manifest, trades, daily, monthly, summary y report.

### Fase 3 - Sensibilidad

- [ ] Sensibilidad por ventana ORB: 15 vs 30 minutos.
- [x] Sensibilidad por ancho del opening range del spread.
- [ ] Sensibilidad por hora de ruptura.
- [ ] Sensibilidad por regimen de volatilidad diaria/intradia.
- [x] Sensibilidad por costes base/conservador/stress.
- [x] Revisar concentracion de PnL por sesiones.

### Fase 4 - Extension

- [ ] Si H2.2 tiene senal, probar H2.1 como filtro direccional.
- [ ] Si H2.2 tiene senal, probar H2.3 como confirmacion cross-asset.
- [x] Probar H2.4 como filtro de calidad del rango.
- [x] Si continuation falla, probar H2.5 failed ORB.
- [ ] Solo despues de validar equity ORB, conectar con Options ORB.

## Promotion Gates Iniciales

Gates minimos para pasar de research a freeze review:

- PnL neto positivo en validation con costes conservadores.
- PnL neto positivo en test confirmatorio.
- Al menos 4 folds/trimestres con actividad.
- No depender de un unico par.
- Top 5 sesiones no explican mas del 50% del PnL neto.
- Superar ORB direccional simple despues de costes.
- Superar random same-frequency y same-hour controls.
- Mantener PnL razonable en coste stress.
- Correlacion de PnL con `SPY` suficientemente baja para justificar que es una
  hipotesis relativa y no beta encubierta.

## Artefactos Esperados

```text
configs/strategy/equity_orb_pairs_v1.yaml
results/strategy/equity_orb_pairs/5min/manifest.yaml
results/strategy/equity_orb_pairs/5min/trades.parquet
results/strategy/equity_orb_pairs/5min/daily.parquet
results/strategy/equity_orb_pairs/5min/monthly.parquet
results/strategy/equity_orb_pairs/5min/summary.parquet
results/strategy/equity_orb_pairs/5min/report.md
configs/strategy/equity_orb_failed_pairs_v1.yaml
results/strategy/equity_orb_failed_pairs/5min/manifest.yaml
results/strategy/equity_orb_failed_pairs/5min/trades.parquet
results/strategy/equity_orb_failed_pairs/5min/daily.parquet
results/strategy/equity_orb_failed_pairs/5min/monthly.parquet
results/strategy/equity_orb_failed_pairs/5min/summary.parquet
results/strategy/equity_orb_failed_pairs/5min/report.md
configs/strategy/equity_orb_range_quality_v1.yaml
results/strategy/equity_orb_range_quality/5min/manifest.yaml
results/strategy/equity_orb_range_quality/5min/range_quality_thresholds.parquet
results/strategy/equity_orb_range_quality/5min/trades.parquet
results/strategy/equity_orb_range_quality/5min/daily.parquet
results/strategy/equity_orb_range_quality/5min/monthly.parquet
results/strategy/equity_orb_range_quality/5min/summary.parquet
results/strategy/equity_orb_range_quality/5min/report.md
```

## Siguiente Accion

H2.2 continuation, H2.5 failed ORB/reversion y H2.4 range-quality no justifican
promotion. No merece pasar a H2.1/H2.3 como filtros sobre una base negativa.

Decision actual: aparcar la rama ORB equity salvo que se plantee una hipotesis
nueva con mecanismo economico distinto. No conectar con Options ORB desde estos
resultados.

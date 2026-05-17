# Options ORB Hypothesis

Hipotesis independiente para estudiar una estrategia intradia de compra de
opciones de vencimiento corto sobre `SPY` y `QQQ`, basada en Open Range
Breakout (ORB).

Este documento no activa paper trading ni define todavia una fuente de datos
cerrada. La preferencia actual para datos es:

```text
IBKR > Databento > ThetaData
```

La eleccion final se deja para el momento de implementacion, despues de probar
si IBKR devuelve historico suficiente de opciones expiradas y si el coste de
Databento para una descarga acotada es aceptable.

## Hipotesis

Cuando `SPY` o `QQQ` rompen con claridad el rango inicial de la sesion regular,
la continuacion direccional intradia puede capturarse comprando opciones
liquidas de `0-2 DTE`. La convexidad de opciones cercanas a vencimiento deberia
amplificar los movimientos favorables, pero solo sera explotable si el edge
direccional sobrevive a spread, slippage, theta e IV crush.

La estrategia compra opciones y las cierra vendiendo la opcion. El ejercicio no
forma parte de la estrategia, salvo caso operacional excepcional.

## Universo

- Subyacentes: `SPY`, `QQQ`.
- Sesion: regular US equities, `09:30-16:00 America/New_York`.
- Ventanas ORB iniciales:
  - `09:30-09:45`.
  - `09:30-10:00`.
- Direccion:
  - ruptura alcista: comprar call.
  - ruptura bajista: comprar put.
- Vencimientos:
  - `0DTE`.
  - `1DTE`.
  - `2DTE`.
- Tipo de posicion:
  - long premium only.
  - sin venta de opciones descubiertas.
  - sin spreads en la primera version.

## Selector De Contratos

El selector debe priorizar liquidez y ejecutabilidad, no maximizar gamma.

Reglas iniciales:

- Elegir contratos con vencimiento `0-2 DTE`.
- Preferir strike ATM o cercano al precio del subyacente en el momento de
  entrada.
- Si hay Greeks fiables point-in-time, registrar delta, gamma, vega, theta e IV.
- Usar delta solo como descriptor/control de exposicion, no como objetivo de
  optimizacion inicial.
- Rechazar contratos con bid cero.
- Rechazar contratos con spread excesivo:
  - filtro primario candidato: `spread / mid <= 0.15`.
  - probar tambien `0.08` y `0.20` en sensibilidad.
- Rechazar contratos con datos incompletos de bid/ask alrededor de entrada o
  salida.
- Registrar volumen, open interest, bid size y ask size si la fuente los aporta.

Variables a guardar por trade:

- `underlying_symbol`
- `option_symbol`
- `session`
- `dte`
- `right`
- `strike`
- `moneyness`
- `delta_entry`
- `gamma_entry`
- `vega_entry`
- `theta_entry`
- `iv_entry`
- `iv_exit`
- `spread_entry`
- `spread_pct_entry`
- `bid_entry`
- `ask_entry`
- `mid_entry`
- `bid_exit`
- `ask_exit`
- `mid_exit`

## Senales

Variantes a testear sin mezclar optimizacion y confirmacion:

1. ORB limpio:
   - long call si el subyacente rompe por encima del high del opening range.
   - long put si rompe por debajo del low del opening range.
2. ORB con confirmacion:
   - requiere cierre de vela de 1 minuto fuera del rango.
3. ORB con filtro de rango:
   - operar solo si el ancho del opening range esta dentro de percentiles
     razonables frente a ATR intradia o volatilidad realizada reciente.

Reglas anti-leakage:

- La entrada se calcula con datos disponibles en el instante de ruptura.
- La ejecucion se simula en el siguiente precio ejecutable, no en el precio que
  dispara la senal si eso introduce lookahead.
- Los datos diarios externos, si se usan, deben estar disponibles antes de la
  sesion o asignados point-in-time.

## Salidas

Salidas candidatas:

- Stop por invalidacion del breakout en el subyacente.
- Stop por perdida de prima de la opcion.
- Take profit por multiple de riesgo.
- Time stop si no hay continuacion despues de N minutos.
- Cierre forzoso antes del final de sesion.

Restricciones:

- No mantener `0DTE` overnight.
- No mantener opciones abiertas hasta ejercicio automatico como conducta normal.
- Vender la opcion para cerrar.

## Costes Y Slippage

El backtest no debe usar mid como ejecucion base.

Modelo primario:

- Entrada long option: comprar en `ask`.
- Salida long option: vender en `bid`.
- Comision configurable por contrato.
- Rechazo de trades con spread excesivo.

Sensibilidad:

- Base: `buy=ask`, `sell=bid`.
- Conservador: `buy=ask + 25% spread`, `sell=bid - 25% spread`.
- Stress: `buy=ask + 50% spread`, `sell=bid - 50% spread`.

Si solo hay trades y no hay bid/ask historico, el resultado queda marcado como
screening, no como backtest ejecutable.

## IV, Vega Y Gamma

La estrategia no debe seleccionar por gamma maxima en la primera version.
`0-2 DTE` ya introduce gamma alta por construccion, y perseguir la gamma maxima
puede acabar eligiendo contratos muy ruidosos, caros de cruzar o demasiado OTM.

Analisis requerido:

- Buckets de gamma de entrada.
- Buckets de vega de entrada.
- Buckets de IV de entrada.
- Cambio de IV entre entrada y salida.
- Relacion entre PnL y compresion/expansion de IV.
- Separar PnL direccional aproximado de PnL explicado por IV/theta cuando sea
  razonable.

Objetivo: saber si la estrategia gana por direccion y convexidad, o si depende
de comprar IV en momentos favorables de forma fragil.

## Datos

Preferencia actual:

1. IBKR
   - Mejor para live, paper y empezar a grabar bid/ask desde ahora.
   - Muy barato si la cuenta esta como non-professional.
   - Riesgo principal: historico de opciones expiradas puede no estar disponible
     o no ser suficiente para un backtest de 1-2 anos.
2. Databento
   - Candidato para descarga historica acotada.
   - Ventaja: OPRA historico y pricing por uso.
   - Riesgo: coste final depende de GB descargados; hay que estimarlo antes.
3. ThetaData
   - Fallback si necesitamos una tarifa plana mensual para research de opciones.
   - Ventaja: facil de razonar por plan mensual.
   - Riesgo: mas caro que IBKR y potencialmente mas de lo necesario si el scope
     se mantiene muy pequeno.

Scope de datos para reducir coste:

- No descargar cadenas completas historicas.
- Primero detectar senales ORB con datos del subyacente.
- Solo despues pedir contratos `0-2 DTE` alrededor del ATM.
- Limitar a `SPY` y `QQQ`.
- Empezar con los ultimos 24 meses, no 10 anos.
- Reportar resultados por trimestre para detectar cambios de regimen.

## Metricas De Decision

Metricas primarias:

- PnL neto por contrato.
- Avg trade neto.
- Profit factor.
- Win rate.
- Payoff ratio.
- Max drawdown.
- Sharpe diario, solo como metrica secundaria.
- Trades por fold/trimestre.
- Concentracion por dia y por semana.
- Sensibilidad a costes y slippage.

Comparaciones obligatorias:

- ORB sobre subyacente sin opciones.
- Posicion equivalente por delta en el subyacente.
- Breakout sin filtro de liquidez de opciones.
- Random same-frequency control.
- Same-hour control.

La estrategia solo merece paper si las opciones superan al proxy direccional
equivalente despues de bid/ask, slippage y comisiones.

## Roadmap

### Fase 0 - Especificacion

- [x] Escribir hipotesis y alcance inicial.
- [x] Definir prioridad de fuentes de datos: `IBKR > Databento > ThetaData`.
- [ ] Convertir esta hipotesis en un contrato YAML versionado.
- [ ] Definir nombres de artefactos y paths de salida.
- [ ] Definir promotion gates antes de mirar resultados.

### Fase 1 - Data probe

- [ ] Probar IBKR para cadenas actuales de `SPY` y `QQQ`.
- [ ] Probar IBKR para bid/ask historico de una opcion expirada concreta.
- [ ] Probar IBKR para historical bars `BID`, `ASK`, `BID_ASK` y `TRADES` en
  opcion activa y opcion expirada.
- [ ] Probar captura live de quotes OPRA L1 via IBKR paper.
- [ ] Estimar coste de Databento para una muestra acotada:
  `SPY/QQQ`, `0-2 DTE`, strikes ATM +/- pocos strikes, ultimos 24 meses.
- [ ] Decidir fuente historica solo despues de la prueba anterior.

### Fase 2 - Dataset minimo

- [ ] Preparar datos intradia de subyacente para `SPY` y `QQQ`.
- [ ] Calcular opening ranges de 15 y 30 minutos.
- [ ] Generar eventos ORB por sesion y subyacente.
- [ ] Construir selector de contratos elegibles en cada evento.
- [ ] Guardar quotes/trades de opciones solo para contratos candidatos.
- [ ] Calcular IV y Greeks si la fuente no los aporta point-in-time.
- [ ] Validar timestamps, timezone y disponibilidad point-in-time.

### Fase 3 - Backtest

- [ ] Implementar backtest ORB sobre subyacente como control.
- [ ] Implementar backtest long option con bid/ask real.
- [ ] Implementar costes base/conservador/stress.
- [ ] Implementar salidas: stop, take profit, time stop y force-flat.
- [ ] Generar trades, daily, monthly, summary y manifest.
- [ ] Separar resultados por `SPY/QQQ`, call/put, DTE, ventana ORB y trimestre.

### Fase 4 - Sensibilidad

- [ ] Analizar sensibilidad por DTE.
- [ ] Analizar sensibilidad por moneyness/delta descriptiva.
- [ ] Analizar sensibilidad por gamma.
- [ ] Analizar sensibilidad por vega.
- [ ] Analizar sensibilidad por IV entry e IV change.
- [ ] Analizar sensibilidad por spread y liquidez.
- [ ] Analizar concentracion por dia, hora, semana y eventos extremos.

### Fase 5 - Decision

- [ ] Rechazar si no supera controles sobre el subyacente.
- [ ] Rechazar si el edge desaparece con coste conservador.
- [ ] Rechazar si PnL depende de muy pocos dias.
- [ ] Rechazar si solo funciona en un DTE o bucket demasiado pequeno.
- [ ] Si pasa gates, preparar paper trading read-only/signal-only.
- [ ] Solo despues de paper signal-only, evaluar ejecucion real en paper.

## Promotion Gates Iniciales

Gates minimos antes de paper:

- Al menos 80 trades totales en validation/test combinados.
- Al menos 4 trimestres con actividad.
- PnL neto positivo con coste conservador.
- Profit factor > 1.10 con coste conservador.
- Max drawdown aceptable frente al capital por contrato definido.
- Top 5 dias no explican mas del 50% del PnL neto.
- Opciones superan al delta-equivalent del subyacente despues de costes.
- No hay dependencia exclusiva de `0DTE`.
- No hay dependencia exclusiva de spreads anormalmente estrechos.
- Resultados reproducibles desde manifest y config.

## Artefactos Esperados

```text
configs/strategy/options_orb_v1.yaml
results/strategy/options_orb/SPY/1min/manifest.yaml
results/strategy/options_orb/SPY/1min/trades.parquet
results/strategy/options_orb/SPY/1min/daily.parquet
results/strategy/options_orb/SPY/1min/monthly.parquet
results/strategy/options_orb/SPY/1min/summary.parquet
results/strategy/options_orb/SPY/1min/report.md
results/strategy/options_orb/QQQ/1min/manifest.yaml
results/strategy/options_orb/QQQ/1min/trades.parquet
results/strategy/options_orb/QQQ/1min/daily.parquet
results/strategy/options_orb/QQQ/1min/monthly.parquet
results/strategy/options_orb/QQQ/1min/summary.parquet
results/strategy/options_orb/QQQ/1min/report.md
```

## Decision Actual

Estado: `research_spec`.

Siguiente accion concreta: implementar una prueba pequena de disponibilidad de
datos con IBKR antes de comprar datos historicos externos.

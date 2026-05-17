# Estrategias de trading cuantitativo de corto plazo

**Rol:** investigador cuantitativo especializado en estrategias sistemáticas de corto plazo para acciones, índices y FX.  
**Horizonte:** intradía a máximo 5 sesiones.  
**Universos:** acciones líquidas de EE. UU. y Europa, ETFs, futuros de índices, FX líquido, baskets long/short y pares relativos.  

> Este documento es un research memo. No constituye recomendación de inversión. Todas las ideas deben evaluarse netas de costes, con datos point-in-time y con backtests event-driven realistas.

---

## Resumen ejecutivo

Se proponen cinco estrategias, priorizando aquellas con hipótesis económica clara y posibilidad razonable de validación empírica:

1. **Earnings continuation intradía condicionado**  
2. **Reversión de gap fundamental no confirmado por sector ni peers**  
3. **Read-through de earnings de líder sectorial hacia peers**  
4. **Cascade de revisiones de analistas post-earnings**  
5. **Shock FX/tipos sobre compañías con exposición identificable**  

Las tres estrategias más atractivas para comenzar serían:

1. **Earnings continuation intradía condicionado**: mejor equilibrio entre claridad, frecuencia y testeabilidad.  
2. **Read-through de líder sectorial hacia peers**: hipótesis económica fuerte, aunque exige buen mapping sectorial.  
3. **Cascade de revisiones de analistas**: más dependiente de vendors, pero potencialmente escalable.

Base empírica general:

- El **post-earnings-announcement drift** se ha documentado como posible reacción retardada a resultados.
- Las revisiones de analistas pueden acelerar o modificar la incorporación de información.
- Los cambios de recomendación de analistas muestran, en ciertos contextos, reacción inicial incompleta.
- Existe evidencia de transferencia de información entre compañías del mismo sector alrededor de earnings.

---

# 1. Earnings continuation intradía condicionado

Esta es la versión formal de la idea:

> Después de earnings con sorpresa positiva fuerte, gap moderado, volumen alto y sector confirmando, se compra a los 30 minutos y se cierra al cierre o al día siguiente.

---

## 1.1 Hipótesis económica

La hipótesis es que el mercado **incorpora lentamente información fundamental positiva** cuando la sorpresa de resultados es clara, pero el gap inicial no ha agotado la reacción.

El edge no viene del momentum puro, sino de una combinación de:

- sorpresa de EPS, ventas o guidance;
- absorción de oferta en la apertura;
- volumen anormal que confirma atención institucional;
- sector y peers validando que la noticia no es puramente idiosincrática;
- actualización progresiva de expectativas durante la sesión.

La condición crítica es que el gap sea **moderado**, no extremo. Un gap demasiado grande puede implicar sobre-reacción o mala relación payoff/riesgo.

---

## 1.2 Universo

Acciones líquidas de EE. UU. y Europa con earnings pre-market o after-market.

Criterios iniciales:

- market cap superior a 2–5 bn USD/EUR;
- ADV superior a 20–50 mn USD/EUR;
- spread medio inferior a 10–20 bps;
- precio superior a 5 USD/EUR;
- cobertura mínima de analistas;
- excluir small caps, biotech binario, SPACs, acciones con borrow problemático;
- sectores preferentes: software, semiconductores, retail, consumo discrecional, industriales, bancos y lujo europeo.

Para EE. UU., puede cubrirse con acciones del S&P 500, Nasdaq 100 y Russell 1000 líquido. Para Europa, STOXX Europe 600 líquido, EuroStoxx 50 y grandes mid caps.

---

## 1.3 Evento o señal

Definir la sorpresa de EPS como:

```text
S_EPS = (EPS_reportado - EPS_consenso) / sigma(EPS_surprises)
```

O alternativamente:

```text
S_EPS = (EPS_reportado - EPS_consenso) / |EPS_consenso|
```

Señal long si se cumplen simultáneamente:

- sorpresa de EPS en percentil alto de su histórico;
- sorpresa de ingresos positiva o no negativa;
- guidance positivo o ausencia de recorte;
- gap overnight positivo entre, por ejemplo, 0.5 y 2.5 desviaciones estándar de la volatilidad reciente;
- volumen relativo en los primeros 30 minutos superior al volumen esperado para esa franja;
- retorno sectorial positivo o neutral;
- peer basket con retorno positivo o, al menos, no contrario;
- precio a los 30 minutos por encima del VWAP o del punto medio del rango inicial;
- excluir si la call revela deterioro de márgenes, one-off gains o contabilidad dudosa.

Los umbrales concretos deben optimizarse con walk-forward, no fijarse ex ante como verdad empírica.

---

## 1.4 Reglas operativas

### Entrada

- EE. UU.: 10:00 ET, es decir, 30 minutos después de la apertura.
- Europa: 30 minutos después de la apertura local.
- Precio de referencia: último precio negociado o midquote, no close anterior.
- Confirmación mínima: precio por encima del VWAP de 30 minutos y sector no cayendo.

### Salida

- Salida principal: cierre de la misma sesión mediante MOC/LOC.
- Alternativa: mantener hasta T+1 10:30–12:00 si el cierre es fuerte y no hay reversión overnight.
- Salida por señal: pérdida del VWAP con sector girándose negativo.
- Stop: por debajo del mínimo del rango inicial o pérdida de 1–1.5 veces la volatilidad intradía esperada.
- Take profit opcional: solo si el movimiento alcanza una extensión extrema respecto a volatilidad intradía.

### Tamaño de posición

- Posición proporcional a volatilidad inversa.
- Riesgo máximo por trade definido por stop, no por nominal.
- Máximo 5–15 posiciones simultáneas.
- Beta hedge opcional con sector ETF o índice futuro.
- Variante preferida: long stock, short sector ETF en beta equivalente si se quiere aislar alpha idiosincrático.

### Tipo

- Long-only para empezar.
- Market-neutral en versión institucional.

---

## 1.5 Distribución y probabilidades

Para cada trade, medir retorno neto:

```text
r_i_net = r_i_stock - beta_i * r_i_sector/index - costes_i
```

Estimar:

```text
P(r > 0 | señal) = (1/N) * sum(1(r_i_net > 0))
```

```text
E[r | señal] = (1/N) * sum(r_i_net)
```

También:

- media y mediana;
- percentiles 5/25/75/95;
- hit rate;
- payoff ratio;
- volatilidad condicional;
- MAE y MFE usando trayectoria intradía;
- drawdown esperado mediante bootstrap por día de evento;
- intervalo de confianza por bootstrap clusterizado por fecha;
- sensibilidad a costes: repetir con 1x, 2x y 3x spread + slippage.

Payoff ratio:

```text
Payoff ratio = E[r | r > 0] / |E[r | r < 0]|
```

No conviene fijar un tamaño muestral arbitrario. Debe hacerse power analysis. Como ejemplo hipotético: si la desviación estándar neta por trade fuera 80 bps y se quisiera detectar una media de 10 bps con intervalo razonable, se necesitarían varios cientos de trades independientes. Ese cálculo debe hacerse con la volatilidad real observada.

---

## 1.6 Backtest realista

Datos necesarios:

- intradía trade/quote o barras de 1 minuto;
- calendario de earnings con timestamp real;
- consenso disponible antes del anuncio;
- EPS, ventas, guidance y revisiones;
- clasificación sectorial;
- sector ETF o futuro;
- bid/ask;
- datos de subasta de cierre si se opera MOC;
- flags de halted stocks.

Errores a evitar:

- usar consenso actualizado después del evento;
- timestamp incorrecto de earnings after-market/pre-market;
- survivorship bias;
- operar acciones que no eran líquidas en la fecha;
- ignorar spreads en nombres europeos;
- usar close oficial cuando la orden real habría ejecutado en auction;
- ignorar latencia de parsing de resultados;
- no controlar earnings simultáneos de peers.

Train/test:

- entrenar umbrales por periodos;
- test fuera de muestra por año;
- walk-forward por trimestre;
- separar regímenes de alta/baja volatilidad;
- analizar earnings season por separado.

---

## 1.7 Robustez

Pruebas:

- variar entrada: 15, 30, 45 y 60 minutos;
- variar salida: cierre, T+1 open, T+1 close;
- usar gap moderado vs gap extremo;
- separar EPS surprise de revenue surprise;
- exigir o no confirmación sectorial;
- excluir mega caps;
- excluir días FOMC, CPI, NFP;
- bootstrap por día para evitar concentración en una earnings season;
- repetir en EE. UU. y Europa por separado.

---

## 1.8 Riesgos

Principales riesgos:

- earnings ya completamente descontados;
- titulares positivos pero guidance débil;
- short-term crowding;
- reversión tras la conference call;
- errores de parsing;
- spreads amplios en la apertura;
- halts;
- noticia simultánea del sector;
- cambio de régimen en reacción a earnings;
- dependencia del vendor de consenso.

---

## 1.9 Implementación práctica

Infraestructura mínima:

- feed intradía de precios y quotes;
- vendor de earnings con timestamps reales;
- consenso pre-evento;
- NLP o reglas para guidance;
- motor de señales intradía;
- ejecución algorítmica VWAP/TWAP o smart order router;
- módulo de hedging sectorial;
- control de riesgo en tiempo real;
- logging completo de señal, timestamp, orden, fill y coste.

Esta estrategia es probablemente la **más testeable** del conjunto.

---

# 2. Reversión de gap fundamental no confirmado por sector ni peers

---

## 2.1 Hipótesis económica

Explota **sobrerreacción inicial** y presión de liquidez en la apertura. Tras una noticia fundamental, el mercado puede vender o comprar agresivamente antes de distinguir entre:

- impacto económico real;
- titular ambiguo;
- noticia ya descontada;
- read-through inexistente;
- flujo forzado de apertura.

La señal no es “gap reversal” técnico. La clave es que el movimiento de la acción sea grande, pero **no confirmado por el sector, los peers ni variables relacionadas**.

---

## 2.2 Universo

Acciones líquidas de EE. UU. y Europa:

- large caps y mid caps líquidas;
- sectores con peers comparables: bancos, semiconductores, autos, aerolíneas, lujo, energía, software y retail;
- excluir compañías sin comparables claros;
- evitar eventos binarios: litigios, FDA, fraude, M&A, profit warning muy severo;
- preferible operar con hedge sectorial.

---

## 2.3 Evento o señal

Definir:

```text
Gap_i = (Open_i - Close_{i,-1}) / Close_{i,-1}
```

Y retorno residual:

```text
Residual_i = r_i - beta_sector * r_sector - beta_peer * r_peer
```

Señal contraria si el gap es extremo frente a la distribución reciente, pero no existe confirmación externa.

### Ejemplo de long reversal

- Acción abre -4%.
- Sector cae solo -0.5%.
- Peer basket cae menos de -1%.
- Noticia negativa pero no cambia guidance agregado.
- Durante los primeros 30–60 minutos la acción deja de hacer nuevos mínimos.
- Recupera VWAP o punto medio del rango inicial.

### Ejemplo de short reversal

- Acción abre +5%.
- Sector plano.
- Peers no reaccionan.
- Noticia es una subida de recomendación o titular no fundamental.
- Volumen inicial alto pero momentum se agota.

---

## 2.4 Reglas operativas

### Entrada

- No entrar en la apertura.
- Esperar 30–60 minutos.
- Long si recupera VWAP o rompe el máximo del rango de estabilización.
- Short si pierde VWAP tras gap positivo no confirmado.
- Precio de referencia: midquote o VWAP de ejecución.

### Salida

- Salida primaria al cierre.
- Salida alternativa T+1 open si la reversión no se completa pero la tesis sigue válida.
- Stop: mínimo/máximo del rango inicial.
- Take profit: retorno al 50–75% del gap o al VWAP diario.
- Cancelar operación si aparece noticia adicional confirmatoria.

### Tamaño

- Menor que en estrategia #1, porque el riesgo de headline es mayor.
- Beta-neutral contra sector ETF.
- Máximo 5–10 posiciones.
- No operar si borrow es caro o no disponible en shorts.

---

## 2.5 Distribución y probabilidades

Separar por tipo de noticia:

- earnings;
- guidance;
- downgrade;
- litigation;
- M&A rumor;
- macro shock;
- sector read-through.

Métricas:

```text
P(r_net > 0 | gap no confirmado)
```

```text
E[r_net | gap no confirmado]
```

Además:

- percentiles 5/25/75/95;
- MAE;
- MFE;
- ratio MFE/MAE;
- probabilidad de tocar stop antes de target;
- skewness;
- pérdida media condicional en el peor 5%.

MAE es especialmente importante: una estrategia de reversión puede tener buena media pero colas negativas severas.

Intervalos de confianza: bootstrap por fecha y por ticker, porque la misma acción puede aportar muchos eventos correlacionados.

---

## 2.6 Backtest realista

Datos necesarios:

- noticias con timestamp;
- clasificación de noticia;
- intradía 1 minuto o tick;
- sector ETF;
- peer basket;
- bid/ask;
- short availability;
- registro de halts;
- eventos corporativos.

Errores críticos:

- clasificar manualmente noticias después de ver el resultado;
- usar peers que no estaban en el universo en esa fecha;
- ignorar que algunas noticias salen antes de la apertura y otras durante la sesión;
- asumir fills al VWAP sin modelar liquidez;
- no excluir eventos verdaderamente estructurales.

---

## 2.7 Robustez

Pruebas:

- entrada 30/45/60/90 minutos;
- gaps de 1.5, 2, 2.5 y 3 desviaciones estándar;
- peer confirmation estricta vs laxa;
- solo long, solo short y long/short;
- por sector;
- por régimen VIX alto/bajo;
- con costes duplicados;
- eliminando los mejores y peores 1% de trades.

---

## 2.8 Riesgos

Riesgos altos:

- confundir falta de confirmación con información idiosincrática real;
- gaps que continúan;
- titulares incompletos;
- short squeeze;
- downgrade con ventas institucionales persistentes;
- eventos legales o regulatorios;
- slippage fuerte en apertura;
- baja muestra si se filtra correctamente.

---

## 2.9 Implementación práctica

Se necesita:

- motor de detección de gaps;
- news classifier;
- construcción automática de peer baskets;
- cálculo intradía de residual vs sector;
- integración con locate/borrow;
- reglas de kill-switch si aparece segunda noticia;
- ejecución pasiva/agresiva según liquidez.

Esta estrategia puede tener edge, pero es más frágil que la #1 porque depende mucho de clasificar bien la noticia.

---

# 3. Read-through de earnings de líder sectorial hacia peers

---

## 3.1 Hipótesis económica

Una compañía líder revela información sobre demanda, precios, márgenes, inventarios o costes que afecta a sus peers. El mercado puede reaccionar rápido en el líder, pero más lentamente en compañías relacionadas.

Ejemplos:

- NVIDIA → semiconductores / AI supply chain;
- ASML → semis europeos;
- LVMH → lujo europeo;
- JPMorgan → bancos;
- Delta / United → aerolíneas;
- Walmart / Target → retail;
- Caterpillar → industriales / ciclo global.

La hipótesis es de **transferencia de información intra-sectorial**.

---

## 3.2 Universo

Baskets de peers líquidos:

- mínimo 5–20 compañías por grupo;
- líder con alta cuota de mercado o alta relevancia informacional;
- peers con volumen suficiente;
- excluir peers que reportan earnings en las siguientes 24–48 horas;
- construir grupos por industria económica real, no solo GICS.

Posibles universos:

- semiconductores: NVDA, AMD, AVGO, MU, ASML, TSM ADR, SOXX;
- lujo: LVMH, Hermès, Kering, Richemont, Moncler;
- bancos EE. UU.: JPM, BAC, C, WFC, GS, MS;
- bancos Europa: BNP, SAN, BBVA, ING, UBS;
- aerolíneas: DAL, UAL, AAL, LHA, IAG, AF-KLM;
- energía: XOM, CVX, SHEL, BP, TotalEnergies.

---

## 3.3 Evento o señal

Evento:

- líder reporta earnings/guidance;
- sorpresa positiva o negativa significativa;
- retorno anormal del líder en los primeros 30–60 minutos;
- volumen relativo alto.

Señal long peers:

```text
Signal = Surprise_leader > 0
```

Y:

```text
r_peers_0-60m < lambda * r_leader_0-60m
```

donde `lambda` se estima históricamente.

La idea es comprar peers si han reaccionado menos de lo esperado.

Señal short peers:

- líder cae con sorpresa negativa;
- peers apenas caen;
- sector ETF empieza a confirmar;
- no hay noticia propia positiva en peers.

---

## 3.4 Reglas operativas

### Entrada

- 30–90 minutos después de la apertura si el líder reportó pre-market.
- Si reportó after-market, entrada en peers durante la siguiente apertura tras confirmar reacción.
- Evitar entrar si todos los peers ya han ajustado.
- Usar basket ponderado por liquidez y exposición histórica al líder.

### Salida

- Cierre del día.
- T+1 o T+2 si la señal es fuerte y los peers siguen retrasados.
- Cerrar antes del earnings propio del peer.
- Stop por basket: pérdida de 1–1.5 veces volatilidad esperada.
- Salida por invalidación si líder revierte completamente o sector gira.

### Tamaño

- Long/short sector-neutral.
- El líder puede no operarse; el edge está en peers.
- Hedge con sector ETF.
- Máximo 3–5 temas sectoriales simultáneos.
- Limitar concentración por industria.

---

## 3.5 Distribución y probabilidades

Medir retornos netos del basket:

```text
r_basket_net = sum_j(w_j * r_j) - beta * r_sector - costes
```

Estimar:

- `P(r_basket_net > 0 | leader surprise)`;
- esperanza por evento;
- mediana;
- percentiles;
- hit rate;
- payoff ratio;
- volatilidad condicional;
- MAE/MFE por basket;
- drawdown por cluster sectorial.

Separar buckets:

- líder mega cap vs mid cap;
- sorpresa positiva vs negativa;
- alta vs baja concentración del sector;
- peers que reportan pronto vs no;
- EE. UU. vs Europa;
- pre-market vs after-market.

El problema aquí no es solo el número de eventos, sino la independencia. Muchos eventos del mismo líder no son independientes. Usaría bootstrap por líder-sector-fecha.

---

## 3.6 Backtest realista

Datos necesarios:

- earnings del líder;
- consenso;
- timestamps;
- intradía de líder y peers;
- mapping económico de peers;
- sector ETFs;
- earnings calendar de peers;
- corporate actions;
- liquidez y spreads.

Errores a evitar:

- elegir peers después de mirar resultados;
- incluir compañías que no eran comparables en esa fecha;
- ignorar earnings simultáneos;
- no distinguir información común de shock competitivo;
- asumir que buena noticia para el líder siempre es buena para peers. En algunos sectores puede ser negativa si implica ganancia de cuota.

---

## 3.7 Robustez

Pruebas:

- diferentes definiciones de peer basket;
- excluir el peer más correlacionado;
- excluir mega caps;
- probar 30/60/90/120 minutos;
- horizonte intradía, T+1, T+2, T+5;
- variación de hedge sectorial;
- análisis por tipo de sector;
- control por mercado global;
- test fuera de muestra por industria.

---

## 3.8 Riesgos

Principales riesgos:

- read-through competitivo mal interpretado;
- sector ya ajustado por ETF/algo trading;
- señal muy crowded en mega caps;
- baja frecuencia;
- earnings simultáneos;
- relación económica cambiante;
- clasificación sectorial pobre;
- alta dependencia de vendor de eventos.

---

## 3.9 Implementación práctica

Se requiere:

- taxonomía propia de relaciones económicas;
- motor de eventos earnings;
- cálculo de sorpresa y retorno anormal del líder;
- generación automática de peer basket;
- ejecución basket;
- hedge sectorial;
- monitor de eventos de peers;
- control de exposición temática.

Es una de las estrategias más interesantes, pero exige mucho cuidado en la construcción de relaciones económicas.

---

# 4. Cascade de revisiones de analistas post-earnings

---

## 4.1 Hipótesis económica

Tras earnings o guidance, los analistas no actualizan todos al mismo tiempo. Las primeras revisiones relevantes pueden anticipar una **cascada de cambios de estimaciones**, targets o recomendaciones.

La hipótesis es de **revisión lenta de expectativas**, no de momentum técnico.

---

## 4.2 Universo

Acciones con alta cobertura de analistas:

- large/mid caps de EE. UU. y Europa;
- mínimo 8–10 analistas cubriendo;
- alta liquidez;
- excluir compañías con cobertura escasa;
- excluir microcaps;
- preferir sectores con sensibilidad a estimates: software, semis, bancos, retail, lujo, industriales y salud no binaria.

---

## 4.3 Evento o señal

Señal long:

- earnings/guidance positivo;
- varios analistas revisan EPS, revenue, target price o rating al alza en las primeras 24–48 horas;
- revisiones no son solo target price por expansión de múltiplo, sino estimates reales;
- precio no ha hecho gap extremo;
- sector no contradice.

Score posible:

```text
RevisionScore_i =
    z(Delta EPS_FY1)
  + z(Delta EPS_FY2)
  + z(Delta Revenue_FY1)
  + z(Delta TargetPrice)
  + RatingChangeScore
```

Señal short:

- recortes coordinados de EPS/guidance;
- downgrades relevantes;
- recorte de target con reducción de estimates;
- precio aún no ha ajustado completamente.

---

## 4.4 Reglas operativas

### Entrada

- No operar el primer titular de un solo broker salvo que sea broker muy influyente.
- Entrar tras confirmarse un mínimo de revisiones independientes.
- Entrada posible al cierre de T, apertura de T+1 o intradía cuando el vendor publique revisión.
- Precio de referencia: midquote tras timestamp de revisión.

### Salida

- Holding 2–5 sesiones.
- Salida por tiempo.
- Salida si aparece revisión contraria.
- Salida si el precio alcanza un movimiento extremo frente a sector.
- Stop por retorno residual vs sector.
- No mantener a través de un evento binario nuevo.

### Tamaño

- Long/short.
- Preferible market-neutral por sector.
- Posiciones ponderadas por liquidez y fuerza del score.
- Máximo 20–50 posiciones si la estrategia escala como basket.
- Limitar exposición a una sola earnings season.

---

## 4.5 Distribución y probabilidades

Medir:

```text
r_i_net = s_i * (r_i - beta_sector * r_sector) - costes
```

Donde:

```text
s_i = +1 para revisiones positivas
s_i = -1 para revisiones negativas
```

Estimar:

- `P(r_net > 0 | RevisionScore > q)`;
- media;
- mediana;
- percentiles;
- hit rate;
- payoff ratio;
- volatilidad condicional;
- MAE/MFE por ventana de 1, 2, 3 y 5 días;
- drawdown por basket;
- sensibilidad a retraso de ejecución.

Buckets importantes:

- revisión de EPS vs solo target price;
- upgrade/downgrade vs estimate revision;
- número de analistas;
- dispersión previa de estimates;
- compañías con guidance vs sin guidance;
- sectores.

El coste informacional importa: si el vendor entrega la revisión tarde, el edge puede desaparecer. Por tanto, conviene medir performance con retrasos artificiales de 5, 15, 30, 60 minutos y T+1.

---

## 4.6 Backtest realista

Datos necesarios:

- historial point-in-time de estimates;
- timestamps reales de cada revisión;
- identificador de broker;
- rating changes;
- target price changes;
- earnings calendar;
- precios intradía;
- bid/ask;
- short availability;
- corporate actions.

Errores críticos:

- usar consenso final del día en vez del consenso disponible en el momento;
- ignorar revisiones publicadas antes de mercado;
- contar reiteraciones como revisiones;
- no distinguir cambios de estimates de cambios de múltiplo;
- no controlar que muchas revisiones vienen del mismo evento de earnings.

---

## 4.7 Robustez

Pruebas:

- score con y sin target price;
- solo EPS FY1/FY2;
- solo revisiones de brokers top-tier;
- entrada inmediata vs T+1;
- holding 1/2/3/5 días;
- excluir eventos con gap extremo;
- costes duplicados;
- bootstrap por compañía y fecha;
- test fuera de muestra por año.

---

## 4.8 Riesgos

Riesgos:

- vendors caros y con latencia variable;
- señal crowded;
- revisiones mecánicas sin información nueva;
- baja independencia de eventos;
- sesgo de publicación;
- analistas optimistas de forma estructural;
- impacto fuerte de costes si se opera demasiado rápido;
- short constraints en señales negativas.

---

## 4.9 Implementación práctica

Necesario:

- vendor point-in-time de estimates;
- ingestión intradía de revisiones;
- normalización por broker y tipo de revisión;
- motor de scoring;
- ejecución basket;
- hedge sectorial;
- monitor de eventos pendientes;
- control de latencia.

Esta estrategia es menos intradía pura que la #1, pero puede ser más escalable como basket long/short.

---

# 5. Shock FX/tipos sobre compañías con exposición identificable

---

## 5.1 Hipótesis económica

Movimientos abruptos de FX o tipos afectan a compañías con exposiciones económicas claras:

- exportadores;
- importadores;
- compañías con ingresos en USD y costes en EUR;
- aerolíneas con costes de combustible USD;
- lujo europeo sensible a USD/CNY;
- industriales europeos con ventas globales;
- bancos sensibles a curvas de tipos;
- utilities/REITs sensibles a yields.

La hipótesis es que el mercado primero mueve índice/sector de forma genérica y luego discrimina entre ganadores y perdedores por exposición.

La exposición cambiaria a nivel firma es difícil de estimar y varía en el tiempo. Por eso debe medirse con cautela mediante datos fundamentales point-in-time y betas históricas.

---

## 5.2 Universo

Baskets relativos dentro de sectores:

- Europa: lujo, autos, industriales, semiconductores, aerolíneas, turismo;
- EE. UU.: multinationals con alta foreign revenue vs domésticas;
- Japón: exportadores vs domésticas;
- bancos: high-rate sensitivity vs low-rate sensitivity;
- ETFs sectoriales como hedge;
- FX líquidos: EURUSD, USDJPY, GBPUSD, USDCHF, USDCNH si está disponible;
- futuros de tipos: Bund, Treasury, SOFR/Euribor futures, según infraestructura.

Criterios:

- liquidez alta;
- datos fundamentales de revenue por región;
- exposición histórica estimada por regresión;
- evitar compañías con hedge accounting opaco si no hay suficiente evidencia empírica.

---

## 5.3 Evento o señal

Shock FX:

```text
FXShock_t = Delta FX_t / sigma(Delta FX)
```

Señal si:

- movimiento de FX intradía o overnight supera umbral estadístico;
- shock está asociado a macro release, central bank, inflación, tipos o evento geopolítico;
- el sector se mueve de forma agregada, pero la dispersión por exposición aún es baja.

Construcción de exposición:

```text
Exposure_i =
    w1 * beta_FX_i_hist
  + w2 * ForeignRevenue_i
  + w3 * ImportCostProxy_i
  + w4 * AnalystTextExposure_i
```

Trade:

- long basket beneficiado por el shock;
- short basket perjudicado;
- hedge sectorial e índice.

Ejemplo:

- EURUSD cae con fuerza;
- exportadores europeos con ventas USD deberían beneficiarse;
- importadores europeos con costes USD deberían sufrir;
- long high USD revenue / short high USD cost dentro del mismo sector.

---

## 5.4 Reglas operativas

### Entrada

- Después de confirmar el shock en FX/tipos.
- No entrar durante el primer minuto de macro release.
- Ventana típica: 15–60 minutos tras release o apertura.
- En acciones europeas, especial atención a shocks USD ocurridos durante sesión estadounidense y reacción al día siguiente.

### Salida

- Intradía si el movimiento se corrige.
- T+1 a T+5 si el shock afecta expectativas de beneficios.
- Salida si FX revierte más del 50%.
- Stop por basket residual.
- Cierre antes de earnings propios si quedan cerca.

### Tamaño

- Market-neutral.
- Sector-neutral.
- Beta-neutral.
- Máximo 5–10 baskets temáticos.
- Limitar exposición neta a FX si no se cubre directamente.

---

## 5.5 Distribución y probabilidades

Retorno del trade:

```text
r_net = r_long_basket - r_short_basket - costes
```

Condicionar por:

- magnitud del shock FX;
- tipo de evento;
- sector;
- exposición estimada;
- volatilidad de mercado;
- dirección del movimiento.

Métricas:

- `P(r_net > 0 | FXShock, ExposureSpread)`;
- esperanza;
- mediana;
- percentiles;
- payoff ratio;
- hit rate;
- volatilidad condicional;
- MAE/MFE;
- drawdown por macro-event cluster;
- sensibilidad a costes;
- sensibilidad a error en beta FX.

El punto crítico es validar que el retorno proviene de exposición FX/tipos y no de beta de mercado. Por eso el retorno debe evaluarse residualizado frente a índice, sector y, si aplica, commodity.

---

## 5.6 Backtest realista

Datos necesarios:

- intradía FX;
- intradía acciones;
- calendario macro con timestamps;
- yields/futuros de tipos;
- fundamentales point-in-time por revenue geography;
- estimates;
- sector ETFs;
- bid/ask;
- datos de ADRs si se usan.

Errores a evitar:

- estimar exposición FX usando datos futuros;
- no actualizar exposiciones con cambios de negocio;
- confundir FX con shock general de riesgo;
- no ajustar por commodities;
- operar Europa con shocks ocurridos fuera del horario local sin modelar gap;
- ignorar hedging corporativo;
- usar revenue geography no point-in-time.

---

## 5.7 Robustez

Pruebas:

- exposición fundamental vs exposición estimada por retornos;
- shocks intradía vs overnight;
- EURUSD, USDJPY, GBPUSD por separado;
- sectores exportadores vs importadores;
- hedge con índice vs hedge con sector;
- exclusión de eventos macro grandes;
- costes duplicados;
- test por subperiodos;
- bootstrap por evento macro.

---

## 5.8 Riesgos

Riesgos:

- exposición mal medida;
- compañías cubren FX;
- shock FX coincide con shock de riesgo global;
- movimiento ya descontado;
- baja frecuencia;
- execution gap en Europa;
- cambios de régimen de correlaciones;
- dependencia de datos fundamentales caros;
- riesgo de basis entre ADR/local listing.

---

## 5.9 Implementación práctica

Necesario:

- feed FX y rates en tiempo real;
- calendario macro;
- base de exposiciones por compañía;
- motor de basket construction;
- optimizador market/sector neutral;
- ejecución basket;
- hedging con futuros/ETFs;
- monitor de reversión FX;
- revisión periódica de exposiciones.

Esta estrategia tiene una tesis económica fuerte, pero es más compleja de implementar que earnings.

---

# Marco común de backtest y validación

Para todas las estrategias, el backtest debe ser **event-driven**, no solo bar-based.

---

## Datos mínimos

- precios intradía 1 minuto o tick;
- bid/ask;
- volumen intradía;
- subastas de apertura/cierre si se usan;
- eventos con timestamps reales;
- calendario de earnings;
- consenso point-in-time;
- estimates point-in-time;
- sector/industry mapping histórico;
- corporate actions;
- short availability;
- borrow fees;
- datos de liquidez;
- flags de halts.

La subasta de cierre merece especial cuidado si se usa como salida. En EE. UU., la literatura muestra que la closing auction maneja una proporción relevante y creciente del volumen diario, y que el precio de subasta puede desviarse del midquote pre-cierre.

---

## Costes

Modelar:

```text
Coste = spread/2 + slippage + fees + borrow + market_impact
```

Sensibilidad obligatoria:

- costes base;
- 2x costes;
- 3x costes;
- slippage por volatilidad;
- slippage por participación en volumen;
- ejecución con retraso.

---

## Sesgos a controlar

- survivorship bias;
- look-ahead bias;
- consenso posterior al evento;
- timestamps incorrectos;
- clasificación sectorial retrospectiva;
- universe selection bias;
- vendor revision bias;
- disponibilidad de shorts;
- restricciones regulatorias;
- earnings simultáneos;
- eventos macro solapados.

---

## Validación

- train/test temporal;
- walk-forward trimestral o anual;
- test fuera de muestra;
- bootstrap por fecha;
- bootstrap por ticker;
- análisis por régimen de volatilidad;
- análisis por sector;
- análisis pre/post cambios de microestructura;
- exclusión de outliers;
- test de capacidad.

---

# Comparativa final

| Estrategia | Universo | Tipo de hipótesis | Señal | Horizonte | Frecuencia esperada | Datos necesarios | Dificultad de ejecución | Edge a estimar | Principales riesgos | Expectativa de robustez |
|---|---|---|---|---|---|---|---|---|---|---|
| Earnings continuation condicionado | Acciones líquidas EE. UU./Europa con earnings | Incorporación lenta de información + absorción institucional | Sorpresa positiva, gap moderado, volumen alto, sector confirmando | Intradía a T+1 | Alta durante earnings season | Earnings, consenso, intradía, sector, bid/ask | Media | Drift residual neto tras costes | Guidance contradictorio, gap agotado, crowding | Alta si los filtros son estrictos |
| Reversión de gap no confirmado | Acciones líquidas con noticia fundamental | Sobrerreacción + presión de liquidez | Gap extremo sin confirmación de peers/sector | Intradía a T+1 | Media | Noticias, intradía, peer basket, sector, short data | Media-alta | Reversión residual del gap | Clasificación errónea de noticia, colas negativas | Media; sensible a filtros |
| Read-through de líder a peers | Baskets sectoriales | Transferencia intra-sectorial de información | Earnings/guidance de líder y peers retrasados | Intradía a T+2/T+3 | Baja-media | Earnings, consenso, intradía, mapping económico de peers | Alta | Retorno de peer basket residual | Read-through competitivo mal interpretado, baja muestra | Alta en sectores bien definidos |
| Cascade de revisiones de analistas | Acciones con alta cobertura | Revisión lenta de expectativas | Revisiones coordinadas de EPS/target/rating | T+1 a T+5 | Media | Estimates point-in-time, timestamps, ratings, intradía | Alta | Drift tras revisiones neto de costes | Vendor latency, señal crowded, revisiones mecánicas | Media-alta si hay buen dato point-in-time |
| Shock FX/tipos por exposición | Baskets de acciones expuestas a FX/rates | Repricing de cash flows por shock exógeno | Movimiento extremo en FX/tipos + exposición firm-level | Intradía a T+5 | Baja-media | FX, rates, intradía, revenue geography, betas, macro calendar | Alta | Spread high-exposure vs low-exposure | Exposición mal medida, hedging corporativo, shock de riesgo global | Media; fuerte tesis pero implementación difícil |

---

# Ranking inicial de investigación

## 1. Earnings continuation condicionado

Mejor relación entre claridad, frecuencia y testeabilidad.

## 2. Read-through de líder a peers

Tesis económica fuerte, pero requiere mapping de relaciones muy bueno.

## 3. Cascade de revisiones de analistas

Potencialmente escalable, dependiente de datos caros y timestamps fiables.

## 4. FX/tipos por exposición

Elegante, pero compleja y con riesgo de contaminación macro.

## 5. Reversión de gap no confirmado

Atractiva, pero más peligrosa por colas y clasificación de noticias.

---

# Referencias orientativas

- Bernard, V. L., & Thomas, J. K. (1989). *Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?* Journal of Accounting Research.
- Womack, K. L. (1996). *Do Brokerage Analysts' Recommendations Have Investment Value?* Journal of Finance.
- Estudios sobre transferencia intra-sectorial de información alrededor de earnings.
- Estudios sobre exposición de firmas a movimientos de FX y su impacto en retornos.
- Estudios sobre microestructura y subastas de cierre.


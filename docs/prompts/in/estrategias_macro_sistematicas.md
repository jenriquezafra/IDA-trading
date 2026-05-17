# Estrategias sistemáticas de trading basadas en fundamentos macroeconómicos

> Documento de investigación cuantitativa. No constituye recomendación personalizada de inversión.

Este documento propone cuatro estrategias sistemáticas basadas en fundamentos económicos, macroeconómicos y financieros. El criterio común es que cada estrategia tenga una tesis causal clara, señales observables ex ante, implementación líquida y posibilidad de evaluación histórica robusta.

Las cuatro estrategias conservadas son:

1. **Rotación global de renta variable por país: valoración + calidad macro.**
2. **FX real carry con filtro de fragilidad externa.**
3. **Commodities: escasez, backwardation e inflación.**
4. **Global rates: duración y curva tras política monetaria restrictiva.**

Se descartan deliberadamente ideas más débiles, como commodities long-only como cobertura permanente de inflación, FX carry sin filtro de riesgo, rotación sectorial basada solo en PMI o duración long-only estructural.

---

## 1. Equity country rotation: valoración + calidad macro

### 1.1. Tesis económica

Los índices de renta variable por país son derechos sobre beneficios corporativos locales, pero también incorporan riesgo político, moneda, ciclo de crédito, inflación, márgenes y condiciones externas. La estrategia explota una combinación de tres primas o ineficiencias:

**Value macro:** los países baratos frente a sus beneficios normalizados tienden a ofrecer mayor retorno esperado. Sin embargo, valoración barata por sí sola puede ser una trampa.

**Calidad macro-financiera:** los países baratos con inflación controlada, balances externos razonables y beneficios no colapsando tienen más probabilidad de rerating que países baratos por deterioro estructural.

**Segmentación y restricciones:** muchos inversores asignan por región, benchmark o mandato, no por valoración relativa global. Esto permite que las divergencias entre países persistan.

No se acepta una estrategia simple de “comprar el CAPE más bajo”. La versión defendible es una estrategia de **valor relativo entre países con veto macro**, porque la señal fundamental es lenta, barata de implementar y menos dependiente del timing exacto.

### 1.2. Universo de inversión

Usaría índices líquidos de países desarrollados y emergentes grandes.

**Desarrollados:**

- Estados Unidos
- Canadá
- Reino Unido
- Alemania
- Francia
- Italia
- España
- Suiza
- Japón
- Australia
- Suecia
- Países Bajos

**Emergentes líquidos:**

- México
- Brasil
- India
- Corea
- Taiwán
- China
- Sudáfrica
- Polonia
- Indonesia
- Tailandia

Instrumentos posibles:

- Futuros sobre índices cuando existan.
- ETFs de país.
- Swaps sobre índices MSCI/FTSE.
- UCITS equivalentes para inversores europeos.

Para backtest, usaría índices MSCI o FTSE de retorno total, no ETFs, porque los ETFs tienen historiales más cortos.

### 1.3. Variables fundamentales

Para cada país \(i\):

**Valoración:**

\[
Val_i = \text{avg}\left[z(1/CAPE_i), z(E/P_i), z(DY_i), z(B/P_i)\right]
\]

Donde:

- \(CAPE\): cyclically adjusted price-to-earnings.
- \(E/P\): earnings yield.
- \(DY\): dividend yield.
- \(B/P\): book-to-price.

**Calidad corporativa:**

\[
Qual_i = \text{avg}\left[z(ROE_i), z(margen\ operativo_i), -z(leverage_i), z(revisiones\ EPS_i)\right]
\]

**Macro solvencia:**

\[
Macro_i = \text{avg}\left[z(CA/GDP_i), -z(inflación_i), -z(CDS_i), -z(crédito/GDP\ excesivo_i)\right]
\]

**Ciclo:**

- PMI.
- Producción industrial.
- Desempleo.
- Crédito.
- OECD Composite Leading Indicator.

### 1.4. Reglas de entrada y salida

Rebalanceo mensual.

Defino la puntuación total:

\[
S_i = 0.50 Val_i + 0.25 Qual_i + 0.25 Macro_i
\]

Los z-scores se winsorizan al 5 % y se calculan con ventana expansiva o rolling de mínimo 10 años.

**Entrada long:**

Comprar países en el top 25 % de \(S_i\), siempre que no activen veto macro.

**Entrada short:**

Vender países en el bottom 25 % de \(S_i\), solo si el instrumento es líquido, prestable y con coste razonable.

En versión long-only, no se vende en corto: simplemente se infrapondera o excluye.

**Veto macro para largos:**

No comprar un país aunque sea barato si se cumple cualquiera de estas condiciones:

\[
\pi_{YoY} > 10\% \text{ y acelerando}
\]

\[
FX_{6m} < -15\% \text{ contra USD/EUR y reservas cayendo}
\]

\[
CDS\ soberano \text{ en top 10 % histórico y subiendo}
\]

\[
CA/GDP < -4\% \text{ y crédito privado/GDP creciendo agresivamente}
\]

**Salida:**

Salir si:

- El país cae por debajo del percentil 50 de \(S_i\).
- Activa veto macro.
- La liquidez del instrumento se deteriora.

No usaría un stop-loss mecánico de precio. Preferiría un **stop de tesis**.

### 1.5. Construcción de cartera

Dos versiones posibles.

#### Versión long-only benchmark-aware

- Comprar 6–10 países con mejor señal.
- Peso inicial por inverse volatility.
- Máximo 20 % por país.
- Tracking error objetivo frente a MSCI ACWI: 4–8 % anual.

#### Versión long-short institucional

- Long top 25 %.
- Short bottom 25 %.
- Beta global neutral frente a MSCI ACWI.
- Exposición bruta: 150–200 %.
- Exposición neta: entre -10 % y +30 %.
- Máximo 12 % bruto por país.
- Emergentes máximo 35 % bruto.

Rebalanceo mensual, con turnover cap: no cambiar más del 25 % de la cartera mensual salvo veto macro.

### 1.6. Gestión de riesgo

Principales riesgos:

- **Value trap:** un país barato puede seguir barato durante años.
- **Riesgo de divisa:** parte del retorno de país viene de moneda, no de equity local.
- **Riesgo político y controles de capital:** especialmente en emergentes.
- **Concentración sectorial:** algunos países son proxies de bancos, energía, semiconductores o commodities.

Controles:

- Volatilidad objetivo: 10–12 % en long-short; 8–10 % en long-only.
- Hedge parcial de divisa en desarrollados.
- No cubrir automáticamente emergentes: la divisa forma parte del riesgo macro.
- Liquidez mínima diaria.
- Bid-ask spread máximo permitido.
- Stress tests: Asia 1997, 2008, eurozona 2011, China/EM 2015, COVID 2020, inflación 2022.

### 1.7. Backtest propuesto

**Periodo:**

- 1988–2026 si se dispone de MSCI country total return, valoraciones y macro.
- Para emergentes, probablemente submuestras desde 1995–2000.

**Frecuencia:** mensual.

**Datos:**

- MSCI/FTSE country total return.
- Worldscope/FactSet/Bloomberg para valoración y EPS.
- FMI/Banco Mundial/OCDE/FRED para macro.
- CDS/soberanos desde Bloomberg/Markit.
- Datos vintage cuando sea posible.

**Benchmarks:**

- MSCI ACWI.
- MSCI World.
- MSCI EM.
- Equal-weight country portfolio.

**Costes:**

- 10–25 bps por operación en desarrollados.
- 35–75 bps por operación en emergentes.
- Repetir test con costes duplicados.

**Métricas clave:**

- Sharpe.
- Information ratio.
- Drawdown.
- Turnover.
- Hit rate por país.
- Alpha contra ACWI.
- Beta.
- Exposición a value global.
- Exposición a USD.
- Performance por régimen de inflación y crecimiento.

### 1.8. Riesgos de sobreajuste

Lo más vulnerable es el peso de las variables: 50/25/25 puede ser arbitrario. Validaría usando:

- Pesos iguales.
- Rankings simples.
- Señales separadas.
- Walk-forward: entrenar 1988–2004, validar 2005–2015, holdout 2016–2026.
- Leave-one-country-out.
- DM-only y EM-only.
- Sustitución de CAPE por E/P y B/P.
- Retraso de datos fundamentales de 2–3 meses.

La estrategia solo pasa si la señal de valoración sigue funcionando sin optimizar pesos.

### 1.9. Implementación práctica

Para un inversor institucional:

- Futuros de índices.
- Swaps sobre MSCI country indices.
- Cestas de ETFs.

Para implementación líquida:

- ETFs de país.
- UCITS equivalentes.
- Futuros sobre S&P 500, Euro Stoxx 50, DAX, FTSE, Nikkei, Hang Seng y MSCI EM.

### 1.10. Refutación crítica

La objeción fuerte es que el CAPE predice mejor retornos a 7–10 años que a 6–18 meses. Por tanto, una rotación mensual puede generar ruido. La estrategia solo sería aceptable si el backtest muestra que el ranking funciona con baja rotación y que la mayor parte del retorno viene de diferenciales de valoración persistentes, no de timing macro puntual.

---

## 2. FX real carry con filtro de fragilidad externa

### 2.1. Tesis económica

La estrategia explota la desviación persistente de la paridad descubierta de tipos. Las monedas con tipos altos no siempre se deprecian lo suficiente para eliminar el carry. Pero esta prima no es gratis: compensa riesgo de crash, iliquidez, funding stress y desapalancamiento.

La versión defendible no es “comprar la moneda con mayor tipo”. Es:

> Comprar carry real alto solo cuando la moneda no parece sobrevalorada y el país no muestra fragilidad externa extrema.

### 2.2. Universo de inversión

**G10:**

- USD
- EUR
- JPY
- GBP
- CHF
- CAD
- AUD
- NZD
- NOK
- SEK

**Emergentes líquidos:**

- MXN
- BRL
- ZAR
- PLN
- HUF
- CZK
- KRW
- INR
- IDR
- CLP
- COP
- SGD

Preferiría empezar con G10 + MXN + ZAR + BRL + PLN + KRW por liquidez y datos.

Instrumentos:

- Forwards 1M/3M.
- NDFs para algunas divisas emergentes.
- Futuros de divisas líquidos.
- ETFs de divisas solo para versión simplificada.

### 2.3. Variables fundamentales

Para cada divisa \(i\):

**Carry nominal:**

\[
Carry_i = r_i^{3m} - r_{funding}^{3m}
\]

o forward discount implícito.

**Carry real:**

\[
RealCarry_i = (r_i^{3m} - \pi_i^{12m}) - (r_{funding}^{3m} - \pi_{funding}^{12m})
\]

**Valoración externa:**

\[
REERVal_i = -z(REER_i)
\]

Positivo si la moneda está barata frente a su media histórica.

**Fragilidad externa:**

\[
Frag_i = -z(CA/GDP_i) - z(Reservas/STDebt_i) + z(inflación_i) + z(CDS_i)
\]

**Riesgo global:**

- VIX.
- Volatilidad FX realizada.
- Spreads de crédito.
- USD funding stress.
- Drawdown de equity global.

### 2.4. Reglas de entrada y salida

Rebalanceo mensual; control de riesgo semanal.

Señal principal:

\[
S_i = 0.45z(RealCarry_i) + 0.25z(REERVal_i) - 0.20z(Frag_i) + 0.10z(Momentum_{6m,i})
\]

**Entrada long:**

Comprar las divisas en el top 25 % de \(S_i\).

**Entrada short/funding:**

Vender las divisas en el bottom 25 % de \(S_i\), preferentemente JPY, CHF, EUR o USD cuando tengan bajo carry real y baja inflación.

**Veto para largos emergentes:**

No comprar si:

\[
CA/GDP < -4\%
\]

\[
Reservas/STDebt < 1
\]

\[
\pi_{YoY} > 12\% \text{ y acelerando}
\]

\[
REER_i > +1.5\sigma \text{ y } RealCarry_i \text{ alto}
\]

Esto evita comprar high yield currencies que pagan mucho porque están cerca de una devaluación.

**Risk-off gate:**

Reducir exposición bruta al 50 % si:

\[
VIX > P_{85}
\]

o:

\[
FXVol > P_{85}
\]

Poner la cartera plana si dos o más condiciones extremas se activan:

\[
VIX > P_{95}, \quad FXVol > P_{95}, \quad MSCI\ World_{drawdown} < -10\%, \quad USD\ funding\ stress > P_{90}
\]

**Salida:**

Salir si:

- La divisa cae por debajo del percentil 50 de \(S_i\).
- Activa veto macro.
- Pierde más de 2 desviaciones estándar frente a su carry esperado en menos de un mes y el régimen global está en risk-off.

### 2.5. Construcción de cartera

Long-short, dollar-neutral o EUR-neutral según base del inversor.

Peso por inverse volatility:

\[
w_i \propto \frac{S_i}{\sigma_i}
\]

Límites:

- Máximo 10 % de riesgo por divisa.
- Máximo 35 % de riesgo total en emergentes.
- Máximo 40 % de funding en una sola moneda.
- Volatilidad objetivo: 7–9 % anual.
- Rebalanceo mensual.
- Reducción dinámica semanal.

### 2.6. Gestión de riesgo

El riesgo clave es la **asimetría negativa**: muchos meses de carry positivo pueden perderse en pocos días. Por eso no usaría leverage alto.

Controles:

- Volatilidad FX realizada.
- Concentración en emergentes.
- Correlación entre largos.
- Funding currency squeeze.
- Eventos de bancos centrales.
- Riesgo de intervención.

En versión institucional, compraría ocasionalmente opciones OTM sobre funding currencies cuando la volatilidad implícita sea barata. No como fuente de alpha, sino como seguro de crash.

### 2.7. Backtest propuesto

**Periodo:**

- G10 desde 1989–1990.
- Emergentes desde 2000 aproximadamente, según liquidez.

**Datos:**

- Spot FX.
- Forwards 1M/3M.
- Tipos interbancarios/OIS.
- Inflación.
- REER BIS.
- Cuenta corriente.
- Reservas.
- CDS.
- Volatilidad global.

**Frecuencia de señal:** mensual.

**Frecuencia de riesgo:** semanal o diaria para gates.

**Costes:**

- 1–3 bps G10 por lado.
- 5–25 bps EM por lado.
- Slippage superior en crisis.
- Test obligatorio con costes duplicados y spreads de crisis.

**Benchmark:**

- Carry naïve HML FX.
- Índices FX carry de Deutsche/Bloomberg si están disponibles.
- Cash.

**Métricas:**

- Sharpe.
- Skewness.
- Expected shortfall.
- Máximo drawdown.
- Retorno en meses de VIX alto.
- Beta a equity global.
- Beta a USD.
- Performance por régimen de inflación.

### 2.8. Riesgos de sobreajuste

Los thresholds de VIX/volatilidad son vulnerables a data mining. Validaría sustituyendo percentiles 80/85/90/95 y comprobando que la lógica no depende de un punto exacto.

También probaría:

- G10-only.
- EM-only.
- Sin momentum.
- Sin REER.
- Con carry nominal en vez de real carry.

La estrategia debe seguir siendo razonable con carry simple + filtro de fragilidad. Si solo funciona con una combinación precisa de pesos, se descarta.

### 2.9. Implementación práctica

Institucional:

- FX forwards.
- NDFs.

Simplificada:

- Futuros de divisas líquidos para G10 y algunas emergentes.
- ETFs de divisas solo si la liquidez y tracking son aceptables.

Para un inversor europeo, es clave definir si el objetivo es generar retorno en EUR o USD. La moneda base cambia la interpretación del funding.

### 2.10. Refutación crítica

La objeción fuerte es que el carry es una prima por vender seguro contra crisis. El filtro de riesgo puede sacar a la estrategia demasiado tarde o demasiado pronto. Solo se acepta si el backtest muestra menor drawdown y menor skew negativa que carry naïve, aunque sacrifique retorno medio.

---

## 3. Commodities: escasez, backwardation e inflación

### 3.1. Tesis económica

Los futuros de commodities no son una clase homogénea. Cada mercado tiene inventarios, estacionalidad, capacidad de almacenamiento, shocks geopolíticos y demanda industrial distintos.

La prima defendible no es “comprar commodities siempre”, sino explotar señales de **escasez física y presión de cobertura**:

**Backwardation:** cuando el contrato cercano cotiza por encima del diferido, suele reflejar escasez inmediata o convenience yield alto.

**Inventarios bajos:** menor colchón físico aumenta el valor de asegurar suministro.

**Momentum:** shocks de oferta/demanda suelen propagarse lentamente.

**Inflación:** commodities, especialmente energía y metales, pueden responder bien a inflación inesperada, aunque la evidencia de cobertura no es estable para todos los subsectores.

### 3.2. Universo de inversión

Futuros líquidos:

**Energía:**

- WTI
- Brent
- Gasolina
- Heating oil
- Gas natural

**Metales preciosos:**

- Oro
- Plata

**Metales industriales:**

- Cobre
- Aluminio
- Zinc
- Níquel, si la liquidez lo permite

**Granos:**

- Maíz
- Trigo
- Soja

**Softs:**

- Café
- Azúcar
- Algodón

**Livestock:**

- Live cattle
- Lean hogs

Evitaría contratos con baja liquidez, problemas de entrega física o curvas difíciles de replicar.

### 3.3. Variables fundamentales

Para cada commodity \(c\):

**Carry / roll yield:**

\[
Carry_c = \ln\left(\frac{F_{1,c}}{F_{4,c}}\right) \times 4
\]

Positivo si la curva está en backwardation.

**Momentum:**

\[
Mom_c = R_{12m,c} - R_{1m,c}
\]

Retorno de 12 meses excluyendo el último mes.

**Inventarios / escasez:**

\[
Scarcity_c = -z(Inventory_c / Demand_c)
\]

Ejemplos:

- Granos: stock-to-use ratio.
- Metales: inventarios LME/COMEX ajustados por consumo.
- Energía: inventarios EIA/IEA ajustados por demanda.

**Inflación macro:**

- Inflación sorpresa.
- PPI.
- Breakevens.
- Inflation swaps.
- USD real rate.
- PMI manufacturero.

### 3.4. Reglas de entrada y salida

Rebalanceo mensual.

Señal por contrato:

\[
S_c = 0.50z(Carry_c) + 0.30z(Mom_c) + 0.20z(Scarcity_c)
\]

**Entrada long:**

Comprar commodities con \(S_c > 0\) y en el top 40 % del universo.

**Entrada short opcional:**

Vender commodities con \(S_c < 0\) y en el bottom 40 %. Solo en versión institucional. En long-only, simplemente excluir.

**Overlay macro de inflación:**

Multiplicar exposición bruta por 1.25 si:

\[
InflationSurprise > 0
\]

\[
Breakevens_{3m} > 0
\]

\[
USD\ real\ rate_{3m} \leq 0
\]

Reducir exposición bruta a 0.75 si:

\[
USD_{3m} > 0
\]

\[
RealRates_{3m} > +50bps
\]

\[
PMI < 50
\]

**Salida:**

Salir si:

- \(S_c < 0\).
- El contrato pasa a contango extremo.
- El volumen/open interest cae por debajo del umbral mínimo.
- Se aproxima un evento de entrega física que el vehículo no puede gestionar.

### 3.5. Construcción de cartera

Versión preferida: long-only relativa, no commodity beta puro.

Peso por inverse volatility con límites:

- Máximo 40 % energía.
- Máximo 30 % agricultura.
- Máximo 30 % metales.
- Máximo 12 % por contrato.
- Volatilidad objetivo: 10–12 %.
- Collateral: T-bills o instrumentos monetarios de alta calidad.

La versión long-short puede tener mejor neutralidad a beta commodity, pero introduce riesgo de squeeze en contratos físicamente escasos. La usaría solo con infraestructura institucional.

### 3.6. Gestión de riesgo

Riesgos principales:

- Squeezes en contratos cercanos.
- Riesgo de roll y metodología de curva.
- Eventos geopolíticos.
- Clima y cosechas.
- Regulación de posiciones.
- Fuerte dependencia de energía si no se controla.

Controles:

- Usar contratos con suficiente open interest.
- Rolar antes del periodo de entrega.
- Sector caps estrictos.
- Volatility targeting.
- Stress tests para shocks de petróleo, crisis de gas, sequías, subidas bruscas del USD y recesiones industriales.

### 3.7. Backtest propuesto

**Periodo:**

- Idealmente 1991–2026.
- Si hay datos fiables, extender a los años setenta para energía, metales y granos.

**Datos:**

- Curvas de futuros.
- Precios de vencimientos individuales.
- Volúmenes.
- Open interest.
- Inventarios.
- CPI/PPI.
- Breakevens.
- USD index.
- Real rates.

**Benchmark:**

- Bloomberg Commodity Index.
- S&P GSCI.
- Cash + T-bills.
- Commodity beta equal-weight.

**Costes:**

- 2–10 bps por lado en contratos líquidos.
- Costes mayores en softs/livestock.
- Incluir slippage de roll.
- Probar roll mensual, roll por liquidez y roll por señal.

**Métricas:**

- Sharpe.
- Drawdown.
- Skewness.
- Beta a inflación.
- Beta a equity.
- Beta a USD.
- Retorno en shocks inflacionarios.
- Contribución por sector.
- Turnover.
- Capacidad.

### 3.8. Riesgos de sobreajuste

El peligro está en elegir vencimientos, ventana de momentum y peso de carry. Validaría:

- Carry con \(F_1/F_3\), \(F_1/F_6\) y pendiente completa.
- Momentum 6m, 9m, 12m.
- Sin inventarios.
- Long-only vs long-short.
- Excluyendo energía.
- Excluyendo cada sector uno a uno.

La estrategia solo se acepta si carry + momentum sobreviven sin depender de un vencimiento exacto.

### 3.9. Implementación práctica

Institucional:

- Futuros CME/ICE/LME.
- Colateral en T-bills.

Práctica simplificada:

- ETFs amplios de commodities.
- ETFs/ETCs sectoriales.

Para una implementación seria, preferiría futuros porque la tesis depende precisamente de la curva.

### 3.10. Refutación crítica

La objeción fuerte es que muchos retornos históricos de commodities dependen del esquema de ponderación y del periodo. Además, un basket long-only de commodities puede pasar décadas mediocres. La estrategia solo sería aceptable como exposición **selectiva por escasez y carry**, no como exposición permanente.

---

## 4. Global rates: duración y curva tras política monetaria restrictiva

### 4.1. Tesis económica

Los bonos soberanos nominales reflejan tres componentes:

1. Tipos reales esperados.
2. Inflación esperada.
3. Term premium.

Una estrategia defendible busca duración cuando:

1. La política monetaria ya es restrictiva.
2. El crecimiento se desacelera.
3. La inflación deja de acelerar.
4. El term premium ofrece compensación.

La tesis es que cuando el banco central está cerca del final del ciclo de subidas y el crecimiento se enfría, el mercado tiende a descontar recortes o menor tipo terminal. Eso favorece duración y, a menudo, steepeners de curva. Pero si la inflación vuelve a acelerar o el déficit fiscal eleva el term premium, la estrategia falla.

### 4.2. Universo de inversión

Futuros de tipos líquidos:

**Estados Unidos:**

- 2Y Treasury futures.
- 5Y Treasury futures.
- 10Y Treasury futures.
- Ultra Bond futures.

**Europa:**

- Schatz.
- Bobl.
- Bund.
- BTP si se asume riesgo periférico.

**Otros:**

- Gilts.
- Canadá.
- Australia.
- Japón, si la liquidez y la microestructura son adecuadas.

Versión ETF:

- Treasuries 1–3Y.
- Treasuries 7–10Y.
- Treasuries 20Y+.
- Inflation-linked bonds.
- ETFs de bonos soberanos globales.

Para ejecución táctica, futuros son superiores.

### 4.3. Variables fundamentales

Por país \(j\):

**Restrictividad monetaria:**

\[
Restrict_j = z\left((PolicyRate_j - CoreInflation_j) - r^*_j\right)
\]

Si \(r^*\) no es fiable, usar el real policy rate relativo a su propia distribución histórica.

**Crecimiento:**

\[
GrowthDecline_j = -z(\Delta PMI_j, \Delta CLI_j, \Delta Claims_j)
\]

**Inflación:**

\[
InflDecline_j = -z(\pi^{3m}_{core,ann} - \pi^{12m}_{core})
\]

Complementar con breakevens o inflation swaps cuando existan.

**Term premium / curva:**

\[
TP_j = z(TermPremium_{10Y,j})
\]

Si no hay term premium fiable:

\[
Slope_j = z(Yield_{10Y} - Yield_{3M})
\]

### 4.4. Reglas de entrada y salida

Señal de duración:

\[
S_j = 0.35Restrict_j + 0.25GrowthDecline_j + 0.25InflDecline_j + 0.15TP_j
\]

**Long duration:**

\[
S_j > +0.5
\]

Comprar futuros 5Y/10Y o bonos 7–10Y/20Y según convexidad deseada.

**Short duration:**

\[
S_j < -0.5
\]

Y además:

\[
InflationMomentum_j > 0
\]

O:

\[
PMI_j > 52
\]

Vender duración.

**Neutral:**

\[
-0.5 \leq S_j \leq +0.5
\]

**2s10s steepener:**

Entrar en steepener DV01-neutral si:

\[
Slope_{2s10s} < -50bps
\]

\[
Restrict_j > +0.5
\]

\[
GrowthDecline_j > 0
\]

Salir si:

\[
Slope_{2s10s} > +50bps
\]

O si la inflación core 3m anualizada vuelve a superar la inflación 12m.

### 4.5. Construcción de cartera

Dos bloques:

- 60 % riesgo: duración direccional global.
- 40 % riesgo: curvas DV01-neutral.

Ponderación por riesgo, no por notional.

Límites:

- Máximo 35 % del riesgo en Estados Unidos.
- Máximo 20 % por mercado no estadounidense.
- Máximo drawdown tolerado antes de reducción: 8–10 % de la estrategia.
- Volatilidad objetivo: 7–9 %.
- Rebalanceo mensual.
- Ajuste semanal si inflación o yields rompen umbrales.

### 4.6. Gestión de riesgo

Escenarios de fallo:

- Inflación reacelera.
- Shock fiscal eleva term premium.
- Banco central mantiene tipos altos más tiempo.
- Recesión con inflación alta: stagflation.
- Correlación positiva equity-bond, como en shocks inflacionarios.

Controles:

- No estar long duration si inflación core 3m anualizada sube durante dos meses consecutivos y breakevens suben.
- No concentrar todo en 30Y.
- Preferir 5Y/10Y cuando la incertidumbre fiscal es alta.
- Separar duración de curva.
- Stress tests: años setenta, 1994, 2008, 2013 taper tantrum, 2020, 2022.

### 4.7. Backtest propuesto

**Periodo:**

- 1990–2026 para futuros líquidos globales.
- Extender con yields cero-cupón si se dispone de datos.

**Datos:**

- Futuros de bonos.
- Yields.
- OIS.
- Inflación.
- PMIs.
- OECD CLI.
- Claims.
- Breakevens.
- Term premia.
- Curvas 2Y/10Y/30Y.

Datos macro con vintage son imprescindibles, porque inflación y crecimiento se revisan o publican con retraso.

**Benchmark:**

- Bloomberg Global Treasury.
- US Treasury 7–10Y.
- Cash.
- Trend-following rates.

**Costes:**

- Muy bajos en Treasury/Bund futures.
- Probar 2x y 5x costes.
- Incluir collateral yield.

**Métricas:**

- Sharpe.
- Drawdown.
- Beta a equity.
- Retorno en recesiones.
- Retorno en inflación alta.
- DV01 neta.
- Convexidad.
- Hit rate por ciclo monetario.

### 4.8. Riesgos de sobreajuste

La estimación de \(r^*\) y term premium es frágil. Validaría con versiones sin \(r^*\), usando solo percentiles históricos del real policy rate.

También probaría señales con y sin term premium.

La estrategia no debe depender de acertar el nivel exacto de neutral rate. Debe funcionar porque identifica un patrón amplio:

> Política restrictiva + crecimiento cayendo + inflación no acelerando.

### 4.9. Implementación práctica

Instrumentos principales:

- Futuros sobre Treasuries.
- Bund futures.
- Gilt futures.
- JGB futures.
- Futuros de Canadá y Australia.

Para carteras simples:

- ETFs de duración corta/media/larga.

Para Europa, distinguir claramente entre duration core —Bund— y riesgo spread periférico —BTP—. No mezclar ambos sin modelar spread soberano.

### 4.10. Refutación crítica

La objeción fuerte es que tras 2020–2022 quedó claro que los bonos nominales pueden dejar de diversificar si el shock dominante es inflación, no crecimiento. La estrategia solo sería aceptable con veto inflacionario explícito y con capacidad de ir neutral o short duration.

---

## 5. Cartera multi-estrategia propuesta

No combinaría estas estrategias por capital fijo, sino por **presupuesto de riesgo**.

Asignación inicial razonable para investigación:

| Estrategia | Riesgo asignado |
|---|---:|
| Equity country value-quality | 25 % |
| FX real carry filtrado | 25 % |
| Commodities carry/momentum/escasez | 25 % |
| Global rates duration/curve | 25 % |

Luego ajustaría por correlación:

\[
w \propto \Sigma^{-1} \mu
\]

Pero con \(\mu\) conservador, sin optimizar retornos históricos de forma agresiva. En producción usaría equal-risk-contribution con volatilidad objetivo total de 8–10 %.

### 5.1. Regla de protección de cartera

Reducir exposición bruta total 30–50 % si tres condiciones coinciden:

\[
EquityDrawdown < -10\%
\]

\[
FXVol > P_{90}
\]

\[
CreditSpreads > P_{90}
\]

\[
LiquidityStress > P_{90}
\]

No liquidaría automáticamente todas las estrategias:

- Commodities pueden ayudar en inflación.
- Duration puede ayudar en recesión.
- FX carry y equity country value suelen necesitar reducción en estrés de liquidez.

---

## 6. Protocolo de backtest común

Para que el backtest sea creíble:

### 6.1. Datos point-in-time

Usar datos disponibles en la fecha de decisión, no series revisadas. Para datos macro estadounidenses, ALFRED permite recuperar vintages de datos económicos. Para países no estadounidenses, usar OCDE, bancos centrales, Bloomberg o proveedores con release dates.

### 6.2. Retrasos realistas

- Macro: usar con retraso de publicación.
- Fundamentales: lag de 2–3 meses.
- CAPE/EPS: evitar usar beneficios revisados.
- PMI/encuestas: menor lag, pero respetar fecha de publicación.

### 6.3. Ejecución

- Operar al cierre o siguiente sesión tras señal.
- No operar en el precio del dato publicado antes de la publicación.

### 6.4. Costes

Incluir:

- Bid-ask.
- Comisiones.
- Slippage.
- Roll.
- Borrow cost.
- Financing.
- Coste de divisa.

Probar costes base, 2x y 5x.

### 6.5. Sesgos a evitar

- Survivorship bias.
- Look-ahead bias.
- Selección ex post de países/contratos.
- Data snooping de ventanas.
- Benchmarks incorrectos.
- No incluir collateral return en futuros.

### 6.6. Métricas mínimas

- Retorno anualizado.
- Volatilidad.
- Sharpe.
- Sortino.
- Max drawdown.
- Calmar.
- Skewness.
- Expected shortfall.
- Turnover.
- Capacidad.
- Beta a equity.
- Beta a inflación.
- Beta a USD.
- Performance por régimen.

### 6.7. Validación robusta

- Walk-forward.
- Holdout temporal.
- Submuestras geográficas.
- Exclusión de activos uno a uno.
- Variación de ventanas.
- Parámetros redondeados y no optimizados.

---

## 7. Tabla comparativa final

| Estrategia | Fundamento económico | Universo | Horizonte temporal | Complejidad de implementación | Principales riesgos | Expectativa de robustez | Datos necesarios |
|---|---|---|---|---|---|---|---|
| Equity country value-quality | Valoración relativa entre países + calidad macro evita value traps | Índices/ETFs/futuros de países DM y EM líquidos | 6–36 meses | Media | Value puede tardar años; divisa; política; emergentes | Media-alta si se aplica con baja rotación y filtros macro simples | MSCI/FTSE country returns, CAPE, E/P, B/P, ROE, EPS, inflación, cuenta corriente, CDS, PMI/CLI |
| FX real carry filtrado | Prima por carry real, compensación por crash risk y funding liquidity | G10 + EM líquidos vía forwards/futuros/NDFs | 1–12 meses | Alta | Crash risk, intervención, liquidez, capital controls, skew negativa | Media; mejora si el filtro de fragilidad reduce colas sin matar carry | Spot/forward FX, tipos 3M/OIS, inflación, REER, reservas, CA/GDP, CDS, VIX, FX vol |
| Commodities carry/momentum/escasez | Backwardation e inventarios bajos reflejan escasez/convenience yield; momentum captura shocks persistentes | Futuros líquidos de energía, metales, granos, softs, livestock | 1–12 meses | Media-alta | Squeezes, roll risk, concentración energía, shocks geopolíticos, estacionalidad | Media-alta si no depende de long-only beta y sobrevive por sectores | Curvas de futuros, inventarios, open interest, CPI/PPI, breakevens, USD, real rates |
| Global rates duration/curve | Política restrictiva + crecimiento desacelerando + inflación no acelerando favorece duración/steepeners | Futuros soberanos US, Bund, Gilts, JGB, Canadá, Australia | 3–18 meses | Media | Reaceleración inflacionaria, fiscal risk, term premium shock, modelos r* frágiles | Media; buena en shocks de crecimiento, débil en stagflation | Yields, OIS, inflación, breakevens, PMIs, CLI, claims, term premium, curvas 2Y/10Y/30Y |

---

## 8. Estrategias descartadas

### 8.1. Rotación sectorial basada solo en PMI

La descartaría porque suele convertirse en market timing encubierto y es muy sensible a la ventana elegida.

### 8.2. Commodities long-only como hedge permanente de inflación

La descartaría porque la evidencia es demasiado dependiente del periodo, sector y esquema de roll.

### 8.3. FX carry sin filtro de liquidez

La descartaría porque la prima existe, pero la cola izquierda domina justo cuando el capital escasea.

### 8.4. Duración long-only estructural

La descartaría como diversificador universal. Funciona bien en shocks deflacionarios, pero puede fallar severamente cuando inflación y term premium suben juntos.

---

## 9. Conclusión

La combinación más defendible sería una cartera diversificada de las cuatro estrategias, con:

- Riesgo equilibrado.
- Señales lentas.
- Pocos parámetros.
- Control explícito de colas.
- Datos point-in-time.
- Validación fuera de muestra.
- Costes de transacción conservadores.

La parte más robusta conceptualmente es que las cuatro estrategias no dependen de una única prima: combinan valoración, carry, escasez física, política monetaria, ciclo e inflación. La parte más vulnerable es la interacción entre filtros macro y timing: si se optimizan demasiado, el backtest puede parecer convincente pero perder validez fuera de muestra.

---

## 10. Referencias orientativas

Estas referencias sirven como punto de partida para una revisión bibliográfica y no sustituyen una replicación propia:

- Asness, C., Moskowitz, T., & Pedersen, L. H. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Brunnermeier, M., Nagel, S., & Pedersen, L. H. (2008). *Carry Trades and Currency Crashes*. NBER.
- Gorton, G., & Rouwenhorst, K. G. (2004). *Facts and Fantasies about Commodity Futures*. NBER.
- Erb, C., & Harvey, C. (2006). *The Strategic and Tactical Value of Commodity Futures*. Financial Analysts Journal.
- Campbell, J. Y., & Shiller, R. J. (1991). *Yield Spreads and Interest Rate Movements: A Bird’s Eye View*. Review of Economic Studies.
- Federal Reserve Bank of St. Louis. *ALFRED: Archival Federal Reserve Economic Data*.
- OECD. *Composite Leading Indicator*.

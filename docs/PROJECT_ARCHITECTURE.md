# 01. Problema

## Objetivo

Diseñar una estrategia intradía sobre **SPY** a frecuencia **5 minutos**, sin overnight, usando:

- HMM para detectar regímenes latentes.

- Modelo predictivo supervisado para estimar dirección.

- Backtest realista con costes, slippage y ejecución causal.

La estrategia no busca maximizar accuracy, sino generar **PnL neto positivo y robusto**.

---

## Información disponible

En cada instante $begin:math:text$t$end:math:text$:

$begin:math:display$

X\_t \= \\text\{información disponible hasta el cierre de la vela \} t

$end:math:display$

La señal se calcula al cierre de la vela $begin:math:text$t$end:math:text$, pero la entrada se ejecuta en:

$begin:math:display$

Open\_\{t\+1\}

$end:math:display$

Esto evita usar el cierre de la vela $begin:math:text$t$end:math:text$ como precio de ejecución después de haberlo usado para construir la señal.

---

## Qué se predice

Se predice la dirección del retorno futuro de SPY en un horizonte corto:

$begin:math:display$

h \\in \\\{1\,2\,3\\\}

$end:math:display$

El horizonte principal será:

$begin:math:display$

h \= 2

$end:math:display$

es decir, **10 minutos**.

---

## Target

El retorno futuro se define como:

$begin:math:display$

r\_\{t\,t\+h\} \=

\\log\\left\(

\\frac\{Open\_\{t\+h\+1\}\}\{Open\_\{t\+1\}\}

\\right\)

$end:math:display$

La clase objetivo es:

$begin:math:display$

y\_t \=

\\begin\{cases\}

\+1 \& \\text\{si \} r\_\{t\,t\+h\} \> \\delta\_t \\\\

0 \& \\text\{si \} \|r\_\{t\,t\+h\}\| \\leq \\delta\_t \\\\

\-1 \& \\text\{si \} r\_\{t\,t\+h\} \< \-\\delta\_t

\\end\{cases\}

$end:math:display$

donde $begin:math:text$\\delta\_t$end:math:text$ es una zona neutral que debe cubrir costes y ruido.

---

## Zona neutral

La zona neutral se define como:

$begin:math:display$

\\delta\_t \=

\\max\(

\\text\{coste round\-trip\}\_t \+ \\text\{buffer\}\,

\\lambda \\sigma\_\{h\,t\}

\)

$end:math:display$

con valores iniciales:

```text

buffer = 0.5 bps

lambda = 0.25

h = 2

```

La idea es no forzar al modelo a predecir movimientos demasiado pequeños que no son operables después de costes.

---

## Edge real

Hay edge real si:

$begin:math:display$

E\[\\text\{posición\}\_t \\cdot r\_\{t\,t\+h\} \\mid X\_t\]

\-

\\text\{costes\}\_t

\> 0

$end:math:display$

No basta con:

- buena accuracy;

- buen AUC;

- buen resultado sin costes;

- buen resultado en train.

La estrategia solo tiene sentido si produce beneficio neto bajo ejecución realista.

---

## Hipótesis de mercado

La hipótesis principal es que SPY tiene distintos regímenes intradía:

- régimen de calma;

- régimen tendencial;

- régimen de reversión;

- régimen de alta volatilidad;

- régimen de shock o transición.

El HMM no es el predictor principal. Su función es estimar el régimen latente para condicionar el modelo predictivo.

---

## Regla de rechazo

Si la estrategia no sobrevive a:

- costes realistas;

- validación walk-forward;

- robustez de parámetros;

- comparación contra benchmarks simples;

entonces debe rechazarse.

# 02. Arquitectura

## Pipeline general

```text

raw_data

    ↓

cleaned_data

    ↓

features

    ↓

HMM

    ↓

modelo predictivo

    ↓

signal

    ↓

position

    ↓

backtest

    ↓

evaluation

```

---

## Objetivo de cada bloque

### `raw_data`

Datos originales:

- OHLCV de SPY a 5 minutos.

- Datos de volumen.

- Variables externas opcionales como VIX.

- Calendario de mercado.

---

### `cleaned_data`

Datos limpios y alineados:

- solo regular session;

- timestamps correctos;

- sin barras duplicadas;

- sin gaps críticos;

- precios ajustados si procede;

- eliminación de días incompletos.

---

### `features`

Construcción de variables predictivas y de régimen:

- retornos;

- volatilidad realizada;

- rango;

- volumen relativo;

- tendencia;

- drawdown intradía;

- variables temporales;

- features externas disponibles en $begin:math:text$t$end:math:text$.

---

### `HMM`

Modelo de regímenes latentes.

Produce:

- estado más probable;

- probabilidades por estado;

- entropía de régimen;

- persistencia de régimen.

---

### `model`

Modelo supervisado que predice:

$begin:math:display$

P\(y\_t \= \+1 \\mid X\_t\)\, \\quad

P\(y\_t \= 0 \\mid X\_t\)\, \\quad

P\(y\_t \= \-1 \\mid X\_t\)

$end:math:display$

Modelo base recomendado:

```text

Logistic Regression multinomial regularizada

```

Modelo challenger:

```text

XGBoost

```

---

### `signal`

Convierte probabilidades en decisión operativa:

```text

long / short / flat

```

La señal depende de:

- probabilidad direccional;

- zona neutral;

- régimen HMM;

- confianza del HMM;

- filtros de riesgo.

---

### `position`

Convierte señal en posición:

- tamaño;

- leverage;

- exposición máxima;

- cooldown;

- límites diarios;

- stop loss;

- cierre intradía.

---

### `backtest`

Backtest realista barra a barra:

- señal al cierre de $begin:math:text$t$end:math:text$;

- entrada en $begin:math:text$Open\_\{t\+1\}$end:math:text$;

- salida temporal;

- stops;

- costes;

- slippage;

- no overnight.

---

### `evaluation`

Evalúa:

- Sharpe neto;

- drawdown;

- profit factor;

- hit ratio;

- PnL por régimen;

- PnL por hora;

- robustez;

- ablation.

---

## Arquitectura final recomendada

```text

SPY 5min OHLCV

    ↓

cleaning + calendar alignment

    ↓

causal features

    ↓

Gaussian HMM K=4

    ↓

filtered HMM probabilities

    ↓

Logistic Regression

    ↓

probabilistic signal

    ↓

risk-managed position

    ↓

event-driven backtest

    ↓

walk-forward evaluation

```

---

## Principio clave

El HMM no debe ser tratado como una caja mágica que predice el mercado.

Debe ser usado como:

```text

extractor de régimen + filtro contextual

```

El edge debe venir de la interacción entre:

```text

features predictivas + régimen + gestión de riesgo + costes realistas

# 03. Datos

## Universo

```text
Activo: SPY
Frecuencia: 5 minutos
Sesión: regular session
Overnight: no
Horizonte: 1–3 velas
```

---

## Datos principales

### OHLCV

Campos mínimos:

```text
timestamp
open
high
low
close
volume
```

Uso:

- construir retornos;
- construir targets;
- estimar volatilidad;
- calcular rango;
- simular ejecución;
- medir volumen relativo.

Riesgos:

- timestamps mal alineados;
- barras faltantes;
- datos duplicados;
- precios no ajustados;
- usar precios no ejecutables;
- mezclar premarket con regular session.

Decisión:

```text
Usar solo regular session.
Eliminar días incompletos.
No permitir overnight.
```

---

## Volumen

Uso:

- detectar actividad anómala;
- normalizar liquidez;
- construir volumen relativo;
- filtrar señales en momentos ilíquidos.

Feature principal:

$begin:math:display$
RelVol\_t \=
\\frac\{Volume\_t\}
\{\\text\{mediana histórica del volumen en la misma barra intradía\}\}
$end:math:display$

Riesgo:

```text
No usar volumen total del día, porque no está disponible en tiempo real.
```

---

## Volatilidad

Uso:

- medir régimen;
- estimar neutral zone;
- ajustar tamaño de posición;
- activar kill switch;
- detectar shocks.

Medidas:

```text
realized_vol_3
realized_vol_6
realized_vol_12
realized_vol_24
range
ATR
```

Riesgo:

```text
La volatilidad debe calcularse solo con datos pasados.
```

---

## Tiempo intradía

Uso:

- modelar apertura;
- modelar cierre;
- capturar menor actividad en mitad de sesión;
- evitar señales cerca del cierre.

Features:

```text
bar_index
sin_time
cos_time
minutes_to_close
is_open_window
is_close_window
```

Riesgo:

```text
El modelo puede sobreaprender patrones horarios.
```

Decisión:

```text
Usar tiempo intradía en el modelo predictivo,
pero no como input principal del HMM.
```

---

## VIX y variables externas

Uso posible:

- proxy de riesgo sistémico;
- volatilidad implícita;
- aversión al riesgo;
- contexto de mercado.

Opciones:

```text
VIX previous close
VIX daily return previous close
VIX 5min si hay dato intradía real
```

Decisión inicial:

```text
Usar solo VIX previous close y VIX previous daily return.
```

Motivo:

```text
Evitar leakage con el close diario del mismo día.
```

---

## Calendario

Necesario para:

- festivos;
- medias sesiones;
- cambios horarios;
- sesiones incompletas;
- evitar overnight.

Decisión:

```text
Crear columna session.
Crear columna bar_index.
Eliminar sesiones con número anómalo de barras.
```

---

## Requisitos de calidad

Antes de modelar:

```text
sin timestamps duplicados
sin barras fuera de sesión
sin NaN críticos
sin saltos artificiales
sin targets cruzando cierre
sin features con futuro
```

---

## Output esperado

Archivo limpio:

```text
data/cleaned/spy_5min_clean.parquet
```

Columnas mínimas:

```text
timestamp
session
bar_index
open
high
low
close
volume
```


# 04. Features

## Principio general

Todas las features deben estar disponibles en $begin:math:text$t$end:math:text$.

La regla es:

```text
Features calculadas al cierre de la vela t.
Entrada simulada en open_{t+1}.
```

Nunca se usan datos de $begin:math:text$t\+1$end:math:text$ o posteriores para construir $begin:math:text$X\_t$end:math:text$.

---

# Features para HMM

El HMM debe recibir variables que describan el estado del mercado, no necesariamente las más predictivas.

## Retornos

```text
ret_1
ret_3
```

Definición:

$begin:math:display$
ret\\\_1\_t \= \\log\\left\(\\frac\{Close\_t\}\{Close\_\{t\-1\}\}\\right\)
$end:math:display$

$begin:math:display$
ret\\\_3\_t \= \\log\\left\(\\frac\{Close\_t\}\{Close\_\{t\-3\}\}\\right\)
$end:math:display$

Uso:

- dirección reciente;
- presión compradora/vendedora;
- estado tendencial.

---

## Volatilidad realizada

```text
rv_6
rv_12
```

Definición:

$begin:math:display$
rv\_\{n\,t\} \=
\\sqrt\{
\\sum\_\{i\=t\-n\+1\}\^\{t\} ret\_i\^2
\}
$end:math:display$

Uso:

- diferenciar calma de alta volatilidad;
- detectar shocks;
- separar régimen normal de régimen estresado.

---

## Rango

```text
range_t = log(high_t / low_t)
```

Uso:

- medir amplitud intrabar;
- proxy de microvolatilidad;
- detección de eventos.

---

## Volumen relativo

```text
rel_volume_t =
volume_t / median_volume_same_bar_past_N_days
```

Uso:

- detectar actividad inusual;
- distinguir movimientos con o sin participación.

Riesgo:

```text
No usar volumen futuro del día.
```

---

## Tendencia

```text
trend_12_t = close_t / sma_12_t - 1
```

Uso:

- identificar momentum local;
- separar tendencia de ruido.

---

## Drawdown intradía

```text
intraday_drawdown_t =
close_t / max(close desde apertura hasta t) - 1
```

Uso:

- identificar presión bajista;
- distinguir selloff de volatilidad simétrica.

---

# Features para modelo predictivo

## Momentum

```text
ret_1
ret_2
ret_3
ret_6
ret_12
trend_6
trend_12
```

Uso:

- capturar continuación de corto plazo;
- detectar presión direccional.

---

## Reversión

```text
zscore_ret_1
zscore_ret_3
dist_vwap
dist_sma_12
rsi_like_6
```

Uso:

- capturar sobreextensión;
- identificar mean reversion intradía.

---

## Volatilidad

```text
rv_3
rv_6
rv_12
range
atr_6
range_zscore
```

Uso:

- ajustar umbrales;
- detectar condiciones de mercado;
- evitar operar ruido extremo.

---

## Volumen

```text
rel_volume
volume_zscore
dollar_volume
```

Uso:

- confirmar movimientos;
- filtrar barras ilíquidas;
- detectar eventos.

---

## Tiempo

```text
sin_time
cos_time
minutes_to_close
open_window
close_window
midday
```

Uso:

- capturar estacionalidad intradía;
- evitar operar demasiado cerca del cierre;
- distinguir apertura, mitad y cierre.

---

## HMM

El modelo predictivo recibe:

```text
hmm_state_argmax
hmm_p0
hmm_p1
hmm_p2
hmm_p3
hmm_entropy
hmm_max_prob
```

Donde:

$begin:math:display$
hmm\\\_entropy\_t \=
\-\\sum\_k p\_\{k\,t\}\\log\(p\_\{k\,t\}\)
$end:math:display$

Uso:

- condicionar la señal por régimen;
- medir incertidumbre;
- evitar operar cuando el régimen no está claro.

---

## Lista inicial de features HMM

```python
HMM_COLS = [
    "ret_1",
    "ret_3",
    "rv_6",
    "rv_12",
    "range",
    "rel_volume",
    "trend_12",
    "intraday_drawdown",
]
```

---

## Lista inicial de features del modelo

```python
MODEL_COLS = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_6",
    "ret_12",
    "rv_3",
    "rv_6",
    "rv_12",
    "range",
    "atr_6",
    "trend_6",
    "trend_12",
    "dist_vwap",
    "rel_volume",
    "sin_time",
    "cos_time",
    "minutes_to_close",
    "hmm_p0",
    "hmm_p1",
    "hmm_p2",
    "hmm_p3",
    "hmm_entropy",
    "hmm_max_prob",
]
```


# 05. HMM

## Objetivo

El HMM se usa para identificar regímenes latentes intradía.

No se usa como predictor directo de retorno.

Su función es generar variables de contexto:

```text
estado más probable
probabilidades por estado
entropía
confianza de régimen
persistencia
```

---

## Modelo base

Modelo recomendado:

```text
Gaussian HMM
```

Configuración inicial:

```python
GaussianHMM(
    n_components=4,
    covariance_type="diag",
    n_iter=500,
    random_state=seed
)
```

---

## Inputs

```python
HMM_COLS = [
    "ret_1",
    "ret_3",
    "rv_6",
    "rv_12",
    "range",
    "rel_volume",
    "trend_12",
    "intraday_drawdown",
]
```

Estos inputs capturan:

- dirección reciente;
- volatilidad;
- amplitud de vela;
- volumen;
- tendencia;
- presión bajista intradía.

---

## Normalización

La normalización se ajusta solo en train.

Pipeline:

```text
winsorization en train
RobustScaler o StandardScaler fit en train
transform en validation/test
```

Prohibido:

```text
hacer fit del scaler con todo el dataset
```

---

## Separación por sesiones

El HMM debe entrenarse respetando sesiones independientes.

Ejemplo:

```python
hmm.fit(X_train_hmm, lengths=session_lengths_train)
```

Motivo:

```text
Evitar que el modelo aprenda transiciones artificiales de 16:00 a 9:30.
```

---

## Número de estados

Candidatos:

```text
K ∈ {2, 3, 4, 5, 6}
```

Decisión inicial:

```text
K = 4
```

Justificación:

- suficientemente rico para distinguir calma, tendencia, selloff y shock;
- no demasiado complejo;
- más estable que K alto;
- interpretable.

---

## Selección de estados

Criterios:

```text
BIC/AIC en train
ocupación mínima
estabilidad entre seeds
interpretabilidad económica
impacto OOS en ablation
```

Restricciones:

```text
ocupación mínima por estado > 5%
ningún estado debe dominar > 80%
ningún estado debe aparecer solo en un periodo concreto
```

---

## Interpretación económica

La numeración del HMM es arbitraria. Después de entrenar se deben interpretar los estados según sus estadísticas.

Ejemplo:

| Estado | Posible interpretación | Rasgos |
|---|---|---|
| 0 | Calma | baja volatilidad, bajo rango |
| 1 | Tendencia positiva | retornos positivos, vol media |
| 2 | Selloff | retornos negativos, alta vol |
| 3 | Shock/transición | alta vol, alto volumen |

---

## Uso de probabilidades

No usar solo:

```text
estado más probable
```

Usar también:

```text
probabilidad de cada estado
entropía
probabilidad máxima
```

Motivo:

```text
La incertidumbre del régimen es información útil.
```

---

## Filtrado online

Importante:

```text
No usar smoothing forward-backward en test.
```

Se necesita:

$begin:math:display$
P\(z\_t \= k \\mid x\_1\, x\_2\, \.\.\.\, x\_t\)
$end:math:display$

No:

$begin:math:display$
P\(z\_t \= k \\mid x\_1\, x\_2\, \.\.\.\, x\_T\)
$end:math:display$

La segunda usa futuro y genera leakage.

---

## Pseudocódigo filtro online

```python
def online_hmm_filter(model, X_day):
    log_likelihood = model._compute_log_likelihood(X_day)
    log_A = np.log(model.transmat_)
    log_pi = np.log(model.startprob_)

    alphas = []

    log_alpha = log_pi + log_likelihood[0]
    log_alpha = normalize_log_probs(log_alpha)
    alphas.append(np.exp(log_alpha))

    for t in range(1, len(X_day)):
        pred = logsumexp(log_alpha[:, None] + log_A, axis=0)
        log_alpha = pred + log_likelihood[t]
        log_alpha = normalize_log_probs(log_alpha)
        alphas.append(np.exp(log_alpha))

    return np.array(alphas)
```

---

## Diagnóstico de estabilidad

Para cada fold y seed:

```text
ocupación por estado
media de retornos por estado
media de volatilidad por estado
matriz de transición
duración media del estado
PnL por estado
```

Rechazar el HMM si:

```text
los estados cambian radicalmente entre seeds
un estado desaparece
las probabilidades son casi uniformes siempre
no mejora al modelo sin HMM
```


# 06. Modelo predictivo

## Objetivo

El modelo predictivo estima:

$begin:math:display$
P\(y\_t \= \+1 \\mid X\_t\)
$end:math:display$

$begin:math:display$
P\(y\_t \= 0 \\mid X\_t\)
$end:math:display$

$begin:math:display$
P\(y\_t \= \-1 \\mid X\_t\)
$end:math:display$

donde $begin:math:text$X\_t$end:math:text$ incluye:

- features de precio;
- features de volatilidad;
- features de volumen;
- tiempo intradía;
- probabilidades HMM;
- entropía HMM.

---

## Modelos evaluados

Se comparan dos modelos:

```text
Logistic Regression
XGBoost
```

---

# Logistic Regression

## Ventajas

- robusta;
- interpretable;
- menos propensa a overfitting;
- probabilidades más controlables;
- buena baseline cuantitativa;
- fácil de calibrar;
- rápida de entrenar en walk-forward.

## Desventajas

- captura peor no linealidades;
- depende más de la calidad de features;
- puede quedarse corta si hay interacciones fuertes.

---

# XGBoost

## Ventajas

- captura no linealidades;
- maneja interacciones complejas;
- puede explotar efectos régimen × momentum × volatilidad.

## Desventajas

- mayor riesgo de overfitting;
- calibración peor;
- más hiperparámetros;
- puede aprender artefactos temporales;
- mayor riesgo de optimización accidental contra validation.

---

## Decisión

Modelo principal:

```text
Logistic Regression multinomial regularizada
```

Modelo challenger:

```text
XGBoost
```

La estrategia se desarrolla primero con Logistic Regression. XGBoost solo se acepta si mejora OOS de forma estable, no solo en train.

---

## Configuración inicial

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.2,
    C=0.1,
    class_weight="balanced",
    max_iter=5000,
    multi_class="multinomial"
)
```

---

## Uso del HMM

El HMM entra en el modelo como features:

```text
hmm_p0
hmm_p1
hmm_p2
hmm_p3
hmm_entropy
hmm_max_prob
hmm_state_onehot
```

No se recomienda entrenar un modelo distinto por estado al principio.

Primera arquitectura:

```text
features base + HMM probabilities → Logistic Regression
```

Arquitectura alternativa para ablation:

```text
modelo separado por régimen
```

---

## Calibración

Las probabilidades deben calibrarse temporalmente.

Proceso:

```text
fit modelo en train
calibrar probabilidades en validation
usar probabilidades calibradas en test
```

Métodos posibles:

```text
Platt scaling
isotonic calibration
temperature scaling
```

Decisión inicial:

```text
CalibratedClassifierCV con split temporal manual
```

---

## Output del modelo

El modelo devuelve:

```text
p_down
p_neutral
p_up
```

A partir de ellas:

$begin:math:display$
score\_t \= p\\\_up\_t \- p\\\_down\_t
$end:math:display$

La señal no se basa solo en la clase con mayor probabilidad, sino en umbrales de probabilidad y edge esperado.

---

## Criterio de aceptación del modelo

El modelo debe mejorar:

```text
benchmark random
momentum simple
reversión simple
modelo sin HMM
```

y debe hacerlo:

```text
después de costes
en walk-forward
en varios periodos
sin concentración extrema del PnL
```



# 07. Target

## Definición

En cada instante $begin:math:text$t$end:math:text$:

$begin:math:display$
X\_t \= \\text\{información disponible hasta el cierre de la vela \} t
$end:math:display$

La entrada simulada ocurre en:

$begin:math:display$
Open\_\{t\+1\}
$end:math:display$

La salida temporal ocurre en:

$begin:math:display$
Open\_\{t\+h\+1\}
$end:math:display$

Por tanto, el retorno futuro es:

$begin:math:display$
r\_\{t\,t\+h\}
\=
\\log\\left\(
\\frac\{Open\_\{t\+h\+1\}\}\{Open\_\{t\+1\}\}
\\right\)
$end:math:display$

---

## Horizonte

Horizonte principal:

```text
h = 2
```

Equivale a:

```text
10 minutos
```

Horizontes de robustez:

```text
h = 1
h = 3
```

---

## Clases

Se define una clasificación ternaria:

$begin:math:display$
y\_t \=
\\begin\{cases\}
\+1 \& \\text\{si \} r\_\{t\,t\+h\} \> \\delta\_t \\\\
0 \& \\text\{si \} \|r\_\{t\,t\+h\}\| \\leq \\delta\_t \\\\
\-1 \& \\text\{si \} r\_\{t\,t\+h\} \< \-\\delta\_t
\\end\{cases\}
$end:math:display$

Interpretación:

```text
+1: oportunidad long
0: no trade
-1: oportunidad short
```

---

## Zona neutral

La zona neutral evita etiquetar ruido como señal.

$begin:math:display$
\\delta\_t \=
\\max\(
\\text\{coste round\-trip\}\_t \+ \\text\{edge buffer\}\,
\\lambda \\sigma\_\{h\,t\}
\)
$end:math:display$

Valores iniciales:

```text
edge_buffer = 0.5 bps
lambda = 0.25
```

---

## Estimación de sigma

$begin:math:display$
\\sigma\_\{h\,t\}
$end:math:display$

debe estimarse solo con datos disponibles hasta $begin:math:text$t$end:math:text$.

Ejemplo:

```python
sigma_h_t = realized_vol_12_t * sqrt(h)
```

No usar volatilidad futura realizada entre $begin:math:text$t\+1$end:math:text$ y $begin:math:text$t\+h$end:math:text$.

---

## Evitar leakage

Prohibido:

```text
usar close_t como precio de entrada
usar open_{t+1} en features
usar high/low futuros
usar volatilidad futura
usar datos posteriores al cierre de t
permitir que el target cruce overnight
normalizar con todo el dataset
```

Permitido:

```text
usar OHLCV completo de la vela t
entrar en open_{t+1}
salir en open_{t+h+1}
usar rolling features cerradas en t
```

---

## Pseudocódigo

```python
def make_labels(df, h, cost_bps, edge_buffer_bps, neutral_vol_mult):
    df = df.copy()

    df["entry_px"] = df.groupby("session")["open"].shift(-1)
    df["exit_px"] = df.groupby("session")["open"].shift(-(h + 1))

    df["fwd_ret"] = np.log(df["exit_px"] / df["entry_px"])

    df["sigma_h"] = df["rv_12"] * np.sqrt(h)

    cost_ret = bps_to_ret(cost_bps + edge_buffer_bps)

    df["neutral_band"] = np.maximum(
        cost_ret,
        neutral_vol_mult * df["sigma_h"]
    )

    df["y"] = 0
    df.loc[df["fwd_ret"] > df["neutral_band"], "y"] = 1
    df.loc[df["fwd_ret"] < -df["neutral_band"], "y"] = -1

    df = df.dropna(subset=["entry_px", "exit_px", "y"])

    return df
```

---

## Regla crítica

El target debe representar una oportunidad operable después de costes.

Si el target clasifica como positivos movimientos que son menores que el coste total, el modelo aprenderá ruido.



# 08. Señal

## Inputs

El modelo produce:

```text
p_up      = P(y = +1 | X_t)
p_neutral = P(y = 0  | X_t)
p_down    = P(y = -1 | X_t)
```

A partir de esto se define:

$begin:math:display$
score\_t \= p\\\_up\_t \- p\\\_down\_t
$end:math:display$

---

## Señal básica

Regla inicial:

```python
if p_up > threshold:
    signal = +1
elif p_down > threshold:
    signal = -1
else:
    signal = 0
```

Pero esta regla es demasiado simple.

---

## Señal recomendada

Parámetros iniciales:

```text
theta_prob = 0.55
theta_score = 0.10
max_neutral = 0.55
max_hmm_entropy = 0.90
```

Regla:

```python
if hmm_entropy_t > max_hmm_entropy:
    signal = 0

elif p_up > theta_prob and score_t > theta_score and p_neutral < max_neutral:
    signal = +1

elif p_down > theta_prob and score_t < -theta_score and p_neutral < max_neutral:
    signal = -1

else:
    signal = 0
```

---

## Interpretación

Se abre long si:

```text
la probabilidad alcista es suficientemente alta
la diferencia p_up - p_down es positiva
la probabilidad neutral no domina
el régimen HMM no es demasiado incierto
```

Se abre short si:

```text
la probabilidad bajista es suficientemente alta
la diferencia p_up - p_down es negativa
la probabilidad neutral no domina
el régimen HMM no es demasiado incierto
```

---

## Filtros por régimen

Para cada régimen $begin:math:text$k$end:math:text$, se estima en validation:

```text
PnL long en régimen k
PnL short en régimen k
número de trades
profit factor
avg trade net
```

Se permite operar un lado solo si:

```text
n_trades >= min_trades
avg_trade_net > 0
profit_factor > 1.05
```

Parámetro inicial:

```text
min_trades = 30
```

---

## Pseudocódigo

```python
def generate_signal(row, thresholds, regime_allow):
    p_up = row["p_up"]
    p_down = row["p_down"]
    p_neutral = row["p_neutral"]

    score = p_up - p_down
    regime = row["hmm_state"]

    if row["hmm_entropy"] > thresholds["max_hmm_entropy"]:
        return 0

    if p_up > thresholds["theta_prob"]:
        if score > thresholds["theta_score"] and p_neutral < thresholds["max_neutral"]:
            if regime_allow[regime]["long"]:
                return +1

    if p_down > thresholds["theta_prob"]:
        if score < -thresholds["theta_score"] and p_neutral < thresholds["max_neutral"]:
            if regime_allow[regime]["short"]:
                return -1

    return 0
```

---

## Reglas adicionales

No abrir nueva señal si:

```text
quedan menos de 15 minutos para el cierre
kill switch activo
drawdown diario excedido
spread estimado demasiado alto
volatilidad extrema
cooldown activo
máximo de trades diario alcanzado
```

---

## Objetivo de la señal

La señal debe ser selectiva.

En intradía, operar demasiado suele destruir el edge por:

```text
spread
slippage
comisiones
ruido
sobreajuste
```


# 09. Risk Management

## Objetivo

El risk management debe evitar que una señal estadística débil se convierta en una estrategia frágil.

Debe controlar:

```text
tamaño de posición
pérdida por trade
pérdida diaria
número de trades
exposición
cierre intradía
condiciones anómalas
```

---

## Posición base

La posición puede ser:

```text
-1 short
 0 flat
+1 long
```

Primera versión:

```text
sin leverage
posición fija
```

Después:

```text
volatility scaling
```

---

## Sizing

Tamaño base:

$begin:math:display$
Notional\_t \= Equity\_t \\cdot leverage\_t
$end:math:display$

Con:

```text
base_leverage = 1.0
max_leverage = 1.0
```

---

## Volatility scaling

$begin:math:display$
vol\\\_scale\_t \=
clip\\left\(
\\frac\{\\sigma\_\{target\}\}\{\\sigma\_\{h\,t\}\}\,
0\.25\,
1\.00
\\right\)
$end:math:display$

Entonces:

$begin:math:display$
Notional\_t \=
Equity\_t \\cdot base\\\_leverage \\cdot vol\\\_scale\_t
$end:math:display$

Si el HMM tiene baja confianza:

```python
if hmm_max_prob < 0.50:
    vol_scale *= 0.5
```

---

## Stop loss

Stop inicial:

$begin:math:display$
stop\\\_distance\_t \=
\\max\(
1\.5 \\cdot ATR\_\{6\,t\}\,
2\.0 \\cdot \\sigma\_\{h\,t\} \\cdot Price\_t
\)
$end:math:display$

Para long:

$begin:math:display$
stop \= entry \- stop\\\_distance
$end:math:display$

Para short:

$begin:math:display$
stop \= entry \+ stop\\\_distance
$end:math:display$

---

## Time stop

La posición se cierra por tiempo en:

```text
open_{t+h+1}
```

con:

```text
h = 2
```

---

## Stop intrabar

Si se usan barras OHLCV y el stop se toca dentro de una vela:

```text
long: si low <= stop, se ejecuta stop
short: si high >= stop, se ejecuta stop
```

Si stop y take-profit ocurren en la misma vela:

```text
asumir peor caso
```

---

## Límite de pérdida diaria

```text
max_daily_loss = -0.75% del equity
```

Si se alcanza:

```text
cerrar posición
no abrir más trades ese día
```

---

## Límite de trades

```text
max_trades_per_day = 15
cooldown_after_trade = 2 velas
```

Motivo:

```text
reducir churn
reducir costes
evitar sobreoperar ruido
```

---

## Cierre intradía

Reglas:

```text
no abrir nuevas posiciones después de 15:45 ET
cerrar cualquier posición antes de 15:55 ET
no mantener overnight
```

---

## Kill switch

Activar flat si:

```text
datos faltantes
NaN en features críticas
spread estimado excesivo
volatilidad > percentil 99 train
drawdown diario excedido
latencia anómala
HMM entropy alta durante muchas barras
órdenes rechazadas
```

---

## Reglas de rechazo

La estrategia se rechaza si:

```text
necesita leverage alto para ser atractiva
depende de stops optimistas
pierde más en costes que lo que gana en señal
tiene drawdowns intradía no asumibles
```



# 10. Costes

## Principio

En intradía, los costes no son un detalle.

Una estrategia puede parecer rentable antes de costes y desaparecer completamente después de:

```text
spread
slippage
comisiones
fees
impacto
```

Por tanto, el backtest debe calcular siempre PnL neto.

---

## Coste por lado

$begin:math:display$
cost\\\_side\_t \=
commission\_t
\+
half\\\_spread\_t
\+
slippage\_t
\+
fees\_t
\+
impact\_t
$end:math:display$

Coste round-trip:

$begin:math:display$
cost\\\_\{rt\,t\} \= 2 \\cdot cost\\\_side\_t
$end:math:display$

---

## Escenarios de coste

Si solo se tiene OHLCV, usar escenarios.

| Escenario | Coste round-trip |
|---|---:|
| cero | 0 bps |
| base | 1 bps |
| conservador | 2 bps |
| stress | 5 bps |

---

## Regla

```text
El escenario cero solo sirve como diagnóstico.
No sirve para aceptar la estrategia.
```

La estrategia debe sobrevivir al menos al escenario base.

---

## Slippage

Modelo inicial:

$begin:math:display$
slippage\\\_bps\_t \=
base\\\_slippage\\\_bps
\+
impact\\\_bps\_t
$end:math:display$

Impacto aproximado:

$begin:math:display$
impact\\\_bps\_t \=
impact\\\_coef \\cdot \\sqrt\{participation\_t\}
$end:math:display$

donde:

$begin:math:display$
participation\_t \=
\\frac\{order\\\_size\_t\}\{bar\\\_volume\_t\}
$end:math:display$

Restricción:

```text
participation_t < 1%
```

---

## Spread

Si no se tiene bid/ask:

```text
usar estimación conservadora en bps
```

Para SPY, el spread suele ser bajo, pero en una estrategia de 5 minutos incluso costes pequeños importan.

---

## Comisiones

Parámetro configurable:

```yaml
commission_per_share: 0.0035
min_commission: 0.35
```

O simplificación inicial:

```yaml
commission_bps: 0.1
```

---

## Coste total en el backtest

Para long:

$begin:math:display$
PnL \=
\(exit \- entry\) \\cdot shares
\-
costes
$end:math:display$

Para short:

$begin:math:display$
PnL \=
\(entry \- exit\) \\cdot shares
\-
costes
$end:math:display$

---

## Coste en target

La zona neutral debe incluir costes:

$begin:math:display$
\\delta\_t \\geq cost\\\_\{rt\,t\}
$end:math:display$

Si no, el modelo aprende movimientos que no son operables.

---

## Reglas de aceptación

La estrategia debe reportarse en:

```text
0 bps
1 bps
2 bps
5 bps
```

Y debe mostrar:

```text
Sharpe neto
avg trade net
profit factor neto
turnover
```

---

## Rechazo automático

Rechazar si:

```text
solo funciona a 0 bps
avg trade net < coste plausible
beneficio desaparece con costes x2
necesita muchísimos trades para generar PnL
```


# 11. Backtesting

## Objetivo

El backtest debe aproximar una ejecución realista.

Regla central:

```text
La señal se calcula al cierre de t.
La entrada ocurre en open_{t+1}.
```

No se permite:

```text
usar close_t para generar señal y entrar en close_t
```

---

## Flujo por barra

Para cada vela $begin:math:text$t$end:math:text$:

```text
1. Cierra la vela t.
2. Se calculan features X_t.
3. Se actualizan probabilidades HMM filtradas.
4. El modelo predice probabilidades.
5. Se genera señal.
6. La orden se ejecuta en open_{t+1}.
7. La posición se cierra por:
   - time stop
   - stop loss
   - cierre intradía
   - kill switch
```

---

## Ejecución

Entrada:

$begin:math:display$
entry \= Open\_\{t\+1\} \+ slippage
$end:math:display$

Salida temporal:

$begin:math:display$
exit \= Open\_\{t\+h\+1\} \- slippage
$end:math:display$

Para short, el signo del slippage se invierte de manera desfavorable.

---

## Walk-forward

Esquema inicial:

```text
fit: 5 meses
validation: 1 mes
test: 1 mes
step: 1 mes
```

Ejemplo:

```text
Fold 1:
fit        Jan-May
validation Jun
test       Jul

Fold 2:
fit        Feb-Jun
validation Jul
test       Aug
```

---

## Qué se entrena en cada fold

En cada fold:

```text
1. Fit scaler HMM en train.
2. Fit HMM en train.
3. Generar HMM probs en train/validation/test con filtro online.
4. Fit scaler modelo en train.
5. Fit modelo predictivo en train.
6. Calibrar probabilidades en validation.
7. Elegir thresholds en validation.
8. Evaluar una sola vez en test.
```

---

## Purge y embargo

Como el target usa un horizonte $begin:math:text$h$end:math:text$, se debe evitar solapamiento entre train, validation y test.

```text
purge = h + 1 barras
embargo = h + 1 barras
```

---

## No overnight

Eliminar señales que no puedan cerrarse dentro de la sesión.

Regla:

```text
si t+h+1 excede la sesión, no crear target ni trade
```

---

## Pseudocódigo walk-forward

```python
for fold in walkforward_splits(data):

    train = data.loc[fold.train]
    val = data.loc[fold.val]
    test = data.loc[fold.test]

    features_train = make_features(train)
    features_val = make_features(val)
    features_test = make_features(test)

    labels_train = make_labels(train)
    labels_val = make_labels(val)
    labels_test = make_labels(test)

    hmm_scaler.fit(features_train[HMM_COLS])
    Xh_train = hmm_scaler.transform(features_train[HMM_COLS])

    hmm = fit_hmm(Xh_train, train_session_lengths)

    hmm_probs_train = online_filter_by_day(hmm, Xh_train)

    Xh_val = hmm_scaler.transform(features_val[HMM_COLS])
    hmm_probs_val = online_filter_by_day(hmm, Xh_val)

    Xh_test = hmm_scaler.transform(features_test[HMM_COLS])
    hmm_probs_test = online_filter_by_day(hmm, Xh_test)

    X_train = join_model_features(features_train, hmm_probs_train)
    X_val = join_model_features(features_val, hmm_probs_val)
    X_test = join_model_features(features_test, hmm_probs_test)

    model_scaler.fit(X_train)
    X_train_scaled = model_scaler.transform(X_train)
    X_val_scaled = model_scaler.transform(X_val)
    X_test_scaled = model_scaler.transform(X_test)

    clf = fit_model(X_train_scaled, labels_train)

    thresholds = select_thresholds(
        clf,
        X_val_scaled,
        labels_val,
        cost_model
    )

    trades = run_backtest(
        test,
        X_test_scaled,
        clf,
        thresholds,
        hmm_probs_test,
        risk_config,
        cost_model
    )

    save_results(fold, trades)
```

---

## Backtest event-driven

Debe ser preferible a uno puramente vectorizado porque permite:

```text
stops intrabar
cooldown
kill switch
límite de trades
estado de posición
cierre intradía
```

---

## Reglas de rechazo

Rechazar si:

```text
el backtest usa close_t como entrada
permite overnight accidental
no incluye costes
no implementa walk-forward
selecciona parámetros mirando test
```


# 12. Evaluación

## Objetivo

Evaluar si la estrategia genera edge real después de costes, no si predice bien en abstracto.

La evaluación principal debe hacerse sobre:

```text
PnL neto diario
trades netos
drawdown
estabilidad OOS
```

---

## Métricas principales

```text
Sharpe neto
max drawdown
profit factor
hit ratio
avg trade net bps
median trade net bps
número de trades
turnover
exposure
PnL por régimen
PnL por hora del día
PnL long vs short
```

---

## Sharpe

Calcular sobre retornos diarios netos:

$begin:math:display$
Sharpe \=
\\frac\{
mean\(r\_\{daily\}\)
\}\{
std\(r\_\{daily\}\)
\}
\\sqrt\{252\}
$end:math:display$

No calcular Sharpe principal sobre trades individuales.

---

## Max drawdown

$begin:math:display$
DD\_t \=
\\frac\{Equity\_t\}\{\\max\_\{s \\leq t\} Equity\_s\} \- 1
$end:math:display$

Reportar:

```text
max drawdown
duración del drawdown
tiempo hasta recuperación
```

---

## Profit factor

$begin:math:display$
ProfitFactor \=
\\frac\{
GrossProfit
\}\{
\|GrossLoss\|
\}
$end:math:display$

Criterio mínimo inicial:

```text
Profit factor > 1.10
```

---

## Hit ratio

$begin:math:display$
HitRatio \=
\\frac\{
\\text\{número de trades ganadores\}
\}\{
\\text\{número total de trades\}
\}
$end:math:display$

No debe interpretarse solo.

Una estrategia puede tener bajo hit ratio y ser rentable si el payoff ratio es alto.

---

## Avg trade net

$begin:math:display$
AvgTradeNet \=
mean\(PnL\\\_neto\\\_por\\\_trade\)
$end:math:display$

Debe ser positivo después de costes.

Criterio crítico:

```text
avg trade net > 0
```

---

## PnL por régimen

Para cada trade, guardar:

```text
hmm_state_entry
hmm_max_prob_entry
hmm_entropy_entry
```

Reportar por régimen:

```text
n_trades
avg_net_bps
hit_ratio
profit_factor
total_pnl
max_drawdown
```

---

## PnL por hora

Agrupar por:

```text
bar_index
hora
bloque intradía
```

Bloques:

```text
open
morning
midday
afternoon
close
```

Objetivo:

```text
detectar si el edge viene solo de una franja horaria.
```

---

## Calibración de probabilidades

Evaluar:

```text
Brier score
calibration curve
reliability plot
predicted probability buckets
realized hit ratio por bucket
```

La señal debe ser más rentable cuando el modelo tiene mayor confianza.

---

## Métricas mínimas de aceptación

```text
Sharpe neto OOS > 1.0
Profit factor > 1.10
Avg trade net > 0
Max drawdown razonable
Resultado positivo en varios folds
HMM mejora frente a modelo sin HMM
Costes x2 no destruyen completamente el resultado
```

---

## Señales de alerta

```text
PnL concentrado en pocos días
Sharpe alto con pocos trades
mucho mejor train que test
drawdown creciente en test
profit factor frágil
rentabilidad solo en 0 bps
```



# 13. Robustez

## Objetivo

Comprobar que la estrategia no depende de una configuración concreta.

La robustez no busca encontrar el mejor parámetro, sino verificar que el comportamiento es estable.

---

## Parámetros a testear

### Horizonte

```text
h ∈ {1, 2, 3}
```

El horizonte principal es $begin:math:text$h\=2$end:math:text$. Los demás son pruebas de robustez.

---

### Estados HMM

```text
K ∈ {2, 3, 4, 5, 6}
```

Comprobar:

```text
ocupación por estado
interpretabilidad
estabilidad OOS
mejora en ablation
```

---

### Costes

```text
round-trip cost ∈ {1, 2, 5 bps}
```

Regla:

```text
Si solo funciona con costes bajos irreales, rechazar.
```

---

### Umbrales

```text
theta_prob ∈ {0.53, 0.55, 0.57, 0.60}
theta_score ∈ {0.05, 0.10, 0.15}
neutral_vol_mult ∈ {0.15, 0.25, 0.35}
```

---

### Ventana de entrenamiento

```text
train_window ∈ {3, 6, 12 meses}
```

Objetivo:

```text
ver si el modelo necesita demasiada historia o si se adapta rápido.
```

---

## Robustez temporal

Separar resultados por:

```text
años
trimestres
meses
alta volatilidad
baja volatilidad
mercado alcista
mercado bajista
periodos macro
```

---

## Robustez por sesión

Separar por:

```text
apertura
mañana
midday
tarde
cierre
```

Si todo el PnL viene de una sola franja, analizar si tiene sentido económico.

---

## Robustez de HMM

Para cada fold y seed:

```text
state occupancy
mean return by state
mean volatility by state
transition matrix
state duration
state persistence
```

Rechazar HMM si:

```text
los estados no son estables
los estados no son interpretables
las probabilidades son siempre similares
un estado domina casi todo
```

---

## Robustez de costes

Test obligatorio:

```text
base cost
2x base cost
5 bps round-trip
```

Criterio:

```text
La estrategia puede empeorar, pero no debería colapsar completamente con costes razonables.
```

---

## Robustez de seeds

Para HMM y modelos no deterministas:

```text
seeds = [1, 3, 7, 11, 42]
```

Reportar:

```text
media
desviación
percentiles
peor seed
```

---

## Robustez de features

Eliminar bloques de features:

```text
sin volumen
sin volatilidad
sin tiempo
sin HMM
sin VIX
```

Objetivo:

```text
ver qué bloque aporta realmente.
```

---

## Rechazo

Rechazar si:

```text
el resultado depende de un solo parámetro
solo funciona en un periodo
solo funciona en un seed
solo funciona con un coste irreal
solo funciona con una configuración concreta de HMM
```


# 14. Ablation

## Objetivo

Determinar qué componentes aportan valor real.

La pregunta principal:

```text
¿El HMM mejora realmente la estrategia?
```

No se acepta el HMM por elegancia matemática. Se acepta solo si mejora resultados OOS después de costes.

---

## Experimentos

| Código | Modelo | Objetivo |
|---|---|---|
| A0 | Logistic sin HMM | baseline predictivo |
| A1 | Logistic + hard HMM state | valor del estado discreto |
| A2 | Logistic + HMM probabilities | valor de incertidumbre de régimen |
| A3 | Logistic + HMM filters | valor como filtro operativo |
| A4 | modelos separados por régimen | dependencia régimen-específica |
| A5 | XGBoost sin HMM | challenger no lineal |
| A6 | XGBoost + HMM probs | challenger completo |

---

## A0: Logistic sin HMM

Features:

```text
precio
volatilidad
volumen
tiempo
VIX opcional
```

Sin:

```text
hmm_state
hmm_probs
hmm_entropy
```

Objetivo:

```text
medir el edge base sin regímenes.
```

---

## A1: Logistic + hard state

Añadir:

```text
hmm_state_argmax
```

Codificado one-hot.

Objetivo:

```text
ver si el estado discreto aporta información.
```

---

## A2: Logistic + HMM probabilities

Añadir:

```text
hmm_p0
hmm_p1
hmm_p2
hmm_p3
hmm_entropy
hmm_max_prob
```

Objetivo:

```text
ver si la incertidumbre de régimen aporta más que el estado duro.
```

---

## A3: Logistic + filtros HMM

Mismo modelo que A0 o A2, pero con filtros:

```text
permitir long solo en ciertos regímenes
permitir short solo en ciertos regímenes
bloquear señales con alta entropía
```

Objetivo:

```text
ver si el HMM es más útil como filtro que como feature.
```

---

## A4: Modelos separados por régimen

Entrenar un modelo por régimen dominante.

Riesgo:

```text
poca muestra por régimen
sobreajuste
inestabilidad
```

Solo usar si A2/A3 muestran evidencia fuerte.

---

## A5: XGBoost sin HMM

Challenger no lineal.

Objetivo:

```text
ver si la no linealidad mejora al modelo lineal.
```

---

## A6: XGBoost + HMM

Modelo complejo completo.

Aceptar solo si:

```text
mejora OOS
mejora después de costes
no aumenta demasiado turnover
no reduce estabilidad
```

---

## Métricas por ablation

Para cada experimento:

```text
Sharpe neto
profit factor
avg trade net
max drawdown
número de trades
turnover
PnL por régimen
cost stress
estabilidad por fold
```

---

## Criterio de aceptación del HMM

Mantener HMM solo si:

```text
A2 o A3 mejora a A0
la mejora es OOS
la mejora sobrevive a costes
la mejora aparece en varios folds
la mejora no viene de un solo mes
```

---

## Criterio de rechazo del HMM

Eliminar HMM si:

```text
no mejora al baseline
añade inestabilidad
solo mejora en train
solo funciona con un K concreto
aumenta turnover sin mejorar PnL neto
```


# 15. Benchmarks

## Objetivo

La estrategia debe superar alternativas simples.

Si no supera benchmarks básicos, no merece complejidad adicional.

---

## Benchmark 1: Buy & Hold

Benchmark clásico:

```text
comprar SPY y mantener
```

No es perfectamente comparable porque la estrategia no mantiene overnight, pero sirve como referencia general.

---

## Benchmark 2: Intraday Buy & Hold

Más comparable:

```text
comprar en open
vender en close
sin overnight
```

Objetivo:

```text
comparar contra exposición intradía simple.
```

---

## Benchmark 3: Momentum simple

Regla:

```python
if ret_3_t > threshold:
    signal = +1
elif ret_3_t < -threshold:
    signal = -1
else:
    signal = 0
```

La entrada se ejecuta en:

```text
open_{t+1}
```

Debe incluir los mismos costes que la estrategia principal.

---

## Benchmark 4: Reversión simple

Regla:

```python
if zscore_ret_3_t < -1:
    signal = +1
elif zscore_ret_3_t > 1:
    signal = -1
else:
    signal = 0
```

Objetivo:

```text
comparar contra mean reversion sencilla.
```

---

## Benchmark 5: Random

Benchmark aleatorio pero controlado.

Debe igualar:

```text
número de trades
proporción long/short
horario de entrada
holding period
```

Objetivo:

```text
descartar que el PnL venga solo de exposición o de operar ciertas horas.
```

---

## Benchmark 6: Modelo sin HMM

Mismo modelo predictivo:

```text
Logistic Regression
mismas features base
mismos costes
mismo backtest
```

Pero sin:

```text
hmm_state
hmm_probs
hmm_entropy
filtros HMM
```

Este es el benchmark más importante.

---

## Benchmark 7: Always Flat

```text
no operar nunca
```

Sirve para recordar que una estrategia con PnL negativo no añade valor.

---

## Comparación

Todos los benchmarks deben evaluarse con:

```text
mismo periodo
mismos costes
misma ejecución
mismo risk management básico
mismo walk-forward si aplica
```

---

## Criterio

La estrategia final debe superar:

```text
random benchmark
momentum simple
reversión simple
modelo sin HMM
```

Si no supera al modelo sin HMM, el HMM se elimina.


# 16. Estructura del proyecto

## Objetivo

Crear una estructura limpia, reproducible y escalable.

La estructura debe separar:

```text
datos
features
modelos
backtests
configuraciones
reportes
```

---

## Estructura recomendada

```text
project/
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── features/
│
├── src/
│   ├── data_loader.py
│   ├── calendar.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── labels.py
│   ├── hmm_model.py
│   ├── hmm_filter.py
│   ├── predictive_model.py
│   ├── signal.py
│   ├── risk.py
│   ├── costs.py
│   ├── execution.py
│   ├── backtest.py
│   ├── walkforward.py
│   ├── evaluation.py
│   ├── ablation.py
│   ├── robustness.py
│   └── utils.py
│
├── models/
│   ├── hmm/
│   ├── classifiers/
│   └── scalers/
│
├── backtest/
│   ├── trades/
│   ├── equity_curves/
│   └── fold_results/
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── hmm.yaml
│   ├── model.yaml
│   ├── signal.yaml
│   ├── risk.yaml
│   └── costs.yaml
│
├── reports/
│   ├── data_quality.md
│   ├── walkforward_summary.md
│   ├── regime_diagnostics.md
│   ├── ablation.md
│   └── robustness.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_diagnostics.ipynb
│   ├── 03_hmm_diagnostics.ipynb
│   └── 04_backtest_review.ipynb
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## `data/`

```text
raw: datos originales sin modificar
cleaned: datos limpios
features: matrices listas para modelar
```

No sobrescribir raw data.

---

## `src/`

Código modular.

Cada archivo debe tener una responsabilidad clara.

---

## `models/`

Guardar artefactos por fold:

```text
scaler_hmm
hmm_model
scaler_model
predictive_model
calibrator
thresholds
```

---

## `backtest/`

Guardar:

```text
trades por fold
equity curve por fold
resultados agregados
diagnósticos
```

---

## `configs/`

Toda configuración debe estar en YAML.

Evitar hiperparámetros hardcodeados.

---

## `reports/`

Reportes automáticos:

```text
calidad de datos
diagnóstico HMM
walk-forward
ablation
robustez
```

---

## Principio

El proyecto debe poder ejecutarse de nuevo desde cero con:

```bash
python -m src.walkforward --config configs/base.yaml
```


# 17. Librerías

## Core

```python
pandas
numpy
scipy
```

Uso:

```text
manipulación de datos
arrays
estadística
cálculos numéricos
```

---

## Machine Learning

```python
scikit-learn
xgboost
```

Uso:

```text
Logistic Regression
calibración
pipelines
scalers
métricas
XGBoost challenger
```

---

## HMM

Opciones:

```python
hmmlearn
pomegranate
```

Decisión inicial:

```text
hmmlearn
```

Motivo:

```text
suficiente para GaussianHMM
simple de integrar
rápido para prototipo
```

---

## Calendario de mercado

```python
exchange_calendars
pandas_market_calendars
```

Uso:

```text
regular session
festivos
medias sesiones
alineación temporal
```

---

## Almacenamiento

```python
pyarrow
fastparquet
```

Formato recomendado:

```text
parquet
```

Motivo:

```text
rápido
compacto
mantiene tipos
adecuado para data pipelines
```

---

## Serialización de modelos

```python
joblib
pickle
```

Recomendación:

```text
usar joblib para scalers/modelos sklearn
```

---

## Visualización

```python
matplotlib
plotly
```

Uso:

```text
equity curves
drawdowns
PnL por régimen
diagnósticos HMM
```

---

## Configuración

```python
pyyaml
omegaconf
```

Uso:

```text
leer configs YAML
mantener experimentos reproducibles
```

---

## Logging

```python
logging
loguru
```

Uso:

```text
tracking de folds
warnings de datos
errores de ejecución
diagnósticos
```

---

## Testing

```python
pytest
```

Tests mínimos:

```text
no leakage en labels
no overnight
features disponibles en t
walk-forward sin solapamiento
costes aplicados correctamente
```

---

## Librerías no prioritarias

Evitar inicialmente:

```text
backtesting.py
vectorbt
zipline
backtrader
```

Motivo:

```text
pueden ocultar detalles de ejecución.
```

Para esta estrategia conviene un backtester propio, explícito y simple.

# 18. Roadmap

## Fase 1: dataset limpio

Objetivo:

```text
SPY 5min limpio, regular session, sin gaps críticos.
```

Tareas:

```text
descargar datos
normalizar timestamps
filtrar regular session
detectar barras faltantes
eliminar días incompletos
guardar parquet limpio
```

Entregables:

```text
data/cleaned/spy_5min_clean.parquet
reports/data_quality.md
```

---

## Fase 2: baseline

Objetivo:

```text
medir si hay señal antes de añadir HMM.
```

Implementar:

```text
random benchmark
momentum simple
reversión simple
logistic sin HMM
```

Entregables:

```text
baseline_backtest.parquet
baseline_report.md
```

---

## Fase 3: HMM

Objetivo:

```text
identificar regímenes estables e interpretables.
```

Tareas:

```text
crear features HMM
normalizar causalmente
entrenar HMM K=2..6
generar probabilidades filtradas
diagnóstico por estado
test de estabilidad por seed
```

Entregables:

```text
models/hmm/
reports/regime_diagnostics.md
```

---

## Fase 4: modelo predictivo

Objetivo:

```text
estimar probabilidades direccionales calibradas.
```

Tareas:

```text
crear target ternario
entrenar Logistic Regression
calibrar probabilidades
añadir HMM features
comparar con modelo sin HMM
entrenar XGBoost challenger
```

Entregables:

```text
models/classifiers/
reports/model_diagnostics.md
```

---

## Fase 5: backtest

Objetivo:

```text
evaluar la estrategia bajo ejecución realista.
```

Tareas:

```text
implementar ejecución next-open
añadir costes
añadir slippage
añadir stops
añadir kill switch
añadir cierre intradía
walk-forward completo
```

Entregables:

```text
backtest/trades/
backtest/equity_curves/
reports/walkforward_summary.md
```

---

## Fase 6: robustez

Objetivo:

```text
ver si el resultado sobrevive a cambios razonables.
```

Tests:

```text
horizonte h
K del HMM
costes
thresholds
ventana de entrenamiento
seeds
periodos
bloques de features
```

Entregables:

```text
reports/robustness.md
```

---

## Fase 7: ablation

Objetivo:

```text
medir qué componentes aportan valor.
```

Comparar:

```text
sin HMM
hard state
HMM probabilities
HMM filters
XGBoost sin HMM
XGBoost con HMM
```

Entregable:

```text
reports/ablation.md
```

---

## Fase 8: paper trading

Solo pasar a paper trading si:

```text
OOS neto positivo
cost stress razonable
HMM aporta valor
resultados estables por fold
drawdown aceptable
```

Paper trading mínimo:

```text
4 a 8 semanas
```

Registrar:

```text
señales
probabilidades
régimen
órdenes
fills
slippage
spread
latencia
PnL simulado vs PnL real
```

---

## Criterio final

No avanzar a paper trading si:

```text
el backtest no sobrevive a costes
no mejora al baseline
depende de parámetros frágiles
tiene leakage potencial no resuelto
```

# 19. Riesgos

## Riesgo 1: Overfitting

Síntomas:

```text
Sharpe alto en train
Sharpe bajo en test
muchos parámetros
thresholds extremos
PnL concentrado en pocos días
XGBoost mejora solo in-sample
```

Mitigación:

```text
modelo lineal base
regularización
walk-forward
ablation
cost stress
benchmarks simples
```

---

## Riesgo 2: Leakage

Fuentes habituales:

```text
normalizar con todo el dataset
usar close_t como ejecución
usar volumen total del día
usar VIX close del mismo día
usar HMM smoothing en test
permitir target overnight
usar high/low futuros
```

Mitigación:

```text
scalers fit solo en train
entrada en open_{t+1}
features cerradas en t
HMM filtering online
purge y embargo
drop de barras cercanas al cierre
```

---

## Riesgo 3: Costes eliminan el edge

Síntomas:

```text
PnL positivo a 0 bps
PnL negativo a 1–2 bps
avg trade net muy bajo
turnover excesivo
```

Mitigación:

```text
zona neutral
thresholds más exigentes
límite de trades
cost stress
modelo de slippage
```

---

## Riesgo 4: HMM inestable

Síntomas:

```text
estados cambian entre seeds
un estado desaparece
estado dominante > 80%
probabilidades uniformes
interpretación económica débil
```

Mitigación:

```text
K bajo
covariance diag
multi-seed
diagnóstico por fold
usar probabilidades, no solo hard state
```

---

## Riesgo 5: Modelo mal calibrado

Síntomas:

```text
probabilidades altas no implican mejor PnL
overconfidence
mala reliability curve
thresholds poco estables
```

Mitigación:

```text
calibración temporal
Brier score
calibration plots
probability buckets
```

---

## Riesgo 6: Dependencia de un régimen

Síntomas:

```text
todo el PnL viene de un estado
mal comportamiento fuera de ese estado
pocos trades reales
```

Mitigación:

```text
PnL por régimen
filtros por régimen
ablation
evaluación por periodo
```

---

## Riesgo 7: Backtest demasiado optimista

Fuentes:

```text
ejecución en close_t
stops optimistas
no slippage
no spread
no comisiones
fills perfectos
no restricciones intradía
```

Mitigación:

```text
event-driven backtest
next-open execution
costes conservadores
peor caso si stop y take-profit coinciden
kill switch
```

---

## Riesgo 8: Edge no explotable

Puede ocurrir que exista señal estadística, pero no sea monetizable.

Ejemplo:

```text
el modelo predice dirección ligeramente mejor que random,
pero el movimiento esperado es menor que los costes.
```

Regla:

```text
Si no hay PnL neto, no hay estrategia.
```


# 20. Condición crítica

## Regla principal

Si la estrategia no sobrevive a costes y validación temporal, debe rechazarse.

No se debe intentar salvar una estrategia frágil añadiendo complejidad.

---

## Condiciones mínimas de aceptación

La estrategia solo se acepta si cumple:

```text
Sharpe neto OOS > 1.0
profit factor neto > 1.10
avg trade net > 0
drawdown controlado
resultado positivo en varios folds
mejora frente a modelo sin HMM
sobrevive a costes base
no colapsa con costes x2
```

---

## Condiciones de rechazo automático

Rechazar si:

```text
solo funciona sin costes
solo funciona en train
solo funciona en un fold
solo funciona con un seed
solo funciona con un K concreto del HMM
solo funciona con thresholds extremos
no supera benchmarks simples
el HMM no aporta valor
el PnL está concentrado en pocos días
```

---

## Jerarquía de evidencia

Orden de importancia:

```text
1. PnL neto walk-forward
2. Robustez a costes
3. Ablation contra modelo sin HMM
4. Estabilidad por fold
5. Estabilidad por régimen
6. Métricas predictivas
```

Accuracy, AUC o log loss no son suficientes.

---

## Criterio sobre el HMM

El HMM se mantiene solo si:

```text
mejora el modelo sin HMM
la mejora es OOS
la mejora sobrevive a costes
la mejora es estable por fold
la mejora es interpretable
```

Si no:

```text
eliminar HMM
mantener modelo base
```

---

## Criterio sobre XGBoost

XGBoost se acepta solo si:

```text
mejora Logistic Regression OOS
no aumenta inestabilidad
no empeora calibración
no necesita hiperparámetros frágiles
no genera exceso de turnover
```

Si no:

```text
mantener Logistic Regression
```

---

## Principio final

La complejidad debe estar justificada por resultados robustos.

Si una versión simple funciona igual que una versión compleja, usar la versión simple.


# 21. Pseudocódigo

## Pipeline principal

```python
def main(config):
    raw = load_raw_data(config.data)
    clean = clean_data(raw, config.calendar)

    features = make_features(clean)
    labels = make_labels(features, config.target)

    results = []

    for fold in walkforward_splits(features, config.walkforward):
        fold_result = run_fold(features, labels, fold, config)
        results.append(fold_result)

    report = evaluate_results(results, config)
    save_report(report)
```

---

## Construcción de features

```python
def make_features(df):
    df = df.copy()

    df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_2"] = np.log(df["close"] / df["close"].shift(2))
    df["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    df["ret_6"] = np.log(df["close"] / df["close"].shift(6))
    df["ret_12"] = np.log(df["close"] / df["close"].shift(12))

    df["rv_3"] = rolling_rv(df["ret_1"], 3)
    df["rv_6"] = rolling_rv(df["ret_1"], 6)
    df["rv_12"] = rolling_rv(df["ret_1"], 12)

    df["range"] = np.log(df["high"] / df["low"])
    df["atr_6"] = rolling_atr(df, 6)

    df["sma_6"] = df["close"].rolling(6).mean()
    df["sma_12"] = df["close"].rolling(12).mean()

    df["trend_6"] = df["close"] / df["sma_6"] - 1
    df["trend_12"] = df["close"] / df["sma_12"] - 1

    df["intraday_high"] = df.groupby("session")["close"].cummax()
    df["intraday_drawdown"] = df["close"] / df["intraday_high"] - 1

    df["vwap"] = compute_intraday_vwap(df)
    df["dist_vwap"] = df["close"] / df["vwap"] - 1

    df["bar_index"] = df.groupby("session").cumcount()
    df["sin_time"] = np.sin(2 * np.pi * df["bar_index"] / 78)
    df["cos_time"] = np.cos(2 * np.pi * df["bar_index"] / 78)

    df["minutes_to_close"] = compute_minutes_to_close(df.index)
    df["rel_volume"] = compute_relative_volume(df)

    return df
```

---

## Labels

```python
def make_labels(df, config):
    h = config.horizon_bars

    df["entry_px"] = df.groupby("session")["open"].shift(-1)
    df["exit_px"] = df.groupby("session")["open"].shift(-(h + 1))

    df["fwd_ret"] = np.log(df["exit_px"] / df["entry_px"])

    df["sigma_h"] = df["rv_12"] * np.sqrt(h)

    neutral_band = np.maximum(
        bps_to_ret(config.cost_bps + config.edge_buffer_bps),
        config.neutral_vol_mult * df["sigma_h"]
    )

    df["y"] = 0
    df.loc[df["fwd_ret"] > neutral_band, "y"] = 1
    df.loc[df["fwd_ret"] < -neutral_band, "y"] = -1

    df = df.dropna(subset=["entry_px", "exit_px", "y"])

    return df
```

---

## Fold walk-forward

```python
def run_fold(features, labels, fold, config):
    train = features.loc[fold.train]
    val = features.loc[fold.val]
    test = features.loc[fold.test]

    y_train = labels.loc[fold.train, "y"]
    y_val = labels.loc[fold.val, "y"]
    y_test = labels.loc[fold.test, "y"]

    hmm, hmm_scaler = fit_hmm_pipeline(train, config.hmm)

    hmm_train = get_hmm_features(train, hmm, hmm_scaler)
    hmm_val = get_hmm_features(val, hmm, hmm_scaler)
    hmm_test = get_hmm_features(test, hmm, hmm_scaler)

    X_train = build_model_matrix(train, hmm_train)
    X_val = build_model_matrix(val, hmm_val)
    X_test = build_model_matrix(test, hmm_test)

    model, model_scaler = fit_predictive_model(X_train, y_train, config.model)

    probs_val = predict_proba(model, model_scaler, X_val)

    thresholds = select_thresholds(
        probs_val,
        y_val,
        val,
        config.signal,
        config.costs
    )

    probs_test = predict_proba(model, model_scaler, X_test)

    trades = run_backtest(
        data=test,
        probs=probs_test,
        hmm_features=hmm_test,
        thresholds=thresholds,
        config=config
    )

    return evaluate_fold(trades)
```

---

## HMM pipeline

```python
def fit_hmm_pipeline(train, config):
    X = train[HMM_COLS].dropna()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    lengths = compute_session_lengths(train.loc[X.index])

    hmm = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.max_iter,
        random_state=config.seed
    )

    hmm.fit(X_scaled, lengths=lengths)

    return hmm, scaler
```

---

## Señal

```python
def generate_signal(row, thresholds, regime_allow):
    p_up = row["p_up"]
    p_down = row["p_down"]
    p_neutral = row["p_neutral"]

    score = p_up - p_down
    regime = row["hmm_state"]

    if row["hmm_entropy"] > thresholds["max_hmm_entropy"]:
        return 0

    if p_up > thresholds["theta_prob"]:
        if score > thresholds["theta_score"] and p_neutral < thresholds["max_neutral"]:
            if regime_allow[regime]["long"]:
                return 1

    if p_down > thresholds["theta_prob"]:
        if score < -thresholds["theta_score"] and p_neutral < thresholds["max_neutral"]:
            if regime_allow[regime]["short"]:
                return -1

    return 0
```

---

## Backtest loop

```python
def run_backtest(data, probs, hmm_features, thresholds, config):
    portfolio = init_portfolio(config)
    trades = []

    for t in range(len(data) - config.target.horizon_bars - 1):
        row = data.iloc[t]
        next_row = data.iloc[t + 1]

        portfolio.update_intrabar(row)

        if portfolio.stop_hit(row):
            trade = portfolio.close_by_stop(row, config.costs)
            trades.append(trade)
            continue

        if portfolio.must_close_by_time(row):
            trade = portfolio.close(next_row["open"], config.costs)
            trades.append(trade)
            continue

        if kill_switch_active(portfolio, row, config):
            portfolio.force_flat(next_row["open"])
            continue

        signal_row = build_signal_row(row, probs.iloc[t], hmm_features.iloc[t])
        signal = generate_signal(signal_row, thresholds, config.regime_allow)

        if not can_trade(row, portfolio, config):
            signal = 0

        if signal != 0 and portfolio.is_flat():
            entry_price = apply_slippage(next_row["open"], signal, config.costs)
            size = compute_size(row, portfolio, config.risk)
            portfolio.open(signal, size, entry_price, row.name)

    return pd.DataFrame(trades)
```


# 22. Resumen operativo

## Arquitectura final

```text
SPY 5min OHLCV
    ↓
limpieza de datos
    ↓
features causales
    ↓
HMM Gaussian K=4
    ↓
probabilidades filtradas online
    ↓
Logistic Regression multinomial
    ↓
señal long / short / flat
    ↓
risk management
    ↓
backtest walk-forward
    ↓
evaluación neta
```

---

## Decisiones concretas

```text
Activo: SPY
Frecuencia: 5 minutos
Sesión: regular session
Overnight: no
Horizonte principal: h = 2
Target: ternario {-1, 0, +1}
Modelo de régimen: Gaussian HMM
Estados iniciales: K = 4
Modelo predictivo base: Logistic Regression
Modelo challenger: XGBoost
Ejecución: next-open
Costes: obligatorios
Validación: walk-forward
```

---

## Regla de entrada

```text
Se genera señal al cierre de t.
Se entra en open_{t+1}.
```

---

## Regla de salida

```text
Salida temporal en open_{t+h+1}
Stop loss si se activa
Cierre obligatorio antes del final de sesión
No overnight
```

---

## Señal

```text
Long si:
p_up > threshold
p_up - p_down > score_threshold
p_neutral no domina
HMM entropy no es excesiva
régimen permite long

Short si:
p_down > threshold
p_up - p_down < -score_threshold
p_neutral no domina
HMM entropy no es excesiva
régimen permite short

Flat en caso contrario
```

---

## Backtest

Debe incluir:

```text
costes
slippage
spread estimado
stops
cooldown
límite diario de trades
kill switch
walk-forward
purge y embargo
```

---

## Evaluación principal

Métricas clave:

```text
Sharpe neto
profit factor neto
avg trade net
max drawdown
número de trades
turnover
PnL por régimen
PnL por hora
cost stress
ablation
```

---

## Condición crítica

La estrategia se rechaza si:

```text
no sobrevive a costes
no mejora al modelo sin HMM
depende de parámetros frágiles
solo funciona en un periodo
el PnL está concentrado
el HMM es inestable
el paper trading no replica el backtest
```

---

## Orden correcto de desarrollo

```text
1. Dataset limpio
2. Baselines simples
3. Modelo sin HMM
4. HMM estable
5. Modelo con HMM
6. Backtest realista
7. Robustez
8. Ablation
9. Paper trading
```

---

## Principio final

El HMM no es la estrategia.

La estrategia es la combinación de:

```text
features causales
régimen estable
probabilidades calibradas
señal selectiva
costes realistas
risk management
validación walk-forward
```

Si esa combinación no genera PnL neto robusto, la estrategia debe descartarse.
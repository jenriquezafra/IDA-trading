# H8 - Bayesian Regime HMM Mejorado

Fecha: 2026-05-16

## Estado

`spec_registered`.

Esta hipotesis nace del indicador TradingView `Hidden Markov Model: Regime
Probability [AlgoPoint]`, pero no se acepta como estrategia por si solo. La
primera fase valida si el filtro de regimen tiene informacion economica
estable; solo despues se anade una senal operable.

## Hipotesis Economica

Un filtro bayesiano de regimen, alimentado por momentum normalizado,
volatilidad normalizada y eficiencia direccional, puede separar cuatro
contextos intradia utiles para QQQ/SPY:

- `bull_trend`: avance direccional con volatilidad no explosiva.
- `bear_stress`: caida direccional con expansion de volatilidad.
- `chop_compression`: falta de direccion con volatilidad comprimida.
- `volatile_noise`: volatilidad alta sin direccion eficiente.

El edge no debe venir de comprar o vender el estado directamente. Debe venir de
usar el estado para decidir que setups tienen permiso para operar.

## Base: HMM De TradingView

La version original usa:

- `mom_z`: ROC de 1 barra suavizado por EMA y normalizado por rolling z-score.
- `vol_z`: ATR normalizado por rolling z-score.
- tres estados: bull, bear y chop.
- matriz de transicion fija con persistencia alta en bull/bear y menor en chop.
- actualizacion bayesiana causal:

```text
posterior_t(state) = likelihood(obs_t | state) * prior_t(state)
prior_t = posterior_{t-1} * transition_matrix
```

## Mejora H8a - Filtro Bayesiano Robusto

Primero se mejora el filtro antes de anadir ninguna senal:

- calcular en log-space para evitar underflow numerico;
- resetear el prior por sesion intradia;
- tratar warm-up/NaN como periodo no observable y reiniciar en la siguiente
  barra valida;
- separar `chop_compression` de `volatile_noise`;
- anadir `eff_z`, z-score de eficiencia direccional, para distinguir tendencia
  real de desplazamiento ruidoso;
- exponer `max_prob` y `entropy` como confianza del filtro;
- no mirar retornos futuros para nombrar estados.

Artefacto inicial:

- `src/bayesian_regime_hmm.py`
- `src/bayesian_regime_h8.py`
- `configs/hmm_bayesian_regime_h8_spy_15min.yaml`

## Mejora H8b - Calibracion Walk-Forward

Despues de validar el H8a manual, calibrar sin tocar test:

- estimar medias/sigmas de emisiones solo con train;
- comparar 3 estados vs 4 estados;
- seleccionar parametros por estabilidad de estados en validation, no por PnL;
- mantener test como confirmacion unica;
- medir sensibilidad de `length`, persistencia y `min_prob`.

No se permite optimizar los parametros contra PnL de test.

Runner:

```bash
.venv/bin/python -m src.bayesian_regime_h8 --config configs/hmm_bayesian_regime_h8_spy_15min.yaml
```

Artefactos esperados:

- `h8_posteriors.parquet`: probabilidades por barra, fold y variante.
- `h8_regime_profiles.parquet`: ocupacion, duracion, confianza y perfil de
  features por regimen.
- `h8_directional_gate_diagnostics.parquet`: primera prueba de direccion:
  long si `P(bull_trend)` supera threshold, short si `P(bear_stress)` supera
  threshold.
- `h8_directional_gate_aggregate.parquet`: lectura agregada por folds para no
  seleccionar una fila aislada.
- `h8_model_registry.parquet`: modelos H8b entrenados por fold.

## Add-On H8c - Senal Operable

Solo si H8a/H8b separan estados interpretables, anadir una senal sencilla:

### Variante Principal: Regime-Gated Continuation

Long:

- `state == bull_trend`;
- `max_prob >= threshold`;
- `entropy <= threshold`;
- precio sobre VWAP;
- breakout o cierre cerca de maximo reciente;
- confirmacion cross-asset: breadth/credito no contradicen.

Short:

- `state == bear_stress`;
- `max_prob >= threshold`;
- `entropy <= threshold`;
- precio bajo VWAP;
- breakdown o cierre cerca de minimo reciente;
- confirmacion cross-asset: credito/risk-off no contradicen.

No-trade:

- `chop_compression`;
- `volatile_noise`, salvo que una rama posterior demuestre mean reversion neta.

## Baselines Obligatorios

- sin HMM, misma senal de continuation;
- HMM original de tres estados;
- HMM H8a de cuatro estados;
- filtro simple por momentum/ATR equivalente;
- random same-hour/same-frequency;
- inversion de direccion;
- breakout simple sin filtro de regimen.

## Metricas

- estado: ocupacion, duracion media, transiciones, entropy, concentracion por
  hora, estabilidad por fold;
- economia de estados: forward returns por estado despues de nombrarlos;
- estrategia: trades, avg trade neto, hit rate, profit factor, daily Sharpe,
  max drawdown, concentracion por dia y sensibilidad a costes;
- controles: mejora contra random, contra breakout simple y contra filtro
  momentum/ATR.

## Promotion Gates Iniciales

- estados interpretables en al menos dos folds;
- ningun estado aceptado explicado solo por hora del dia;
- `bull_trend` y `bear_stress` con duracion media razonable, no flip-barra;
- validation y test positivos con coste conservador;
- avg trade neto mayor que coste stress;
- mejora neta contra continuation sin HMM;
- no depender de un unico mes o top 5 dias;
- survives `min_prob`, entropy y costes sin colapsar.

## Decision Rules

- Si H8a no separa estados interpretables: cerrar o volver a emisiones.
- Si H8a separa estados pero no mejora setups: mantenerlo como diagnostico, no
  estrategia.
- Si H8c supera controles: crear runner formal, manifest y freeze review.

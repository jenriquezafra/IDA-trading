# TODO — Estrategia intradía SPY 5min con HMM

## 0. Setup inicial

- [x] Crear repositorio del proyecto.
- [x] Crear estructura de carpetas:

```text
project/
├── data/
├── src/
├── models/
├── docs/
├── backtest/
├── configs/
├── reports/
└── notebooks/
```

- [x] Crear entorno virtual.
- [x] Instalar dependencias base:

```bash
pip install pandas numpy scipy scikit-learn hmmlearn xgboost pyarrow matplotlib pyyaml joblib pytest
```

- [x] Crear `requirements.txt`.
- [x] Crear `README.md`.
- [x] Crear configuración base `configs/base.yaml`.

---

## 1. Datos

- [x] Conseguir datos OHLCV de SPY a 5 minutos.
- [ ] Evaluar fuente definitiva para histórico intradía 5min multi-año:

```text
Polygon
Databento
Tiingo
IBKR
otro proveedor
```

- [ ] Sustituir dataset inicial de yfinance por histórico suficiente para walk-forward 5/1/1.
- [x] Verificar que los datos tienen columnas:

```text
timestamp
open
high
low
close
volume
```

- [x] Convertir timestamps a timezone correcto.
- [x] Filtrar solo regular session.
- [x] Eliminar premarket.
- [x] Eliminar after-hours.
- [x] Crear columna `session`.
- [x] Crear columna `bar_index`.
- [x] Verificar número esperado de barras por sesión.
- [x] Detectar días incompletos.
- [x] Eliminar o marcar días incompletos.
- [x] Detectar timestamps duplicados.
- [x] Eliminar duplicados.
- [x] Detectar NaN críticos.
- [x] Detectar precios imposibles o outliers extremos.
- [x] Guardar dataset limpio en:

```text
data/cleaned/spy_5min_clean.parquet
```

- [x] Crear reporte de calidad:

```text
reports/data_quality.md
```

---

## 2. Calendario

- [x] Integrar calendario de mercado.
- [x] Detectar festivos.
- [x] Detectar medias sesiones.
- [x] Decidir si eliminar medias sesiones.
- [x] Verificar que no hay targets cruzando overnight.
- [x] Verificar que no hay trades que puedan quedar abiertos al cierre.

---

## 3. Features base

- [x] Implementar `src/feature_engineering.py`.
- [x] Calcular retornos:

```text
ret_1
ret_2
ret_3
ret_6
ret_12
```

- [x] Calcular volatilidad realizada:

```text
rv_3
rv_6
rv_12
rv_24
```

- [x] Calcular rango:

```text
range = log(high / low)
```

- [x] Calcular ATR:

```text
atr_6
atr_12
```

- [x] Calcular medias móviles:

```text
sma_6
sma_12
sma_24
```

- [x] Calcular tendencias:

```text
trend_6
trend_12
trend_24
```

- [x] Calcular VWAP intradía.
- [x] Calcular distancia a VWAP:

```text
dist_vwap
```

- [x] Calcular drawdown intradía:

```text
intraday_drawdown
```

- [x] Calcular volumen relativo:

```text
rel_volume
```

- [x] Calcular features temporales:

```text
sin_time
cos_time
minutes_to_close
open_window
close_window
midday
```

- [x] Verificar que todas las features están disponibles en $begin:math:text$t$end:math:text$.
- [x] Guardar features en:

```text
data/features/features_base.parquet
```

---

## 4. Target

- [x] Implementar `src/labels.py`.
- [x] Definir horizonte principal:

```text
h = 2
```

- [x] Crear precio de entrada:

```text
entry_px = open_{t+1}
```

- [x] Crear precio de salida:

```text
exit_px = open_{t+h+1}
```

- [x] Calcular retorno futuro:

```text
fwd_ret = log(exit_px / entry_px)
```

- [x] Calcular estimación ex ante de volatilidad:

```text
sigma_h = rv_12 * sqrt(h)
```

- [x] Definir zona neutral.
- [x] Crear target ternario:

```text
-1
 0
+1
```

- [x] Eliminar filas cuyo target cruce el cierre de sesión.
- [x] Verificar que no hay leakage en el target.
- [x] Guardar labels en:

```text
data/features/labels.parquet
```

---

## 5. Baselines simples

- [x] Implementar benchmark always flat.
- [x] Implementar benchmark random.
- [x] Implementar benchmark intraday buy & hold.
- [x] Implementar momentum simple.
- [x] Implementar reversión simple.
- [x] Aplicar mismos costes a todos los benchmarks.
- [x] Aplicar misma lógica next-open.
- [x] Generar reporte inicial:

```text
reports/baseline_report.md
```

---

## 6. HMM

- [x] Implementar `src/hmm_model.py`.
- [x] Implementar `src/hmm_filter.py`.
- [x] Definir columnas HMM:

```text
ret_1
ret_3
rv_6
rv_12
range
rel_volume
trend_12
intraday_drawdown
```

- [x] Implementar normalización fit solo en train.
- [x] Implementar entrenamiento por sesiones usando `lengths`.
- [x] Entrenar HMM con:

```text
K = 4
covariance_type = diag
```

- [x] Probar también:

```text
K = 2, 3, 5, 6
```

- [x] Implementar filtro online.
- [x] Evitar smoothing forward-backward en test.
- [x] Generar probabilidades filtradas por barra.
- [x] Calcular:

```text
hmm_p0
hmm_p1
hmm_p2
hmm_p3
hmm_state
hmm_entropy
hmm_max_prob
```

- [x] Diagnosticar ocupación por estado.
- [x] Diagnosticar retorno medio por estado.
- [x] Diagnosticar volatilidad media por estado.
- [x] Diagnosticar duración media por estado.
- [x] Diagnosticar matriz de transición.
- [x] Probar estabilidad con varios seeds.
- [x] Crear reporte:

```text
reports/regime_diagnostics.md
```

---

## 7. Modelo predictivo base

- [x] Implementar `src/predictive_model.py`.
- [x] Crear matriz de features sin HMM.
- [x] Entrenar Logistic Regression sin HMM.
- [x] Usar regularización elastic-net.
- [x] Usar scaler fit solo en train.
- [x] Evaluar probabilidades.
- [x] Calibrar probabilidades en validation.
- [x] Guardar modelo por fold.
- [x] Guardar scaler por fold.

---

## 8. Modelo predictivo con HMM

- [x] Añadir probabilidades HMM a las features.
- [x] Añadir `hmm_entropy`.
- [x] Añadir `hmm_max_prob`.
- [x] Añadir estado HMM one-hot.
- [x] Entrenar Logistic Regression con HMM.
- [x] Comparar contra modelo sin HMM.
- [x] Verificar mejora OOS.
- [x] Verificar mejora después de costes.

---

## 9. Señal

- [x] Implementar `src/signal.py`.
- [x] Definir probabilidades:

```text
p_up
p_neutral
p_down
```

- [x] Definir score:

```text
score = p_up - p_down
```

- [x] Implementar thresholds iniciales:

```text
theta_prob = 0.55
theta_score = 0.10
max_neutral = 0.55
max_hmm_entropy = 0.90
```

- [x] Implementar señal long.
- [x] Implementar señal short.
- [x] Implementar señal flat.
- [x] Implementar filtro por entropía HMM.
- [x] Implementar filtros por régimen.
- [x] Seleccionar thresholds solo en validation.
- [x] No optimizar thresholds en test.

---

## 10. Costes

- [x] Implementar `src/costs.py`.
- [x] Definir coste round-trip base:

```text
1 bps
```

- [x] Definir coste conservador:

```text
2 bps
```

- [x] Definir coste stress:

```text
5 bps
```

- [x] Implementar comisiones.
- [x] Implementar spread estimado.
- [x] Implementar slippage.
- [x] Implementar impacto por participación.
- [x] Calcular PnL bruto.
- [x] Calcular PnL neto.
- [x] Verificar que todos los reportes usan PnL neto.

---

## 11. Risk management

- [x] Implementar `src/risk.py`.
- [x] Definir posición base:

```text
-1, 0, +1
```

- [x] Implementar sizing fijo.
- [x] Implementar volatility scaling.
- [x] Implementar máximo leverage.
- [x] Implementar stop loss.
- [x] Implementar time stop.
- [x] Implementar max daily loss.
- [x] Implementar máximo de trades por día.
- [x] Implementar cooldown tras trade.
- [x] Implementar no abrir trades después de 15:45 ET.
- [x] Implementar cierre forzoso antes de 15:55 ET.
- [x] Implementar kill switch.
- [x] Verificar que no hay overnight.

---

## 12. Backtest

- [x] Implementar `src/backtest.py`.
- [x] Implementar backtest event-driven.
- [x] Entrada siempre en `open_{t+1}`.
- [x] Prohibir entrada en `close_t`.
- [x] Implementar salida temporal.
- [x] Implementar salida por stop.
- [x] Implementar salida por cierre intradía.
- [x] Implementar salida por kill switch.
- [x] Aplicar costes en entrada.
- [x] Aplicar costes en salida.
- [x] Guardar trades.
- [x] Guardar equity curve.
- [x] Guardar PnL diario.
- [x] Validar manualmente algunos trades.

---

## 13. Walk-forward

- [x] Implementar `src/walkforward.py`.
- [x] Definir esquema:

```text
fit: 5 meses
validation: 1 mes
test: 1 mes
step: 1 mes
```

- [x] Implementar purge.
- [x] Implementar embargo.
- [x] Entrenar scaler HMM solo en train.
- [x] Entrenar HMM solo en train.
- [x] Generar HMM probs en train, validation y test.
- [x] Entrenar modelo solo en train.
- [x] Calibrar en validation.
- [x] Elegir thresholds en validation.
- [x] Evaluar una sola vez en test.
- [x] Guardar resultados por fold.

Nota: la implementación está lista y probada con datos sintéticos multi-mes. Con el dataset actual de yfinance solo hay 2026-02, 2026-03 y 2026-04, por lo que no se generan folds reales para el esquema 5/1/1.

Pendiente antes de usar walk-forward como evidencia: elegir proveedor de datos intradía histórico y cargar al menos 7 meses, preferiblemente 18-24 meses o más.

---

## 14. Evaluación

- [x] Implementar `src/evaluation.py`.
- [x] Calcular Sharpe neto diario.
- [x] Calcular max drawdown.
- [x] Calcular profit factor.
- [x] Calcular hit ratio.
- [x] Calcular avg trade net.
- [x] Calcular median trade net.
- [x] Calcular turnover.
- [x] Calcular exposure.
- [x] Calcular PnL long vs short.
- [x] Calcular PnL por régimen.
- [x] Calcular PnL por hora.
- [x] Calcular PnL por fold.
- [x] Calcular métricas de calibración.
- [x] Crear reporte:

```text
reports/walkforward_summary.md
```

---

## 15. Robustez

- [ ] Implementar `src/robustness.py`.
- [ ] Testear horizontes:

```text
h = 1, 2, 3
```

- [ ] Testear estados HMM:

```text
K = 2, 3, 4, 5, 6
```

- [ ] Testear costes:

```text
1, 2, 5 bps
```

- [ ] Testear thresholds.
- [ ] Testear ventanas de entrenamiento.
- [ ] Testear varios seeds.
- [ ] Testear distintos periodos.
- [ ] Testear por régimen de volatilidad.
- [ ] Testear por franja horaria.
- [ ] Crear reporte:

```text
reports/robustness.md
```

---

## 16. Ablation

- [ ] Implementar `src/ablation.py`.
- [ ] Ejecutar modelo sin HMM.
- [ ] Ejecutar modelo con hard HMM state.
- [ ] Ejecutar modelo con HMM probabilities.
- [ ] Ejecutar modelo con filtros HMM.
- [ ] Ejecutar modelos separados por régimen.
- [ ] Ejecutar XGBoost sin HMM.
- [ ] Ejecutar XGBoost con HMM.
- [ ] Comparar todos los resultados.
- [ ] Verificar si HMM mejora realmente.
- [ ] Crear reporte:

```text
reports/ablation.md
```

---

## 17. XGBoost challenger

- [ ] Implementar XGBoost baseline.
- [ ] Limitar profundidad.
- [ ] Regularizar.
- [ ] Evitar grid search excesivo.
- [ ] Calibrar probabilidades.
- [ ] Comparar contra Logistic Regression.
- [ ] Evaluar turnover.
- [ ] Evaluar estabilidad por fold.
- [ ] Rechazar si solo mejora in-sample.

---

## 18. Tests anti-leakage

- [ ] Testear que ningún scaler usa test.
- [ ] Testear que el HMM se entrena solo en train.
- [ ] Testear que el modelo se entrena solo en train.
- [ ] Testear que thresholds se eligen solo en validation.
- [ ] Testear que ninguna feature usa futuro.
- [ ] Testear que el target no cruza overnight.
- [ ] Testear que la entrada es next-open.
- [ ] Testear que no se usa close_t como ejecución.
- [ ] Testear que volumen relativo no usa volumen futuro.
- [ ] Testear que VIX no usa close del mismo día si no está disponible.

---

## 19. Reportes finales

- [ ] Crear reporte de datos.
- [ ] Crear reporte de HMM.
- [ ] Crear reporte de modelo predictivo.
- [ ] Crear reporte de backtest.
- [ ] Crear reporte de robustez.
- [ ] Crear reporte de ablation.
- [ ] Crear resumen ejecutivo.
- [ ] Documentar condiciones de rechazo.
- [ ] Documentar decisión final:

```text
aceptar / rechazar / seguir investigando
```

---

## 20. Criterios de aceptación

- [ ] Sharpe neto OOS mayor que 1.
- [ ] Profit factor neto mayor que 1.10.
- [ ] Avg trade net positivo.
- [ ] Drawdown razonable.
- [ ] PnL positivo en varios folds.
- [ ] PnL no concentrado en pocos días.
- [ ] Sobrevive a costes base.
- [ ] No colapsa totalmente con costes x2.
- [ ] Supera benchmark random.
- [ ] Supera momentum simple.
- [ ] Supera reversión simple.
- [ ] Supera modelo sin HMM.
- [ ] HMM estable entre folds.
- [ ] HMM estable entre seeds.
- [ ] Paper trading replica razonablemente el backtest.

---

## 21. Criterios de rechazo

- [ ] Rechazar si solo funciona sin costes.
- [ ] Rechazar si solo funciona en train.
- [ ] Rechazar si solo funciona en un fold.
- [ ] Rechazar si depende de un único seed.
- [ ] Rechazar si depende de un único K del HMM.
- [ ] Rechazar si no supera al modelo sin HMM.
- [ ] Rechazar si el PnL está concentrado.
- [ ] Rechazar si el turnover es excesivo.
- [ ] Rechazar si el HMM es inestable.
- [ ] Rechazar si el backtest usa supuestos optimistas.
- [ ] Rechazar si paper trading no replica el comportamiento esperado.

---

## 22. Paper trading

- [ ] Conectar fuente de datos en tiempo real o delayed controlado.
- [ ] Ejecutar mismo pipeline que en backtest.
- [ ] Registrar cada señal.
- [ ] Registrar probabilidades del modelo.
- [ ] Registrar régimen HMM.
- [ ] Registrar orden teórica.
- [ ] Registrar fill simulado.
- [ ] Registrar spread.
- [ ] Registrar slippage.
- [ ] Registrar latencia.
- [ ] Comparar señal paper vs señal backtest equivalente.
- [ ] Comparar PnL paper vs PnL esperado.
- [ ] Ejecutar durante 4–8 semanas.
- [ ] Decidir si pasar a capital real o rechazar.

---

## 23. Primera versión mínima viable

- [ ] Dataset limpio.
- [ ] Features base.
- [ ] Target ternario.
- [ ] Logistic Regression sin HMM.
- [ ] Backtest next-open.
- [ ] Costes base.
- [ ] Walk-forward simple.
- [ ] Métricas netas.
- [ ] Benchmark momentum.
- [ ] Benchmark random.

---

## 24. Segunda versión

- [ ] Añadir HMM.
- [ ] Añadir probabilidades HMM.
- [ ] Añadir filtros por régimen.
- [ ] Añadir ablation.
- [ ] Añadir robustez de K.
- [ ] Añadir robustez de costes.

---

## 25. Tercera versión

- [ ] Añadir XGBoost challenger.
- [ ] Añadir calibración avanzada.
- [ ] Añadir slippage más realista.
- [ ] Añadir análisis por régimen.
- [ ] Añadir paper trading.

---

## Decisión final

- [ ] Aceptar estrategia.
- [ ] Rechazar estrategia.
- [ ] Mantener como investigación.
- [ ] Rehacer target.
- [ ] Rehacer features.
- [ ] Eliminar HMM.
- [ ] Mantener Logistic Regression.
- [ ] Probar otro universo.
- [ ] Probar otra frecuencia.

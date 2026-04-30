# IDA Trading

Proyecto de investigacion para una estrategia intradia sobre SPY a frecuencia de 5 minutos.

El objetivo es evaluar si una combinacion de features causales, regimes HMM, modelo predictivo supervisado y backtest realista puede generar PnL neto positivo y robusto sin posiciones overnight.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura

```text
backtest/      utilidades y salidas especificas de backtest
configs/       configuraciones reproducibles
data/          datos locales
  raw/         datos originales
  cleaned/     datos limpios
  features/    features y labels
docs/          documentacion y TODOs
models/        artefactos entrenados
notebooks/     analisis exploratorio
reports/       reportes generados
src/           codigo del proyecto
tests/         pruebas
```

## Configuracion base

La configuracion inicial vive en `configs/base.yaml` e incluye parametros de sesion, labeling, HMM, modelo, costes, backtest y walk-forward.

## Limpieza de datos

Para descargar un dataset inicial reciente de SPY 5 minutos con Yahoo Finance:

```bash
source .venv/bin/activate
python -m src.data_download --config configs/base.yaml
```

Esto guarda `data/raw/spy_5min.parquet`. El historico intradia gratuito suele estar limitado a una ventana reciente, por lo que este dataset sirve para desarrollar el pipeline, no para una validacion robusta final.

El pipeline espera datos OHLCV de SPY a 5 minutos en `data/raw/spy_5min.parquet` o, ajustando `configs/base.yaml`, en un CSV equivalente. Columnas requeridas:

```text
timestamp
open
high
low
close
volume
```

Para limpiar y validar:

```bash
source .venv/bin/activate
python -m src.data_cleaning --config configs/base.yaml
```

La salida se guarda en `data/cleaned/spy_5min_clean.parquet` y el reporte en `reports/data_quality.md`.

El limpiador usa calendario NYSE cuando `calendar.enabled: true`: descarta festivos, detecta medias sesiones y, por defecto, elimina medias sesiones para mantener sesiones comparables de 78 barras. Tambien anade columnas de seguridad para evitar targets y entradas que puedan cruzar el cierre.

## Features base

Para generar las features causales:

```bash
source .venv/bin/activate
python -m src.feature_engineering --config configs/base.yaml
```

La salida se guarda en `data/features/features_base.parquet`. Los calculos rolling se reinician por sesion y `rel_volume` usa solo sesiones anteriores para el mismo `bar_index`.

## Labels

Para generar labels ternarios con entrada en `open_{t+1}` y salida en `open_{t+h+1}`:

```bash
source .venv/bin/activate
python -m src.labels --config configs/base.yaml
```

La salida se guarda en `data/features/labels.parquet`. Las filas cuyo target cruzaria el cierre de sesion se eliminan.

## Baselines

Para generar benchmarks simples:

```bash
source .venv/bin/activate
python -m src.baselines --config configs/base.yaml
```

El reporte se guarda en `reports/baseline_report.md` y los trades por benchmark en `reports/baseline_trades.parquet`.

## HMM

Para entrenar el HMM y generar probabilidades filtradas:

```bash
source .venv/bin/activate
python -m src.hmm_model --config configs/base.yaml
```

La salida se guarda en `data/features/hmm_features.parquet`, el modelo en `models/hmm_k4.joblib` y el diagnostico en `reports/regime_diagnostics.md`.

## Modelo Predictivo Base

Para entrenar la Logistic Regression base sin HMM:

```bash
source .venv/bin/activate
python -m src.predictive_model --config configs/base.yaml
```

Las probabilidades se guardan en `data/features/predictive_base_predictions.parquet`, el reporte en `reports/predictive_base_report.md` y los artefactos en `models/predictive_base/fold_0/`.

Para entrenar la variante con HMM:

```bash
source .venv/bin/activate
python -m src.predictive_model_hmm --config configs/base.yaml
```

Las probabilidades se guardan en `data/features/predictive_hmm_predictions.parquet`, el reporte en `reports/predictive_hmm_report.md` y los artefactos en `models/predictive_hmm/fold_0/`.

## Señales

Para convertir probabilidades en senales `long/short/flat`:

```bash
source .venv/bin/activate
python -m src.signal --config configs/base.yaml
```

Las senales se guardan en `data/features/signals.parquet` y el reporte en `reports/signal_report.md`.

## Backtest

Para ejecutar el backtest event-driven:

```bash
source .venv/bin/activate
python -m src.backtest --config configs/base.yaml
```

El backtest usa costes de `src.costs`, restricciones de `src.risk`, entra siempre en `open_{t+1}` y guarda `reports/backtest_trades.parquet`, `reports/equity_curve.parquet`, `reports/daily_pnl.parquet` y `reports/backtest_report.md`.

## Walk-Forward

Para ejecutar el pipeline walk-forward mensual:

```bash
source .venv/bin/activate
python -m src.walkforward --config configs/base.yaml
```

El esquema configurado es fit 5 meses, validation 1 mes, test 1 mes y step 1 mes. Con el dataset inicial de yfinance no hay meses suficientes para generar folds reales; el reporte se guarda en `reports/walkforward_folds_summary.md`.

## Evaluacion

Para generar el resumen neto de evaluacion:

```bash
source .venv/bin/activate
python -m src.evaluation --config configs/base.yaml
```

El reporte se guarda en `reports/walkforward_summary.md` y las metricas agregadas en `reports/evaluation_metrics.parquet`. Con el dataset inicial el reporte calcula metricas sobre los artefactos disponibles, pero marca que no hay folds walk-forward reales suficientes para evidencia OOS robusta.

## Documentacion

- `docs/PROJECT_ARCHITECTURE.md`: definicion del problema, arquitectura y decisiones de diseno.
- `docs/TODOs.md`: roadmap de implementacion por bloques.

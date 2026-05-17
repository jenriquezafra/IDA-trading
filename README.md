# IDA Trading

IDA Trading se esta reconstruyendo como un research stack modular para pasar de
alpha research a estrategia, backtest y paper/live con trazabilidad. El codigo
antiguo queda como base reutilizable, pero la arquitectura activa vive
directamente bajo `src/`.

## Arquitectura activa

```text
data -> features -> alpha -> strategy -> backtesting -> triage -> paper/live
```

La primera vertical activa es alpha research intradia:

```text
QQQ 15min
  -> cross_asset_liquid_15min features
  -> declarative alpha specs
  -> StrategySpec
  -> cost-aware backtest
  -> promotion gates
```

## Modulos nuevos

```text
src/
  alpha/         alpha specs, thresholds, confirmation gates
  data/          ingesta/tratamiento de datos externos point-in-time
  strategy/      StrategySpec, entry/exit/position/risk contracts
  backtesting/   metricas comunes para research
  research/      manifests, run ids, artifact contracts
```

Los modulos pendientes son `features`, `risk` y `execution`. Los scripts legacy
de `src/*.py` se migraran o eliminaran por fases.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Alpha research config

La config inicial vive en:

```text
configs/alpha/alpha_research_v1.yaml
```

Incluye:

- target/timeframe y split policy;
- feature path declarativo;
- coste primario, conservador y stress;
- alpha families;
- confirmation gates;
- promotion gates.

## Datos externos Cboe

Descargar y tratar volatilidad/put-call diarios point-in-time:

```bash
source .venv/bin/activate
python -m src.data.cboe_risk_context --config configs/data/cboe_risk_context.yaml
```

Esto genera:

```text
data/external/cboe/volatility_indices_daily.parquet
data/external/cboe/put_call_ratios_daily.parquet
data/external/cboe/risk_context_daily.parquet
reports/data_external/cboe_risk_context.md
```

El dataset `risk_context_daily.parquet` usa `available_session`: los datos de
cierre Cboe de una fecha solo quedan disponibles para la siguiente sesion NYSE.

## Contrato de estrategia

Una estrategia operable debe poder expresarse como YAML:

```yaml
strategy_id: qqq_15min_risk_off_short_h6_v1
target_symbol: QQQ
timeframe: 15min
feature_set_id: cross_asset_liquid_15min
alpha_id: risk_off_short_h6_q80
entry_rule: next_open
exit_rule:
  horizon_bars: 6
position:
  side: short_only
  max_gross_exposure: 1.0
risk:
  no_new_trades_after: "15:45"
  force_flat_before: "15:55"
  max_turnover: 4.0
cost_profile_id: bps_2
split_policy_id: wf_24m_6m_6m_step6m_embargo1
```

## Alpha research CLI

Validar la config y mostrar el universo de busqueda:

```bash
source .venv/bin/activate
python -m src alpha-research --config configs/alpha/alpha_research_v1.yaml --dry-run
```

Ejecutar el runner y escribir artefactos:

```bash
source .venv/bin/activate
python -m src alpha-research --config configs/alpha/alpha_research_v1.yaml
```

## Risk-off EDA

Generar el analisis exploratorio de la hipotesis risk-off short:

```bash
source .venv/bin/activate
python -m src.alpha.risk_off_eda
```

Esto escribe:

```text
reports/eda/risk_off_short/risk_off_short_eda.md
reports/eda/risk_off_short/bucket_summary.parquet
reports/eda/risk_off_short/condition_summary.parquet
reports/eda/risk_off_short/yearly_summary.parquet
reports/eda/risk_off_short/control_pnl.parquet
```

## Risk-off strategy runner

Ejecutar la primera estrategia rule-based con trades no solapados, costes,
walk-forward folds y controles:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/trades.parquet
results/strategy/risk_off_short/QQQ/15min/daily.parquet
results/strategy/risk_off_short/QQQ/15min/monthly.parquet
results/strategy/risk_off_short/QQQ/15min/summary.parquet
results/strategy/risk_off_short/QQQ/15min/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/report.md
```

La vertical nueva todavia no usa ML. La estrategia actual es intencionadamente
rule-based: primero se valida si la hipotesis economica produce edge neto y
estable; despues se decide si merece modelos.

La validacion sigue el esquema del libro en lo importante: datos point-in-time,
walk-forward train/validation/test, validation para seleccionar, test para
confirmar, costes y controles. El splitter activo aplica `embargo_sessions: 1`
para ser conservador en los bordes entre train y validation.

## Risk-off h=6 triage

Diagnosticar el horizonte superviviente con controles agregados, franja
intradia, dia de semana, concentracion por sesiones y sensibilidad de thresholds
seleccionada solo con validation:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_triage
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/triage/report.md
results/strategy/risk_off_short/QQQ/15min/triage/controls_rollup.parquet
results/strategy/risk_off_short/QQQ/15min/triage/candidate_by_fold.parquet
results/strategy/risk_off_short/QQQ/15min/triage/hour_summary.parquet
results/strategy/risk_off_short/QQQ/15min/triage/bucket_summary.parquet
results/strategy/risk_off_short/QQQ/15min/triage/weekday_summary.parquet
results/strategy/risk_off_short/QQQ/15min/triage/session_concentration.parquet
results/strategy/risk_off_short/QQQ/15min/triage/threshold_sensitivity.parquet
results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_confirmation.parquet
results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_trades.parquet
results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_controls.parquet
results/strategy/risk_off_short/QQQ/15min/triage/selected_threshold_concentration.parquet
results/strategy/risk_off_short/QQQ/15min/triage/promotion_gates.parquet
results/strategy/risk_off_short/QQQ/15min/triage/promotion_decision.yaml
```

Promotion gates son reglas duras para decidir si un candidato puede pasar de
research a `freeze_review`. No buscan mejorar el backtest; bloquean candidatos
fragiles aunque el PnL sea positivo. En H1 cubren muestra minima, folds
positivos, retorno neto, coste stress, mejora contra controles y concentracion
por sesiones. La implementacion comun vive en `src/research/promotion.py`; H1
solo aporta sus artefactos y su `candidate_label`.

## Risk-off promotion-aware sweep

Buscar variantes de H1 que reparen la concentracion sin seleccionar con test:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_promotion_sweep
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/report.md
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/validation_sweep.parquet
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/validation_gates.parquet
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/selected_variant.yaml
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/selected_controls.parquet
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/selected_concentration.parquet
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/selected_gates.parquet
results/strategy/risk_off_short/QQQ/15min/promotion_sweep/selected_decision.yaml
```

El sweep actual evalua `128` variantes de thresholds y franja horaria. Ninguna
pasa todos los gates de validation. La variante seleccionada por validation es
`riskq80__vixq50__all`; mejora muestra y PnL, pero la decision final sigue en
`continue_research` porque falla concentracion por sesiones:
`validation_top5_abs_share` y `test_top5_abs_share`.

## Risk-off H1b concentration sweep

Reformular H1 con filtros economicos adicionales para reparar concentracion:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_h1b_sweep
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/report.md
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/validation_sweep.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/validation_gates.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_variant.yaml
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_trades.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_controls.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_concentration.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_gates.parquet
results/strategy/risk_off_short/QQQ/15min/h1b_concentration_sweep/selected_decision.yaml
```

El sweep H1b evalua `840` variantes con filtros de credito, defensivos,
breadth, VIX extremo y term structure. `26` variantes pasan todos los gates de
validation. La variante seleccionada por validation es
`riskq55__vixq45__credit_weak_q50`, que exige credito debil
(`spread_credit_12` por debajo de su mediana de train). La confirmacion final
pasa todos los gates de validation y test, por lo que la decision queda en
`freeze_review`.

## H1b StrategySpec freeze

La variante H1b congelada vive en:

```text
configs/strategy/qqq_15min_risk_off_short_h1b_v1.yaml
```

Congelar spec, thresholds por fold y fingerprints de artefactos:

```bash
source .venv/bin/activate
python -m src.strategy.freeze_risk_off_short_h1b
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/strategy_spec.yaml
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/fold_thresholds.parquet
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/freeze_review_decision.yaml
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1b_v1/manifest.yaml
```

El `manifest.yaml` es el artefacto canonico de freeze review: incluye
fingerprints de features, risk context, sweep H1b, selected trades/controls,
promotion gates, spec congelado y thresholds entrenados por fold.

## H1b pre-paper robustness

Ejecutar robustez local desde el `StrategySpec` congelado:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_h1b_robustness
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/report.md
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/local_threshold_sweep.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/local_threshold_gates.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/cost_sensitivity.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/subperiod_summary.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/fold_stability.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1b_v1/robustness_decision.yaml
```

Resultado actual: `needs_more_research`. La variante ancla pasa gates, y `6/27`
variantes locales pasan todos los gates, pero todas dependen de `credit_q50`.
Mover el filtro de credito a `q45` o `q55` degrada el edge. La estrategia tambien
deja de ser positiva a `7.5` y `10` bps. No esta rechazada, pero no pasa a paper
hasta reparar o justificar la fragilidad del filtro de credito.

## H1c credit repair

Reparar el filtro de credito exacto con reglas economicas interpretables:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_h1c_credit_repair
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/report.md
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/validation_sweep.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/validation_gates.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_variant.yaml
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_trades.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_controls.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_concentration.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_gates.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_cost_sensitivity.parquet
results/strategy/risk_off_short/QQQ/15min/h1c_credit_repair/selected_decision.yaml
```

Resultado actual: `credit_repaired`. La variante seleccionada por validation es
`riskq50__vixq45__credit_spread_lte_0`: exige que `spread_credit_12 <= 0`, una
regla economica interpretable equivalente a HYG no liderando a LQD. Pasa todos
los promotion gates en validation y test, queda positiva a `7.5 bps`, y solo
falla como warning a `10 bps`.

## H1c StrategySpec freeze

La variante H1c congelada vive en:

```text
configs/strategy/qqq_15min_risk_off_short_h1c_v1.yaml
```

Congelar spec, thresholds por fold y fingerprints:

```bash
source .venv/bin/activate
python -m src.strategy.freeze_risk_off_short_h1c
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/strategy_spec.yaml
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/fold_thresholds.parquet
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/freeze_review_decision.yaml
results/strategy/risk_off_short/QQQ/15min/freeze_review/qqq_15min_risk_off_short_h1c_v1/manifest.yaml
```

## H1c pre-paper robustness

Ejecutar robustez local desde el `StrategySpec` H1c congelado:

```bash
source .venv/bin/activate
python -m src.strategy.risk_off_short_h1c_robustness
```

Esto escribe:

```text
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/report.md
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/manifest.yaml
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/local_threshold_sweep.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/local_threshold_gates.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/cost_sensitivity.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/subperiod_summary.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/fold_stability.parquet
results/strategy/risk_off_short/QQQ/15min/robustness/qqq_15min_risk_off_short_h1c_v1/robustness_decision.yaml
```

Resultado actual: `paper_candidate`. `6/9` variantes locales pasan todos los
gates, con soporte en `3` quantiles de risk y `2` quantiles de VIX. La estrategia
sigue positiva a `7.5 bps` en validation y test; `10 bps` queda como warning.

## IBKR Gateway read-only

Config paper read-only:

```text
configs/execution/ibkr_paper_readonly.yaml
```

Validar config sin conectar:

```bash
source .venv/bin/activate
python -m src.execution.ibkr_read_only --validate-only
```

Health-check contra IB Gateway paper:

```bash
source .venv/bin/activate
python -m src.execution.ibkr_read_only
```

Snapshot read-only de account summary, posiciones y open trades:

```bash
source .venv/bin/activate
python -m src.execution.ibkr_read_only --snapshot
```

El cliente solo permite `trading_mode: paper`, `read_only: true`,
`allow_orders: false` y puerto paper de IB Gateway/TWS (`4002` o `7497`). Esta
fase no envia ordenes ni liquida posiciones.

## Tests focalizados

```bash
source .venv/bin/activate
pytest tests/test_alpha_specs.py tests/test_alpha_research_runner.py tests/test_backtesting_metrics.py tests/test_cboe_risk_context.py tests/test_ibkr_read_only.py tests/test_promotion_gates.py tests/test_risk_off_eda.py tests/test_risk_off_short_strategy.py tests/test_risk_off_short_triage.py tests/test_risk_off_short_promotion_sweep.py tests/test_risk_off_short_h1b_sweep.py tests/test_risk_off_short_h1b_freeze.py tests/test_risk_off_short_h1b_robustness.py tests/test_risk_off_short_h1c_credit_repair.py tests/test_risk_off_short_h1c_freeze.py tests/test_risk_off_short_h1c_robustness.py tests/test_strategy_manifest.py
```

## Estado legacy

El repo aun contiene datos, resultados, reportes y scripts antiguos. No son la guia
activa. Se conservaran solo mientras se migren piezas utiles a los paquetes
activos bajo `src/`.

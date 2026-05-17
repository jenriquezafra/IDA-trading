# IDA Trading — Roadmap técnico y de producto

## 0. Tesis central

Sí tiene sentido construir la app **ahora**, pero no como “plataforma de trading” ni como dashboard visual libre. Debe empezar como un **research control plane local, read-only, artifact-driven y con disciplina experimental**.

La app no debe intentar descubrir edge por sí misma. Debe ayudarte a:

1. encontrar rápidamente qué candidatos sobreviven a costes, folds, ablations, leakage audits y baselines;
2. detectar fragilidad antes de tocar holdout/test;
3. comparar runs solo cuando sean comparables;
4. dejar trazabilidad completa de cada decisión;
5. preparar candidatos congelados para paper trading sin mezclar investigación histórica y monitorización live.

El mayor riesgo no es técnico. Es que la UI facilite mirar demasiadas curvas, ordenar por el mejor Sharpe y racionalizar después. Por tanto, el producto debe introducir **fricción experimental útil**.

---

# 1. Decisión recomendada

## Construir app: sí, pero con alcance limitado

| Pregunta | Recomendación |
|---|---|
| ¿Construir ahora? | Sí, si empieza como visor reproducible de artefactos y registry de runs. |
| ¿Construir una app completa? | No. Primero dashboard read-only + manifest + comparación disciplinada. |
| ¿Meter agente LLM desde el inicio? | Solo una versión read-only y evidence-driven después de tener manifests/index. |
| ¿Paper/live desde el inicio? | No. Diseñar la frontera, pero no mezclar todavía. |
| ¿UI interactiva de tuning? | Evitar al principio. Alto riesgo de overfitting visual. |

## Cuándo sí

Construye la app ahora si cumple estas condiciones:

- Lee artefactos existentes sin reimplementar lógica de research.
- Cada vista responde a una pregunta de decisión:
  - aceptar/rechazar hipótesis;
  - detectar leakage o fragilidad;
  - comparar contra baseline;
  - preparar paper trading.
- Todo run/candidato tiene:
  - `run_id`;
  - `candidate_id`;
  - config;
  - git commit;
  - dataset;
  - timeframe;
  - target;
  - feature set;
  - date range;
  - cost profile;
  - split policy;
  - artifact paths.
- La UI diferencia claramente:
  - exploratory;
  - validation;
  - confirmatory;
  - holdout/test;
  - paper/live.

## Cuándo no

No construyas todavía una app más ambiciosa si:

- todavía no hay contrato estable de artefactos;
- los resultados no tienen manifests reproducibles;
- no puedes reconstruir qué config generó qué parquet;
- quieres usar la UI para buscar “el mejor equity curve”;
- el dashboard te haría ejecutar más experimentos sin hipótesis previa;
- la app obliga a mover lógica de `src/` a callbacks de UI;
- el agente LLM tendría permiso para proponer conclusiones sin evidencia.

## Alcance inicial razonable

Primera versión útil:

1. **Run Browser**: lista de runs con manifest, estado y artefactos.
2. **Candidate Explorer**: tabla de candidatos con métricas netas, costes, folds, leakage, ablation y robustness.
3. **Run Comparison**: comparación controlada entre runs compatibles.
4. **HMM State Viewer**: estados, estabilidad, economía y ocupación.
5. **Evidence Viewer**: markdown reports, figuras y parquet ligados al run.
6. **Decision Log**: registrar decisiones humanas.
7. **Freeze Candidate Draft**: preparar, no aprobar automáticamente, candidato para paper.

No meter todavía:

- live execution;
- broker integration;
- optimizadores visuales;
- backtests lanzados desde cada widget;
- agente con permisos amplios.

## Cómo medir si la app ayuda a encontrar edge real

No lo midas por “más gráficos” ni por “más runs”. Mídelo por reducción de errores de investigación.

| Métrica | Buena señal | Mala señal |
|---|---:|---:|
| % runs con manifest completo | >95% | runs huérfanos |
| Tiempo hasta diagnosticar candidato | baja | sube por complejidad UI |
| Nº candidatos rechazados por fragilidad antes de holdout | sube | todo llega a test |
| Nº accesos a test/holdout | bajo y registrado | uso repetido para seleccionar |
| Nº comparaciones incompatibles bloqueadas o marcadas | sube inicialmente | comparas todo con todo |
| % decisiones con evidence paths | >90% | decisiones narrativas sin artefactos |
| Candidatos congelados que pasan a paper | pocos pero bien documentados | muchos por impulso |
| Diferencia validation → paper | estable/explicable | colapso sistemático |

La app aporta valor si aumenta la **calidad de falsación**: rechazar antes candidatos bonitos pero frágiles.

---

# 2. Arquitectura propuesta por fases

## Fase 0 — Preparación mínima del repo

Objetivo: crear un contrato estable entre pipeline y app.

### Entregables

- Manifest por run.
- Registry local SQLite.
- Indexador de artefactos existentes.
- Convención de paths.
- Checks de reproducibilidad.
- Separación clara `src/` vs `app/`.

### Principios

- `src/` sigue conteniendo toda la lógica de datos, features, HMM, modelos, señales, backtests y evaluación.
- `app/` solo lee, filtra, visualiza, registra decisiones y lanza scripts existentes cuando llegue Fase 3.
- Ningún cálculo de trading vive dentro de la UI.

### Estructura sugerida

```text
IDA-Trading/
  src/
    ida_trading/
      data/
      features/
      hmm/
      models/
      signals/
      backtests/
      evaluation/
      registry/
        __init__.py
        schemas.py
        db.py
        indexer.py
        manifest.py
        validators.py
      artifacts/
        __init__.py
        loader.py
        query.py
        reports.py
      agents/
        __init__.py
        tools.py
        prompts.py
        orchestrator.py
        permissions.py
        schemas.py

  app/
    ida_dashboard.py
    pages/
      01_Run_Browser.py
      02_Candidate_Explorer.py
      03_Compare_Runs.py
      04_HMM_States.py
      05_Robustness.py
      06_Reports.py
      07_Decisions.py
      08_Agent.py
      09_Paper_Monitor.py
    components/
      filters.py
      metric_cards.py
      warnings.py
      tables.py
      charts.py
    services/
      registry_service.py
      artifact_service.py
      agent_service.py

  configs/
  data/
  models/
  results/
    ida_registry.sqlite
    runs/
      RUN_YYYYMMDD_HHMMSS_xxxxx/
        manifest.yml
        config_snapshot.yml
        artifacts/
  reports/
  tests/
```

---

## Fase 1 — Research dashboard read-only

Objetivo: valor rápido sin introducir riesgo operativo.

### Funcionalidad

- Lee `manifest.yml`, parquet, YAML, markdown y figuras.
- No ejecuta backtests.
- No modifica configs.
- No genera señales nuevas.
- No toca paper/live.

### Vistas mínimas

- Run Browser.
- Candidate Explorer.
- Compare Runs.
- HMM State Viewer.
- Robustness / Cost / Leakage.
- Report Viewer.

### Resultado esperado

Puedes responder en menos de 2 minutos:

> “¿Este candidato sigue siendo interesante después de costes, folds, ablation, leakage audit y baseline?”

---

## Fase 2 — Experiment registry / run manifest

Objetivo: que no haya runs sin genealogía.

### Funcionalidad

- Generación de `run_id`.
- Registro de:
  - config hash;
  - git commit;
  - dirty flag;
  - dataset hash;
  - split policy;
  - cost profile;
  - target;
  - feature set;
  - artifacts.
- Tablas SQLite para indexar metadata.
- DuckDB para consultar parquet grandes directamente.

DuckDB encaja bien aquí porque permite consultar Parquet directamente desde SQL, evitando cargar todos los resultados en pandas para cada vista. SQLite encaja como registry local porque Python trae `sqlite3` en la librería estándar y ofrece persistencia transaccional sin desplegar servidor.

---

## Fase 3 — Launch panel controlado

Objetivo: ejecutar scripts existentes con trazabilidad, no convertir la UI en motor de research.

### Permitido

- Seleccionar script existente.
- Seleccionar config existente.
- Crear config snapshot.
- Mostrar diff.
- Exigir aprobación explícita.
- Lanzar proceso local.
- Registrar stdout/stderr, exit code y nuevos artefactos.

### Prohibido

- Modificar configs silenciosamente.
- Ejecutar grid search libre desde widgets.
- Ejecutar sobre holdout/test sin workflow confirmatory.
- Generar candidatos sin manifest.

---

## Fase 4 — Agente LLM integrado para research

Objetivo: asistente evidence-driven, no oráculo.

### Funcionalidad inicial

- Analyst read-only.
- Skeptic read-only.
- Planner con proposals, no ejecución.
- Operator solo con aprobación explícita.
- Decision logs.

El agente debe usar function/tool calling para conectarse con herramientas internas que consulten manifests, parquet, reports y registry. Para salidas estructuradas —por ejemplo experiment proposals, decision logs o candidate reviews— conviene usar JSON Schema o modelos Pydantic.

---

## Fase 5 — Paper trading monitor

Objetivo: monitorizar una estrategia congelada, sin contaminar research histórico.

### Separación estricta

```text
results/research/
results/paper/
results/live/
```

Paper debe tener su propio:

- `paper_strategy_id`;
- candidate freeze record;
- signal snapshots;
- market data timestamps;
- expected orders;
- simulated fills;
- latency assumptions;
- realized paper PnL;
- health checks;
- incident logs.

---

## Fase 6 — Production/live monitor, si procede

Objetivo: observabilidad, no autonomía del agente.

### Solo si antes existe

- estrategia paper estable;
- execution spec documentado;
- reconciliación paper vs expected;
- kill-switch manual;
- broker adapter testeado;
- alerting;
- runbooks;
- control de riesgos.

### Regla crítica

El agente puede explicar estado, diagnosticar discrepancias y redactar incident reports. **No debe enviar órdenes.**

---

# 3. Stack recomendado

## Decisión de stack

| Opción | Veredicto | Motivo |
|---|---|---|
| **Streamlit** | **Recomendado Fase 1-4** | Python-first, rápido, multipage, suficiente para dashboards locales y chat. |
| Dash | Alternativa si necesitas callbacks complejos o Plotly-heavy | Más estructura reactiva, más boilerplate. |
| FastAPI + React | Later | Útil para multiusuario, APIs, live monitor serio o frontend complejo. Prematuro ahora. |
| Notebooks | Mantener para exploración puntual | No son buen registry ni app reproducible. |
| CLI | Mantener | Excelente para pipelines, tests y reproducibilidad. |

## Dependencias a añadir

Must-have:

```text
streamlit
duckdb
pydantic
```

Should-have:

```text
openai          # Fase agente
plotly          # solo si necesitas charts interactivos más ricos
watchdog        # mejora DX con Streamlit, opcional
```

Later:

```text
fastapi
uvicorn
apscheduler
websockets
```

Evitar al principio:

```text
sqlalchemy      # sqlite3 basta inicialmente
celery          # demasiado pesado
redis           # innecesario local-first
postgres        # prematuro
react frontend  # prematuro
vector db       # no hace falta hasta tener mucho texto
langchain       # excesivo si solo necesitas tools controladas
mlflow          # útil, pero puede ser demasiado si tu manifest ya cubre lineage
```

## Mantener como scripts existentes

Todo esto debe seguir fuera de la UI:

- data cleaning;
- feature generation;
- HMM training;
- model training;
- signal generation;
- walk-forward;
- backtesting;
- cost sensitivity;
- leakage audits;
- ablation;
- risk filters;
- baseline comparison.

La UI llama o lee esos outputs. No debe duplicarlos.

---

# 4. Diseño funcional de la app

## Filtros globales

Estos filtros deben estar en sidebar y propagarse a todas las vistas:

| Filtro | Uso |
|---|---|
| `instrument` | SPY ahora; extensible. |
| `timeframe` | 5min inicialmente. |
| `date_range` | Filtra runs/candidatos por cobertura. |
| `split_role` | train / validation / test / holdout / paper. |
| `target` | Dirección, return, threshold, etc. |
| `feature_set` | Familia de features. |
| `hmm_spec` | n_states, features, seed, fit window. |
| `model_family` | baseline, HMM, XGB, ensemble, etc. |
| `cost_profile` | bps, spread, slippage, commissions. |
| `execution_model` | causal `open_{t+1}`. |
| `candidate_status` | exploratory, rejected, frozen, paper_candidate. |
| `leakage_status` | pass, warning, fail, unknown. |
| `run_tag` | alpha refinement, ablation, risk filters, etc. |

---

## Página 1 — Decision Board

Pregunta que responde:

> “¿Qué candidatos merecen atención y por qué?”

Elementos:

- Nº runs indexados.
- Nº candidatos por estado.
- Nº candidatos con leakage fail.
- Nº candidatos net positive after realistic costs.
- Nº candidatos robustos across folds.
- Nº candidatos congelados.
- Tabla “candidate triage”.

Columnas mínimas:

```text
candidate_id
run_id
status
validation_net_pnl
validation_sharpe
max_drawdown
trade_count
avg_trade_bps_net
turnover
cost_profile
break_even_cost_bps
fold_positive_ratio
worst_fold_pnl
baseline_delta
ablation_fragility_score
leakage_status
hmm_state_dependency
last_decision
```

Regla de diseño: default sort por `status`, `leakage_status`, `fold_positive_ratio`, `baseline_delta`, no por Sharpe máximo.

---

## Página 2 — Run Browser

Pregunta:

> “¿Qué produjo exactamente este run?”

Elementos:

- Manifest.
- Config snapshot.
- Git commit.
- Dirty flag.
- Dataset hash.
- Split policy.
- Cost profile.
- Artifacts list.
- Reports linked.
- Candidate IDs generados.

Vista ejemplo:

```text
RUN_20260504_142233_a91f3

Code:
  git_commit: 9f4a7c1
  dirty: false

Data:
  dataset_id: spy_5m_cleaned_v17
  data_hash: sha256:...
  train: 2016-01-04 → 2021-12-31
  validation: 2022-01-03 → 2024-12-31
  holdout: 2025-01-02 → 2025-12-31

Execution:
  decision_time: close_t
  fill_time: open_t+1
  overnight: false
  cost_profile: realistic_spy_5m_v3

Artifacts:
  results/.../candidate_metrics.parquet
  results/.../fold_metrics.parquet
  reports/.../baseline_comparison.md
  reports/.../leakage_audit.md
```

---

## Página 3 — Candidate Explorer

Pregunta:

> “¿Este candidato es robusto o solo parece bueno?”

### Tabla principal

Debe incluir:

- métricas gross y net;
- métricas por coste;
- folds;
- ablation;
- leakage;
- baseline;
- concentración;
- dependencia de régimen/HMM;
- estabilidad temporal.

### Drilldown por candidato

Secciones:

1. **Summary**
   - net PnL;
   - Sharpe;
   - max DD;
   - Calmar;
   - avg trade bps;
   - trade count;
   - exposure;
   - turnover;
   - capacity proxy.

2. **Costs**
   - coste base;
   - coste conservador;
   - break-even bps;
   - slope de PnL frente a coste;
   - porcentaje de edge consumido por costes.

3. **Folds**
   - fold metrics table;
   - worst fold;
   - positive fold ratio;
   - variance across folds;
   - performance by year/month.

4. **Ablation**
   - top features;
   - removal impact;
   - suspicious single-feature dependence;
   - feature family dependence.

5. **HMM**
   - PnL por estado;
   - exposure por estado;
   - signal frequency por estado;
   - state occupancy;
   - state stability across seeds/windows.

6. **Leakage**
   - causal alignment;
   - target leakage;
   - split leakage;
   - duplicate timestamps;
   - market calendar issues;
   - open/close alignment.

7. **Decision**
   - reject;
   - keep exploring;
   - freeze candidate;
   - send to paper review.

---

## Página 4 — Compare Runs

Pregunta:

> “¿La mejora viene de una hipótesis real o de cambiar demasiadas cosas a la vez?”

### Regla fundamental

La comparación debe mostrar un warning si difieren más de X dimensiones:

```text
dataset_id
date_range
split_policy
target
feature_set
cost_profile
execution_model
baseline
universe
timeframe
```

Ejemplo de warning:

```text
⚠️ Comparación no limpia:
- run_A usa cost_profile realistic_spy_5m_v2
- run_B usa cost_profile optimistic_v1
- run_A usa validation 2022-2023
- run_B usa validation 2022-2024

Conclusión causal no permitida. Use esta comparación solo como exploratoria.
```

### Comparaciones útiles

- candidate vs baseline;
- same features, different model;
- same model, different features;
- same signal, different cost profile;
- same candidate, risk filter on/off;
- HMM states on/off;
- execution delay sensitivity.

---

## Página 5 — HMM State Lab

Pregunta:

> “¿Los estados HMM aportan estructura económica estable o solo segmentan ruido?”

Vistas necesarias:

| Vista | Decisión que ayuda a tomar |
|---|---|
| State occupancy over time | Detectar estados raros o inestables. |
| Transition matrix | Ver persistencia/regime switching. |
| State economics | Saber si un estado tiene edge neto. |
| State feature profile | Interpretabilidad económica. |
| State stability across seeds/windows | Evitar estados semánticamente arbitrarios. |
| Candidate exposure by state | Ver si el candidato depende de un único régimen. |
| State mapping across runs | Comparar estados sin asumir mismo label numérico. |

Regla: no llamar a un estado “bull”, “bear”, “chop” salvo que el report lo justifique con evidencia. Por defecto: `state_0`, `state_1`, etc.

---

## Página 6 — Cost, Execution & Slippage

Pregunta:

> “¿El edge sobrevive a costes realistas y ejecución causal?”

Vistas:

- Net PnL vs cost bps.
- Break-even cost.
- Sharpe vs cost.
- Trade count vs threshold.
- Avg trade bps gross/net.
- Delay tests:
  - `open_{t+1}`;
  - `vwap_next_bar`;
  - `close_{t+1}` si aplica;
  - missed fill sensitivity.
- Cost profile comparison:
  - optimistic;
  - realistic;
  - conservative;
  - stress.

Warning obligatorio:

```text
⚠️ Este candidato tiene avg_trade_bps_net = 0.7 bps y break_even_cost_bps = 1.2.
Cualquier subestimación de slippage puede eliminar el edge.
```

---

## Página 7 — Robustness, Ablation & Folds

Pregunta:

> “¿La señal es estable o depende de una configuración concreta?”

Vistas:

- Walk-forward fold table.
- Worst fold diagnostics.
- Fold dispersion.
- Year/month heatmap.
- Parameter sensitivity.
- Feature ablation.
- Risk filter on/off.
- Baseline comparison.
- Recent period performance.

Debe destacar:

- performance concentrada en pocos días;
- edge concentrado en un HMM state raro;
- un único fold explica el PnL;
- ablation mata la estrategia al quitar una feature sospechosa;
- la mejora frente a baseline es menor que el error esperado.

---

## Página 8 — Leakage & Causality Audit

Pregunta:

> “¿Estoy mirando el futuro sin darme cuenta?”

Checks visibles:

```text
timestamp_monotonic
duplicate_bars
market_calendar_alignment
feature_lag_consistency
target_shift_consistency
decision_time_before_fill_time
open_t+1_available_only_after_decision
train_validation_test_disjoint
scaler_fit_only_train
hmm_fit_only_train_or_wf_window
model_fit_no_future_data
costs_applied_after_signal
overnight_positions_absent
```

Estados:

```text
PASS
WARNING
FAIL
UNKNOWN
```

Regla: un candidato con `FAIL` no puede pasar a freeze.

---

## Página 9 — Reports & Evidence

Pregunta:

> “¿Dónde está la evidencia generada por el pipeline?”

Funcionalidad:

- Render de markdown reports.
- Figuras asociadas al run.
- Links a parquet.
- Schema preview.
- Hash del artifact.
- Extractos citables por el agente.
- Botón “add to decision log”.

---

## Página 10 — Freeze / Paper Prep

Pregunta:

> “¿Este candidato está suficientemente definido para evaluarse fuera de muestra o en paper?”

Checklist:

```text
[ ] Manifest completo
[ ] Config congelada
[ ] Git commit limpio
[ ] Dataset snapshot congelado
[ ] Split policy documentada
[ ] Holdout no usado para selección
[ ] Leakage audit PASS
[ ] Cost profile realista/conservador
[ ] Baseline comparison positiva
[ ] Fold stability aceptable
[ ] Ablation sin dependencia sospechosa
[ ] HMM state dependence razonable
[ ] Risk limits definidos
[ ] Execution assumptions definidas
[ ] Paper duration definida
[ ] Paper success/failure criteria definidos
[ ] Decision log creado
```

---

# 5. Diseño del agente tipo ChatGPT/Codex

## Arquitectura

```text
Streamlit Chat UI
      |
      v
agent_service.py
      |
      v
ida_trading.agents.orchestrator
      |
      +--> Tool registry
      |       list_runs
      |       load_run_manifest
      |       query_results
      |       read_report
      |       inspect_candidate
      |       compare_runs
      |       create_decision_log
      |       run_pipeline_step_with_approval
      |
      +--> Permission layer
      |
      +--> Evidence formatter
      |
      +--> SQLite agent memory/logs
```

## Contrato del agente

El agente debe responder siempre con:

```text
Conclusión
Evidencia usada
Limitaciones
Siguiente acción recomendada
```

Y cuando sea relevante:

```text
No puedo concluir X porque falta Y.
```

No debe decir:

```text
“Parece robusto”
```

sin especificar:

```text
run_id
candidate_id
artifact path
metric
split
cost_profile
date range
```

---

## Evidence schema

```python
class EvidenceRef(BaseModel):
    run_id: str | None
    candidate_id: str | None
    artifact_path: str
    artifact_type: str
    metric_name: str | None
    metric_value: float | str | None
    split_role: str | None
    query: str | None
    row_filter: dict | None
    timestamp_utc: str
```

Respuesta ejemplo:

```text
Conclusión:
El candidato C_017 no debería congelarse todavía.

Evidencia:
- validation_net_pnl cae de 42.1k a 5.4k al pasar de realistic_v2 a conservative_v1.
  Fuente: results/runs/RUN_.../artifacts/cost_sensitivity.parquet
- 78% del PnL viene de fold_2023_03_2023_06.
  Fuente: results/runs/RUN_.../artifacts/fold_metrics.parquet
- leakage_audit tiene WARNING en feature_lag_consistency.
  Fuente: reports/RUN_.../leakage_audit.md

Limitación:
No he consultado holdout/test. Esta conclusión es solo sobre validation.

Siguiente acción:
Ejecutar ablation específica de feature family `intraday_gap_reversal` antes de cualquier freeze.
```

---

## Permisos por modo

| Permiso | Descripción |
|---|---|
| `READ_ARTIFACTS` | Leer manifests, parquet, markdown, figuras metadata. |
| `WRITE_NOTES` | Crear research notes y decision logs. |
| `PROPOSE_CONFIG` | Generar config propuesta, no guardarla como activa. |
| `WRITE_CONFIG_DRAFT` | Guardar config draft en carpeta `configs/proposals/`. |
| `RUN_APPROVED_SCRIPT` | Ejecutar script existente tras aprobación explícita. |
| `READ_PAPER_STATE` | Leer paper/live monitor. |
| `PLACE_ORDERS` | Nunca conceder al agente. |

---

## Memoria/contexto

Guardar en SQLite:

```text
agent_sessions
agent_messages
agent_tool_calls
agent_evidence_refs
decision_logs
experiment_proposals
```

Memoria por:

- `session_id`;
- `run_id`;
- `candidate_id`;
- `hypothesis_id`;
- `paper_strategy_id`.

Regla: la memoria no sustituye evidencia. El agente puede recordar que “se rechazó C_017”, pero debe citar el decision log.

---

## Aprobación antes de lanzar scripts

El agente debe producir un objeto `PendingAction`:

```yaml
action_type: run_pipeline_step
proposed_by: agent
mode: Operator
script: src/ida_trading/evaluation/run_ablation.py
config_input: configs/proposals/ablation_C_017_v1.yml
expected_outputs:
  - results/runs/{new_run_id}/manifest.yml
  - results/runs/{new_run_id}/artifacts/ablation_summary.parquet
reason: >
  Validar si el candidato C_017 depende excesivamente de intraday_gap_reversal.
safety_checks:
  uses_holdout: false
  modifies_raw_data: false
  modifies_existing_artifacts: false
  creates_new_run_id: true
requires_user_approval: true
```

La UI debe mostrar:

```text
[Approve exact command]
[Reject]
[Edit proposal]
```

---

## Qué no debe poder hacer

- Enviar órdenes.
- Cambiar raw data.
- Sobrescribir artefactos.
- Modificar configs congeladas.
- Acceder a holdout/test para seleccionar candidatos.
- Ocultar runs negativos.
- Ordenar candidatos por test PnL.
- Crear nuevas features directamente en `src/` sin revisión humana.
- “Arreglar” resultados.
- Cambiar costes para que un candidato pase.
- Declarar edge sin baseline, costes y folds.

---

# 6. Modos del agente

## Tabla resumen

| Modo | Capacidades | Inputs | Outputs | Permisos | Límites | Prompts útiles |
|---|---|---|---|---|---|---|
| Analyst | Resume runs, compara candidatos, explica métricas. | `run_id`, `candidate_id`, filtros. | Resumen con evidencia. | `READ_ARTIFACTS` | No propone cambios agresivos. | “Resume C_023 frente a baseline y costes realistas.” |
| Skeptic | Busca fallos, leakage, fragilidad, concentración. | candidato/run/familia. | Red flags, falsación, missing evidence. | `READ_ARTIFACTS` | No aprueba estrategias. | “Intenta desmontar C_023 antes de freeze.” |
| Planner | Propone próximos experimentos con hipótesis. | hallazgos previos, constraints. | Experiment proposal estructurado. | `READ_ARTIFACTS`, `PROPOSE_CONFIG` | No ejecuta. No usa test para selección. | “Propón 3 experimentos para falsar dependencia de HMM state_2.” |
| Operator | Lanza scripts existentes con aprobación. | script, config, proposal. | PendingAction, run log, manifest. | `RUN_APPROVED_SCRIPT` | No ejecuta sin aprobación. No toca raw/frozen. | “Prepara ejecución de ablation para C_023, sin holdout.” |
| Monitor | Observa paper/live, explica estado. | `paper_strategy_id`, fechas. | Estado, alertas, discrepancias. | `READ_PAPER_STATE` | No opera órdenes. | “Explica por qué hoy no hubo trades en paper.” |

---

## Analyst

### Capacidades

- Resumir un run.
- Explicar un candidato.
- Comparar run A vs run B.
- Extraer evidencia de reports.
- Generar “evidence bundle”.
- Traducir resultados técnicos a decisión.

### Inputs

```text
run_id
candidate_id
cost_profile
split_role
date_range
baseline_id
```

### Output esperado

```text
- Conclusión
- Métricas principales
- Comparación contra baseline
- Evidencia
- Limitaciones
```

### Ejemplo de prompt

```text
Analyst: compara RUN_20260504_A y RUN_20260504_B.
Solo concluye si dataset, split, target, timeframe y cost_profile son compatibles.
```

---

## Skeptic

### Capacidades

- Buscar overfitting visual.
- Buscar dependencia de un fold.
- Buscar dependencia de un HMM state raro.
- Revisar leakage warnings.
- Revisar coste break-even.
- Revisar ablation fragility.
- Revisar trade concentration.
- Señalar que falta evidencia.

### Output esperado

```text
Verdict: reject / needs more evidence / freeze candidate possible

Red flags:
1. ...
2. ...

Critical missing evidence:
- ...

Recommended falsification:
- ...
```

### Ejemplo de prompt

```text
Skeptic: intenta rechazar candidate_id C_041.
No uses holdout. Prioriza costes, folds, ablation, leakage y concentración temporal.
```

---

## Planner

### Capacidades

- Proponer experimentos falsables.
- Definir hipótesis.
- Definir criterio de éxito.
- Definir criterio de fallo.
- Definir split permitido.
- Generar config draft.

### Output schema

```yaml
proposal_id: EXP_20260504_001
hypothesis: >
  La mejora viene de filtrar señales en HMM state_2, no de leakage ni de costes optimistas.
experiment_type: ablation
candidate_id: C_041
allowed_splits:
  - train
  - validation
forbidden_splits:
  - holdout
  - test
success_criteria:
  - validation_net_pnl remains positive under realistic and conservative costs
  - positive_fold_ratio >= 0.65
  - baseline_delta_net > 0
failure_criteria:
  - pnl explained by one fold > 50%
  - break_even_cost_bps < realistic_cost_bps * 1.5
  - leakage audit != PASS
outputs_expected:
  - ablation_summary.parquet
  - cost_sensitivity.parquet
  - leakage_audit.md
```

### Ejemplo de prompt

```text
Planner: propón un experimento mínimo para saber si el alpha refinement mejoró edge real o solo redujo trades malos en validation.
```

---

## Operator

### Capacidades

- Preparar comando exacto.
- Crear config draft.
- Mostrar diff.
- Validar que no usa holdout/test indebidamente.
- Ejecutar tras aprobación.
- Registrar nuevo run.

### Límites

- No decide por ti.
- No ejecuta sin aprobación explícita.
- No modifica scripts.
- No edita raw data.
- No sobrescribe artefactos.

### Ejemplo de prompt

```text
Operator: prepara, pero no ejecutes, el script de cost sensitivity para C_041 usando conservative_cost_v2 y validation only.
```

---

## Monitor

### Capacidades

- Leer paper signals.
- Leer simulated fills.
- Explicar PnL paper.
- Detectar divergencia expected vs actual.
- Revisar health checks.
- Crear incident notes.

### Límites

- No lanza órdenes.
- No cambia risk limits.
- No recalibra modelos.
- No mezcla paper con research selection.

### Ejemplo de prompt

```text
Monitor: resume el estado paper de STRAT_SPY_5M_001 esta semana.
Distingue signal quality, execution assumptions y PnL attribution.
```

---

# 7. Herramientas internas del agente

| Tool | Propósito | Inputs | Outputs | Riesgos | Safeguards |
|---|---|---|---|---|---|
| `list_runs` | Listar runs indexados. | filtros globales. | tabla resumida. | Exceso de runs irrelevantes. | paginación, filtros, status. |
| `load_run_manifest` | Cargar manifest completo. | `run_id`. | manifest dict. | Manifest incompleto. | validar schema, marcar missing fields. |
| `query_results` | Consultar parquet/SQLite. | SQL restringido, artifact type. | dataframe/json + evidence. | SQL amplio o lento. | solo SELECT, limit obligatorio, whitelist paths. |
| `read_report` | Leer markdown report. | path, section. | extracto + path. | Citas fuera de contexto. | limitar longitud, incluir sección. |
| `compare_runs` | Comparar runs compatibles. | run_ids, metrics. | diff + warnings. | Comparaciones inválidas. | compatibility check obligatorio. |
| `inspect_candidate` | Resumen de candidato. | `candidate_id`. | metrics, artifacts, warnings. | Sesgo por métrica única. | devuelve costes/folds/leakage siempre. |
| `inspect_hmm_state` | Ver estado HMM. | `run_id`, `state_id`. | occupancy, economics, stability. | Interpretación semántica falsa. | no nombrar estado sin report. |
| `cost_sensitivity_summary` | Resumir robustez a costes. | `candidate_id`. | break-even, slope, stress. | Costes optimistas. | incluir realistic y conservative. |
| `fold_stability_summary` | Ver estabilidad WFO. | `candidate_id`. | fold table, worst fold. | Ocultar folds malos. | mostrar todos los folds. |
| `ablation_summary` | Ver dependencia de features. | `candidate_id`. | feature impact. | Confundir correlación con causalidad. | marcar como evidence, no prueba definitiva. |
| `leakage_audit_summary` | Revisar leakage. | `run_id`/`candidate_id`. | pass/warn/fail. | Ignorar warnings. | fail bloquea freeze. |
| `propose_experiment` | Crear experimento falsable. | hipótesis, candidato. | proposal YAML. | Data snooping. | forbidden_splits explícitos. |
| `create_decision_log` | Registrar decisión humana. | decisión, evidencia. | markdown/yaml log. | Narrativas post-hoc. | evidence refs obligatorios. |
| `create_config_draft` | Crear config propuesta. | base config, patch. | draft path + diff. | Cambios invisibles. | diff obligatorio, no overwrite. |
| `run_pipeline_step_with_approval` | Ejecutar script existente. | pending action approved. | new run_id, logs. | Ejecución accidental. | confirmación exacta, allowlist scripts. |
| `load_paper_state` | Leer paper monitor. | `paper_strategy_id`. | current state. | Mezclar research/paper. | namespace separado. |
| `create_incident_note` | Registrar incidente paper/live. | event, evidence. | incident log. | Diagnóstico especulativo. | evidence refs obligatorios. |

---

# 8. Diseño anti-overfitting

## Principios

1. La UI debe dificultar el cherry-picking.
2. El test/holdout no es una herramienta de exploración.
3. Cada candidato debe tener genealogía.
4. Cada comparación debe declarar compatibilidad.
5. Toda decisión debe registrar evidencia.
6. Cada cambio después de mirar validation/test crea una nueva rama experimental.
7. El agente debe actuar como auditor, no como vendedor de estrategias.

---

## Separación de datasets

```text
train:
  fit scalers, HMM, models, thresholds where applicable

validation:
  model/candidate selection
  ablation
  cost sensitivity
  risk filters

test/holdout:
  confirmatory only
  limited access
  no iterative tuning

paper:
  forward simulation using frozen candidate

live:
  production observation/execution, separated from research
```

## Exploratory vs confirmatory

| Tipo | Permitido | Prohibido |
|---|---|---|
| Exploratory | buscar hipótesis, comparar variantes, mirar diagnostics | declarar edge final |
| Validation | seleccionar candidatos, ajustar filtros predefinidos | tocar holdout repetidamente |
| Confirmatory | evaluar candidato congelado | cambiar config tras ver resultado |
| Paper | forward check operacional | usar resultados para recalibrar sin nuevo research cycle |
| Live | monitorización y control de riesgo | investigación ad hoc dentro de producción |

---

## Warnings/gates en UI

Ejemplos:

```text
⚠️ Candidate selected using holdout metric.
Este candidato no puede marcarse como confirmatory. Cree una nueva familia experimental.
```

```text
⚠️ Incompatible comparison.
Los runs difieren en target, cost_profile y date_range. No interprete el delta como efecto del modelo.
```

```text
⚠️ Cost fragility.
Break-even cost = 1.1 bps; realistic cost = 0.9 bps. Margen insuficiente.
```

```text
⚠️ Fold concentration.
El 64% del PnL viene de 1 de 12 folds.
```

```text
⚠️ HMM state dependency.
El 82% del PnL viene de state_3, con occupancy media 6.4%.
```

```text
❌ Leakage audit FAIL.
Freeze bloqueado.
```

---

## Workflow de freeze candidate

```text
1. Candidate pasa triage en validation.
2. Se crea freeze proposal.
3. Se bloquea config snapshot.
4. Se registra git commit limpio.
5. Se registra dataset snapshot/hash.
6. Se documenta hipótesis.
7. Se documentan criterios de éxito/fallo.
8. Se verifica que holdout/test no se usó para selección.
9. Se ejecuta confirmatory una sola vez, o se manda a paper.
10. Cualquier cambio posterior crea candidate_id nuevo.
```

## Gates mínimos para freeze

```text
[ ] leakage_audit == PASS
[ ] execution_model == open_t+1 causal
[ ] no overnight
[ ] validation net PnL > 0 under realistic costs
[ ] validation net PnL > 0 or acceptable under conservative costs
[ ] baseline_delta_net > 0
[ ] positive_fold_ratio >= threshold predefinido
[ ] worst_fold_loss tolerable
[ ] no single fold explains > threshold PnL
[ ] break_even_cost_bps >= realistic_cost_bps * margin
[ ] trade_count >= minimum
[ ] avg_trade_bps_net sufficiently above noise
[ ] ablation does not reveal single suspicious feature dependency
[ ] HMM state dependency documented
[ ] decision log exists
```

---

# 9. Modelo de datos

## Run manifest recomendado

```yaml
schema_version: 1

run:
  run_id: RUN_20260504_142233_a91f3
  created_at_utc: "2026-05-04T12:22:33Z"
  created_by: enrique
  project: IDA Trading
  run_type: alpha_refinement
  status: completed
  parent_run_id: RUN_20260503_180812_b773c
  tags:
    - spy
    - 5min
    - hmm
    - no_overnight

code:
  git_commit: 9f4a7c1d8b2
  git_branch: main
  git_dirty: false
  python_version: "3.11.8"
  environment_file: requirements.txt

data:
  instrument: SPY
  timeframe: 5min
  dataset_id: spy_5m_cleaned_aligned_v17
  dataset_path: data/aligned/spy_5m_v17.parquet
  dataset_hash: sha256:abc123
  calendar: XNYS
  timezone: America/New_York
  data_start: "2016-01-04"
  data_end: "2025-12-31"

splits:
  split_policy_id: wf_spy_5m_v4
  train:
    start: "2016-01-04"
    end: "2021-12-31"
  validation:
    start: "2022-01-03"
    end: "2024-12-31"
  holdout:
    start: "2025-01-02"
    end: "2025-12-31"
  test_access_count_before_run: 0

research:
  hypothesis_id: HYP_20260504_001
  hypothesis: >
    HMM state-conditioned reversal filter improves net PnL after realistic costs.
  experiment_stage: validation
  allowed_for_selection: true

config:
  config_path: configs/alpha_refinement/spy_5m_hmm_reversal.yml
  config_snapshot_path: results/runs/RUN_20260504_142233_a91f3/config_snapshot.yml
  config_hash: sha256:def456

features:
  feature_set_id: fs_intraday_hmm_v9
  feature_manifest_path: data/features/fs_intraday_hmm_v9_manifest.yml
  feature_hash: sha256:ghi789

hmm:
  enabled: true
  n_states: 4
  hmm_features:
    - intraday_vol
    - overnight_gap_proxy
    - realized_vol_30m
  fit_policy: walk_forward_train_only
  seed: 42

model:
  model_family: xgboost
  model_id: xgb_spy_5m_v12
  target_id: next_bar_direction_thresholded_v3
  model_artifact_path: models/RUN_20260504_142233_a91f3/model.joblib

signal:
  signal_id: sig_hmm_reversal_v5
  threshold_policy: validation_locked
  no_overnight: true

execution:
  decision_time: close_t
  fill_time: open_t_plus_1
  allow_overnight: false
  position_sizing: fixed_notional
  max_position: 1

costs:
  cost_profile_id: realistic_spy_5m_v3
  commission_bps: 0.0
  spread_bps: 0.5
  slippage_bps: 0.7
  total_cost_bps_roundtrip: 1.2

artifacts:
  - artifact_id: candidate_metrics
    type: parquet
    path: results/runs/RUN_20260504_142233_a91f3/artifacts/candidate_metrics.parquet
    hash: sha256:...
    schema_id: candidate_metrics_v1
  - artifact_id: fold_metrics
    type: parquet
    path: results/runs/RUN_20260504_142233_a91f3/artifacts/fold_metrics.parquet
    hash: sha256:...
    schema_id: fold_metrics_v1
  - artifact_id: leakage_audit
    type: markdown
    path: reports/RUN_20260504_142233_a91f3/leakage_audit.md
    hash: sha256:...

summary_metrics:
  validation_net_pnl: 18420.5
  validation_sharpe: 1.21
  validation_max_drawdown: -6200.0
  trade_count: 912
  positive_fold_ratio: 0.67
  break_even_cost_bps: 2.4
  baseline_delta_net: 7300.2
  leakage_status: PASS

candidates:
  - candidate_id: C_20260504_001
```

---

## Esquema mínimo SQLite

```sql
runs(
  run_id text primary key,
  created_at_utc text,
  run_type text,
  status text,
  git_commit text,
  git_dirty integer,
  config_hash text,
  dataset_id text,
  dataset_hash text,
  instrument text,
  timeframe text,
  target_id text,
  feature_set_id text,
  split_policy_id text,
  cost_profile_id text,
  experiment_stage text,
  manifest_path text
);

artifacts(
  artifact_id text primary key,
  run_id text,
  candidate_id text,
  artifact_type text,
  logical_name text,
  path text,
  hash text,
  schema_id text,
  created_at_utc text
);

candidates(
  candidate_id text primary key,
  run_id text,
  candidate_family_id text,
  status text,
  hypothesis_id text,
  signal_id text,
  model_id text,
  feature_set_id text,
  cost_profile_id text,
  created_at_utc text
);

candidate_metrics(
  candidate_id text,
  run_id text,
  split_role text,
  cost_profile_id text,
  metric_name text,
  metric_value real,
  artifact_path text,
  primary key(candidate_id, split_role, cost_profile_id, metric_name)
);

fold_metrics(
  candidate_id text,
  run_id text,
  fold_id text,
  split_role text,
  metric_name text,
  metric_value real,
  artifact_path text
);

reports(
  report_id text primary key,
  run_id text,
  candidate_id text,
  report_type text,
  path text,
  hash text
);

decision_logs(
  decision_id text primary key,
  created_at_utc text,
  human_owner text,
  decision_type text,
  run_id text,
  candidate_id text,
  decision text,
  rationale text,
  evidence_json text,
  next_action text
);

experiment_proposals(
  proposal_id text primary key,
  created_at_utc text,
  created_by text,
  mode text,
  hypothesis text,
  candidate_id text,
  status text,
  proposal_yaml_path text
);

candidate_freeze_records(
  freeze_id text primary key,
  candidate_id text,
  created_at_utc text,
  git_commit text,
  config_hash text,
  dataset_hash text,
  allowed_next_stage text,
  freeze_record_path text
);

agent_sessions(
  session_id text primary key,
  created_at_utc text,
  mode text,
  linked_run_id text,
  linked_candidate_id text
);

agent_messages(
  message_id text primary key,
  session_id text,
  created_at_utc text,
  role text,
  content text,
  evidence_json text
);
```

---

# 10. Roadmap con milestones

## Milestone 0 — Artifact contract & manifests

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Que cada run sea reconstruible. |
| Entregables | `manifest.yml`, schema Pydantic, validator, config snapshot. |
| Checklist técnico | crear `registry/schemas.py`; crear `manifest.py`; hash configs/datasets; capturar git commit; tests. |
| Criterios de aceptación | 10 runs existentes pueden indexarse; manifests pasan validación; no hay artifact sin path. |
| Riesgos | Descubrir artefactos inconsistentes. |
| Mitigación | Permitir manifests parciales con `UNKNOWN`, pero mostrar warning. |

---

## Milestone 1 — Local registry index

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Consultar runs/candidatos sin escanear carpetas cada vez. |
| Entregables | `results/ida_registry.sqlite`, indexer CLI. |
| Checklist | tablas SQLite; indexar manifests; indexar reports; indexar parquet metadata; tests de idempotencia. |
| Aceptación | `python -m ida_trading.registry.indexer --results results/` rellena DB reproduciblemente. |
| Riesgos | Duplicados, paths rotos. |
| Mitigación | unique constraints + artifact hash. |

---

## Milestone 2 — Read-only Streamlit dashboard

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Ver runs y candidatos sin tocar pipeline. |
| Entregables | `app/ida_dashboard.py`, páginas Run Browser y Candidate Explorer. |
| Checklist | filtros globales; tablas; links a artifacts; markdown viewer; warnings básicos. |
| Aceptación | puedes abrir un run, ver manifest, métricas y reports en menos de 3 clicks. |
| Riesgos | UI se convierte en notebook glorificado. |
| Mitigación | cada sección debe tener pregunta de decisión. |

---

## Milestone 3 — Candidate diagnostics

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Diagnosticar robustez de candidatos. |
| Entregables | vistas de cost sensitivity, folds, ablation, leakage, HMM dependency. |
| Checklist | drilldown candidato; cost slope; fold table; leakage status; ablation impact; state exposure. |
| Aceptación | para cualquier candidato se puede producir verdict: reject / continue / freeze review. |
| Riesgos | demasiados gráficos. |
| Mitigación | priorizar tablas y warnings; equity curve secundaria. |

---

## Milestone 4 — Compare Runs con compatibility gates

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Comparar runs sin conclusiones falsas. |
| Entregables | página Compare Runs. |
| Checklist | compatibility matrix; metric deltas; config diff; cost diff; dataset diff. |
| Aceptación | si dos runs difieren en dimensiones críticas, la UI muestra warning claro. |
| Riesgos | racionalización post-hoc. |
| Mitigación | marcar comparación como exploratory si no es limpia. |

---

## Milestone 5 — Decision logs & freeze workflow

| Campo | Detalle |
|---|---|
| Prioridad | Must-have |
| Objetivo | Registrar decisiones y preparar paper sin cherry-picking. |
| Entregables | decision log form; freeze draft; freeze validator. |
| Checklist | evidence refs obligatorios; checklist freeze; bloqueo por leakage fail; candidate status. |
| Aceptación | no se puede marcar candidato como frozen sin manifest completo y leakage pass. |
| Riesgos | fricción excesiva. |
| Mitigación | plantillas cortas y autocompletado desde registry. |

---

## Milestone 6 — Agente Analyst/Skeptic read-only

| Campo | Detalle |
|---|---|
| Prioridad | Should-have |
| Objetivo | Hacer análisis textual evidence-driven. |
| Entregables | página Agent; tools read-only; evidence formatter. |
| Checklist | `list_runs`; `inspect_candidate`; `compare_runs`; `read_report`; `leakage_audit_summary`. |
| Aceptación | el agente responde citando paths/run_ids/candidate_ids y reconoce falta de evidencia. |
| Riesgos | hallucinations. |
| Mitigación | tools estructuradas + respuesta con EvidenceRefs + no raw claims. |

---

## Milestone 7 — Planner + experiment proposals

| Campo | Detalle |
|---|---|
| Prioridad | Should-have |
| Objetivo | Proponer experimentos falsables. |
| Entregables | proposal schema; proposal viewer; config draft optional. |
| Checklist | hipótesis; success/failure criteria; forbidden splits; expected artifacts. |
| Aceptación | cada proposal puede aprobarse/rechazarse y queda registrado. |
| Riesgos | generar demasiados experimentos. |
| Mitigación | experiment budget por candidate family. |

---

## Milestone 8 — Operator con aprobación

| Campo | Detalle |
|---|---|
| Prioridad | Later / Should-have si hay fricción alta |
| Objetivo | Ejecutar scripts existentes desde UI con trazabilidad. |
| Entregables | launch panel; pending actions; subprocess logs. |
| Checklist | allowlist scripts; config diff; approval; new run_id; log stdout/stderr. |
| Aceptación | ningún script corre sin aprobación explícita. |
| Riesgos | UI como motor de overfitting. |
| Mitigación | no grid search libre; no holdout salvo confirmatory. |

---

## Milestone 9 — Paper monitor

| Campo | Detalle |
|---|---|
| Prioridad | Later |
| Objetivo | Monitorizar estrategia congelada en forward paper. |
| Entregables | paper namespace; paper strategy manifest; signal/fill/PnL views. |
| Checklist | no mezcla research; expected vs actual; health checks; incident logs. |
| Aceptación | paper run reproducible y ligado a freeze record. |
| Riesgos | contaminar research con paper tweaks. |
| Mitigación | cambios generan nuevo research cycle. |

---

## Milestone 10 — Live monitor

| Campo | Detalle |
|---|---|
| Prioridad | Later |
| Objetivo | Observabilidad de producción. |
| Entregables | broker read adapter, health dashboard, risk limits, kill switch manual. |
| Checklist | paper estable; runbooks; reconciliation; alerting. |
| Aceptación | monitor explica estado sin agente ejecutando órdenes. |
| Riesgos | complejidad operacional. |
| Mitigación | agente read-only; órdenes fuera del agente. |

---

# 11. Qué NO construir todavía

## No construir ahora

| Feature | Motivo |
|---|---|
| Optimizer visual interactivo | Alto riesgo de overfitting visual. |
| Drag-and-drop strategy builder | Prematuro y rompe reproducibilidad. |
| Broker integration | No aporta a encontrar edge histórico robusto. |
| Multiuser auth | App local primero. |
| React frontend | Sobrediseño. |
| Realtime streaming complejo | Innecesario antes de paper. |
| Vector DB | Primero metadata estructurada y paths. |
| Backtest engine dentro de UI | Duplica `src/` y aumenta bugs. |
| AutoML loops desde dashboard | Incentiva búsqueda ciega. |
| “Best candidate leaderboard” por Sharpe | Fomenta cherry-picking. |
| Agente que edita código | Riesgo alto, bajo valor inicial. |
| Agente que opera órdenes | Prohibido por diseño. |
| Test leaderboard | Contamina holdout. |

## Cosas que deben seguir como scripts

- generación de features;
- entrenamiento HMM;
- walk-forward;
- backtesting;
- ablation;
- leakage audits;
- candidate search;
- risk filters;
- baseline comparison;
- report generation.

---

# 12. Plan inicial de implementación: 1-2 semanas

## Semana 1 — Dashboard útil sin agente

### Día/Bloque 1 — Schemas y manifest

Crear:

```text
src/ida_trading/registry/schemas.py
src/ida_trading/registry/manifest.py
src/ida_trading/registry/validators.py
tests/test_manifest_schema.py
```

Checklist:

```text
[ ] RunManifest Pydantic model
[ ] ArtifactRef model
[ ] CandidateRef model
[ ] CostProfile model
[ ] SplitPolicy model
[ ] load_manifest(path)
[ ] validate_manifest(path)
[ ] hash_file(path)
[ ] get_git_commit()
[ ] get_git_dirty()
```

---

### Día/Bloque 2 — SQLite registry

Crear:

```text
src/ida_trading/registry/db.py
src/ida_trading/registry/indexer.py
tests/test_registry_indexer.py
```

Comando:

```bash
python -m ida_trading.registry.indexer --results results --reports reports --db results/ida_registry.sqlite
```

Checklist:

```text
[ ] Crear tablas si no existen
[ ] Indexar manifests
[ ] Indexar artifacts
[ ] Indexar reports
[ ] Idempotencia
[ ] Warnings para paths rotos
```

---

### Día/Bloque 3 — Artifact loader

Crear:

```text
src/ida_trading/artifacts/loader.py
src/ida_trading/artifacts/query.py
src/ida_trading/artifacts/reports.py
tests/test_artifact_loader.py
```

Checklist:

```text
[ ] read_parquet_sample(path, limit)
[ ] query_parquet(path, sql_template)
[ ] read_markdown(path)
[ ] list_artifacts(run_id)
[ ] schema preview
```

---

### Día/Bloque 4 — Streamlit skeleton

Crear:

```text
app/ida_dashboard.py
app/components/filters.py
app/components/warnings.py
app/services/registry_service.py
app/services/artifact_service.py
```

Comando:

```bash
streamlit run app/ida_dashboard.py
```

Checklist:

```text
[ ] Sidebar filters
[ ] DB connection cached
[ ] Run table
[ ] Candidate table
[ ] Manifest viewer
[ ] Report viewer básico
```

---

### Día/Bloque 5 — Candidate Explorer

Crear:

```text
app/pages/02_Candidate_Explorer.py
app/components/metric_cards.py
app/components/tables.py
```

Checklist:

```text
[ ] Candidate triage table
[ ] Drilldown
[ ] Cost summary
[ ] Fold summary
[ ] Leakage status
[ ] Baseline delta
[ ] Links a artifacts
```

---

### Día/Bloque 6 — Compare Runs

Crear:

```text
app/pages/03_Compare_Runs.py
src/ida_trading/registry/compatibility.py
tests/test_run_compatibility.py
```

Checklist:

```text
[ ] Compatibility matrix
[ ] Config diff
[ ] Dataset/split/cost diff
[ ] Metric deltas
[ ] Warning si comparación no limpia
```

---

### Día/Bloque 7 — HMM + Reports + Decisions

Crear:

```text
app/pages/04_HMM_States.py
app/pages/06_Reports.py
app/pages/07_Decisions.py
src/ida_trading/registry/decisions.py
tests/test_decision_logs.py
```

Checklist:

```text
[ ] HMM state economics
[ ] State occupancy
[ ] State stability
[ ] Markdown report viewer
[ ] Decision log form
[ ] Evidence refs obligatorios
```

---

## Semana 2 — Agente mínimo y freeze workflow

### Día/Bloque 8 — Freeze records

Crear:

```text
src/ida_trading/registry/freeze.py
app/pages/10_Freeze_Candidate.py
tests/test_freeze_candidate.py
```

Checklist:

```text
[ ] Freeze checklist
[ ] Bloqueo si leakage fail
[ ] Bloqueo si manifest incompleto
[ ] Candidate freeze YAML
[ ] Status update
```

---

### Día/Bloque 9 — Agent tools read-only

Crear:

```text
src/ida_trading/agents/tools.py
src/ida_trading/agents/schemas.py
tests/test_agent_tools.py
```

Tools iniciales:

```text
list_runs
load_run_manifest
inspect_candidate
compare_runs
read_report
cost_sensitivity_summary
fold_stability_summary
ablation_summary
leakage_audit_summary
```

Checklist:

```text
[ ] Todas las tools devuelven EvidenceRefs
[ ] Ninguna escribe
[ ] Ninguna accede a holdout salvo petición explícita y marcada
[ ] Tests con fixtures
```

---

### Día/Bloque 10 — Chat UI simple

Crear:

```text
app/pages/08_Agent.py
app/services/agent_service.py
src/ida_trading/agents/orchestrator.py
src/ida_trading/agents/prompts.py
```

Checklist:

```text
[ ] Chat local en Streamlit
[ ] Modo Analyst
[ ] Modo Skeptic
[ ] Respuestas con evidencia
[ ] Sin Operator todavía
[ ] Fallback si no hay OPENAI_API_KEY
```

---

### Día/Bloque 11 — Planner sin ejecución

Crear:

```text
src/ida_trading/agents/proposals.py
tests/test_experiment_proposals.py
```

Checklist:

```text
[ ] propose_experiment
[ ] YAML proposal
[ ] forbidden_splits
[ ] success/failure criteria
[ ] Guardar en configs/proposals/
```

---

### Día/Bloque 12 — Documentación y DoD

Crear:

```text
docs/research_app.md
docs/agent_contract.md
docs/anti_overfitting_workflow.md
```

Checklist:

```text
[ ] Cómo indexar runs
[ ] Cómo ejecutar dashboard
[ ] Cómo registrar decisión
[ ] Cómo congelar candidato
[ ] Qué no hacer
```

---

## Dependencias iniciales

```bash
pip install streamlit duckdb pydantic
```

Cuando metas agente:

```bash
pip install openai
```

Comandos operativos:

```bash
python -m ida_trading.registry.indexer \
  --results results \
  --reports reports \
  --db results/ida_registry.sqlite

streamlit run app/ida_dashboard.py

pytest tests/test_manifest_schema.py \
       tests/test_registry_indexer.py \
       tests/test_run_compatibility.py \
       tests/test_agent_tools.py
```

## Cómo evitar bloquear la app por el agente

La app debe funcionar sin agente.

```text
[ ] Dashboard no depende de OPENAI_API_KEY
[ ] Agent page muestra “disabled” si no hay provider
[ ] Tools son funciones Python testeables sin LLM
[ ] Analyst/Skeptic usan mismos services que la UI
[ ] Decision logs se pueden crear manualmente
```

---

# 13. Definition of Done

## Primera versión útil del research dashboard

Está lista cuando:

```text
[ ] Indexa runs existentes
[ ] Cada run visible tiene manifest o warning claro
[ ] Candidate Explorer muestra costes, folds, baseline, leakage y ablation
[ ] Compare Runs detecta incompatibilidades
[ ] Reports markdown/figuras están ligados al run
[ ] La UI no ejecuta scripts
[ ] La lógica de trading no está en la UI
[ ] Hay tests para manifest/indexer/compatibility
[ ] Un candidato puede terminar en: reject / continue / freeze review
```

## Primera versión útil del agente

Está lista cuando:

```text
[ ] Solo modo read-only por defecto
[ ] Toda afirmación importante cita run_id/candidate_id/path
[ ] Puede decir “no hay evidencia suficiente”
[ ] Tiene Analyst y Skeptic
[ ] Usa tools estructuradas
[ ] Guarda mensajes relevantes
[ ] Puede crear decision log con aprobación humana
[ ] No ejecuta scripts
[ ] No modifica configs
[ ] No toca paper/live
```

## Pasar de research dashboard a paper monitor

Solo cuando:

```text
[ ] Existe candidate freeze record
[ ] La config está congelada
[ ] Dataset/code/config tienen hashes
[ ] Leakage audit PASS
[ ] Cost profile realista documentado
[ ] Execution assumptions definidas
[ ] Paper strategy manifest creado
[ ] Paper namespace separado
[ ] Success/failure criteria paper definidos antes de empezar
```

## Considerar que la app ayuda a decidir estrategias

La app aporta valor si:

```text
[ ] Reduce candidatos ambiguos
[ ] Aumenta decisiones con evidencia
[ ] Detecta fragilidad antes de holdout/paper
[ ] Reduce comparaciones inválidas
[ ] Registra por qué se rechazó cada candidato
[ ] Evita repetir experimentos equivalentes
[ ] Permite reconstruir cualquier decisión meses después
```

## Considerar una estrategia lista para paper trading

Checklist mínimo:

```text
[ ] Causal execution open_t+1 validada
[ ] No overnight validado
[ ] Leakage audit PASS
[ ] Net PnL positivo con realistic costs
[ ] Stress costs no destruyen completamente el perfil o el riesgo está documentado
[ ] Baseline delta positivo
[ ] Folds razonablemente estables
[ ] Drawdown tolerable
[ ] Trade count suficiente
[ ] Avg trade bps net > ruido operacional estimado
[ ] Ablation no revela dependencia sospechosa
[ ] HMM state dependency entendida
[ ] Risk limits definidos
[ ] Paper duration definida
[ ] Paper failure criteria definidos
[ ] Freeze record aprobado
```

---

# 14. Ejemplos concretos

## Ejemplo de pantalla: Candidate Explorer

```text
Candidate: C_20260504_001
Run: RUN_20260504_142233_a91f3
Stage: validation
Status: freeze_review

Summary:
  validation_net_pnl:      18,420
  validation_sharpe:       1.21
  max_drawdown:           -6,200
  trade_count:             912
  avg_trade_bps_net:       1.8
  break_even_cost_bps:     2.4
  realistic_cost_bps:      1.2
  positive_fold_ratio:     0.67
  baseline_delta_net:      7,300
  leakage_status:          PASS

Warnings:
  ⚠️ 44% of PnL comes from two folds.
  ⚠️ state_2 contributes 61% of net PnL with 18% occupancy.
```

---

## Ejemplo de query DuckDB: coste

```sql
SELECT
  candidate_id,
  cost_profile_id,
  total_cost_bps,
  net_pnl,
  sharpe,
  max_drawdown,
  trade_count
FROM read_parquet('results/runs/RUN_20260504_142233_a91f3/artifacts/cost_sensitivity.parquet')
WHERE candidate_id = 'C_20260504_001'
ORDER BY total_cost_bps;
```

## Ejemplo de query: folds

```sql
SELECT
  fold_id,
  split_role,
  net_pnl,
  sharpe,
  max_drawdown,
  trade_count,
  avg_trade_bps_net
FROM read_parquet('results/runs/RUN_20260504_142233_a91f3/artifacts/fold_metrics.parquet')
WHERE candidate_id = 'C_20260504_001'
ORDER BY fold_id;
```

## Ejemplo de query: dependencia HMM

```sql
SELECT
  hmm_state,
  occupancy_pct,
  signal_count,
  net_pnl,
  avg_trade_bps_net,
  sharpe
FROM read_parquet('results/runs/RUN_20260504_142233_a91f3/artifacts/state_economics.parquet')
WHERE candidate_id = 'C_20260504_001'
ORDER BY net_pnl DESC;
```

---

## Prompts útiles al agente

```text
Analyst: resume C_20260504_001 usando solo validation y realistic_cost_v3.
Incluye baseline delta, fold stability, cost sensitivity y leakage status.
```

```text
Skeptic: intenta rechazar C_20260504_001.
Busca dependencia de fold, coste, HMM state, feature ablation y leakage.
No uses holdout.
```

```text
Planner: propón un experimento mínimo para falsar si state_2 explica artificialmente el edge.
Debe usar validation only y no tocar configs congeladas.
```

```text
Operator: prepara un pending action para ejecutar ablation sobre C_20260504_001.
No ejecutes nada. Muestra comando, config draft, expected outputs y safeguards.
```

```text
Monitor: para paper_strategy_id STRAT_SPY_5M_001, explica la divergencia entre expected fills y simulated fills durante la última sesión.
No propongas cambios de estrategia.
```

---

## Ejemplo de decision log

```yaml
decision_id: DEC_20260504_001
created_at_utc: "2026-05-04T16:45:00Z"
owner: enrique
decision_type: freeze_review
candidate_id: C_20260504_001
run_id: RUN_20260504_142233_a91f3

decision: continue_validation
rationale: >
  El candidato tiene PnL neto positivo y mejora frente a baseline,
  pero la dependencia de HMM state_2 y la concentración en dos folds
  requieren falsación adicional antes de freeze.

evidence:
  - path: results/runs/RUN_20260504_142233_a91f3/artifacts/candidate_metrics.parquet
    metric: validation_net_pnl
    value: 18420.5
  - path: results/runs/RUN_20260504_142233_a91f3/artifacts/fold_metrics.parquet
    metric: pnl_top_2_folds_pct
    value: 44.0
  - path: results/runs/RUN_20260504_142233_a91f3/artifacts/state_economics.parquet
    metric: state_2_pnl_share_pct
    value: 61.0
  - path: reports/RUN_20260504_142233_a91f3/leakage_audit.md
    metric: leakage_status
    value: PASS

next_action: >
  Ejecutar ablation de state-conditioned filter y stress cost conservative_v2.
forbidden_actions:
  - use_holdout_for_selection
  - modify_frozen_configs
```

---

## Ejemplo de candidate freeze record

```yaml
freeze_id: FREEZE_20260504_001
candidate_id: C_20260504_001
candidate_family_id: FAM_HMM_REVERSAL_003
created_at_utc: "2026-05-04T18:10:00Z"
approved_by: enrique

frozen_inputs:
  run_id: RUN_20260504_142233_a91f3
  git_commit: 9f4a7c1d8b2
  git_dirty: false
  config_snapshot_path: results/runs/RUN_20260504_142233_a91f3/config_snapshot.yml
  config_hash: sha256:def456
  dataset_id: spy_5m_cleaned_aligned_v17
  dataset_hash: sha256:abc123
  feature_set_id: fs_intraday_hmm_v9
  model_id: xgb_spy_5m_v12
  signal_id: sig_hmm_reversal_v5

execution_spec:
  instrument: SPY
  timeframe: 5min
  decision_time: close_t
  fill_time: open_t_plus_1
  allow_overnight: false
  position_sizing: fixed_notional
  max_position: 1

costs:
  primary_cost_profile_id: realistic_spy_5m_v3
  stress_cost_profile_id: conservative_spy_5m_v2

validation_evidence:
  leakage_status: PASS
  validation_net_pnl: 18420.5
  positive_fold_ratio: 0.67
  baseline_delta_net: 7300.2
  break_even_cost_bps: 2.4

paper_plan:
  paper_strategy_id: STRAT_SPY_5M_001
  start_date: "2026-05-05"
  min_duration_trading_days: 60
  success_criteria:
    - no causality or execution mismatch
    - realized paper slippage <= conservative assumption
    - signal frequency within validation confidence band
    - drawdown within predefined limit
  failure_criteria:
    - execution mismatch unresolved
    - live/paper avg trade bps materially below validation
    - risk breach
    - missing data incidents repeated

status: frozen_for_paper
```

---

## Ejemplo de análisis del agente para candidato frágil

```text
Verdict:
No congelaría C_20260504_017. La evidencia sugiere fragilidad a costes y concentración temporal.

Evidencia:
1. Costes:
   - realistic_cost_v3: net_pnl = 9,800
   - conservative_cost_v2: net_pnl = -1,250
   - break_even_cost_bps = 1.35
   - realistic_cost_bps = 1.20
   Fuente: results/runs/RUN_20260504_150011_f31ac/artifacts/cost_sensitivity.parquet

2. Folds:
   - 1 de 10 folds explica 57% del PnL.
   - positive_fold_ratio = 0.50.
   Fuente: results/runs/RUN_20260504_150011_f31ac/artifacts/fold_metrics.parquet

3. Ablation:
   - Al eliminar feature family `gap_reversal_open`, net_pnl pasa de 9,800 a -430.
   Fuente: results/runs/RUN_20260504_150011_f31ac/artifacts/ablation_summary.parquet

4. Leakage:
   - leakage_status = WARNING por `feature_lag_consistency`.
   Fuente: reports/RUN_20260504_150011_f31ac/leakage_audit.md

Limitaciones:
No he consultado holdout/test. Esta revisión usa validation y artifacts de robustness.

Conclusión operativa:
Rechazar para freeze. Próximo experimento razonable: rehacer feature lag audit y repetir ablation de `gap_reversal_open` con coste conservative, validation only.
```

---

# Recomendación final

Construye la app, pero con esta prioridad:

```text
1. Manifest + registry
2. Read-only dashboard
3. Candidate diagnostics
4. Compatibility-aware comparisons
5. Decision logs + freeze workflow
6. Agente Analyst/Skeptic read-only
7. Planner
8. Operator con aprobación
9. Paper monitor
10. Live monitor
```

La primera versión no debe sentirse como una plataforma. Debe sentirse como un **laboratorio disciplinado que te impide engañarte**. Esa es precisamente la parte que más puede ayudarte a encontrar una estrategia con PnL neto positivo después de costes realistas.

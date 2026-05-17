# IDA Research App (legacy / headless)

Esta carpeta queda como infraestructura legacy de indexado y lectura de
artefactos. No es la app principal del proyecto.

El research operativo de IDA Trading se hace por codigo: scripts, configs,
tests, reports y manifests. La UI no debe sustituir ese flujo ni intentar
convertirse en un IDE de investigacion.

La app principal para gestion visual es ahora `src/candidate_app/`, enfocada
solo en candidatos que ya han salido de research y pueden estar en revision o
paper trading.

## Papel actual

`research_app` puede seguir siendo util para tareas headless:

- indexar metadata local de `results/` y `reports/`;
- leer parquet, markdown y YAML;
- mantener compatibilidad con tests y tooling existente;
- consultar artefactos legacy cuando haga falta depurar historico.

No se debe anadir nueva funcionalidad de producto aqui salvo que sea
estrictamente necesaria para compatibilidad.

## Indexado legacy

```bash
python -m src.research_app.registry --results results --reports reports --db results/ida_registry.sqlite --reset
```

El indexador crea:

- `runs`: grupos legacy inferidos por target/timeframe/experimento.
- `artifacts`: parquet de `results/`.
- `reports`: markdown, figuras y parquet de `reports/`.
- `candidates`: filas resumen desde `candidate_registry`, `decisions`, `triage`, `selected_specs` y `selected_validation`.
- `decision_logs`: decisiones humanas con evidencia obligatoria.

Los artefactos legacy se marcan con warning porque todavia no tienen manifest completo.

## UI legacy

```bash
streamlit run app/ida_dashboard.py
```

Esta UI queda disponible solo como visor antiguo. No es el dashboard recomendado
para el trabajo diario.

La UI oficial para candidatos y paper trading es:

```bash
uvicorn src.candidate_app.api:app --host 127.0.0.1 --port 8700
```

## Backend legacy

`src/research_app/api.py` se conserva para compatibilidad:

```bash
uvicorn src.research_app.api:app --host 127.0.0.1 --port 8600
```

Endpoints iniciales:

- `GET /health`: registry summary y estado read-only del daemon H1c.
- `GET /registry/summary`: contadores del SQLite.
- `POST /registry/index`: reindexa metadata local de `results/` y `reports/`.
- `GET /registry/snapshot`: paquete completo para una UI custom.
- `GET /runs`, `/candidates`, `/artifacts`, `/reports`.
- `GET /reports/markdown?path=...`: preview markdown dentro del workspace.
- `GET /artifacts/parquet-preview?path=...`: preview parquet dentro del workspace.
- `GET /operations/daemon-status`: lee `daemon_status.yaml`, sin mutar nada.
- `GET /operations/h1c`: vista read-only de estado, auto-runs, ordenes
  planificadas/enviadas, eventos de PnL, eventos de estado y series para plots
  de precio QQQ 15m / PnL realizado.
- `POST /decisions`: registra una decision humana con evidencia.

## Frontera

Permitido:

- leer parquet, markdown y YAML;
- indexar metadata local;
- registrar decisiones;
- preparar trazabilidad para manifests futuros.

No permitido en esta fase:

- convertir esta app en el flujo principal de research;
- modificar configs desde la UI;
- lanzar backtests desde widgets;
- seleccionar por test/holdout;
- meter logica de trading dentro de callbacks;
- ejecutar ordenes.

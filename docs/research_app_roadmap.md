# IDA Research App Roadmap (archived)

Este roadmap queda archivado. La direccion actual es no construir una UI de
research: el research se ejecuta y se revisa por codigo.

La app viva del proyecto es `src/candidate_app/`, que empieza despues de la fase
de research y gestiona candidatos, paper trading y ledger de resultados.

## Principios

- La app es un plano de observabilidad y decision humana, no un motor de trading.
- El backend expone contratos read-only por defecto; las unicas escrituras
  permitidas son metadata local de registry y decision logs con evidencia.
- Ninguna pantalla debe modificar configs, lanzar runners, enviar/cancelar
  ordenes ni mutar `results/paper/h1c_state/state.yaml`.
- El automatismo H1c sigue siendo propietario de execution, reconciliation,
  order planning, state transitions y guardrails.

## Estado actual

- [x] Streamlit Research Lab en `app/ida_dashboard.py`.
- [x] Registry SQLite local en `results/ida_registry.sqlite`.
- [x] Decision logs con evidencia obligatoria.
- [x] Capa de servicio reusable en `src/research_app/service.py`.
- [x] Backend HTTP inicial en `src/research_app/api.py`.
- [x] UI custom inicial servida por FastAPI en `/`.
- [x] Endpoint read-only de estado del daemon:
  `/operations/daemon-status`.
- [x] Endpoint operacional H1c:
  `/operations/h1c`.
- [x] Vista principal orientada a ordenes enviadas, PnL y estado paper.
- [x] Plots de precio QQQ 15m y PnL realizado acumulado.
- [x] Tests unitarios para registry, manifests, decisiones, servicio y rutas API.

## Fase 1 - Contrato backend estable

- [x] Extraer acceso a registry/parquet/markdown fuera de Streamlit.
- [x] Exponer endpoints para summary, runs, candidates, reports, artifacts,
  parquet preview y markdown preview.
- [x] Mantener writes limitadas a `POST /registry/index` y `POST /decisions`.
- [ ] Anadir tests HTTP con `TestClient` cuando `httpx` este disponible en la
  `.venv`.
- [ ] Versionar schemas de respuesta para candidates, reports y daemon status.
- [ ] Anadir paginacion real por cursor o offset en tablas grandes.
- [ ] Anadir endpoint de manifests con prioridad sobre heuristicas legacy.

## Fase 2 - Frontend custom legacy

- [x] Crear primera UI custom sin build step en `src/research_app/web/`.
- [x] Decision actual: no seguir invirtiendo en una UI de research.
- [x] Implementar layout operacional inicial: sidebar, top status bar y tabs.
- [ ] Rehacer Decision Board consumiendo `/candidates`.
- [ ] Rehacer Run Browser consumiendo `/runs`, `/artifacts` y `/reports`.
- [ ] Rehacer Report Viewer consumiendo `/reports/markdown`.
- [ ] Rehacer Candidate Explorer con parquet preview bajo demanda.
- [ ] Implementar Decision Log con formulario y validacion de evidencia.
- [x] Sustituir la orientacion de producto por `candidate_app`.

## Fase 3 - Observabilidad paper

La observabilidad paper se mueve a `candidate_app`, asociada a candidatos
activos y ledger por estrategia. `research_app` no debe ser propietario de esta
superficie.

## Fase 4 - Hardening

- [ ] Configurar auth local basica antes de exponer fuera de localhost.
- [ ] Anadir auditoria para cualquier write de decision/index.
- [ ] Separar DB de registry de artefactos generados por trading.
- [ ] Definir permisos por rol: research, observer, operator.
- [ ] Anadir pruebas Playwright del frontend.
- [ ] Empaquetar launch local separado del daemon H1c.

## Comandos

Indexar registry:

```bash
python -m src.research_app.registry --results results --reports reports --db results/ida_registry.sqlite --reset
```

Arrancar Streamlit legacy:

```bash
streamlit run app/ida_dashboard.py
```

Arrancar API legacy:

```bash
uvicorn src.research_app.api:app --host 127.0.0.1 --port 8600
```

Arrancar app oficial de candidatos:

```bash
uvicorn src.candidate_app.api:app --host 127.0.0.1 --port 8700
```

Instalar dependencias si falta FastAPI:

```bash
pip install -r requirements.txt
```

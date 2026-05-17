# IDA Paper/Live Control Center

App oficial para observar y controlar estrategias candidatas que ya llegaron a
paper/live.

Research sigue viviendo en codigo. Esta app empieza despues: cuando una
hipotesis ya paso gates, fue promovida a candidato operable y esta trabajando
en paper o live.

## Arranque

```bash
uvicorn src.candidate_app.api:app --host 127.0.0.1 --port 8700
```

La UI queda en:

```text
http://127.0.0.1:8700/
```

## Alcance actual

Permitido:

- ver solo candidatos paper/live activos;
- separar vista `Paper`, vista `Live` y vista `Conexion`;
- leer estado real de `results/paper/...`;
- ver daemon, scheduler, ultimo run y eventos de estado;
- ver PnL historico si existe `pnl_events.parquet`;
- ver ledger operativo por candidato, desde artefactos reales o ledger local;
- ver precio del activo operado con marcas `buy`, `sell` y `hold`;
- mostrar alertas operativas por candidato;
- activar/desactivar kill switch local;
- guardar politica operativa de capital por estrategia;
- usar la UI como ventana de control, no como research dashboard.

No permitido:

- ejecutar trading real;
- lanzar backtests;
- editar hipotesis de research;
- modificar configs de estrategia;
- modificar configs del runner sin una accion explicita del operador;
- promocionar candidatos desde la UI;
- comparar hipotesis/candidatos pre-paper;
- mezclar decisiones de research con estado operativo paper/live.

## Storage local

Por defecto usa:

```text
results/candidate_app/candidates.sqlite
```

La base local se genera con seed data si no existen candidatos. No se versiona.

## Paper ledger

La fuente preferida del ledger operativo son los artefactos reales del runner:

```text
results/paper/h1c_auto_runner/
results/paper/h1c_state/state.yaml
results/paper/h1c_state/events.parquet
results/paper/h1c_state/pnl_events.parquet
```

La tabla local `candidate_paper_ledger` queda disponible para eventos manuales o
integraciones futuras. Si no existe `pnl_events.parquet`, la pantalla principal
usa esta tabla como fallback para la curva de PnL y el ledger.

Cada fila del ledger manual representa un evento paper: senal, orden planeada,
orden enviada, fill, marca de mercado, fee, ajuste o nota.

Campos principales:

- `candidate_id`
- `event_at`
- `event_type`
- `strategy_run_id`
- `symbol`
- `side`
- `quantity`
- `price`
- `gross_pnl`
- `fees`
- `slippage_bps`
- `net_pnl`
- `exposure`
- `notes`
- `metadata`

La app calcula por candidato:

- PnL neto acumulado;
- PnL bruto;
- fees;
- win rate sobre eventos con PnL;
- media por evento;
- drawdown de PnL acumulado;
- slippage medio;
- ultimo evento;
- curva de PnL acumulado.

La app incluye un candidato demo `KO Defensive Mean Reversion` para validar la
experiencia visual: precio KO simulado, marcas de compra/venta/hold, ledger y
PnL acumulado. Es un ejemplo local, no una fuente de trading real.

## Control operativo

Cada fuente operativa puede declarar:

- `mode`: `paper` o `live`;
- `kill_switch_path`;
- estado de daemon;
- estado paper/live;
- ruta de eventos;
- ruta de PnL;
- config del runner;
- config de conexion.

En la UI se puede guardar:

- encendida/apagada;
- capital como fraccion del neto/buying power;
- capital absoluto en USD como cap de notional por orden;
- si la politica debe quedarse solo en la DB local o aplicarse al config del
  runner.

Nota: el runner H1C actual soporta fracciones sobre buying power/funds y cap de
notional. Un importe absoluto fijo por estrategia requiere ampliar la logica de
sizing del runner.

## Conexion / VPN

La vista `Conexion` ejecuta checks ligeros:

- reachability TCP del gateway configurado;
- modo de trading esperado;
- cuenta esperada;
- existencia y frescura del status file del daemon.

## Endpoints principales

- `GET /control-center`
- `GET /control-center/connections`
- `GET /control-center/{candidate_id}`
- `POST /control-center/{candidate_id}/control`
- `PATCH /control-center/{candidate_id}/runtime`
- `GET /candidates`
- `POST /candidates`
- `GET /candidates/paper-trading`
- `GET /candidates/{candidate_id}`
- `PATCH /candidates/{candidate_id}/status`
- `GET /compare`
- `GET /paper-ledger`
- `POST /paper-ledger`
- `GET /paper-ledger/summary`

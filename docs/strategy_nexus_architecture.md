# Strategy Nexus Architecture

Fecha: 2026-05-16

Este documento fija el rumbo de IDA Trading como nexo de estrategias. No cambia
la implementacion actual de H1c ni H3; solo define el marco al que migrar nuevas
verticales y futuros contratos.

## Decision

IDA Trading deja de definirse como un proyecto intradia y pasa a definirse como
un research stack multi-estrategia:

```text
hypothesis -> data contract -> research candidate -> strategy contract
  -> validation -> promotion gates -> paper/live -> monitoring -> portfolio
```

La estrategia intradia sigue siendo una familia de primera clase, pero no es el
molde universal. H1c queda como vertical intradia operable. H3 queda como
vertical event-driven en research. Las nuevas lineas deben declarar su familia,
horizonte, universo, datos y reglas de promocion sin heredar supuestos intradia
si no aplican.

## Strategy Families

Cada estrategia nueva debe clasificarse al menos por:

- `strategy_family`: intraday, event_driven, swing, macro, relative_value,
  options_probe u otra familia explicita.
- `asset_class`: equities, etf, futures, options, fx, rates, credit o multi_asset.
- `holding_period`: intraday, overnight, multi_day, weekly, monthly.
- `can_hold_overnight`: booleano explicito.
- `universe_id`: universo versionado usado en research.
- `data_contract_id`: contrato de datos point-in-time necesario.
- `execution_mode`: research_only, paper, live_read_only, live_orders.
- `portfolio_bucket`: bucket de capital/riesgo si llega a operable.

## Contract Direction

El contrato actual de estrategia sirve para estrategias intradia de barras con
entrada `next_open`, horizonte en barras y reglas de flat intradia. No debe ser
forzado sobre estrategias que:

- tienen eventos externos como earnings, macro o analyst revisions;
- pueden salir en T+1 o en varias sesiones;
- requieren hedge, basket, pair/spread o portfolio construction;
- dependen de fuentes point-in-time externas con snapshots;
- tienen riesgo overnight o gap risk.

La evolucion recomendada es introducir un contrato general, por ejemplo
`StrategyContract`, y dejar el contrato actual como especializacion intradia.
Eso evita romper H1c/H3 y permite que cada familia declare su propio entry/exit
sin simular que todo es `horizon_bars`.

## Promotion Model

Todas las familias comparten la misma filosofia:

- seleccion en train/validation, confirmacion en test;
- datos point-in-time y reglas causales;
- costes, slippage y sensibilidad a ejecucion;
- controles economicos especificos de la hipotesis;
- concentracion por fecha, ticker, sector, regimen y evento;
- decision registrada antes de paper/live;
- degradacion monitorizada contra expectativa congelada.

Las gates concretas no tienen que ser identicas. Una estrategia intradia puede
fallar por slippage de minutos; una event-driven puede fallar por revision de
consenso, survivorship bias, halts o colas overnight.

## Repository Boundaries

Direccion recomendada para nuevas verticales:

```text
configs/strategy/       contratos de estrategia y familias
configs/data/           contratos de datos externos
src/data/               ingesta y auditoria point-in-time
src/strategy/           logica de estrategia por familia
src/backtesting/        metricas comunes y simulacion reutilizable
src/research/           manifests, splits, promotion gates, decisions
src/execution/          paper/live por estrategia operable
src/candidate_app/      gestion de candidatos, paper ledger y observabilidad pre-live
src/research_app/       indexador/visor legacy de artefactos de research
```

No se debe mezclar el motor de una familia con otra salvo en utilidades comunes
claramente neutrales: calendario, costes, splits, manifests, metricas y registry.

## Near-Term Migration

Orden recomendado:

1. Mantener H1c y H3 congeladas funcionalmente.
2. Crear nuevos metadatos de clasificacion solo para nuevas estrategias.
3. Disenar `StrategyContract` general sin sustituir el spec intradia actual.
4. Promocionar solo estrategias con evidencia congelada a `candidate_app`.
5. Migrar verticales antiguas solo cuando haya necesidad real, no como refactor
   preventivo.

La regla practica: ninguna migracion debe cambiar resultados historicos de una
estrategia existente salvo que ese cambio sea el objetivo explicito del trabajo.

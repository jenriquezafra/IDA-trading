# Anti-Overfitting Workflow

## Reglas de uso de la app

1. Cada decision debe citar evidencia: path, run_id y candidate_id cuando existan.
2. No seleccionar candidatos por orden de Sharpe maximo.
3. Validation sirve para seleccionar; test/holdout sirven para confirmar.
4. Un candidato con leakage `fail` no puede congelarse.
5. Un candidato que solo sobrevive a un coste irreal queda marcado como research-only o rejected.
6. Comparaciones entre runs incompatibles deben tratarse como exploratorias.
7. Cada freeze posterior debe incluir config, git commit, dataset, coste y split policy.

## Decision log minimo

```text
decision_type: reject | keep_in_research | freeze_draft | paper_candidate | note
decision: texto corto
candidate_id: opcional
run_id: opcional
evidence:
  - path: results/.../candidate_decisions.parquet
rationale: por que se acepta/rechaza
next_action: siguiente experimento o cierre
```

## Criterios de rechazo por defecto

- no mejora baseline;
- falla costes conservadores;
- PnL concentrado en pocos dias/meses/folds;
- baja muestra de trades;
- HMM inestable o no interpretable;
- resultado depende de mirar test;
- no hay reporte de leakage limpio.

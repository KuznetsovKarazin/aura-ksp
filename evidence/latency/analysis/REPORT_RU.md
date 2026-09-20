# AURA-KSP: результат GPU latency microbenchmark

Этот этап измеряет только стоимость существующих K64 checkpoints. Обучение и оценка качества не выполнялись.

Итоговый latency gate: **PASS**.

| Cache regime | Full-4, ms | Stream-3, ms | Saving | LCB | Требование | Результат |
|---|---:|---:|---:|---:|---:|---|
| warm_reuse | 65.2755 | 48.9959 | 24.94% | 24.88% | 20.00% | PASS |
| fresh_mapping | 68.5097 | 51.8634 | 24.30% | 24.22% | 20.00% | PASS |

PASS разрешает заморозить отдельный confirmatory quality protocol. Он не является подтверждением качества сам по себе.
Независимое подтверждение качества требует нового seed и девяти обучений: 3 folds × 3 retained streams.

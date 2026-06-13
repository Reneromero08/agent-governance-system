# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {0: 115, 1: 125}
- mean delta ns: 241045.554

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `delta_abs_ns_threshold` | 0.602679 | 0.575083 | 0.027595 |
| `mode_majority` | 0.578125 | 0.588988 | -0.010863 |
| `elapsed_a_ns_threshold` | 0.555804 | 0.600446 | -0.044643 |
| `sum_ns_threshold` | 0.508929 | 0.568571 | -0.059643 |
| `carrier_low_majority` | 0.5 | 0.567297 | -0.067297 |
| `elapsed_b_ns_threshold` | 0.497768 | 0.577864 | -0.080096 |
| `offset_iters_majority` | 0.433036 | 0.585714 | -0.152679 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

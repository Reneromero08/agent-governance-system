# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {1: 123, 0: 117}
- mean delta ns: 173821.633

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `mode_majority` | 0.555617 | 0.611235 | -0.055617 |
| `carrier_low_majority` | 0.499444 | 0.569024 | -0.06958 |
| `sum_ns_threshold` | 0.535039 | 0.614478 | -0.079439 |
| `offset_iters_majority` | 0.474972 | 0.577143 | -0.102171 |
| `elapsed_b_ns_threshold` | 0.451613 | 0.580357 | -0.128744 |
| `elapsed_a_ns_threshold` | 0.456062 | 0.611235 | -0.155172 |
| `delta_abs_ns_threshold` | 0.441046 | 0.60178 | -0.160734 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

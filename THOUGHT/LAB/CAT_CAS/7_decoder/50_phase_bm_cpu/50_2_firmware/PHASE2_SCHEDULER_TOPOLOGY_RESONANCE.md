# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_CANDIDATE`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {1: 126, 0: 114}
- mean delta ns: 139694.8

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `sum_ns_threshold` | 0.633333 | 0.583333 | 0.05 |
| `elapsed_b_ns_threshold` | 0.633333 | 0.598214 | 0.035119 |
| `elapsed_a_ns_threshold` | 0.583333 | 0.609428 | -0.026094 |
| `delta_abs_ns_threshold` | 0.55 | 0.588571 | -0.038571 |
| `carrier_low_majority` | 0.5 | 0.548943 | -0.048943 |
| `mode_majority` | 0.5 | 0.591429 | -0.091429 |
| `offset_iters_majority` | 0.4 | 0.585714 | -0.185714 |

## Interpretation

At least one topology feature beat the shuffled-answer null. Rerun on fresh seed windows before promotion.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

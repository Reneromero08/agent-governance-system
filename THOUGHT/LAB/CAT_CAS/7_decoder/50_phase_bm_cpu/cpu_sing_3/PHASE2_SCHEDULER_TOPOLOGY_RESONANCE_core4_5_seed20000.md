# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_CANDIDATE`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {1: 126, 0: 114}
- mean delta ns: 238650.542

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `sum_ns_threshold` | 0.666667 | 0.566667 | 0.1 |
| `delta_abs_ns_threshold` | 0.633333 | 0.566667 | 0.066667 |
| `elapsed_a_ns_threshold` | 0.6 | 0.568409 | 0.031591 |
| `elapsed_b_ns_threshold` | 0.6 | 0.571429 | 0.028571 |
| `mode_majority` | 0.5 | 0.591429 | -0.091429 |
| `carrier_low_majority` | 0.45 | 0.553167 | -0.103167 |
| `offset_iters_majority` | 0.4 | 0.585714 | -0.185714 |

## Interpretation

At least one topology feature beat the shuffled-answer null. Rerun on fresh seed windows before promotion.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

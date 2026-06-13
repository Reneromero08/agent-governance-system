# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {0: 116, 1: 124}
- mean delta ns: 146200.317

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `elapsed_b_ns_threshold` | 0.555061 | 0.600667 | -0.045606 |
| `carrier_low_majority` | 0.516685 | 0.566964 | -0.050279 |
| `mode_majority` | 0.5 | 0.578125 | -0.078125 |
| `elapsed_a_ns_threshold` | 0.506674 | 0.588988 | -0.082314 |
| `sum_ns_threshold` | 0.506674 | 0.588988 | -0.082314 |
| `offset_iters_majority` | 0.483315 | 0.577143 | -0.093828 |
| `delta_abs_ns_threshold` | 0.468854 | 0.571429 | -0.102574 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

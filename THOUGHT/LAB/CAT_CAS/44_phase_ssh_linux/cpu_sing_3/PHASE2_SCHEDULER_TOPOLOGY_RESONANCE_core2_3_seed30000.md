# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {0: 115, 1: 125}
- mean delta ns: 1021.992

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `elapsed_a_ns_threshold` | 0.569196 | 0.578125 | -0.008929 |
| `mode_majority` | 0.578125 | 0.588988 | -0.010863 |
| `sum_ns_threshold` | 0.533482 | 0.590278 | -0.056796 |
| `carrier_low_majority` | 0.5 | 0.566392 | -0.066392 |
| `elapsed_b_ns_threshold` | 0.53125 | 0.598443 | -0.067193 |
| `delta_abs_ns_threshold` | 0.462054 | 0.589544 | -0.12749 |
| `offset_iters_majority` | 0.433036 | 0.585714 | -0.152679 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

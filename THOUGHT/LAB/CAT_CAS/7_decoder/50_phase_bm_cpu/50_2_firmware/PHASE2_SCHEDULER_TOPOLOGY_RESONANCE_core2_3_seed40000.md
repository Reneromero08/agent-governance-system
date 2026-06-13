# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {1: 123, 0: 117}
- mean delta ns: 28685.496

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `elapsed_b_ns_threshold` | 0.595106 | 0.584175 | 0.010931 |
| `carrier_low_majority` | 0.532814 | 0.566185 | -0.03337 |
| `mode_majority` | 0.555617 | 0.611235 | -0.055617 |
| `delta_abs_ns_threshold` | 0.55673 | 0.620536 | -0.063806 |
| `sum_ns_threshold` | 0.489433 | 0.581201 | -0.091769 |
| `elapsed_a_ns_threshold` | 0.517241 | 0.615684 | -0.098443 |
| `offset_iters_majority` | 0.474972 | 0.577143 | -0.102171 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

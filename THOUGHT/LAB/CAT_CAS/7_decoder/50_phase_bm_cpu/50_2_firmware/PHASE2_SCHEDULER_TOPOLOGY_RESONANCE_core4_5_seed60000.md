# PHASE2_SCHEDULER_TOPOLOGY_RESONANCE

## Verdict

`SCHEDULER_TOPOLOGY_RESONANCE_NOT_CONFIRMED`

Core-pair phase-offset topology probe with shuffled-answer nulls.

## Summary

- rows: 240
- restore failures: 0
- answer balance: {1: 110, 0: 130}
- mean delta ns: 151986.013

| Feature | bAcc | Shuffle p95 | Margin |
|---|---:|---:|---:|
| `carrier_low_majority` | 0.5 | 0.538721 | -0.038721 |
| `mode_majority` | 0.5 | 0.544643 | -0.044643 |
| `elapsed_b_ns_threshold` | 0.528365 | 0.574163 | -0.045798 |
| `delta_abs_ns_threshold` | 0.524472 | 0.584821 | -0.06035 |
| `offset_iters_majority` | 0.491657 | 0.566964 | -0.075307 |
| `sum_ns_threshold` | 0.493882 | 0.583333 | -0.089451 |
| `elapsed_a_ns_threshold` | 0.467186 | 0.593891 | -0.126706 |

## Interpretation

No topology feature beat the shuffled-answer null with the required margin.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

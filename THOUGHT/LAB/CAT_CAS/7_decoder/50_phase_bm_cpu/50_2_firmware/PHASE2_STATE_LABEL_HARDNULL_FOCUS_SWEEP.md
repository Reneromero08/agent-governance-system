# PHASE2_STATE_LABEL_HARDNULL_FOCUS_SWEEP

## Verdict

`STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET`

Compact read-only sweep over seed windows and row durations.

## Summary

- runs: 3
- candidate runs: 0
- acceptance: at least 3 candidate runs with zero restore failures

| File | Verdict | Seed | Rounds | Rows | Restore failures | Elapsed bAcc | Shuffle p95 | Shuffle margin | State bAcc | Mode bAcc | Core bAcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PHASE2_STATE_LABEL_HARDNULL_FOCUS_seed5750_r4096_t16.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 5750 | 4096 | 256 | 0 | 0.543651 | 0.608333 | -0.064682 | 0.5 | 0.436508 | 0.436508 |
| `PHASE2_STATE_LABEL_HARDNULL_FOCUS_seed6000_r4096_t16.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 6000 | 4096 | 256 | 0 | 0.556207 | 0.59433 | -0.038123 | 0.501955 | 0.48436 | 0.51564 |
| `PHASE2_STATE_LABEL_HARDNULL_FOCUS_seed6250_r4096_t16.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 6250 | 4096 | 256 | 0 | 0.668627 | 0.593137 | 0.07549 | 0.5 | 0.609804 | 0.515686 |

## Interpretation

The timing edge remains live but unstable. It should not be counted as CPU-sings evidence yet.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

# PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP

## Verdict

`STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET`

Compact read-only sweep over seed windows and row durations.

## Summary

- runs: 6
- candidate runs: 2
- acceptance: at least 3 candidate runs with zero restore failures

| File | Verdict | Seed | Rounds | Rows | Restore failures | Elapsed bAcc | State bAcc | Mode bAcc | Core bAcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PHASE2_STATE_LABEL_SWEEP_seed1000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 1000 | 4096 | 128 | 0 | 0.61336 | 0.5 | 0.483806 | 0.5 |
| `PHASE2_STATE_LABEL_SWEEP_seed3000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 3000 | 4096 | 128 | 0 | 0.491667 | 0.5 | 0.566667 | 0.566667 |
| `PHASE2_STATE_LABEL_SWEEP_seed5000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 5000 | 4096 | 128 | 0 | 0.666667 | 0.5 | 0.5 | 0.5 |
| `PHASE2_STATE_LABEL_SWEEP_seed7000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 7000 | 4096 | 128 | 0 | 0.481781 | 0.5 | 0.467611 | 0.61336 |
| `PHASE2_STATE_LABEL_SWEEP_seed1000_r8192.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 1000 | 8192 | 128 | 0 | 0.548583 | 0.5 | 0.483806 | 0.5 |
| `PHASE2_STATE_LABEL_SWEEP_seed5000_r8192.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 5000 | 8192 | 128 | 0 | 0.382353 | 0.5 | 0.5 | 0.5 |

## Interpretation

The timing edge remains live but unstable. It should not be counted as CPU-sings evidence yet.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

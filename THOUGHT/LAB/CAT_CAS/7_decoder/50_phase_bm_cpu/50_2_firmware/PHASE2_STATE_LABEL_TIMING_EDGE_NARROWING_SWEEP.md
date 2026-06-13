# PHASE2_STATE_LABEL_TIMING_EDGE_NARROWING_SWEEP

## Verdict

`STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET`

Compact read-only sweep over seed windows and row durations.

## Summary

- runs: 16
- candidate runs: 2
- acceptance: at least 3 candidate runs with zero restore failures

| File | Verdict | Seed | Rounds | Rows | Restore failures | Elapsed bAcc | Shuffle p95 | Shuffle margin | State bAcc | Mode bAcc | Core bAcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PHASE2_STATE_LABEL_NARROW_seed1000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 1000 | 4096 | 128 | 0 | 0.510121 |  |  | 0.5 | 0.483806 | 0.5 |
| `PHASE2_STATE_LABEL_NARROW_seed1500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 1500 | 4096 | 128 | 0 | 0.59375 |  |  | 0.5 | 0.5625 | 0.5 |
| `PHASE2_STATE_LABEL_NARROW_seed2000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 2000 | 4096 | 128 | 0 | 0.531746 |  |  | 0.5 | 0.436508 | 0.468254 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 2500 | 4096 | 128 | 0 | 0.67451 |  |  | 0.5 | 0.531373 | 0.578431 |
| `PHASE2_STATE_LABEL_NARROW_seed3000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 3000 | 4096 | 128 | 0 | 0.516667 |  |  | 0.5 | 0.566667 | 0.566667 |
| `PHASE2_STATE_LABEL_NARROW_seed3500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 3500 | 4096 | 128 | 0 | 0.609804 |  |  | 0.5 | 0.468627 | 0.515686 |
| `PHASE2_STATE_LABEL_NARROW_seed4000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 4000 | 4096 | 128 | 0 | 0.587302 |  |  | 0.5 | 0.5 | 0.595238 |
| `PHASE2_STATE_LABEL_NARROW_seed4500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 4500 | 4096 | 128 | 0 | 0.523529 |  |  | 0.5 | 0.468627 | 0.452941 |
| `PHASE2_STATE_LABEL_NARROW_seed5000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 5000 | 4096 | 128 | 0 | 0.637255 |  |  | 0.5 | 0.5 | 0.5 |
| `PHASE2_STATE_LABEL_NARROW_seed5500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 5500 | 4096 | 128 | 0 | 0.52381 |  |  | 0.5 | 0.5 | 0.5 |
| `PHASE2_STATE_LABEL_NARROW_seed6000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 6000 | 4096 | 128 | 0 | 0.698039 |  |  | 0.5 | 0.468627 | 0.468627 |
| `PHASE2_STATE_LABEL_NARROW_seed6500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 6500 | 4096 | 128 | 0 | 0.559524 |  |  | 0.5 | 0.563492 | 0.5 |
| `PHASE2_STATE_LABEL_NARROW_seed7000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 7000 | 4096 | 128 | 0 | 0.518219 |  |  | 0.5 | 0.467611 | 0.61336 |
| `PHASE2_STATE_LABEL_NARROW_seed7500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 7500 | 4096 | 128 | 0 | 0.398039 |  |  | 0.5 | 0.390196 | 0.452941 |
| `PHASE2_STATE_LABEL_NARROW_seed8000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 8000 | 4096 | 128 | 0 | 0.417749 |  |  | 0.5 | 0.448052 | 0.58658 |
| `PHASE2_STATE_LABEL_NARROW_seed8500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 8500 | 4096 | 128 | 0 | 0.476471 |  |  | 0.5 | 0.421569 | 0.343137 |

## Interpretation

The timing edge remains live but unstable. It should not be counted as CPU-sings evidence yet.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

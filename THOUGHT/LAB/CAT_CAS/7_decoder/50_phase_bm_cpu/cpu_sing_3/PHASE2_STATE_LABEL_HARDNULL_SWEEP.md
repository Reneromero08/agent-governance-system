# PHASE2_STATE_LABEL_HARDNULL_SWEEP

## Verdict

`STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET`

Compact read-only sweep over seed windows and row durations.

## Summary

- runs: 5
- candidate runs: 1
- acceptance: at least 3 candidate runs with zero restore failures

| File | Verdict | Seed | Rounds | Rows | Restore failures | Elapsed bAcc | Shuffle p95 | Shuffle margin | State bAcc | Mode bAcc | Core bAcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PHASE2_STATE_LABEL_HARDNULL_seed4500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 4500 | 4096 | 128 | 0 | 0.515686 | 0.601732 | -0.086046 | 0.558824 | 0.468627 | 0.452941 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 5000 | 4096 | 128 | 0 | 0.666667 | 0.622727 | 0.04394 | 0.5 | 0.5 | 0.5 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 5500 | 4096 | 128 | 0 | 0.587302 | 0.658333 | -0.071031 | 0.5 | 0.5 | 0.5 |
| `PHASE2_STATE_LABEL_HARDNULL_seed6000_r4096.json` | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 6000 | 4096 | 128 | 0 | 0.660784 | 0.598039 | 0.062745 | 0.5 | 0.468627 | 0.468627 |
| `PHASE2_STATE_LABEL_HARDNULL_seed6500_r4096.json` | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 6500 | 4096 | 128 | 0 | 0.47619 | 0.65 | -0.17381 | 0.5 | 0.563492 | 0.5 |

## Interpretation

The timing edge remains live but unstable. It should not be counted as CPU-sings evidence yet.

## Boundary

- No platform setting changes.
- No candidate image construction.
- No external instrumentation.

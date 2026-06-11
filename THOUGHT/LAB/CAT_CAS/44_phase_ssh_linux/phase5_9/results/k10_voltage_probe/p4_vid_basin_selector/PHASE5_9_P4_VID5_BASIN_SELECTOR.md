# Phase 5.9 P4 VID+5 Basin Selector Probe

Verdict: `BASIN_SELECTOR_FOUND_SYSCALL_HIGH_BIAS`

Objective: hold P4 VID+5 / decoded 1.1625V and test whether preconditions bias collapsed, mid, or high carrier basin selection.

Basin bins:

- collapsed: thickness < 100
- mid: 100 <= thickness < 5000
- high: thickness >= 5000

All 24 runs completed with 0 restoration failures.

## Selector Summary

| Selector | n | Collapsed | Mid | High | Mean thickness | Max thickness | Mean CV |
|----------|---|-----------|-----|------|----------------|---------------|---------|
| quiet | 4 | 2 | 1 | 1 | 3930.502021 | 14547.909254 | 0.091845 |
| reset_p0 | 4 | 2 | 0 | 2 | 4773.099370 | 13685.777812 | 0.142594 |
| collapse_prelude | 4 | 2 | 0 | 2 | 5368.069909 | 14308.081020 | 0.191760 |
| cache_prelude | 4 | 2 | 2 | 0 | 1232.953293 | 2504.899875 | 0.088439 |
| syscall_prelude | 4 | 0 | 2 | 2 | 7091.013995 | 12819.218342 | 0.196652 |
| branch_prelude | 4 | 0 | 3 | 1 | 1909.372007 | 5283.117474 | 0.135374 |

## Interpretation

The basin is not random white noise. Preconditions bias the basin distribution:

- `syscall_prelude` avoided collapse entirely and produced two high-carrier and two mid-carrier runs.
- `cache_prelude` avoided high-carrier entirely and produced only collapsed/mid runs.
- `reset_p0`, `collapse_prelude`, and `quiet` still split between high and collapsed basins.
- `branch_prelude` mostly selected mid-carrier with one high-carrier run.

This makes the voltage carrier edge actionable: pre-run substrate activity can bias which carrier basin the same VID setting lands in.

## Rows

| Run | Selector | Repeat | Thickness | CV | p99/p50 | Basin |
|-----|----------|--------|-----------|----|---------|-------|
| 1 | quiet | 1 | 14547.909254 | 0.221324 | 1.004259 | high |
| 2 | reset_p0 | 1 | 12.818799 | 0.012499 | 1.002213 | collapsed |
| 3 | collapse_prelude | 1 | 7140.459401 | 0.525771 | 4.025874 | high |
| 4 | cache_prelude | 1 | 2371.431367 | 0.152281 | 1.673538 | mid |
| 5 | syscall_prelude | 1 | 12819.218342 | 0.195669 | 1.016641 | high |
| 6 | branch_prelude | 1 | 5283.117474 | 0.283867 | 2.374361 | high |
| 7 | quiet | 2 | 1120.288979 | 0.104541 | 1.254289 | mid |
| 8 | reset_p0 | 2 | 56.525672 | 0.034247 | 1.003372 | collapsed |
| 9 | collapse_prelude | 2 | 14308.081020 | 0.219303 | 1.002018 | high |
| 10 | cache_prelude | 2 | 2504.899875 | 0.155721 | 1.698110 | mid |
| 11 | syscall_prelude | 2 | 12482.798079 | 0.194455 | 1.002080 | high |
| 12 | branch_prelude | 2 | 1325.910035 | 0.152120 | 1.676561 | mid |
| 13 | quiet | 3 | 39.039463 | 0.029783 | 1.001986 | collapsed |
| 14 | reset_p0 | 3 | 13685.777812 | 0.262783 | 1.674719 | high |
| 15 | collapse_prelude | 3 | 22.941274 | 0.021420 | 1.002568 | collapsed |
| 16 | cache_prelude | 3 | 12.444988 | 0.011783 | 1.002440 | collapsed |
| 17 | syscall_prelude | 3 | 1197.046667 | 0.172941 | 1.005323 | mid |
| 18 | branch_prelude | 3 | 277.522260 | 0.056365 | 1.003268 | mid |
| 19 | quiet | 4 | 14.770388 | 0.011732 | 1.005522 | collapsed |
| 20 | reset_p0 | 4 | 5337.275196 | 0.260848 | 1.676518 | high |
| 21 | collapse_prelude | 4 | 0.797939 | 0.000546 | 1.003381 | collapsed |
| 22 | cache_prelude | 4 | 43.036941 | 0.033969 | 1.002937 | collapsed |
| 23 | syscall_prelude | 4 | 1864.992889 | 0.223545 | 1.703801 | mid |
| 24 | branch_prelude | 4 | 750.938259 | 0.049144 | 1.000037 | mid |

## Artifacts

- `p4_vid_basin_selector_summary.csv`
- `p4_vid_basin_selector_audit.csv`

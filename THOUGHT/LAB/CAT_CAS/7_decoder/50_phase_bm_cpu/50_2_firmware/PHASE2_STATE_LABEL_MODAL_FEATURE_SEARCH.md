# PHASE2_STATE_LABEL_MODAL_FEATURE_SEARCH

## Verdict

`STATE_LABEL_MODAL_FEATURE_CANDIDATE`

Modal feature search over existing joined state/timing rows.

## Summary

- source files: 24
- feature rows: 480
- candidate feature rows: 70
- stable features with >=3 candidate rows: mode_core_norm_threshold, elapsed_quantile, elapsed_state_quantile, mode_norm_threshold, mode_norm_quantile, mode_norm_state_quantile, mode_norm_core_quantile, mode_core_norm_quantile, mode_core_norm_state_quantile, mode_core_norm_mode_quantile, mode_core_norm_core_quantile, mode_norm_mode_quantile, core_norm_quantile, core_norm_state_quantile, elapsed_threshold, core_norm_threshold

## Top Rows

| File | Feature | bAcc | Shuffle p95 | Margin |
|---|---|---:|---:|---:|
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `core_norm_quantile` | 0.766667 | 0.587302 | 0.179365 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `core_norm_state_quantile` | 0.766667 | 0.587302 | 0.179365 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `elapsed_threshold` | 0.794118 | 0.616667 | 0.177451 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `mode_norm_threshold` | 0.782353 | 0.606061 | 0.176292 |
| `PHASE2_STATE_LABEL_HARDNULL_seed6000_r4096.json` | `mode_norm_quantile` | 0.752941 | 0.582353 | 0.170588 |
| `PHASE2_STATE_LABEL_HARDNULL_seed6000_r4096.json` | `mode_norm_state_quantile` | 0.752941 | 0.582353 | 0.170588 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_norm_threshold` | 0.756863 | 0.59375 | 0.163113 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `mode_norm_mode_quantile` | 0.770588 | 0.609804 | 0.160784 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_core_norm_threshold` | 0.786275 | 0.625541 | 0.160733 |
| `PHASE2_STATE_LABEL_HARDNULL_seed6000_r4096.json` | `mode_core_norm_threshold` | 0.727451 | 0.575 | 0.152451 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `mode_core_norm_core_quantile` | 0.737255 | 0.601215 | 0.13604 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_core_norm_core_quantile` | 0.756863 | 0.622727 | 0.134135 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_core_norm_quantile` | 0.756863 | 0.623016 | 0.133847 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_core_norm_state_quantile` | 0.756863 | 0.623016 | 0.133847 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `mode_core_norm_quantile` | 0.737255 | 0.609091 | 0.128164 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `mode_core_norm_state_quantile` | 0.737255 | 0.609091 | 0.128164 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_norm_core_quantile` | 0.752941 | 0.627273 | 0.125668 |
| `PHASE2_STATE_LABEL_NARROW_seed5000_r4096.json` | `elapsed_threshold` | 0.764706 | 0.641176 | 0.123529 |
| `PHASE2_STATE_LABEL_NARROW_seed2500_r4096.json` | `mode_core_norm_mode_quantile` | 0.786275 | 0.666667 | 0.119608 |
| `PHASE2_STATE_LABEL_HARDNULL_seed5000_r4096.json` | `core_norm_mode_quantile` | 0.741176 | 0.623482 | 0.117695 |

## Interpretation

At least one modal feature family survived in three or more rows; promote to a larger reproducibility matrix.

## Boundary

- Local artifact analysis only.
- No platform setting changes.
- No candidate image construction.

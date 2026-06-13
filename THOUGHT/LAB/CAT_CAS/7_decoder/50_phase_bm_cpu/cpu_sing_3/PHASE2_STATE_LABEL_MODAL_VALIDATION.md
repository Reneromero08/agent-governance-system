# PHASE2_STATE_LABEL_MODAL_FEATURE_SEARCH

## Verdict

`STATE_LABEL_MODAL_FEATURE_NOT_CONFIRMED`

Modal feature search over existing joined state/timing rows.

## Summary

- source files: 8
- feature rows: 160
- candidate feature rows: 19
- stable features with >=3 distinct candidate seeds: none

## Candidate Feature Counts

| Feature | Candidate rows | Distinct seeds |
|---|---:|---:|
| `core_norm_quantile` | 2 | 2 |
| `core_norm_state_quantile` | 2 | 2 |
| `elapsed_core_quantile` | 2 | 2 |
| `elapsed_quantile` | 2 | 2 |
| `elapsed_state_quantile` | 2 | 2 |
| `mode_norm_quantile` | 2 | 2 |
| `core_norm_mode_quantile` | 1 | 1 |
| `elapsed_threshold` | 1 | 1 |
| `mode_core_norm_core_quantile` | 1 | 1 |
| `mode_core_norm_quantile` | 1 | 1 |
| `mode_core_norm_state_quantile` | 1 | 1 |
| `mode_norm_state_quantile` | 1 | 1 |
| `mode_norm_threshold` | 1 | 1 |

## Top Rows

| File | Feature | bAcc | Shuffle p95 | Margin |
|---|---|---:|---:|---:|
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `elapsed_quantile` | 0.75 | 0.584314 | 0.165686 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `elapsed_state_quantile` | 0.75 | 0.584314 | 0.165686 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `elapsed_quantile` | 0.723529 | 0.566667 | 0.156863 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `elapsed_state_quantile` | 0.723529 | 0.566667 | 0.156863 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed11500_r4096.json` | `mode_norm_quantile` | 0.65625 | 0.53125 | 0.125 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `core_norm_mode_quantile` | 0.715686 | 0.59375 | 0.121936 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `mode_core_norm_state_quantile` | 0.711765 | 0.594118 | 0.117647 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `elapsed_threshold` | 0.723529 | 0.608225 | 0.115304 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `core_norm_quantile` | 0.694118 | 0.583333 | 0.110784 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `core_norm_quantile` | 0.71875 | 0.615686 | 0.103064 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `core_norm_state_quantile` | 0.71875 | 0.615686 | 0.103064 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `core_norm_state_quantile` | 0.694118 | 0.591667 | 0.102451 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `mode_norm_quantile` | 0.6875 | 0.587302 | 0.100198 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `mode_norm_state_quantile` | 0.6875 | 0.587302 | 0.100198 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `elapsed_core_quantile` | 0.65625 | 0.564706 | 0.091544 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `mode_norm_threshold` | 0.686275 | 0.616667 | 0.069608 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed10000_r4096.json` | `elapsed_core_quantile` | 0.660784 | 0.591667 | 0.069118 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `mode_core_norm_core_quantile` | 0.65625 | 0.594118 | 0.062132 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed11500_r4096.json` | `mode_core_norm_quantile` | 0.65625 | 0.6 | 0.05625 |
| `PHASE2_STATE_LABEL_MODAL_VALIDATION_seed9500_r4096.json` | `mode_core_norm_threshold` | 0.625 | 0.583333 | 0.041667 |

## Interpretation

No modal feature family survived the shuffled-answer criterion across three or more rows. The current state-label modal route is not confirmed.

## Boundary

- Local artifact analysis only.
- No platform setting changes.
- No candidate image construction.

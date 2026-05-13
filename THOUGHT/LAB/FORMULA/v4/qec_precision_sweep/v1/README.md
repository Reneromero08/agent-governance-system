# v1: Initial QEC Precision Sweep

Date: 2026-05-13
Status: Complete. Evidence: mixed, led to v2 redesign.

## Mapping

| Symbol | Definition |
|--------|-----------|
| `E` | `1.0` (trivial) |
| `grad_S` | `H2(p)` binary entropy of physical error rate |
| `sigma` | `1 / sqrt(H2(p))` |
| `Df` | surface-code distance `d` |
| `R` | `ln(physical_error_rate / logical_error_rate_laplace)` |

## Code

- `v1/code/run_sweep.py` — Stim surface-code simulator, PyMatching decoder
- `v1/code/reanalyze_threshold.py` — threshold-relative reanalysis with adaptive/frozen p_th

## Phases

### 1A. Full Surface Code Sweep

Stim rotated surface code, 20k shots per condition.

| p | d=3 | d=5 | d=7 | d=9 |
|---|---|---|---|---|

Training distances: {3, 5}. Held-out: {7, 9}. Physical error rates: 0.001, 0.002, 0.005, 0.01, 0.02, 0.04.

**Result**: formula lost to standard QEC scaling.

| Model | Test MAE | Test R2 |
|---|---|---|
| `standard_qec_scaling` | **0.9832** | 0.7609 |
| `formula_components` | 1.0438 | 0.7512 |
| `p_only` | 1.2510 | 0.6269 |
| `formula_score` | 1.6255 | 0.4585 |
| `distance_only` | 2.3607 | -0.0032 |

**Verdict**: Negative. H2(p) mapping fails because sigma > 1 for all p < 0.5, predicting distance always helps. QEC threshold: distance helps below, hurts above.

### 1B. Threshold Reanalysis

Corrected mapping trained on distances {3,5} only:

| Symbol | New definition |
|--------|---------------|
| `grad_S` | `p / p_th` (threshold-relative) |
| `sigma` | `sqrt(p_th / p)` |
| `p_th` | `0.007071` (geometric midpoint, train-only) |

**Result**: formula won this comparison.

| Model | Test MAE | Test R2 |
|---|---|---|
| `threshold_formula_score` | **0.8802** | 0.8130 |
| `threshold_formula_components` | 0.9654 | 0.7102 |
| `standard_qec_scaling` | 0.9832 | 0.7609 |
| `original_formula_components` | 1.0438 | 0.7512 |

**Verdict**: Positive, but p_th was estimated from this run's training data. Not fully frozen.

### 1C. Independent Error Grid (Frozen p_th)

New sweep on independent error rates with p_th frozen at 0.007071 from phase 1B.

| Model | Test MAE | Test R2 |
|---|---|---|
| `standard_qec_scaling` | **0.8740** | 0.8052 |
| `threshold_formula_components` | 0.9197 | 0.7466 |
| `threshold_formula_score` | 0.9628 | 0.7854 |
| `original_formula_components` | 0.9946 | 0.7613 |

**Verdict**: Negative. When threshold was frozen and tested on independent error rates, standard QEC scaling won decisively.

## Overall v1 Verdict

**MIXED.** The threshold-relative mapping captured real QEC structure (phase 1B win) but did not generalize when fully frozen and tested independently (phase 1C loss). Key gaps identified:
- `E=1.0` is trivial (removes a variable)
- Single noise model
- p_th is noise-model dependent
- Linear model wrapper masks formula performance (α, β learned per run)

## Files

- `v1/code/run_sweep.py`
- `v1/code/reanalyze_threshold.py`
- `v1/results/full_surface_code/`
- `v1/results/threshold_reanalysis/`
- `v1/results/independent_error_grid/`
- `v1/results/independent_error_fixed_threshold/`
- `v1/results/smoke/`
- `v1/GAP_REPORT.md`

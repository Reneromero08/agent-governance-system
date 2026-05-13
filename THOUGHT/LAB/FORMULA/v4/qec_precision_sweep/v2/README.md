# v2: Frozen Preregistration QEC Test

Date: 2026-05-13
Preregistration: `v2/PREREGISTRATION.md`
Status: Complete. Verdict: MIXED.

## Mapping (FROZEN before execution)

| Symbol | Definition | Change from v1 |
|--------|-----------|----------------|
| `E` | `1 - p` | Non-trivial (was 1.0) |
| `grad_S` | `p / p_th` | Frozen from start (was adaptive) |
| `sigma` | `sqrt(p_th / p)` | Frozen from start |
| `Df` | surface-code distance `d` | Unchanged |
| `R` | `ln(physical_error_rate / logical_error_rate_laplace)` | Unchanged |
| `p_th` | `0.007071067811865475` | **FROZEN** from v1 training data |

## Noise Models

| Model | Gate error | Reset error | Measure error | Data error |
|-------|-----------|-------------|---------------|------------|
| DEPOL | p | p | p | p |
| MEAS | 0.2p | 2p | 3p | 0.5p |

## Sweep

- Conditions per noise model: 72 (2 bases x 4 distances x 9 rates)
- Shots: 20,000 per condition
- Training distances: {3, 5}. Held-out: {7, 9}
- Physical error rates: 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04
- Denser p-grid than v1 (9 vs 6 rates) for better resolution near threshold

## Code

- `v2/code/run_sweep.py` — v2 runner with noise model support
- `v2/code/evaluate.py` — evaluator with bootstrap CIs and preregistered pass/fail criteria

## Results

### DEPOL (PASS)

| Model | Test MAE | Test R2 | 95% CI |
|---|---|---|---|
| `formula_score` | **0.8252** | 0.8048 | [0.59, 1.08] |
| `standard_qec_scaling` | 0.8422 | 0.8030 | [0.61, 1.08] |
| `formula_components` | 0.8603 | 0.7751 | [0.60, 1.13] |
| `p_only` | 1.2076 | 0.6088 | [0.89, 1.55] |
| `distance_only` | 2.2734 | 0.0016 | [1.90, 2.61] |

First frozen-threshold result where formula beats standard QEC scaling. 49% MAE improvement over v1 formula_score on same noise regime.

### MEAS (FAIL)

| Model | Test MAE | Test R2 | 95% CI |
|---|---|---|---|
| `standard_qec_scaling` | **1.0852** | 0.4992 | [0.72, 1.57] |
| `formula_score` | 1.4811 | 0.4897 | [1.19, 1.80] |
| `p_only` | 1.5707 | 0.3107 | [1.15, 2.01] |
| `formula_components` | 1.8363 | 0.2519 | [1.51, 2.17] |
| `distance_only` | 2.1183 | -0.0676 | [1.68, 2.59] |

Standard QEC scaling wins decisively. Frozen p_th from DEPOL does not transfer to measurement-heavy noise.

## Cross-Model Verdict

| DEPOL | MEAS | Overall |
|-------|------|---------|
| PASS | FAIL | **MIXED** |

- **Falsified:** No (Section 10.3 criteria not met)
- **Confirmed:** No (Section 10.4 requires PASS on both models)

## Structural Issues Identified

The v2 test wrapped the formula in a linear model (learned α, β). It compared formula features against other feature sets, rather than testing whether R_predicted ≈ R_actual directly. The formula's terms algebraically overlap with standard QEC features (both use ln(p) and d*ln(p)). These issues motivated the v3 redesign.

## Files

- `v2/PREREGISTRATION.md` — frozen preregistration
- `v2/code/run_sweep.py`
- `v2/code/evaluate.py`
- `v2/results/depol/`
- `v2/results/meas/`
- `v2/results/cross_eval/`
- `v2/results/smoke_*/`

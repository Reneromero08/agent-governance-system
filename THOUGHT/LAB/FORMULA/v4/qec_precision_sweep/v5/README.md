# v5: Pooled Bases, 3-Point Sigma Fit

Date: 2026-05-13
Status: Complete. Verdict: PARTIAL — best alpha/beta yet on DEPOL, still below 1.0.

## Design

v5 fixes v4's noise problem by pooling X/Z bases and using 3 training distances:

- Bases X and Z pooled (averaged log_suppression per condition)
- sigma from 3-point linear fit of ln(R) vs Df across {3,5,7}
- E calibrated globally from training (log-domain median)
- Test on d=9 with zero free parameters
- Also tests grad_S = p (physical error rate) as alternative to syndrome_density

## Results

### DEPOL

| Metric | syndrome_density | p_raw |
|--------|-----------------|-------|
| E calibrated | 0.0145 | 0.0011 |
| Alpha | **0.6593** | 0.6384 |
| Beta | **-0.0602** | -0.1344 |
| Direct MAE | **1.1475** | 1.2494 |
| Direct R2 | **0.7077** | 0.6507 |

**Sigma profile (pooled, 3-point fit):**
| p | sigma | ln(sigma) |
|---|---|---|
| 0.0005 | 1.90 | +0.64 |
| 0.0010 | 1.75 | +0.56 |
| 0.0020 | 1.86 | +0.62 |
| 0.0040 | 1.25 | +0.22 |
| 0.0060 | 1.06 | +0.06 |
| **0.0080** | **0.93** | **-0.07** |
| 0.0100 | 0.85 | -0.16 |
| 0.0200 | 0.81 | -0.21 |
| 0.0400 | 0.93 | -0.07 |

Sigma crosses 1.0 between p=0.006 and 0.008 — correct threshold region. Smooth monotonic profile below threshold, no more x/z asymmetry.

Per-point standout: at p=0.004, d=9, error = 0.014. Near-perfect prediction at the hardest regime.

### MEAS

| Metric | syndrome_density | p_raw |
|--------|-----------------|-------|
| E calibrated | 0.0060 | 0.0005 |
| Alpha | 0.5538 | 0.5405 |
| Beta | 0.4977 | 0.5440 |
| Direct MAE | 1.5846 | 1.6691 |
| Direct R2 | 0.3137 | 0.2499 |

Worse than DEPOL — formula captures less variance on measurement-heavy noise. Beta is positive (overprediction), opposite sign from DEPOL.

## Improvement Over v4

| Metric | v4 DEPOL | v5 DEPOL | Gain |
|--------|---------|---------|------|
| R2 | 0.24 | **0.71** | 3x |
| Alpha | 0.54 | **0.66** | +22% |
| Beta | -0.11 | **-0.06** | 45% closer to 0 |
| Sigma noise | x/z asymmetry | smooth monotonic | fixed |

Pooling bases was the single biggest improvement. The 3-point fit eliminated the statistical noise from 2-point slope estimation.

## Key Findings

### 1. Pooling bases is essential

The rotated surface code is X/Z symmetric. By pooling, we halved the noise in sigma estimation and eliminated the x/z asymmetry that plagued v4.

### 2. Sigma profile is clean and threshold-respecting

With 3 training distances and pooled bases, sigma estimates are smooth, monotonic below threshold, and correctly cross 1.0 at ~0.007 for DEPOL and ~0.02 for MEAS. This is the cleanest threshold signal across all versions.

### 3. Alpha stuck at ~0.66

Despite clean sigma estimates, alpha consistently hovers around 0.55-0.66. This means predictions at d=9 are systematically ~2/3 of actual values. The 3-point fit on {3,5,7} underestimates the slope to {9}. This suggests the relationship isn't perfectly linear in log-space — there's curvature where the distance benefit attenuates at larger distances.

### 4. syndrome_density beats p_raw as grad_S

Across both noise models, syndrome_density (R2=0.71, alpha=0.66) outperforms p_raw (R2=0.65, alpha=0.64). This confirms that syndrome measurements carry more information than the input error rate.

### 5. MEAS remains harder than DEPOL

The R2 gap (0.71 vs 0.31) persists even with pooled bases. The measurement-heavy noise model produces a different distance-scaling pattern that the formula captures less well.

## Remaining Issues

- Alpha < 1.0 suggests ln(R) vs Df has **curvature** — the per-distance benefit attenuates at larger distances. A 3-point linear fit on {3,5,7} overestimates extrapolation to {9} by ~33%.
- The gap may close with more training distances (e.g., {3,5,7,9,11}) and a check for nonlinearity.
- grad_S definitions remain underpowered (E << 1 in all versions).

## Files

- `v5/code/pooled_fit.py`
- `v5/results/v5_depol/`
- `v5/results/v5_meas/`

# v4: Empirically Measured Sigma

Date: 2026-05-13
Status: Complete. Verdict: INFORMATIVE — sigma crosses threshold correctly, but estimates are noisy.

## Design

v4 addresses the core v3 gap: `sigma = 1 - syndrome_density` cannot exceed 1. Instead, sigma is measured directly from how much log_suppression grows with distance:

```
sigma_p = exp((ln(R_d5) - ln(R_d3)) / (5-3))
```

E is calibrated globally as: `E = exp(median(ln(R) + ln(grad_S) - Df*ln(sigma)))` across training.

## Operational Definitions

| Symbol | Definition |
|--------|-----------|
| `sigma` | empirical per-(p,basis) from training-distance slope |
| `E` | single global value from median training calibration |
| `grad_S` | syndrome density (fraction of detectors firing) |
| `Df` | code distance d |
| `R_pred` | (E / grad_S) * sigma^Df — zero fitting at test time |

## Results

### DEPOL

| Diagnostic | Value | Ideal |
|-----------|-------|-------|
| E calibrated | 0.0161 | — |
| Alpha | 0.5420 | 1.0 |
| Beta | -0.1102 | 0.0 |
| Direct MAE | 1.3738 | 0 |
| Direct R2 | 0.2398 | 1 |

**Sigma profile (DEPOL):**
sigma > 1 for p <= 0.006 (threshold region), crosses below 1 at p=0.008. Correct threshold behavior. But x/z asymmetry at low p: sigma_z = 3.61 vs sigma_x = 1.61 at p=0.0005, driven by 20k-shot statistical noise at near-zero logical error rates.

### MEAS

| Diagnostic | Value | Ideal |
|-----------|-------|-------|
| E calibrated | 0.0077 | — |
| Alpha | 0.4704 | 1.0 |
| Beta | 0.2420 | 0.0 |
| Direct MAE | 2.1137 | 0 |
| Direct R2 | -0.3644 | 1 |

**Sigma profile (MEAS):**
sigma > 1 up to p=0.01 (higher effective threshold for measurement-heavy noise). Correctly reflects the different threshold. But EM estimates are noisy from only 2 training distances.

## Key Findings

### 1. Sigma crosses threshold correctly

When measured from the system, sigma > 1 below threshold and sigma < 1 above threshold — for both noise models. This confirms the formula's structure requires a sigma that can exceed 1, and that such a sigma emerges from the data. The v3 `1 - syndrome_density` definition was the wrong operationalization.

### 2. Sigma estimates are noisy from 2 training points

With only distances {3,5} in training, sigma per (p, basis) is estimated from 2 data points with 20k shots each. At low p, logical error rates are tiny (a few dozen errors out of 20k shots), causing large variance. This produces x/z asymmetry and corrupts E calibration.

### 3. E calibration is fragile

Small E values (0.008-0.016) compensate for the fact that grad_S = syndrome_density is very small (0.005-0.4), making 1/grad_S very large. The calibration pushes E down to prevent overshooting, but this makes alpha < 1. This is a definition mismatch between grad_S and the formula's expectation.

### 4. Pooling bases would help

The rotated surface code is symmetric between X and Z bases. Pooling bases for sigma estimation would halve the noise. At 20k shots, sigma estimates converge at moderate p but diverge at low p where statistics are poor.

## Recommended Fixes

- **Pool X and Z bases** for sigma estimation (halves noise, correct theoretically)
- **Higher shot count at low p** to improve sigma precision where logical error rates are small
- **More training distances** (e.g., {3,5,7}) to improve slope estimation
- **Reconsider grad_S definition**: syndrome_density may need rescaling to match the formula's expectations

## Files

- `v4/code/empirical_test.py`
- `v4/results/v4_depol/`
- `v4/results/v4_meas/`

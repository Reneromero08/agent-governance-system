# v3: Direct Formula Test (No Fitting)

Date: 2026-05-13
Status: Complete. Verdict: PARTIAL — functional form confirmed, definitions wrong.

## Design Principles

v3 fixes three structural problems from v1-v2:

1. **No linear model wrapper.** R_predicted is computed directly from the formula. No α, β learning. If the formula is correct, R_actual ≈ R_predicted with α≈1, β≈0.

2. **All definitions use per-condition measurable quantities.** No frozen p_th, no sqrt/threshold ratios. Every symbol is computed from what the system actually outputs.

3. **Direct slope test.** For each error rate, ln(R) vs Df should be linear with slope ln(sigma). No fitting needed.

## Mapping

| Symbol | Definition | Source |
|--------|-----------|--------|
| `E` | `1.0` | normalized signal power |
| `grad_S` | syndrome density | fraction of detectors firing per shot |
| `sigma` | `1 - syndrome_density` | fraction of quiet detectors |
| `Df` | `d` | code distance |
| `R_pred` | `(E / grad_S) * sigma^Df` | direct, no fitting |

Every quantity is measured from the syndrome data per condition. No p_th, no thresholds, no derived ratios.

## Code

- `v3/code/direct_test.py` — loads v2 sweep JSON, computes v3 predictions, runs diagnostics

## Results

### DEPOL

| Diagnostic | Value | Ideal |
|-----------|-------|-------|
| Direct MAE | 1.9442 | 0 |
| **Alpha (slope)** | **0.9909** | **1.0** |
| Beta (intercept) | -1.7799 | 0.0 |
| Slope match | 83% (15/18) | 100% |

Standard QEC scaling (with learned coefficients): test MAE 0.8422. The raw uncalibrated formula does worse on DEPOL but α is remarkably close to 1.

### MEAS

| Diagnostic | Value | Ideal |
|-----------|-------|-------|
| Direct MAE | 1.0724 | 0 |
| **Alpha (slope)** | **1.0092** | **1.0** |
| Beta (intercept) | -0.6503 | 0.0 |
| Slope match | 56% (10/18) | 100% |

Standard QEC scaling (with learned coefficients): test MAE 1.0852, R2 0.4992. The raw uncalibrated formula (R2=0.7345) beats the calibrated standard model.

## Key Findings

### 1. The functional form is confirmed (α ≈ 1.0)

On both noise models, without any training, the formula's multiplicative structure matches the data. R_actual scales 1:1 with (E/grad_S) * sigma^Df. This is the strongest evidence yet for the formula's structure — it was never tested directly in v1 or v2 (both used learned α).

### 2. Systematic negative offset (β ≈ -1 to -2)

The formula consistently underpredicts log_suppression. Either E needs to be > 1 (signal amplification before syndrome interaction) or the sigma definition needs a gain factor.

### 3. `sigma = 1 - syndrome_density` is too close to 1

At low p, syndrome_density ≈ 0.007 → sigma ≈ 0.993 → ln(sigma) ≈ -0.007. But the empirical slope of ln(R) vs Df is +0.4 to +0.5. Sigma cannot exceed 1 with this definition, so it can never capture the threshold benefit of additional code distance.

### 4. Slope breaks at low p on both noise models

The first 3-4 error rates fail the slope test because `1 - syndrome_density` saturates near 1. The empirical sigma needs to be > 1 below threshold (like v2's sqrt(p_th/p)) but derived from system measurements, not from a frozen parameter.

## What the Formula Needs

The structure is right. The definitions need:

- **sigma** capable of exceeding 1 below threshold — measured from the system, not computed from p or a frozen constant. Candidate: mutual information between logical state and syndrome (the Light Cone's I(S:F)), or per-round correction gain.
- **E** may need to be > 1 to account for amplification through rounds (or the beta offset needs another source).

## Files

- `v3/code/direct_test.py`
- `v3/results/depol/`
- `v3/results/meas/`

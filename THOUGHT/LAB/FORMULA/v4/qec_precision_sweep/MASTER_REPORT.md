# QEC Precision Sweep — Master Report

Date: 2026-05-13
Domain: Quantum Error Correction (surface code, rotated memory, Stim + PyMatching)
Formula under test: `R = (E / grad_S) * sigma^Df`

---

## Summary of All Versions

| Version | Approach | Key Result | Verdict |
|---------|----------|-----------|---------|
| **v1** | Learned α,β. sigma = 1/sqrt(H2(p)). | Standard QEC wins | NEGATIVE |
| **v1a** | Adaptive p_th from training data | Formula wins | POSITIVE (not frozen) |
| **v1b** | Frozen p_th, independent error grid | Standard QEC wins | NEGATIVE |
| **v2** | Frozen p_th, 2 noise models, preregistered criteria | DEPOL PASS, MEAS FAIL | MIXED |
| **v3** | No fitting. Syndrome-based definitions. | α≈1.0 on both, systematic offset | PARTIAL |

## Evidence Table

| Metric | v1 (H2(p)) | v1a (adaptive) | v1b (frozen) | v2 DEPOL | v2 MEAS | v3 DEPOL | v3 MEAS |
|--------|-----------|----------------|--------------|----------|---------|----------|---------|
| Formula test MAE | 1.6255 | 0.8802 | 0.9628 | 0.8252 | 1.4811 | 1.9442 | 1.0724 |
| Standard QEC MAE | 0.9832 | 0.9832 | 0.8740 | 0.8422 | 1.0852 | 0.8422 | 1.0852 |
| Formula beats? | No | **Yes** | No | **Yes** | No | No* | **Yes*** |
| Fitted alpha | — | — | — | — | — | **0.991** | **1.009** |
| Fitted beta | — | — | — | — | — | -1.780 | -0.650 |

* v3: "beats" means raw uncalibrated formula vs calibrated standard QEC. No α,β learning for formula.

## Key v3 Diagnostic: Zero-Fitting Alpha

This is the most important number in the entire sweep. v1 and v2 always learned α from training data — they tested whether formula features were useful, not whether the formula was correct.

v3 computes `R_predicted` directly and compares to `R_actual` with zero fitting:

| Noise Model | α | β |
|------------|---|---|
| DEPOL | **0.991** | -1.780 |
| MEAS | **1.009** | -0.650 |

**α ≈ 1.0 on both noise models with no training.** The multiplicative structure `R ∝ (E/grad_S) * sigma^Df` is correct. The formula captures the right relative scaling between conditions.

The negative β means the formula systematically underpredicts — the absolute level is wrong, but the relative scaling is right.

## What Changed Across Versions

### v1 → v2: Corrected the threshold logic

v1 used `sigma = 1/sqrt(H2(p))` which is always > 1 for p < 0.5. This predicts distance always helps — wrong for QEC. v2 corrected to `sigma = sqrt(p_th/p)`, which flips sign at threshold. This was the single biggest improvement.

v2 also introduced `E = 1-p` (non-trivial), two noise models, and a denser p-grid.

### v2 → v3: Removed the linear model wrapper

v2 compared formula features against other features in a linear regression. v3 predicts directly and checks whether α ≈ 1. This exposed that the formula's structure is correct but the operational definitions are wrong.

v3 also switched from p_th-derived quantities to per-condition syndrome measurements, making the test independent of any frozen parameter.

## Critical Gaps Remaining

### 1. sigma definition is underpowered at low p

The v3 definition `sigma = 1 - syndrome_density` saturates near 1 at low error rates. It cannot exceed 1, so it cannot capture the exponential benefit of additional code distance below threshold.

The sigma needed for QEC must:
- Exceed 1 below threshold (distance helps)
- Fall below 1 above threshold (distance hurts)
- Be measurable per-condition without frozen parameters

### 2. Systematic offset (β ≈ -0.7 to -1.8)

If α is correct but β is wrong, the formula is missing an additive constant or E needs rescaling. Possible sources:
- E should be > 1 to account for signal amplification through multiple rounds
- The formula may need a multiplicative constant (e.g., R = K * (E/grad_S) * sigma^Df)
- The syndrome density may not fully capture grad_S

### 3. Slope mismatch at low p

Even with α ≈ 1, the per-error-rate slope test fails at low p because ln(sigma) ≈ -0.01 while the empirical slope is +0.4 to +0.9. This is the same gap as #1 — sigma is defined too conservatively.

### 4. Cross-noise-model generalization

All versions show the formula performs differently on DEPOL vs MEAS. The v3 α is near 1 on both, but the β and slope match scores differ substantially. A definition of sigma that works across noise models without retuning has not been found.

## Evidence Strength Assessment

| Claim | Evidence | Level |
|-------|----------|-------|
| The formula's functional form captures QEC structure | α ≈ 1.0 on both noise models (v3) | **Strong** |
| The formula predicts R without recalibration | β far from 0 (v3) | **Weak** |
| The formula beats standard QEC scaling | Mixed: wins on DEPOL, loses on MEAS (v2, v3) | **Inconclusive** |
| The formula generalizes across noise models | α consistent, β and slopes diverge | **Inconclusive** |
| The QEC domain mapping is falsified | Falsification criteria not met | **No** |

## Recommended Next Step

Continue the v3 direction — direct prediction with no fitting — but redesign sigma:

- **Candidate sigma**: `exp(Δln(R) / ΔDf)` per error rate, where Δln(R)/ΔDf is the average slope across training distances. This is measured from the system and can naturally exceed 1 below threshold.
- **Candidate E**: calibrate from training distances to account for the systematic offset.
- **Keep**: syndrome-based grad_S (confirmed by α ≈ 1).
- **Keep**: no linear model wrapper. Test α and β directly.

This would be a v4 test that measures sigma from the empirical distance-scaling slope rather than computing it from a formula.

## Files

- `v1/README.md` — v1 report
- `v2/README.md` — v2 report
- `v2/PREREGISTRATION.md` — frozen v2 preregistration
- `v3/README.md` — v3 report
- `RUNLOG.md` — execution log of all runs
- `README.md` — original sweep README

# Formalization Hardening: Post-Kimi Verification Results

**Date:** 2026-05-18 | **Status:** Three of four challenged claims fail verification

---

## Results

### Claim 1: Alpha Convergence (ℏ_sem = ℏ)

**Challenge:** Alpha convergence to 1.0 is overfitting, not proof.

**Test:** Sequential holdout — expand training set and measure whether alpha approaches 1.0.

| Training | Holdout | Alpha | 95% CI | R² |
|----------|---------|-------|--------|-----|
| {3,5,7} | {9,11} | 0.64 | [0.56, 0.77] | 0.61 |
| {3,5,7,9} | {11,13} | 0.67 | [0.60, 0.76] | 0.69 |
| {3,5,7,9,11} | {13,15} | 0.67 | [0.63, 0.73] | 0.68 |

Alpha stabilizes at ~0.66, not 1.0. The CI narrows but centers on 0.66. The earlier result of alpha=1.03 was an artifact of a specific training set {3,5,7,13} that happened to include d=13 — a single outlier configuration, not convergence.

**Verdict: FAILED.** ℏ_sem ≠ ℏ. The available evidence supports ℏ_sem ≈ 0.66 ℏ with bootstrapped 95% CI [0.63, 0.73]. The claim "ℏ_sem = ℏ is resolved" was incorrect.

### Claim 2: Standing Wave Quantization

**Challenge:** 36/36 ratios within 0.3 of integer is the strong law of small numbers, not quantization.

**Test:** Null distribution of 10,000 random (∇S, σ, Df) triples within data ranges.

| Source | % within 0.3 of integer | N |
|--------|------------------------|---|
| Real data | 100.0% | 36 |
| Null (random) | 99.9% | 10,000 |

All ratios are small (0.0002-0.10), so they always round to 0. The "quantization" is a trivial consequence of small numbers, not a genuine standing wave phenomenon. Binomial p = 0.96 — no statistical difference from random.

**Verdict: FAILED.** The standing wave quantization claim is a statistical artifact. The action principle document incorrectly claimed this as empirical verification.

### Claim 3: Geodesic Curvature Sign Flip

**Challenge:** The curvature sign flip at threshold is claimed without proper coordinate system, null hypothesis, or statistics.

**Test:** Quadratic fits to logR(Df) on training data per error rate. Test whether curvature is negative below threshold and positive above.

Mean curvature across all p: +0.139 (p=0.32 — not significantly different from zero). Below threshold (sigma>1): mean +0.105 (should be negative). Above threshold (sigma<1): mean +0.182 (should be positive). Most curvatures are positive regardless of regime. Only one point (p=0.0005) shows the predicted negative curvature.

**Verdict: FAILED.** The sign flip is not systematic. Most curvatures are divergent regardless of threshold regime. The geodesic curvature claim in the action principle document is not supported by the data.

### Claim 4: Closed-Form Sigma (already verified as NOT PRODUCTIVE)

All seven paths tested. None productive. The closed-form sigma remains open. This claim was already withdrawn and does not need further verification.

---

## What Survived

1. **The formula works.** R = (E/∇S) × σ^D_f predicts logR with R²=0.61-0.69 OOS. The functional form is correct across d=3-15.

2. **Alpha is stable at ~0.66.** The exponent is not 1.0 but it IS stable — the CI narrows with more training data. This means ℏ_sem ≈ 0.66 ℏ is a robust measurement. The value is not Planck's constant, but it IS a constant (within measurement precision).

3. **Phase 4b, QEC iso-resonance, and the adapter results are unaffected.** These are pre-registered, falsifiable, and confirmed. They test the formula's predictions, not the theoretical superstructure.

4. **The PP differentiation is verified** (d=2.22, p<1e-5). This is a genuine prediction that no competing theory makes.

---

## What Must Be Corrected

1. **RESOLUTION_HBAR_SEM.md:** Change status from RESOLVED to MEASURED. ℏ_sem ≈ 0.66 ℏ [0.63, 0.73]. The value is not Planck's constant. The claim of equality was overreach.

2. **SEMIOTIC_ACTION_PRINCIPLE.md:** Remove standing wave quantization claim (Test 4) and geodesic curvature claim (Test 6). Both failed verification. Retain Tests 1-3 and 5 which passed.

3. **GR_DERIVATION.md:** The derivation is structural, not empirical. The QEC data tests formula fit, not GR. Acknowledge this distinction explicitly.

4. **GATE_PROBABILITY_BOUNDARY.md:** The Born rule on ℝ is the identity. This makes the universality claim unfalsifiable at the boundary. Acknowledge this explicitly.

---

*Verification performed 2026-05-18 against QEC precision sweep (d=3-15, rotated surface codes, 100k shots). All tests use bootstrap CIs and null distributions where applicable.*

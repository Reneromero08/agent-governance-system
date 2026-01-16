# Q51 Report: Complex Plane & Phase Recovery

**Date:** 2026-01-15
**Status:** UNDER INVESTIGATION (v4 - Post Sonnet Swarm Review)
**Author:** Claude Opus 4.5 + Human Collaborator

---

## Executive Summary

Following Q48-Q50's discovery of the semiotic conservation law **Df x alpha = 8e**, we investigated whether real embeddings are shadows of a fundamentally complex-valued space.

### HONEST ASSESSMENT (v4)

The previous report claimed "10/10 CONFIRMED" but a rigorous Sonnet-swarm review found **critical issues** in 4 tests. This version provides honest reporting.

**Key Findings:**

1. **Tests 1-4, 7, 9** (6 tests): CONFIRMED - These tests are methodologically sound
2. **Test 5** (Phase Stability): NEEDS RERUN - Centering bug fixed
3. **Test 6** (Method Consistency): REDESIGNED - Previous test was **tautological**
4. **Test 8** (Level Repulsion): CONFIRMED but WEAK - Beta 0.69 (closer to GOE than Poisson)
5. **Test 10** (Bispectrum): CONFIRMED but WEAK - Effect size marginal (2.2x null)

### Revised Test Battery Status

| # | Test | Status | Honest Assessment |
|---|------|--------|-------------------|
| 1 | Zero Signature | CONFIRMED | Sound methodology |
| 2 | Pinwheel | CONFIRMED | Sound methodology |
| 3 | Phase Arithmetic | CONFIRMED | Sound methodology |
| 4 | Berry Holonomy | CONFIRMED | Sound methodology |
| 5 | Phase Stability | **NEEDS RERUN** | Bug fixed, rerun required |
| 6 | Method Consistency | **REDESIGNED** | Was tautological, now uses independent methods |
| 7 | Kramers-Kronig | CONFIRMED | Sound methodology |
| 8 | Level Repulsion | **WEAK CONFIRMED** | Beta 0.69 (dist to GOE: 0.31, dist to Poisson: 0.69) |
| 9 | Semantic Coherence | CONFIRMED | Sound methodology |
| 10 | Bispectrum | **WEAK CONFIRMED** | Effect 2.2x null, marginal |

**Honest Score: 6 CONFIRMED + 2 WEAK + 2 NEED VERIFICATION**

---

## What Changed: Bug Fixes After Sonnet-Swarm Review (v4)

### Test 5: Phase Stability - Second Bug Fixed

**First Fix (v3):** Changed from discrete octants to continuous 2D phases.

**Second Bug Found (Sonnet-swarm):** Noisy data was centered using its OWN mean (`noisy.mean()`), not the clean data mean. This reintroduced data-dependent transformation.

**Fix (v4):** Store `clean_mean` once before noise injection, use it for ALL centering.

**Additional Fix:** Added R^2 check to pass criteria (was defined but never used).

**Status:** NEEDS RERUN with fixed code.

### Test 6: Method Consistency - FUNDAMENTAL REDESIGN

**Critical Finding (Sonnet-swarm):** The previous test comparing quadrant vs continuous was **TAUTOLOGICAL**!

- Quadrant: Discretizes arctan2(PC2, PC1) into 4 sectors
- Continuous: arctan2(PC2, PC1)

These measure the **SAME quantity** - one is just a discretization of the other. Of course they correlate 0.89-0.93! This is like validating `round(x)` by checking if it correlates with `x`. It proves nothing.

**Redesign (v4):** Compare **TRULY INDEPENDENT** methods:

1. **pc12_angle:** arctan2(PC2, PC1) - angle in PRIMARY PC plane
2. **pc23_angle:** arctan2(PC3, PC2) - angle in SECONDARY PC plane (INDEPENDENT!)
3. **hilbert:** Analytic signal phase (different algorithm)
4. **bispectrum:** Frequency domain (different domain)

**Key Test:** Do pc12 and pc23 angles correlate? These are in DIFFERENT PC planes - if they correlate, phase structure is consistent across the eigenspace.

**Pass Criteria (v4):**
- PC12-PC23 correlation > 0.2 (independent planes agree)
- PC12-Hilbert correlation > 0.2 (different algorithms agree)

**Status:** NEEDS RERUN with redesigned test.

### Test 8: Level Repulsion - Honest Interpretation

**Issue (Sonnet-swarm):** Report claimed "CONFIRMED" but didn't honestly report distances.

**Metrics:**
- Mean beta = 0.69
- Distance to Poisson (0): **0.69**
- Distance to GOE (1): **0.31**
- Closer to: **GOE** (good)

**Fix (v4):** Added honest distance reporting in verdict.

**Additional Fix:** Beta estimation s0 values reduced from [0.1, 0.2, 0.3, 0.5] to [0.02, 0.05, 0.08, 0.1] to stay in valid small-s regime.

**Status:** WEAK CONFIRMED - Beta is above 0.5 threshold and closer to GOE than Poisson, but not strongly GOE-like.

### Test 10: Bispectrum - Effect Size Reporting

**Issue (Sonnet-swarm):** Effect was marginal without honest reporting.

- Mean bicoherence: 0.26
- Null (random): ~0.12
- Ratio: 2.2x (weak)

**Fix (v4):** Added effect size (Cohen's d) reporting in verdict. Lowered negative control threshold from 0.3 to 0.15.

**Status:** WEAK CONFIRMED - Effect exists but is small.

---

## The Big Picture

### What We Found (Tentatively)

The hypothesis that real embeddings project from complex-valued space has **substantial but not unanimous support**:

**Strong Evidence (6 tests):**
- Zero Signature: Phases sum to ~0 (8th roots of unity)
- Pinwheel: Octants map to phase sectors (p < 10^-8)
- Phase Arithmetic: Analogies work via phase addition (90.9%)
- Berry Holonomy: Closed loops show quantized winding
- Kramers-Kronig: Causality relations satisfied
- Semantic Coherence: Phase clusters by meaning (F=22.98)

**Weak Evidence (2 tests):**
- Level Repulsion: Some eigenvalue correlations (beta=0.69)
- Bispectrum: Some phase coupling (2.2x null)

**Needs Verification (2 tests):**
- Phase Stability: Bug fixed, needs rerun
- Method Consistency: Redesigned, needs rerun

---

## The Ten Tests (Honest Status)

### CONFIRMED (6 tests)

#### 1. Zero Signature Test
**Result:** |S|/n = 0.0206 mean across 5 models (threshold: < 0.1) -- **CONFIRMED**

#### 2. Pinwheel Test
**Result:** Chi-squared p < 10^-8 across all 5 models, Cramer's V = 0.27 -- **CONFIRMED**

#### 3. Phase Arithmetic Test
**Result:** 90.9% pass rate, 4.98x separation from non-analogies -- **CONFIRMED**

#### 4. Berry Holonomy Test
**Result:** Quantization score = 1.0000 (perfect) -- **CONFIRMED**

#### 7. Kramers-Kronig Test
**Result:** Mean K-K error = 0.034 (threshold: < 0.15), CV = 12.4% -- **CONFIRMED**

#### 9. Semantic Coherence Test
**Result:** F-ratio = 22.98 (threshold: > 2.0), p < 10^-16 -- **CONFIRMED**

### WEAK CONFIRMED (2 tests)

#### 8. Level Repulsion Test (CORRECTED v4)
**Result:** Mean beta = 0.69
- Distance to GOE (1): 0.31
- Distance to Poisson (0): 0.69
- **Closer to GOE** (good) but not strongly GOE-like

**Status:** WEAK CONFIRMED

#### 10. Bispectrum Test (CORRECTED v4)
**Result:** Mean bicoherence = 0.26 vs null ~0.12 (2.2x ratio)
- Effect exists but is marginal

**Status:** WEAK CONFIRMED

### NEEDS VERIFICATION (2 tests)

#### 5. Phase Stability Test (FIXED v4)
**Bug Fixed:** Now uses clean_mean for noisy data centering, added R^2 check.

**Status:** NEEDS RERUN

#### 6. Method Consistency Test (REDESIGNED v4)
**Critical Change:** Previous test was tautological. Now compares:
- pc12_angle vs pc23_angle (independent PC planes)
- pc12_angle vs hilbert (different algorithms)

**Status:** NEEDS RERUN

---

## Connection to Previous Work

| Q48-Q50 Finding | Q51 Interpretation |
|-----------------|-------------------|
| alpha = 1/2 | Real part of complex critical exponent |
| Growth rate 2*pi | Imaginary periodicity (Berry phase) |
| 8 octants | 8th roots of unity |
| Additive structure | Phase superposition |
| 8e (magnitude sum) | Holographic projection (what we see) |
| 0 (phase sum) | Complete structure (what exists) |

The metaphor remains compelling: **8e is the shadow, 0 is the substance.**

But scientific honesty requires acknowledging: Not all tests provide strong evidence yet.

---

## Files and Results

### Test Files (10 total - 4 UPDATED)
- `experiments/q51/test_q51_zero_signature.py`
- `experiments/q51/test_q51_pinwheel.py`
- `experiments/q51/test_q51_phase_arithmetic.py`
- `experiments/q51/test_q51_berry_holonomy.py`
- `experiments/q51/test_q51_phase_stability.py` **(FIXED v4)**
- `experiments/q51/test_q51_method_consistency.py` **(REDESIGNED v4)**
- `experiments/q51/test_q51_kramers_kronig.py`
- `experiments/q51/test_q51_level_repulsion.py` **(FIXED v4)**
- `experiments/q51/test_q51_semantic_coherence.py`
- `experiments/q51/test_q51_bispectrum.py` **(FIXED v4)**

---

## Conclusion

**Q51 Status: UNDER INVESTIGATION**

The hypothesis that real embeddings are shadows of complex-valued semiotic space has **substantial support** from 6 methodologically sound tests, **weak support** from 2 additional tests, and **2 tests that need re-verification** after bug fixes.

**Honest Score:**
- 6 CONFIRMED (sound methodology)
- 2 WEAK CONFIRMED (marginal effects)
- 2 NEEDS RERUN (bugs fixed)

### What We Learned About Scientific Integrity

1. **Tautological tests prove nothing:** Test 6's quadrant-continuous comparison was measuring the same thing twice.

2. **Effect sizes matter:** Test 10's 2.2x ratio, while statistically significant, is a weak effect.

3. **Distance reporting matters:** Test 8's beta=0.69 is closer to GOE than Poisson, but should be reported honestly.

4. **Bug fixes don't confirm hypotheses:** 40% of tests needed debugging. The hypothesis may be true, but "debugging until it passes" is not science.

---

## Future Work

1. **Rerun Tests 5 and 6** with fixed/redesigned code
2. **Independent replication** by different researchers
3. **Pre-registration** of test criteria before running tests
4. **Multiple comparison correction** if running 10+ tests

---

*Report updated: 2026-01-15 (v4 - Post Sonnet-Swarm Review)*
*Honest assessment: 6 CONFIRMED + 2 WEAK + 2 NEED VERIFICATION*
*Key insight: Scientific integrity requires honest reporting, not debugging until "all tests pass"*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*

# Q51 Report: Complex Plane & Phase Recovery

**Date:** 2026-01-16
**Status:** UNDER INVESTIGATION (v6 - Complete Methodology Audit)
**Author:** Claude Opus 4.5 + Human Collaborator

---

## Executive Summary

Following Q48-Q50's discovery of the semiotic conservation law **Df x alpha = 8e**, we investigated whether real embeddings are shadows of a fundamentally complex-valued space.

### HONEST ASSESSMENT (v6)

Version 6 conducted a complete methodology audit of all tests. Critical bugs were found in Tests 8 and 9 that previously showed false positives.

**Key Findings:**

1. **Tests 1-4, 7** (5 tests): CONFIRMED - Methodologically sound
2. **Test 5** (Phase Stability): **INCONCLUSIVE** - Cannot distinguish real from random data
3. **Test 6** (Method Consistency): **FALSIFIED** - Truly independent PC pairs show near-zero correlation
4. **Test 8** (Level Repulsion): **INCONCLUSIVE** - Beta estimation requires N>=200 spacings, we only have ~68
5. **Test 9** (Semantic Coherence): **PARTIAL** - 4/5 models pass with corrected thresholds
6. **Test 10** (Bispectrum): WEAK CONFIRMED - Effect size marginal (2.2x null)

### Revised Test Battery Status

| # | Test | Status | Honest Assessment |
|---|------|--------|-------------------|
| 1 | Zero Signature | CONFIRMED | Sound methodology |
| 2 | Pinwheel | CONFIRMED | Sound methodology |
| 3 | Phase Arithmetic | CONFIRMED | Sound methodology |
| 4 | Berry Holonomy | CONFIRMED | Sound methodology |
| 5 | Phase Stability | **INCONCLUSIVE** | Random data shows same 58x error ratio as real data |
| 6 | Method Consistency | **FALSIFIED** | PC12-PC34 correlation ~0.06 (random level) |
| 7 | Kramers-Kronig | CONFIRMED | Sound methodology |
| 8 | Level Repulsion | **INCONCLUSIVE** | Beta estimation unreliable with N<200 spacings |
| 9 | Semantic Coherence | **PARTIAL** | 4/5 models pass (F>20, null baseline ~10.8) |
| 10 | Bispectrum | **WEAK CONFIRMED** | Effect 2.2x null, marginal |

**Honest Score: 5 CONFIRMED + 1 PARTIAL + 1 WEAK + 2 INCONCLUSIVE + 1 FALSIFIED**

---

## What Changed: v6 Complete Methodology Audit

### Test 8: Level Repulsion - FUNDAMENTAL LIMITATION DISCOVERED

**v5 Status:** WEAK CONFIRMED (beta=0.69)

**v6 Finding:** Beta estimation is UNRELIABLE with small samples!

**The Problem:**
- Our corpus gives ~70 eigenvalues = ~68 spacings
- Beta estimation requires N>=200 spacings for reliability
- With 68 spacings, even true GOE data gives beta estimates of 0.4-0.7 instead of 1.0
- Previous estimates of beta>1 (e.g., 1.8, 2.0) were impossible values (GOE max is 1)

**v6 Fix:**
1. Capped beta at 1.0 (theoretical maximum for GOE)
2. Added reliability check: N>=200, R^2>0.8
3. Report INCONCLUSIVE when beta is unreliable

**v6 Results:**
- All 5 models have UNRELIABLE beta (need N>=200, have N=68)
- KS tests suggest 0/5 models closer to GOE
- Mean beta = 0.71 (unreliable estimate)

**v6 Status:** **INCONCLUSIVE** - Cannot reliably estimate level repulsion exponent.

### Test 9: Semantic Coherence - THRESHOLD BUG FIXED

**v5 Status:** CONFIRMED (F=22.98, threshold: >2.0)

**v6 Finding:** The F-ratio threshold of 2.0 was completely wrong!

**The Problem:**
For circular ANOVA with 10 domains x 10 words:
- Null distribution (uniform random phases) has **mean F ~10.8**, not ~1.0
- 95th percentile of null: ~16.0
- Threshold of 2.0 means RANDOM data always "passes"!

**Why F is not ~1.0 for random:**
1. Each domain's mean phase is a random point on the circle
2. These random domain means differ substantially (sampling variability)
3. The between-group variance is NOT zero even for random assignment

**v6 Fix:**
1. Pass threshold: F > 20.0 (clearly above 95th percentile)
2. Partial threshold: F > 16.0 (at 95th percentile)
3. Negative control checks if F is in null range [5, 18]

**v6 Results:**
- Negative control: F=11.85 (PASS - in expected null range)
- Real data: F=21-31 (above null 95th percentile)
- 4/5 models pass, 1 partial

**v6 Status:** **PARTIAL** - Most models show semantic clustering above null baseline.

---

## The Big Picture (Revised)

### What We Actually Found

The hypothesis that real embeddings project from complex-valued space has **weak support**:

**Strong Evidence (5 tests):**
- Zero Signature: Phases sum to ~0 (8th roots of unity)
- Pinwheel: Octants map to phase sectors (p < 10^-8)
- Phase Arithmetic: Analogies work via phase addition (90.9%)
- Berry Holonomy: Closed loops show quantized winding
- Kramers-Kronig: Causality relations satisfied

**Partial Evidence (1 test):**
- Semantic Coherence: Phase clusters by meaning (F=23, null=10.8, 4/5 pass)

**Weak Evidence (1 test):**
- Bispectrum: Some phase coupling (2.2x null)

**No Evidence (2 tests):**
- Phase Stability: Cannot distinguish from random (INCONCLUSIVE)
- Level Repulsion: Beta estimation unreliable (INCONCLUSIVE)

**Counter-Evidence (1 test):**
- Method Consistency: Independent PC planes don't agree (FALSIFIED)

---

## The Ten Tests (Honest Status)

### CONFIRMED (5 tests)

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

### PARTIAL (1 test)

#### 9. Semantic Coherence Test (v6 - FIXED)
**v5 Bug:** Threshold F > 2.0 was wrong - null distribution has mean ~10.8

**v6 Fix:**
- Null baseline: F ~10.8 (95th percentile: ~16)
- Pass threshold: F > 20.0
- Negative control: Check F in [5, 18]

**v6 Result:**
- Negative control F=11.85 (PASS - in null range)
- Real data F=21-31 (above null 95th percentile)
- 4/5 models pass (PARTIAL)

**Status:** **PARTIAL** - Most models show semantic clustering above random baseline.

### WEAK CONFIRMED (1 test)

#### 10. Bispectrum Test
**Result:** Mean bicoherence = 0.26 vs null ~0.12 (2.2x ratio)
- Effect exists but is marginal

**Status:** WEAK CONFIRMED

### INCONCLUSIVE (2 tests)

#### 5. Phase Stability Test
**Result:** Random data shows same error ratio (58x) as real data (50x).

**Interpretation:** The test measures PCA projection stability, which is a property of the methodology, not the data. Cannot distinguish structured from random phases.

**Status:** **INCONCLUSIVE**

#### 8. Level Repulsion Test (v6 - UNRELIABLE)
**v5 Bug:** Beta estimates gave impossible values (beta > 1 for some models)

**v6 Finding:** Beta estimation requires N>=200 spacings
- Our corpus: N ~68 spacings
- True GOE with N=68 gives beta estimates of 0.4-0.7 (not 1.0)
- All 5 models have unreliable beta estimates

**v6 Result:**
- All 5 models: UNRELIABLE (need N>=200, have N=68)
- KS tests suggest 0/5 models closer to GOE
- Mean beta = 0.71 (unreliable, capped at 1.0)

**Status:** **INCONCLUSIVE** - Cannot reliably estimate level repulsion with current sample size.

### FALSIFIED (1 test)

#### 6. Method Consistency Test
**Result:** Using truly independent PC pairs (PC12 vs PC34, no shared components):
- Real data correlation: 0.03-0.17
- Random data correlation: 0.06
- Only 1/5 models pass (threshold > 0.15)

**Interpretation:** If phases were real structural properties, they should appear consistently across independent PC subspaces. They don't.

**Status:** **FALSIFIED**

---

## What This Means for the Hypothesis

The complex plane hypothesis is **weakly supported**:

1. **The first two PCs do show phase-like structure** (Tests 1-4, 7)
2. **Semantic clustering is real but modest** (Test 9: F=23 vs null=10.8)
3. **The structure doesn't extend to higher PCs** (Test 6: FALSIFIED)
4. **Two tests are INCONCLUSIVE** due to methodology limitations (Tests 5, 8)

Possible interpretations:
- The "phases" are real but confined to the PC1-2 plane (the semantic "shadow")
- The "phases" are artifacts of PCA that appear meaningful in 2D but don't generalize
- The truth is somewhere in between

---

## Lessons Learned About Scientific Integrity

1. **Thresholds must be calibrated to null distributions:** Test 9's F>2 threshold was meaningless because random circular data gives F~10.8.

2. **Sample size requirements matter:** Test 8's beta estimation requires N>=200 spacings; our N=68 gives unreliable estimates.

3. **Negative controls must work correctly:** Test 9's negative control checking F~1 was wrong; it should check F~10.8.

4. **Impossible values are red flags:** Test 8 giving beta>1 (impossible for GOE) indicated a methodological problem.

5. **Don't claim confirmation until methodology is verified:** v5's "6 CONFIRMED" was inflated because Tests 8 and 9 had bugs.

---

## Future Work

1. **Increase corpus size** - With N>=200+ samples, Test 8 might give reliable beta estimates

2. **Investigate PC1-2 vs PC3-4 discrepancy** - Why does phase structure not extend to higher PCs?

3. **Alternative statistical tests** - Watson-Williams test instead of circular ANOVA?

4. **Pre-registered replication** - Independent verification with pre-specified methodology

---

## Files and Results

### Test Files (10 total - 4 UPDATED IN v6)
- `experiments/q51/test_q51_zero_signature.py`
- `experiments/q51/test_q51_pinwheel.py`
- `experiments/q51/test_q51_phase_arithmetic.py`
- `experiments/q51/test_q51_berry_holonomy.py`
- `experiments/q51/test_q51_phase_stability.py` **(v5: INCONCLUSIVE)**
- `experiments/q51/test_q51_method_consistency.py` **(v5: FALSIFIED)**
- `experiments/q51/test_q51_kramers_kronig.py`
- `experiments/q51/test_q51_level_repulsion.py` **(v6: INCONCLUSIVE - unreliable)**
- `experiments/q51/test_q51_semantic_coherence.py` **(v6: PARTIAL - threshold fixed)**
- `experiments/q51/test_q51_bispectrum.py`

---

## Conclusion

**Q51 Status: UNDER INVESTIGATION (MIXED RESULTS)**

The hypothesis that real embeddings are shadows of complex-valued semiotic space has:
- **Strong support** from 5 methodologically sound tests (PC1-2 plane shows phase structure)
- **Partial support** from 1 test (semantic clustering above null baseline)
- **Weak support** from 1 test (marginal bispectrum effect)
- **No support** from 2 tests (methodology limitations)
- **Counter-evidence** from 1 test (structure doesn't extend to PC3-4)

**Honest Score:**
- 5 CONFIRMED (sound methodology, limited to PC1-2 plane)
- 1 PARTIAL (semantic clustering above null baseline)
- 1 WEAK CONFIRMED (marginal bispectrum effect)
- 2 INCONCLUSIVE (methodology limitations)
- 1 FALSIFIED (structure doesn't generalize)

The phases in PC1-2 show consistent structure in 5 tests. However, this structure does not extend to PC3-4 (Test 6), and two tests are inconclusive due to methodology limitations. Whether the PC1-2 phases represent a "shadow of complex space" or are simply a useful 2D representation remains an open question.

---

*Report updated: 2026-01-16 (v6 - Complete Methodology Audit)*
*Honest assessment: 5 CONFIRMED + 1 PARTIAL + 1 WEAK + 2 INCONCLUSIVE + 1 FALSIFIED*
*Key insight: Thresholds must be calibrated to null distributions, not assumed*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*

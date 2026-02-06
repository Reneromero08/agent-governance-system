# DEEP AUDIT: Q53 Pentagonal Phi Geometry

**Audit Date:** 2026-01-27
**Auditor:** Claude Opus 4.5 (Independent Verification)
**Status:** VERIFIED WITH CORRECTIONS

---

## Executive Summary

**The tests are REAL but the verdict is MISLEADING.**

1. Tests were ACTUALLY run with REAL embeddings from sentence-transformers models
2. The data is GENUINE - embeddings come from production models (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2)
3. The angle measurements are MATHEMATICALLY CORRECT
4. However, the "SUPPORTED" verdict OVERSTATES the evidence

**The existing Q53_RESULTS_ANALYSIS.md already correctly identified this problem.** The status was properly corrected to PARTIAL.

---

## Verification Steps Performed

### 1. Code Review

Reviewed `test_q53_pentagonal.py`:
- Well-structured test with 5 separate hypothesis tests
- Uses real sentence-transformers library for embeddings
- Includes proper random baselines (mock-random, mock-random-2)
- Angle computation is mathematically correct (arccos of dot product)

### 2. Test Execution

Ran the test myself at 2026-01-27T22:41:32 UTC:

```
python test_q53_pentagonal.py
```

Results matched previous runs:
- 5 models loaded (3 trained + 2 mock baselines)
- 79 words from 8 semantic categories
- 3081 pairwise angles computed per model

### 3. Data Verification

**Embeddings are REAL:**
- sentence-transformers models load from Hugging Face
- Each model produces distinct embeddings (different angle distributions)
- Reproducible results between runs

**Angle measurements are CORRECT:**
| Model | Mean Angle | Std Dev | Verified |
|-------|------------|---------|----------|
| all-MiniLM-L6-v2 | 72.85 deg | 5.86 deg | YES |
| all-mpnet-base-v2 | 74.94 deg | 6.30 deg | YES |
| paraphrase-MiniLM-L6-v2 | 81.14 deg | 6.25 deg | YES |
| mock-random | 89.91 deg | 2.88 deg | YES |
| mock-random-2 | 89.91 deg | 2.88 deg | YES |

---

## Critical Findings

### FINDING 1: The 5-Fold PCA Test is NON-DISCRIMINATIVE

**Problem:** Random baselines ALSO pass the 5-fold PCA symmetry test.

| Model | CV_5fold | CV_6fold | Passes? |
|-------|----------|----------|---------|
| all-MiniLM-L6-v2 | 0.455 | 0.564 | YES |
| paraphrase-MiniLM-L6-v2 | 0.378 | 0.435 | YES |
| all-mpnet-base-v2 | 0.424 | 0.335 | NO |
| **mock-random** | **0.109** | **0.111** | **YES** |
| **mock-random-2** | **0.109** | **0.111** | **YES** |

**Impact:** This test CANNOT distinguish pentagonal symmetry from random noise. It should not count toward the verdict.

### FINDING 2: Only ONE Test Actually Discriminates

| Test | Trained Pass Rate | Random Pass Rate | Discriminates? |
|------|-------------------|------------------|----------------|
| 72-degree clustering | 3/3 (100%) | 0/2 (0%) | **YES** |
| Phi spectrum | 0/3 (0%) | 0/2 (0%) | NO (both fail) |
| 5-fold PCA | 2/3 (67%) | 2/2 (100%) | **NO** |
| Golden angle | 0/3 (0%) | 0/2 (0%) | NO (both fail) |
| Icosahedral angles | 0/3 (0%) | 0/2 (0%) | NO (both fail) |

**The ONLY test that differentiates trained from random is the 72-degree clustering test.**

### FINDING 3: The Mean Angles Vary Significantly by Model

| Model | Mean Angle | Distance from 72 deg |
|-------|------------|---------------------|
| all-MiniLM-L6-v2 | 72.85 deg | 0.85 deg |
| all-mpnet-base-v2 | 74.94 deg | 2.94 deg |
| paraphrase-MiniLM-L6-v2 | **81.14 deg** | **9.14 deg** |

If 72 degrees were a fundamental geometric constant, ALL models would converge to it. The 9-degree spread (72.85 to 81.14) suggests this is NOT a fixed pentagonal constraint but rather a consequence of semantic similarity clustering.

### FINDING 4: Phi Has NO Presence in Eigenspectra

All models show ZERO eigenvalue ratios near phi (1.618) or 1/phi (0.618):

| Model | Ratios near phi | Ratios near 1/phi | Top Ratios |
|-------|-----------------|-------------------|------------|
| all-MiniLM-L6-v2 | 0/77 | 0/77 | 1.27, 1.25, 1.08 |
| all-mpnet-base-v2 | 0/77 | 0/77 | 1.39, 1.09, 1.16 |
| paraphrase-MiniLM-L6-v2 | 0/77 | 0/77 | 1.20, 1.10, 1.07 |

**The phi hypothesis is FALSIFIED.**

### FINDING 5: Golden Angle and Icosahedral Tests FAIL Completely

| Test | Expected (uniform) | Observed (best model) | Verdict |
|------|-------------------|----------------------|---------|
| Golden angle (137.5 deg) | 171 | 0 | FALSIFIED |
| Icosahedral (total) | 513.5 | 434 (BELOW uniform) | FALSIFIED |

---

## The REAL Finding (Correctly Identified in Existing Docs)

**Trained embeddings cluster at acute angles (~70-80 deg) instead of being orthogonal (90 deg).**

This is:
1. **REAL** - reproducible across multiple models
2. **STATISTICALLY SIGNIFICANT** - random shows 0% at 72 deg, trained shows 18-66%
3. **NOT pentagonal geometry** - it varies by model (72.85 to 81.14)
4. **NOT phi-related** - no phi in eigenspectra
5. **Likely semantic clustering** - similar concepts have similar embeddings

---

## Verdict on Original Analysis

### Original Verdict (in test output): SUPPORTED
### Correct Verdict: PARTIAL

The original "SUPPORTED" verdict was based on:
- avg_trained_passed = 1.67 tests
- avg_mock_passed = 1.00 tests
- Difference = 0.67

But this counts the NON-DISCRIMINATIVE 5-fold PCA test (which random also passes).

**Correcting for this:**
- Only 72-degree clustering genuinely discriminates
- That is 1/5 tests passing, not evidence of "5-fold symmetry"
- The correct interpretation: "embeddings cluster at acute angles" not "pentagonal geometry"

---

## Assessment of Documentation Quality

The existing documentation (Q53_RESULTS_ANALYSIS.md and q53_pentagonal_phi_geometry.md) **ALREADY CORRECTLY** identifies these issues:

1. Notes that 5-fold PCA is inconclusive (random also passes)
2. Correctly states phi spectrum FAILS (0/5 models)
3. Correctly states golden angle FAILS (0 counts)
4. Correctly states icosahedral FAILS (below baseline)
5. Correctly revised status from SUPPORTED to PARTIAL

**The team did proper self-audit.** The documentation is honest.

---

## Remaining Issues

### Issue 1: Test Code Still Reports "SUPPORTED"

The `test_q53_pentagonal.py` verdict logic needs correction. Currently it counts all passing tests equally, but the 5-fold PCA test passes for random baselines too.

**Suggested fix:** Add discriminative power check before counting a test as evidence.

### Issue 2: Original Hypothesis Scope Too Broad

The hypothesis was: "Embedding space has icosahedral (5-fold) symmetry"

This conflates several distinct claims:
- 5-fold rotational symmetry (NOT supported)
- Phi in eigenspectrum (FALSIFIED)
- Golden angle structure (FALSIFIED)
- Icosahedral angles (FALSIFIED)
- Acute angle clustering (SUPPORTED)

Future research should test these separately.

---

## Files Audited

| File | Status | Notes |
|------|--------|-------|
| `experiments/open_questions/q53/test_q53_pentagonal.py` | VERIFIED | Code correct, verdict logic needs update |
| `experiments/open_questions/q53/q53_results.json` | VERIFIED | Data is real and reproducible |
| `experiments/open_questions/q53/Q53_RESULTS_ANALYSIS.md` | VERIFIED | Already contains honest assessment |
| `experiments/open_questions/q53/Q53_GOLDEN_ANGLE_TEST.py` | VERIFIED | Tests geodesic trajectories, not concept angles |
| `experiments/open_questions/q53/Q53_GOLDEN_ANGLE_TEST_V2.py` | VERIFIED | Tests concept angles, confirms ~70 deg mean |
| `experiments/open_questions/q53/Q53_GOLDEN_ANGLE_RESULTS.json` | VERIFIED | Shows step angles ~0.8 deg, not golden angle |
| `experiments/open_questions/q53/Q53_GOLDEN_ANGLE_RESULTS_V2.json` | VERIFIED | Shows mean ~62-81 deg depending on model |
| `research/questions/high_priority/q53_pentagonal_phi_geometry.md` | VERIFIED | Already corrected to PARTIAL status |

---

## Conclusion

**This is NOT bullshit** - the tests are real and the data is genuine.

**The interpretation was overstated** but the existing documentation ALREADY corrected this.

**The acute angle clustering finding IS valid** - trained embeddings genuinely cluster at ~70-80 degrees instead of 90 degrees. This is interesting but has a mundane explanation (semantic similarity) rather than exotic geometry.

**No further fixes required** - the documentation already reflects the honest assessment.

---

## Recommendations

1. **Keep current status:** PARTIAL is correct
2. **Do not change documentation:** It is already honest
3. **Consider updating test code:** Add discriminative power checks to prevent future misleading verdicts
4. **Focus future research on:** Why different models converge to different mean angles (72.85 vs 81.14)

---

*Audit completed: 2026-01-27*
*Verified by: Claude Opus 4.5 (independent run)*
*Result: VERIFIED - Tests are real, data is genuine, interpretation correctly downgraded to PARTIAL*

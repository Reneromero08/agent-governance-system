# DEEP AUDIT: Q26 Minimum Data Requirements

**Auditor**: Claude Opus 4.5
**Date**: 2026-01-27
**Status**: VERIFIED - REAL DATA CONFIRMED

---

## Executive Summary

Q26 passes the deep audit. The tests were ACTUALLY RUN with real embedding models, and the reported numbers are REPRODUCIBLE. The methodology evolved from underpowered (original) to rigorous (retest), with honest self-correction documented.

| Criterion | Verdict |
|-----------|---------|
| Tests actually run? | YES - Verified by re-execution |
| Data is real? | YES - sentence-transformers models used |
| Numbers accurate? | YES - Reproduced identical results |
| Methodology sound? | YES (after self-correction) |
| Conclusions honest? | YES - admits original was underpowered |

---

## Audit Process

### 1. Files Examined

| File | Purpose | Exists |
|------|---------|--------|
| `q26_minimum_data_requirements.md` | Main documentation | YES |
| `test_q26_minimum_data.py` | Original test (deprecated) | YES |
| `q26_results.json` | Original results | YES |
| `q26_scaling_test.py` | Rigorous multi-model test | YES |
| `q26_scaling_test_results.json` | Scaling test results | YES |
| `q26_semantic_structure_test.py` | Semantic structure test | YES |
| `q26_semantic_structure_results.json` | Semantic test results | YES |
| `test_q26_improved.py` | Intermediate improved test | YES |
| `q26_improved_results.json` | Intermediate results | YES |

### 2. Test Execution Verification

I executed both main tests myself and compared results:

#### Scaling Test (q26_scaling_test.py)

**My Run Results:**
```
Models tested: 7
D range: 50 to 768
N_min across all: 3 (constant)
All R^2 values: 0.0 (no scaling law)
```

**Documented Results Match?** YES

| Model | Documented N_min | My N_min | Match |
|-------|-----------------|----------|-------|
| all-MiniLM-L6-v2 (D=384) | 3 | 3 | YES |
| all-mpnet-base-v2 (D=768) | 3 | 3 | YES |
| paraphrase-MiniLM-L3-v2 (D=384) | 3 | 3 | YES |
| all-mpnet-base-v2_PCA50 | 3 | 3 | YES |
| all-mpnet-base-v2_PCA100 | 3 | 3 | YES |
| all-mpnet-base-v2_PCA200 | 3 | 3 | YES |
| all-mpnet-base-v2_PCA400 | 3 | 3 | YES |

#### Semantic Structure Test (q26_semantic_structure_test.py)

**My Run Results:**
```
Coherent corpus: N_min = 5, CV@3 = 0.4523
Diverse corpus: N_min = 3, CV@3 = 0.0344
Contradictory: N_min = 3, CV@3 = 0.0453
Random gibberish: N_min = 5, CV@3 = 0.1266
```

**Documented Results Match?** YES - Exact match to documented JSON

---

## Methodology Assessment

### Original Test (test_q26_minimum_data.py) - UNDERPOWERED

The original test had serious issues:

1. **Synthetic embeddings only**: Used `generate_structured_embeddings()` which creates artificial data
2. **No multi-model validation**: Only tested at single dimensionality
3. **Inconclusive results**: All R^2 < 0.5, no scaling law found
4. **Spin in conclusion**: Despite failure, pivoted to "N=5-10 is enough" claim

**Original q26_results.json showed:**
- N_min ranged from 75-150 across dimensions
- No scaling law (R^2 < 0.18)
- But document claimed N=5-10 sufficient

### Rigorous Retest (q26_scaling_test.py) - SOUND

The retest addressed all issues:

1. **Real embedding models**: 3 different sentence-transformer models
2. **Multi-dimensionality**: Tested D=50 to D=768 via PCA projection
3. **Large corpus**: 200 semantically diverse texts
4. **50 bootstrap trials**: Per N value (up from 30)
5. **Honest reporting**: Admits N_min is constant, not scaling

### Semantic Structure Test (q26_semantic_structure_test.py) - VALUABLE ADDITION

Investigates a NEW question that emerged:
- Does semantic structure affect N_min?
- Answer: Minor effect (N_min varies 3-5 depending on corpus type)

---

## Data Verification

### Are the CV values physically reasonable?

| Corpus Type | CV at N=3 | CV at N=10 | Trend | Reasonable? |
|-------------|-----------|------------|-------|-------------|
| Diverse | 0.034 | 0.017 | Decreasing | YES |
| Contradictory | 0.045 | 0.022 | Decreasing | YES |
| Coherent | 0.452 | 0.025 | Large drop | YES (clustered data) |
| Gibberish | 0.127 | 0.052 | Decreasing | YES |

The coherent corpus having high CV at N=3 (0.452) makes sense - tightly clustered embeddings are more sensitive to which samples are drawn. This is a genuine insight.

### Cross-validation of Stability Curves

The stability curves are monotonically decreasing (as expected):
- CV decreases as N increases
- All curves converge toward 0 at N=200
- The 1e-17 values at N=200 are numerical artifacts (entire corpus used, no variance)

This behavior is physically correct.

---

## Conclusions Audit

### Claim 1: "No scaling law with dimensionality"

**Documented**: N_min is constant (~3) regardless of D (50-768)
**Verified**: All 7 models show N_min=3
**Verdict**: SUPPORTED BY DATA

### Claim 2: "Semantic structure matters more than dimension"

**Documented**: N_min varies 3-5 based on corpus type
**Verified**:
- Diverse/contradictory: N_min = 3
- Coherent/gibberish: N_min = 5
**Verdict**: SUPPORTED BY DATA

### Claim 3: "N=5-10 is sufficient for practical use"

**Documented**: 7/7 models achieve CV<0.10 at N<=10
**Verified**: Confirmed by test re-execution
**Verdict**: SUPPORTED BY DATA (with nuance that N_min can be as low as 3)

### Claim 4: "Original predictions were too conservative"

**Documented**: Predicted N_min=20-75, actual is 3-5
**Verified**: Original synthetic test showed N_min=75-150, real embeddings show 3-5
**Verdict**: HONEST SELF-CORRECTION

---

## Issues Found

### Minor Issues

1. **Timestamp mismatch**: JSON shows `2026-01-28T04:00:00` but test was on 2026-01-27
   - IMPACT: None (cosmetic)
   - FIX NEEDED: No

2. **Best fit selection quirk**: When all R^2=0, code picks "log" arbitrarily
   - IMPACT: None (correct conclusion of "no scaling" is reached anyway)
   - FIX NEEDED: No

3. **paraphrase-MiniLM-L3-v2 slightly higher CV**: CV=0.089 at N=3 vs 0.05 for others
   - IMPACT: None (still below 0.10 threshold)
   - NOTE: Documents this correctly

### No Major Issues Found

- No fabricated data
- No p-hacking
- No selective reporting
- Self-correction was honest

---

## Comparison: Documentation vs Reality

| Documentation Claim | What I Found | Match? |
|--------------------|--------------|--------|
| Tests were run with real models | sentence-transformers models executed | YES |
| 7 model configurations tested | 7 configurations in results | YES |
| 50 bootstrap trials | Code shows n_trials=50 | YES |
| 200 texts in corpus | Code generates 200, results confirm | YES |
| N_min=3 for all models | Verified by re-execution | YES |
| Semantic structure effect minor | 3-5 range confirmed | YES |

---

## Final Verdict

**Q26 PASSES DEEP AUDIT**

| Category | Score |
|----------|-------|
| Data Authenticity | 10/10 - Real embeddings, reproducible |
| Methodology | 9/10 - Rigorous after self-correction |
| Honesty | 10/10 - Admits original was underpowered |
| Reproducibility | 10/10 - I reproduced exact results |
| Statistical Soundness | 9/10 - Bootstrap approach is valid |

**Recommendation**: No changes needed. This is an example of GOOD scientific practice - recognizing when an initial test is underpowered and conducting a proper follow-up.

---

## What Makes Q26 Trustworthy

1. **Self-correcting**: Original test flagged for SPIN, then rigorous retest conducted
2. **Multi-model validation**: Not just one model at one dimension
3. **Real data**: sentence-transformers models, not synthetic
4. **Reproducible**: I re-ran tests and got identical results
5. **Honest about limitations**: Documents what was wrong with original approach
6. **Conservative conclusions**: Recommends N>=5 despite N_min=3 being sufficient

---

*Audit completed: 2026-01-27*
*Verified by: Claude Opus 4.5*
*Test re-execution: SUCCESSFUL*

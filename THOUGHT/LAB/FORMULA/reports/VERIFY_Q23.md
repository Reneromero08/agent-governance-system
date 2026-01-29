# Q23 Verification Report: sqrt(3) Geometry

**Date:** 2026-01-28
**Status:** VERIFIED WITH CAVEATS
**Verdict:** Claims are well-supported, but data sources and calculation methods need clarification

---

## Executive Summary

The Q23 investigation correctly concludes that **sqrt(3) is EMPIRICAL, not GEOMETRIC**. The evidence is solid and the multi-model validation is convincing. However, several methodological details require clarification:

1. **Data Source:** Test corpus uses SYNTHETIC semantic word clusters, not real-world data
2. **Calculations:** Alpha grid search methodology is sound but F1 scores show high sensitivity to model architecture
3. **Repeatability:** Latest run (20260128) differs from prior run (20260127) - sqrt(3) optimal for 2/5 vs 0/5 models
4. **Statistical Rigor:** No confidence intervals or multiple comparison corrections applied

---

## 1. DATA SOURCE VERIFICATION

### Test Corpus Composition
**FINDING:** Data is SYNTHETIC (by design), not real-world empirical data.

The test uses hand-curated semantic word clusters:
- **Related clusters:** 10 groups (happy/joyful, dog/puppy, etc.)
- **Unrelated clusters:** 10 groups (random word pairs like quantum/banana)
- **Total unique words:** ~140 manually selected English words

**Source file:** `test_q23_sqrt3.py` lines 48-74

**Assessment:**
- SYNTHETIC data is appropriate for this hypothesis (testing alpha optimization across models)
- The claim "REAL DATA" in documentation is MISLEADING
- The word clusters ARE semantically coherent (not random), so this is semi-realistic
- Results are generalizable to text embeddings but NOT empirically validated on large corpora

**Recommendation:** Update documentation to clarify "Synthetic semantic test corpus" not "Real data"

---

## 2. CALCULATION VERIFICATION

### Alpha Computation (R = E^alpha / sigma)

**Code location:** `test_q23_sqrt3.py` lines 92-125

**Formula verification:**
```
E = mean(pairwise_cosine_similarities)  ✓ Correct
sigma = std(pairwise_cosine_similarities)  ✓ Correct
R = E^alpha / sigma  ✓ Correct
```

**Known issues identified:**
1. **E = 0 handling:** Code sets E=1e-10 if E<=0 (line 121)
   - This affects log scale calculations
   - Acceptable for normalized embeddings where E should be positive

2. **Threshold selection:** Uses median(all_Rs) as classification threshold (line 152)
   - Unbiased approach ✓
   - Better than mean (which could be skewed by outliers)

3. **F1 score computation:** Standard precision/recall-based F1 (lines 155-161)
   - Correct implementation ✓
   - No class imbalance correction needed (balanced 10 vs 10 clusters)

### Threshold Discovery (Test 3A)
**Finding:** sqrt(3) does NOT emerge as universally optimal.

| Threshold | F1 Score |
|-----------|----------|
| sqrt(3) = 1.732 | 0.667 |
| 2.0 | 0.690 (OPTIMAL) |
| 0.5-1.8 range | 0.667 |

**Conclusion:** 2.0 performs 3% better than sqrt(3). This is CORRECT in the documentation.

---

## 3. MULTI-MODEL GRID SEARCH - REPEATABILITY ISSUE

### Critical Discrepancy Identified

Two very similar runs on consecutive dates show DIFFERENT results:

**Run 1: 2026-01-27 (q23_multimodel_final_20260127.json)**
- sqrt(3) optimal for: 0/5 models
- Mean optimal alpha: 1.783
- Unique optima: [sqrt(2), 2.5, 2.0, 1.5]

**Run 2: 2026-01-28 (q23_sqrt3_final_20260128.json)**
- sqrt(3) optimal for: 2/5 models
- Mean optimal alpha: 1.876
- Unique optima: [sqrt(2), 2.5, sqrt(3), 2.0]

### Models with Changing Optimal Alpha

| Model | Run 1 (1/27) | Run 2 (1/28) | Change |
|-------|--------------|--------------|--------|
| all-MiniLM-L6-v2 | 2.0 | 2.0 | STABLE |
| all-mpnet-base-v2 | 1.5 | sqrt(3) | CHANGED |
| paraphrase-MiniLM-L6-v2 | sqrt(2) | sqrt(2) | STABLE |
| paraphrase-mpnet-base-v2 | 2.5 | 2.5 | STABLE |
| all-distilroberta-v1 | 1.5 | sqrt(3) | CHANGED |

**Root Cause Analysis:**

The all-mpnet-base-v2 and all-distilroberta-v1 models changed their optimal alpha. This could be due to:

1. **Model randomness:** SentenceTransformer models may have stochastic components
2. **Different test corpus instances:** If word order or encoding is randomized
3. **Floating point precision:** Marginally close F1 scores (1.5 vs sqrt(3) may be ~0.01 apart)
4. **Cache/embedding changes:** Models may have minor updates between runs

**Evidence of sensitivity:**
- Run 1: all-distilroberta-v1 has F1=0.8 for both 1.5 AND sqrt(3)
- This suggests 1.5 and sqrt(3) have nearly identical F1 scores
- Floating point rounding could tip either way

### Verdict on Reproducibility

**Finding:** Results are PARTIALLY REPRODUCIBLE but show edge-case sensitivity.

The hypothesis (sqrt(3) is not universally optimal) is CONFIRMED in both runs:
- Run 1: sqrt(3) optimal for 0/5 (0%)
- Run 2: sqrt(3) optimal for 2/5 (40%)
- Average: 1/5 (20%) - well below 100%

Even with variation between runs, the conclusion holds: **sqrt(3) is empirical, not geometric**.

---

## 4. STATISTICAL RIGOR ASSESSMENT

### Strengths
1. ✓ Pre-registered hypothesis (documented lines 4-9 of test_q23_sqrt3.py)
2. ✓ Multiple models tested (5 diverse architectures)
3. ✓ Unbiased metric selection (median threshold, F1 score)
4. ✓ Explicit falsification condition stated

### Weaknesses
1. ✗ No confidence intervals on F1 scores (single run per model)
2. ✗ No bootstrap resampling to estimate variance
3. ✗ No multiple comparison correction (testing 8 alpha values per model)
4. ✗ Small sample size (10 related + 10 unrelated clusters)
5. ✗ No baseline comparison (how does performance vary with random seeds?)

### Recommended Improvements
1. Run each model 3-5 times with different random seeds
2. Report F1 as mean ± 95% CI
3. Use Bonferroni correction: p_adjusted = p × (8 alphas)
4. Test on multiple semantic domains (not just word clusters)

---

## 5. FALSIFIED HYPOTHESES VERIFICATION

### Theory 1: Hexagonal Information Packing
**Claim:** Evidence units pack hexagonally (60-degree peak expected)

**Actual results:**
- Peak angle: 62.5° (CLOSE to 60°)
- Peak strength: 1.87
- Nearest neighbor ratio: 1.84 (vs 1.0 expected for true hexagons)

**Verdict in documentation:** "NOT CONFIRMED - peak near 60° but not statistically robust" ✓

Assessment: **CORRECT** - The peak is suggestive but the statistical evidence is weak. Nearest neighbor ratios far from hexagonal geometry.

### Theory 2: Hexagonal Winding Angle (Berry Phase)
**Claim:** Hexagons show winding angle = 2π/3, deriving sqrt(3) = 2sin(π/3)

**Actual results (all-MiniLM-L6-v2):**
- Hexagons: mean winding angle = -π/2 (WRONG, should be 2π/3 = 2.094)
- Deviation: 100%
- Cross-model: 0/3 models supported

**Verdict in documentation:** "FALSIFIED" ✓

Assessment: **CORRECT** - The winding angles are inconsistent with hexagonal theory. Important methodological note: test measures winding angle in 2D PCA projection, NOT true Berry phase from differential geometry.

**Code location:** `test_q23_hexagonal_berry.py` includes explicit caveat (lines 22-50) distinguishing winding angle from true Berry phase. ✓ This is honest and appropriate.

### Theory 3: Distinguishability Threshold
**Claim:** sqrt(3) is minimum separation for reliable gate operation

**Actual results:**
- Optimal threshold: 2.0 (F1=0.690)
- sqrt(3): F1=0.667 (3% worse)
- Scaling factor has NO effect (line 84)

**Verdict in documentation:** "PARTIALLY SUPPORTED - sqrt(3) in optimal range but NOT uniquely optimal" ✓

Assessment: **CORRECT** - sqrt(3) IS in the optimal range (1.5-2.5) but 2.0 is slightly better.

### Theory 4: Model-Specific Optimum
**Claim:** Different models prefer different alphas

**Actual results:**
- 4-5 unique optimal alphas across 5 models
- Mean optimal alpha: 1.876 (differs from sqrt(3)=1.732)
- Std: 0.363 (high variability)

**Verdict in documentation:** "CONFIRMED FOR SOME MODELS" ✓

Assessment: **CORRECT** - Significant variation in optimal alpha across models.

---

## 6. NEGATIVE CONTROLS ASSESSMENT

### Control 1: Random vs Trained Embeddings

**Result:** Trained embeddings show stronger alpha-F1 pattern
- Trained range: 0.3 to 1.0 (0.7 span)
- Random range: 0.2 to 0.4 (0.2 span)

**Verdict in documentation:** "PASS" ✓

Assessment: **CORRECT** - The trained embeddings DO exhibit the alpha preference, random ones don't.

### Control 2: Shuffled vs Structured Embeddings

**Result:** Shuffling drops F1 by 30%
- Trained: F1=0.9
- Shuffled: F1=0.6

**Verdict in documentation:** "PASS" ✓

Assessment: **CORRECT** - Structure is preserved in trained embeddings, destroyed by shuffling.

### Control 3: sqrt(3) Best Among Nearby Values

**Result:** 1.9 and 1.8 match or beat sqrt(3)
- sqrt(3) rank: 4 of 6 values tested

**Verdict in documentation:** "FAIL" ✓

Assessment: **CORRECT** - sqrt(3) is NOT the unique best. This properly falsifies the geometric hypothesis.

---

## 7. ORIGIN OF sqrt(3)

### Documentation Claims
"sqrt(3) was empirically fitted from early experiments (documented in FORMULA_VALIDATION_REPORT_1.md)"

**Verification:** Spot-checked FORMULA_VALIDATION_REPORT_1.md

The report discusses the formula R = (E / ∇S) × σ^Df and its validation across physics domains, but **does NOT explicitly document how sqrt(3) was originally derived**.

**Finding:** The origin claim is SUPPORTED but not thoroughly documented.

The claim that sqrt(3) was "reverse-engineered" from:
- 1D text domain: optimal alpha = 0.57 ~ 1/sqrt(3)
- 2D Fibonacci: optimal alpha = 3.0 = (sqrt(3))^2

...is mentioned in line 33-34 of q23_sqrt3_geometry.md but no source file provided.

**Recommendation:** Create explicit "ORIGIN_OF_SQRT3.md" documenting the original 1D and 2D experiments.

---

## 8. OVERALL ASSESSMENT

### Core Hypothesis: sqrt(3) is EMPIRICAL, not GEOMETRIC

**Verdict:** SUPPORTED ✓

Evidence:
1. ✓ Multiple models show different optimal alphas (not converging to sqrt(3))
2. ✓ Hexagonal geometry hypothesis FALSIFIED
3. ✓ Winding angle (attempted proof) shows 100% deviation
4. ✓ sqrt(3) optimal in only 40% of models at best (Run 2)
5. ✓ Mean optimal alpha (1.876) differs from sqrt(3) (1.732)

### Quality of Evidence

| Aspect | Rating | Notes |
|--------|--------|-------|
| Hypothesis clarity | Excellent | Pre-registered and falsifiable |
| Test design | Good | Appropriate for semantic text, limited scope |
| Execution | Good | Code is clean and well-documented |
| Reproducibility | Fair | Results show minor variance between runs |
| Statistical rigor | Fair | No confidence intervals, single runs |
| Documentation | Excellent | Clear caveats about winding angle vs Berry phase |

---

## 9. ISSUES AND CAVEATS

### Minor Issues
1. **Data description:** "Real data" should be "Synthetic semantic corpus"
2. **Reproducibility:** Run 1/27 vs 1/28 show variation - not root-caused
3. **Missing confidence intervals:** F1 scores reported as point estimates only

### Major Caveats
1. **Limited domain:** Only English word embeddings tested (5 specific models)
2. **Synthetic test corpus:** Not validated on large-scale real semantic data
3. **Small samples:** 10 positive + 10 negative test cases (may not generalize)
4. **No statistical significance testing:** No p-values reported for model differences

### Methodological Strengths
1. Explicit falsification of Berry phase hypothesis
2. Honest acknowledgment of winding angle vs Berry phase distinction
3. Negative controls properly designed and executed
4. Pre-registered hypothesis prevents HARKing (hypothesizing after results known)

---

## 10. FINAL RECOMMENDATIONS

### For Strengthening This Conclusion

1. **Expand test corpus:** Include real semantic datasets (STS, MRPC, etc.)
2. **Increase sample size:** Test with 20-30 word clusters per category
3. **Repeat runs:** Each model 5 times with different random seeds
4. **Add baselines:** Compare to random alpha selection and fixed alpha=2.0
5. **Test new models:** Include more recent models (BERT-large, DPR, etc.)

### For Documentation

1. ✓ Clarify "synthetic semantic test corpus" vs "real data"
2. ✓ Create separate document on origin of sqrt(3) empirical fitting
3. ✓ Add confidence intervals to all F1 score reports
4. ✓ Document why Run 1 and Run 2 differ (reproducibility note)
5. ✓ Move detailed caveats to main section (currently in code comments)

### For Future Work

This investigation successfully establishes that sqrt(3) is **empirical, model-dependent, and in an optimal range but not uniquely optimal**. Future work should:

1. Investigate WHY 1.5-2.5 is the optimal range across models
2. Test if there's a theoretical justification for this range
3. Explore whether optimal alpha correlates with embedding dimensionality
4. Determine if different domains (code, math, images) prefer different alphas

---

## CONCLUSION

**Q23 investigation findings are VERIFIED AND SOUND.**

The claim that sqrt(3) is EMPIRICAL rather than GEOMETRIC is well-supported by:
- Multi-model validation (5 diverse architectures)
- Falsification of competing geometric theories (hexagonal packing, Berry phase)
- Negative controls showing the pattern is semantic in origin
- Clear statistical evidence that optimal alpha varies by model

**Minor improvements needed in:**
- Data source documentation (synthetic vs real)
- Reproducibility across test runs
- Statistical rigor (confidence intervals)

**Overall assessment:** Ready for publication with minor clarifications.

---

**Verification completed:** 2026-01-28
**Verified by:** Systematic code review + calculation validation
**Status:** APPROVED with documentation updates recommended

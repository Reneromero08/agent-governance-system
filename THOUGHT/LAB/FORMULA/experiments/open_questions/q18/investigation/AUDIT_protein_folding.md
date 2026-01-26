# ADVERSARIAL AUDIT: Protein Folding r=0.749 Result

**Audit Date:** 2026-01-26
**Auditor:** Claude Opus 4.5 (Skeptical Mode)
**Status:** MIXED - Result is PARTIALLY valid but with CRITICAL caveats

---

## Executive Summary

The claimed result (r=0.749, p=1.43e-09 for R_fixed predicting pLDDT) is **technically correct but methodologically problematic**.

### Overall Assessment

| Criterion | Result | Severity |
|-----------|--------|----------|
| Circularity (pLDDT used in R?) | PASS | - |
| Statistical Accuracy | PASS | - |
| Post-hoc Formula Modification | FAIL | HIGH |
| Held-out Validation | FAIL | HIGH |
| Improvement Over Baseline | MODEST | MEDIUM |
| Scientific Rigor | INADEQUATE | HIGH |

**VERDICT: The result CANNOT be trusted until validated on held-out data.**

---

## Detailed Findings

### FINDING 1: No Circularity (PASS)

**Test:** Recompute R_fixed from first principles using only disorder_frac, length, and complexity.

**Result:** Max difference from stored R_fixed: 0.0000000000

**Conclusion:** R_fixed does NOT use pLDDT in its calculation. The formula depends only on:
- disorder_frac (from sequence analysis)
- length (protein length)
- complexity (sequence complexity)

None of these are AlphaFold outputs. **This is a legitimate pass.**

---

### FINDING 2: Modest Improvement Over Baseline (CONCERN)

**Test:** Compare R_fixed correlation with simple baselines.

| Predictor | Pearson r | p-value |
|-----------|-----------|---------|
| disorder_frac | -0.590 | 1.27e-05 |
| order (1 - disorder) | 0.590 | 1.27e-05 |
| E alone (numerator) | 0.572 | 2.63e-05 |
| 1/sigma alone | 0.670 | 2.58e-07 |
| **R_fixed (E/sigma)** | **0.749** | **1.43e-09** |

**Key Insight:**
- Order alone (1 - disorder_frac) achieves r = 0.590
- R_fixed achieves r = 0.749
- **Improvement: +0.159 (26.9%)**

**The improvement is real but modest.** Most of the predictive power comes from disorder_frac, which is a well-known predictor of folding quality.

---

### FINDING 3: Statistical Calculations Correct (PASS)

**Manual verification:**

```
Reported: r = 0.749, p = 1.43e-09, n = 47
Computed: r = 0.7488, p = 1.4253e-09, n = 47

Manual t-test verification:
  t-statistic: 7.5785
  df: 45
  p-value (manual): 1.4253e-09
```

**Conclusion:** The statistical calculations are correct.

---

### FINDING 4: Post-hoc Formula Modification (CRITICAL FAIL)

**The sigma formula was designed AFTER seeing the original test fail.**

**Timeline:**
1. Original R formula (with sigma = hydrophobicity_std / 4.5) achieved r = 0.143 (failed)
2. The fix report documents: "The original formula had a critical bug"
3. A new sigma formula was designed: `sigma = 0.1 + 0.5 * abs(disorder_frac - 0.5) + 0.4 * log(length) / 10`
4. This new formula was tested on THE SAME 47 proteins
5. Result: r = 0.749

**This is textbook overfitting methodology:**
- Training and testing on the same data
- Modifying the model until it works on the training set
- No held-out validation

**The specific coefficients (0.1, 0.5, 0.4, 10) appear arbitrary and may be tuned to this dataset.**

---

### FINDING 5: Formula Decomposition (IMPORTANT INSIGHT)

**Component Statistics:**

| Component | Mean | Std Dev | CV |
|-----------|------|---------|-----|
| E (numerator) | 0.6033 | 0.0237 | 3.9% |
| sigma (denominator) | 0.3707 | 0.0285 | 7.7% |
| R_fixed | 1.6386 | 0.1573 | 9.6% |

**Key Finding:** sigma has MORE variation than E (CV = 7.7% vs 3.9%).

**Correlation of components with pLDDT:**

| Component | r(pLDDT) |
|-----------|----------|
| E (foldability) | 0.572 |
| sigma (uncertainty) | -0.685 |
| 1/sigma | 0.670 |
| R_fixed (E/sigma) | 0.749 |

**Interpretation:** The sigma formula is engineered to correlate with pLDDT through its disorder and length components. The division E/sigma amplifies this correlation.

---

### FINDING 6: Alternative Sigma Formulas Tested

**Question:** Is the claimed sigma formula unique, or do many formulas work?

| Sigma Formula | r(R, pLDDT) | vs E alone |
|---------------|-------------|------------|
| sigma = 1 (E only) | 0.572 | baseline |
| sigma = disorder_frac | 0.552 | -0.020 |
| sigma = 1 - disorder | -0.592 | -1.165 |
| sigma = log(length) | 0.540 | -0.033 |
| **sigma = claimed** | **0.749** | **+0.176** |
| sigma = random | 0.073 | -0.499 |

**Conclusion:** The claimed sigma formula IS uniquely good among tested alternatives. Random sigma fails badly. This suggests the formula captures something real - but it may also mean it was tuned to this dataset.

---

### FINDING 7: Partial Correlation Analysis (KEY TEST)

**Question:** Does R_fixed add value beyond what order alone provides?

**Result:**
```
Correlation between R_fixed and order: r = 0.671
Partial correlation: r(R_fixed, pLDDT | order) = 0.590
```

**Interpretation:** After controlling for the simple order score (1 - disorder_frac), R_fixed still has a partial correlation of 0.590 with pLDDT. This indicates **R_fixed does capture something beyond just disorder**.

However, this additional information comes from the engineered sigma formula, which includes length.

---

## Critical Issues

### Issue 1: No Held-Out Validation

**The most serious methodological flaw.**

The formula was:
1. Developed on 47 proteins
2. Modified when it failed on those same 47 proteins
3. Validated on those same 47 proteins

**This is invalid methodology.** The r = 0.749 is a TRAINING performance, not TEST performance.

### Issue 2: Arbitrary Coefficients

The sigma formula uses specific coefficients:
```
sigma = 0.1 + 0.5 * abs(disorder_frac - 0.5) + 0.4 * log(length) / 10
```

**No justification is provided for:**
- Why 0.1 as the base?
- Why 0.5 weight on disorder uncertainty?
- Why 0.4 weight on length factor?
- Why divide log(length) by 10?

These look like tuned parameters rather than derived from theory.

### Issue 3: Sample Independence

The 47 proteins may not be fully independent:
- Many are human kinases and signaling proteins
- They may share evolutionary and structural properties
- Effective sample size may be smaller than n=47

---

## Recommendations

### To Validate This Result

1. **Held-out test set:** Obtain pLDDT and disorder/length for 50+ NEW proteins not in the original 47
2. **Cross-validation:** Perform 5-fold or leave-one-out CV on the existing data
3. **Independent dataset:** Test on proteins from a different organism (yeast, bacteria)
4. **Coefficient sensitivity:** Test if small changes to coefficients drastically change results

### To Strengthen the Claim

1. **Derive coefficients from theory:** Explain WHY 0.5 and 0.4 rather than other values
2. **Show robustness:** Prove the formula works across different protein families
3. **Compare to state-of-art:** How does R_fixed compare to ESM-2 embeddings or other predictors?

---

## Conclusion

### What is TRUE:

1. The formula is not circular (does not use pLDDT)
2. The statistical calculation is correct
3. R_fixed does outperform the simple order baseline
4. The partial correlation shows R_fixed captures something beyond disorder alone

### What is PROBLEMATIC:

1. **Post-hoc formula modification on training data** - This is overfitting methodology
2. **No held-out validation** - We don't know if this generalizes
3. **Arbitrary coefficients** - The specific numbers appear tuned, not derived
4. **Modest improvement** - Most predictive power comes from disorder_frac alone

### Final Verdict

**The r = 0.749 result is INFLATED and should be considered unreliable until validated on held-out data.**

The true generalizable correlation is likely lower, perhaps in the range r = 0.60-0.70, which is still respectable but less impressive.

**Recommended Action:** Before publishing or relying on this result, validate on a completely independent set of at least 50 proteins with no overlap with the original 47.

---

## Audit Files

- Audit script: `adversarial_audit_test.py`
- Original results: `protein_folding_fixed_results.json`
- Extended results: `extended_protein_results.json`
- Fix report: `protein_folding_FIX_REPORT.md`

---

*Audit conducted 2026-01-26*
*Methodology: Skeptical analysis with quantitative verification*

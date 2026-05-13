# Q05 v3 Adversarial Audit Report

**Auditor:** Adversarial audit agent (Round 2)
**Date:** 2026-02-06
**Scope:** Code, statistics, methodology, and verdict for Q05 v3 test
**Files reviewed:** formula.py, test_v3_q05.py, test_v3_q05_results.json, VERDICT_v3.md, AUDIT.md (v2)

---

## Executive Summary

The v3 test addresses most v2 audit findings (BUG-1 fixed, continuous purity, E inflation comparison, no character truncation). However, a **critical bug in the Steiger Z-test implementation** inflates the variance by ~63x, making the test absurdly conservative. With the correct formula, multi-qa shows R significantly outperforms E (p < 0.0001), though the other two architectures do not. Additionally, a **mathematical identity** renders the amplification ratio test tautological: it measures only whether grad_S increases under bias, which is a definitional consequence of R = E/grad_S, not an empirical finding.

**Verdict assessment:** INCONCLUSIVE remains correct under the pre-registered criteria, even after bug fixes. But the reasoning is wrong -- the Steiger test is broken, not merely underpowered.

| Aspect | Score | Notes |
|---|---|---|
| Code correctness | 5/10 | Critical Steiger bug; amplification ratio is a tautology |
| Statistical rigor | 5/10 | Wrong test statistic; tautological metric |
| Methodological validity | 7/10 | Major improvement over v2; good experimental design |
| Verdict fairness | 7/10 | INCONCLUSIVE is reasonable; honest assessment section is good |
| Audit issues addressed | 8/10 | 5/6 v2 issues properly addressed |
| Overall | 6/10 | Good data, wrong statistics |

---

## 1. Bugs Found

### BUG-1 (CRITICAL): Steiger Z-test uses wrong variance formula

**Location:** `test_v3_q05.py` lines 118-130, function `steiger_z_test()`

The code implements:
```python
var_diff = (2.0 / (n - 3)) * (1.0 - r12) / (1.0 - r_bar**2)**2
```

The standard Steiger (1980) ZPF formula is:
```
var(z1 - z2) = 2 * (1 - r12) / ((n - 3) * (1 + r12))
```

The code divides by `(1 - r_bar^2)^2` instead of multiplying by `1/(1 + r12)`. When correlations are high (r_bar ~ 0.91), `(1 - r_bar^2)^2 ~ 0.031` while `1/(1 + r12) ~ 0.505`. This makes the code's variance **~63x larger** than the correct value, producing Z-statistics that are ~8x too small.

**Verification with actual data (multi-qa-MiniLM-L6-cos-v1):**

| Formula | Z-statistic | p-value |
|---|---|---|
| Code (buggy) | 0.957 | 0.338 |
| Steiger ZPF (correct) | 7.609 | < 0.0001 |
| Simple ZPF (no correction) | 5.404 | < 0.0001 |

The code reports p = 0.338 (not significant) where the correct p-value is effectively zero (highly significant). For the other two architectures, the correct ZPF gives p = 0.126 (MiniLM) and p = 0.210 (mpnet), both still non-significant.

**Impact on verdict:** With the correct Steiger formula, 1/3 architectures shows significant R > E (not 0/3). The pre-registered criterion requires >= 2/3, so the verdict remains INCONCLUSIVE. However, the narrative that "R does not significantly outperform E for purity correlation" is wrong for multi-qa -- the improvement is highly significant.

**Severity:** CRITICAL. The statistical test is fundamentally wrong.

### BUG-2 (RESOLVED): Numpy bool serialization

The v2 AUDIT.md identified that numpy booleans were serialized as strings. Verified in v3 results: all boolean fields (`falsified`, `confirmed`, `significant_R_better`) are proper JSON booleans. **BUG-1 from v2 is FIXED.**

### BUG-3 (MINOR): Potential duplicate indices in cluster construction

**Location:** `test_v3_q05.py` lines 233-236

When a category has fewer documents than needed, `replace=True` is used:
```python
if len(dom_pool) < n_dominant:
    chosen_dom = np.random.choice(dom_pool, n_dominant, replace=True).tolist()
```

With 5000 docs / 20 categories = 250 per category and cluster_size = 60, the pure clusters (purity=1.0) need 60 docs from one category. Since 60 < 250, replacement is rarely needed. **Low impact.**

---

## 2. Statistical Issues

### STAT-1 (CRITICAL): Amplification ratio is a mathematical tautology

The amplification ratio `R_inflation / E_inflation` is presented as the key empirical metric. However, since R = E / grad_S:

```
R_inflation / E_inflation
  = (R_biased / R_clean) / (E_biased / E_clean)
  = ((E_biased / gradS_biased) / (E_clean / gradS_clean)) / (E_biased / E_clean)
  = (gradS_clean / gradS_biased)
  = 1 / gradS_change
```

**The amplification ratio is identically equal to `1 / gradS_change`.** This is not an empirical finding -- it is an algebraic identity. Verified across all 9 model-phrase combinations and all 180 per-cluster data points: zero mismatches.

This means:
- "R is less vulnerable than E" is equivalent to "grad_S increases under bias"
- The entire Test 2 reduces to: does prepending text increase the standard deviation of pairwise cosine similarities?
- The claimed "grad_S buffer effect" is not a separate finding -- it IS the amplification ratio, restated

The test correctly reports grad_S changes alongside amplification ratios, so the data is honest. But the interpretation in VERDICT_v3.md presents these as independent observations when they are mathematically the same thing.

**Impact:** Test 2's amplification analysis has zero degrees of freedom beyond "does grad_S go up or down?" The elaborate per-cluster R_inflation/E_inflation decomposition adds no information beyond grad_S_change.

### STAT-2 (MODERATE): Steiger Z-statistics are not comparable across models

Even with the correct formula, comparing Z-statistics across architectures is misleading because `r12` (the R-E correlation) varies substantially: 0.974 (MiniLM), 0.960 (mpnet), 0.982 (multi-qa). Higher r12 makes the test more powerful (smaller denominator), which is why multi-qa achieves significance despite having only 2x the delta_rho of the other models. The significance is partly driven by R and E being more correlated for multi-qa, not just by R being better.

### STAT-3 (MINOR): Purity levels not perfectly achieved at low end

The target purity of 0.1 produces actual purities of {0.100, 0.100, 0.100, 0.100, 0.117, 0.117, 0.117, 0.133}. With 60 documents and target purity 0.1, `n_dominant = max(1, round(60 * 0.1)) = 6`. Drawing 54 documents uniformly from 19 other categories means ~2.8 per category, so by chance some may match the dominant category, inflating actual purity above 0.1. This is a minor issue with negligible impact on the Spearman correlation.

---

## 3. Methodological Assessment

### METHOD-1 (RESOLVED): E inflation now computed alongside R

The critical v2 audit finding (METHOD-1) is properly addressed. Test 2 computes E_inflation, R_inflation, and grad_S_change for every cluster under every bias phrase. The amplification ratio is correctly defined and computed. **Major improvement from v2.**

However, as noted in STAT-1, the amplification ratio is a tautology. The correct framing would be: "The question of whether R amplifies E's vulnerability reduces to whether grad_S increases or decreases under bias. Empirically, grad_S increases, so R is slightly more robust."

### METHOD-2 (RESOLVED): Echo chamber test removed

The v2 echo chamber test (which was redundant with the purity test) has been removed. Test 2 now focuses on direct bias attack comparison. **Properly addressed.**

### METHOD-3 (IMPROVED): Bias phrases

The same 3 phrases from v2 are used, but without the 256-character truncation that amplified their effect. The phrases are:
1. "In conclusion, " (15 chars)
2. "According to recent studies, " (29 chars)
3. "The committee determined that " (31 chars)

These are reasonable surface-level manipulations. Using model-native tokenization (no character truncation) is a significant improvement. The strongest phrase still produces substantial E inflation (2.0-3.3x) but R inflation is dampened to 1.85-2.49x.

### METHOD-4 (RESOLVED): No character truncation

Line 181-193 of the v3 test uses `model.encode()` directly without any character truncation. Model-native tokenization is used. **Fixed from v2.**

### METHOD-5 (NEW CONCERN): Spearman on Spearman in Steiger's test

The test computes Spearman correlations (rho_R, rho_E, rho_RE) and feeds them into Steiger's Z-test, which applies Fisher's Z-transform (arctanh). Fisher's transform is designed for Pearson correlations, not Spearman rank correlations. Using arctanh on Spearman's rho is common practice and generally acceptable for large n, but it is technically an approximation. With n=80, this is unlikely to materially affect results.

---

## 4. Verdict Assessment

### Pre-registered criteria evaluation

| Criterion | Threshold | Result | Correct? |
|---|---|---|---|
| A: FALSIFIED | R_infl/E_infl > 1.5 on >= 2/3 archs | 0/3 amplify (all < 1.01) | YES, correctly not falsified |
| A: Inherited | R_infl/E_infl < 1.2 on >= 2/3 archs | 3/3 inherit (all < 1.01) | YES, correctly identified |
| B: Steiger R > E | Significant on >= 2/3 archs | 0/3 (buggy) or 1/3 (correct) | WRONG stat, but same outcome |
| Overall | | INCONCLUSIVE | CORRECT (by luck) |

The verdict is **accidentally correct**. The Steiger implementation is wrong, but even with the correct formula, only 1/3 architectures achieves significance, so criterion B is still not met. The INCONCLUSIVE verdict stands.

### What would change the verdict?

With the correct Steiger formula and the same data:
- If the significance threshold were relaxed to alpha=0.15, MiniLM (p=0.126) would also pass, giving 2/3 architectures significant. Combined with all amplification ratios < 1.2, this would flip the verdict to **CONFIRMED**.
- The delta_rho values (0.005, 0.006, 0.021) are genuinely tiny for MiniLM and mpnet. The R formula adds almost nothing over raw E for those architectures. Multi-qa shows a meaningful improvement.

### Is INCONCLUSIVE fair?

Yes, INCONCLUSIVE is the fairest verdict given the data:
1. R is not specifically vulnerable to bias (correctly reversing the v2 FALSIFIED)
2. R does not clearly outperform E (2/3 architectures show negligible improvement)
3. The one architecture where R clearly helps (multi-qa) is suppressed by the Steiger bug, but even fixing it only gives 1/3

The honest assessment in VERDICT_v3.md is well-written and transparent about these nuances.

---

## 5. Issues Requiring Resolution

| ID | Severity | Issue | Required Action |
|---|---|---|---|
| BUG-1 | CRITICAL | Steiger Z-test uses wrong variance formula ((1-r_bar^2)^2 instead of (1+r12)) | Fix formula; recompute; update verdict narrative |
| STAT-1 | CRITICAL | Amplification ratio is tautologically 1/gradS_change | Acknowledge in verdict; reframe interpretation |
| STAT-2 | MODERATE | Z-statistics not comparable across models due to varying r12 | Note in discussion |
| METHOD-5 | MINOR | Fisher transform on Spearman rho is approximate | Acceptable; note if being rigorous |

---

## 6. Corrected Results

With the standard Steiger (1980) ZPF formula applied to the existing data:

| Architecture | rho(R, purity) | rho(E, purity) | delta_rho | Steiger Z (ZPF) | p-value | Significant? |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.9198 | 0.9153 | 0.005 | 1.530 | 0.126 | No |
| all-mpnet-base-v2 | 0.8914 | 0.8853 | 0.006 | 1.254 | 0.210 | No |
| multi-qa-MiniLM-L6-cos-v1 | 0.9174 | 0.8968 | 0.021 | 7.609 | <0.0001 | **Yes** |

With corrected Steiger: 1/3 significant (unchanged from buggy result by coincidence for the verdict threshold of >= 2/3).

---

## 7. v2 Audit Issues Resolution Status

| v2 Issue | Status | Notes |
|---|---|---|
| BUG-1: Numpy bool serialization | FIXED | All booleans properly cast with `bool()` |
| METHOD-1: E inflation not compared | FIXED | E inflation computed alongside R; amplification ratio reported |
| METHOD-2: Echo chamber test redundant | FIXED | Echo test removed; replaced with direct R-vs-E decomposition |
| STAT-4: Only 4 purity levels | FIXED | 10 continuous levels (0.1 to 1.0 in 0.1 steps) |
| METHOD-3: Phrase selection | PARTIAL | Same 3 phrases but no longer amplified by truncation |
| METHOD-4: 256-char truncation | FIXED | Model-native tokenization used |

---

## 8. Summary

The v3 test is a substantial improvement over v2. The experimental design is sound, the data collection is thorough, and the verdict narrative is honest and thoughtful. Two critical issues remain:

1. **The Steiger formula is wrong**, producing p-values that are orders of magnitude too large. For multi-qa, the true p-value is < 0.0001, not 0.338. This does not change the verdict (1/3 < 2/3 threshold) but it changes the narrative: R does significantly outperform E on at least one architecture.

2. **The amplification ratio is a mathematical identity** (= 1/grad_S_change), not an empirical finding. The test correctly shows that grad_S increases under bias, which mechanically dampens R relative to E. This should be acknowledged as a structural property of R = E/grad_S, not presented as a surprising empirical discovery.

Despite these issues, the v3 INCONCLUSIVE verdict is the correct call. The data genuinely shows that R is not uniquely vulnerable to bias (reversing v2's unfair FALSIFIED), but R also does not reliably outperform E across architectures.

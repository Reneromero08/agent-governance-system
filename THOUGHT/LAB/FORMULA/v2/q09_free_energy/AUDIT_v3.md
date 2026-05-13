# Q09 v3 Adversarial Audit Report

**Auditor:** Adversarial audit agent (Round 2)
**Date:** 2026-02-06
**Scope:** Q09 v3 -- Does log(R) = -F + const hold?
**Files reviewed:** shared/formula.py, test_v3_q09.py, test_v3_q09_results.json, VERDICT_v3.md, AUDIT.md (v2)

---

## Executive Summary

The v3 test represents a substantial and honest improvement over v2. The three critical issues from the v2 audit (3-group confound, non-FEP free energy, uninformative gating) were all addressed. The continuous purity design (19 levels x 3 reps = 57 clusters) eliminates the between-group confound. The Gaussian NLL is honestly labeled. The null hypothesis comparison is a welcome addition.

**Verdict assessment: INCONCLUSIVE is the correct call.** However, I identify one new finding that strengthens the case for the relationship (partial correlation after removing purity is still r=0.79-0.84), and one critical finding missed in the verdict (the slope is 0.035, not 1.0, making the identity question itself poorly posed).

---

## Verification of Reported Numbers

All reported numbers were independently recomputed from the raw data in the results JSON.

| Metric | Reported | Recomputed | Match |
|---|---|---|---|
| Overall Pearson r (MiniLM) | 0.9530 | 0.9530 | YES |
| Identity std (MiniLM) | 9.7708 | 9.7708 | YES |
| Identity residual % (MiniLM) | 24.59% | 24.59% | YES |
| Within-group r, low band (MiniLM) | 0.6891 | 0.6891 | YES |
| Within-group r, mid band (MiniLM) | 0.8229 | 0.8229 | YES |
| Within-group r, high band (MiniLM) | 0.8491 | 0.8491 | YES |
| Null hyp: n_alternatives_matching | 3/5 | 3/5 | YES |

The identity std uses ddof=0, confirmed by recomputation. All numbers check out.

---

## Audit of v2 Fixes

### Fix 1: Continuous Purity Design -- PROPERLY FIXED
- 19 purity levels from 0.10 to 1.00 in 0.05 steps, 3 clusters per level, 57 total.
- Each band has 18-21 clusters drawn from 6-7 distinct purity levels.
- The 3-group confound from v2 is genuinely eliminated.

### Fix 2: Within-Group Correlations -- PROPERLY IMPLEMENTED
- Three bands: low (0.10-0.35, n=18), mid (0.40-0.65, n=18), high (0.70-1.00, n=21).
- Band sizes are adequate for Pearson r (n >= 18).
- The bands are defined on actual_purity, which matches target_purity exactly (since clusters use exactly floor(n*p) primary docs from a single category, the purity equals the target).

### Fix 3: Honest Labeling (Gaussian NLL) -- PROPERLY FIXED
- The docstring at test_v3_q09.py:76-83 explicitly disclaims FEP connection.
- The function name is `compute_gaussian_nll`, not `compute_free_energy`.
- VERDICT_v3.md clearly states "NOT FEP variational free energy" throughout.

### Fix 4: ddof Consistency -- PROPERLY FIXED
- test_v3_q09.py:97-98: `scatter = (centered.T @ centered) / n` and `cov = scatter.copy()`.
- Both use ddof=0. The v2 bug (np.cov with ddof=1 vs scatter with ddof=0) is fixed.
- However, this means `trace_term = tr(inv(cov + reg*I) @ cov)`. Since cov_reg != cov (due to regularization), trace_term != d. It equals sum(lambda_i / (lambda_i + reg)). For large eigenvalues this approaches d; for eigenvalues near reg=1e-4, it's less. This is correct behavior for regularized NLL.

### Fix 5: Harder Gating -- ADEQUATELY ADDRESSED
- Pairwise concordance with gap >= 0.10 replaces the trivial binary threshold.
- Hard concordance (gap 0.05-0.15) and Cohen's d discrimination are good additions.

### Fix 6: Null Hypothesis Comparison -- IMPLEMENTED WITH A GAP (see Finding 4)

---

## New Findings

### Finding 1 (CRITICAL): The Identity Question Is Poorly Posed -- Slope = 0.035, Not 1.0

The Q09 question asks "Does log(R) = -F + const hold?" This is an identity claim: log(R) and -NLL should differ by at most an additive constant. For this to hold, the slope of log(R) vs (-NLL) must be approximately 1.0.

**Actual slope: 0.035** (MiniLM). The dynamic ranges are:
- log(R): spans 1.39 units
- -NLL: spans 39.73 units
- Range ratio: 28.5x

This means log(R) = 0.035 * (-NLL) + const, not log(R) = -NLL + const. The identity fails not because of noise or residual scatter, but because the quantities live on completely different scales. A unit change in NLL produces only 0.035 units of change in log(R).

The residual percentage (24.6%) actually understates the problem. The identity residual is computed as std(log(R) + NLL) / range(NLL). Since log(R) contributes only ~3.5% of the scale of NLL, the sum (log(R) + NLL) is almost entirely NLL, and its std reflects NLL's own variance. The residual appears modest (24.6%) because NLL dominates the sum. The true failure is that the slope is off by a factor of 28.5.

**This was not flagged in either v2 or v3.** The identity check tests std(log_R + NLL) = const, which is correct, but neither the test nor the verdict discusses the slope. The question itself -- "Does log(R) = -F + const?" -- is poorly posed when the quantities differ in scale by 28.5x.

**Impact on verdict:** The identity criterion is correctly failing, but for a deeper reason than the 24% residual suggests. The relationship is proportional (log(R) ~ k*(-NLL) + C with k=0.035), not an identity. This makes the CONFIRMED threshold of "residual < 10%" irrelevant -- even with zero noise, the identity cannot hold because the slope is wrong. The right question is whether log(R) is proportional to -NLL, and the answer is yes (r=0.95, very tight linear fit with slope 0.035).

### Finding 2 (SIGNIFICANT): Partial Correlation After Removing Purity Is Still Strong

I computed the partial correlation of log(R) vs -NLL after linearly regressing out purity from both variables:

| Architecture | Partial r (purity removed) | Partial rho | p-value |
|---|---|---|---|
| all-MiniLM-L6-v2 | 0.7942 | 0.7770 | 1.68e-13 |
| all-mpnet-base-v2 | 0.8353 | 0.8192 | 6.58e-16 |
| multi-qa-MiniLM-L6-cos-v1 | 0.8210 | 0.8149 | 5.32e-15 |

**This is the strongest evidence that the correlation is not a purity artifact.** After completely removing the linear effect of purity, log(R) and -NLL still correlate at r=0.79-0.84. This means: at the SAME purity level, clusters with higher log(R) genuinely tend to have lower NLL.

Additionally, same-purity concordance (within each of the 19 purity levels, comparing the 3 replicates) is 77-81%, well above the 50% chance baseline.

**This analysis was not done in v3.** The within-group band correlations (Test 2) partially address this but still allow purity variation within bands. The partial correlation fully removes the purity confound and shows the relationship survives cleanly.

### Finding 3 (MODERATE): Narrow-Band Analysis Shows Heterogeneity

I computed correlations within 6 narrow bands (each spanning 0.10-0.15 in purity, n=9 each):

| Narrow Band | MiniLM r | mpnet r | multi-qa r |
|---|---|---|---|
| very_low (0.10-0.20) | 0.656 | 0.686 | 0.408 |
| low-mid (0.25-0.35) | 0.550 | 0.832 | 0.770 |
| mid (0.40-0.50) | 0.954 | 0.886 | 0.728 |
| mid-high (0.55-0.65) | 0.760 | 0.756 | 0.823 |
| high (0.70-0.80) | 0.904 | 0.737 | 0.818 |
| very_high (0.85-1.00) | 0.843 | 0.883 | 0.856 |

Notable:
- The relationship is weakest at the very lowest purities (0.10-0.20), especially for multi-qa (r=0.408, p=0.276).
- This makes physical sense: at 10-20% purity, clusters are almost random mixtures, and both R and NLL are in a noisy floor where the signal is weak.
- The relationship is strongest in the mid-to-high purity range (0.40-1.00), consistently r > 0.7 across architectures.
- With only 9 data points per narrow band, individual r values have wide confidence intervals. The pattern matters more than any single value.

### Finding 4 (MODERATE): Null Hypothesis Test Is Incomplete -- Only Overall, Not Within-Group

The v3 null hypothesis comparison (Test 3) shows that trace(cov), mean_variance, and mean_euclidean_dist all correlate with -NLL at |r| >= R's |r| overall. This correctly demonstrates that R is not uniquely correlated with NLL at the overall level.

**However, the null hypothesis comparison was NOT done within groups.** The key v3 finding is that R has strong WITHIN-GROUP correlations with NLL. If trace(cov) also has within-group r=0.8, then R is truly not special. If trace(cov) has within-group r=0.3, then R IS special in a meaningful way.

This gap means the verdict's claim that "R is not special" is only proven for overall correlation, not for the within-group correlation that is the v3's central result.

**Severity: MODERATE.** Given that trace(cov) and mean_variance are direct inputs to the logdet in NLL, they likely also have strong within-group correlations. But this should be tested, not assumed. The claim "R is not special" should be qualified as "R is not special for overall correlation; within-group comparison not tested."

### Finding 5 (LOW): mean_l2_norm Returns NaN for All Architectures

In Test 3 results, mean_l2_norm consistently shows NaN Pearson r across all architectures. This is likely because sentence-transformer embeddings (with normalize_embeddings=False) have nearly constant L2 norm, producing effectively zero variance in mean_l2_norm across clusters. The code at test_v3_q09.py:129 computes mean(norms) which would be near-constant if all embeddings have similar L2 norm.

This is not a bug -- it correctly identifies that mean_l2_norm is useless for this comparison. But it means only 4 of 5 alternatives were actually tested, and 3 of those 4 matched R. The count should arguably be "3 out of 4 valid alternatives" rather than "3 out of 5."

### Finding 6 (LOW): Gating Test Concordance Direction Assumption

Test 4 (test_v3_q09.py:665-676) assumes higher purity should yield higher R. This assumption is encoded but not explicitly justified. For the formula R = E/grad_S:
- Higher purity --> tighter cluster --> higher mean cosine similarity (E increases)
- Higher purity --> more homogeneous similarities --> lower std (grad_S decreases)
- Both effects push R higher, so the assumption is correct.

The concordance rate of 95-97% (gap >= 0.10) confirms this. No issue here.

---

## Code Quality

### No New Bugs Found

The v3 code is clean and well-structured. The imports, formula usage, and statistical computations are all correct. The ddof fix is properly applied. The cluster construction logic correctly implements continuous purity with category rotation.

### Minor Observations (Not Bugs)

1. **test_v3_q09.py:36-41:** Absolute path for formula import. This is fragile but works for the test environment and was inherited from v2.

2. **test_v3_q09.py:98:** `cov = scatter.copy()` -- the copy is redundant since cov_reg creates a new array, but it's harmless and makes the intent clear.

3. **test_v3_q09.py:260-263:** When available_primary < n_primary, it pads with replacement sampling from the same (small) pool. This could create duplicate documents. At purity=0.10 with cluster_size=200, n_primary=20, and max_per_cat=250, available_primary is always >= n_primary, so this path is likely never hit.

---

## Verdict Assessment

### Is INCONCLUSIVE Correct?

**Yes. INCONCLUSIVE is the correct verdict.** The pre-registered criteria are:

- CONFIRMED requires: within-group |r| > 0.7 on >= 2/3 arch **AND** identity residual < 10%.
  - Within-group: PASS (3/3 architectures have >= 2/3 bands above 0.7)
  - Identity: FAIL (residual 23.6-24.6%, and more fundamentally, slope = 0.035 not 1.0)
  - **CONFIRMED fails on identity.**

- FALSIFIED requires: overall |r| < 0.5 **OR** within-group |r| < 0.3 on all arch.
  - Overall r: 0.953-0.962 (far above 0.5)
  - Within-group: 0.69-0.89 (far above 0.3)
  - **FALSIFIED conditions not met.**

- INCONCLUSIVE: otherwise. **This is the correct bucket.**

### Could It Be FALSIFIED?

No. The within-group correlations (0.69-0.89) and partial correlations (0.79-0.84) demonstrate a genuine relationship beyond group structure. The finding is real. FALSIFIED is not justified.

### Could It Be CONFIRMED?

No. The identity log(R) = -NLL + const fails fundamentally (slope=0.035, not 1.0). Even with zero noise, the identity cannot hold because the scales differ by 28.5x. The question as posed ("Does log(R) = -F + const?") has a clear answer: NO.

However, the proportionality log(R) ~ k*(-NLL) + C holds strongly (r=0.95, stable across architectures). If the question were reformulated as "Is log(R) proportional to -NLL?", the answer would be a clear YES.

### Is the Verdict Honestly Written?

**Yes, with one mild criticism.** VERDICT_v3.md is notably more honest than v2:
- The dual-verdict problem from v2 is gone. A single clear INCONCLUSIVE.
- The "What the test does NOT show" section is excellent.
- The honest relabeling (Gaussian NLL, not FEP) is properly maintained.
- The null hypothesis finding is clearly stated.

The mild criticism: the verdict does not discuss the slope problem (Finding 1). The 24% residual is presented as the reason the identity fails, but the deeper issue -- that the quantities live on different scales (factor 28.5x) -- is not mentioned. This matters because it means the 10% threshold for CONFIRMED was never achievable, making that criterion misleadingly close to reasonable.

---

## Summary of Issues

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | CRITICAL | Slope is 0.035, not 1.0 -- identity question is poorly posed | NEW - not in v2 or v3 |
| 2 | SIGNIFICANT | Partial correlation (purity removed) is r=0.79-0.84 -- relationship is real | NEW - strengthens v3 |
| 3 | MODERATE | Narrow-band heterogeneity: very low purity (0.10-0.20) is weak | NEW |
| 4 | MODERATE | Null hypothesis not tested within-group | NEW gap |
| 5 | LOW | mean_l2_norm is NaN (constant-norm embeddings) | Minor |
| 6 | LOW | Gating direction assumption correct but implicit | Minor |

No bugs found. No statistical errors in the computations. All reported numbers verified.

---

## Recommendations for Future Work

### If Q09 is revisited:

1. **Reformulate the question.** "Does log(R) = -F + const?" has a clear answer (NO, slope=0.035). The interesting question is "Is log(R) proportional to -NLL, and if so, what determines the proportionality constant k?" This is testable and the current data already suggests k varies narrowly (~0.035 for 384-dim models).

2. **Complete the null hypothesis.** Test whether trace(cov), mean_variance, and mean_euclidean_dist also have strong within-group and partial correlations with -NLL. This determines whether R's within-group relationship is unique or shared by all tightness metrics.

3. **Investigate the proportionality constant.** If log(R) ~ k * (-NLL) + C, does k depend on dimensionality d? On the embedding model? On the data domain? This could reveal whether the relationship is a genuine structural law or an empirical coincidence.

---

## Final Assessment

**INCONCLUSIVE is the correct verdict.** The v3 test is well-designed and honestly executed. The within-group correlations (r=0.69-0.89) are genuine and survive partial correlation analysis (r=0.79-0.84). This is a real empirical finding. However, the identity claim fails fundamentally due to a 28.5x scale mismatch, not just a 24% residual. The question as posed cannot be confirmed. R is also not demonstrated to be special compared to other tightness metrics within groups.

**Honesty grade: A.** v3 represents a significant improvement in scientific rigor over v2. The honest relabeling, continuous purity design, within-group analysis, and null hypothesis comparison all address the v2 audit's critical findings. The remaining gaps (scale analysis, within-group null hypothesis) are genuine oversights, not motivated reasoning.

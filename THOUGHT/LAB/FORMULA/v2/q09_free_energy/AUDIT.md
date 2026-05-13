# Q09 Adversarial Audit Report

**Auditor:** Adversarial audit agent
**Date:** 2026-02-06
**Scope:** Q09 v2 -- Does log(R) = -F + const hold?
**Files reviewed:** shared/formula.py, test_v2_q09_fixed.py, test_v2_q09_fixed_results.json, VERDICT_v2.md

---

## CRITICAL FINDING: THE r=0.97 CORRELATION IS NOT TAUTOLOGICAL, BUT IT IS STRUCTURALLY CONFOUNDED

### The Make-or-Break Question: Is F Independent of R's Components?

**R_simple = E / grad_S** where:
- E = mean pairwise cosine similarity (formula.py:13-42)
- grad_S = std of pairwise cosine similarities (formula.py:45-69)

**F = 0.5 * (d*log(2*pi) + log|cov + reg*I| + tr(inv(cov+reg*I) @ scatter))** where:
- cov = covariance of raw embedding vectors (test_v2_q09_fixed.py:91)
- scatter = centered.T @ centered / n (test_v2_q09_fixed.py:89)

**Verdict on tautology: F is NOT directly computed from E and grad_S.** R operates on pairwise cosine similarities (a 1D summary of angle distributions). F operates on the full d-dimensional covariance matrix of raw embedding vectors. These are different mathematical objects computed through different pathways. The correlation is **not a mathematical identity** in the way that, say, correlating x/y with 1/y would be.

**However, the correlation IS a confounded structural relationship.** Both R and F are driven by the same latent factor: cluster geometric tightness. For text embeddings:
- Tight clusters (pure topics) -> high mean cosine similarity (high E) -> high E/grad_S (high R) -> low covariance spread (low F)
- Diffuse clusters (random mix) -> low mean cosine similarity (low E) -> low E/grad_S (low R) -> high covariance spread (high F)

The r=0.97 reflects that cosine similarity concentration and Gaussian covariance are both measuring the same thing (how clustered the embeddings are), NOT that log(R) = -F + const is a structural law.

**This is neither tautological (trivially true by construction) nor a genuine deep connection. It is a confound.**

---

## Code Bugs Found

### Bug 1: Covariance Computed Twice Differently (MEDIUM)
- **Location:** test_v2_q09_fixed.py:89-91
- **Issue:** `scatter = (centered.T @ centered) / n` uses division by n. Then `cov = np.cov(embeddings, rowvar=False)` uses numpy's default ddof=1, so it divides by (n-1). When cov_inv @ scatter is computed, the trace_term is off by a factor of n/(n-1). For n=200, this is a factor of 200/199 = 1.005 -- negligible but technically wrong.
- **Impact:** Negligible for n=200 clusters. The trace_term should be exactly d (degrees of freedom) when the model matches the data; the n vs n-1 discrepancy means it evaluates to d*n/(n-1) instead. This creates a tiny constant offset in F but does not affect correlations (additive constant cancels).

### Bug 2: Regularization Bakes In a Constant Offset (LOW)
- **Location:** test_v2_q09_fixed.py:72-73, 92
- **Issue:** reg=1e-4 is added to the covariance diagonal. For 384-dimensional embeddings, this adds 384 * 1e-4 = 0.0384 to the trace_term (via inv(cov+reg*I)). The logdet also shifts. This is architecture-dependent (384d vs 768d).
- **Impact:** It changes the absolute value of F between architectures but does not affect within-architecture correlations. The cross-architecture comparison of Pearson r values is unaffected because Pearson r is invariant to linear transforms.

### Bug 3: No NaN Handling for R_full in Gating Test (LOW)
- **Location:** test_v2_q09_fixed.py:404-406
- **Issue:** When R_full is NaN, it's replaced with 0.0. If a cluster genuinely has NaN R_full, this assigns it the minimum quality score, which could misclassify it. But with n=200 clusters of 200 documents each, NaN is extremely unlikely.
- **Impact:** Negligible in practice.

---

## Statistical Errors

### Error 1: CRITICAL -- Confound Masquerading as Correlation (SEVERE)
- **Issue:** The test creates 3 sharply separated groups: pure (purity=1.0), mixed (purity~0.08), degraded (purity~0.52). Both R and F are monotonically related to cluster tightness. Correlating them across these 3 extreme groups guarantees high r, just as correlating ANY two measures that respond to cluster quality would yield high r.
- **Evidence:** Look at the raw data. log(R) for pure clusters: [0.0 to 0.54]. log(R) for mixed clusters: [-1.08 to -0.83]. log(R) for degraded: [-0.87 to -0.39]. These form 3 well-separated bands. Similarly for -F. The r=0.97 is ALMOST ENTIRELY driven by the between-group separation, not by within-group tracking.
- **Test:** If you computed the correlation within each group separately, I predict r would drop dramatically (possibly below 0.5). This is not tested.
- **Severity:** This is the single most important statistical issue. The test proves that "R and F both respond to cluster purity," not that "log(R) = -F + const."

### Error 2: Pearson r Is Inflated by Group Structure (MODERATE)
- **Issue:** With 3 groups of ~20 data points each, and the groups being far apart, the data approximates 3 point masses. Pearson r between 3 point masses can be trivially high (3 points always have r near +/-1.0 unless they form a non-monotone pattern). The effective degrees of freedom are closer to 3 (one per group) than 60 (one per cluster).
- **Impact:** The reported p-values (e.g., 2.7e-38) are dramatically overstated. The real effective sample size is approximately 3 groups, not 60 clusters. With 3 groups, a Pearson r of 0.97 has a p-value around 0.15 -- not significant.

### Error 3: Identity Check Is Done Correctly But Underweighted (LOW)
- **Issue:** The identity check std(log(R)+F) = 13.2 with range(F) = 44.6 correctly shows the identity does NOT hold (residual is 30% of range). The verdict acknowledges this. But this should have been the LEAD finding, not a footnote. The test is titled "Does log(R) = -F + const?" and the answer from their own identity check is "No."

### Error 4: Spearman rho Partially Mitigates But Does Not Solve the Group Problem (LOW)
- **Issue:** Spearman rho (0.94-0.96) is slightly lower than Pearson r (0.97), which is consistent with some non-linearity. Spearman is rank-based and thus less sensitive to between-group magnitude, but it is still dominated by the clean 3-group separation.

---

## Methodological Issues

### Issue 1: Task Design Creates Guaranteed High Correlation (SEVERE)
- **Issue:** The 3 cluster types (pure/mixed/degraded) create an artificially wide dynamic range in both R and F. Any measure of cluster quality would correlate with any other measure of cluster quality across this range. The correlation is not specific to the log(R) vs -F relationship.
- **Better design:** Use clusters that vary continuously in coherence (e.g., mixing ratios from 0% to 100% in 5% increments), not just 3 extreme types. Or compute correlations WITHIN each cluster type.

### Issue 2: Free Energy Definition Is Not Standard FEP (MODERATE)
- **Issue:** The "variational free energy" computed here is the negative log-likelihood under a fitted Gaussian. In the Free Energy Principle (Friston), variational free energy is F = D_KL(q || p) + E_q[-log p(o|s)] where q is a recognition density, p is a generative model, and o are observations. The Gaussian NLL used here is a degenerate case where q = delta(mean) and the generative model is Gaussian. This is not the FEP's free energy -- it's just a Gaussian fit quality score.
- **Impact:** Even if log(R) = -F_gaussian perfectly, this would NOT establish a connection to the Free Energy Principle. It would establish that R is related to Gaussian fit quality, which is a much weaker and less interesting claim.

### Issue 3: Gating Test Purity Thresholds Are Too Extreme (MODERATE)
- **Issue:** High quality = purity > 0.8 (all 20 pure clusters qualify). Low quality = purity < 0.4 (all 20 mixed clusters qualify). No degraded clusters fall in either bin. This is a 20-vs-20 binary classification where the two classes occupy completely different regions of feature space. ANY reasonable metric achieves F1=1.0.
- **Impact:** The gating test is completely uninformative. The verdict correctly identifies this but does not flag it as a design flaw.

### Issue 4: 20 Newsgroups Is One Dataset (LOW)
- **Issue:** All results come from one text corpus. Generalization to other domains (images, audio, tabular data, different text corpora) is not tested.

### Issue 5: All Architectures Are Sentence Transformers (LOW)
- **Issue:** All 3 architectures are sentence-transformer models trained with similar contrastive objectives. They may share representation biases that produce similar R-F relationships. Testing with fundamentally different architectures (e.g., BoW, TF-IDF, random projections) would be more convincing.

---

## Verdict Assessment

### The Dual Verdict Is Problematic
- "INCONCLUSIVE (formally) / PARTIALLY CONFIRMED (substantively)" is trying to have it both ways. Pick one.
- The formal criteria produce INCONCLUSIVE. That should be the verdict.
- Editorializing that it is "substantively partially confirmed" undermines the pre-registration. If you pre-register criteria and they produce INCONCLUSIVE, report INCONCLUSIVE. Do not add qualifiers.

### The Correct Verdict Should Be INCONCLUSIVE with Stronger Caveats
Given the confound analysis above, even the correlation finding is weaker than presented:
1. r=0.97 is confounded by 3-group structure (effective N~3, not 60)
2. The identity check (std=13.2) explicitly disconfirms the equality
3. The "free energy" is Gaussian NLL, not FEP free energy
4. The gating test is uninformative

### What the Test Actually Shows
The test demonstrates that R (E/grad_S) and negative Gaussian log-likelihood co-vary across clusters with very different coherence levels. This is expected: both measure aspects of cluster tightness. The test does NOT show:
- That log(R) = -F + const (identity check fails)
- That R is connected to the Free Energy Principle (wrong F definition)
- That R has practical advantages over simpler metrics (gating test is at ceiling)
- That the relationship holds within coherence levels (not tested)

---

## Issues Requiring Resolution

### P0 - Must Fix Before Any Claim Stands

1. **Compute within-group correlations.** Report Pearson and Spearman r for log(R) vs -F separately within each cluster type (pure only, mixed only, degraded only). If within-group r drops below 0.3, the r=0.97 is entirely driven by group membership, not by a genuine R-F relationship.

2. **Use continuous mixing ratios.** Create clusters with mixing proportions at 0%, 10%, 20%, ..., 100% noise. This eliminates the 3-group confound.

3. **Compare against null hypothesis correlations.** Compute the correlation between -F and OTHER simple cluster statistics (mean Euclidean distance, mean L2 norm, trace of covariance, first eigenvalue, etc.). If these also show r > 0.9, then the finding is not specific to R.

### P1 - Should Fix

4. **Use actual FEP free energy.** If the claim is about the Free Energy Principle, compute free energy under a model that could plausibly be an FEP generative model (e.g., Gaussian mixture, or at minimum acknowledge that the Gaussian NLL is not FEP-standard).

5. **Design a harder gating task.** Use purity thresholds that create overlap (e.g., 0.5-0.7 vs 0.3-0.5). Or use a completely different operational task.

### P2 - Nice to Have

6. **Test on non-sentence-transformer embeddings.** Use TF-IDF, BoW, or random projections to verify the relationship is not architecture-specific.
7. **Test on non-text data.** Image embeddings, tabular data embeddings.

---

## What Would Change the Verdict

| Finding | New Verdict |
|---|---|
| Within-group r > 0.7 for all 3 types | Strengthens PARTIALLY CONFIRMED -- the relationship goes beyond group membership |
| Within-group r < 0.3 for all 3 types | INCONCLUSIVE with severe caveats -- r=0.97 is a group-structure artifact |
| Other simple metrics also correlate with F at r > 0.9 | INCONCLUSIVE -- R is not special, everything correlates with F |
| Continuous mixing ratios still show r > 0.8 | Genuine empirical finding (but still not FEP) |
| FEP-proper F still correlates with log(R) at r > 0.8 | Would upgrade to PARTIALLY CONFIRMED (FEP connection) |
| Within-group r > 0.7 AND proper FEP F AND continuous mixing | CONFIRMED |

---

## Summary

The r=0.97 correlation is **not tautological** (F and R compute different things from the same data), but it is **confounded by a 3-group design** that guarantees high between-measure correlations for any pair of cluster-quality metrics. The effective sample size is ~3 (one per group), not 60 (one per cluster). The identity check within the test's own results (std(log(R)+F) = 13.2) already disconfirms the exact equality. The "free energy" used is Gaussian NLL, not FEP free energy, so even a perfect correlation would not establish an FEP connection.

**Recommended verdict: INCONCLUSIVE** -- with the finding reframed as "R and Gaussian NLL both respond to cluster coherence, as expected" rather than "log(R) approximately equals -F."

**Severity: 3 critical issues** (confounded group structure, non-FEP free energy, uninformative gating test), **1 moderate issue** (identity check disconfirms the exact claim), **2 minor code issues**.

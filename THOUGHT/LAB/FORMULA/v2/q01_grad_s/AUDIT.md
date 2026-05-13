# Q01 Audit Report: grad_S Independent Predictive Value

## Verdict Under Review: FALSIFIED

---

## Code Bugs Found

### BUG-01: Partial correlation implementation uses hybrid Spearman/Pearson (lines 206-244)

The `partial_correlation()` function rank-transforms the variables (correct for Spearman), then regresses ranks on ranks via `linregress` (linear regression on ranks = Spearman-style), then computes **Pearson** correlation of the residuals (`stats.pearsonr` at line 236). This is the standard textbook method for Spearman partial correlation and is **correct** in the general case. However, there is a subtlety: the p-value adjustment at lines 239-242 uses a t-distribution with `df=n-3`, which is the classical formula for partial correlation p-values. This is valid for Pearson partial correlation under normality of residuals. For rank-based residuals with n=20, the approximation is reasonable but imperfect. **Severity: LOW** -- the method is standard and defensible, but the p-values are approximate, not exact.

### BUG-02: No bug -- formula.py functions are used correctly (lines 85-86)

The test correctly imports `compute_E` and `compute_grad_S` from the shared formula.py. The local `compute_pairwise_sims()` (lines 59-67) duplicates the normalization logic from formula.py but is only used for MAD/IQR computation, not for E or grad_S. No inconsistency in the E/grad_S values. **Verified correct.**

### BUG-03: Silhouette computation uses binary labels (line 132)

At line 132, the silhouette is computed with binary labels: `[1]*n_target + [0]*n_other`. This means the silhouette score measures how well a specific cluster separates from "everything else" as a single blob. This is a valid interpretation for the question being asked (cluster cohesion/separation quality), but it differs from the standard multi-class silhouette score. The consequence is that the silhouette scores are potentially inflated (a cluster only needs to be distinct from the average of all others, not from each individual other cluster). **Severity: LOW-MEDIUM** -- the metric is self-consistent across comparisons, but the absolute values may not match standard silhouette interpretations.

### BUG-04: No seed isolation between architectures for data subsampling (line 284)

Each architecture call creates `rng = np.random.RandomState(RANDOM_SEED)` at line 284. This means **all three architectures use the exact same document subsample** (same 200 docs per cluster). This is actually good for controlled comparison across architectures, but it means the 3 architectures are not independent -- they analyze the exact same documents. The "consistent across architectures" claim in the verdict (Finding 4) is therefore weaker than presented: the same 4,000 documents are used each time, and the primary source of variation is the embedding model. **Severity: MEDIUM** -- does not invalidate results but the "three independent replications" framing is slightly misleading.

### BUG-05: Silhouette sampling RNG is also reset identically (line 307)

`rng2 = np.random.RandomState(RANDOM_SEED + 1)` is created per architecture. This means the same target/background samples are drawn for silhouette computation each time. Combined with BUG-04, the silhouette scores across architectures differ only due to different embeddings, not due to sampling variation. This is fine for controlled comparison but means the three results are not statistically independent.

### BUG-06: `background_per_cluster` parameter in `subsample_dataset` is unused (lines 247-275)

The function signature includes `background_per_cluster=100` but this parameter is never used inside the function body. The function only uses `per_cluster`. This is dead code, not a bug that affects results, but it suggests a planned-but-not-implemented sampling strategy. **Severity: NONE** for results.

### BUG-07: Win/loss counting threshold is asymmetric and potentially misleading (lines 416-422)

A "win" requires `p_value < 0.01` (R0 significantly better) and a "loss" requires `p_value > 0.99` (R0 significantly worse). The p-value computed at line 191 is `np.mean(deltas <= 0)`, which is the proportion of bootstrap samples where R0 does NOT beat the alternative. So p < 0.01 means R0 beats the alternative in >99% of bootstraps. This is correct but extremely conservative -- it's testing at alpha=0.01 one-sided, which is effectively alpha=0.005 two-sided. The "0 wins" finding could be partly due to this high bar. **Severity: LOW** -- the criterion is pre-registered and internally consistent.

### BUG-08: The `subsample_dataset` function has an unused `all_indices` return (line 275)

The function returns `(sub_texts, sub_labels, cluster_map, all_indices)` but `all_indices` is captured as `original_indices` at line 288 and never used. Minor dead code. **Severity: NONE**.

---

## Statistical Errors

### STAT-01: n=20 is severely underpowered for partial correlation detection (CRITICAL)

With n=20 data points (one per cluster), the partial correlation test has very low statistical power. A partial correlation needs to be very large to reach significance at p<0.05 with df=n-3=17. Using the standard formula, the critical rho value for p<0.05 (two-tailed) at df=17 is approximately |rho| > 0.456.

Looking at the results:
- Architecture 1: partial_rho_gradS = -0.243 (not significant, but very small n)
- Architecture 2: partial_rho_gradS = -0.373 (p=0.115, approaching significance)
- Architecture 3: partial_rho_gradS = -0.173 (not significant)

The effect sizes are small-to-moderate and consistently negative. With only 20 data points, the test simply lacks power to detect effects of this magnitude. A power analysis shows that to detect a partial correlation of rho=-0.37 at alpha=0.05 with 80% power, you need approximately n=55 data points. The test is operating at roughly 30-40% power for the observed effect sizes.

**This is the single biggest issue.** The verdict says grad_S has "zero independent predictive value" but the data actually shows consistent negative partial correlations (grad_S negatively predicts quality after controlling for E in all 3 architectures). The non-significance is due to insufficient power, not absence of effect. The correct interpretation is "insufficient evidence to detect independent predictive value at this sample size," NOT "zero predictive value."

### STAT-02: The falsification criterion is too easily triggered

The pre-registered falsification criterion says: "grad_S partial corr non-significant in ALL architectures" triggers FALSIFIED. But significance testing with n=20 will fail to detect anything short of very large effects. The criterion effectively guarantees falsification unless grad_S has a massive effect (|rho| > 0.46). This is not a fair test -- it sets up a straw man where grad_S must demonstrate an unreasonably large independent effect to survive.

A better criterion would incorporate effect size or a meta-analytic combination of the three partial correlations.

### STAT-03: No multiple comparison correction discussed, but also not strictly needed

The test compares R0 against 4 alternatives, plus runs partial correlations. Given that the bootstrap tests all fail to show R0 superiority, multiple comparison correction would only make the results more negative for R0. In the other direction, the partial correlations are tested individually but the conclusion (non-significant in ALL architectures) is already the most conservative combination. **Severity: LOW** -- does not change conclusions.

### STAT-04: Spearman correlation with n=20 has limited precision

The bootstrap CIs confirm this: CIs for the delta between R0 and E alone span roughly +/-0.08 in the best case. This means differences in Spearman rho of <0.08 are undetectable. Since all the observed rho values are clustered between 0.88 and 0.94, the test genuinely cannot distinguish between the variants at this sample size. **Severity: MEDIUM** -- correctly acknowledged in Limitations section of verdict.

### STAT-05: The direction of partial correlations is informative but ignored

All three architectures show negative partial correlations of grad_S with silhouette (controlling for E): -0.243, -0.373, -0.173. The probability of getting all three negative by chance (if the true partial correlation is zero) is 0.5^3 = 0.125. While not significant on its own, this consistent direction is suggestive evidence that grad_S may actually HURT prediction after controlling for E. The verdict mentions this ("weakly associated with LOWER cluster quality") but does not quantify or formally test the sign consistency. A sign test or meta-analytic combination across architectures would be more appropriate.

### STAT-06: Architecture 2 partial correlation of R0|E approaches significance

For all-mpnet-base-v2, the partial correlation of R0 with silhouette controlling for E is rho=0.464, p=0.045. This IS significant at p<0.05. The verdict buries this in a table and never discusses it. This result suggests that in 768-dim space, dividing by grad_S DOES add some predictive value beyond E alone. However, this is only in one of three architectures, and the partial correlation of grad_S itself is not significant (p=0.115). These two facts are in mild tension: R0=E/grad_S adds value beyond E (p=0.045) but grad_S alone does not predict beyond E (p=0.115). The resolution is that the ratio E/grad_S is not the same as the additive effect of grad_S. **Severity: MEDIUM** -- this should have been discussed.

---

## Methodological Issues

### METH-01: 20 Newsgroups produces only 20 data points

This is the fundamental design limitation. Each "observation" is an entire cluster (newsgroup category), and there are only 20 categories. With 5 variants tested and partial correlations to compute, n=20 is marginal at best. Alternative designs that would produce more data points:
- Use subclusters (e.g., topic modeling within each newsgroup to create 100+ clusters)
- Use multiple datasets (Wikipedia categories, StackOverflow tags, etc.)
- Use leave-one-out or cross-validation designs
- Use hierarchical clustering to generate variable numbers of clusters

**Severity: HIGH** -- this limits the power of all statistical tests.

### METH-02: Silhouette score may not be the right ground truth

The question asks whether grad_S adds "independent predictive value" for "cluster quality." Silhouette score (using cosine distance) is one definition of cluster quality, but it is computed from the SAME embedding space as E and grad_S. This means E and silhouette are related by construction: both are derived from cosine similarities of the same vectors. A cluster where all vectors point in a similar direction will have high E AND high silhouette by mathematical necessity.

This creates a near-tautological relationship between E and silhouette, making it very hard for grad_S to add independent value. A more independent ground truth would be:
- Human-judged cluster quality
- Downstream task performance (classification accuracy)
- Cluster stability under perturbation
- Normalized Mutual Information with ground truth labels

**Severity: HIGH** -- the strong mathematical link between E and silhouette biases the test against grad_S.

### METH-03: All clusters are the same size (200 docs)

By subsampling exactly 200 documents per cluster, the test eliminates natural variation in cluster size, which is a property that grad_S might be sensitive to. In real-world scenarios, clusters have varying sizes, and the normalization by grad_S might be more important when comparing clusters of different sizes. **Severity: MEDIUM** -- limits ecological validity.

### METH-04: The 200 docs per cluster cap may distort grad_S

For newsgroups with more than 200 documents, subsampling 200 docs introduces sampling variance in grad_S. The std of pairwise similarities is sensitive to outliers, and a random sample of 200 from a larger pool may not capture the true dispersion. E (the mean) is more robust to subsampling. This could systematically handicap grad_S relative to E. **Severity: LOW-MEDIUM**.

### METH-05: Three sentence-transformer models are not "three architectures" in the strong sense

All three models are sentence transformers trained on similar data with similar objectives. Two are MiniLM-based (384-dim) and one is MPNet-based (768-dim). They will produce highly correlated embedding spaces. True architectural diversity would include:
- Non-transformer models (word2vec, GloVe averages)
- Domain-specific models
- Randomly initialized models (as a control)
- Models with very different dimensionalities (e.g., 50, 384, 1024, 4096)

**Severity: MEDIUM** -- the "robust across architectures" claim is overstated.

### METH-06: Headers, footers, quotes removed but text quality varies

The test removes headers/footers/quotes from 20 Newsgroups (line 482) and filters documents with fewer than 10 characters (line 489). However, 20 Newsgroups documents still vary enormously in length and quality. Very short documents will produce embeddings close to the model's default behavior, potentially creating a floor effect in E. This affects all variants equally and is not specific to grad_S, but it adds noise. **Severity: LOW**.

### METH-07: The test asks the wrong question for grad_S's role

Q01 asks whether grad_S adds "independent predictive value beyond E." But in the formula R = E/grad_S, grad_S is not supposed to be an independent predictor -- it is a normalizer. The correct question would be: "Does normalizing E by grad_S improve the signal-to-noise ratio of E as a predictor of cluster quality?" This is subtly different from asking whether grad_S has independent predictive value.

Consider an analogy: in a t-test, the standard error normalizes the mean difference. The standard error by itself does not "predict" anything, but normalizing by it produces a more informative statistic. Similarly, grad_S might improve E's utility without having its own independent predictive power.

The partial correlation test (does grad_S predict beyond E?) is asking whether grad_S has additive predictive value. But the formula uses grad_S as a divisor (multiplicative relationship), not an additive term. The linear-regression-based partial correlation may miss a purely multiplicative interaction. **Severity: MEDIUM-HIGH** -- this is a conceptual issue with the test design.

### METH-08: Bootstrap with n=20 produces coarse estimates

Bootstrap resampling with only 20 data points means each bootstrap sample draws 20 items with replacement from 20 items. Many bootstrap samples will be missing several clusters entirely. The bootstrap distribution will be discrete and lumpy. While 10,000 resamples is standard, the granularity of the underlying data limits the resolution. **Severity: LOW-MEDIUM** -- correctly acknowledged but under-weighted.

---

## Verdict Assessment

- **Current verdict:** FALSIFIED
- **Auditor recommendation:** CHANGE TO INCONCLUSIVE
- **Confidence:** HIGH
- **Reasoning:**

The FALSIFIED verdict overclaims. The evidence shows:

1. **True finding:** E/grad_S does not SIGNIFICANTLY outperform E alone at conventional thresholds. This is solid.

2. **Overclaim:** "grad_S has zero independent predictive value" (verdict line 5, Finding 2). The data shows consistent small-to-moderate NEGATIVE partial correlations across all three architectures. The non-significance is primarily a power issue (n=20). The correct statement is "insufficient evidence for independent predictive value at this sample size."

3. **Missed nuance:** The mpnet architecture shows partial corr(R0, sil | E) = 0.464 with p=0.045, which IS significant. This means in at least one architecture, dividing by grad_S does add detectable predictive value beyond E alone. The verdict does not discuss this finding.

4. **Methodological bias:** The near-tautological relationship between E and silhouette (both derived from cosine similarities) biases the test heavily in favor of E alone, making it very hard for any normalizer to add value. This is the most serious issue.

5. **Wrong statistical question:** Partial correlation tests additive predictive value, but grad_S enters the formula as a divisor (multiplicative). The test may be insensitive to the actual mechanism by which grad_S could improve prediction.

The pre-registered criteria are asymmetric and favor falsification: the confirmation bar is extremely high (significant in ALL architectures) while the falsification bar is easily triggered (non-significant in all architectures with n=20, which has low power). The pre-registration is not invalid, but the criteria were poorly calibrated for the available sample size.

**Bottom line:** The evidence supports "E/grad_S is not demonstrably superior to E alone for predicting silhouette score in 20 Newsgroups data with n=20 clusters." It does NOT support "grad_S has zero independent predictive value" or a strong FALSIFIED verdict. The correct verdict is INCONCLUSIVE with a lean toward FALSIFIED.

---

## Issues Requiring Resolution

1. **Power analysis required.** Calculate and report the minimum detectable effect size for partial correlation with n=20 at alpha=0.05, power=0.80. This will show that the test could not have detected the observed effect sizes.

2. **Increase sample size.** Use subclusters, multiple datasets, or hierarchical clustering to generate 50-100+ data points per architecture.

3. **Discuss the mpnet partial correlation.** Architecture 2 shows significant partial corr(R0|E) at p=0.045. This contradicts the "zero value" claim and must be addressed.

4. **Use ground truth independent of cosine similarity.** Classification accuracy, retrieval metrics, or human judgments would avoid the E-silhouette tautology.

5. **Test multiplicative interaction explicitly.** Instead of partial correlation, test whether E/grad_S predicts silhouette better than E alone using cross-validation (e.g., leave-one-out prediction error). This directly tests the ratio form.

6. **Meta-analyze across architectures.** Combine the three partial correlations using Fisher's z-transformation or similar meta-analytic method rather than requiring significance in each individual architecture.

7. **Correct the verdict language.** Replace "zero independent predictive value" with "no statistically significant independent predictive value at the tested sample size."

8. **Acknowledge the consistent sign.** All three partial correlations of grad_S|E are negative. The probability of this under H0 is 12.5%. While not formally significant, it suggests a real (if small) negative relationship.

9. **Vary cluster sizes.** Test with unequal cluster sizes to better evaluate grad_S's normalization role.

10. **Add a control variant.** Include E/random_noise as a control to calibrate the baseline performance.

---

## What Would Change the Verdict

### To strengthen FALSIFIED:
- Increase sample size to n>=60 and still find non-significant partial correlations
- Use independent ground truth (not cosine-based) and still find no grad_S value
- Show that E alone cross-validates as well as E/grad_S (prediction error comparison)
- Demonstrate that the consistent negative partial correlations are due to a known statistical artifact

### To flip to CONFIRMED:
- Increase sample size and find that the negative partial correlations of grad_S become significant (showing grad_S REDUCES cluster quality prediction, i.e., dividing by it helps)
- Use hierarchical clustering to produce 100+ clusters and find that E/grad_S significantly outperforms E alone
- Use non-cosine ground truth where the E-silhouette tautology does not apply
- Show that in cross-validation, E/grad_S has lower prediction error than E alone

### To remain INCONCLUSIVE:
- Increase sample size to n>=60 and find partial correlations near zero with tight confidence intervals -- this would confirm that grad_S truly adds nothing
- Use multiple independent ground truths with mixed results across metrics

---

## Summary

The test code is implemented correctly with no critical bugs. The statistical methodology is standard. However, the test is **severely underpowered** (n=20) and uses a **ground truth that is mathematically coupled to E** (cosine-based silhouette). The pre-registered criteria are **asymmetric and favor falsification** at small sample sizes. The verdict overclaims by stating "zero independent predictive value" when the evidence only supports "insufficient evidence at this sample size." One architecture (mpnet) actually shows a significant partial correlation of R0|E (p=0.045), which contradicts the "zero value" narrative.

**Recommended verdict: INCONCLUSIVE** (with the evidence leaning toward FALSIFIED, but not strongly enough to declare it given the power limitations and methodological concerns).

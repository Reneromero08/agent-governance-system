# Q15 Adversarial Audit Report

**Auditor**: Adversarial audit agent
**Date**: 2026-02-06
**Scope**: Code, statistics, methodology, and verdict for Q15 v2 (Bayesian interpretation of R)
**Verdict under review**: INCONCLUSIVE

---

## Code Bugs Found

### BUG-1 (CRITICAL): Effective Sample Size is a deterministic function of E, not a "non-trivial Bayesian quantity"

**Location**: `test_v2_q15_fixed.py:121-131`, verdict criterion A

The code defines:
```python
def compute_effective_sample_size(embeddings):
    mean_corr = compute_E(embeddings)
    ess = n * (1.0 - mean_corr)
    return float(max(ess, 1.0))
```

All 60 clusters have n=200 (lines 197-199, 221-223, 238-246). Therefore ESS = 200 * (1 - E), which is a simple affine transformation of E. Spearman rank correlation is invariant to monotone transformations, so `rho(R_full, ESS)` = `-rho(R_full, E)` exactly.

The results confirm this: E vs eff_sample_size = -1.000 across all architectures.

**Impact**: The test labels ESS as a "non-trivial Bayesian quantity" alongside bootstrap posterior precision (line 265-266). The verdict criterion A uses `nontrivial_keys = ["bootstrap_post_prec", "eff_sample_size"]` (line 899). Because ESS is trivially related to E, only bootstrap_post_prec is genuinely non-trivial. The criterion uses the max absolute rho across these two keys (line 904). Since `|rho(R_full, ESS)| > |rho(R_full, bootstrap_post_prec)|` in every architecture, **the criterion A pass is driven by the trivial ESS metric, not by bootstrap posterior precision**.

Verifying from results:
- MiniLM: R_full vs ESS = |-0.952| = 0.952, R_full vs bootstrap = |0.930| = 0.930. Max = **0.952 (from ESS)**
- mpnet: R_full vs ESS = |-0.978| = 0.978, R_full vs bootstrap = |0.957| = 0.957. Max = **0.978 (from ESS)**
- multi-qa: R_full vs ESS = |-0.957| = 0.957, R_full vs bootstrap = |0.944| = 0.944. Max = **0.957 (from ESS)**

The `max_rhos_by_arch` in the verdict data confirms this: values are -0.952, -0.978, -0.957 -- these are the ESS correlations.

**Criterion A nominally passes but is using a mathematically trivial metric as its strongest evidence.** The bootstrap posterior precision also passes (>0.93), but the code was *designed* to pick the stronger one, and the stronger one happens to be trivial.

**Severity**: HIGH. The code's claim to use "non-trivial" quantities in criterion A is undermined. The pass still holds with bootstrap alone, but the code logic and the reported max_rhos are misleading.

### BUG-2 (MODERATE): Bootstrap posterior precision is dominated by E through the covariance trace

**Location**: `test_v2_q15_fixed.py:99-118`

The bootstrap posterior precision is defined as `1/trace(Var(bootstrap centroids))`. By the bootstrap central limit theorem, for large n and iid samples:

```
Var(bootstrap centroid) ~ Cov(X) / n
```

So `trace(Var(bootstrap centroids)) ~ trace(Cov(X)) / n`.

For unit-norm embeddings (which sentence transformer embeddings approximately are after L2 normalization), `trace(Cov(X))` is directly related to mean pairwise cosine distance, which is `1 - E`. Specifically:

```
trace(Cov(X)) = E[||x||^2] - ||E[x]||^2
```

For unit-norm vectors: `E[||x||^2] = 1`, and `||E[x]||^2 = E[x]^T E[x]`, which for high-dimensional near-unit-norm data is closely related to E (the mean pairwise similarity drives the norm of the mean vector).

The empirical evidence confirms this: E vs bootstrap_post_prec has rho = 0.952, 0.978, 0.980 across architectures. These are extremely high, suggesting that bootstrap posterior precision is *nearly* a deterministic function of E for this data.

**Impact**: The "non-trivial" Bayesian quantity is not as independent from E as claimed. It is NOT mathematically identical (hence rho < 1.0), but it is dominated by the same signal. The correlation of R_full with bootstrap_post_prec (0.930-0.957) is therefore largely measuring how well R_full tracks E, not how well R captures genuinely independent Bayesian structure.

**Severity**: MODERATE. The quantity is technically not identical to E, but its near-identity makes the "non-trivial" label misleading.

### BUG-3 (MODERATE): Intensive property test has misleading aggregation

**Location**: `test_v2_q15_fixed.py:493-514`, verdict criterion B

The overall CV is computed as the mean of 3 text CVs and 10 housing CVs:
- Text (384-dim): CV = 0.383, 0.356, 0.362 (mean 0.367, all FAIL the 0.15 threshold)
- Housing (8-dim): CV = 0.026, 0.022, 0.056, 0.036, 0.018, 0.035, 0.034, 0.033, 0.024, 0.009 (mean 0.029, all pass)

The overall mean = 0.107 < 0.15, passing the threshold. But this is because 10 housing clusters outnumber 3 text clusters. The verdict acknowledges this (VERDICT_v2.md line 98) but still reports a PASS.

**Impact**: The pre-registered criterion says "CV < 0.15" without specifying aggregation method. The unweighted mean over a 10:3 mix of easy:hard cases is biased. A domain-weighted mean (0.367 + 0.029) / 2 = 0.198 would FAIL. A "worst domain" check would definitely FAIL.

**Severity**: MODERATE. The verdict text honestly notes this concern, but the pre-registered criterion still records a PASS.

### BUG-4 (LOW): Test 3 gating task is trivially easy, providing no discriminative information

**Location**: `test_v2_q15_fixed.py:533-678`

The gating test compares pure single-category clusters vs random 200-doc mixtures across 20 newsgroup categories. Every method except 1/grad_S achieves F1 = 1.000. This means the test has zero discriminative power between R_full, R_simple, E, and 1/trace(cov).

The test design created a task so easy that the ceiling effect makes the criterion (R_full > E + 5%) impossible to satisfy in either direction. This is a design flaw, not a code bug per se, but it renders criterion C informationally null.

**Severity**: LOW for code correctness, HIGH for test design.

### BUG-5 (LOW): Test 4 has only 20 clusters per seed, giving very weak statistical power

**Location**: `test_v2_q15_fixed.py:686-872`

With 20 geographic clusters per seed, and minimum 30 samples per cluster, the per-seed Spearman correlations have n=20. For Spearman with n=20, a correlation needs to exceed approximately |0.45| to be significant at p<0.05. Mean rho across 10 seeds = 0.024 is clearly null, but the test could not detect a moderate effect even if one existed.

**Severity**: LOW. The test correctly reports non-significance, and the aggregation across 10 seeds helps somewhat.

---

## Statistical Errors

### STAT-1 (CRITICAL): The E > R_simple > R_full ordering is NOT tested for statistical significance

The verdict's central interpretive claim is: "E alone has STRONGER Bayesian connection than R_full" and "additional formula components degrade the Bayesian correlation." But the differences are:

For bootstrap_post_prec (the only genuinely non-trivial quantity):
- MiniLM: E = 0.952, R_simple = 0.942, R_full = 0.930. Differences: 0.010, 0.022
- mpnet: E = 0.978, R_simple = 0.964, R_full = 0.957. Differences: 0.014, 0.021
- multi-qa: E = 0.980, R_simple = 0.954, R_full = 0.944. Differences: 0.026, 0.036

Are these differences statistically significant? Comparing two Spearman correlations from the SAME sample requires a dependent-samples test (e.g., Steiger's Z-test or Williams' test). This was never performed.

For n=60, a Spearman rho of 0.93 vs 0.95 is a very small difference. The Fisher z-transformation gives z(0.93) = 1.658, z(0.95) = 1.832. The standard error for comparing dependent correlations with shared variable is approximately 1/sqrt(n-3) * correction_factor. With n=60 and high intercorrelation between E and R_full, the correction factor could make this non-significant.

**Impact**: The headline finding ("E alone has stronger Bayesian connection") may be within sampling error. Without a formal comparison test, we cannot conclude that the ordering E > R_full is real rather than noise.

**Severity**: CRITICAL. The interpretive framework of the entire verdict rests on an untested statistical claim.

### STAT-2 (MODERATE): Spearman rho is used but the monotone relationship may be better captured by Pearson

Spearman measures monotone association. For the comparison E vs bootstrap_post_prec, the relationship is likely near-linear (since bootstrap_post_prec ~ 1/(trace(Cov)/n) and trace(Cov) is near-linearly related to 1-E for unit-norm data). Pearson correlation might show different relative rankings of E vs R_full because R_full introduces nonlinear distortion through sigma^Df.

**Impact**: The choice of rank correlation may actually favor E over R_full because R_full's sigma^Df multiplicative term preserves rank ordering less perfectly than it preserves linear structure.

**Severity**: LOW-MODERATE. Not necessarily wrong, but an unexamined choice.

### STAT-3 (MODERATE): P-values in Test 1 are misleading due to lack of independence

The 60 clusters are not independent: they are formed from the same embedding space (same model, same corpus). The 20 pure clusters sample from the same 20 categories; the 20 mixed clusters pair categories; the 20 random clusters draw from the same pool. Correlations among the 60 data points (R_full values, E values, etc.) inflate the effective sample size used in Spearman p-value computation.

All reported p-values (e.g., p = 1.6e-31) are therefore anti-conservative. The *actual* effective degrees of freedom are much lower than 58 (n-2).

**Impact**: The p-values are unreliable. However, with rho > 0.93, significance would hold even with greatly reduced df (e.g., rho = 0.93 with effective n=15 gives p < 0.001).

**Severity**: MODERATE. Doesn't change the qualitative conclusions but the reported p-values are wrong.

---

## Methodological Issues

### METH-1 (CRITICAL): The "Bayesian quantity" used as ground truth is not a standard Bayesian metric

The test claims to compare R against "Bayesian quantities." Let us evaluate each:

1. **1/trace(cov)**: This is the "likelihood precision trace." The code acknowledges this is trivially related to E (rho = 1.000). This is NOT a Bayesian quantity -- it is a frequentist statistic (inverse covariance trace). Calling it "likelihood precision" is a misnomer: in Bayesian statistics, the likelihood precision would be the inverse variance of the data-generating distribution's noise term, not the inverse trace of the sample covariance of embeddings.

2. **Bootstrap posterior precision**: Defined as `1/trace(Var(bootstrap centroids))`. This is a frequentist bootstrap quantity, NOT a Bayesian posterior. The bootstrap does not produce a posterior distribution in the Bayesian sense (it requires no prior, no likelihood model, and no Bayes' rule application). It approximates the *sampling distribution* of the estimator, which is a frequentist concept. Calling this "posterior precision" conflates frequentist and Bayesian frameworks.

   In proper Bayesian statistics, posterior precision of the mean under a normal model with known variance sigma^2 would be: `(sigma^{-2} * n + precision_prior)`, where the prior precision comes from the prior distribution. The bootstrap quantity has no prior component.

3. **Effective sample size (ESS)**: As shown in BUG-1, this is `n * (1 - E)`, a trivial function of E. While ESS has Bayesian uses (e.g., in MCMC diagnostics for correlated samples), the formula used here is just a linear transform of the mean pairwise cosine similarity. It does not correspond to any standard Bayesian ESS definition.

4. **Fisher information (Test 4)**: The diagonal Fisher information `E[grad_log_p(y|x,theta)^2]` is a frequentist quantity. It appears in Bayesian analysis as the curvature of the negative log-posterior (when the prior is flat), but computing it from a trained neural network with MSE loss on geographic clusters is a non-standard application. The connection between embedding structure (R) and parameter-space curvature (Fisher) is not theoretically motivated.

**Impact**: The entire test framework conflates frequentist statistics with Bayesian quantities. The "Bayesian interpretation" being tested is not well-defined. A proper Bayesian test would need:
- An explicit prior distribution on the embeddings or their parameters
- A likelihood model for observed data given embeddings
- Computation of the actual posterior distribution via Bayes' rule
- Comparison of R with a quantity derived from this posterior (e.g., marginal likelihood, posterior predictive density, Bayes factor)

None of these exist in the test.

**Severity**: CRITICAL. The test does not actually test a Bayesian interpretation. It tests correlation with frequentist statistics that have superficial naming connections to Bayesian concepts.

### METH-2 (HIGH): The comparison "E alone vs R_full" is the wrong comparison for the Bayesian question

Q15 asks: "Does R have a valid Bayesian interpretation?" The test operationalizes this as: "Does R correlate with Bayesian quantities better than E alone?" But this is a straw-man comparison.

R = (E / grad_S) * sigma^Df. Since R is a function of E, finding that R correlates with E-correlated quantities is expected. The question should be: "Does R have a Bayesian interpretation *that E alone does not have*?" which is a different (and much harder) question.

A proper test would ask: "Given E, does the additional information in grad_S, sigma, Df provide Bayesian-relevant information?" This would require a partial correlation analysis: compute the correlation of R_full with the Bayesian quantity after controlling for E. If R_full adds Bayesian information beyond E, the partial correlation should be significant.

No partial correlation analysis is performed.

**Severity**: HIGH. The comparison framework cannot answer the question being asked.

### METH-3 (MODERATE): 60 clusters with fixed n=200 gives no leverage on the intensive property question

The intensive property test (Test 2) varies n from 20 to 500 within a single cluster. But the Bayesian correlation test (Test 1) uses only n=200 clusters. The interaction between sample size and Bayesian correlation is never tested.

If R's Bayesian connection depends on the intensive property (i.e., R is only meaningful when it is stable across sample sizes), then the Bayesian test should be conducted at multiple sample sizes.

**Severity**: MODERATE. A design limitation.

### METH-4 (MODERATE): The cross-domain test (California Housing in Test 2) uses raw features, not embeddings

In Test 2, Part B uses `StandardScaler().fit_transform(housing.data)` -- 8-dimensional raw feature vectors. These are NOT embeddings from a neural model. The E formula (mean pairwise cosine similarity) has a clear geometric interpretation for unit-norm embeddings but is less interpretable for standardized tabular features.

Furthermore, the comparison with Test 1 (which uses 384-dim sentence embeddings) is confounded by dimensionality, data type, and normalization method.

**Severity**: MODERATE. The test correctly identifies that R_full behaves differently in low vs high dimensions, but the interpretation is muddied by mixing data types.

### METH-5 (LOW): The verdict_data.json contains DIFFERENT criteria than the main test

The `verdict_data.json` file (from an earlier run?) shows different criteria:
- "rho > 0.7 with sqrt(lik_prec)" (the main test uses bootstrap_post_prec and ESS)
- "F1 > 5% better than trivial" -> True (main test: diff = 0.000)
- "Falsification reproduces" concept is absent from the main test
- Threshold "all rho < 0.2" vs main test's "all rho < 0.3"

This file appears to be a leftover from an earlier version (v1). Its presence in the results directory is confusing.

**Severity**: LOW. It is labeled as a separate file and the main results are clear.

---

## Verdict Assessment

### Is INCONCLUSIVE the right call?

The pre-registered logic is:
- CONFIRM = A AND B AND C
- FALSIFY = A_falsify OR (B_falsify AND C_falsify)
- Otherwise = INCONCLUSIVE

Given the results: A=PASS, B=PASS(marginal), C=FAIL, the pre-registered verdict is INCONCLUSIVE. **The mechanical application of the decision rules is correct.**

However, the *interpretation* in the verdict document goes far beyond what the pre-registered criteria support. The verdict claims "E alone has STRONGER Bayesian connection than R_full" and "additional formula components degrade the Bayesian correlation." These are interpretive claims that require:
1. A significance test for the E-vs-R_full correlation difference (not performed -- STAT-1)
2. "Bayesian quantities" that are actually Bayesian (they are not -- METH-1)
3. A partial correlation analysis showing R adds nothing beyond E (not performed -- METH-2)

### Should this be FALSIFIED?

Arguments for FALSIFIED:
- R adds nothing over E for gating (F1 diff = 0.000)
- The "Bayesian quantities" are not actually Bayesian
- In the neural network setting (Test 4), R shows zero correlation with Fisher information
- The strong correlations in Test 1 are explained by E's dominance in R

Arguments against FALSIFIED:
- The pre-registered falsification criteria are not met (rho > 0.3 everywhere)
- R_full does correlate at >0.93 with bootstrap_post_prec, even if E does better
- The question is not "does R add value over E" but "does R have a Bayesian interpretation" -- and a correlation of 0.93 is hard to dismiss

**My assessment**: The verdict should remain INCONCLUSIVE, but the interpretive text should be more honest about the limitations. The "Bayesian interpretation" being tested is not properly Bayesian, which makes the entire question ill-defined.

### Should this be PARTIALLY CONFIRMED?

Arguments for:
- R_full correlates >0.93 with a quantity that has some Bayesian flavor
- R_full is intensive in favorable conditions (low dim)

Arguments against:
- The Bayesian quantities are not properly Bayesian (METH-1)
- The correlation is dominated by E, and R's additional terms may add noise
- The intensive property only holds in a specific regime

**My assessment**: PARTIALLY CONFIRMED would be an overclaim given METH-1.

### Is "additional components degrade correlation" a fair characterization?

It is an *observed pattern* but not a *tested claim*. The differences (0.95 vs 0.93 in bootstrap_post_prec correlation) are small and have not been tested for significance. "The additional components do not demonstrably improve correlation" would be more honest.

### Could the degradation be within statistical error?

Almost certainly yes. The differences are 0.02-0.04 in Spearman rho at n=60. A Steiger test for dependent correlations would very likely show p > 0.05 for these differences. The "degradation" narrative is not statistically supported.

---

## Issues Requiring Resolution

### MUST-FIX (5 items)

1. **Remove ESS from the "non-trivial" criterion A computation.** ESS = 200*(1-E) is mathematically identical to E in rank. The criterion A should use ONLY bootstrap_post_prec. Re-check whether the criterion still passes (it does: rho > 0.93 for all architectures, so the fix is cosmetic but necessary for intellectual honesty).

2. **Perform Steiger's test or Williams' test** for the difference between rho(E, bootstrap_post_prec) and rho(R_full, bootstrap_post_prec) to determine if the "E beats R_full" claim is statistically supported.

3. **Perform partial correlation analysis**: compute rho(R_full, bootstrap_post_prec | E) to determine whether R_full adds any information beyond E for Bayesian-like quantities.

4. **Redesign Test 3** with a harder discrimination task (e.g., pure vs mixed-2, or continuous purity prediction) so that the gating criterion is not trivially saturated at F1=1.0 for all methods.

5. **Reframe the "Bayesian" language.** The quantities tested are frequentist statistics, not Bayesian posteriors. The question should be reframed as "Does R correlate with standard statistical measures of cluster quality?" or the test should be redesigned with actual Bayesian machinery (priors, posteriors, marginal likelihoods).

### SHOULD-FIX (3 items)

6. **Use domain-weighted averaging** for the intensive property criterion, or test each domain separately. The current 10:3 ratio biases toward the low-dimensional case.

7. **Report confidence intervals** for the Spearman correlations, not just point estimates and p-values.

8. **Remove or relabel the stale verdict_data.json** file, which contains criteria from an earlier version.

---

## What Would Change the Verdict

### To CONFIRM:
- Define a **proper Bayesian quantity**: compute marginal likelihood p(X|model) under a Bayesian model (e.g., Gaussian mixture with conjugate priors). Show R correlates with log-marginal-likelihood better than E alone.
- Design a **harder gating task** where R_full outperforms E by >5% F1.
- Show via **partial correlation** that R_full captures Bayesian-relevant information beyond E.

### To FALSIFY:
- Perform Steiger's test showing E **significantly** outperforms R_full in correlation with a genuinely Bayesian quantity.
- Show that R_full's apparent correlation is **entirely explained** by E (partial rho ~ 0 after controlling for E).
- Run the test on a **wider range of dimensionalities and sample sizes** showing the pattern is robust.

### To maintain INCONCLUSIVE with stronger justification:
- Acknowledge that the current "Bayesian quantities" are not properly Bayesian.
- Report the partial correlation results.
- Acknowledge that the E-vs-R_full difference is not tested for significance.

---

## Summary

| Category | Count | Critical | High | Moderate | Low |
|----------|-------|----------|------|----------|-----|
| Code Bugs | 5 | 1 | 0 | 2 | 2 |
| Statistical Errors | 3 | 1 | 0 | 2 | 0 |
| Methodological Issues | 5 | 1 | 1 | 2 | 1 |
| **Total** | **13** | **3** | **1** | **6** | **3** |

**Critical issues**:
1. ESS is trivially = E, inflating criterion A's apparent rigor (BUG-1)
2. The E > R_full correlation ordering is not significance-tested (STAT-1)
3. None of the "Bayesian quantities" are actually Bayesian (METH-1)

**Bottom line**: The INCONCLUSIVE verdict is mechanically correct from the pre-registered criteria, but the interpretive narrative ("E alone has stronger Bayesian connection") rests on untested statistical claims and improperly defined Bayesian quantities. The test does not actually test a Bayesian interpretation -- it tests correlation with frequentist statistics that have been given Bayesian-sounding names. The entire Q15 framing needs conceptual revision before a meaningful answer can be given.

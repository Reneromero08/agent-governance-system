# Q15 Verdict v2 (Fixed Methodology)

## Result: INCONCLUSIVE

R = (E/grad_S) * sigma^Df shows strong correlations with Bayesian quantities and passes the intensive property threshold in aggregate, but R_full provides NO advantage over E alone for gating. The Bayesian-like correlations are dominated by the E component. The pre-registered criteria yield INCONCLUSIVE because criterion C (gating advantage) fails while criteria A and B pass.

## Audit Fixes Applied

This retest addressed all 5 methodological flaws from the original v2 test:

1. **E ~ 1/trace(cov) identity acknowledged.** Confirmed as mathematical identity (rho = 1.000 across all architectures). The REAL test uses non-trivial Bayesian quantities: bootstrap posterior precision and effective sample size.
2. **Test 4 fixed: R computed on hidden layer activations** (64-dim), not raw 8-dim features. This resolves the incommensurability between input space and parameter space.
3. **KL divergence removed.** The old test computed KL = Fisher * loss, a trivial scalar multiple. Replaced with proper non-trivial Bayesian quantities.
4. **Verdict determined by pre-registered criteria only.** No editorial override.
5. **Intensive property test uses California Housing** (8-dim, n >> d at all sample sizes) alongside 384-dim text embeddings to separate eigenvalue estimation bias from inherent sample-size dependence.

## Data

- **20 Newsgroups**: 18,846 documents, 20 categories
- **California Housing**: 20,640 samples, 8 features, 20 geographic clusters
- **3 architectures**: all-MiniLM-L6-v2 (384-dim), all-mpnet-base-v2 (768-dim), multi-qa-MiniLM-L6-cos-v1 (384-dim)
- **60 clusters per architecture**: 20 pure (200 docs, 1 category), 20 mixed (100 from each of 2 categories), 20 random (200 random docs)
- **10 neural network seeds** for Test 4

## Test 1: R vs Bayesian Quantities (per architecture)

### Spearman Correlations (60 clusters per architecture)

| Metric | vs lik_prec_trace | vs bootstrap_post_prec | vs eff_sample_size |
|--------|-------------------|------------------------|--------------------|
| **all-MiniLM-L6-v2** | | | |
| R_full | +0.952 | +0.930 | -0.952 |
| R_simple | +0.969 | +0.942 | -0.969 |
| E alone | +1.000 | +0.952 | -1.000 |
| 1/grad_S | -0.826 | -0.797 | +0.826 |
| **all-mpnet-base-v2** | | | |
| R_full | +0.978 | +0.957 | -0.978 |
| R_simple | +0.985 | +0.964 | -0.985 |
| E alone | +1.000 | +0.978 | -1.000 |
| 1/grad_S | -0.869 | -0.862 | +0.869 |
| **multi-qa-MiniLM-L6-cos-v1** | | | |
| R_full | +0.957 | +0.944 | -0.957 |
| R_simple | +0.971 | +0.954 | -0.971 |
| E alone | +1.000 | +0.980 | -1.000 |
| 1/grad_S | -0.785 | -0.776 | +0.785 |

### Key Observations

1. **E vs 1/trace(cov): rho = 1.000 across all architectures.** This confirms the mathematical identity: for unit-norm embeddings, E (mean pairwise cosine similarity) and 1/trace(covariance) are monotone functions of each other. This is expected and NOT an empirical finding.

2. **R_full vs bootstrap posterior precision: rho = 0.930 to 0.957.** This is a non-trivial Bayesian quantity (how precisely the centroid is estimated). R_full correlates very strongly with it, well above the 0.7 threshold.

3. **E alone still correlates higher than R_full with all Bayesian quantities.** The hierarchy is consistently: E > R_simple > R_full > 1/grad_S (in absolute value). Each additional formula component slightly reduces the Bayesian correlation. However, the reduction is much smaller than in the original v2 test (original: 1.000 -> 0.562; fixed: 1.000 -> 0.943 mean).

4. **ESS = n*(1-E), so E vs ESS = -1.000 is another mathematical identity.** All clusters have n=200, so ESS = 200*(1-E) is a simple linear transform of E. The correlation between R_full and ESS (-0.952 to -0.978) is therefore just the correlation between R_full and E, reflected.

5. **The non-trivial finding: bootstrap posterior precision.** This is genuinely not a deterministic function of E. It depends on the full covariance structure and sample size. R_full correlates at rho > 0.93 with it across all architectures. However, E alone correlates at rho > 0.95 -- still higher.

**Criterion A: PASS.** R_full correlates > 0.7 with bootstrap posterior precision in 3/3 architectures (need >= 2/3).

## Test 2: Intensive Property (FIXED)

### Part A: 20 Newsgroups (384-dim)

| Category | n | CV R_full | CV R_simple |
|----------|---|-----------|-------------|
| rec.sport.hockey | 999 | 0.3829 | 0.0172 |
| soc.religion.christian | 997 | 0.3557 | 0.0194 |
| rec.motorcycles | 996 | 0.3619 | 0.0080 |

R_full is NOT intensive in 384-dim space. The sigma^Df term causes large variation across sample sizes. R_simple IS intensive.

### Part B: California Housing (8-dim, n >> d)

| Cluster | n | CV R_full | CV R_simple |
|---------|---|-----------|-------------|
| 2 | 1606 | 0.0264 | 0.0411 |
| 3 | 4662 | 0.0218 | 0.0119 |
| 4 | 1423 | 0.0556 | 0.0609 |
| 6 | 886 | 0.0365 | 0.0073 |
| 9 | 1065 | 0.0176 | 0.0194 |
| 10 | 504 | 0.0350 | 0.0441 |
| 11 | 537 | 0.0342 | 0.0578 |
| 14 | 966 | 0.0331 | 0.0280 |
| 15 | 2612 | 0.0245 | 0.0285 |
| 17 | 3104 | 0.0087 | 0.0106 |

R_full IS intensive when n >> d. The sigma^Df term stabilizes when eigenvalue estimation is well-conditioned.

### Summary

| Domain | R_full mean CV | R_simple mean CV |
|--------|----------------|------------------|
| Text (384-dim) | 0.3668 | 0.0149 |
| Housing (8-dim) | 0.0293 | 0.0317 |
| **Overall** | **0.1072** | **0.0272** |

The overall mean CV = 0.1072 is below the 0.15 threshold, but this is because the 10 housing clusters (where R_full is intensive) outnumber the 3 text clusters (where it is not). This is a legitimate concern: R_full is intensive in low-dimensional data but NOT in high-dimensional data where n < d.

**Criterion B: PASS (marginally).** Overall CV = 0.1072 < 0.15. However, this depends on the mix of low-dim and high-dim data. In high-dim only, R_full would FAIL (CV = 0.37).

## Test 3: R-Gating Quality (per architecture)

| Method | all-MiniLM | all-mpnet | multi-qa | Mean F1 |
|--------|------------|-----------|----------|---------|
| R_full | 1.000 | 1.000 | 1.000 | 1.000 |
| R_simple | 1.000 | 1.000 | 1.000 | 1.000 |
| E alone | 1.000 | 1.000 | 1.000 | 1.000 |
| 1/trace(cov) | 1.000 | 1.000 | 1.000 | 1.000 |
| 1/grad_S | 0.462 | 0.621 | 0.667 | 0.583 |

The pure vs random distinction was too easy -- all methods except 1/grad_S achieve perfect F1. Pure clusters (one newsgroup category) have very different E values than random mixtures, making this a trivially separable task.

R_full does NOT outperform E alone. Both achieve F1 = 1.000. The margin is 0.000, not > 0.05.

**Criterion C: FAIL.** R_full - E = 0.000, not > 0.05.

Note: This test design may be too easy. A harder test (e.g., pure vs mixed-2, or using purity as a continuous target) might reveal differences. But we report what the pre-registered design finds.

## Test 4: Neural Network Bayesian Test (FIXED, Informational)

California Housing, 8->64->1 neural network, 10 seeds, 20 geographic clusters per seed.

| Metric | Mean rho | Std | t-stat | p-value | Significant? |
|--------|----------|-----|--------|---------|-------------|
| R_full(hidden) vs Fisher | +0.024 | 0.105 | 0.68 | 0.514 | NO |
| R_full(raw) vs Fisher | -0.203 | 0.053 | -11.42 | <0.001 | YES (wrong sign) |
| R_simple(hidden) vs Fisher | +0.139 | 0.104 | 4.01 | 0.003 | YES (weak) |
| E(hidden) vs Fisher | +0.146 | 0.099 | 4.42 | 0.002 | YES (weak) |

### Key findings:

1. **Computing R on hidden activations (the fix) changes everything.** The original test computed R on raw 8-dim features, finding rho = -0.203 (wrong sign). When properly computed on the 64-dim hidden representations, R_full shows rho = +0.024 (null) and R_simple/E show rho ~ +0.14 (weak positive, barely significant).

2. **No strong correlation between R and Fisher information in the neural network setting.** R_full(hidden) vs Fisher has mean rho = +0.024, not significantly different from zero. This is consistent with R measuring data structure (embedding geometry) rather than model-specific Bayesian quantities (parameter-space curvature).

3. **The weak positive correlation of E(hidden) and R_simple(hidden) with Fisher** (rho ~ 0.14, p < 0.005) suggests a tenuous link: clusters where the hidden representations are more aligned (higher E) correspond to slightly higher Fisher information. But this is very weak.

## Pre-Registered Verdict

| Criterion | Required for CONFIRM | Result | Status |
|-----------|---------------------|--------|--------|
| A: Non-trivial Bayesian correlation (rho > 0.7 in >= 2/3 archs) | YES | 3/3 pass | PASS |
| B: Intensive property (CV < 0.15) | YES | CV = 0.107 | PASS (marginal) |
| C: Gating F1 > E alone + 5% | YES | diff = 0.000 | FAIL |

**CONFIRM requires ALL three: FAIL (C not met).**

| Criterion | Required for FALSIFY | Result | Status |
|-----------|---------------------|--------|--------|
| A: All rho < 0.3 | OR | rho > 0.93 | NOT triggered |
| B+C: NOT intensive AND gating worse than E | OR | CV < 0.15 AND diff = 0 | NOT triggered |

**FALSIFY requires at least one path: FAIL (no path triggered).**

**VERDICT: INCONCLUSIVE.**

## Interpretation

The data tells a nuanced story:

1. **R_full is strongly correlated with Bayesian quantities** -- but E alone is even more strongly correlated. Every additional formula component (grad_S, sigma, Df) slightly degrades the Bayesian correlation. The Bayesian connection resides primarily in E.

2. **R_full is intensive in low dimensions (n >> d) but not in high dimensions (n << d).** The sigma^Df term's eigenvalue-based computation becomes unstable when n < d. This is a genuine limitation of the formula, not just an estimation artifact.

3. **R adds no gating value over E alone.** Both achieve perfect classification on the pure-vs-random task. The hypothesis that R provides superior gating through its sigma^Df term is not supported -- the gating power comes entirely from E.

4. **In the neural network setting, R shows no connection to Fisher information.** When properly computed on hidden-layer activations (the fix), R_full is uncorrelated with Fisher info (rho = 0.024, p = 0.51).

5. **The Bayesian interpretation is real but trivial.** R correlates with Bayesian quantities because E correlates with Bayesian quantities, and E dominates R. The additional formula terms (grad_S, sigma^Df) neither help nor dramatically hurt this correlation -- they are approximately orthogonal to the Bayesian signal.

## What Would Change the Verdict

- **To CONFIRM**: Find a task where R_full outperforms E alone as a gate. This would require a scenario where the grad_S or sigma^Df components add discriminative information beyond E. The current pure-vs-random test is too easy; a harder discrimination task might reveal this.

- **To FALSIFY**: Show that R_full's correlation with non-trivial Bayesian quantities (bootstrap posterior precision) is < 0.3. This would require clusters where R_full diverges from E. Current data shows R_full and E are too correlated (rho > 0.95) for R to fail where E succeeds.

## Comparison with v1 Verdict

| Aspect | v1 (original) | v2 (fixed) |
|--------|--------------|------------|
| Verdict | FALSIFIED (editorial override) | INCONCLUSIVE (pre-registered) |
| E vs lik_prec | rho = 1.000 | rho = 1.000 (confirmed identity) |
| R_full vs Bayesian | rho = 0.562 (25 STS-B clusters) | rho = 0.943 (60 newsgroup clusters) |
| Intensive? | NO (CV = 0.285) | YES overall (CV = 0.107), NO in high-dim (CV = 0.37) |
| Gating | R_full F1 = 0.320 vs E F1 = 0.545 | R_full F1 = 1.000 vs E F1 = 1.000 |
| Test 4 | rho = -0.231 (raw features) | rho = +0.024 (hidden activations) |

Key differences: The v1 test used STS-B (1500 samples, small clusters, singular covariance) while v2 uses 20 Newsgroups (18846 samples, 200-doc clusters, better conditioned). The v1 test computed R on raw features in Test 4; v2 correctly uses hidden activations. The v1 verdict was editorially overridden from INCONCLUSIVE to FALSIFIED; v2 lets the pre-registered criteria decide.

## Data Artifacts

- Code: `THOUGHT/LAB/FORMULA/v2/q15_bayesian/code/test_v2_q15_fixed.py`
- Results: `THOUGHT/LAB/FORMULA/v2/q15_bayesian/results/test_v2_q15_fixed_results.json`
- Embedding cache: `THOUGHT/LAB/FORMULA/v2/q15_bayesian/cache/` (3 .npy files)

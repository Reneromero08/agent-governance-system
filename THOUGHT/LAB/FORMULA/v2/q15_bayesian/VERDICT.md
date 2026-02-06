# Q15 Verdict

## Result: FALSIFIED

The full formula R = (E/grad_S) * sigma^Df does NOT have a genuine Bayesian interpretation. Any Bayesian-like correlation is driven entirely by the E component (mean pairwise cosine similarity), which trivially equals a function of likelihood precision. The sigma^Df term actively degrades the Bayesian connection and introduces sample-size dependence. R adds nothing beyond what E alone provides, and E's Bayesian connection is a trivial mathematical relationship, not a discovery.

## Full R vs Bayesian Quantities (Test 1)

25 STS-B clusters, encoded with all-MiniLM-L6-v2.

| Variant | vs lik_prec (rho) | vs sqrt(lik_prec) (rho) | vs post_prec (rho) |
|---------|-------------------|-------------------------|---------------------|
| R_full | +0.562 (p=0.003) | +0.562 (p=0.003) | +0.530 (p=0.006) |
| R_simple (E/grad_S) | +0.737 (p<0.001) | +0.737 (p<0.001) | +0.268 (p=0.194) |
| Trivial_R (1/grad_S) | -0.384 (p=0.058) | -0.384 (p=0.058) | +0.316 (p=0.124) |
| **E alone** | **+1.000 (p=0.000)** | **+1.000 (p=0.000)** | -0.072 (p=0.731) |

**Critical finding:** E alone correlates perfectly (rho=1.0) with likelihood precision. This is not surprising -- E is mean pairwise cosine similarity, and 1/trace(cov) is directly related to the tightness of the same embedding distribution. They are measuring the same thing.

**The hierarchy tells the story:**
- E alone: rho = 1.000 (perfect)
- E/grad_S: rho = 0.737 (adding 1/grad_S hurts by 0.263)
- (E/grad_S) * sigma^Df: rho = 0.562 (adding sigma^Df hurts by another 0.175)

Each additional term in the formula DEGRADES the Bayesian correlation. The Bayesian connection lives entirely in E. The formula's other components (grad_S, sigma, Df) are noise from a Bayesian perspective.

**Pre-registered criterion (rho > 0.7 with sqrt(lik_prec)): FAILED.** R_full achieves only 0.562.

## Intensive Property (Test 2)

3 large STS-B clusters (score tertiles), subsampled at N = 10, 20, 50, 100, 200.

| Quantity | Mean CV | Median CV | Max CV | Intensive (CV < 0.15)? |
|----------|---------|-----------|--------|------------------------|
| R_full | 0.2850 | 0.2591 | 0.3410 | NO |
| R_simple (E/grad_S) | 0.0635 | 0.0427 | 0.1159 | YES |
| 1/grad_S | 0.0526 | 0.0575 | 0.0799 | YES |
| Posterior precision | 0.9020 | 0.9016 | 0.9031 | NO (extensive, as expected) |

**Critical finding:** R_full is NOT intensive. The sigma^Df term introduces strong sample-size dependence (sigma via eigenvalue participation ratio changes with N; Df via eigenvalue decay slope changes with N). Both R_simple and the trivial 1/grad_S ARE intensive, confirming that E/grad_S stabilizes but sigma^Df does not.

R_full values by N for cluster 0:
- N=10: 0.070
- N=20: 0.102
- N=50: 0.136
- N=100: 0.152
- N=200: 0.147

This monotonic increase (tripling from N=10 to N=200) shows R_full is clearly NOT intensive.

**Pre-registered criterion (CV < 0.15): FAILED.** Mean CV = 0.285, nearly double the threshold.

## Gating Comparison (Test 3)

53 micro-clusters from STS-B, labeled high-quality (score > 4.0) or low-quality (score < 2.0). 50/50 train/test split.

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| R_full | 0.211 | 0.667 | 0.320 |
| R_simple | 0.211 | 0.667 | 0.320 |
| Trivial_R (1/grad_S) | 0.154 | 0.333 | 0.211 |
| **E alone** | **0.600** | **0.500** | **0.545** |
| **Bayesian posterior** | **0.600** | **0.500** | **0.545** |

R_full does outperform trivial 1/grad_S by +0.109 F1 (meeting the >5% criterion), but this advantage comes entirely from the E component, which dominates R_full. Both E alone and Bayesian posterior precision outperform all R variants by a large margin (+0.225 F1 over R_full).

**The R formula adds complexity but reduces gating quality compared to its own E component used directly.**

**Pre-registered criterion (R_full > trivial + 5% F1): PASSED**, but this is misleading because R_full is strictly worse than its own E component.

## Falsification Reproduction (Test 4)

California Housing, 32-unit single-hidden-layer neural network, 10 seeds, 20 data subgroups per seed.

| Metric | R_full mean rho | p-value | Significant? |
|--------|-----------------|---------|--------------|
| vs Hessian precision | -0.231 | 0.006 | YES (wrong sign) |
| vs Fisher information | -0.171 | 0.067 | NO |
| vs KL divergence | -0.171 | 0.067 | NO |

**Critical finding:** R_full shows a statistically significant NEGATIVE correlation with Hessian-based posterior precision. Higher R corresponds to LOWER Bayesian precision. This is the opposite of what the Bayesian interpretation claims.

For comparison, the trivial 1/grad_S shows a weak POSITIVE correlation with Hessian precision (mean rho = +0.219, p = 0.0002), which is in the correct direction.

The v1 falsification partially reproduces: the full formula does not positively correlate with standard Bayesian quantities in a neural network setting. The negative correlation found here is worse than the null result in v1 -- it suggests R is actively anti-correlated with Bayesian precision.

**Pre-registered criterion (falsification does NOT reproduce): TECHNICALLY PASSED** -- there is a significant correlation. But the correlation is negative, which contradicts the hypothesis even more strongly than no correlation would.

## Key Finding

**R does not have a genuine Bayesian interpretation. The appearance of one is a compositional artifact:**

1. E (mean pairwise cosine similarity) trivially correlates with likelihood precision because both measure embedding tightness. This is a mathematical relationship, not a discovery.

2. The formula's additional terms (grad_S, sigma^Df) actively DEGRADE the Bayesian connection:
   - E alone: rho = 1.000 with likelihood precision
   - E/grad_S: rho = 0.737 (worse)
   - (E/grad_S) * sigma^Df: rho = 0.562 (even worse)

3. The sigma^Df term makes R non-intensive (CV = 0.285), destroying the one claimed advantage of R as a quality gate.

4. In a proper Bayesian setting (neural network + Hessian), R_full is significantly NEGATIVELY correlated with posterior precision (rho = -0.231, p = 0.006).

5. For practical gating, E alone (F1 = 0.545) outperforms R_full (F1 = 0.320) by a wide margin.

**Bottom line:** The Bayesian interpretation was a misattribution. The E component has a trivial mathematical connection to likelihood precision (both measure distributional tightness). Building E into a more complex formula (R) dilutes this connection rather than strengthening it. The v1 "rescue" was correct that 1/std correlates with sqrt(1/std^2), but that tautology does not extend to the actual formula.

## Data

- **STS-B validation set**: 1500 sentence pairs, encoded with all-MiniLM-L6-v2 (384 dimensions)
- **California Housing**: 20,640 samples, 8 features, used for Hessian-based falsification
- **Clusters**: 25 score-binned clusters for Test 1; 3 tertile clusters for Test 2; 53 micro-clusters for Test 3
- **Neural network**: 1-hidden-layer (32 units), ReLU, MSE loss, Adam optimizer, 200 epochs
- **Seeds**: 42 (global); 10 seeds for Test 4

## Limitations

1. **BIC and det-based precision returned NaN.** For 384-dimensional embeddings with cluster sizes of 10-100, the covariance matrix is singular or near-singular, preventing BIC computation. This reduced Test 1 to only trace-based and bootstrap precision measures.

2. **Small cluster sizes.** STS-B validation has only 1500 samples. Score-based binning into 50 bins yields clusters of 10-100, which is small for 384-dimensional data. This affects covariance estimation quality.

3. **Single embedding model.** Only all-MiniLM-L6-v2 was tested. The README called for multiple models.

4. **Test 3 class imbalance.** 40 low-quality vs 13 high-quality groups. This asymmetry affects F1 interpretation.

5. **Test 4 uses feature embeddings, not sentence embeddings.** California Housing has 8 raw features, not high-dimensional embeddings. E and grad_S behave differently in 8-D vs 384-D spaces.

6. **The negative correlation in Test 4 may be a spurious artifact** of data ordering within train/test splits creating systematic patterns in the 20 contiguous subgroups. However, it is consistent across 10 random seeds, which argues against this.

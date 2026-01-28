# Question 25: What determines sigma? (R: 1260)

**STATUS: RESOLVED - SIGMA IS PREDICTABLE**

## Question
Is there a principled way to derive sigma, or is it always empirical?

## Pre-Registration
- **HYPOTHESIS**: Sigma is predictable from dataset properties (R^2 > 0.7)
- **FALSIFICATION**: If R^2 < 0.5, sigma is irreducibly empirical
- **DATA**: 22 synthetic datasets across 7 domains (NLP, market, image, graph, baseline, structure, tails)
- **METHODOLOGY**: Grid search for optimal sigma (minimizing bootstrap CV of R), then linear regression on log(sigma)

## Results Summary

**Date**: 2026-01-27

### Key Finding: SIGMA IS PREDICTABLE

Cross-validated R^2 = **0.8617** (exceeds 0.7 threshold)

### Predictive Formula

```
log(sigma) = 3.4560 + 0.9396 * log(mean_dist) - 0.0872 * log(effective_dim) - 0.0212 * eigenvalue_ratio
```

Or equivalently:
```
sigma = 31.7 * (mean_pairwise_distance)^0.94 * (effective_dim)^-0.09 * exp(-0.02 * eigenvalue_ratio)
```

### Interpretation

1. **Mean pairwise distance dominates**: Sigma scales almost linearly with the typical distance between points in the embedding space (exponent ~ 0.94). This makes physical sense - sigma is a "scale" parameter.

2. **Effective dimensionality has mild negative effect**: Higher intrinsic dimensionality slightly reduces optimal sigma, but the effect is weak (exponent ~ -0.09).

3. **Eigenvalue concentration barely matters**: The eigenvalue ratio (concentration on first PC) has almost no predictive power once distance is accounted for.

### Feature Set Comparison

| Features | R^2 (train) | R^2 (CV) |
|----------|-------------|----------|
| log_mean_dist, log_effective_dim, eigenvalue_ratio | 0.9555 | **0.8617** |
| log_mean_dist, log_std_dist | 0.9558 | 0.8613 |
| intrinsic_scale, log_intrinsic_scale | 0.6784 | 0.5179 |
| log_n_samples, log_n_dimensions | 0.3288 | 0.1387 |
| entropy, effective_dim, eigenvalue_ratio | 0.2409 | 0.0156 |

### Sigma Distribution Across Datasets

- **Min**: 4.15
- **Max**: 100.0
- **Mean**: 46.1
- **Std**: 33.0
- **Range ratio**: 24x

### Domain-Specific Patterns

| Domain | Mean Sigma | Std Sigma | N |
|--------|------------|-----------|---|
| NLP | 40.4 | 28.8 | 3 |
| Market | 54.2 | 29.7 | 3 |
| Image | 64.6 | 14.7 | 2 |
| Graph | 26.0 | 13.6 | 2 |
| Baseline | 41.1 | 34.9 | 4 |
| Structure | 43.3 | 36.4 | 6 |
| Tails | 62.5 | 37.5 | 2 |

Within-domain variance is high, confirming that domain type alone doesn't determine sigma - the scale properties matter more.

### Prediction Accuracy Examples

| Dataset | Actual Sigma | Predicted | Ratio |
|---------|--------------|-----------|-------|
| market_calm | 12.46 | 12.49 | 1.00 |
| lowrank_tight | 17.63 | 17.97 | 1.02 |
| multimodal_far | 100.00 | 102.32 | 1.02 |
| aniso_extreme | 15.70 | 15.24 | 0.97 |
| market_extreme | 70.67 | 121.54 | 1.72 (outlier) |

Most predictions within 25% of actual. Outliers occur with heavy-tailed or extreme distributions.

## Implications

### For the Formula R = (E/nabla_H) * sigma^Df

1. **Sigma is not magic**: It can be estimated from simple dataset statistics before any fitting.

2. **Practical guidance**: Compute mean pairwise distance, take its log, multiply by ~0.94, add ~3.5, exponentiate. This gives a good starting point for sigma.

3. **First-principles derivation possible**: The strong relationship to mean distance suggests sigma is fundamentally about "how far apart are your observations in embedding space" - a geometric quantity.

4. **Domain-agnostic**: The formula works across NLP, market, image, and graph embeddings, suggesting sigma captures something universal about data geometry.

## Remaining Questions

1. **Why 0.94 and not 1.0?** The slight deviation from linearity may indicate curvature effects or the distinction between L2 distance and the error metric used in R.

2. **Heavy tails**: The formula underperforms on heavy-tailed data. May need robust distance measures.

3. **Real-world validation**: These results are on synthetic data. Need to test on actual GEO, market, and NLP benchmarks.

## Files

- Test script: `experiments/open_questions/q25/test_q25_sigma.py`
- Results: `experiments/open_questions/q25/q25_results.json`

## Verdict

**HYPOTHESIS CONFIRMED**: Sigma is predictable from dataset properties with R^2_cv = 0.8617 > 0.7.

The primary determinant is the mean pairwise distance in embedding space. Sigma is essentially a scale parameter that should match the "natural" error scale of the data. This is not just curve-fitting - it has a clear geometric interpretation.

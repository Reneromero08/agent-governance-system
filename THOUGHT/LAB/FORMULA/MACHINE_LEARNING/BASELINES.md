# Baselines

Every ML experiment in this folder must compare against simpler or established
metrics.

## Required Baselines

1. `E`
   - mean pairwise cosine similarity
2. `grad_S`
   - standard deviation of pairwise cosine similarities
3. `1 / grad_S`
   - pure dispersion inverse
4. `sigma`
   - normalized participation ratio
5. `Df`
   - spectral complexity proxy
6. `effective_rank`
   - entropy-based or participation-ratio effective dimension
7. `isotropy_score`
   - anisotropy/uniformity proxy
8. `random`
   - shuffled labels or random scores

## Preferred Additional Baselines

- alignment / uniformity metrics from contrastive learning
- intrinsic dimensionality estimators
- class margin or centroid separation when labels exist
- neural-collapse metrics for supervised late-training regimes

## Decision Rule

If `R_simple` or `R_full` does not beat simpler baselines, the result is still
valid. The correct conclusion is that the formula is not currently useful in
that setting.

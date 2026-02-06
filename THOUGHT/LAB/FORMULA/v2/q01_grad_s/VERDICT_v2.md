# Q01 Verdict (v2 - Fixed Methodology)

## Result: FALSIFIED

grad_S (std of pairwise cosine similarities) is NOT the "correct" normalization for E. Dividing by grad_S adds no statistically significant predictive value beyond E alone across any of three tested architectures, and the partial correlation of grad_S with cluster quality (controlling for E) is non-significant in all three.

## Pre-Registered Criteria Applied

- **CONFIRM** required: E/grad_S outperforms >= 4/4 alternatives (p<0.01) across ALL 3 architectures AND partial corr of grad_S significant (p<0.05) in ALL architectures.
- **FALSIFY** required: E/grad_S loses to >= 3 alternatives in ANY architecture OR grad_S partial corr non-significant in ALL architectures.

**Outcome: grad_S partial correlation is non-significant in ALL 3 architectures (p = 0.316, 0.115, 0.478) --> FALSIFIED.**

## Per-Architecture Results

### Summary Table

| Model                        | dim | rho(R0) | rho(E) | R0 Wins (p<0.01) | Losses | Partial corr(grad_S\|E) p |
|------------------------------|-----|---------|--------|-------------------|--------|--------------------------|
| all-MiniLM-L6-v2             | 384 | +0.926  | +0.941 | 0/4               | 0/4    | 0.316 (n.s.)             |
| all-mpnet-base-v2            | 768 | +0.937  | +0.922 | 0/4               | 0/4    | 0.115 (n.s.)             |
| multi-qa-MiniLM-L6-cos-v1   | 384 | +0.938  | +0.937 | 1/4               | 0/4    | 0.478 (n.s.)             |

### All Spearman Correlations with Silhouette Score

**all-MiniLM-L6-v2:**

| Variant      | Spearman rho | p-value    |
|--------------|-------------|------------|
| E alone      | +0.941      | < 0.000001 |
| E/grad_S     | +0.926      | < 0.000001 |
| E/IQR        | +0.916      | < 0.000001 |
| E/MAD        | +0.907      | < 0.000001 |
| E/grad_S^2   | +0.832      | 0.000006   |

**all-mpnet-base-v2:**

| Variant      | Spearman rho | p-value    |
|--------------|-------------|------------|
| E/grad_S     | +0.937      | < 0.000001 |
| E alone      | +0.922      | < 0.000001 |
| E/MAD        | +0.886      | < 0.000001 |
| E/IQR        | +0.878      | < 0.000001 |
| E/grad_S^2   | +0.785      | 0.000041   |

**multi-qa-MiniLM-L6-cos-v1:**

| Variant      | Spearman rho | p-value    |
|--------------|-------------|------------|
| E/grad_S     | +0.938      | < 0.000001 |
| E alone      | +0.937      | < 0.000001 |
| E/IQR        | +0.917      | < 0.000001 |
| E/MAD        | +0.910      | < 0.000001 |
| E/grad_S^2   | +0.759      | 0.000103   |

### Bootstrap Comparison (R0 = E/grad_S vs each alternative, n=10000)

**all-MiniLM-L6-v2:**

| vs Alternative | Delta(rho) | p-value | 95% CI             |
|----------------|-----------|---------|---------------------|
| vs E/grad_S^2  | +0.099    | 0.097   | [-0.036, +0.315]    |
| vs E/MAD       | +0.019    | 0.334   | [-0.041, +0.109]    |
| vs E/IQR       | +0.009    | 0.407   | [-0.042, +0.075]    |
| vs E alone     | -0.020    | 0.816   | [-0.082, +0.033]    |

**all-mpnet-base-v2:**

| vs Alternative | Delta(rho) | p-value | 95% CI             |
|----------------|-----------|---------|---------------------|
| vs E/grad_S^2  | +0.157    | 0.027   | [+0.000, +0.421]    |
| vs E/MAD       | +0.055    | 0.075   | [-0.006, +0.204]    |
| vs E/IQR       | +0.061    | 0.108   | [-0.009, +0.227]    |
| vs E alone     | +0.017    | 0.326   | [-0.062, +0.107]    |

**multi-qa-MiniLM-L6-cos-v1:**

| vs Alternative | Delta(rho) | p-value | 95% CI             |
|----------------|-----------|---------|---------------------|
| vs E/grad_S^2  | +0.187    | 0.008   | [+0.026, +0.487]    |
| vs E/MAD       | +0.031    | 0.136   | [+0.000, +0.121]    |
| vs E/IQR       | +0.024    | 0.160   | [+0.000, +0.097]    |
| vs E alone     | +0.004    | 0.441   | [-0.046, +0.055]    |

### Partial Correlation Results (Test 2)

| Model                        | Partial corr(R0, sil \| E) | p       | Partial corr(grad_S, sil \| E) | p       |
|------------------------------|---------------------------|---------|--------------------------------|---------|
| all-MiniLM-L6-v2             | +0.162                    | 0.509   | -0.243                         | 0.316   |
| all-mpnet-base-v2            | +0.464                    | 0.045   | -0.373                         | 0.115   |
| multi-qa-MiniLM-L6-cos-v1   | +0.324                    | 0.176   | -0.173                         | 0.478   |

## Interpretation

### Finding 1: E alone is as good or better than E/grad_S

Across all three architectures, E (mean pairwise cosine similarity) alone achieves a Spearman correlation with silhouette score between +0.922 and +0.941. Dividing by grad_S does not significantly improve this. In the all-MiniLM-L6-v2 model, E alone actually has a HIGHER correlation (+0.941) than E/grad_S (+0.926).

The bootstrap tests confirm this: the comparison of R0 vs E alone yields p-values of 0.816, 0.326, and 0.441 -- far from significance in any architecture.

### Finding 2: grad_S adds no independent predictive value

The partial correlation tests show that grad_S, after controlling for E, has no significant relationship with cluster quality (silhouette score). The partial correlations are small and negative (-0.24, -0.37, -0.17) with p-values well above 0.05 in all three architectures. If anything, higher grad_S is weakly associated with LOWER cluster quality after controlling for E, but this effect is not significant.

This directly falsifies the claim that dividing by grad_S is "structurally forced." The data shows that grad_S carries no information about cluster quality beyond what E already captures.

### Finding 3: E/grad_S does beat E/grad_S^2 (but not the right comparison)

R0 = E/grad_S consistently outperforms E/grad_S^2 (precision-weighted), reaching significance in the multi-qa model (p=0.008). This merely shows that dividing by grad_S is better than dividing by grad_S^2 -- i.e., that over-normalizing is worse than moderate normalization. It does NOT show that dividing by grad_S is better than not normalizing at all.

### Finding 4: Results are consistent across architectures

The pattern is robust across:
- all-MiniLM-L6-v2 (384-dim, general purpose)
- all-mpnet-base-v2 (768-dim, larger model)
- multi-qa-MiniLM-L6-cos-v1 (384-dim, QA-optimized)

This rules out the possibility that the finding is an artifact of a single architecture.

### Contrast with v1 Verdict

The v1 test (using STS-B with similarity bins) found the same overall pattern: E/grad_S does not significantly outperform alternatives. The v1 bridge test additionally showed that E_gaussian and E_cosine are unrelated (rho = -0.098). The v2 fixed test strengthens this with:
- 3 architectures instead of 1
- Natural topic clusters instead of artificial similarity bins
- Silhouette score (geometric cluster quality) instead of human similarity ratings
- Partial correlation analysis isolating grad_S's independent contribution

Both tests agree: **FALSIFIED.**

## Methodology

1. **Data:** 20 Newsgroups (sklearn), 18,275 documents across 20 topic categories. Headers, footers, and quotes removed.
2. **Subsampling:** 200 documents per cluster (4,000 total per model) for formula computation. Same subsampled set used for silhouette scoring.
3. **Silhouette computation:** For each cluster, sampled 100 target docs + 1,900 background docs, computed silhouette_samples with cosine metric, averaged over target docs.
4. **Formula variants:** E/grad_S, E/grad_S^2, E/MAD, E/IQR, E alone.
5. **Correlation:** Spearman rank correlation between each variant and silhouette score across 20 clusters.
6. **Bootstrap:** 10,000 bootstrap resamples for pairwise comparison of R0 vs each alternative.
7. **Partial correlation:** Spearman partial correlation of R0 (and grad_S) with silhouette controlling for E, with t-test adjusted for n-3 degrees of freedom.
8. **Random seed:** 42 (fixed for reproducibility).

## Limitations

1. **Only 20 data points per architecture.** With 20 clusters, the bootstrap CIs are wide, reducing power to detect small differences. However, the CIs for R0 vs E alone span zero in all cases, meaning any real advantage of grad_S normalization is small at best.
2. **Silhouette score is one measure of cluster quality.** Other ground truth measures (NMI, ARI, retrieval metrics) might yield different results.
3. **Subsampling reduces sample size.** Using 200 docs per cluster instead of all available docs adds noise to the E and grad_S estimates.
4. **Only the E/grad_S core tested.** The full formula R = (E/grad_S) * sigma^Df was not tested. The fractal scaling might change the picture, but that is a separate question (not Q01).
5. **Domain scope.** Results apply to text embedding clusters from 20 Newsgroups. Other domains (images, audio, molecular) were not tested.

## Raw Data

Results JSON: `results/test_v2_q01_fixed_results.json`
Test code: `code/test_v2_q01_fixed.py`

Runtime: 1,927 seconds (~32 minutes) on CPU.

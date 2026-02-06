# Q01 Verdict

## Result: FALSIFIED

The hypothesis that E/grad_S is the "correct" normalization for E is not supported by the data. Two independent failures were observed.

## Evidence

### Test 1: Normalization Comparison (STS-B, 250 clusters)

| Normalization | Spearman rho | p-value   | Rank |
|---------------|-------------|-----------|------|
| E/grad_S^2    | 0.2652      | 2.15e-05  | 1    |
| **E/grad_S**  | **0.1367**  | **0.031** | **2**|
| SNR           | 0.0215      | 0.735     | 3    |
| E/IQR         | 0.0134      | 0.833     | 4    |
| E/MAD         | 0.0055      | 0.931     | 5    |
| Raw E         | 0.0040      | 0.950     | 6    |

E/grad_S beats 4/5 alternatives in raw |rho| ranking. However:
- **None of the wins are statistically significant** (all bootstrap p > 0.11).
- **E/grad_S^2 (precision-weighted) significantly outperforms E/grad_S** (bootstrap p = 0.9994 that grad_S is better; equivalently p < 0.001 that grad_S^2 is better). The 95% CI for the rho difference is [-0.182, -0.075], entirely below zero.
- E/grad_S itself only reaches p = 0.031 for correlation with human scores -- below the pre-registered p < 0.01 threshold.
- Significant wins at p < 0.01: **0 out of 5**.

### Test 2: Bridge (E_gaussian vs E_cosine)

| Metric | Value |
|--------|-------|
| Spearman rho | -0.0976 |
| Spearman p   | 0.124 (not significant) |
| Pearson r    | -0.1090 |
| Pearson p    | 0.086 (not significant) |

The correlation between E_gaussian (Gaussian kernel) and E_cosine (mean pairwise cosine similarity) is **effectively zero and slightly negative**. The two E definitions are unrelated on this data.

- E_cosine range: [0.0016, 0.071], mean = 0.029
- E_gaussian range: [0.692, 0.706], mean = 0.700

E_gaussian is nearly constant across all clusters (std = 0.003), providing almost no discriminative information. E_cosine varies more but operates in a completely different range.

### Verdict Logic (Pre-registered)

- CONFIRM requires: sig wins >= 4 at p<0.01 AND bridge rho > 0.8
- FALSIFY requires: losses >= 3 OR bridge rho < 0.3

**Bridge rho = -0.098 < 0.3 --> FALSIFIED**

Additionally, even if we only look at Test 1: zero significant wins at the pre-registered p < 0.01 threshold. The formula E/grad_S^2 outperforms E/grad_S with strong statistical significance.

## Data

- **Dataset:** STS-B (Semantic Textual Similarity Benchmark), train + validation splits
- **Source:** HuggingFace `glue/stsb`
- **N items:** 7,249 sentence pairs
- **N unique sentences:** 13,197
- **N clusters tested:** 250 (50 per similarity bin)
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensions)
- **Cluster sizes:** 10-30 sentences each

## Methodology

1. Loaded STS-B from HuggingFace (train + validation; test labels are hidden).
2. Encoded all unique sentences with all-MiniLM-L6-v2.
3. Grouped sentence pairs into 5 bins by human similarity score (0-1, 1-2, 2-3, 3-4, 4-5).
4. For each bin, randomly sampled 50 clusters of 10-30 sentences from sentences appearing in that bin's pairs.
5. For each cluster, computed six normalization variants of the formula.
6. Measured Spearman correlation of each normalization with mean human similarity score of the bin.
7. Used bootstrap resampling (10,000 iterations) for pairwise significance testing.
8. Separately computed E_cosine and E_gaussian for all 250 clusters and measured their correlation.

## Limitations

1. **Cluster construction is coarse.** Clusters are random subsets of sentences within a similarity bin, not natural semantic groups. A more sophisticated clustering approach (e.g., k-means on embeddings) might yield different results.
2. **Ground truth granularity.** Using bin-averaged human scores means each cluster gets a noisy aggregate label. Pair-level analysis would be more granular.
3. **Single embedding model.** Results may differ with other models (e.g., larger transformers, domain-specific models).
4. **Single dataset.** STS-B is one benchmark. Cross-domain validation (MTEB, clinical NLP, etc.) was not performed.
5. **E_gaussian formulation.** The per-dimension z-score approach used here is one interpretation. Other formulations of the Gaussian bridge might behave differently.
6. **Only R_simple tested.** The full formula R = (E/grad_S) * sigma^Df was not tested here -- only the E/grad_S core. The fractal scaling factor might change the outcome.

## Raw Results

```
Test 1 - Spearman correlations with human similarity:
  E/grad_S:     rho = 0.1367, p = 0.031
  Raw E:        rho = 0.0040, p = 0.950
  E/grad_S^2:   rho = 0.2652, p = 2.15e-05
  E/MAD:        rho = 0.0055, p = 0.931
  E/IQR:        rho = 0.0134, p = 0.833
  SNR:          rho = -0.0215, p = 0.735

Bootstrap pairwise (E/grad_S vs alternative, mean |rho| difference):
  vs Raw E:     +0.085 [-0.101, +0.169] p=0.136
  vs E/grad_S^2: -0.128 [-0.182, -0.075] p=0.999  <-- E/grad_S LOSES
  vs E/MAD:     +0.085 [-0.097, +0.169] p=0.136
  vs E/IQR:     +0.085 [-0.091, +0.164] p=0.116
  vs SNR:       +0.085 [-0.057, +0.220] p=0.126

Test 2 - Bridge correlation:
  Spearman: rho = -0.098, p = 0.124
  Pearson:  r = -0.109, p = 0.086
```

## Interpretation

Two findings stand out:

1. **E/grad_S^2 is the better normalization**, not E/grad_S. Dividing by the variance (grad_S^2) rather than the standard deviation (grad_S) produces nearly double the correlation with human judgments (0.265 vs 0.137). This is the precision-weighted form familiar from Bayesian statistics. The difference is highly significant.

2. **The theoretical E (Gaussian kernel) and operational E (cosine similarity) are unrelated.** The v1 derivation showing log(R) = -F was proven on E_gaussian. But E_gaussian and E_cosine show zero correlation on real embedding data. This means the theoretical justification (location-scale likelihood) does not transfer to the operational formula.

Together, these findings mean: (a) grad_S is not the optimal normalizer even within the cosine-similarity framework, and (b) the theoretical argument for why it should be grad_S does not connect to the actual computation.

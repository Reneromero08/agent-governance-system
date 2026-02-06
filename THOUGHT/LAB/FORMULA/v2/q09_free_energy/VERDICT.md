# Q09 Verdict

## Result: FALSIFIED

The hypothesis that log(R) = -F + const, where F is variational free energy, does not hold when R is computed using operational E (mean pairwise cosine similarity) on real embedding data. The correlation is negligible, R-gating underperforms simpler alternatives, and the one "passing" test (upper bound) is a dimensional artifact, not a genuine structural property.

---

## Test 1: log(R) vs -F Correlation

### Real Embedding Data (STS-B, all-MiniLM-L6-v2, 20 clusters of 40 sentences)

| Metric | log(R_simple) vs -F | log(R_simple) vs -NLE | log(R_full) vs -F |
|--------|--------------------|-----------------------|-------------------|
| Pearson r | **-0.1298** | -0.1387 | -0.2223 |
| Pearson p | 0.5855 | 0.5598 | 0.3462 |
| Spearman rho | **-0.2421** | -0.2406 | -0.3534 |
| Spearman p | 0.3038 | 0.3069 | 0.1264 |

**Interpretation:** The correlation between log(R) and -F is essentially zero. Pearson r = -0.13 is not statistically significant (p = 0.59). The sign is *wrong* -- the hypothesis predicts a positive correlation (log(R) = -F + const implies they move together), but the observed direction is weakly negative. This is consistent with R and F being unrelated quantities.

**Identity check:** If log(R) = -F + const, then log(R) + F should have zero variance. Observed: std(log(R) + F) = 5.35 with mean = -1300.86. The coefficient of variation is small (0.4%), but this is misleading: F values range from -1313 to -1293 (range ~20) while log(R) values range from -1.48 to -0.77 (range ~0.7). The "constant-ish" sum is dominated by F values -- log(R) contributes less than 0.1% of the magnitude. The quantities live on completely different scales.

**Pre-registered threshold: FALSIFY (r = -0.13 < 0.3)**

## Test 2: R-Gating vs Alternatives

Clusters classified as "good" (human similarity score > 3.5, n=6) vs "bad" (score < 2.0, n=7).

| Method | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|-----|----------|
| R > threshold | 0.5000 | 1.0000 | **0.6667** | 0.5385 |
| 1/variance > threshold | 0.6667 | 1.0000 | **0.8000** | 0.7692 |
| Raw E > threshold | 0.6667 | 1.0000 | **0.8000** | 0.7692 |
| Random baseline | 0.4827 | 0.5167 | 0.4668 | 0.5000 |

**Interpretation:** R-gating is the *worst* non-random method. Both 1/variance and raw E outperform R by 13.3 percentage points in F1 (a 20% relative improvement over R). This is the opposite of what the hypothesis predicts. The simple inverse-variance filter -- which R is supposed to improve upon -- is strictly better.

Note: R *does* beat random (F1 0.67 vs 0.47), so it has some signal. But the signal comes from E (cosine similarity), and dividing by grad_S (the std of pairwise similarities) actually *degrades* performance.

**Pre-registered threshold: R underperforms 1/var by 16.7%. This is in FALSIFY territory (R no better than 1/var within 5%), and in fact R is significantly worse.**

## Test 3: Upper Bound on Surprise

-log(R) >= Surprise for **20/20 clusters (100%)**.

**However, this result is a dimensional artifact and must be discounted.**

Here is why: In 384-dimensional embedding space, the surprise S = 0.5 * (d*log(2pi) + log(det(cov)) + avg_mahalanobis) yields *negative* values (approximately -1300) because the log-determinant of the covariance is very negative (the eigenspectrum drops off rapidly in high dimensions). Meanwhile, -log(R) is a small positive number (~1.1 to 1.5) because R is a ratio of cosine similarities near zero.

So the "bound" -log(R) >= S becomes approximately 1.2 >= -1300, which is trivially true for *any* positive number. This tells us nothing about whether -log(R) is a meaningful upper bound on surprise. A random positive constant like 42 would also "pass" this test with 100% success rate.

**For this test to be meaningful,** -log(R) and S would need to be on comparable scales, and the margin would need to be tight (small). The observed margins are ~1300, which is the full magnitude of the surprise itself. This is not an upper bound relationship; it is dimensional incompatibility.

**Honest assessment: VOID (test not informative due to scale mismatch)**

## Test 4: Non-Gaussian Distributions

Synthetic clusters in 50 dimensions with various distributions.

### Overall Correlation
| Metric | All distributions | Non-Gaussian only |
|--------|------------------|-------------------|
| Pearson r | 0.2091 | 0.2486 |
| Pearson p | 0.4929 | 0.4885 |
| Spearman rho | 0.1374 | 0.0788 |
| Spearman p | 0.6545 | 0.8287 |

### Per-Type (n=3 each, interpret with extreme caution)
| Distribution | Pearson r | Spearman rho |
|-------------|-----------|-------------|
| Bimodal | 0.5625 | 0.5000 |
| Skewed (log-normal) | **0.9928** | **1.0000** |
| Heavy-tailed (Student-t) | N/A (only 2 valid) | N/A |
| Uniform | N/A (only 2 valid) | N/A |
| Gaussian baseline | -0.5623 | -0.5000 |

**Interpretation:** The overall correlation is weak and non-significant. The skewed (log-normal) distribution shows a striking correlation of 0.99, but with only n=3 data points this has no statistical power -- three monotonically varying quantities will always show high correlation. The Gaussian baseline itself shows r = -0.56 (wrong sign), confirming the identity does not hold even for the distribution family where it "should" work according to the v1 derivation.

**The non-Gaussian test does not rescue the hypothesis.**

## Data

- **Dataset:** STS-B test split (mteb/stsbenchmark-sts), 1379 sentence pairs
- **Encoder:** all-MiniLM-L6-v2 (384 dimensions)
- **Clusters:** 20 clusters of 40 sentences (20 pairs), sorted by human similarity score
- **Synthetic data:** 15 clusters of 40 points in 50 dimensions (bimodal, skewed, heavy-tailed, uniform, Gaussian)
- **Random seed:** 42
- **Elapsed time:** 29.5 seconds

## Why This Fails (Diagnosis)

The v1 "identity" log(R) = -F + const was proven by defining E(z) = exp(-z^2/2), which IS the Gaussian kernel. Under that definition, R = E/std is algebraically equal to the Gaussian likelihood divided by the standard deviation. The "proof" was: log(Gaussian/std) = -GaussianNLL - log(std), which is trivially true.

The operational E (mean pairwise cosine similarity) has no such algebraic relationship to free energy. Specifically:

1. **E is bounded in [-1, 1]** while F can be any real number. There is no affine transformation that maps one to the other.
2. **E measures pairwise similarity** (a second-order statistic of angles) while F measures likelihood under a generative model (depends on the full covariance structure, not just average pairwise cosine).
3. **grad_S (std of pairwise cosines)** is not the scale parameter of any likelihood family. It is a measure of similarity dispersion, unrelated to the variance parameter that appears in the Gaussian free energy.
4. **R = E/grad_S** is a signal-to-noise ratio in cosine-similarity space. Free energy is a likelihood-based quantity in the full embedding space. These live in different mathematical worlds.

## Limitations

1. **Cluster size:** 40 sentences per cluster is modest. Larger clusters might show different behavior, though the fundamental scale mismatch would remain.
2. **Single encoder:** Only tested with all-MiniLM-L6-v2. Different encoders produce different embedding geometries.
3. **Gaussian generative model for F:** The free energy was computed under a Gaussian assumption. The embeddings may not be Gaussian, which affects F but should not affect the test logic (the hypothesis claims the identity holds generally).
4. **Regularization:** Covariance matrices were regularized with lambda=0.0001 to avoid singularity. This slightly inflates the determinant.
5. **Test 3 is uninformative:** The scale mismatch makes the upper bound test vacuous. A proper test would require defining surprise in the same space as R, but this would require specifying what generative model R implicitly uses -- which is exactly the unresolved question.
6. **Small sample for non-Gaussian tests:** Only 3 samples per distribution type. Statistical power is negligible.

## Conclusion

The Free Energy Principle connection to R was constructed by reverse-engineering E to equal the Gaussian kernel, making the identity tautological. When tested with the actual operational E (cosine similarity), the connection vanishes:

- Correlation is negligible (r = -0.13, wrong sign)
- R-gating is worse than simple alternatives (F1 16.7% below 1/variance)
- The upper bound test passes trivially due to dimensional incompatibility, not structural connection
- Non-Gaussian distributions show no meaningful correlation

**The claim that R-maximization equals surprise minimization is not supported by evidence when R uses its defined operational E.**

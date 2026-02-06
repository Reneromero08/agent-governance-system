# Q02 Verdict

## Result: FALSIFIED

The formula R = (E / grad_S) * sigma^Df fails the pre-registered falsification criteria. The modus tollens test fails on held-out real data, and only 3 of 5 component tests pass. The specific ratio form E/grad_S is not privileged over alternatives, and in fact performs worse than E alone.

## Evidence

### Test 1A: E Falsification -- PASS
- Spearman rho(cosine_similarity, human_score) = **0.8203** (p < 1e-300)
- Criterion: rho > 0.5
- N = 1379 STS-B test pairs
- **Interpretation:** Cosine similarity from all-MiniLM-L6-v2 is a strong proxy for semantic similarity. This is well-established in the literature and confirms that the E component (mean pairwise cosine similarity) captures a real signal. However, this is a property of the embedding model, not a property of the formula.

### Test 1B: grad_S Falsification -- PASS
- Spearman rho(grad_S, bootstrap_variance) = **0.8667** (p = 0.0012)
- 10 score buckets across STS-B test split
- Criterion: positive correlation with p < 0.05
- **Interpretation:** grad_S (std of pairwise cosine similarities) does track empirical uncertainty measured by bootstrap variance of cluster centroids. This validates grad_S as a dispersion measure. However, the correlation is across only 10 data points (score buckets), so the statistical power is moderate.

### Test 1C: Functional Form Comparison -- FAIL
- Ranking by |Spearman rho| with mean human scores across 20 clusters:
  1. R4 = E/(grad_S + 0.1): |rho| = 0.3203, p = 0.169
  2. R2 = E * exp(-grad_S): |rho| = 0.3008, p = 0.198
  3. E alone (baseline): |rho| = 0.2962, p = 0.205
  4. R3 = log(E)/(grad_S + 1): |rho| = 0.2602, p = 0.268
  5. **R0 = E/grad_S (original): |rho| = 0.2436, p = 0.301**
  6. R1 = E - grad_S: |rho| = 0.0150, p = 0.950
- Criterion: R0 must be in top 2. R0 ranks **5th out of 6**.
- **Critical finding: E alone (no grad_S division) outperforms R0.** Dividing by grad_S actually degrades the correlation with ground truth. The regularized version E/(grad_S + 0.1) and the exponential form E * exp(-grad_S) both outperform the claimed ratio form.
- **None of the correlations are statistically significant** (all p > 0.15). The formula has no detectable relationship with ground-truth quality in this cluster-level analysis.

### Test 1D: sigma Falsification -- FAIL
- Sigma values across 5 score buckets: [0.178, 0.193, 0.200, 0.259, 0.220]
- Coefficient of variation = **0.134** (criterion: CV > 0.2)
- Sigma does correlate with score bucket (rho = 0.90, p = 0.037), but its absolute variation is small: it ranges from 0.178 to 0.259. The participation ratio is nearly constant across semantic quality levels, providing minimal discriminative power.

### Test 1E: Df Falsification -- PASS (marginal)
- Df values across 5 score buckets: [0.907, 0.715, 0.962, 1.088, 1.038]
- Coefficient of variation = **0.137** (criterion: CV > 0.1, barely passes)
- Df does NOT correlate with score diversity (rho = 0.0, p = 1.0)
- **Interpretation:** Df varies enough to technically pass the CV criterion, but shows zero correlation with any interpretable property of the data. Its variation appears to be noise rather than signal.

### Component Summary: 3/5 PASS (1A, 1B, 1E)
The two passing components (E and grad_S) are essentially validations of cosine similarity and its standard deviation as statistical measures -- not unique to this formula. Df passes on a technicality (CV > 0.1) but carries no interpretable signal.

## Modus Tollens Test -- FAIL

### Calibration (training split, N=50 groups)
- Threshold T = 0.2996
- Quality minimum Q_min = 2.332 (10th percentile of scores when R > T)
- Pre-registered conditional: "R > 0.300 implies mean_human_score > 2.332"

### Held-out Test (test split, N=50 groups)
- Groups with R > T: 6 out of 50
- **Violations: 2 out of 6 (33.3%)**
  - Group with R=0.314 had mean_score=2.306 (below Q_min)
  - Group with R=0.302 had mean_score=2.286 (below Q_min)
- Criterion: violation rate < 10%
- **R vs ground-truth score correlation on test data: rho = -0.061, p = 0.675**
- R has essentially zero correlation with ground-truth quality on held-out groups. The modus tollens conditional is violated at a 33% rate, far exceeding the 10% threshold.
- **This is the core falsification result:** R does not reliably predict quality even in the weak conditional form "high R implies above-minimum quality."

## Adversarial Results -- PASS

| Case | E | grad_S | R_simple | Behavior | Correct? |
|------|-------|--------|----------|----------|----------|
| A: Near-duplicates | 0.850 | 0.100 | 8.461 | HIGH R | Yes (genuine agreement) |
| B: Random mixture | 0.058 | 0.080 | 0.727 | LOW R | Yes |
| C: Cross-domain | 0.160 | 0.151 | 1.061 | LOW R | Yes |

R behaves correctly on all 3 adversarial cases: high for genuine agreement, low for random/cross-domain mixtures. However, the adversarial test reveals the fundamental limitation: R cannot distinguish genuine agreement from echo chambers. Case A's sentences all say the same thing, which makes high R correct -- but a set of 20 sentences all confidently stating a wrong fact would produce the same R value.

## Data
- **STS-B (Semantic Textual Similarity Benchmark):** 5749 train pairs, 1379 test pairs
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)
- **Formula:** Shared implementation from `THOUGHT/LAB/FORMULA/v2/shared/formula.py`
- **Seed:** 42
- **Runtime:** 40.0 seconds

## Limitations

1. **Single embedding model.** All results use all-MiniLM-L6-v2. A different model might change the relative performance of formula variants. However, the formula claims to be model-agnostic, so this is a fair test.

2. **Cluster-level analysis.** Tests 1C-1E group pairs into score buckets and compute formula values per bucket. With 5-20 buckets, statistical power is limited. None of the correlations in Test 1C are individually significant at p < 0.05.

3. **Random group sampling.** The modus tollens test (Test 2) uses randomly sampled groups of 20-50 pairs. Different random seeds could yield different violation counts, though with only 6 groups exceeding the threshold, the test has low power. That said, a 33% violation rate vs. a 10% criterion is not a borderline failure.

4. **No sigma/Df cross-domain test.** The v2 test plan called for testing sigma stability across domains. STS-B covers multiple domains implicitly through its diverse sentence pairs, but a dedicated cross-domain test would be stronger.

5. **Adversarial tests are qualitative.** The adversarial cases test ordinal behavior (is R higher for agreement than noise?) but do not test calibration.

## Honest Assessment

The formula fails for two distinct reasons:

**Reason 1: The ratio form E/grad_S is not special.** When tested against alternatives (E-grad_S, E*exp(-grad_S), log(E)/(grad_S+1), E/(grad_S+0.1)), the original ratio form ranks 5th out of 6. E alone, without any grad_S term, performs better. This means the division by grad_S actively degrades the formula's predictive power on real data.

**Reason 2: R does not predict quality.** On held-out data, R shows near-zero correlation with ground-truth similarity scores (rho = -0.06, p = 0.68). The modus tollens conditional is violated 33% of the time. High R does not imply high quality.

The components that DO work (E correlating with human similarity, grad_S correlating with bootstrap variance) are simply properties of cosine similarity and its standard deviation -- they are not unique to or dependent on this formula. Assembling these components into R = E/grad_S produces a quantity with no predictive value beyond what E alone provides.

**Can the formula be falsified?** Technically yes -- the modus tollens test provides a clear falsification mechanism. But the formula's looseness (what counts as "observations"? what is the group boundary? how many observations are needed?) provides escape hatches. One could always argue the groups were chosen wrong, the embedding model was wrong, or the task was wrong. A truly falsifiable formula would specify these conditions precisely.

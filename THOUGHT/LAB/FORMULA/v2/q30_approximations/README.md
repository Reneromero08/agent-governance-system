# Q30: Fast R Approximations Exist with Bounded Error

## Hypothesis
Faster approximations of R exist (sampling-based, centroid-based, projected) that preserve gate behavior for large-scale systems. Specifically, random sampling achieves 100-300x speedup over exact O(n^2) pairwise computation with 100% gate decision accuracy.

## v1 Evidence Summary
- 6/6 tests passed.
- 8 approximation methods tested: exact, sampled (k=20, 50, 100), centroid, projected, Nystrom, streaming, combined.
- Best Pareto point: sampled_20 at 297.9x speedup with 100% gate accuracy (at n=500).
- 7/8 methods achieved 100% gate agreement with exact computation.
- Combined method had 83.3% accuracy due to over-aggressive projection.
- Scaling: sampled (fixed k) achieves O(n^0.18), effectively constant time.
- At n=1000, sampling achieves 390x speedup.
- R-value error up to 250% (gate decisions still correct due to binary threshold).

## v1 Methodology Problems
1. **Synthetic data only**: All benchmarks use synthetic embeddings with controlled agreement levels. Real embeddings have heavy-tailed similarity distributions, cluster structure, and other properties that may break sampling assumptions.
2. **Single gate threshold**: All tests use threshold = 0.8. Gate accuracy near a threshold depends on R-value distribution relative to that threshold. The 100% accuracy may hold only because test scenarios produce R values far from 0.8.
3. **No analytical error bounds**: All accuracy claims are empirical. No concentration inequalities, no confidence intervals, no worst-case analysis. The CLT justification hand-waves over the fact that pairwise similarities are correlated (they share embedding vectors).
4. **250% R-value error buried in limitations**: The actual R value can have 250% error from sampling. This is only mentioned in the Limitations section, not in key findings. For any non-gate use of R (ranking, monitoring, comparison), this error is a showstopper.
5. **No threshold-proximity testing**: The critical case -- R values near the gate threshold -- is not specifically tested. If R is at 0.79 (exact) vs 0.81 (sampled), the gate decision flips. How often does this happen?
6. **Correlated pairwise similarities**: The k*(k-1)/2 pairs from k sampled vectors are highly correlated because they share vectors. The effective sample size is much smaller than k*(k-1)/2. No analysis of this correlation structure is provided.

## v2 Test Plan

### Test 1: Real Data Gate Accuracy
- Compute exact and approximate R on real embedding data:
  (a) SNLI sentence pairs (all-MiniLM-L6-v2, n=100 to n=5000)
  (b) STS Benchmark pairs (n=100 to n=1000)
  (c) 20 Newsgroups document embeddings (n=500 to n=10000)
- For each approximation method and each k, report gate decision agreement with exact R at thresholds {0.3, 0.5, 0.8, 1.0, 2.0}.
- Report agreement rate separately for cases where exact R is within 20% of the threshold ("boundary cases") vs. far from threshold.

### Test 2: Threshold Proximity Analysis
- Identify all data points where exact R is within {5%, 10%, 20%} of the gate threshold.
- Report the gate decision flip rate for each approximation method at each proximity band.
- Determine the minimum k required to achieve 95% gate accuracy in the 5%-proximity band.

### Test 3: Analytical Error Bounds
- Derive a concentration bound for the sampling error: P(|R_sample - R_exact| > epsilon) as a function of k and the empirical similarity distribution.
- Account for pairwise correlation by computing the effective sample size.
- Validate the bound against empirical error distributions.
- Report: "For k=X, the gate decision error probability is at most Y at threshold Z."

### Test 4: Cluster-Structured Data
- Generate real-world cluster-structured embeddings by grouping documents by topic (20 Newsgroups).
- Test whether random sampling misses minority clusters.
- Compare random sampling vs. stratified sampling (sample proportionally from each cluster).
- Report gate accuracy for both sampling strategies on cluster-structured data.

### Test 5: R-Value Accuracy (Not Just Gate)
- For each approximation method and k, report:
  (a) Mean absolute error of R relative to exact R
  (b) Mean relative error of R
  (c) Spearman rank correlation between approximate R and exact R
  (d) Maximum observed error
- Determine the k required for 95th percentile R-value error < 10%, 20%, 50%.
- This addresses the 250% error problem by finding the k at which R-value accuracy becomes acceptable.

### Test 6: Speedup Under Real Conditions
- Benchmark wall-clock time for exact vs. approximate R computation on real hardware.
- Test with real embedding dimensions (384, 768, 1024).
- Include embedding computation time (not just similarity computation) to measure end-to-end speedup.
- Report speedup as a function of n, k, and embedding dimensionality.

## Required Data
- **SNLI** (~570K sentence pairs, HuggingFace)
- **STS Benchmark** (~8K pairs, HuggingFace)
- **20 Newsgroups** (~18K documents, sklearn.datasets)
- **Wikipedia random articles** (for large-n scaling tests)

## Pre-Registered Criteria
- **Success (confirm):** An approximation method achieves >= 95% gate decision accuracy on real data across at least 3 thresholds AND >= 10x speedup, including for boundary cases (R within 10% of threshold). Analytical error bound matches empirical error within 2x.
- **Failure (falsify):** No method achieves >= 90% gate accuracy on boundary cases with k < n/2 (i.e., you need to compute most of the exact similarities anyway), OR the analytical bound is too loose to be useful (> 10x the empirical error).
- **Inconclusive:** High accuracy on non-boundary cases but poor accuracy on boundary cases, or accuracy depends strongly on the specific threshold value chosen.

## Baseline Comparisons
- Exact pairwise computation (O(n^2) baseline)
- Centroid-only approximation (O(n), simplest possible)
- Random projection + exact computation (Johnson-Lindenstrauss)
- Locality-sensitive hashing (LSH) for approximate nearest neighbors
- Mini-batch computation (streaming, fixed memory)

## Salvageable from v1
- `test_q30_approximations.py`: The 8-method comparison framework is well-structured and reusable. The Pareto frontier analysis is clean.
- The `compute_r_fast()` implementation is a reasonable starting point for the sampling method.
- Speed benchmarks from v1 can serve as a reference for v2 comparisons on the same hardware.
- The finding that combined (sample + project) had lower accuracy (83.3%) is a genuine negative result worth preserving.

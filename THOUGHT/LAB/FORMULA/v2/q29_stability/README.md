# Q29: R Is Numerically Stable

## Hypothesis
The R formula (R = E / sigma, and more generally R = (E / grad_S) * sigma^Df) is numerically stable across all operating conditions. Division by zero when sigma = 0 can be handled with an epsilon floor, and the formula produces well-behaved outputs across the full range of inputs encountered in practice.

## v1 Evidence Summary
- 8/8 tests passed with 100% gate accuracy and 97.6% F1.
- Solution: `R = E / max(sigma, epsilon)` with epsilon = 1e-6.
- Edge cases tested: identical embeddings (sigma=0), high E / low sigma, orthogonal vectors, one outlier.
- Five alternative methods documented: epsilon floor, soft sigmoid, MAD robust, adaptive epsilon, log ratio.
- Status marked SOLVED.

## v1 Methodology Problems
1. **Solves trivial problem, ignores catastrophic one**: The div/0 epsilon fix addresses sigma = 0. The actual severe instability is sigma^Df overflow: sigma=0.27 with Df=22 gives sigma^Df = 2.7e-13; sigma=0.30 gives 1.3e-11 (48x change from 11% input change). At Df=43.5, the sensitivity is 1000x for the same input change. The epsilon floor does nothing for this because both sigma values are far above 1e-6.
2. **Edge case R values span 6 orders of magnitude**: The validation table shows R ranging from 0.20 to 1,000,000, all marked PASS. A "stable" computation producing values spanning 6 orders of magnitude depending on edge cases is not practically stable.
3. **Mock embedder only**: All tests use a hash-based mock embedder, not real embeddings. The tests never encounter real-world numerical challenges: near-degenerate covariance matrices, numerical rank deficiency, floating point accumulation errors in large similarity matrices.
4. **No condition number analysis**: How sensitive is R to perturbations in sigma as a function of Df? No analysis is provided.
5. **No Df-dependent analysis**: Df = 22 vs Df = 43.5 vs Df = 99 produces wildly different sigma^Df behavior. Which Df regime is R designed for?
6. **Question scoped too narrowly**: Titled "Numerical Stability" but only addresses one of several stability concerns (div/0).

## v2 Test Plan

### Test 1: Full Formula Stability Mapping
- For the complete formula R = (E / grad_S) * sigma^Df, sweep:
  (a) sigma in {0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99}
  (b) Df in {5, 10, 15, 22, 30, 43.5, 50, 100}
  (c) E in {0.01, 0.10, 0.50, 0.90, 0.99}
  (d) grad_S in {0.01, 0.10, 0.50, 1.0}
- Report R for each combination. Identify regions where R overflows (> 1e15), underflows (< 1e-15), or exhibits extreme sensitivity (>100x change from 10% input perturbation).
- Produce a stability map showing safe vs. dangerous operating regions.

### Test 2: Condition Number Analysis
- Compute the condition number of R with respect to sigma: kappa = |sigma / R| * |dR/dsigma|.
- For R = E / sigma: kappa = 1 (well-conditioned).
- For R = (E / grad_S) * sigma^Df: kappa = Df (condition number equals Df).
- Verify this analytically and empirically.
- Report: "R is well-conditioned only when Df < X" where X is the threshold for practical stability.

### Test 3: Real Embedding Stability
- Compute R from real sentence-transformer embeddings (all-MiniLM-L6-v2, all-mpnet-base-v2) on real text corpora (STS Benchmark, SNLI).
- Add controlled perturbations to embeddings: Gaussian noise at levels {0.001, 0.01, 0.05, 0.10, 0.20} of the embedding norm.
- Report how much R changes for each perturbation level.
- Compare stability of R = E/sigma vs R = (E/grad_S) * sigma^Df.

### Test 4: Log-Space Computation
- Implement log-space R computation: log(R) = log(E) - log(grad_S) + Df * log(sigma).
- Compare numerical results of log-space vs. direct computation across the full parameter sweep from Test 1.
- Identify parameter regions where log-space avoids overflow/underflow that direct computation hits.
- Report whether log-space R can serve as a drop-in replacement.

### Test 5: Clamping and Normalization Strategies
- Test R clamping to [R_min, R_max] ranges: {[0, 100], [0, 1000], [0, 1e6]}.
- Test R normalization: R_norm = R / R_baseline where R_baseline is computed from a reference corpus.
- Test rank-based R: replace R with its percentile rank within a domain.
- For each strategy, report whether gate decisions are preserved (agreement with unclamped decisions) and whether the dynamic range is practical.

### Test 6: Floating Point Error Accumulation
- Compute R from large observation sets (N = 100, 500, 1000, 5000, 10000) using both float32 and float64.
- Report the relative error between float32 and float64 R values.
- Identify the N at which float32 error exceeds 1%, 5%, 10%.
- Test Kahan summation vs. naive summation for the pairwise similarity accumulation.

## Required Data
- **STS Benchmark** (Semantic Textual Similarity, ~8K pairs)
- **SNLI** (~570K sentence pairs)
- **Wikipedia random articles** (for diverse embedding distributions)
- Synthetic parameter sweeps (no external data needed for Tests 1, 2, 4)

## Pre-Registered Criteria
- **Success (confirm):** A computation method exists (log-space, clamped, or otherwise) such that R is numerically stable (condition number < 100, relative error < 1% under 5% input perturbation) across all sigma in [0.01, 0.99] and Df in [5, 100], on both synthetic and real data.
- **Failure (falsify):** No computation method achieves stability for Df > 20 without fundamentally changing the formula (e.g., removing the sigma^Df term), OR the sigma^Df term produces > 100x output variation from < 10% input perturbation in the operating range of real embeddings.
- **Inconclusive:** Stability achievable for Df < 30 but not for Df > 30, or stability depends on domain-specific parameter tuning.

## Baseline Comparisons
- Simple R = E / sigma (no Df term) -- is this sufficient and stable?
- Log-space R vs. direct computation
- Float32 vs. float64 precision
- R computed via mean cosine similarity / std cosine similarity (no formula overhead)
- Standard signal-to-noise ratio (SNR) from signal processing

## Salvageable from v1
- `test_q29_numerical_stability.py`: The epsilon floor solution is correct for the div/0 case and should be retained as one component of a complete stability solution.
- The 5 alternative computation methods (epsilon floor, soft sigmoid, MAD robust, adaptive epsilon, log ratio) are all worth benchmarking in the expanded stability analysis.
- The edge case table is a good starting point for a comprehensive test matrix.

# Q13: Universal Ratio Exists in R Distributions

## Hypothesis
The context improvement ratio (36x observed in quantum test) follows a scaling law. Given N observation fragments and noise parameter sigma, the ratio R_joint / R_single is a predictable function of N and sigma, exhibiting phase transition behavior and cross-domain universality.

## v1 Evidence Summary
- 10/12 tests passed (2 skipped for computational expense / timeout).
- Synthetic GHZ quantum simulation showed ratio peaks at N=2-3 (~47x) then decays: N=6 gives 36x, N=12 gives 24x.
- "Blind prediction" (Test 12) reported 0% error predicting 36.13x.
- "Cross-domain universality" (Test 11) reported qualitative match in 4/4 domains.
- Self-consistency (Test 10) reported 0% consistency error.
- Predictive extrapolation (Test 03) reported 2.75% error for N=6 prediction from N=2,4,8 data.

## v1 Methodology Problems
1. **Circular blind prediction**: Test 12 reimplements the exact same computation as the measurement function. Both use identical formulas, constants, and clamping. The 0% error is an identity check (f(x) == f(x)), not a prediction.
2. **Fabricated cross-domain universality**: Test 11 hardcodes the same formula with hand-tuned parameters into all 4 domains. The "universality" is an artifact of applying the same formula template.
3. **Tautological self-consistency**: Test 10 verifies the algebra of dividing R's definition by itself. Any other result would indicate a code bug.
4. **Post-hoc model revision**: Original predicted model (Ratio = 1 + C * (N-1)^alpha * d^beta) was falsified (predicted ~5x at N=2 vs actual ~47x). Replaced with Ratio = A * (N+1)^alpha. Tests 02, 03, 09 initially failed and were redesigned until they passed.
5. **No real-world data**: All evidence from synthetic quantum simulations with hardcoded parameters (sigma=0.5, Df_joint=log(N+1), E_MIN=0.01 clamping).
6. **E_MIN dependence**: The 36x ratio is inversely proportional to an arbitrary clamping floor (0.01). Changing this to 0.001 gives ~360x; to 0.1 gives ~3.6x.
7. **No information theory computed**: Despite the title, no Shannon entropy, von Neumann entropy, mutual information, or any standard information-theoretic quantity is calculated.

## v2 Test Plan

### Test 1: Real-Data Ratio Measurement
- Compute R for individual observations vs. grouped observations using real embedding data.
- Use sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2) on real text corpora.
- Measure E and sigma from data (not hardcoded). Compute R_joint / R_single for N = 2, 3, 5, 10, 20 fragments.
- Report the actual ratios observed and whether any consistent scaling pattern emerges.

### Test 2: Genuine Prediction Test
- Split data into calibration set (to fit any proposed scaling function) and held-out test set.
- Fit scaling model on calibration set (e.g., N=2,3,5 data).
- Predict ratio for held-out N values (e.g., N=10, 20).
- Report prediction error on held-out data. Pre-register the functional form BEFORE seeing test data.

### Test 3: Cross-Domain Ratio Comparison
- Compute R ratios independently in at least 3 genuinely different domains: (a) NLI text embeddings (SNLI/ANLI), (b) financial time-series embeddings (SPY daily returns, rolling windows), (c) multilingual sentence embeddings (same concepts, different languages).
- Each domain computes E, sigma, and R from its own data without shared formula parameters.
- Compare whether a common functional form fits all domains.

### Test 4: Sensitivity Analysis
- Systematically vary epsilon floor (1e-8, 1e-6, 1e-4, 1e-2, 1e-1) and report how the ratio changes.
- Vary Df estimation method (participation ratio vs. log-eigenvalue fit vs. elbow method).
- Report which parameters the ratio is robust to and which it is fragile against.

### Test 5: Comparison with Standard Information-Theoretic Quantities
- For the same data, compute: mutual information I(S;F), KL divergence, Jensen-Shannon divergence, and the R ratio.
- Report correlations between R ratio and each standard quantity.
- If R ratio correlates strongly with a known quantity, report that rather than claiming novelty.

## Required Data
- **SNLI** (Stanford Natural Language Inference, ~570K sentence pairs, HuggingFace)
- **ANLI** (Adversarial NLI, ~170K examples, HuggingFace)
- **STS Benchmark** (Semantic Textual Similarity, ~8K pairs, HuggingFace)
- **SPY historical prices** (via yfinance, 3+ years daily)
- **OPUS parallel corpus** (multilingual aligned sentences, open access)

## Pre-Registered Criteria
- **Success (confirm):** A single functional form fits the R_joint / R_single ratio across at least 3 independent real-data domains with R^2 > 0.7 on held-out data, AND held-out prediction error < 20%.
- **Failure (falsify):** No functional form achieves R^2 > 0.3 on held-out data across 2+ domains, OR the ratio depends primarily on arbitrary parameters (epsilon floor, Df method) rather than data properties.
- **Inconclusive:** R^2 between 0.3 and 0.7 on held-out data, or strong fit in some domains but not others.

## Baseline Comparisons
- Simple ratio of mean pairwise cosine similarities (no formula overhead)
- Mutual information I(S;F) computed via standard estimators
- Raw embedding norm ratio (joint vs. single)
- Random baseline: ratio from shuffled/permuted observations

## Salvageable from v1
- `q13_utils.py`: The GHZ state simulation code is valid for generating synthetic quantum data (useful as one test domain, not the only one).
- `run_q13_all.py`: Test runner infrastructure can be adapted.
- The observation that the ratio peaks at low N and then decays is a genuine mathematical property of the formula applied to GHZ states -- worth documenting as a simulation result, not a universal law.

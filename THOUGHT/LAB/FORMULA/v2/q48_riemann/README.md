# Q48: Eigenvalue Statistics Connect to Riemann Zeta

## Hypothesis
The eigenvalue spectrum of semantic embedding covariance matrices follows the same universal statistics as Riemann zeta zeros. Specifically: eigenvalue spacings match GUE (Gaussian Unitary Ensemble) statistics as predicted by the Montgomery-Odlyzko law, a spectral zeta function zeta_sem(s) = sum(lambda_k^(-s)) exhibits properties analogous to the Riemann zeta function (functional equation, critical line, zero distribution), and there exists a conservation law Df * alpha = 8e connecting the participation ratio and spectral decay exponent.

## v1 Evidence Summary
- **GUE spacing hypothesis REJECTED.** All 3 tested models (MiniLM, MPNet, GloVe) matched Poisson statistics, not GUE. KL divergence: GUE 1.46-2.82, Poisson 0.37-0.56. No level repulsion detected.
- Cumulative variance followed exponential saturation C(k) = a * (1 - exp(-b*k)) + c with R^2 = 0.994-0.999 across 3 models.
- Spectral zeta function zeta_sem(s) was computed. Critical exponent sigma_c = 1/alpha found at ~2.09 (MiniLM).
- Functional equation test FAILED: all symmetry tests show "is_symmetric: False" with CV > 2.6.
- Euler product test FAILED: "No Semantic Primes -- ADDITIVE Structure."
- No real zeros found. No complex near-zeros found.
- Df * alpha product: mean = 21.84, CV = 2.69% across 6 models. 8e = 21.746 was selected as best-matching constant from 5 candidates (7*pi, 22, e^3, 8e, pi^2*2).
- Alpha averaged ~0.5 across transformer models but GloVe-100 had alpha = 0.84 (68% deviation from 0.5).

## v1 Methodology Problems
1. **Core hypothesis falsified, then bait-and-switched.** GUE spacing was the testable Riemann connection. It was cleanly rejected. The pivot to "conservation law" and "critical line" retained the "Riemann" branding despite having no mathematical connection to the Riemann zeta function.
2. **Spectral zeta function has none of the Riemann zeta's properties.** No functional equation (tested, failed). No Euler product (tested, failed). No zeros (tested, none found). No analytic continuation (code claims it but just computes the direct sum). The connection is purely nominal.
3. **8e is post-hoc.** Five candidate constants were tested; 8e was selected as the best fit. 7*pi = 21.991 and 22 are also within 1%. The constant 4*pi*sqrt(3) = 21.77 actually fits better. With enough candidate expressions, matching to <1% is expected.
4. **alpha = 0.5 is common.** Power-law exponents near 0.5 appear throughout statistics (Marchenko-Pastur edge scaling, 1/f noise covariance, Zipf-derived spectra). Sharing the numerical value 0.5 with the Riemann critical line is coincidence, not connection.
5. **Fixed vocabulary of 74 words.** All models were tested on exactly 74 words. The covariance matrices are at most 74x74. The eigenvalue structure is heavily constrained by this fixed dimensionality. "Universality" may be an artifact of fixed sample size.
6. **Incorrect unfolding procedure.** Spacing statistics used global mean normalization instead of local density estimation, biasing toward Poisson-like statistics. The GUE rejection may be partially artifactual (though the true answer is likely still Poisson for covariance matrices).
7. **Escalating language for weakening evidence.** The strongest result (GUE rejection) got honest treatment. The weakest results (alpha ~0.5, Df*alpha ~22) got language like "BREAKTHROUGH" and "NUMERICAL IDENTITY." The evidence-to-rhetoric ratio is inverted.

## v2 Test Plan

### Test 1: Correct GUE Spacing Analysis
- Repeat eigenvalue spacing statistics with PROPER spectral unfolding (local polynomial density estimation, not global mean).
- Test on covariance matrices from 500+ words (not 74) to reduce finite-size effects.
- Compare to Poisson, GUE, GOE, and Marchenko-Pastur distributions using KL divergence and Kolmogorov-Smirnov tests.
- Report full spacing distribution histograms with confidence bands.

### Test 2: Vocabulary Size Dependence of Df * alpha
- Compute Df * alpha using vocabulary sizes of 50, 100, 200, 500, 1000, 5000 words.
- Use 5+ random vocabulary subsets at each size to get error bars.
- Test whether Df * alpha is constant, increases, decreases, or follows a specific function of N.
- If Df * alpha depends on N, the "conservation law" is a finite-size artifact.

### Test 3: Architecture Independence
- Test Df * alpha on genuinely independent architectures: count-based (PMI matrix), word2vec (skip-gram), GloVe (matrix factorization), BERT (transformer MLM), GPT (transformer CLM), CLIP (contrastive), and at least one non-English model trained independently.
- For each architecture, test on 3+ vocabulary subsets.
- Report per-architecture Df * alpha with confidence intervals.
- A genuine conservation law should hold across all architectures, not just within the transformer family.

### Test 4: Spectral Zeta Function Properties
- Compute zeta_sem(s) with rigorous numerical methods.
- Attempt actual analytic continuation (not just the direct sum at real s).
- Systematically search for zeros in the complex plane using contour integration.
- Test the functional equation with proper error analysis.
- If none of these yield Riemann-like properties, conclude the connection is nominal.

### Test 5: Null Distribution for Df * alpha
- Generate 10,000 random matrices with the same rank, condition number, and approximate spectral shape as real embedding matrices (structure-preserving randomization, not i.i.d. Gaussian).
- Compute the distribution of Df * alpha under this null.
- Compute p-value: what fraction of null matrices produce Df * alpha within 5% of the observed mean?
- A proper Monte Carlo test with an appropriate null, not the failed p=0.55 test from v1.

### Test 6: Candidate Constant Discrimination
- Pre-register exactly 3 candidate constants: 8e, 7*pi, 22.
- Collect Df * alpha measurements on 100+ genuinely independent model+vocabulary combinations.
- Apply Bayesian model comparison (Bayes factor) to determine which constant (if any) is preferred.
- If no constant is significantly preferred, report "Df * alpha ~ 22 (approximate)" as the honest conclusion.

## Required Data
- **Pre-trained embeddings:** GloVe (50d, 100d, 200d, 300d), Word2Vec (300d), FastText (300d), BERT-base (768d), GPT-2 (768d), CLIP text encoder (512d), XLM-R (768d)
- **Vocabulary lists:** Standard frequency-ranked lists from 50 to 5000 words, multiple random subsets
- **Non-English models:** Independent Chinese (not translated), Arabic, Japanese embedding models
- **Random matrix baselines:** Marchenko-Pastur theory for analytical comparison
- **Riemann zero tables:** First 10,000 non-trivial zeros from Odlyzko's published tables

## Pre-Registered Criteria
- **Success (GUE connection):** Properly unfolded eigenvalue spacings match GUE distribution (KS test p > 0.05) for at least 3 independent architectures.
- **Failure (GUE connection):** All spacings match Poisson (KS p > 0.05 for Poisson, p < 0.01 for GUE).
- **Success (conservation law):** Df * alpha is constant (CV < 5%) across vocabulary sizes 50-5000 AND across all tested architectures. Bayes factor > 10 for one specific constant over alternatives.
- **Failure (conservation law):** Df * alpha depends on vocabulary size (regression slope significantly nonzero, p < 0.01) OR CV > 15% across architectures.
- **Inconclusive:** GUE/Poisson not clearly distinguished, or Df * alpha constant within transformer family but not across architectures.

## Baseline Comparisons
- **Marchenko-Pastur distribution** for random matrix eigenvalue spacings (proper null for covariance matrices).
- **Poisson spacing** (independent eigenvalues -- no correlations).
- **GUE/GOE spacing** (quantum chaos / random matrix universality).
- **i.i.d. Gaussian random matrices** matched for dimension and sample size (baseline Df * alpha).
- **Structure-preserving randomized matrices** (eigenvalue-matched but eigenvector-randomized).

## Salvageable from v1
- test_q48_riemann_bridge.py has working GUE/Poisson comparison code, though the unfolding procedure needs correction.
- test_q48_spectral_zeta.py has a basic spectral zeta implementation that can be extended with proper analytic continuation.
- test_q48_universal_constant.py has Df and alpha computation code that works correctly.
- The clean GUE rejection is a valuable negative result that should be reported prominently in v2.
- Cross-model eigenvalue loading infrastructure (multiple architectures) is reusable.

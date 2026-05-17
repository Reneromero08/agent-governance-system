# Q49 Verification Report: Df * alpha = 8e

**Date:** 2026-05-17
**Status:** FALSIFIED
**Reviewer:** Fresh verification from v1 code + new methodology

---

## Claim Under Test

The product of participation ratio (D_f) and eigenvalue decay exponent (alpha), computed from the covariance matrix of embedding dimensions across N vocabulary words, is a universal conservation law equal to 8e = 21.746 across all trained embedding models.

---

## Method

v1 code (`test_q49_falsification.py`) was run verbatim. Then a vocabulary-size sweep was added (N = 20 to 150 words) using the identical computational pipeline: `np.cov(centered.T)` → eigenvalue decomposition → D_f = (Σλ)² / Σλ², alpha = -slope of log-log polyfit on first half of eigenvalues. Random-baseline normalization was applied by computing matched random matrices (i.i.d. Gaussian, same N and D) at each N.

Models tested: all-MiniLM-L6-v2 (D=384), all-mpnet-base-v2 (D=768).

---

## Results

### 1. v1 code run verbatim

v1 falsification battery result: **2 of 4 tests FAILED.**

- T1.1 (Random matrix baseline): FAILED — random matrices also converge (to 14.5, not 21.7, but low CV = 2.6% means convergence alone is not evidence)
- T1.2 (Permutation): Passed
- T1.3 (Vocab independence at N=75): Passed — different words at same N=75 give CV = 1.4%
- T1.4 (Monte Carlo specialness): **FAILED — p = 0.534.** 53% of random constants in [15,30] fit the data as well as 8e. 8e is not statistically distinguishable from a random number.

### 2. N-dependence

D_f × alpha is a function of vocabulary size N, not a constant:

| N | MiniLM product | MPNet product | Random baseline |
|---|---------------|---------------|-----------------|
| 20 | 4.38 | 4.48 | 2.68 |
| 30 | 7.46 | 7.41 | 4.84 |
| 50 | 15.07 | 15.58 | 8.84 |
| **75** | **21.85** | **21.81** | **14.40** |
| 100 | 28.31 | 28.26 | 20.14 |
| 150 | 39.91 | 39.47 | 32.14 |

Linear fit: product ≈ 0.26 × N + C (R² > 0.99). The product equals 8e = 21.75 only at N ≈ 75, which is the vocabulary size hardcoded throughout all v1 tests.

### 3. Random-baseline normalization

After subtracting the random-noise floor (matched N, D, i.i.d. Gaussian), the semantic signal is the delta (real - random):

| N | MiniLM delta | MPNet delta |
|---|-------------|------------|
| 30 | 2.61 | 3.97 |
| 75 | 7.45 | 10.46 |
| 100 | 8.17 | 11.71 |
| 150 | 7.77 | 12.44 |

The delta does not converge to 8e or any other constant. It varies with N and model.

### 4. Root cause

The covariance matrix `np.cov(centered.T)` has shape D×D estimated from N samples. When N < D, the eigenspectrum is dominated by sample-size effects (rank = N-1). As N increases, more eigenvalues become non-zero, the participation ratio grows, and the product grows. This is true for both real embeddings and random noise — it's a property of eigenvalue spectra in the under-sampled regime, not a semantic conservation law.

The v1 tests never varied N. Test 1.3 ("vocabulary independence") only randomized which words at fixed N=75. The "constant" 8e is simply where the product curve crosses ~22 for these specific models at N=75.

---

## Verdict

**FALSIFIED.** D_f × alpha is not a universal conservation law. The value 8e = 21.746 is an artifact of measuring at the specific vocabulary size N ≈ 75 in the under-sampled covariance regime. No normalization recovers a universal constant.

- v1 adversarial verdict (Phase 4, verdict_4_Q49.md) rating "FAIL — Numerology with post-hoc rationalization" is **sustained**.
- Real embeddings do carry ~50-90% more D_f × alpha than random noise at matched N, but the gap is N-dependent and model-specific. Not a law. Not a constant. Not 8e.

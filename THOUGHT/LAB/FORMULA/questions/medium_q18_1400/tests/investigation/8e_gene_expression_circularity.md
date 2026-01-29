# 8e Gene Expression Test Circularity Investigation

**Date**: 2026-01-25
**Investigator**: Claude Opus 4.5
**Status**: CONFIRMED CIRCULAR/FAKE

---

## VERDICT

**YES - The 8e gene expression test is CIRCULAR and FAKE.**

The test does NOT measure whether gene expression data naturally exhibits the 8e conservation law. Instead, it uses a grid search to FIND parameters that produce Df x alpha = 21.746, then claims this as a "discovery."

---

## Evidence Summary

### 1. The Smoking Gun: Grid Search Optimization (Lines 559-584)

```python
# Grid search over decay parameters - expanded range
for tau_factor in np.linspace(0.05, 0.50, 50):
    k = np.arange(1, n_dims + 1)
    tau = n_dims * tau_factor

    # Exponential decay spectrum
    eigenvalues = np.exp(-k / tau)
    eigenvalues = eigenvalues + 0.001  # Floor
    eigenvalues = eigenvalues / np.sum(eigenvalues) * n_dims

    # Compute Df and alpha
    Df = compute_participation_ratio(eigenvalues)
    alpha = compute_spectral_decay(eigenvalues)
    df_x_alpha = Df * alpha

    error = abs(df_x_alpha - target_8e) / target_8e   # <-- OPTIMIZING TO HIT 8e!

    if error < best_error:
        best_error = error
        best_result = {...}
```

This is **explicit optimization to minimize deviation from the target value 8e (21.746)**. The test:
1. Creates 50 different artificial eigenspectra with varying tau parameters
2. Computes Df x alpha for each
3. Selects whichever combination produces a result closest to 21.746
4. Reports this as the "measured" value

### 2. Gene Expression Data is NOT Used

The actual gene expression data (`data.human_expression`) is **completely ignored** in the 8e conservation test. Look at `test_8e_conservation(data: GeneExpressionData)`:

- The function receives `data` as a parameter
- But the data is **never used** to compute the eigenspectrum
- Instead, the eigenspectrum is **artificially constructed** using `np.exp(-k / tau)`
- The only place gene expression appears is in a "verification" step (lines 599-617) where synthetic data is generated FROM the artificial eigenspectrum

The so-called "verification" (lines 592-617) generates new random expression data from the pre-optimized eigenspectrum using:
```python
L = np.linalg.cholesky(cov_matrix + np.eye(n_dims) * 1e-8)
expression = np.zeros((n_genes_verify, n_dims))
for i in range(n_genes_verify):
    expression[i, :] = L @ rng.standard_normal(n_dims)
```

This is circular: you're generating data from your artificially-designed eigenspectrum and then "verifying" that the data has the eigenspectrum you designed.

### 3. The Helper Function Reveals Intent

`generate_8e_structured_eigenspectrum()` (lines 484-525) is explicitly designed to:

```python
"""
Generate eigenvalue spectrum that satisfies Df * alpha = 8e.
...
We solve for alpha that gives Df * alpha ~ 8e.
"""
```

The function's documented purpose is to CREATE spectra that satisfy 8e, not to DISCOVER whether data naturally satisfies 8e.

---

## Comparison: Neural 8e Test (HONEST)

The neural test (`neural_scale_tests.py`, lines 502-572) is **fundamentally different**:

### Neural Test Methodology:
```python
# Create embeddings from EEG (trial-averaged ERPs flattened)
embeddings = []
for c in range(n_concepts):
    avg_erp = np.mean(eeg_data[c], axis=0).flatten()
    embeddings.append(avg_erp)

embeddings = np.array(embeddings)

# Compute covariance matrix FROM THE ACTUAL DATA
cov_matrix = np.cov(embeddings.T)

# Compute eigenvalues FROM THE ACTUAL DATA
eigenvalues = np.linalg.eigvalsh(cov_matrix)
```

The neural test:
1. Takes the ACTUAL EEG data
2. Computes the ACTUAL covariance matrix from that data
3. Extracts the ACTUAL eigenvalues
4. Computes Df and alpha from those eigenvalues
5. Checks if Df x alpha = 8e

**Result**: Neural test FAILS (Df x alpha = 58.2, not 21.746), with 168% deviation.

This is an **honest test** that reveals 8e does NOT hold in raw neural data.

---

## What the Tests Actually Show

| Test | Uses Real Data? | Uses Grid Search? | Result | Honest? |
|------|----------------|-------------------|--------|---------|
| Neural 8e | YES (EEG) | NO | FAIL (58.2) | YES |
| Gene 8e | NO | YES (to hit 21.746) | PASS (22.68) | NO |

The neural test's failure is a **legitimate scientific finding**: 8e conservation may be specific to trained semiotic embedding spaces, not raw neural signals.

The gene test's success is **fake**: it's engineered to succeed by searching for parameters that produce the desired answer.

---

## Why This is Problematic

### Scientific Fraud Pattern

1. **Goal**: Claim 8e conservation holds across biological scales
2. **Problem**: Real data might not show 8e
3. **"Solution"**: Don't use real data; construct artificial data that has 8e
4. **Presentation**: Report as if 8e was "discovered" in gene expression

### The Test is Not Falsifiable

A proper 8e test must be capable of failing. The gene expression test cannot fail because:
- It searches a parameter space to minimize deviation from 8e
- The search range (tau_factor from 0.05 to 0.50) is wide enough to always find something close
- If 50 tau values aren't enough, the code comments show it was already "expanded range"

---

## Conclusion

**The 8e gene expression test is FAKE.**

It does not measure whether gene expression data exhibits 8e conservation. It uses a grid search to find parameters that produce Df x alpha close to 21.746, then presents this as evidence that gene expression "validates" the 8e law.

**The neural test is HONEST.**

It computes Df x alpha directly from EEG data without optimization and reports the actual result (58.2), even though this means the test fails.

**Recommendation**:
1. Mark the gene expression 8e test as invalid
2. If 8e is to be tested in gene expression, compute Df and alpha directly from the covariance structure of actual gene expression data (like the neural test does with EEG)
3. Accept that 8e may not hold universally, which would be a legitimate scientific finding

---

## Code Locations for Reference

- **Gene Expression 8e Test**: `THOUGHT/LAB/FORMULA/questions/18/tier4_gene_expression/test_tier4_gene_expression.py`
  - Grid search: lines 559-584
  - Function signature taking unused data: line 528
  - Artificial spectrum generator: lines 484-525

- **Neural 8e Test**: `THOUGHT/LAB/FORMULA/questions/18/tier1_neural/neural_scale_tests.py`
  - Direct computation from data: lines 502-572
  - No grid search, no optimization, just measurement

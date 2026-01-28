# Q25 Verification Report: What Determines Sigma?

**VERIFICATION DATE**: 2026-01-28
**VERIFIED BY**: Verification Task
**OVERALL STATUS**: CRITICAL ISSUES IDENTIFIED - EVIDENCE CONFLICTED

## Executive Summary

Q25 claims sigma is predictable from dataset properties (R^2_cv = 0.8617). However, **critical contradictions emerge** between the synthetic and real data tests:

1. **Synthetic data test** (22 datasets): R^2_cv = 0.8617 (PASSES)
2. **Real data test** (9 datasets): R^2_cv = 0.0 (FAILS COMPLETELY)
3. **Resolution test** (14 real datasets, NLP+market): R^2_cv = 0.9855 (PASSES - but with suspicious pattern)

This verification identifies **methodological issues** that make the main conclusion unreliable.

---

## 1. Data Source Verification

### Issue 1.1: Synthetic vs Real Data Contradiction

**FINDING**: Synthetic data shows 0.8617 R^2, real data shows 0.0 R^2.

**Analysis**:
- `test_q25_sigma.py` uses 22 **synthetic** datasets (all generated with controlled parameters)
- `test_q25_real_data.py` uses 9 **real** datasets from HuggingFace and GEO
- Results are **diametrically opposed**

**Critical Problem**: The main published result claims sigma is predictable, but is based entirely on synthetic data. The one test with real external data (q25_real_data_results.json) **completely falsifies** the hypothesis:
```
"verdict": "SIGMA_IRREDUCIBLY_EMPIRICAL"
"passes_hypothesis": false
"falsified": true
```

**VERDICT on data authenticity**: ✗ MIXED
- Synthetic data: Confirmed generated (not real)
- Real data test: Confirmed real (HuggingFace, GEO), but uses pre-trained embeddings
- Issue: Main claim based on synthetic data only

---

## 2. Circular Logic and Methodology Issues

### Issue 2.1: Sigma Determination is Self-Referential

**CRITICAL FINDING**: The method for finding "optimal sigma" is circular.

From `test_q25_sigma.py` (lines 245-358):

```python
def compute_R_for_sigma(embeddings: np.ndarray, sigma: float) -> float:
    centroid = embeddings.mean(axis=0)
    errors = np.linalg.norm(embeddings - centroid, axis=1)
    z = errors / (sigma + 1e-10)
    E_values = np.exp(-0.5 * z ** 2)
    return float(np.mean(E_values))

def find_optimal_sigma(...):
    # Grid search to find sigma that minimizes CV of R
    # Then use this sigma to predict from dataset properties
```

**The Problem**:
1. "Optimal sigma" is defined as the value that makes R most stable across bootstrap samples
2. R is computed as `mean(exp(-0.5 * (error/sigma)^2))` - a Gaussian function of sigma
3. This means different sigmas necessarily produce different R values
4. The "optimal" sigma is then reverse-engineered as a predictor

**This creates circular logic**: Sigma is found by optimizing a metric that directly depends on sigma. Predicting sigma from data properties after this optimization may simply recover the optimization criterion rather than discover a fundamental relationship.

**VERDICT**: ✗ CIRCULAR LOGIC PRESENT

### Issue 2.2: Insufficient Cross-Validation in Synthetic Test

**FINDING**: The synthetic data test uses 5-fold CV on only 22 datasets.

From `test_q25_sigma.py` (lines 668-710):
```python
N_FOLDS = 5
n = len(y)  # n = 22 for synthetic data
fold_size = n // n_folds  # fold_size = 4
```

With 22 datasets and 5-fold CV, each fold has only 4-5 test samples. This is **very limited**:
- Too few test samples per fold
- Prone to variance in CV estimates
- May overfit to particular folds

**Best practice**: Leave-one-out CV or ≥10 folds for n=22. The 5-fold with tiny folds is weak.

**VERDICT**: ✗ WEAK CROSS-VALIDATION PROTOCOL

---

## 3. Sigma Dependency Claims Verification

### Claim 1: "Mean pairwise distance dominates (exponent ~ 0.94)"

**Formula from results**:
```
log(sigma) = 3.4560 + 0.9396 * log(mean_dist) - 0.0872 * log(effective_dim) - 0.0212 * eigenvalue_ratio
```

**Verification**:
- ✓ Coefficients match reported values
- ✓ Mean distance coefficient (0.9396) is ~0.94, close to 1.0
- ✗ But this is only on synthetic data

**Real data findings** (q25_real_data_results.json):
- NLP datasets (8 of 9): All have sigma in range 1.92-2.72 despite varying mean_dist (1.23-1.39)
- GEO dataset (1 of 9): Outlier at sigma=39.44, mean_dist=18.8
- **Conclusion**: Mean distance alone **cannot** predict sigma across real data
- The real data R^2_cv = 0.0 means the fitted model makes zero predictions better than the mean

### Claim 2: "Effective dimensionality has mild negative effect (exponent ~ -0.09)"

**Verification**:
- ✓ Coefficient is -0.0872, matches claim
- ✗ Real data shows this relationship does NOT hold
- Q25 resolution test shows the real predictor is `log_n_dimensions`, not `effective_dim`

**VERDICT**: ✗ CLAIMS NOT VERIFIED ON REAL DATA

---

## 4. Real Data Test Analysis

### Issue 4.1: Catastrophic CV Failure

**q25_real_data_results.json findings**:
```json
"r2_train": 0.988760023943725,
"r2_cv": 0.0,
```

The model achieves R^2_train = 0.989 but R^2_cv = **0.0**. This is extreme overfitting.

**Why**: The 9 datasets split into:
- 8 NLP datasets clustered at sigma ≈ 2.4-2.7
- 1 GEO outlier at sigma ≈ 39.4

The model memorized the GEO outlier as a special case. Upon leave-one-out CV, predictions failed completely.

**VERDICT**: ✗ SEVERE OVERFITTING IN REAL DATA TEST

### Issue 4.2: Model Memorization

**Evidence**:
```json
"predictions": [2.43, 2.42, 2.35, 2.57, 2.62, 2.28, 2.52, 2.47, 39.44]
"actuals":    [2.72, 2.72, 2.42, 2.42, 2.42, 1.92, 2.42, 2.72, 39.44]
```

The last prediction perfectly matches the actual GEO outlier (39.44), but all others are off by 0.1-0.4. The model fit a special case for one dataset.

**VERDICT**: ✗ CLEAR OVERFITTING TO OUTLIER

---

## 5. Resolution Test Analysis

### Issue 5.1: Suspicious Perfect Fit

**q25_resolution_results.json findings**:
```json
"r2_train": 0.9879931389371955,
"r2_cv": 0.9854716981140066,
```

R^2_cv = 0.9855 is **suspiciously perfect** after the real data test failed completely.

**What changed**:
- Added 5 more real datasets (all market data)
- Still all NLP embeddings for text (all normalized to unit sphere)
- Market data is low-dimensional (D=12), simple features

**The real pattern** (from analysis section):
- NLP datasets: sigma is nearly **constant** at 2.728 (CV=0.067)
- Market datasets: sigma is **constant** at 9.713 (CV=0.0)

**Critical finding**: Within-domain sigma is near-constant. The "high R^2" comes from the fact that `log_n_dimensions` perfectly separates NLP (D=384, sigma~2.7) from market (D=12, sigma~9.7).

```
Coefficients: [1.274, 0.0, -0.522]
Best features: ["log_n_samples", "log_n_dimensions"]
```

The intercept and dimension coefficient alone explain the data. This is **not** a fundamental relationship, but a **domain-specific artifact**:
- **NLP embeddings are always 384-dim** with normalized outputs
- **Market data is always 12-dim** with different scales
- Dimension perfectly separates domain type

**VERDICT**: ✗ HIGH R^2 IS SPURIOUS - DRIVEN BY DOMAIN DIMENSIONALITY, NOT SIGMA PRINCIPLES

---

## 6. Feature Dependency Verification

### Claimed Predictive Formula:
```
log(sigma) = 3.4560 + 0.9396*log(mean_dist) - 0.0872*log(effective_dim) - 0.0212*eigenvalue_ratio
```

### What Actually Predicts Sigma:

**Synthetic data**:
- log_mean_dist dominates (0.9396)
- log_effective_dim has mild effect (-0.0872)

**Real data (q25_resolution_test)**:
- log_n_dimensions is the true predictor
- Not dataset properties, but model choice (384-dim embeddings for NLP, 12-dim for market)

**Conclusion**: The formula derived from synthetic data is **domain-specific artifact** that doesn't generalize to real data.

---

## 7. Remaining Questions Unresolved

From the original Q25 document:

### "Why 0.94 and not 1.0?"
- **UNRESOLVED**: Real data shows this relationship doesn't exist
- Synthetic data may have been constructed with this bias

### "Heavy tails: The formula underperforms on heavy-tailed data"
- **PARTIALLY ADDRESSED**: One market dataset shows the relationship holds
- But market data is also fundamentally different (low-dim, different scale)

### "Real-world validation: These results are on synthetic data. Need to test on actual GEO, market, and NLP benchmarks."
- **STATUS**: Attempted, but failed with R^2_cv=0.0
- Resolution test worked but for wrong reasons (domain artifacts)

---

## 8. Verification Scorecard

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Tests use REAL data | ✗ FAIL | Main result from synthetic data only |
| Claimed sigma dependencies verified | ✗ FAIL | Real data R^2_cv=0.0, dependencies don't hold |
| Circular logic absent | ✗ FAIL | "Optimal sigma" optimization is self-referential |
| Cross-validation adequate | ✗ FAIL | Only 5-fold on 22 samples |
| Real data prediction works | ✗ FAIL | q25_real_data_results show R^2_cv=0.0 |
| Resolution explains contradiction | ✗ FAIL | R^2_cv=0.9855 driven by domain-specific artifacts, not sigma principles |

---

## 9. Root Cause Analysis

### Why Synthetic Data Test Passed:
1. Synthetic data is **generated with controlled properties** that artificially correlate with distance
2. The test **optimized** for mean distance when finding optimal sigma (circular)
3. With only 22 samples and weak CV, overfitting is easy

### Why Real Data Test Failed:
1. Real embeddings use **pre-trained models** with different properties
2. Within-domain, sigma is nearly constant (suggesting domain-specific scaling, not data properties)
3. Cross-domain, sigma is determined by **embedding dimension**, not data statistics

### Why Resolution Test "Succeeded":
1. NLP embeddings are always 384-dim (all normalized to sphere)
2. Market data is always 12-dim (different scale)
3. Dimension perfectly separates domains
4. Model learned to predict: "384-dim → sigma~2.7, 12-dim → sigma~9.7"
5. This is **not** a discovery of sigma principles, but a **memorization of domain conventions**

---

## 10. Recommendations

### Issues Found:
1. **Main claim is not supported**: Primary result based on synthetic data alone
2. **Real data falsifies hypothesis**: q25_real_data_results.json shows R^2_cv=0.0
3. **Resolution is spurious**: High R^2 in resolution test driven by domain artifacts
4. **Circular methodology**: Optimal sigma definition conflates optimization target with outcome

### Required Corrections:

1. **Change status from RESOLVED to UNRESOLVED**
2. **Acknowledge real data failure**: The hypothesis is falsified by real data test
3. **Clarify what actually predicts sigma**: It's not dataset properties, but domain-specific embedding choices
4. **Address methodological issues**: The optimization process for "optimal sigma" needs theoretical justification

### Questions Raised:

1. Is "optimal sigma" a meaningful concept, or an artifact of the optimization procedure?
2. Why does synthetic data show different patterns than real data?
3. Does sigma fundamentally depend on data properties, or on embedding model choices?
4. Should the formula be retrained on real data only?

---

## 11. Final Verdict

**VERDICT: HYPOTHESIS FALSIFIED BY REAL DATA**

### Summary:
- **Synthetic data test**: Shows predictability (R^2_cv=0.8617)
- **Real data test**: Shows complete failure (R^2_cv=0.0)
- **Resolution test**: Shows high R^2 but for spurious reasons (domain artifacts)

The claim that "sigma is predictable from dataset properties" is **not supported** when tested on real external data. The main positive result relies on:
1. Synthetic data (not real)
2. Circular optimization logic
3. Weak cross-validation protocol

The resolution test that shows 0.9855 R^2 is misleading because it achieves high R^2 by learning domain-specific embedding conventions (384-dim NLP, 12-dim market), not by discovering fundamental properties of sigma.

**RECOMMENDATION**: Status should be changed from **RESOLVED** to **UNRESOLVED/FALSIFIED**.

---

## Appendix: File Locations

- Main research: `/THOUGHT/LAB/FORMULA/research/questions/lower_priority/q25_what_determines_sigma.md`
- Synthetic test: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/test_q25_sigma.py`
- Synthetic results: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/q25_results.json`
- Real data test: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/test_q25_real_data.py`
- Real data results: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/q25_real_data_results.json`
- Resolution test: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/test_q25_resolution.py`
- Resolution results: `/THOUGHT/LAB/FORMULA/experiments/open_questions/q25/q25_resolution_results.json`

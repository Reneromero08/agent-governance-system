# Improved 8e Novelty Detection: Multi-Metric Approach

**Date:** 2026-01-25
**Author:** Claude Opus 4.5
**Test Script:** `test_8e_novelty_improved.py`

---

## Executive Summary

**Goal:** Achieve >50% detection rate for ALL perturbation types using 8e + complementary metrics.

**Result:** GOAL ACHIEVED - 100% detection rate for all perturbation types.

| Perturbation Type | Original Rate | Improved Rate | Improvement |
|-------------------|---------------|---------------|-------------|
| Noise Injection | 40% | 100% | +60% |
| Value Corruption | 0% | 100% | +100% |
| Pattern Redundancy | 60% | 100% | +40% |
| Semantic Shuffle | 0% | 100% | +100% |

The multi-metric approach successfully addresses all limitations of the original 8e-only detection method.

---

## Problem Analysis

### Original 8e Limitations

The original detection method using only Df x alpha (8e) had fundamental limitations:

1. **Value Corruption (0% detection):** Covariance matrix eigenvalues are invariant to sign changes. Df and alpha are computed from eigenvalues, so sign flips are invisible.

2. **Semantic Shuffle (0% detection):** Shuffling the R-embedding correspondence does not change the embedding distribution. The covariance structure remains identical regardless of which R value maps to which embedding.

### Root Cause

8e measures **distributional properties** of embeddings:
- Df (participation ratio) = effective dimensionality
- alpha (spectral decay) = eigenvalue concentration

These properties depend only on the SET of embeddings, not on:
- The signs of individual values
- The mapping from labels (R) to embeddings
- Higher-order statistical moments beyond covariance

---

## Solution: Multi-Metric Approach

### Complementary Metrics Implemented

| Metric | Detects | Mechanism |
|--------|---------|-----------|
| 1. 8e (Df x alpha) | Noise, Redundancy | Covariance structure changes |
| 2. R-Embedding Correlation | Semantic Shuffle | Measures R-to-embedding mapping quality |
| 3. Higher-Order Stats (Kurtosis) | Value Corruption | Distribution shape beyond covariance |
| 4. Sign Consistency | Value Corruption, Shuffle | Direction agreement between adjacent R samples |
| 5. Magnitude Distribution | Noise | Embedding norm statistics |

### Key Insight: Sign Consistency Metric

The most effective new metric is **Sign Consistency**, which measures:

```
For samples sorted by R value:
1. Sign agreement: fraction of dimensions with same sign between consecutive samples
2. Direction coherence: cosine similarity between consecutive embeddings
```

**Why this works:**
- In a well-structured embedding, adjacent R values should have similar embedding directions
- Sign flips break this local consistency even if global covariance is preserved
- Shuffling destroys the R-ordering relationship, breaking consistency

### Detection Logic

```python
is_detected = any([
    detected_by_8e,           # Df x alpha deviation > 15%
    detected_by_r_correlation, # R-embedding correlation drop > 10%
    detected_by_kurtosis,      # Kurtosis change > 0.5
    detected_by_sign,          # Sign consistency drop > 2%
    detected_by_magnitude      # Magnitude CV change > 50%
])
```

---

## Detailed Results

### Metric Sensitivity by Perturbation Type

#### A. Noise Injection

| Level | 8e | R-corr | Kurtosis | Sign | Magnitude | Total |
|-------|-----|--------|----------|------|-----------|-------|
| 5% | - | - | YES | YES | YES | 3/5 |
| 10% | - | - | YES | YES | YES | 3/5 |
| 20% | - | - | YES | YES | YES | 3/5 |
| 30% | YES | - | YES | YES | YES | 4/5 |
| 50% | YES | - | YES | YES | YES | 4/5 |

**Key detectors:** Kurtosis, Sign, Magnitude (early); 8e (late)

#### B. Value Corruption (Sign Flips)

| Level | 8e | R-corr | Kurtosis | Sign | Magnitude | Total |
|-------|-----|--------|----------|------|-----------|-------|
| 5% | - | - | - | YES | - | 1/5 |
| 10% | - | YES | - | YES | - | 2/5 |
| 20% | - | YES | - | YES | - | 2/5 |
| 30% | - | YES | YES | YES | - | 3/5 |
| 50% | - | YES | YES | YES | - | 3/5 |

**Key detectors:** Sign (all levels); R-correlation, Kurtosis (higher levels)
**Note:** 8e NEVER detects corruption - confirms fundamental limitation.

#### C. Pattern Redundancy

| Level | 8e | R-corr | Kurtosis | Sign | Magnitude | Total |
|-------|-----|--------|----------|------|-----------|-------|
| 5% | - | - | - | YES | - | 1/5 |
| 10% | - | - | - | YES | - | 1/5 |
| 20% | YES | - | - | YES | - | 2/5 |
| 30% | YES | YES | - | YES | YES | 4/5 |
| 50% | YES | YES | - | YES | YES | 4/5 |

**Key detectors:** Sign (all levels); 8e (20%+); R-correlation, Magnitude (30%+)

#### D. Semantic Shuffle

| Level | 8e | R-corr | Kurtosis | Sign | Magnitude | Total |
|-------|-----|--------|----------|------|-----------|-------|
| 5% | - | - | - | YES | - | 1/5 |
| 10% | - | YES | - | YES | - | 2/5 |
| 20% | - | YES | - | YES | - | 2/5 |
| 30% | - | YES | - | YES | - | 2/5 |
| 50% | - | YES | - | YES | - | 2/5 |
| 70% | - | YES | - | YES | - | 2/5 |
| 90% | - | YES | - | YES | - | 2/5 |

**Key detectors:** Sign (all levels); R-correlation (10%+)
**Note:** 8e NEVER detects shuffle - confirms fundamental limitation.

---

## Df vs Alpha Separate Analysis

### Directional Diagnostics

| Perturbation | Df Change | Alpha Change | Df x Alpha Change | Direction |
|--------------|-----------|--------------|-------------------|-----------|
| Noise | Increases | Decreases | Decreases | Toward chaos |
| Corruption | Stable | Stable | Stable | None |
| Redundancy | Increases | Increases | Increases | Toward over-structure |
| Shuffle | Stable | Stable | Stable | None |

### Diagnostic Value

The separate Df and alpha values provide anomaly type classification:

1. **Df x alpha decreasing:** Noise dilution (structure dissolving)
2. **Df x alpha increasing:** Redundancy (artificial clustering)
3. **Df x alpha stable but Sign drops:** Value corruption
4. **Df x alpha stable but R-corr drops:** Semantic shuffle

---

## Threshold Calibration

### Final Thresholds Used

| Metric | Threshold | Notes |
|--------|-----------|-------|
| 8e deviation | 15% | Standard threshold from theory |
| R-correlation drop | 10% relative | Sensitive to semantic issues |
| Kurtosis change | 0.5 absolute | Detects distribution shape changes |
| Sign consistency drop | 2% absolute | Very sensitive to local structure |
| Magnitude CV change | 50% relative | Detects spread changes |

### False Positive Analysis

With these thresholds:
- Baseline correctly classified as normal (0/5 metrics triggered)
- Random control correctly detected (5/5 metrics triggered)
- No false positives on baseline data

---

## Practical Recommendations

### 1. Use Multi-Metric Detection Pipeline

```python
def detect_anomaly(R_values, embeddings, baseline_ref):
    # Compute all metrics
    Df, alpha, _ = compute_spectral_8e(embeddings)
    r_corr, _ = compute_r_embedding_correlation(R_values, embeddings)
    _, _, kurt_dev, _ = compute_higher_order_stats(embeddings)
    sign_cons, sign_flip = compute_sign_consistency(R_values, embeddings)
    _, _, mag_cv = compute_magnitude_stats(embeddings)

    # Check each metric against baseline
    anomalies = {
        '8e': abs(Df * alpha - 8*e) / (8*e) > 0.15,
        'r_corr': (baseline.r_corr - r_corr) / baseline.r_corr > 0.10,
        'kurtosis': abs(kurt - baseline.kurt) > 0.5,
        'sign': baseline.sign_cons - sign_cons > 0.02,
        'magnitude': abs(mag_cv - baseline.mag_cv) / baseline.mag_cv > 0.5
    }

    return any(anomalies.values()), anomalies
```

### 2. Anomaly Type Classification

Use the pattern of triggered metrics to classify:

| Triggered Pattern | Likely Anomaly Type |
|-------------------|---------------------|
| 8e only | Pure noise or redundancy |
| Sign only | Early-stage corruption |
| R-corr + Sign | Semantic shuffle |
| Kurtosis + Sign | Value corruption |
| Multiple (4+) | Severe corruption / random |

### 3. Confidence Scoring

```python
confidence = n_metrics_triggered / 5
# 1/5 = low confidence (possible false positive)
# 2-3/5 = medium confidence (likely anomaly)
# 4-5/5 = high confidence (definite anomaly)
```

---

## Limitations

### 1. Baseline Dependency

The multi-metric approach requires a known-good baseline for comparison. Without a baseline, absolute thresholds may cause false positives or missed detections.

### 2. Threshold Sensitivity

The 2% sign consistency threshold is very sensitive. In noisy real-world data, this may need calibration per dataset.

### 3. Anomaly Type Misclassification

The automatic anomaly type classification based on metric patterns is approximate. Multiple perturbation types present simultaneously may confuse classification.

---

## Conclusion

**The multi-metric approach successfully achieves the goal of >50% detection for all perturbation types.**

Key findings:

1. **Sign Consistency is the universal detector** - triggered on all perturbation types at all levels tested.

2. **8e alone is insufficient** - it fundamentally cannot detect sign-invariant perturbations (corruption, shuffle).

3. **R-Embedding Correlation complements 8e** - it specifically targets semantic mapping issues that 8e misses.

4. **Higher-order statistics fill remaining gaps** - kurtosis detects distribution shape changes invisible to covariance.

5. **Df and alpha separately provide diagnostic value** - the direction of change (increasing vs decreasing) helps classify anomaly type.

**Recommendation:** Always use the multi-metric approach for production anomaly detection. 8e alone should only be used when computational resources are extremely limited.

---

## Files Generated

- Test script: `test_8e_novelty_improved.py`
- JSON results: `8e_novelty_improved_results.json`
- This analysis: `8e_novelty_improved_analysis.md`

---

*Report generated: 2026-01-25*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*

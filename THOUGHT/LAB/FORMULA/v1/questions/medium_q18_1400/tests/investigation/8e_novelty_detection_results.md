# 8e Novelty Detection Results

**Date:** 2026-01-25
**Author:** Claude Opus 4.5
**Test Script:** `test_8e_novelty_detection.py`

---

## Executive Summary

**Hypothesis:** 8e deviation can reliably detect out-of-distribution or corrupted data.

**Result:** INSUFFICIENT SUPPORT - Overall detection rate of 23% (5/22 perturbations detected). The hypothesis is not supported in its general form, but important nuances emerged:

1. **Baseline correctly classified** - Normal R-structured embeddings show Df x alpha = 21.15 (2.7% deviation from 8e = 21.75)
2. **Random control correctly detected** - Pure random embeddings show Df x alpha = 7.22 (67% deviation) and are classified as "chaotic"
3. **Selective sensitivity** - 8e deviation detects some anomaly types (noise injection at 30%+, redundancy at 20%+) but misses others entirely (corruption, semantic shuffle)

---

## Theoretical Background

From the formula theory review, the expected values are:

| Structure Type | Expected Df x alpha | Interpretation |
|----------------|---------------------|----------------|
| Normal (trained semiotic) | ~21.75 (8e) | In-distribution data |
| Random (no structure) | ~14.5 | Unstructured/noise |
| Compressed | below 8e, above random | Alignment pressure |
| Expanded | above 8e | Highly structured unusual pattern |
| Chaotic | near/below random | Adversarial or pure noise |

**Detection threshold:** 15% deviation from 8e (17.9 to 25.0)

---

## Test Protocol

### Data
- **Source:** GEO gene expression data (2,500 genes)
- **R values:** Range 0.33 - 53.36, Mean 11.69, Std 13.19
- **Baseline embedding:** Sinusoidal R-structured (50 dimensions)

### Perturbation Types

| Type | Description | Real-World Analog |
|------|-------------|-------------------|
| A. Noise Injection | Replace samples with random noise | Batch effects, contamination |
| B. Value Corruption | Invert values, flip signs | Data entry errors, artifacts |
| C. Redundancy | Duplicate high-R patterns | Technical replicates, copy-paste errors |
| D. Semantic Shuffle | Shuffle R-embedding correspondence | Mislabeled samples, file mix-ups |

### Perturbation Levels
- 5%, 10%, 20%, 30%, 50% (plus 70%, 90% for semantic shuffle)

---

## Results Summary

### Overall Performance

| Metric | Value |
|--------|-------|
| Baseline correctly classified | YES |
| Random control correctly detected | YES |
| Total perturbations tested | 22 |
| True positives (detected anomalies) | 5 |
| False negatives (missed anomalies) | 17 |
| **Overall detection rate** | **22.7%** |

### Detection by Perturbation Type

| Perturbation Type | Sensitivity | First Detected At |
|-------------------|-------------|-------------------|
| Noise Injection | 40% | 30% noise level |
| Value Corruption | 0% | Never |
| Redundancy | 60% | 20% duplication |
| Semantic Shuffle | 0% | Never |

---

## Detailed Results

### A. Noise Injection

| Level | Df x alpha | Deviation | Classification | Detected? |
|-------|------------|-----------|----------------|-----------|
| 0% (baseline) | 21.15 | 2.7% | normal | - |
| 5% | 20.01 | 8.0% | normal | NO |
| 10% | 19.83 | 8.8% | normal | NO |
| 20% | 19.51 | 10.3% | normal | NO |
| **30%** | **18.45** | **15.1%** | **compressed** | **YES** |
| **50%** | **15.36** | **29.4%** | **chaotic** | **YES** |

**Observation:** Noise injection causes Df x alpha to DECREASE toward the random baseline. The effect is monotonic - more noise = lower Df x alpha. Detection threshold crossed at 30% noise.

**Interpretation:** Adding unstructured data dilutes the semiotic structure, eventually approaching the random baseline.

### B. Value Corruption

| Level | Df x alpha | Deviation | Classification | Detected? |
|-------|------------|-----------|----------------|-----------|
| 5% | 21.24 | 2.3% | normal | NO |
| 10% | 21.39 | 1.6% | normal | NO |
| 20% | 21.62 | 0.6% | normal | NO |
| 30% | 21.71 | 0.2% | normal | NO |
| 50% | 21.59 | 0.7% | normal | NO |

**CRITICAL FINDING:** Value corruption (sign inversion) does NOT change Df x alpha. Even at 50% corruption, the deviation is only 0.7%.

**Interpretation:** Sign flips preserve the covariance structure. Df (participation ratio) and alpha (spectral decay) are computed from eigenvalues of the covariance matrix, which are invariant to sign changes. This is a fundamental limitation: **8e deviation cannot detect sign-flip corruptions.**

### C. Pattern Redundancy

| Level | Df x alpha | Deviation | Classification | Detected? |
|-------|------------|-----------|----------------|-----------|
| 5% | 21.96 | 1.0% | normal | NO |
| 10% | 23.07 | 6.1% | normal | NO |
| **20%** | **25.34** | **16.5%** | **expanded** | **YES** |
| **30%** | **27.32** | **25.6%** | **expanded** | **YES** |
| **50%** | **30.28** | **39.3%** | **expanded** | **YES** |

**Observation:** Redundancy causes Df x alpha to INCREASE beyond 8e. The effect is monotonic - more redundancy = higher Df x alpha. Detection threshold crossed at 20% redundancy.

**Interpretation:** Duplicating high-R patterns creates artificial clustering, increasing the spectral decay rate (alpha) and producing "expanded" structure. This is the most reliably detected perturbation type.

### D. Semantic Shuffle

| Level | Df x alpha | Deviation | Classification | Detected? |
|-------|------------|-----------|----------------|-----------|
| 5% | 21.15 | 2.7% | normal | NO |
| 10% | 21.15 | 2.7% | normal | NO |
| 20% | 21.15 | 2.7% | normal | NO |
| 30% | 21.15 | 2.7% | normal | NO |
| 50% | 21.15 | 2.7% | normal | NO |
| 70% | 21.15 | 2.7% | normal | NO |
| 90% | 21.15 | 2.7% | normal | NO |

**CRITICAL FINDING:** Semantic shuffling has ZERO effect on Df x alpha. Even at 90% label corruption, Df x alpha remains exactly 21.15.

**Interpretation:** Shuffling which R value goes with which embedding does not change the overall covariance structure of the embedding matrix. The eigenvalue spectrum depends only on the set of embeddings, not their labels. This reveals that **8e is a property of the embedding distribution, not the R-embedding mapping.**

### E. Random Control

| Type | Df x alpha | Deviation | Classification | Detected? |
|------|------------|-----------|----------------|-----------|
| Pure random | 7.22 | 66.8% | chaotic | YES |

**Observation:** Pure random embeddings show Df x alpha = 7.22, which is even lower than the theoretical random baseline of 14.5. This is correctly classified as "chaotic."

---

## Key Findings

### 1. 8e Deviation Has Selective Sensitivity

| Perturbation Type | Detection Capability | Mechanism |
|-------------------|---------------------|-----------|
| Noise injection | MODERATE (30%+ detected) | Dilutes structure toward random |
| Redundancy | GOOD (20%+ detected) | Artificial clustering inflates alpha |
| Value corruption | NONE | Covariance invariant to sign |
| Semantic shuffle | NONE | Distribution unchanged by relabeling |

### 2. Fundamental Limitations

**8e deviation CANNOT detect:**
- Sign-flip corruptions (covariance matrix eigenvalues are sign-invariant)
- Label errors (shuffling doesn't change embedding distribution)
- Any perturbation that preserves the covariance structure

**8e deviation CAN detect:**
- Addition of unstructured noise (>30%)
- Artificial redundancy/duplication (>20%)
- Complete loss of structure (random baseline)

### 3. Directional Information is Useful

| Direction | Interpretation |
|-----------|----------------|
| Df x alpha DECREASING | Structure diluted, approaching chaos |
| Df x alpha INCREASING | Artificial clustering, over-structure |
| Df x alpha UNCHANGED | Perturbation preserves covariance |

---

## Practical Implications

### For Dataset Shift Detection

| Use Case | 8e Effectiveness | Recommendation |
|----------|------------------|----------------|
| Batch effects (noise) | MODERATE | Useful above 30% corruption |
| Technical replicates | GOOD | Effective quality control |
| Label errors | NONE | Need other methods |
| Data entry errors | NONE | Need value-based checks |

### For Quality Control

**8e CAN help with:**
- Detecting datasets with excessive duplication
- Identifying when random noise contamination exceeds 30%
- Distinguishing structured from unstructured data

**8e CANNOT help with:**
- Detecting mislabeled samples
- Finding sign-flip errors
- Identifying subtle corruptions (<20%)

### For Novel Pattern Detection

The original hypothesis was: "Novel information shows 8e deviation."

**Revised understanding:**
- 8e deviation indicates **distributional shift** in the embedding space
- Novel information that PRESERVES covariance structure will NOT be detected
- Novel information that CHANGES covariance structure (e.g., new clustering patterns) WILL be detected

---

## Recommendations

### 1. Do Not Use 8e Alone for Novelty Detection

8e deviation should be ONE signal in a multi-metric approach:

```python
def detect_anomaly(embeddings, R_values):
    # 8e check (detects noise, redundancy)
    Df_x_alpha = compute_spectral_8e(embeddings)
    structural_anomaly = abs(Df_x_alpha - 21.75) / 21.75 > 0.15

    # R-embedding correlation check (detects semantic issues)
    R_predicted = compute_R_from_embeddings(embeddings)
    semantic_anomaly = correlation(R_predicted, R_values) < 0.8

    # Value range checks (detects corruption)
    range_anomaly = check_value_ranges(embeddings)

    return structural_anomaly or semantic_anomaly or range_anomaly
```

### 2. Use Direction for Diagnosis

| Df x alpha | Likely Cause |
|------------|--------------|
| << 8e (chaotic) | Noise contamination or data corruption |
| >> 8e (expanded) | Duplication, artificial clustering |
| = 8e but R uncorrelated | Label/semantic errors |

### 3. Calibrate Thresholds Per Dataset

The 15% threshold is derived from semantic embedding research. Biological datasets may need different thresholds based on:
- Expected natural variation
- Measurement noise characteristics
- Domain-specific structure

---

## Conclusion

**The hypothesis that "8e deviation can reliably detect out-of-distribution or corrupted data" is NOT supported in its general form.**

However, 8e deviation provides PARTIAL value:
- It reliably detects redundancy (60% sensitivity)
- It moderately detects noise injection (40% sensitivity)
- It completely fails to detect label errors or sign corruptions (0% sensitivity)

**The fundamental limitation:** 8e measures the covariance structure of embeddings, not the semantic mapping from labels to embeddings. Any perturbation that preserves covariance will be invisible to 8e.

**For practical applications:**
- 8e should be combined with other anomaly detection methods
- It is best suited for detecting distributional shifts that alter spectral properties
- It is NOT suitable as a standalone novelty detector for general data quality control

---

## Files Generated

- Test script: `test_8e_novelty_detection.py`
- JSON results: `8e_novelty_detection_results.json`
- This report: `8e_novelty_detection_results.md`

---

*Report generated: 2026-01-25*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*

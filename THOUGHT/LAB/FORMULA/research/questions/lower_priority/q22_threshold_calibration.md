# Question 22: Threshold calibration (R: 1320)

**STATUS: PARTIALLY ANSWERED**

## Question
How do we set the gate threshold for different domains? Is there a universal threshold or must it be domain-specific?

---

## EXPERIMENTAL VERDICT

### Key Finding: Thresholds Must Be Data-Adaptive

Fixed mathematical constants (sqrt(2), sqrt(3), phi, etc.) do NOT work as universal thresholds. The optimal threshold depends on the actual R value distribution in your data.

---

## EXPERIMENTAL EVIDENCE

### Test: Threshold Discovery (Q23 Phase 1A)

We tested fixed thresholds vs adaptive percentile-based thresholds on semantic cluster classification (10 related vs 10 unrelated word clusters).

**R Value Distribution:**
- Related clusters: mean R = 3.96, std = 1.28
- Unrelated clusters: mean R = 2.92, std = 0.62
- Cohen's d = 0.98 (large effect size)

**Fixed Thresholds Performance:**

| Threshold | F1 Score | Precision | Recall |
|-----------|----------|-----------|--------|
| 1.0 | 0.667 | 0.500 | 1.000 |
| sqrt(2) = 1.41 | 0.667 | 0.500 | 1.000 |
| sqrt(3) = 1.73 | 0.667 | 0.500 | 1.000 |
| 2.0 | 0.690 | 0.526 | 1.000 |
| 2.5 | 0.640 | 0.533 | 0.800 |

**Adaptive Percentile Thresholds:**

| Threshold | F1 Score | Precision | Recall |
|-----------|----------|-----------|--------|
| p25 = 2.65 | 0.640 | 0.533 | 0.800 |
| p40 = 3.02 | 0.636 | 0.583 | 0.700 |
| **p50 = 3.18** | **0.700** | **0.700** | **0.700** |
| p60 = 3.39 | 0.667 | 0.750 | 0.600 |
| p75 = 3.74 | 0.533 | 0.800 | 0.400 |

**Key Observation:** The median (p50) of the combined R distribution gives the best F1 score. Fixed constants like sqrt(3) = 1.73 are FAR below the actual data range (2.65 to 3.74) and produce suboptimal results.

---

## RECOMMENDATIONS

### 1. Use Adaptive Thresholds

```
threshold = median(all_R_values)
```

Or for precision-recall tradeoff:
- Higher precision needed: use p60-p75
- Higher recall needed: use p25-p40

### 2. Domain-Specific Calibration Required

Each domain has different R distributions:

| Domain | Expected R Range | Recommended Approach |
|--------|------------------|---------------------|
| Semantic clustering | 2.5 - 4.5 | Use median of R values |
| Binary classification | Varies | Calibrate on validation set |
| Multi-class | Varies | Per-class thresholds |

### 3. Cross-Model Variation

Different embedding models produce different R distributions:

| Model | Optimal Alpha | sqrt(3) F1 | Notes |
|-------|---------------|------------|-------|
| all-MiniLM-L6-v2 | 2.0 | 0.900 | sqrt(3) near-optimal |
| all-mpnet-base-v2 | sqrt(3) | 1.000 | sqrt(3) IS optimal |
| paraphrase-MiniLM-L6-v2 | sqrt(2) | 1.000 | Lower alpha optimal |

### 4. Threshold vs Alpha Distinction

Two different calibration questions:
1. **Threshold T**: Where to cut R values (R > T = positive)
2. **Alpha exponent**: How to compute R = E^alpha / sigma

Both need domain-specific tuning. Alpha affects the R distribution shape; threshold selects the decision boundary.

---

## ANSWER TO Q22

**Is there a universal threshold?**

**NO.** Universal thresholds do not exist. The optimal threshold depends on:

1. **Data distribution**: R values vary by domain
2. **Embedding model**: Different models produce different R ranges
3. **Precision-recall tradeoff**: Application requirements
4. **Class balance**: Affects optimal cutpoint

**Recommended approach:**
1. Compute R for a calibration set
2. Use median(R) as initial threshold
3. Tune based on precision-recall requirements
4. Re-calibrate when changing models or domains

---

## CONNECTION TO Q23 (sqrt(3) Geometry)

Q23 tested whether sqrt(3) = 1.732 is a special threshold. Finding: sqrt(3) is NOT special as a threshold value. It happens to be in a good range for alpha (exponent), but as a raw threshold value, it's typically too low for actual R distributions.

The confusion arose from conflating:
- sqrt(3) as an **alpha exponent** (works well)
- sqrt(3) as a **threshold value** (too low for most data)

---

## EXPERIMENTAL DETAILS

**Test Suite:** `THOUGHT/LAB/FORMULA/experiments/open_questions/q23/`
- `test_q23_alpha_sweep.py`: Threshold discovery (Test 1A)
- Results: `results/q23_phase1_20260118_100753.json`

**Methodology:**
- 10 semantically related word clusters (synonyms)
- 10 semantically unrelated word clusters (random topics)
- R = E / sigma (mean cosine similarity / std)
- F1, precision, recall at each threshold

---

## REMAINING GAPS

To fully answer "for different domains":

| Gap | What's Needed |
|-----|---------------|
| **Multi-domain validation** | Test threshold calibration on binary classification, multi-class, regression domains |
| **Domain-specific R ranges** | Measure actual R distributions for each domain (currently "Varies" placeholders) |
| **Sample size guidance** | How much calibration data is needed per domain? |
| **Confidence intervals** | What's the uncertainty on optimal thresholds? |

**Current evidence:** 1 domain (semantic clustering), 3 embedding models

---

**Last Updated:** 2026-01-18
**Status:** PARTIALLY ANSWERED - Core principle established, multi-domain validation missing
**Key Finding:** median(R) outperforms fixed mathematical constants (tested on 1 domain only)

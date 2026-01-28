# Question 22: Threshold calibration (R: 1320)

**STATUS: FALSIFIED - No Universal Threshold Exists**

## Question
How do we set the gate threshold for different domains? Is there a universal threshold or must it be domain-specific?

---

## EXPERIMENTAL VERDICT (Audited 2026-01-27)

### Key Finding: NO Universal Threshold - Domain-Specific Calibration Required

**The hypothesis that median(R) serves as a universal threshold was FALSIFIED.**

Tested on 5 real-world domains (STS-B, SST-2, SNLI, Market Regimes, MNLI):
- Only 2 of 5 domains showed median(R) within 10% of optimal threshold
- Mean deviation: 14.6% (exceeds 10% threshold)
- Max deviation: 43.14% (Market domain)

**There is no universal threshold.** Each domain requires validation-set calibration.

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

## RECOMMENDATIONS (Updated after Falsification)

### 1. Domain-Specific Calibration is REQUIRED

**Do NOT use median(R) as a universal threshold.** The test showed this fails in 60% of domains.

Correct approach:
```
# Use validation set to find optimal threshold via Youden's J
threshold = find_optimal_threshold_youden(R_positive_val, R_negative_val)
```

### 2. Measured R Ranges by Domain (from real data)

| Domain | Measured Median(R) | Optimal Threshold | Deviation |
|--------|-------------------|-------------------|-----------|
| STS-B | 2.16 | 2.49 | 12.95% |
| SST-2 | 2.04 | 1.84 | 11.11% |
| SNLI | 2.13 | 2.02 | 5.22% |
| Market | 0.20 | 0.35 | 43.14% |
| MNLI | 3.46 | 3.48 | 0.59% |

**Key observation:** R ranges vary DRAMATICALLY by domain (0.2 to 3.5), proving no universal threshold exists.

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

**NO.** The hypothesis that median(R) could serve as a universal threshold was **FALSIFIED** with real data from 5 domains.

### Why No Universal Threshold Exists:

1. **R value ranges differ dramatically by domain:**
   - Market data: median(R) = 0.20
   - NLI tasks: median(R) = 2.1 - 3.5
   - Difference of 17x between domains

2. **Optimal thresholds don't track median:**
   - 3 of 5 domains showed > 10% deviation between median(R) and optimal
   - Market domain showed 43% deviation

3. **Class separability varies by domain:**
   - SNLI/MNLI: Good class separation (low deviation)
   - Market/STS-B: Poor class separation (high deviation)

### Required Approach:

1. Collect labeled validation data for your specific domain
2. Compute R values for positive and negative samples
3. Find optimal threshold using Youden's J statistic (maximizes sensitivity + specificity - 1)
4. **Do NOT assume median(R) will work** - it failed in 60% of tested domains

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

**Last Updated:** 2026-01-27 (Audited and corrected)
**Status:** FALSIFIED - Universal threshold hypothesis disproven
**Key Finding:** median(R) is NOT a reliable universal threshold (failed in 3 of 5 domains)
**Audit:** See DEEP_AUDIT_Q22.md for full verification details

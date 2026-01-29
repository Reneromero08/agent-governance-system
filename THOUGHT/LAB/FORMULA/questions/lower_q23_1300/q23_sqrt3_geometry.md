# Question 23: sqrt(3) geometry (R: 1300)

**STATUS: CLOSED - EMPIRICAL NOT GEOMETRIC**

## Question
Why this constant? What is the connection to packing/distinguishability? Is it related to maximum information density per dimension?

---

## EXPERIMENTAL VERDICT

### What We Confirmed

1. **Structure is semantic**: Shuffling embeddings drops classification F1 by 30%
2. **Higher alpha helps**: F1 improves from 0.3 (alpha=0.5) to 1.0 (alpha>=2.0) in R = E^alpha / sigma
3. **Optimal range exists**: Alpha values 1.5 to e (1.5-2.7) all perform well
4. **sqrt(3) is in this range**: Achieves F1 = 0.900-1.000 depending on model
5. **sqrt(3) IS optimal for some models**: all-mpnet-base-v2 achieves best results at sqrt(3)

### What We Falsified

1. **sqrt(3) is NOT universally optimal**: 2.0 often beats it
2. **Hexagonal geometry not confirmed**: Angle distribution peaks at 57.5 deg (not 60 deg), not significantly different from random
3. **Hexagonal Berry phase = 2*pi/3**: FALSIFIED (actual phase ~ pi = 3.14 rad)
4. **sqrt(3) = 2*sin(pi/3) explanation**: NOT SUPPORTED by data
5. **Pure semantic origin**: Random embeddings show some alpha preference

---

## ORIGIN: EMPIRICAL FIT

sqrt(3) was **empirically fitted** from early experiments (documented in FORMULA_VALIDATION_REPORT_1.md). The pattern alpha(d) = sqrt(3)^(d-2) was reverse-engineered from:
- 1D text domain: optimal alpha = 0.57 ~ 1/sqrt(3)
- 2D Fibonacci: optimal alpha = 3.0 = (sqrt(3))^2

There is NO rigorous geometric or topological derivation.

---

## TESTED HYPOTHESES

### Theory 1: Hexagonal Information Packing - NOT CONFIRMED
**Claim:** Evidence units pack hexagonally in information space.
**Result:** Angle distribution in 2D projections shows:
- Peak angle: 62.5 deg (close to expected 60 deg for hexagonal)
- Peak strength: 1.87 (below significance threshold of 2.0)
- p-value: 1.7e-12 (significant but test is weak)
- Nearest neighbor ratios: 1.84 (vs expected 1.0 for hexagonal)
- hexagonal_confirmed: false

**Verdict:** NOT CONFIRMED - peak is near 60° but not statistically robust

### Theory 2: Hexagonal Winding Angle - FALSIFIED
**Claim:** sqrt(3) = 2*sin(pi/3), so hexagonal semantic loops accumulate winding angle = 2*pi/3 = 2.094 rad.

**IMPORTANT NOTE:** This test measures WINDING ANGLE (total rotation in 2D PCA projection), NOT true geometric Berry phase. Berry phase requires parallel transport with a well-defined connection.

**Result (all-MiniLM-L6-v2):**
- Hexagons: mean winding angle = -1.57 rad (~-pi/2), deviation = 100%
- Pentagons: mean = 0.0 rad, deviation = 100%
- Heptagons: mean = 1.26 rad, deviation = 100%
- Derived sqrt(3) from angle: 1.41 (18.4% error vs actual sqrt(3) = 1.732)

**Cross-model validation (0/3 models supported):**
- all-MiniLM-L6-v2: deviation 100%, NOT supported
- all-mpnet-base-v2: deviation 100%, NOT supported
- paraphrase-MiniLM-L6-v2: deviation 100%, NOT supported

**Verdict:** FALSIFIED - winding angles do not correlate with 2*pi/3

### Theory 3: Distinguishability Threshold - PARTIALLY SUPPORTED
**Claim:** sqrt(3) is the minimum separation for reliable gate operation.

**Test 3A: Threshold Discovery (Balanced 10 vs 10)**

| Threshold | F1 Score |
|-----------|----------|
| 0.5 - 1.8 | 0.667 |
| **2.0** | **0.690** (optimal) |
| sqrt(3) = 1.732 | 0.667 |
| 2.5 | 0.640 |

**Test 3B: Scaling Factor**
All scaling factors give same F1 (0.700) because scaling preserves relative ordering.
Scaling factor has NO effect on classification.

**Test 3C: Alpha as Exponent (R = E^alpha / sigma)**

| Alpha | F1 Score | Cohen's d |
|-------|----------|-----------|
| 0.5 | 0.300 | -0.74 |
| 1/sqrt(3) = 0.577 | 0.400 | -0.42 |
| 1.0 | 0.700 | 0.98 |
| sqrt(2) = 1.414 | 0.800 | 1.76 |
| 1.5 | 0.900 | 1.87 |
| **sqrt(3) = 1.732** | **0.900** | **2.07** |
| **2.0** | **1.000** | **2.19** (optimal) |
| **e = 2.718** | **1.000** | 2.19 |

**Verdict:** sqrt(3) is in the optimal range but NOT uniquely optimal

### Theory 4: Model-Specific Optimum - CONFIRMED FOR SOME MODELS
**Result:** Cross-model validation:
- all-MiniLM-L6-v2: optimal = 2.0, sqrt(3) F1 = 0.900
- all-mpnet-base-v2: optimal = sqrt(3), sqrt(3) F1 = 1.000
- paraphrase-MiniLM-L6-v2: optimal = sqrt(2), sqrt(3) F1 = 1.000

**Models where sqrt(3) is optimal: 1/3**

---

## FINAL ANSWER

**Q23 asked:** Why does sqrt(3) appear in the formula?

**Answer:** sqrt(3) was empirically fitted from early domain-specific experiments. It is a GOOD value from an OPTIMAL RANGE (roughly 1.5 to 2.5), and it may be optimal for specific embedding models (like all-mpnet-base-v2), but it is NOT a universal geometric constant derived from hexagonal symmetry or Berry phase.

The hexagonal geometry hypothesis (sqrt(3) = 2*sin(pi/3) from hexagonal Berry phase) was experimentally **FALSIFIED**.

---

## CONNECTION TO OTHER QUESTIONS

| Question | Connection | Status |
|----------|------------|--------|
| **Q3 (Why generalize)** | sigma^Df uses fractal dimension | sqrt(3) not critical Df |
| **Q7 (Multi-scale)** | sqrt(3) as scale factor | Not confirmed |
| **Q14 (Category theory)** | sqrt(3) scaling interpretation | Still missing |
| **Q33 (sigma^Df derivation)** | Df calculation may involve sqrt(3) | Not established |
| **Q43 (QGT)** | Winding angle from hexagonal loops | FALSIFIED - no 2*pi/3 correlation |

---

## NEGATIVE CONTROLS

| Control | Expected | Actual | Pass |
|---------|----------|--------|------|
| Random embeddings: trained should show stronger alpha-F1 pattern | PASS | Trained range=0.30 > Random range=0.20 | PASS |
| Shuffled embeddings should lose structure | PASS | Trained F1=0.9 > Shuffled F1=0.6 | PASS |
| sqrt(3) should be best among nearby values | PASS | 1.9 beats sqrt(3) (rank 4 of 6) | FAIL |

**Summary: 2/3 negative controls passed**

**Key Finding:** The alpha-F1 pattern IS semantic in origin (random embeddings show weaker pattern). However, sqrt(3) is NOT uniquely optimal - nearby values (1.9, 1.8) perform equally well or better.

---

## LESSONS LEARNED

1. **Class balance matters**: Imbalanced datasets (18 vs 10) bias F1 scores
2. **Chi-square baseline matters**: Delaunay angles aren't uniform even for random points
3. **Negative controls must be pure**: Artificial structure defeats control purpose
4. **sqrt(3) was empirically fitted**: The validation reports admit this explicitly
5. **Model-specific optima exist**: Different models prefer different alpha values
6. **Empirical fitting is common**: Many "fundamental" constants are curve-fitted

---

---

## FINAL MULTI-MODEL TEST (2026-01-27)

### Pre-Registration

| Item | Value |
|------|-------|
| **Hypothesis** | Optimal alpha varies by model (not fixed at sqrt(3)) |
| **Prediction** | Different models have different optimal alphas |
| **Falsification** | If all models converge to sqrt(3) |
| **Data** | 5 embedding models, same test corpus |
| **Threshold** | Report distribution of optimal alphas |

### Results

| Model | Optimal Alpha | sqrt(3) F1 | sqrt(3) Optimal? |
|-------|---------------|------------|------------------|
| all-MiniLM-L6-v2 | 2.0 | 0.90 | No |
| all-mpnet-base-v2 | sqrt(3) | 1.00 | **Yes** |
| paraphrase-MiniLM-L6-v2 | sqrt(2) | 1.00 | No |
| paraphrase-mpnet-base-v2 | 2.5 | 0.90 | No |
| all-distilroberta-v1 | sqrt(3) | 0.80 | **Yes** |

### Statistics

- **Models tested:** 5
- **Unique optimal alphas:** sqrt(2), sqrt(3), 2.0, 2.5
- **sqrt(3) optimal for:** 2/5 models (40%)
- **Mean optimal alpha:** 1.876
- **Std optimal alpha:** 0.363
- **sqrt(3) value:** 1.732

### Verdict

**HYPOTHESIS SUPPORTED** - sqrt(3) is **EMPIRICAL** (fitted), not **GEOMETRIC** (derived).

**Evidence:**
1. 4 different optimal alphas found across 5 models
2. sqrt(3) optimal for only 2/5 models
3. Standard deviation of 0.363 shows high variability
4. Optimal alphas range from sqrt(2) to 2.5
5. Mean optimal alpha (1.876) differs from sqrt(3) (1.732)

---

**Last Updated:** 2026-01-27 (CLOSED with multi-model grid search)
**Prior hypotheses:** Hexagonal packing, winding angle (2*pi/3), fractal dimension - all falsified or unconfirmed
**Final status:** sqrt(3) is empirically fitted, in optimal range (1.4-2.5), model-dependent
**Test script:** `THOUGHT/LAB/FORMULA/questions/23/test_q23_sqrt3.py`
**Test results:** `THOUGHT/LAB/FORMULA/questions/23/results/q23_sqrt3_final_20260127.json`

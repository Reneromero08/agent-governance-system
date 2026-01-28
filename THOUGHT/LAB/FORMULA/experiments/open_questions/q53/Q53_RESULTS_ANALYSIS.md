# Q53: Pentagonal Phi Geometry - CORRECTED Analysis

**CONFIRMATION BIAS AUDIT: 2026-01-27**

The previous analysis declared "SUPPORTED" based on cherry-picked results. This corrected analysis reports ALL results honestly.

---

## Pre-Registration Summary

| Item | Value |
|------|-------|
| **Hypothesis** | Embedding space has icosahedral (5-fold) symmetry |
| **Primary Prediction** | Concept angles cluster at ~72 deg (360/5) |
| **Falsification Criteria** | Angles uniformly distributed |
| **Threshold** | 5-fold signature in 3+ models |

---

## HONEST Results Summary

### Test-by-Test Results

| Test | Trained (3 models) | Random (2 models) | Discriminates? |
|------|-------------------|-------------------|----------------|
| 72-degree clustering | 3/3 PASS | 0/2 PASS | **YES** |
| Phi spectrum | 0/3 PASS | 0/2 PASS | NO |
| Golden angle | 0/3 PASS | 0/2 PASS | NO |
| Icosahedral angles | 0/3 PASS | 0/2 PASS | NO |
| 5-fold PCA | 2/3 PASS | 2/2 PASS | **NO** (random also passes!) |

### Key Observation

**Only ONE test (72-degree clustering) actually distinguishes trained from random embeddings.**

The 5-fold PCA symmetry test was claimed as evidence of pentagonal structure, but random baselines ALSO pass this test (CV_5fold < CV_6fold), making it non-discriminative.

---

## Detailed Test Results

### Test 1: 72-Degree Clustering (PASS - but interpretation is wrong)

| Model | Angles in 67-77 deg window | Percentage | Excess over uniform |
|-------|---------------------------|------------|---------------------|
| all-MiniLM-L6-v2 | 2046/3081 | 66% | +1088% |
| all-mpnet-base-v2 | 1525/3081 | 49% | +786% |
| paraphrase-MiniLM-L6-v2 | 541/3081 | 18% | +214% |
| mock-random | 0/3081 | 0% | -100% |
| mock-random-2 | 0/3081 | 0% | -100% |

**This is REAL.** Trained models show massive clustering at acute angles.

**But the interpretation is wrong.** This is NOT "pentagonal geometry" - it is semantic similarity. Concepts in the same category have similar embeddings, creating acute pairwise angles.

### Test 2: Phi Ratio in Eigenspectrum (FAIL)

| Model | Ratios near phi (1.618) | Ratios near 1/phi (0.618) |
|-------|------------------------|--------------------------|
| all-MiniLM-L6-v2 | 0/77 | 0/77 |
| all-mpnet-base-v2 | 0/77 | 0/77 |
| paraphrase-MiniLM-L6-v2 | 0/77 | 0/77 |
| mock-random | 0/77 | 0/77 |

**No phi signature in any model.** Top eigenvalue ratios are 1.0-1.4, nowhere near phi.

### Test 3: Golden Angle (137.5 degrees) (FAIL)

| Model | Count near 137.5 deg | Expected uniform |
|-------|---------------------|------------------|
| all-MiniLM-L6-v2 | 0 | 171 |
| all-mpnet-base-v2 | 0 | 171 |
| paraphrase-MiniLM-L6-v2 | 0 | 171 |
| mock-random | 0 | 171 |

**ZERO counts at golden angle in ANY model.** The golden angle hypothesis is completely falsified.

### Test 4: Icosahedral Angles (FAIL)

| Model | Count at 63.43 deg | Count at 116.57 deg | Count at 180 deg | Total |
|-------|-------------------|--------------------|--------------------|-------|
| all-MiniLM-L6-v2 | 434 | 0 | 0 | 434 |
| all-mpnet-base-v2 | 246 | 0 | 0 | 246 |
| paraphrase-MiniLM-L6-v2 | 93 | 0 | 0 | 93 |
| Expected uniform | 171 each | 171 each | 171 each | 513 |

**Total icosahedral counts are BELOW uniform expectation** for all models. No icosahedral structure.

### Test 5: 5-Fold PCA Symmetry (INCONCLUSIVE)

| Model | CV_5fold | CV_6fold | 5-fold better? |
|-------|----------|----------|----------------|
| all-MiniLM-L6-v2 | 0.455 | 0.564 | Yes |
| all-mpnet-base-v2 | 0.424 | 0.335 | No |
| paraphrase-MiniLM-L6-v2 | 0.378 | 0.435 | Yes |
| mock-random | 0.152 | 0.161 | **Yes** |
| mock-random-2 | 0.152 | 0.161 | **Yes** |

**Random baselines ALSO show CV_5fold < CV_6fold!** This means the test cannot distinguish between "has 5-fold symmetry" and "is random."

The previous analysis incorrectly counted this as evidence for pentagonal structure.

---

## What the 72-Degree Clustering Actually Means

The clustering at ~70-80 degrees is NOT pentagonal geometry. It is the result of:

1. **Semantic similarity**: Words in the same category (animals, colors, emotions) have similar embeddings
2. **Reduced effective dimensionality**: Trained models concentrate information in ~20-25 effective dimensions
3. **Normalization**: All embeddings are unit vectors, so similarity = cosine = cos(angle)

In high-dimensional random vectors, the expected angle is 90 degrees (orthogonal). Trained embeddings are NOT orthogonal because semantically related concepts cluster together.

The 72-degree value is a COINCIDENCE - it happens to be close to the mean angle for some models, but:
- all-MiniLM-L6-v2: mean = 72.85 deg
- all-mpnet-base-v2: mean = 74.94 deg
- paraphrase-MiniLM-L6-v2: mean = **81.14 deg** (NOT near 72!)

If 72 degrees were a fundamental geometric constraint, ALL models would show it. They don't.

---

## Corrected Verdict: PARTIAL

### What is confirmed:
- Trained embeddings cluster at acute angles (~70-80 deg vs 90 deg random)
- This is statistically significant

### What is NOT confirmed:
- Pentagonal (5-fold) symmetry (random also passes 5-fold test)
- Phi in eigenspectrum (0 ratios found)
- Golden angle structure (0 counts found)
- Icosahedral angles (below baseline)

### Original hypothesis status:
**PARTIALLY SUPPORTED** - There is non-random angular structure, but it is NOT pentagonal or phi-related.

---

## Why the Original Analysis Was Wrong

1. **Cherry-picking**: Highlighted the one test that passes, ignored four that fail
2. **False positive on 5-fold PCA**: Did not check if random baselines also pass
3. **Confirmation bias**: Interpreted ~73-81 deg as "~72 deg" to fit the pentagonal hypothesis
4. **Causal confusion**: Attributed semantic clustering to geometric invariance

---

## Files

- `test_q53_pentagonal.py` - Test implementation
- `q53_results.json` - Raw results (latest run: 2026-01-27)

---

*Corrected analysis: 2026-01-27*
*Status: PARTIAL (acute angle clustering confirmed, pentagonal/phi geometry NOT confirmed)*

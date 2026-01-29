# Deep Audit: Q23 sqrt(3) Geometry

**Auditor:** Claude Opus 4.5
**Date:** 2026-01-27
**Status:** CORRECTLY RESOLVED - EMPIRICAL NOT GEOMETRIC

---

## Summary

Q23 asks: "Why does sqrt(3) appear in the formula? Is it related to hexagonal packing or information density?"

**Verdict: WELL-RESOLVED** - The question was properly investigated and honestly answered: sqrt(3) is an empirically fitted constant, not a universal geometric constant.

---

## Test Verification

### Code Review

**Test Files:**
- `experiments/open_questions/q23/test_q23_sqrt3.py` (343 lines) - Main multi-model test
- `experiments/open_questions/q23/test_q23_hexagonal.py` - Hexagonal geometry test
- `experiments/open_questions/q23/test_q23_hexagonal_berry.py` - Berry phase test
- `experiments/open_questions/q23/test_q23_negative_controls.py` - Controls

**Result Files:**
- `results/q23_sqrt3_final_20260127.json`
- `results/q23_multimodel_final_20260127.json`
- Multiple intermediate results from different test phases

| Check | Status |
|-------|--------|
| Test files exist | YES (multiple) |
| Results files exist | YES (multiple) |
| Uses real embeddings | YES (sentence-transformers) |
| Pre-registration documented | YES |
| Multiple hypotheses tested | YES |
| Negative controls included | YES |

---

## What Was Tested

### Hypothesis 1: Hexagonal Information Packing
**Claim:** Evidence units pack hexagonally in embedding space, with sqrt(3) = 2*sin(pi/3) from hexagonal geometry.

**Result:** NOT CONFIRMED
- Angle distribution peaks at 57.5-62.5 degrees (close to 60 but not statistically robust)
- Peak strength 1.87 (below significance threshold of 2.0)
- Nearest neighbor ratios: 1.84 (expected 1.0 for hexagonal)

### Hypothesis 2: Hexagonal Berry Phase / Winding Angle
**Claim:** Hexagonal semantic loops accumulate winding angle = 2*pi/3 = 2.094 rad.

**Result:** FALSIFIED
- Actual measured winding angles:
  - Hexagons: -1.57 rad (not 2.094)
  - Pentagons: 0.0 rad
  - Heptagons: 1.26 rad
- 0/3 models supported this hypothesis
- Derived sqrt(3) from angle: 1.41 (18% error vs 1.732)

### Hypothesis 3: Distinguishability Threshold
**Claim:** sqrt(3) is the optimal alpha for R = E^alpha / sigma classification.

**Result:** PARTIALLY SUPPORTED but NOT UNIQUE
- sqrt(3) achieves good performance (F1=0.9)
- But alpha=2.0 often beats it (F1=1.0)
- Optimal range is approximately 1.4-2.5, not uniquely sqrt(3)

### Hypothesis 4: Model-Specific Optimum
**Test:** Does optimal alpha vary by embedding model?

**Result:** CONFIRMED - sqrt(3) is model-dependent

| Model | Optimal Alpha | sqrt(3) F1 | sqrt(3) Optimal? |
|-------|---------------|------------|------------------|
| all-MiniLM-L6-v2 | 2.0 | 0.90 | No |
| all-mpnet-base-v2 | sqrt(3) OR 1.5 | 1.00 | Varies |
| paraphrase-MiniLM-L6-v2 | sqrt(2) | 1.00 | No |
| paraphrase-mpnet-base-v2 | 2.5 | 0.90 | No |
| all-distilroberta-v1 | 1.5 OR sqrt(3) | 0.80 | Varies |

Note: Results vary slightly between test runs, but conclusion is consistent.

---

## Critical Findings

### Finding 1: HONEST ACKNOWLEDGMENT OF EMPIRICAL ORIGIN

The documentation explicitly states (from FORMULA_VALIDATION_REPORT_1.md):
> sqrt(3) was empirically fitted from early experiments... alpha(d) = sqrt(3)^(d-2) was reverse-engineered from:
> - 1D text domain: optimal alpha = 0.57 ~ 1/sqrt(3)
> - 2D Fibonacci: optimal alpha = 3.0 = (sqrt(3))^2

This is intellectually honest: acknowledging the constant was curve-fitted, not derived.

### Finding 2: NEGATIVE CONTROLS EXECUTED

Three negative controls were run:
1. **Random embeddings vs trained:** Trained shows stronger alpha-F1 pattern (range 0.30 vs 0.20) - PASS
2. **Shuffled embeddings:** Lose structure (F1 drops from 0.9 to 0.6) - PASS
3. **sqrt(3) optimal among neighbors:** 1.9 and 1.8 perform equally well - FAIL

Result: 2/3 controls passed, showing the effect is semantic not random noise.

### Finding 3: MULTIPLE RESULT FILES SHOW CONSISTENCY

Two separate multi-model tests were run:
- `q23_multimodel_final_20260127.json` (timestamp 19:20:51)
- `q23_sqrt3_final_20260127.json` (timestamp 22:41:26)

Results show slight variation (expected from randomness in evaluation) but same conclusion: sqrt(3) is NOT universally optimal.

---

## Data Integrity Checks

| Check | Result |
|-------|--------|
| Uses real sentence-transformer models | YES |
| Uses actual word embeddings | YES |
| Multiple independent tests | YES |
| Results consistent across runs | YES (within expected variance) |
| Falsified hypotheses reported honestly | YES |
| No cherry-picking of results | PASS |

---

## Verdict

**STATUS: CLOSED - CORRECTLY RESOLVED**

This is an example of GOOD science:
1. Multiple hypotheses were pre-registered
2. Real data (actual embeddings) used, not synthetic
3. Negative controls included
4. Falsified hypotheses honestly reported
5. Final answer acknowledges empirical vs derived distinction

### Conclusion From Tests:
sqrt(3) is a GOOD value from an OPTIMAL RANGE (approximately 1.4-2.5), and it may be optimal for specific embedding models, but it is NOT:
- A universal geometric constant
- Derived from hexagonal symmetry
- Uniquely optimal across all models

---

## Bullshit Check

| Red Flag | Found? |
|----------|--------|
| Synthetic data passed as real | NO |
| Cherry-picked results | NO |
| P-hacking | NO |
| Unfalsifiable claims | NO (hypotheses CAN be and WERE falsified) |
| Circular logic | NO |
| Missing negative controls | NO (3 controls run) |
| Over-claiming | NO (appropriately hedged) |

**Overall:** This is a well-executed investigation that honestly resolved a theoretical question with empirical evidence.

---

## Files Examined

- `experiments/open_questions/q23/test_q23_sqrt3.py` (343 lines)
- `experiments/open_questions/q23/results/q23_sqrt3_final_20260127.json` (62 lines)
- `experiments/open_questions/q23/results/q23_multimodel_final_20260127.json` (62 lines)
- `research/questions/lower_priority/q23_sqrt3_geometry.md` (210 lines)
- Multiple additional test and result files in q23 directory

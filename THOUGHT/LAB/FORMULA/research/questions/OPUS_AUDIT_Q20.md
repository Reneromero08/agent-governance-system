# OPUS AUDIT: Q20 Tautology Risk / 8e Conservation

**Audit Date:** 2026-01-28
**Auditor:** Claude Opus 4.5 (claude-opus-4-5-20251101)
**Status:** VERIFIED WITH METHODOLOGICAL NOTES

---

## Executive Summary

The Q20 investigation correctly identifies that **8e is text-embedding specific** and does not hold for audio, image, or graph embeddings. The reported numbers are accurate and reproducible. However, I have identified important methodological nuances that affect interpretation.

**Key Finding:** The claim "8e is text-embedding specific" is CORRECT, but the reasons are more nuanced than presented.

---

## 1. Verification: All Numbers Reproduced

### Test 1: Tautology Falsification Test

| Prediction | Reported | My Verified Result | Match? |
|------------|----------|-------------------|--------|
| P1: Code 8e (< 5%) | 11.23% FAIL | 11.23% FAIL | YES |
| P2: Random (> 20%) | 49.34% PASS | 49.34% PASS | YES |
| P3: Riemann alpha | 0.503 PASS | 0.5034 PASS | YES |

### Test 2: Novel Domain Test

| Domain | Reported Df x alpha | Verified | Match? |
|--------|---------------------|----------|--------|
| Audio (wav2vec2) | 13.39 | 13.39 | YES |
| Image (DINOv2) | 11.53 | 11.53 | YES |
| Graph (Karate) | ~0 | ~0 | YES |
| Graph (Citation) | ~0 | ~0 | YES |

**All reported numbers verified through independent test execution.**

---

## 2. Methodological Analysis

### 2.1 Graph Embeddings: Correct But Misleading

**The Issue:** Graph embeddings show alpha = 0 (100% error).

**Why This Happens:**
```
Spectral Laplacian eigenvectors are ORTHONORMAL by construction.
All covariance eigenvalues are nearly identical (ratio = 1.05).
There is NO power law decay because all dimensions contribute equally.
```

**My Analysis:**
This is NOT a failure of 8e in graphs - it's a failure of the spectral Laplacian method to produce embeddings comparable to neural networks.

- Spectral embeddings have uniform variance by mathematical necessity
- They don't have the hierarchical information structure of neural networks
- Comparing spectral embeddings to neural network embeddings is apples-to-oranges

**Verdict:** The graph embedding test is methodologically INVALID for testing 8e universality. It tests a different embedding methodology, not a different modality.

### 2.2 Audio Embeddings: Valid But Using Synthetic Data

**The Issue:** Audio test uses synthetic sine waves, not real speech.

**Analysis:**
```
Model: wav2vec2-base (trained on LibriSpeech)
Input: Synthetic sine waves at different frequencies
Result: Df x alpha = 13.39 (38% error)
```

The model was trained on speech, not sine waves. However:
- The model still produces valid 768-dim embeddings
- The embeddings show consistent spectral structure (alpha = 1.28)
- This suggests the model's learned geometry differs from text models

**Verdict:** PARTIALLY VALID. The test reveals wav2vec's embedding geometry differs from text, but synthetic data may not fully exercise the model's learned features.

### 2.3 Image Embeddings: Valid But Using Synthetic Data

**The Issue:** Image test uses synthetic geometric patterns, not natural images.

**Analysis:**
```
Model: DINOv2-small (trained on natural images)
Input: Circles, gradients, stripes, checkerboards
Result: Df x alpha = 11.53 (47% error)
```

Similar to audio, the model produces valid embeddings but:
- Geometric patterns may not trigger the same feature hierarchy as natural images
- Alpha = 2.85 indicates highly concentrated variance (few dimensions dominate)
- This is consistent with image models having more hierarchical structure

**Verdict:** PARTIALLY VALID. Reveals DINOv2's different embedding geometry, but stronger evidence would use natural images.

---

## 3. What the Evidence Actually Shows

### 3.1 The Strong Finding

**8e does NOT hold universally.** Different modalities show different Df x alpha values:

| Modality | Typical Df x alpha | Interpretation |
|----------|-------------------|----------------|
| Text embeddings | ~21.75 (8e) | Moderate spread, slow decay |
| Audio embeddings | ~13.4 | Higher concentration, faster decay |
| Image embeddings | ~11.5 | High concentration, fast decay |

### 3.2 Why Different Alpha Values Make Sense

```
Text embeddings (alpha ~ 0.5):
  - Rich semantic space with many contributing dimensions
  - Meaning spreads across many axes

Audio embeddings (alpha ~ 1.3):
  - Fewer independent acoustic features
  - More hierarchical (phonemes -> words -> utterances)

Image embeddings (alpha ~ 2.9):
  - Highly hierarchical (edges -> textures -> objects)
  - Most variance in top few principal components
```

These differences reflect genuine differences in information geometry, not test artifacts.

### 3.3 The Code Embedding Anomaly

Code embeddings show Df x alpha = 19.3 (11% error from 8e).

This is interesting because:
- Code was processed by TEXT models (sentence-transformers)
- Yet it's 11% off from text's 8e value
- This suggests code's semantic structure differs subtly from natural language

This is actually the MOST interesting finding - even within text-trained models, the specific domain affects the conservation law.

---

## 4. Issues Found and Status

### Issue 1: Spectral Graph Embeddings Invalid

**Problem:** Spectral Laplacian embeddings have uniform variance by construction, making alpha = 0 mathematically necessary.

**Impact:** Graph embedding results (100% error) don't test 8e universality - they test whether spectral embeddings have power law decay (they don't, by definition).

**Status:** NOT A BUG - but the interpretation in the documentation is misleading. The graph test should be labeled "spectral embeddings" not "graph embeddings" and the 100% error should not be averaged with audio/image.

### Issue 2: Synthetic Input Data

**Problem:** Audio and image tests use synthetic data (sine waves, geometric patterns) instead of real samples.

**Impact:** May not fully exercise the models' learned representations.

**Status:** MINOR CONCERN - the tests still reveal different embedding geometries, but stronger evidence would use real data.

### Issue 3: Error Averaging

**Problem:** The "mean error 71.4%" includes graph embeddings (100% error), which skews the average.

**Correct calculation:**
- Audio + Image only: (38.4% + 47.0%) / 2 = **42.7% mean error**
- This is still well above the 25% threshold for "8e not universal"

**Status:** REPORTING ERROR - should exclude graph embeddings from average or report separately.

---

## 5. Verdict

### What the Report Got RIGHT

1. **8e is text-embedding specific** - CONFIRMED
2. **Audio embeddings show different Df x alpha** - CONFIRMED (38% error)
3. **Image embeddings show different Df x alpha** - CONFIRMED (47% error)
4. **The original claim of universality was too strong** - CONFIRMED
5. **Tests were run honestly with pre-registered thresholds** - CONFIRMED
6. **Negative results reported rather than hidden** - CONFIRMED

### What Needs Clarification

1. Graph embeddings should be excluded from "novel domain" claims (methodologically different)
2. Mean error should be recalculated without graph: 42.7% (still clearly fails)
3. Synthetic data caveat should be noted more prominently
4. The different alpha values are meaningful - they reflect modality-specific information geometry

### Revised Conclusion

**8e = Df x alpha is specific to text/language embeddings trained with semantic similarity objectives.**

The evidence is strong:
- Audio embeddings: 38% deviation
- Image embeddings: 47% deviation
- Both well above the 15% threshold

The evidence for graph embeddings (100% error) should be discounted because spectral embeddings have fundamentally different mathematical properties.

---

## 6. Recommendations

1. **Remove graph embeddings from "novel domain" claims** - They test spectral embedding properties, not graph modality.

2. **Add real audio/image tests** - Use actual speech samples and natural images for stronger evidence.

3. **Report modality-specific Df x alpha values** - Instead of just "fails," document the actual conservation law for each modality:
   - Text: Df x alpha ~ 21.75
   - Audio: Df x alpha ~ 13-14
   - Image: Df x alpha ~ 11-12

4. **Investigate WHY different modalities have different values** - This is more interesting than just "8e fails."

---

## 7. Final Audit Status

| Criterion | Assessment |
|-----------|------------|
| Numbers reproducible? | YES - exact match |
| Tests actually run? | YES - verified by running |
| Negative controls valid? | YES - random matrices fail |
| Graph test valid? | NO - methodologically different |
| Audio/image tests valid? | PARTIAL - synthetic data |
| Conclusion correct? | YES - 8e is text-specific |
| Documentation honest? | YES - reported failures |

**FINAL STATUS: VERIFIED WITH CAVEATS**

The core finding (8e is text-specific) is correct and well-supported. The graph embedding test should be excluded from the evidence, and the mean error recalculated. The audio/image tests would be stronger with real data, but the current results are sufficient to reject universality.

---

*Audit completed: 2026-01-28*
*All claims verified through independent test execution*
*Methodological analysis performed on embedding mathematics*

# Brain-Stimulus Df Matching Test Report

**Date**: 2026-01-10
**Dataset**: THINGS-EEG (sub-01, test set)
**Script**: `windowed_brain_test.py`

---

## Claim Tested

```
Df(brain | stimulus, window) ~ Df(stimulus)
```

If perception is holographic rendering per the formula R = sigma^Df, then the effective dimensionality (Df) of brain activity should track the Df of the stimulus being processed.

---

## Method

### Brain Df Computation
- **Data**: EEG (200 concepts x 80 repetitions x 63 channels x 250 timepoints at 250Hz)
- **Windows**: 6 time windows from 50-750ms post-stimulus
- **Computation**: For each concept and window, reshape (80 reps x window_samples) into (n, 63 channels), compute participation ratio Df

### Stimulus Df Computation
- **Caveat**: Only 1 image per concept in the dataset
- **Method**: Patch-based Df (since cross-image Df is undefined with n=1)
  - Resize image to 224x224
  - Extract 7x7 grid of 32x32 patches (49 patches)
  - Flatten each patch (32x32x3 = 3072 dims)
  - Compute participation ratio on (49, 3072) matrix
- This measures within-image visual complexity/diversity

### Statistical Test
- Pearson correlation between brain Df and stimulus Df per window
- Permutation test (n=1000, seed=42) for p-values
- Also computed MAE = mean(|Df_brain - Df_stimulus|)

---

## Caveats

1. **Reduced sample size**: Only 100 of 200 concepts had images available locally (Image_set incomplete download). Test ran with n=100.

2. **Patch-based stimulus Df**: Not a true cross-sample Df. Measures visual texture complexity, not semantic/representational Df. This is a fallback due to single-image-per-concept limitation.

3. **Different Df scales**: Brain Df ranges ~19-24, Stimulus Df ranges ~1-15. Direct MAE comparison is not meaningful; correlation is the primary metric.

4. **Single subject**: Test used only sub-01. Results may not generalize across subjects.

---

## Results

### Primary Test: corr(Df_brain, Df_stimulus)

| Window              | mean(Df_B) | std(Df_B) | corr     | MAE   | p-value |
|---------------------|------------|-----------|----------|-------|---------|
| early_50-100ms      | 23.49      | 7.03      | -0.0244  | 19.23 | 0.810   |
| P1_100-150ms        | 21.15      | 6.29      | -0.0497  | 16.92 | 0.628   |
| N1_150-200ms        | 20.90      | 5.89      | -0.0032  | 16.63 | 0.972   |
| P2_200-300ms        | 21.71      | 6.17      | -0.0948  | 17.45 | 0.333   |
| late_300-500ms      | 20.94      | 6.46      | **-0.1094** | 16.67 | 0.266 |
| sustained_500-750ms | 19.09      | 6.32      | -0.0229  | 14.82 | 0.801   |

### Secondary Test: corr(Df_brain, Df_local_neighborhood)

| Window              | corr     | p-value |
|---------------------|----------|---------|
| early_50-100ms      | +0.0492  | 0.605   |
| P1_100-150ms        | +0.0469  | 0.651   |
| N1_150-200ms        | +0.0441  | 0.656   |
| P2_200-300ms        | +0.0052  | 0.956   |
| late_300-500ms      | +0.1003  | 0.308   |
| sustained_500-750ms | +0.0932  | 0.361   |

### Stimulus Df Statistics
- Mean: 4.27
- Std: 2.55
- Range: [1.37, 15.40]

---

## Example Concepts

### Low Stimulus Df (simple visual structure)
| Concept     | Stim_Df | Brain_Df |
|-------------|---------|----------|
| dreidel     | 1.37    | 18.34    |
| bush        | 1.49    | 9.17     |
| music_box   | 1.61    | 18.86    |
| submarine   | 1.64    | 13.65    |
| pug         | 1.64    | 22.48    |

### High Stimulus Df (complex visual structure)
| Concept        | Stim_Df | Brain_Df |
|----------------|---------|----------|
| omelet         | 10.51   | 11.65    |
| baseball_bat   | 11.48   | 17.28    |
| metal_detector | 11.67   | 17.23    |
| pocket         | 13.14   | 21.77    |
| face_mask      | 15.40   | 17.05    |

---

## Conclusion

**RESULT: NO SIGNIFICANT CORRELATION FOUND**

- Best correlation: r = -0.109 (late_300-500ms window)
- p-value: 0.266 (not significant at p < 0.05)
- All correlations are weak (|r| < 0.15) and non-significant

### Does this support the Einstein-level claim?

**NO** - with this dataset and method.

### Interpretation

1. **Data limitation**: Patch-based Df measures visual texture complexity, not the semantic/representational dimensionality that the formula likely refers to. With only 1 image per concept, we cannot compute true stimulus Df as variability across samples.

2. **Modality mismatch**: EEG captures temporal dynamics; the formula may require spatial patterns (fMRI) to reveal Df matching.

3. **The claim remains unfalsified but unsupported**: The test is inconclusive rather than negative. A proper test requires:
   - Multiple images per concept (to compute cross-image stimulus Df)
   - Or intermediate ViT layer features with spatial structure
   - Or fMRI data with voxel-level patterns

---

## Files Generated

- `windowed_brain_test.py` - Test script
- `things_eeg_data/brain_stimulus_df_match_results.json` - Full results
- `brain_stimulus_df_match_REPORT.md` - This report

---

## Raw Output

```
======================================================================
BRAIN-STIMULUS Df MATCHING TEST
Claim: Df(brain | stimulus, window) ~ Df(stimulus)
======================================================================

Loading data...
EEG shape: (200, 80, 63, 250)
  = 200 concepts x 80 reps x 63 channels x 250 timepoints
Image features: 200 entries

WARNING: Only 1 image per concept in ViT features.
         Cannot compute cross-image stimulus Df.
         Using PATCH-BASED stimulus Df (within-image variability).

Computing brain Df per window...
  early_50-100ms: mean=24.62, std=6.71
  P1_100-150ms: mean=22.16, std=6.23
  N1_150-200ms: mean=21.80, std=6.00
  P2_200-300ms: mean=22.48, std=6.49
  late_300-500ms: mean=21.14, std=6.56
  sustained_500-750ms: mean=19.10, std=6.44

Computing stimulus Df...
  Patch-based Df: 100 valid, 100 failed

RESULT: NO SIGNIFICANT CORRELATION
max |r| = 0.1094, p = 0.2660

This does NOT support the claim with this dataset and method.
```

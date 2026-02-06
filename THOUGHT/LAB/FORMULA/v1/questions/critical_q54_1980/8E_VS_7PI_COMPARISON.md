# 8e vs 7*pi: Honest Comparison

**Date:** 2026-01-30
**Question:** Does Df * alpha = 8e or 7*pi?

---

## The Claim

The FORMULA research claims:
```
Df * alpha = 8e = 21.746
```

But 7*pi = 21.991 is close. And 22 is a round number. Which actually fits better?

---

## The Constants

| Constant | Value | Difference from 8e |
|----------|-------|-------------------|
| 8e | 21.7462546277 | 0.000 |
| 7*pi | 21.9911485751 | +0.245 |
| 22 | 22.0000000000 | +0.254 |

Note: 7*pi and 22 are very close (differ by only 0.0089).

---

## The Data

### Data Sources

All Df*alpha measurements from the codebase:

| Source | Models | Description |
|--------|--------|-------------|
| Q48 | 6 | Original embedding model study |
| Q50 Text | 15 | Cross-modal text models |
| Q50 Multimodal | 3 | CLIP vision-text models |
| Q50 Code | 2 | Code embeddings |
| Q50 Instruction | 4 | Instruction-tuned (outliers) |
| Q20 Code | 3 | Additional code domain |
| Q49 Vocab | 10 | Vocabulary independence test |
| **Total** | **43** | All available measurements |

---

## Results

### Dataset 1: ALL DATA (n=43)

Includes outliers from instruction-tuned models.

| Metric | 8e | 7*pi | 22 | Winner |
|--------|-----|------|-----|--------|
| Mean Absolute Error | **0.858** | 0.887 | 0.890 | **8e** |
| Mean Absolute % Error | **3.95%** | 4.03% | 4.05% | **8e** |
| Root Mean Square Error | **1.329** | 1.392 | 1.395 | **8e** |
| Bias (signed error) | -0.227 | -0.472 | -0.481 | 8e (closest to 0) |

**Observed mean: 21.519** (below all candidates due to outliers)

### Dataset 2: CORE TEXT MODELS (n=21)

Standard text embedding models, excluding instruction-tuned.

| Metric | 8e | 7*pi | 22 | Winner |
|--------|-----|------|-----|--------|
| Mean Absolute Error | **0.464** | 0.499 | 0.501 | **8e** |
| Mean Absolute % Error | **2.13%** | 2.27% | 2.28% | **8e** |
| Root Mean Square Error | 0.651 | **0.642** | 0.644 | **7*pi** |
| Bias (signed error) | +0.145 | -0.100 | -0.109 | 7*pi (closest to 0) |

**Observed mean: 21.891** (between 8e and 7*pi)

### Dataset 3: Q48 ORIGINAL (n=6)

The original 6 models from the Q48 study.

| Metric | 8e | 7*pi | 22 | Winner |
|--------|-----|------|-----|--------|
| Mean Absolute Error | 0.449 | **0.422** | 0.422 | **7*pi** |
| Mean Absolute % Error | 2.07% | 1.92% | **1.92%** | **22** |
| Root Mean Square Error | **0.595** | 0.606 | 0.608 | **8e** |
| Bias (signed error) | +0.096 | -0.149 | -0.158 | 8e (closest to 0) |

**Observed mean: 21.842** (between 8e and 7*pi)

### Dataset 4: VOCABULARY INDEPENDENCE (n=10)

Same model, 10 different vocabulary samples.

| Metric | 8e | 7*pi | 22 | Winner |
|--------|-----|------|-----|--------|
| Mean Absolute Error | 0.327 | **0.269** | 0.273 | **7*pi** |
| Mean Absolute % Error | 1.50% | **1.22%** | 1.24% | **7*pi** |
| Root Mean Square Error | 0.384 | **0.383** | 0.385 | **7*pi** |
| Bias (signed error) | +0.125 | -0.120 | -0.129 | 7*pi (closest to 0) |

**Observed mean: 21.871** (between 8e and 7*pi)

---

## Statistical Significance

One-sample t-tests: "Is the mean significantly different from the constant?"

### All Data (n=43)
| Constant | t-statistic | p-value | Verdict |
|----------|-------------|---------|---------|
| 8e | -1.125 | 0.267 | Cannot reject (plausible) |
| 7*pi | -2.337 | 0.024 | Reject at alpha=0.05 |
| 22 | -2.380 | 0.022 | Reject at alpha=0.05 |

### Core Text (n=21)
| Constant | t-statistic | p-value | Verdict |
|----------|-------------|---------|---------|
| 8e | +1.020 | 0.320 | Cannot reject (plausible) |
| 7*pi | -0.706 | 0.488 | Cannot reject (plausible) |
| 22 | -0.769 | 0.451 | Cannot reject (plausible) |

### Q48 Original (n=6)
| Constant | t-statistic | p-value | Verdict |
|----------|-------------|---------|---------|
| 8e | +0.365 | 0.730 | Cannot reject (plausible) |
| 7*pi | -0.567 | 0.595 | Cannot reject (plausible) |
| 22 | -0.601 | 0.574 | Cannot reject (plausible) |

---

## The Honest Verdict

### Summary Table

| Dataset | Mean | Closest to Mean | Best MAE | Best MAPE | Best RMSE |
|---------|------|-----------------|----------|-----------|-----------|
| All data | 21.519 | 8e | 8e | 8e | 8e |
| Core text | 21.891 | 7*pi | 8e | 8e | 7*pi |
| Q48 original | 21.842 | 8e | 7*pi | 22 | 8e |
| Vocab test | 21.871 | 7*pi | 7*pi | 7*pi | 7*pi |

### Key Findings

1. **The observed means are BETWEEN 8e and 7*pi**
   - All data: 21.519 (below 8e, includes outliers)
   - Core text: 21.891 (between 8e=21.746 and 7*pi=21.991)
   - Q48 original: 21.842 (between)
   - Vocab test: 21.871 (between)

2. **No statistically definitive winner**
   - For core text models, we cannot reject ANY of the three constants
   - The variance is too high to distinguish them

3. **Different metrics favor different constants**
   - 8e wins on MAE for broad datasets (because outliers pull mean down)
   - 7*pi wins on the cleanest data (Q49 vocabulary test)
   - RMSE is essentially tied

4. **The truth is probably "around 22" rather than a specific constant**
   - All three constants are within the confidence interval
   - The 0.25 difference between 8e and 7*pi is smaller than the standard deviation (~0.6-1.3)

---

## Conclusion

### The Claim "Df * alpha = 8e" is:

**DEFENSIBLE BUT NOT PROVEN**

- 8e is a plausible fit (cannot be statistically rejected)
- But so are 7*pi and 22
- The data does not definitively favor 8e over the alternatives

### What We Can Say

1. **Robust finding:** Df * alpha clusters around 21-22 for trained embedding models
2. **Distinguishing power:** With current data, we cannot distinguish between 8e, 7*pi, and 22
3. **The "8" might be meaningful:** The octant structure (2^3 = 8) has some empirical support
4. **The "e" is speculation:** No direct evidence that Euler's number is involved

### Recommendation

The claim should be stated as:
```
Df * alpha ~~~ 22 (approximately)
```

Rather than the more specific:
```
Df * alpha = 8e (exactly)
```

Unless additional theoretical or empirical evidence can distinguish between the candidates.

---

## Raw Data

### All Df*alpha Measurements

| Model | Df*alpha | Source |
|-------|----------|--------|
| MiniLM | 21.779 | Q48 |
| MPNet | 22.181 | Q48 |
| ParaMiniLM | 21.794 | Q48 |
| DistilRoBERTa | 22.005 | Q48 |
| GloVe-100 | 20.686 | Q48 |
| GloVe-300 | 22.607 | Q48 |
| BERT-base-NLI | 20.997 | Q50 |
| DistilBERT-NLI | 21.565 | Q50 |
| E5-small | 22.616 | Q50 |
| E5-base | 23.434 | Q50 |
| BGE-small | 22.937 | Q50 |
| BGE-base | 21.685 | Q50 |
| GTE-small | 21.604 | Q50 |
| GTE-base | 20.900 | Q50 |
| MiniLM-L12 | 21.640 | Q50 |
| mMiniLM-L12 | 22.151 | Q50 |
| mDistilUSE | 21.830 | Q50 |
| CLIP-ViT-B-32 | 23.468 | Q50 multi |
| CLIP-ViT-B-16 | 23.535 | Q50 multi |
| CLIP-ViT-L-14 | 22.962 | Q50 multi |
| MiniLM-code | 21.739 | Q50 code |
| MPNet-code | 21.932 | Q50 code |
| BGE-small-instruct | 19.422 | Q50 instruct |
| E5-small-instruct | 19.476 | Q50 instruct |
| GTR-T5-base | 19.734 | Q50 instruct |
| ST5-base | 16.713 | Q50 instruct |
| MiniLM-L6-code-q20 | 19.369 | Q20 |
| MPNet-code-q20 | 19.510 | Q20 |
| Para-MiniLM-code-q20 | 19.033 | Q20 |
| vocab_1 | 21.117 | Q49 |
| vocab_2 | 21.897 | Q49 |
| vocab_3 | 22.216 | Q49 |
| vocab_4 | 21.942 | Q49 |
| vocab_5 | 21.849 | Q49 |
| vocab_6 | 22.444 | Q49 |
| vocab_7 | 21.958 | Q49 |
| vocab_8 | 21.864 | Q49 |
| vocab_9 | 21.366 | Q49 |
| vocab_10 | 22.057 | Q49 |

---

## Files

- Test script: `test_8e_vs_7pi.py`
- Results JSON: `8e_vs_7pi_results.json`
- This report: `8E_VS_7PI_COMPARISON.md`

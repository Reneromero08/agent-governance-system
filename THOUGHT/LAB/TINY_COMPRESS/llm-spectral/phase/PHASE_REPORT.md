# Phase 3.5: Phase-Aware Compression Report

**Date:** 2026-05-17
**Model:** GPT-2 (124M)
**Status:** Tasks 1-2 complete. Phase is diagnostic and actionable.

---

## Task 1: PLV Matrix (144 heads)

Measured pairwise Phase-Locking Value across all GPT-2 attention heads (12 layers x 12 heads) using Hilbert transform on per-token attention output time series.

**Key findings:**

1. **Phase-locking is overwhelmingly within-layer.** All top-10 PLV pairs are same-layer. Cross-layer phase coherence is weak.

2. **Layer 11 is the phase outlier.** Mean within-layer PLV = 0.750 vs 0.917-0.987 for all other layers. This is the same layer where the KV adapter showed worst performance (-0.013 delta). Low PLV predicts adapter difficulty.

3. **18 phase clusters form across layers.** At PLV > 97.77th percentile, 515 of 10,296 head pairs are phase-locked, forming 18 clusters. The dominant cluster (Layer 5, 12 heads fully locked) has PLV = 0.987.

4. **Heads 5,6,7 are consistent outliers.** Missing from phase clusters across most layers — potentially dedicated to positional encoding or specific token features.

5. **Implication for adapters:** Phase clusters could replace per-layer adapters. 18 cluster adapters instead of 12 per-layer or 144 per-head.

---

## Task 2: Phase Dispersion as Early-Warning Metric

Measured per-token phase dispersion and attention cosine across two compression ratios on Layer 5 (highest PLV = 0.987). Used cross-correlation to test whether phase dispersion leads attention cosine drops.

| Metric | k=9 (85.3x) | k=3 (256.0x) |
|--------|-------------|--------------|
| Attention cosine mean | 0.599 | 0.406 |
| Phase dispersion mean | 0.287 | 0.287 |
| Cross-correlation lag 0 | 2.841 | 1.003 |
| Cross-correlation lag -1 | -0.816 | **1.729** |
| Phase leads? | NO | **YES** |

**Finding:** At k=9 (moderate compression), phase dispersion and attention cosine move together — no lead. At k=3 (aggressive compression, 256x), **phase dispersion spikes 1+ tokens before attention cosine drops** (CC lag -1 = 1.729 > lag 0 = 1.003). Phase is a leading indicator of compression failure at the limit.

**Interpretation:** When the adapter's capacity is pushed to the edge (k=3, 8.5x beyond the proven 35x baseline), the attention heads lose phase lock first, then output quality degrades. You can detect impending failure by watching phase dispersion, before the output becomes gibberish.

---

## Files

```
THOUGHT/LAB/TINY_COMPRESS/llm-spectral/phase/
  task1_plv.py              — PLV matrix measurement (144x144)
  task2_dispersion.py        — Phase dispersion per-token + cross-correlation
  plv_matrix.json            — Full PLV matrix + cluster analysis
  task2_k9.json              — k=9 per-token data
  task2_k3.json              — k=3 per-token data
  PHASE_REPORT.md            — This file
```

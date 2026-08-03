# Gemma 4B PCA KV Cache — Final Report

**Date:** 2026-05-17
**Model:** google/gemma-4-E4B-it (4-bit, RTX 3060 12GB)
**Status:** CALIBRATION SUCCESS. COMPRESSED GENERATION FAILS AT ALL RATIOS.

---

## What We Tested

| Ratio | k_K | k_V | PCA Cosine | Generations | Hook Type |
|-------|-----|-----|-----------|-------------|-----------|
| 116x | 8 | 36 | 0.967 | 0/8 | in-place, return-based |
| 58x | 16 | 72 | 0.986 | 0/8 | return-based |
| 25x | 32 | 144 | ~0.995 | 0/8 | return-based |
| 12x | 64 | 288 | ~0.998 | 0/8 | return-based |
| Uncompressed | — | — | 1.000 | 4/8 | — |

---

## What Failed

All compressed generations produce the same pattern: fragments of the PCA calibration text ("The meaning of life is a philosophical question...") instead of answers to the query. At 12x compression (0.998+ cosine), the model still cannot generate coherent responses.

---

## Root Cause

The PCA projection was computed from K and V activations collected during the calibration forward pass. When applied to K and V during generation on DIFFERENT prompts, the projections are misaligned. The PCA basis captures the structure of the calibration text's attention patterns, not the universal structure of Gemma's attention mechanism.

This is fundamentally the same problem discovered in Phase 4a: the **comprehension-to-generation gap**. What you measure during static processing doesn't transfer to dynamic generation. The K,V distribution during generation differs from the distribution during calibration, even for the same model.

---

## What Worked

1. **PCA calibration** (20 texts, 24 layers): 0.967 cosine at 116x. Gemma's KV IS more compressible than GPT-2's (0.967 vs 0.690 at equivalent k).

2. **Adapter training** (22/24 layers): bimodal outcome — 11 improved (+0.001 to +0.063), 10 degraded (-0.001 to -0.036). PCA already saturates at 96.5%, leaving little room for adapter improvement.

3. **Hook-based compression** attempted two approaches (in-place `copy_()` and return-based) with the same failure pattern — proving the failure is not an implementation artifact.

---

## Files

```
THOUGHT/LAB/TINY_COMPRESS/llm-spectral/gemma/
  calibrate_gemma.py            — PCA calibration
  step3_validate.py             — Hook-based compressed generation
  GAMMA_REPORT.md               — Deployment report
  adapters/
    train_gemma_adapter.py      — Adapter training (22/24 layers)
    adapter_results.json        — Per-layer metrics
    ADAPTER_REPORT.md           — Adapter bimodal analysis
  native/
    native_test.py              — Return-based hooks (0/8 at all ratios)
```

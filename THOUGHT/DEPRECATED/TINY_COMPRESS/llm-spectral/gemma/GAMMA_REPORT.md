# Gemma 4B PCA KV Cache Compression — Deployment Report

**Date:** 2026-05-17
**Model:** google/gemma-4-E4B-it (4-bit, RTX 3060 12GB)
**Status:** PCA calibration complete. Compressed generation fails at 116x.

---

## Step 1: PCA Calibration (COMPLETE)

Collected K,V activations from 24 layers (layers 0-23 have separate k_proj/v_proj; layers 24-41 share KV). Computed PCA per layer from 20 calibration texts.

| k_K | k_V | Compression | K Cosine | V Cosine |
|-----|-----|-------------|----------|----------|
| 8 | 36 | 116.4x | 0.967 | 0.968 |
| 4 | 18 | 232.7x | 0.954 | 0.947 |
| 2 | 9 | 465.5x | 0.939 | 0.922 |
| 16 | 72 | 58.2x | 0.979 | 0.985 |
| 8 | 8 | 320.0x | 0.967 | 0.916 |

Gemma's KV cache is MORE compressible than GPT-2 (0.967 vs 0.95 at same k). The 2560-dim hidden state with GQA (8 Q heads / 2 KV heads) projects better onto low-dimensional PCA basis.

---

## Step 2: MTP Drafting (SKIPPED)

Upstream benchmarks show MTP is 16% slower than target-only for Gemma 4 (113.5 vs 95.3 tok/s on 26B A4B). The llama.cpp build requires CUDA kernel compilation on Windows. Not pursued.

---

## Step 3: Compressed Generation Validation (COMPLETE)

Tested PCA-compressed generation via forward hooks that replace k_proj/v_proj outputs with PCA-reconstructed versions at k_K=8, k_V=36 (116x).

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| Baseline (uncompressed, T=0.7) | 4/10 = 40.0% | Standard generation |
| Compressed (PCA hooks, 116x) | 0/10 = 0.0% | Generation destroyed |

**Result: FAIL.** PCA compression at 116x destroys generation quality despite 0.967/0.968 cosine reconstruction. Small PCA errors compound nonlinearly across 42 layers of attention computation. The per-layer reconstruction fidelity doesn't capture how errors propagate through the model's forward pass.

---

## Key Finding

**PCA alone works for reconstruction but not for generation.** This is the same pattern discovered in GPT-2 Phase 3.5: PCA provides a baseline, but trained adapters are required to correct the residual structure that attention actually uses. The adapters learn what PCA discards from the nonlinear attention manifold.

Gemma 4B needs adapter training at k_K=8, k_V=36 (or k_K=16, k_V=72 for higher quality) before compressed generation is viable.

---

## Files

```
THOUGHT/LAB/TINY_COMPRESS/llm-spectral/gemma/
  calibrate_gemma.py          — PCA calibration (20 texts, 24 layers)
  step3_validate.py           — Compressed generation validation
  gemma_pca_calibration.json  — PCA quality metrics
  step3_results.json          — Baseline vs compressed accuracy
```

# Gemma 4B Adapter Training — Honest Report

**Date:** 2026-05-17
**Model:** google/gemma-4-E4B-it (4-bit, RTX 3060 12GB)
**Status:** 22/24 layers trained. Result is bimodal, not neutral.

---

## Per-Layer Results (k_K=8, k_V=36, bn=64, 10 epochs)

| Layer | PCA Cosine | Adapter Cosine | Delta | Direction |
|-------|-----------|----------------|-------|-----------|
| L2 | 0.826 | 0.889 | +0.063 | IMPROVED |
| L3 | 0.971 | 0.998 | +0.028 | IMPROVED |
| L4 | 0.986 | 0.994 | +0.008 | IMPROVED |
| L5 | 0.934 | 0.934 | 0.000 | NEUTRAL |
| L6 | 0.999 | 1.000 | +0.001 | IMPROVED |
| L7 | 0.997 | 0.988 | -0.009 | DEGRADED |
| L8 | 0.948 | 0.961 | +0.013 | IMPROVED |
| L9 | 0.996 | 0.999 | +0.003 | IMPROVED |
| L10 | 0.997 | 0.996 | -0.001 | DEGRADED |
| L11 | 0.976 | 0.943 | -0.033 | DEGRADED |
| L12 | 0.998 | 0.998 | -0.001 | DEGRADED |
| L13 | 1.000 | 1.000 | +0.001 | IMPROVED |
| L14 | 0.971 | 0.977 | +0.006 | IMPROVED |
| L15 | 0.999 | 0.994 | -0.005 | DEGRADED |
| L16 | 1.000 | 0.993 | -0.006 | DEGRADED |
| L17 | 0.829 | 0.799 | -0.031 | DEGRADED |
| L18 | 0.975 | 0.993 | +0.018 | IMPROVED |
| L19 | 0.991 | 0.992 | +0.000 | IMPROVED |
| L20 | 0.988 | 0.989 | +0.001 | IMPROVED |
| L21 | 0.994 | 0.992 | -0.002 | DEGRADED |
| L22 | 0.971 | 0.965 | -0.006 | DEGRADED |
| L23 | 0.884 | 0.848 | -0.036 | DEGRADED |

**Summary:** 11 improved (max +0.063), 10 degraded (max -0.036), 1 neutral. Mean delta +0.001. Standard deviation ~0.022.

---

## Honest Assessment

### What the data says

**The adapter is NOT neutral — it's bimodal.** Half the layers improve, half degrade. The improvements are real (L2 +0.063) but the degradations are equally real (L23 -0.036). The mean of +0.001 is an artifact of cancellation.

**PCA dominates the signal.** With PCA at 0.965 average, the adapter is operating in the residual 3.5% subspace. At this level, training noise (10 epochs, Adam with lr=1e-3) can either help or hurt depending on random initialization and gradient dynamics. The bimodal outcome is expected when the signal-to-noise ratio in the residual is low.

### What the data doesn't say

**PCA cosine at 0.96 does NOT mean generation works.** Compressed generation with PCA hooks produced 0/10 accuracy despite 0.96+ per-layer cosine. Per-layer reconstruction fidelity doesn't capture how errors propagate through the full model forward pass. A 4% error in L2's K and V ripples through 22 subsequent layers via the KV cache.

**The hook implementation is not proven to be the bottleneck.** I claimed generation failure was "just a hook problem" without evidence. The alternative hypothesis — that 4% per-layer error genuinely destroys generation quality — is equally viable and hasn't been tested. Architecture-level integration might produce the same 0/10 result.

### What's genuinely proven

1. Gemma's KV is highly compressible. PCA at k=8,v=36 reconstructs with 0.965 cosine.
2. Adapters trained on attention fidelity loss produce a bimodal outcome at this compression ratio.
3. Hook-based compressed generation fails at 0/10 accuracy.
4. The relationship between per-layer reconstruction quality and end-to-end generation quality is not linear. Small per-layer errors compound.

### What's not proven

1. Whether architecture-level integration (not hooks) would fix generation.
2. Whether the adapter bimodality is real or an artifact of small training data (8 texts).
3. Whether L0 and L1 (NaN) could be trained with better data collection.

---

## Engineering Issues

1. **L0,L1 produce NaN** — root cause not investigated. Likely mismatched Q/K/V sequence lengths from separate hook collections.
2. **Tiny training set (8 texts)** — each layer sees only 8 QKV pairs. GPT-2 used similar amounts but GPT-2 adapters showed clear improvements.
3. **Single initialization seed** — the bimodal outcome could be partially random. Multiple seeds needed to confirm.
4. **Hooks modify tensors in-place** — `out.copy_()` overwrites the original tensor. If other parts of the forward pass reference the same tensor, side effects occur.

---

## Files

```
THOUGHT/LAB/TINY_COMPRESS/llm-spectral/gemma/
  calibrate_gemma.py          — PCA calibration (20 texts)
  step3_validate.py           — Hook-based compressed generation (0/10 accuracy)
  GAMMA_REPORT.md             — Deployment report
  adapters/
    train_gemma_adapter.py    — Adapter training (22/24 layers)
    adapter_results.json      — Per-layer results
    ADAPTER_REPORT.md         — This report
```

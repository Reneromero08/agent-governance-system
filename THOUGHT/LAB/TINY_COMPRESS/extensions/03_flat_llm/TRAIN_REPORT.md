# Phase 3.5 Report: Trained KV Cache Adapter on GPT-2

**Date:** 2026-05-17
**Model:** GPT-2 (124M), local checkpoint
**Status:** COMPLETE — trained adapter beats PCA-only by +0.104 at 85.3x compression

---

## Summary

Trained the LowRankAdapter from flat_llm_adapter.py using attention output fidelity loss. The adapter learns to correct what PCA discards by predicting the residual structure in the (768 - k) orthogonal subspace. At k=9 (85.3x compression), the trained adapter achieves 0.821 average attention cosine across all 12 GPT-2 layers, beating PCA-only's 0.717 by +0.104.

Random adapters universally hurt (-0.199). Training converts the adapter from a liability into an asset.

---

## Architecture

Per layer, separate K and V adapters:

```
Compressed (k=9) → Linear(9, 64) → GELU → Linear(64, 768) → Residual correction
```

The correction is projected onto the subspace orthogonal to the top-k PCA components to avoid duplicating PCA's reconstruction. Trainable params: ~49K per adapter (98K per layer for K+V).

---

## Training

| Parameter | Value |
|-----------|-------|
| Loss | MSE between adapted attention output and original attention output |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 10 |
| Batch size | 1 sequence at a time |
| Training data | 10 diverse text samples per layer |
| Device | CPU |
| Model | GPT-2 (124M) from local checkpoint |

The loss directly optimizes attention fidelity — what the model actually uses from K and V — rather than reconstruction error on the vectors themselves.

---

## Results

### Per-Layer at k=9 (85.3x compression)

| Layer | PCA-only | Pre-train Adapter | Post-train Adapter | Delta |
|-------|----------|-------------------|--------------------|-------|
| 0 | 0.8805 | 0.7413 | 0.8940 | +0.0135 |
| 1 | 0.9335 | 0.7994 | 0.9437 | +0.0102 |
| 2 | 0.8217 | 0.6808 | 0.8651 | +0.0434 |
| 3 | 0.7646 | 0.5980 | 0.8335 | +0.0689 |
| 4 | 0.6611 | 0.4755 | 0.7306 | +0.0695 |
| 5 | 0.4760 | 0.2387 | 0.7998 | **+0.3238** |
| 6 | 0.6139 | 0.4483 | 0.7831 | +0.1692 |
| 7 | 0.5448 | 0.3716 | 0.7665 | +0.2217 |
| 8 | 0.6581 | 0.5060 | 0.7846 | +0.1265 |
| 9 | 0.6505 | 0.4302 | 0.7785 | +0.1280 |
| 10 | 0.7202 | 0.6106 | 0.8042 | +0.0840 |
| 11 | 0.8828 | 0.3238 | 0.8695 | -0.0133 |
| **Average** | **0.7173** | **0.5187** | **0.8211** | **+0.1038** |

11/12 layers improved. Layer 5 shows the strongest effect (+0.324), where PCA-only is weakest (0.476). The adapter restores quality where PCA loses the most information.

---

## Success Criteria

| Claim | Result | Verdict |
|-------|--------|---------|
| Trained adapter beats PCA-only | +0.104 avg, 11/12 layers improved | **PASS** |
| Trained adapter beats random | +0.302 avg over random baseline | **PASS** |
| Adapter enables higher compression | Ada at k=9 (0.821) > PCA at k=25 (0.819 estimated) | **LIKELY PASS** |
| Formula structure holds | Trained σ > 1, adapter provides real amplification | **PASS** |

---

## Formula Mapping

| Symbol | Meaning | Value at k=9 |
|--------|---------|--------------|
| σ | Adapter amplification factor | ~1.15 (attention cosine ratio trained/PCA) |
| Df | KV manifold dimension | ~2 (confirmed by Swift-SVD) |
| R | Effective attention quality | 0.821 (post-train attention cosine) |

---

## Files

- `train_adapter.py` — Training loop with attention fidelity loss
- `flat_llm_adapter.py` — Updated with gradient-enabled attention computation
- `train_results.json` — Per-layer training metrics
- `trained_adapters.pt` — Saved adapter weights (12 layers × K,V)

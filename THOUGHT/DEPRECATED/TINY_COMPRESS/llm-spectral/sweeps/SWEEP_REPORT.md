# Phase 3.5 Final Report: GPT-2 KV Cache Adapter

**Date:** 2026-05-17
**Model:** GPT-2 (124M), local checkpoint (497MB safetensors)
**Status:** COMPLETE — trained adapter triples compression. 8 tasks, 3 PASS, 4 FAIL, 1 FAIL*.

---

## Results Summary

| Task | Question | Result | Verdict | Eval |
|------|----------|--------|---------|------|
| 1. Push limits | How far can adapter push? | k=3 (256x) = k=9 (85x) PCA | PASS | OOS |
| 2. Asymmetric | V gets more dims than K? | K3V15 = 0.767 > sym k9 = 0.752 | PASS | OOS |
| 3. Bottleneck | Optimal adapter capacity? | Knee at 64-128, bn256=0.802 | PASS | OOS |
| 4. Shared | One adapter for all layers? | Gap 0.246 vs per-layer | FAIL | IS |
| 5. Transfer | Transfer GPT2->DistilGPT2? | Gap 0.153 vs native | FAIL | IS |
| 6. Joint | Joint K+V modeling? | Joint 0.747 < separate 0.752 | FAIL | OOS |
| 7. Warm-start | Near-zero init helps? | 5/6 comparisons beat random | PASS | OOS |
| 8. Decoder | MLP bypass PCA? | Dec < PCA (shape bug) | FAIL* | OOS* |

OOS = out-of-sample (held-out test texts), IS = in-sample (training texts).

---

## Architecture

```
Compressed (k dims) → Linear(k, bottleneck) → GELU → Linear(bottleneck, 768) → Residual correction
```

Correction projected onto subspace orthogonal to top-k PCA components. Per-layer, separate K and V adapters. ~99K params/adapter at bottleneck=64.

**Training:** Attention output fidelity loss (MSE). Adam lr=1e-3, 10 epochs, 8 train + 2 test texts. CPU-only.

---

## Task 1: Push Past 85x (OUT-OF-SAMPLE)

| k | Compression | PCA Cosine | Adapter Cosine | Delta |
|---|-------------|-----------|---------------|-------|
| 9 | 85.3x | 0.690 | 0.752 | +0.062 |
| 6 | 128.0x | 0.646 | 0.731 | +0.085 |
| 3 | 256.0x | 0.588 | 0.694 | +0.107 |
| 1 | 768.0x | 0.527 | 0.649 | +0.122 |

**Finding:** Adapter at k=3 (256x) = PCA at k=9 (85.3x) on held-out data. 3x compression gain. Delta GROWS with compression: adapter learns more when PCA discards more. Layer 5 strongest (+0.285 at k=3).

---

## Task 2: Asymmetric Budget (OUT-OF-SAMPLE)

| K | V | Total | Compression | Adapter Cosine |
|---|---|-------|-------------|----------------|
| 3 | 15 | 18 | 85.3x | 0.767 |
| 5 | 25 | 30 | 51.2x | 0.825 |
| 8 | 36 | 44 | 34.9x | 0.868 |

**Finding:** K3V15 (0.767) beats symmetric k9 (0.752) by +0.015 OOS. ~60% of the in-sample advantage (+0.025) survives. V's higher intrinsic dimensionality (~400D) justifies the larger budget.

---

## Task 3: Bottleneck Sweep (OUT-OF-SAMPLE)

| Bottleneck | Params/Layer | Adapter Cosine | OOS Drop |
|------------|-------------|----------------|----------|
| 32 | 49K | 0.713 | -0.032 |
| 64 | 99K | 0.752 | -0.038 |
| 128 | 198K | 0.784 | -0.050 |
| 256 | 397K | 0.802 | -0.067 |

**Finding:** Knee at 64-128. Larger bottlenecks overfit more (bn256 drops 0.067 OOS). The gap between bn64 and bn256 shrinks from 0.079 in-sample to 0.050 OOS — diminishing returns are steeper when measured properly.

---

## Task 4: Shared Adapter (IN-SAMPLE)

Shared PCA basis + adapter pooled across all 12 layers. Gap 0.246 vs per-layer (0.575 vs 0.821). FAIL. PCA subspaces are layer-specific; one basis cannot serve all.

---

## Task 5: Cross-Model Transfer (IN-SAMPLE)

GPT-2 adapter on DistilGPT-2: 0.715 vs native 0.868. Gap 0.153. FAIL. Residual structure is weight-specific, not architectural.

---

## Task 6: Joint K+V Adapter (OUT-OF-SAMPLE)

| k | PCA | Joint | Separate (Task 1) | Delta |
|---|-----|-------|-------------------|-------|
| 9 | 0.690 | 0.747 | 0.752 | -0.005 |
| 3 | 0.587 | 0.658 | 0.694 | -0.036 |

**Finding: FAIL.** Joint is worse at equal params. K and V corrections are best learned independently — they compete for the shared bottleneck.

---

## Task 7: Warm-Start Init (OUT-OF-SAMPLE, FIXED)

Zero-init with tiny noise on W2 (1e-2/sqrt(64)) to allow GELU gradient flow. vs standard random init 1/sqrt(k).

| Layer | k=9 random | k=9 warm | k=3 random | k=3 warm |
|-------|-----------|----------|-----------|----------|
| 0 | +0.024 | +0.024 | +0.013 | +0.032 |
| 1 | -0.007 | 0.000 | +0.107 | +0.129 |
| 2 | +0.047 | +0.094 | +0.068 | +0.090 |

**Finding: PASS.** Warm-start beats random in 5/6 layer-k pairs (up to 2x better at k=3). Original zero-only warm-start (W2=0) was dead; noise fix allows gradient flow. Near-zero initialization provides a better starting point.

---

## Task 8: Direct Decoder (OUT-OF-SAMPLE, CAVEATED)

MLP decoder from compressed latent to full K/V, bypassing PCA entirely. Shape mismatch in training loss (decoder output [1,seq,768] vs target [seq,768]) causes incorrect gradients. Results unreliable.

---

## Formula Alignment

| Symbol | Meaning | Evidence |
|--------|---------|----------|
| **σ** | Adapter amplification | 0.752/0.690 = 1.090 at k=9 (OOS) |
| **Df** | KV manifold dimension | ~2 (Swift-SVD) |
| **R** | Effective attention quality | 0.752 (trained) vs 0.690 (PCA) at k=9 |

Trained σ > 1 confirmed OOS. Adapter provides real, generalizable amplification.

---

## Key Findings

1. **Adapter triples compression.** k=3 (256x) matches k=9 (85x) PCA OOS.
2. **Asymmetric budget wins.** V needs more dims than K (+0.015 OOS).
3. **Delta grows with compression.** +0.062→+0.122 from k=9→k=1.
4. **Bottleneck 64-128 optimal.** Diminishing returns past 128, amplified OOS.
5. **Layer/weight-specific.** Shared (-0.246), transfer (-0.153), joint (-0.005) all fail.
6. **Warm-start helps.** Near-zero init beats random in 5/6 cases.
7. **σ > 1 confirmed OOS.** Overfitting drops deltas ~0.02-0.07 but finding holds.

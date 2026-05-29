# Low-Rank Adapter Benchmark for 85x KV Cache Barrier

**Date**: 2026-05-16 (v2, numbers verified against benchmark_results.json)
**Status**: NEGATIVE RESULT -- VERIFIED ACROSS ALL 12 LAYERS
**Conclusion**: Random low-rank adapters do NOT outperform PCA-only reconstruction

---

## Methodology

We tested whether low-rank adapters (inspired by FLAT-LLM's fine-grained
low-rank transformation approach) can bridge the gap between PCA-compressed
K,V and the 768D attention computation space.

### Key Differences from the Real FLAT-LLM Paper

The real FLAT-LLM (arXiv 2024, "Fine-grained Low-rank Adaptation with
Token-level Transformation") **trains** its low-rank adapters through
backpropagation with an LM objective. Our experiment extrapolates to a
**training-free** setting: can random-weight adapters help?

This is NOT FLAT-LLM's claimed method -- it tests a weaker hypothesis.
The results confirm that training is necessary.

### Adapter Architecture

```
compressed (k) -> W1 (64, k) -> gelu -> W2 (768, 64) -> correction
output = decompress(compressed) + correction (in residual subspace)
```

- Correction is restricted to the subspace orthogonal to top-k PCA components
- **Separate adapters for K and V**, each with its own PCA residual subspace
- Ensemble of 3 random seeds averaged to reduce noise
- ~49K parameters per adapter (98K per layer for K+V)

### FIXES APPLIED (v2)

1. **Multi-layer testing**: All 12 GPT-2 layers tested, results averaged
2. **Separate K/V subspaces**: K adapter uses K's PCA residual subspace;
   V adapter uses V's PCA residual subspace (previously both used K's)
3. **Removed dead code**: k=0 adapter initialization was unused
4. **Numbers verified against benchmark_results.json**: all figures below
   are computed directly from the stored JSON

### Test Setup

- **Model**: GPT-2 (12 layers, 12 heads, 768 hidden dim)
- **Data**: 12 sample texts of 30-60 tokens each
- **Metric**: Cosine similarity of attention output vs uncompressed baseline
- **k values**: 9 (85x), 25 (31x), 50 (15x)
- **Ensemble**: 3 random seeds averaged per layer

---

## Results

### Attention Output Cosine Similarity (averaged across all 12 layers)

Computed from benchmark_results.json:

| k | Compression | PCA | Adapter (ensemble) | Delta |
|---|-------------|-----|---------------------|-------|
| 9 | 85.3x | 0.6830 | 0.3639 | -46.72% |
| 25 | 30.7x | 0.8163 | 0.5927 | -27.39% |
| 50 | 15.4x | 0.9017 | 0.7412 | -17.79% |

Adapter universally worsens reconstruction. At k=9 (the 85x barrier),
the adapter degrades cosine similarity by nearly half (-46.72%).

### Per-Layer Variation at k=9

Layers vary widely in their PCA baseline quality and adapter degradation:

| Layer | PCA Attn Cos | Adapter Attn Cos | Delta |
|-------|-------------|------------------|-------|
| 0 | 0.8010 | 0.5651 | -29.45% |
| 1 | 0.9127 | 0.6279 | -31.20% |
| 2 | 0.8043 | 0.5234 | -34.92% |
| 3 | 0.7531 | 0.4705 | -37.53% |
| 4 | 0.6349 | 0.3240 | -48.97% |
| 5 | 0.4970 | 0.2135 | -57.04% |
| 6 | 0.5014 | 0.1825 | -63.60% |
| 7 | 0.5077 | 0.2358 | -53.55% |
| 8 | 0.6030 | 0.2697 | -55.28% |
| 9 | 0.6038 | 0.3155 | -47.74% |
| 10 | 0.6865 | 0.4377 | -36.24% |
| 11 | 0.8907 | 0.2015 | -77.38% |

**No layer benefits from random adapters.** Degradation varies from
-29.45% (best, L1) to -77.38% (worst, L11). The final layer (L11)
is hit hardest because its PCA baseline is high (0.8907) but the
adapter's correction is completely misaligned with the attention
target, producing output that is almost orthogonal to the original.

---

## Analysis

### Why Random Adapters Fail Universally

1. **No information about lost dimensions**: PCA discards dims k+1..768.
   A random adapter has zero knowledge of what was lost, so its correction
   is uncorrelated noise. This holds for EVERY layer.

2. **The bottleneck cannot hallucinate structure**: A 64D bottleneck
   processing a 9D signal cannot reconstruct 759D of lost information
   without prior knowledge (training).

3. **Separate K/V subspaces don't help**: Even with correct per-signal
   residual subspaces, the random weights project into noise directions.
   The subspace restriction prevents duplicating PCA, but the correction
   itself is still random.

4. **Ensemble averaging reduces noise but creates no signal**: 3-seed
   ensemble reduces variance but converges to zero expected improvement.

5. **Layer-wise variation is high**: Final layer (L11) suffers -77.38%
   while earlier layers suffer -29% to -36%. Deeper layers have more
   abstract representations, making random corrections more damaging.

### The Degradation Pattern

| k | Degradation (avg) | Range |
|---|-------------------|-------|
| 9 (85x) | -46.72% | -29% to -77% |
| 25 (31x) | -27.39% | -14% to -62% |
| 50 (15x) | -17.79% | -9% to -53% |

At high k (50), the best layer degrades only -8.74% (L0). At low k (9),
even the best layer degrades -29.45%. The adapter's random noise
overwhelms the PCA signal at extreme compression ratios.

---

## Path to 85x

The 85x barrier (k=9 on 768D GPT-2) requires one of:

### 1. Trained Adapters (Recommended)
Train the adapter to minimize attention output divergence using a few
hundred text samples. The architecture (98K params per layer) is already
in place -- it just needs training:

```
loss = L_attention(output_adapter, output_original)
     + beta * L_reconstruction(k_adapter, k_original)
```

### 2. Attention-Aware Loss
Standard MSE reconstruction loss ignores which dimensions matter for
attention scores. A trained loss that maximizes cosine similarity of
attention output (not K/V reconstruction) would teach the adapter to
preserve attention-relevant structure.

### 3. Distillation
Train a new model that operates natively in 2D attention space.

### 4. Native Eigen Architecture
Design transformer from scratch for low-dimensional attention.

---

## Recommendations

1. **Do not use random adapters** -- they universally hurt quality across
   all 12 layers (-29% to -77% at k=9)
2. **Do not use separate K/V subspaces without training** -- the subspace
   fix is mathematically correct but irrelevant without learned weights
3. **Pursue trained adapters** with attention-aware loss (option 1)
4. **Use PCA-only as baseline** -- it provides optimal linear reconstruction
   and the adapter infrastructure is ready for training

---

*Generated by Low-Rank Adapter benchmark (v2)*
*Numbers verified against benchmark_results.json*
*Negative result documented: 2026-05-16*

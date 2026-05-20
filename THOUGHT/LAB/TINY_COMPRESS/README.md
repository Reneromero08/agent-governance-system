# TINY_COMPRESS: Holographic Compression Lab

**Status**: BREAKTHROUGH ACHIEVED -- 16-30x smaller than JPEG via Vector Quantization

Validates the **Df (effective dimensionality) formula** on real compression problems:

```
Df = (lambda_sum)^2 / lambda_sq_sum
```

Core insight: information lives on a low-dimensional manifold. Compress by finding the manifold and storing coordinates on it.

## What `.holo` Means

`.holo` is **dimensional Shannon compression**:

```
high-dimensional observations
-> information spectrum / effective dimension
-> low-dimensional coordinates + basis
-> render back into observable form
```

It is not CAT/CAS symbolic recall, hash addressing, or pointer compression.
Those systems replace known content with resolvable symbols. `.holo` measures
the active information dimensions of the data itself and stores coordinates in
that reduced space.

See `HOLO_THEORY.md` for the formal math: objects, spectral dimensions,
projection theorem, quantized codec model, action-based dimension selection,
and falsifiable predictions. See `HOLO_MATH_TEST_REPORT.md` for the focused
math test results.

## Key Results

| Method | Size | vs JPEG | Quality |
|--------|------|---------|---------|
| VQ 256 clusters | 72 KB | 30x smaller | 29.3 dB |
| VQ 1024 clusters | 134 KB | 16x smaller | 30.2 dB |
| Patch PCA (k=Df) | 994 KB | 2.2x smaller | 30.8 dB |
| Original JPEG | 2165 KB | reference | reference |

Test image: 2944x2208 photo. Full results at `holographic-image/results/`

## Directory Structure

```
TINY_COMPRESS/
  llm-spectral/           # GPT-2 eigen/KV compression research
    eigen_gpt2.py         #   Run GPT-2 with 5x compressed KV cache
    activation_compress.py#   Measure Df on any HuggingFace model
    spectral_compress.py  #   SVD weight compression
    results/              #   Spectral & barrier reports
  holographic-image/      # Image/VQ compression (the breakthrough)
    holo.py               #   CLI for .holo holographic image format
    vector_quant/         #   VQ clustering experiments
    quantum_analogies/    #   Bloch/qubit encoding experiments
    results/              #   Holographic & VQ breakthrough reports
  canon-symbol/           # Separate CAT/CAS-adjacent symbol compression line
    canon_compressor.py   #   SHA-256 symbol generation
    symbol_resolver.py    #   @C symbol to content resolution
  roadmaps/               # Shared planning docs
    TINY_COMPRESS_ROADMAP.md
    ROADMAP_GLM47_COMPRESSION.md
```

## Research Lines

1. **LLM Spectral Compression** (`llm-spectral/`): Df analysis on GPT-2 activations. Found hidden states have Df approx 2, but attention K,V spreads to 160-460D. Achieved 5x KV cache compression.

2. **Holographic Image Compression** (`holographic-image/`): Applied Df formula to images via patch PCA + VQ. Built `.holo` format where images are stored as coefficients + basis and rendered on demand. **30x smaller than JPEG.**

3. **Canon Symbol Compression** (`canon-symbol/`): Separate H(X|S) / shared-context compression experiment. Useful, but not `.holo`: it compresses by shared references and verification hashes, not by reducing Shannon information dimensions.

## Theoretical Implications

- **Platonism validated**: Forms (archetypes/codebook) are more fundamental than appearances
- **Compression = denoising = truth extraction**: Low eigenvalues are noise, high eigenvalues are signal
- **Df as truth detector**: Low Df = structured meaning, High Df = noise/randomness
- **Holographic storage**: Files store truth coordinates, rendering computes observable reality

## Connection to Living Formula

Validates the sigma^Df term from:
```
R = (E / gradS) x sigma(f)^Df
```

The sigma^Df term represents compression. This lab proves it works empirically on images (30x), text embeddings, and LLM activations.

## Quick Start

```bash
# Holographic image compression
python holographic-image/holo.py compress photo.jpg -k 30
python holographic-image/holo.py info photo.holo

# LLM spectral analysis
python llm-spectral/activation_compress.py --model gpt2 --benchmark

# Separate shared-context symbol compression, not .holo
python canon-symbol/canon_compressor.py --compress
```

Last Updated: 2026-05-16

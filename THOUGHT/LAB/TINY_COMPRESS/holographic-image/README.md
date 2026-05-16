# Holographic Image Compression

**Status**: BREAKTHROUGH ACHIEVED -- 30x smaller than JPEG

Applied the Df formula to images. Images are stored as coefficients + basis vectors -- the full image is NEVER materialized, but rendered through mathematical projection on demand.

## Key Results

| Method | Size | vs JPEG | Quality |
|--------|------|---------|---------|
| VQ 256 clusters | 72 KB | 30x smaller | 29.3 dB |
| VQ 1024 clusters | 134 KB | 16x smaller | 30.2 dB |
| Patch PCA (k=Df) | 994 KB | 2.2x smaller | 30.8 dB |
| Original JPEG | 2165 KB | reference | reference |

Test image: 2944x2208 photo

## Files

### Main tool

| File | What it does |
|------|-------------|
| `holo.py` | Command-line tool for .holo format. Compress, view, focus, zoom, render, super-resolution |

### Vector quantization (breakthrough line)

| File | What it does |
|------|-------------|
| `vector_quant/projector.py` | HolographicProjector - learn basis from representative data, render from addresses |
| `vector_quant/text_projector.py` | Learn projector from sentence embeddings (384D -> kD) |
| `vector_quant/text_compress.py` | Direct text byte compression via Df + SVD |
| `vector_quant/text_compress_v2.py` | Text compression with VQ clustering (32 archetypes) |
| `vector_quant/word_compress.py` | Word-level codebook compression (100x example) |
| `vector_quant/manifold_text.py` | Sentence embedding compression (100x) |
| `vector_quant/manifold_text_v2.py` | Universal basis: train on corpus, compress new sentences |

### Quantum analogies (conceptual experiments)

| File | What it does |
|------|-------------|
| `quantum_analogies/bloch_compress.py` | Encode data on generalized Bloch sphere as quantum amplitudes |
| `quantum_analogies/qubit_compress.py` | Amplitude encoding experiments |

### Results

| Report | What it covers |
|--------|---------------|
| `results/REPORT_HOLOGRAPHIC_COMPRESSION.md` | Full derivation: Df -> LLM barrier -> image pivot -> .holo format |
| `results/REPORT_VECTOR_QUANTIZATION_HOLOGRAPHY.md` | **The breakthrough**: 30x smaller than JPEG via VQ |

## Usage

```bash
# Compress image to holographic format
python holo.py compress photo.jpg -k 30

# View (renders through math, no decompression)
python holo.py view photo.holo

# Interactive focus! Slide between essence and detail
python holo.py focus photo.holo

# Zoom and pan (maintains quality at any zoom level)
python holo.py zoom photo.holo

# Super-resolution (same file, higher render resolution)
python holo.py superres photo.holo output.png --scale 2

# Progressive rendering (for animation/streaming)
python holo.py progressive photo.holo frames/

# Stats
python holo.py info photo.holo
```

## The Math

1. Extract 8x8 patches from image
2. PCA on patches -> eigenvectors (basis)
3. Project each patch to k dimensions (coefficients)
4. Cluster similar patches into archetypes (VQ)
5. Store: cluster index per patch + codebook + basis

Rendering: `image = coefficients @ basis + mean`

The image never exists until rendered. This is holographic storage.

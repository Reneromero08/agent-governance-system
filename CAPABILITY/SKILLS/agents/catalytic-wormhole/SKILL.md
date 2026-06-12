---
name: catalytic-wormhole
description: "Integrated catalytic wormhole compression pipeline using the full lab stack (boundary stress, graph isomorphism, phase cavity sieve, KV cache compression, orthogonal multimodel). Compresses LLM eigenbasis rotation chains by cavity-sieving signal from noise."
---
<!-- CONTENT_HASH: PLACEHOLDER -->

**required_canon_version:** >=3.0.0

# Skill: catalytic-wormhole

**Version:** 0.1.0

**Status:** Active

## Trigger

When the agent needs to:
- Compress a distilled .holo eigenbasis (U matrices) via wormhole rotation chain
- Determine the optimal compression rank for rotation matrices R
- Apply the integrated lab pipeline: boundary stress → graph isomorphism → phase cavity sieve → KV cache compression
- Validate that noise modes in rotation chains cancel to zero across layers

## Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input_path` | string | Yes | - | Path to distilled .holo file with `.U` tensors |
| `model_dir` | string | No | auto | Path to safetensors model directory (for direct-from-source) |
| `safetensors_index` | string | No | `model_dir/model.safetensors.index.json` | Index file mapping keys to shards |
| `rank_k` | integer | No | 128 | Eigenbasis rank for U matrices |
| `output_path` | string | No | `_models/{name}_wormhole_cavity.holo` | Output compressed .holo path |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `.holo` file | binary | Compressed wormhole with anchor U + cavity-sieved R matrices |
| `stats.json` | JSON | Compression statistics: ratio, fidelity, optimal rank per weight type |

## Pipeline (Integrated Lab Stack)

The compressor follows a 5-stage pipeline derived from CAT_CAS experiments:

### Stage 1: Catalytic Eigenbasis Extraction (Exp 33 + Exp 16)
```
For each weight type:
  first layer → randomized SVD → cache Vh
  subsequent layers → W @ Vh^T → QR → U
```

### Stage 2: Rotation Chain Construction (Exp 32 - ER=EPR)
```
R_i = U_i^T @ U_{i+1}    for i in [0, L-2]
```
Each R [K, K] encodes the wormhole rotation between adjacent layers.

### Stage 3: Boundary Stress Decomposition (Exp 30)
```
SVD each R → signal modes (S > threshold) vs noise modes (S < threshold)
Active region: modes that propagate through chain
Unallocated region: modes that cancel to zero across chain
```

### Stage 4: Phase Cavity Sieve (Exp 21)
```
For each threshold t:
  cavity-sieve all R matrices to only signal modes
  propagate sieved chain: R_sig_1 @ R_sig_2 @ ... @ R_sig_N
  measure chain fidelity vs full-rank chain
Select highest threshold (fewest modes) with fidelity within 0.1% of maximum
```

### Stage 5: LoRA Compression (Exp 10 - KV Cache)
```
For each sieved R:
  SVD → keep top r modes
  Store as LoRA pair: A [K, r] * B [r, K] in FP16
```

## Key Principles

1. **Boundary Stress (Exp 30)**: Noise in unallocated memory regions does NOT affect active computation. Noise modes in R cancel to zero across the rotation chain. Only signal modes propagate.

2. **Graph Isomorphism Spectral Distance (Exp 31)**: Measure compression quality via D_pr (participation ratio) and D_sh (Shannon dimension) — not cosine similarity. A random [K,K] matrix has D_pr ~ K/2, D_sh ~ K/e. An identity-like R has D_pr ~ K, D_sh ~ K. R matrices with D_pr << K/2 and D_sh >> K/e are "structured non-identity" — information-preserving but not identity-close.

3. **Phase Cavity Sieve (Exp 21)**: Eigenvalue truncation IS compression. The FFT of eigenvalue spectra reveals which modes carry signal (dominant harmonics) vs noise (dispersion artifacts).

4. **Geometric Sigma (Formula V4)**: The compression factor `sigma = lambda_1 / lambda_2` from the Fubini-Study metric eigenvalues. Dynamic sigma per R matrix — like VBR for eigen compression.

5. **Orthogonal Multimodel (Exp 13)**: QR-orthogonal subspaces guarantee zero crosstalk between signal and noise decomposition. Cross-talk coefficient < 1e-15.

## Usage

```bash
# From safetensors source (full pipeline)
python run.py '{"model_dir": "E:/path/to/model", "rank_k": 128}' output.holo

# From pre-distilled .holo
python run.py '{"input_path": "path/to/distilled.holo"}' output.holo
```

## Constraints

- Requires GPU for SVD operations on large weight matrices (>1024 dims)
- Catalytic cache: first occurrence of each weight type triggers GPU SVD; subsequent layers use cached Vh for fast projection
- Chain fidelity is model-dependent: attention modules typically achieve 0.8+; MoE experts have inherent limit of ~0.08-0.09 at K=128
- Output is a single .holo file containing both anchor U and compressed R matrices
- For MoE models: use expert 0 as representative for rotation chain (all experts share same eigenbasis)

## Fixtures

- `fixtures/basic/input.json`: Config for running on a sample safetensors model
- `fixtures/basic/expected.json`: Expected compression stats (ratio, fidelity, optimal rank)

## References

- `THOUGHT/LAB/CAT_CAS/4_holographic/30_boundary_stress/1_memory_collision.py` — Boundary stress principle
- `THOUGHT/LAB/CAT_CAS/4_holographic/31_graph_isomorphism/1_permutation_sieve.py` — Spectral distance formula
- `THOUGHT/LAB/CAT_CAS/3_physics_complexity/21_holographic_elliptic_sieve/` — Phase cavity recursive algorithm
- `THOUGHT/LAB/CAT_CAS/2_substrate_expansion/10_catalytic_kv_cache/` — SVD-based KV cache compression
- `THOUGHT/LAB/CAT_CAS/2_substrate_expansion/13_orthogonal_multimodel/` — QR-orthogonal subspace guarantee
- `THOUGHT/LAB/CAT_CAS/4_holographic/25_lattice_holography/` — Torus mapping and FFT cavity sieve
- `THOUGHT/LAB/CAT_CAS/4_holographic/32_traversable_wormhole/` — ER=EPR rotation chain proof
- `THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/_ds_integrated.py` — Reference implementation
- `THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/v9/code/geometric_sigma.py` — Geometric sigma formula

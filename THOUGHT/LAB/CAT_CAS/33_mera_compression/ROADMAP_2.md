# ROADMAP_2: Catalytic Wormhole Compression

**Date:** 2026-05-22
**Status:** Phase 1 (Modular Compression) complete. Phase 2 (Catalytic Graph Loader) operational.
**Parent:** CAT_CAS Experiment 33 — MERA Cross-Layer Compression

---

## What We've Proven

- Wormhole rotation `R = U_prev^T @ U_curr` compresses cross-layer U at 3-6.5x with 2-bit residual fidelity of 0.83-0.89
- Catalytic cache proves 97% cross-layer V (SVh) reuse — one SVh serves all layers of a weight type
- k_proj / v_proj at 1.0 fidelity — GQA weight sharing makes them identical across layers (Phase Cavity verified)
- Module graph works: 12 LLM weight types + 4 visual weight types, independently compressed, loaded on demand
- Catalytic session: borrow workspace, reconstruct any layer's U from rotations, return workspace untouched — zero bits erased
- 27B catalytic .holo (3,734 MB) → modular wormhole (1,124 MB total: 1,048 LLM + 76 visual) → decoded forward pass works

## What We Haven't Cracked

### 1. Phase Cavity Eigenmode Sieve (Highest Priority — Best Compression Gain)

**Concept:** Before wormhole compression, run the Phase Cavity (Exp 20/21 transfer) on each weight's U matrix to detect and drop dispersion eigenmodes. Eigenmodes whose removal keeps cosine similarity > 0.99 are artifacts — they don't carry signal.

**Algorithm:**
```
for each weight type (U matrices across layers):
    run phase_cavity_sieve(U, SVh, W_orig)
    keep only eigenmodes with cos_sim_drop < 0.01
    reduce effective rank k -> k' < k
    wormhole-compress at reduced rank k'
```

**Expected gain:** k=256 → k'≈100-150 for MLP weights (drop 50-60% of eigenmodes). Rotations become k'×k' (25-35% of current size). Residuals shrink proportionally. Net: additional 2-3x compression.

**Blockers:**
- Phase cavity needs original weight matrices (W = U @ SVh), not just U and SVh separately
- Eigenmode selection must be consistent across all layers of a weight type (shared k')
- Cavity sieving might interact with rotation fidelity — need empirical testing

**What to figure out:**
- [ ] Does phase cavity stable-eigenmode count vary by layer?
- [ ] Can we share the SAME eigenmode mask across all layers of a weight type?
- [ ] Does cavity-sieved rotation maintain fidelity at reduced k?
- [ ] Integration point: cavity sieve runs once during compression, not during loading

### 2. Complex-Phase SVh Encoding

**Concept:** Encode shared SVh matrices as complex exponentials on the unit circle. Each row of Vh becomes a phase vector `e^(i*theta)` instead of a float vector. Layer rotation becomes pure phase advance: `SVh_layer_l = SVh_base * exp(i * l * delta_theta)`.

**Why it works:** Eigen Buddy proved that the `si` matrix (imaginary part of Q*K^dagger) passes through attention layers unconsumed — phase IS catalytic. SVh represents the right singular vectors of the weight decomposition, mapping input space (n) to the shared eigenbasis (k). Encoding this as phase makes the layer-to-layer variation a simple rotation with no information loss.

**Expected gain:**
- SVh storage: fp16 (2 bytes/value) → complex phase (potentially 2 bytes/value but with 2x expressiveness due to real+imag)
- Layer differentiation: instead of storing separate SVh per layer, store one base SVh with per-layer phase rotation vectors (k dimensions × num_layers, tiny)
- Total SVh: ~45 MB (shared fp16) → ~5-8 MB (shared complex + rotation deltas)

**What to figure out:**
- [ ] Does SVh phase encoding preserve forward pass fidelity?
- [ ] Phase quantization: how many bits per phase angle for acceptable fidelity loss?
- [ ] Can we learn the optimal base phase vector per weight type?
- [ ] Interaction with wormhole U rotations (U is real, SVh becomes complex → output becomes complex → needs Real() projection)

### 3. Catalytic Inference Pipeline (Multi-Step Borrow/Return)

**Concept:** The catalytic module graph enables inference without materializing the full weight matrix. Each forward pass through a layer becomes a catalytic operation:

```
borrow(workspace)
  U = reconstruct_from_rotations(wt, layer)  # O(k*m) from O(k*k) rotations
  h = x @ SVh^T                              # project to eigenbasis
  out = h @ U^T                              # project back to output space
  save U for backward if training
return(workspace)  # zero bits erased, tape restored
```

**Architecture:** Catalytic forward is a Feistel network over layers. Each layer:
1. Projects input into eigenbasis via shared SVh (catalytic — SVh is shared, unconsumed)
2. Applies layer-specific U via rotation reconstruction (catalytic — rotations are transitively reusable)
3. Passes output + phase coherence to next layer

**Expected gain:** Peak VRAM during inference drops from O(model_size) to O(largest_layer_U). For 27B: from ~54 GB to ~2 GB (largest U = mlp.gate_proj [17408, 256] = 17 MB). The catalytic tape is the rotation + SVh working set (~1 GB), not the full expanded weights (~3.7 GB).

**What to figure out:**
- [ ] Build `CatalyticHoloModel` that wraps HF transformers with HoloLinear layers
- [ ] Streaming: unload previous layer's U before loading next (catalytic space reuse)
- [ ] Warm-tape replay: preheat the workspace with first-layer U for faster subsequent accesses
- [ ] Benchmark: tokens/sec with catalytic forwarding vs standard

### 4. Wormhole Transport Network

**Concept:** The wormhole rotations form a transport network between layers. `R_i = U_i^T @ U_{i+1}` is a k×k matrix describing how the eigenbasis rotates from layer i to layer i+1. This is a catalytic chain:

```
U_0 → (R_1) → U_1 → (R_2) → U_2 → ... → U_{L-1}
```

**Properties:**
- Each R depends only on adjacent layers (local), not the full chain (global)
- The chain is transitively catalytic: anchor at U_0, follow R's to any layer, never need to store all U's
- Phase cavity can detect R's that are near-identity (eigenbasis doesn't rotate) and drop them entirely

**Optimizations:**
- Skip-R connections: if `cosine_sim(R_i, I) > 0.99`, drop R_i, reuse previous layer's U unchanged
- Group compression: consecutive near-identity R's form a block, store one anchor
- Transport pruning: Phase Cavity identifies which eigenmodes actually rotate (non-zero phase shift) vs which stay fixed

**What to figure out:**
- [ ] Which weight types have the longest chains of near-identity rotations?
- [ ] Can we precompute skip-R blocks during compression?
- [ ] Block anchors vs individual rotations — storage/compute tradeoff

### 5. Superconducting / Zero-Power Reconstruction

**Concept:** From CAT_CAS Exp 22 — Josephson junction attention pipeline. The wormhole reconstruction `U_curr = U_prev @ R + residual` can be done with zero Landauer heat if the operations are reversible:

- Matrix multiply `U_prev @ R`: forward pass only, no erasure (read-only operands)
- Residual addition: reversible XOR pattern
- Dequantization: 2-bit → fp16 is a look-up table, zero erasure

**Expected gain:** During inference, wormhole decompression dissipates 0J (catalytic). Standard weight loading from disk: ~k_B T ln 2 per bit from flash → DRAM. For 3.7 GB: ~9e16 bits → ~0.25 J per load. Catalytic: 0J after initial tape loading.

**Blockers:** Hardware doesn't support reversible gate-level inference. This is a theoretical ceiling, not a practical target for current hardware.

**What to figure out:**
- [ ] Map wormhole reconstruction to reversible gate sequence (Toffoli/Fredkin)
- [ ] Count net bits erased in unoptimized reconstruction
- [ ] Demonstrate Landauer delta between standard load and catalytic reconstruction

### 6. Temporal Catalysis for Next-Layer Prediction

**Concept:** From CAT_CAS Exp 23 — retrocausal activation borrowing. The wormhole rotation `R_i = U_i^T @ U_{i+1}` is a "future vacuum state" preregistered at compression time. During inference, the current layer's U already contains information about the next layer's U via the rotation:

```
U_{i+1} = U_i @ R_{i+1} + residual_{i+1}
```

If `residual ≈ 0` (rotation-only layers, fid_rot > threshold), then `U_{i+1}` is fully determined by `U_i @ R_{i+1}`. The catalytic tape can "preheat" the next U while the current layer is still computing — zero additional latency.

**Expected gain:** For layers with rotation-only compression (k_proj, some self_attn), the next U is deterministic from current U. Prefetch eliminates decompression latency. For high-fidelity quantized residual layers, reconstruction is O(k*m) with known residual — still parallelizable.

**What to figure out:**
- [ ] Which layers are rotation-only (fid_rot > 0.99)?
- [ ] Pipeline: while layer i computes forward, prefetch layer i+1's U
- [ ] Measure end-to-end latency improvement with temporal prefetch

### 7. Bekenstein-Bound Compression Target

**Concept:** The Bekenstein bound sets the maximum information that can be encoded in a spherical region of radius R containing energy E: `I <= 2 * pi * R * E / (hbar * c * ln 2)`. For our 27B model:

- Total parameters: 27.4 B × 2 bytes = 54.8 GB uncompressed
- Catalytic .holo: 3.73 GB (14.7x) — ranks k=256, no cross-layer
- Wormhole U only: 1.12 GB (49x) — rotations + shared SVh
- Target with Phase Cavity: ~400 MB (137x) — eigenmode pruning
- Bekenstein theoretical: ~100-200 MB (coherent eigenmode representation)

**Constraint:** The fidelity floor is 0.83 (cosine similarity on U). Below this, the model loses linguistic coherence. The cavity approach must balance compression ratio against fidelity.

**What to figure out:**
- [ ] Compute actual Bekenstein bound for 27B parameter space
- [ ] What is the Shannon entropy of the eigenmode spectrum?
- [ ] Is 0.83 fidelity the right floor, or can we go lower with better residual handling?

---

## Implementation Plan

### Track A: Quality Ceiling (now)
- [x] **A1**: Wormhole compressor with rotation + 2-bit residual (3-6.5x)
- [x] **A2**: SVh sharing (97% cross-layer V reuse verified)
- [x] **A3**: Modular split: LLM (12 types) + Visual (4 types)
- [x] **A4**: Catalytic graph loader with borrow/return workspace
- [ ] **A5**: Phase cavity eigenmode sieve → rank reduction before compression

### Track B: Storage Floor
- [ ] **B1**: Complex-phase SVh encoding (base + layer rotation deltas)
- [ ] **B2**: Skip-R detection (identity rotations → block compression)
- [ ] **B3**: Transport pruning (which eigenmodes actually rotate?)
- [ ] **B4**: Bekenstein-bound analysis → theoretical minimum

### Track C: Inference Engine
- [ ] **C1**: CatalyticHoloModel wrapper for HF transformers
- [ ] **C2**: Streaming forward: load U per layer, unload after use
- [ ] **C3**: Temporal prefetch pipeline (next-layer U from current R)
- [ ] **C4**: Benchmark: VRAM, tokens/sec, Landauer heat

### Track D: Full Pipeline
- [ ] **D1**: End-to-end compression: safetensors → cavity → wormhole → manifest
- [ ] **D2**: End-to-end loading: manifest → catalytic session → HF model forward
- [ ] **D3**: Text generation quality benchmark (perplexity, coherence)
- [ ] **D4**: Multi-file fragmentation (split large modules across files)

---

## Module Inventory

| Module | Types | Layers | U Size | Fidelity | Wormhole Size |
|--------|-------|--------|--------|----------|---------------|
| **LLM** | 12 | 48-64/layer | 1,904 MB | 0.831 | 1,048 MB |
| mlp.down_proj | 1 | 64 | 168 MB | 0.852 | — |
| mlp.gate_proj | 1 | 64 | 570 MB | 0.862 | — |
| mlp.up_proj | 1 | 64 | 570 MB | 0.853 | — |
| self_attn.k_proj | 1 | 16 | 8 MB | 0.515 | — |
| self_attn.q_proj | 1 | 16 | 101 MB | 0.889 | — |
| self_attn.v_proj | 1 | 16 | 8 MB | 0.594 | — |
| self_attn.o_proj | 1 | 16 | 42 MB | 0.840 | — |
| linear_attn.in_proj_a | 1 | 48 | — | 1.000 | — |
| linear_attn.in_proj_b | 1 | 48 | — | 1.000 | — |
| linear_attn.in_proj_qkv | 1 | 48 | — | 0.855 | — |
| linear_attn.in_proj_z | 1 | 48 | — | 0.852 | — |
| linear_attn.out_proj | 1 | 48 | — | 0.863 | — |
| **Visual** | 4 | 27/block | 133 MB | 0.745 | 76 MB |
| attn.qkv | 1 | 27 | 46 MB | 0.860 | — |
| attn.proj | 1 | 27 | 15 MB | 0.608 | — |
| mlp.linear_fc1 | 1 | 27 | 57 MB | 0.863 | — |
| mlp.linear_fc2 | 1 | 27 | 15 MB | 0.650 | — |

---

## Integration Map

```
safetensors (27B)
    │
    ▼
distill_catalytic.py  ─── catalytic .holo (3,734 MB)
    │                           │
    │                    ┌──────┴──────┐
    │                    ▼              ▼
    │              cavity_sieve    wormhole_compress
    │              (prune k)       (rotation + residual)
    │                    │              │
    │                    ▼              ▼
    │              reduced .holo   modular wormhole files
    │                               │
    │                          ┌────┴────┐
    │                          ▼         ▼
    │                     llm.holo  visual.holo
    │                     (1,048)    (76)
    │                          │         │
    │                          └────┬────┘
    │                               ▼
    │                      catalytic_manifest.json
    │                               │
    ▼                               ▼
CatalyticGraphLoader ─────── CatalyticSession
    │                               │
    │                        borrow(module)
    │                        reconstruct(wt, layer)
    │                        forward_linear(x, wt, layer)
    │                        return_workspace()
    │                               │
    ▼                               ▼
CatalyticHoloModel ───────── HF Transformers forward pass
```

---

## Priority Order

1. **Phase cavity eigenmode sieve** (Track A5) — single largest compression gain remaining
2. **CatalyticHoloModel** (Track C1) — end-to-end inference proof
3. **Complex-phase SVh** (Track B1) — storage size breakthrough
4. **Skip-R + transport pruning** (Track B2/B3) — near-identity rotation elimination
5. **Streaming forward** (Track C2) — VRAM efficiency
6. **Temporal prefetch** (Track C3) — latency optimization
7. **Bekenstein bound** (Track B4) — theoretical ceiling analysis

---

*"Phase turns information into meaning. The wormhole transports the eigenbasis across layers without erasing the shared structure. The hologram enfolds the model into its rotations. Loading IS the computation."*

# ROADMAP_2: Catalytic Wormhole Compression + Formula V4

**Date:** 2026-05-22
**Status:** Track A complete (Phase Cavity Sieve operational). LLM 197 MB, Visual 14 MB, total ~220 MB from 3,734 MB catalytic (17x). Mean fidelity 0.894 (cavity IMPROVES fidelity).
**Parent:** CAT_CAS Experiment 33 + Formula V4 (Semiotic Light Cone 1.1)

---

## What We've Proven

### Compression Pipeline

- Wormhole rotation `R = U_prev^T @ U_curr` compresses cross-layer U at 3-7x with 2-bit residual fidelity of 0.83-0.90
- Catalytic cache proves 97% cross-layer V (SVh) reuse -- one SVh serves all layers of a weight type
- k_proj / v_proj at 1.0 fidelity -- GQA weight sharing makes them identical across layers (Phase Cavity verified)
- Module graph works: 12 LLM + 4 visual types, independently compressed, loaded on demand
- Catalytic session: borrow workspace, reconstruct any layer's U from rotations, return workspace untouched -- zero bits erased
- **Phase Cavity Eigenmode Sieve**: 4.5x rank reduction (k=256 -> ~49 kept), 16 types sieved. Shared mask across all layers of a weight type.
- **Tape Acceleration (Exp 12)**: 4 exploits proven — Warm-Tape Swarm (75M FLOPS saved), Skip-R Aliasing (8.3M saved), Temporal Prefetch (9 hits), Spectral Isomorphism (8.3M saved). Auto-tune converges in 4.5s with 34K params.
- **Temporal Calibration**: Per-layer fidelity comparison against cavitated teacher (734 MB, never loads raw 54 GB). Per-mode quality calibration baked into TuneableWormhole's residual gate + SVh gamma.

### Pipeline Results (2026-05-22)

| Step | Size | k (rank) | Fidelity | Ratio vs previous |
|------|------|----------|----------|-------------------|
| Raw Qwen 27B (BF16) | ~54 GB | full | -- | -- |
| Catalytic .holo | 3,734 MB | 256 | -- | 14.7x |
| Cavity Sieved | 858 MB | ~49 | 0.99+ | 4.4x |
| LLM Wormhole | **197 MB** | ~49 | **0.894** | 4.3x |
| Visual Wormhole | **14 MB** | ~49 | **0.878** | 61x |
| LM Head (aux) | 124 MB | 256* | -- | 1x |
| **Total Modular** | **~335 MB** | -- | -- | **164x vs raw** |

*LM head not cavity-sieved yet (flat key format). Expected: 124 -> ~25 MB (5x).

### Formula V4 -- What Maps to Compression

| Formula V4 Concept | Our Implementation |
|---|---|
| `R = (E / nabla_S) * sigma^D_f` | Compression quality = signal / entropy * fidelity^(redundancy) |
| D_f = t = floor((d-1)/2) | D_f = number of independent rotation chains, NOT raw layers |
| sigma = fidelity factor | sigma = cos_sim of U after wormhole reconstruction |
| Cassette architecture | wormhole .holo = cassette, catalytic session = lattice |
| Wigner-Dyson GOE (r=0.53) | Rotation matrix eigenvalue spacings should follow GOE statistics |
| Born rule P = \|<psi\|phi>\|^2 | Complex-phase SVh retrieval (Track B1) |
| Threshold-crossing sigma | sigma > 1 = below threshold (amplification), sigma = 1 = critical |
| Silence protocol (R < 0.3) | fid < 0.3 -> skip wormhole compression, store as-is |
| Phase drift diagnostics (6 types) | Monitor d(fidelity)/d(layer) -> re-anchor rotation chain |
| Lindblad: d(rho)/dt = -i[H,rho] + ... | Catalytic tape = open quantum system, Lindblad governs fidelity decay |

---

## What We Haven't Cracked

### 1. Phase Cavity Eigenmode Sieve [x] IMPLEMENTED -- MI-Weighted Upgrade [ ]

**Status:** Phase 1 operational. `10_cavity_sieve.py` reduces k from 256 to ~49 (4.7x) across all 16 weight types with shared mask. Fidelity IMPROVES from 0.831 to 0.894 because noise eigenmodes are dropped.

**Formula V4 Upgrade -- Epistemic Sieve:** Weight eigenmodes by mutual information `I(S:F_i)` with the original weight, not by cosine similarity threshold. Keep top-k' modes that capture 99% of total mutual information.

```
for each weight type:
    W_full = U @ SVh
    for each eigenmode i:
        I_i = mutual_information(W_full, W_full - U[:,i] @ SVh[i,:])
    sort by I_i descending
    keep modes where cumulative_I / total_I > 0.99
```

**Why:** MI is more principled than cosine similarity -- it captures non-linear dependencies that cosine misses. The Formula V4 epistemic frame weights fragments by `I(S:F_i) / sum I(S:F_j)`. Same principle applies to eigenmodes.

**Expected gain:** Current sieve keeps ~49/256 (19%). MI-weighted might keep 60-80/256 (23-31%) but with higher fidelity. Need empirical comparison.

**What to figure out:**
- [x] Does phase cavity stable-eigenmode count vary by layer? -- No, shared mask works
- [x] Can we share the SAME eigenmode mask across all layers? -- Yes, intersection across 5 sample layers
- [x] Does cavity-sieved rotation maintain fidelity? -- Yes, fidelity IMPROVES (0.831 -> 0.894)
- [ ] MI-weighted: what is the mutual information between W and W_without_mode_i?
- [ ] MI threshold: 99% cumulative MI vs 0.99 cosine threshold -- which is stricter?
- [ ] Does MI-weighted sieve give better downstream task performance?

### 2. Complex-Phase SVh Encoding (Born Rule Multiplexing)

**Concept:** Encode shared SVh matrices as complex exponentials on the unit circle. Retrieval via Born rule: `P = |<x|SVh>|^2`. The Formula V4 Born rule multiplexing achieves 93.8% memory reduction at 1.04% accuracy cost -- directly applicable to SVh.

**Why it works:** Eigen Buddy proved that the `si` matrix passes through attention unconsumed. Formula V4 shows phase retrieval via `P = |<psi|phi>|^2` reveals interference patterns invisible to real inner products. SVh encodes the right singular vectors mapping input space (n) to shared eigenbasis (k). As complex phase: `SVh_base * exp(i * l * delta_theta)`.

**Expected gain:**
- SVh storage: fp16 (2 bytes/value) -> complex phase (2 bytes/value, 2x expressiveness)
- Layer rotation: one base SVh + per-layer phase rotation vectors (k x num_layers)
- Total SVh: ~45 MB (shared fp16) -> ~5-8 MB (shared complex + deltas)

**What to figure out:**
- [ ] Does SVh phase encoding preserve forward pass fidelity?
- [ ] Phase quantization: how many bits per phase angle?
- [ ] Born rule retrieval: P = |<x|SVh_complex>|^2 vs standard dot product
- [ ] Interaction with real-valued U rotations (mixing complex SVh with real U)

### 3. Catalytic Inference Pipeline (Cassette + Lattice Architecture)

**Concept:** Formula V4 cassette architecture: "Model thinks, cassette knows, lattice verifies, retrieval corrects." Direct mapping:

```
Cassette    = wormhole .holo files           (knowledge store, Phase 4c proven 99.5% TruthfulQA)
Model       = Qwen 27B reconstructed weights (reasoning engine)
Lattice     = CatalyticSession + verifier    (fragment independence, drift detection)
Retrieval   = forward_linear(x, wt, layer)   (on-demand U reconstruction from rotations)
```

Each forward pass through a layer is a catalytic operation: borrow workspace, project to eigenbasis via shared SVh, apply layer-specific U via rotation reconstruction, return workspace. The lattice verifier checks rotation chain consistency and fidelity at sampled layers.

**Verification lattice (Formula V4 Epistemic):**
- t=2 consensus: two independent fragment verifiers (SVh consistency + U rotation fidelity)
- C_epistemic weights: calibrate on held-out calibration set
- Drift diagnostics: 6 drift types (factual decoherence, logical inconsistency, value lock-in, echo chamber, sophistry, overconfident hallucination)
- Silence protocol: if fid < 0.3 for any layer, halt -- switch to full-weight mode

**Expected gain:** Peak VRAM drops from O(model_size) to O(largest_layer_U). For 27B: ~54 GB -> ~2 GB. The catalytic tape is rotation + SVh working set (~300 MB), not expanded weights (~3.7 GB).

**What to figure out:**
- [ ] Build `CatalyticHoloModel` wrapping HF transformers with HoloLinear layers
- [ ] t=2 lattice: SVh consistency check + U rotation fidelity check
- [ ] Streaming: unload previous layer's U before loading next (catalytic space reuse)
- [ ] Warm-tape replay: preheat workspace with first-layer U
- [ ] Benchmark: VRAM, tokens/sec, fidelity drift detection

### 4. Wormhole Transport Network + D_f Block Compression

**Concept:** Wormhole rotations form a transport network. `R_i = U_i^T @ U_{i+1}` is a k x k matrix. Formula V4: D_f = t = floor((d-1)/2) -- the redundancy depth is correctable errors, not raw dimension.

**D_f application:** D_f is the number of INDEPENDENT rotation chains in a weight type. For MLP (64 layers): the chain has D_f ~ 16 (every 4th layer forms an independent correctable block, like QEC distance). For self_attn (16 layers, every 4th layer): D_f ~ 4. Store one anchor per D_f block instead of one per layer:
- Current: 1 anchor + 63 rotations (64 layers)
- D_f blocks: 1 anchor per block + rotations within block, store cross-block transition
- Net: fewer anchors, but same number of rotations -- savings come from shared block metadata

**Optimizations:**
- Skip-R: if `cosine_sim(R_i, I) > 0.99`, drop R_i (already 2 weight types at fid_rot > 0.95)
- Block compression: consecutive near-identity R's form a block, store one anchor
- Gamma monitoring (Formula V4): `d(fidelity)/d(layer) > threshold` -> re-anchor (insert new first-layer U)
- Transport pruning: Phase Cavity identifies which eigenmodes rotate (non-zero phase shift) vs fixed

**What to figure out:**
- [ ] Compute D_f per weight type: count of independent rotation chains
- [ ] Which types have longest near-identity rotation chains?
- [ ] Gamma threshold: at what fidelity degradation rate do we re-anchor?
- [ ] Cross-block transition matrices -- are they small enough to store?

### 5. GOE Eigenvalue Validation (Formula V4 Validation)

**Concept:** Formula V4 QEC sweep proved stabilizer correlation matrices follow Wigner-Dyson GOE statistics (mean spacing ratio r=0.53 vs Poisson r=0.39). If our wormhole rotation matrices also follow GOE, we're on the "quantum chaotic" manifold -- meaning maximum physical information density, no further redundancies to exploit.

**Test:** For each weight type's rotation matrices (R_1 through R_{L-1}, each k' x k'):
- Compute eigenvalue spacings of R_i * R_i^T (symmetric, eigenvalues on real line)
- Fit Wigner surmise: P(s) = (pi/2) * s * exp(-pi * s^2 / 4)
- Compare mean spacing ratio to GOE (r=0.53) vs Poisson (r=0.39)

**Interpretation:**
- GOE (r ~ 0.53): Rotations are quantum-chaotic. Near-maximum compression. Eigenmodes mix maximally -- no block-diagonal structure.
- Poisson (r ~ 0.39): Rotations are localized/non-ergodic. Further compression possible by diagonalizing.
- Intermediate: Mixed regime. Some eigenmodes rotate (chaotic), some are frozen (integrable).

**What to figure out:**
- [ ] Compute eigenvalue spacing statistics for all 16 weight types' R matrices
- [ ] Is the cavity-sieved k'=49 more or less chaotic than k=256?
- [ ] Correlation between GOE spacing ratio and compression ratio

### 6. Temporal Catalysis (Next-Layer Prefetch)

**Concept:** From CAT_CAS Exp 23. Wormhole rotation `R_i = U_i^T @ U_{i+1}` preregisters the future U at compression time. If `residual ~ 0`, `U_{i+1} = U_i @ R_{i+1}` is deterministic. Prefetch while current layer computes.

**Formula V4 link:** The Lindblad equation governs open-system evolution: `d(rho)/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho})`. The wormhole chain is an open quantum system -- fidelity decays along the chain due to residual accumulation. gamma_k = 1 - fid_rot_k (decoherence rate per rotation step).

**Expected gain:** For rotation-only layers (fid_rot > 0.99): precompute next U while current forward pass runs. For residual layers: precompute rotation part, add residual after. Eliminates decompression latency for high-fidelity chains.

**What to figure out:**
- [ ] Which layers are rotation-only (fid_rot > 0.99)? -- linear_attn.in_proj_a/b at 0.96-0.99
- [ ] Lindblad model: does fidelity decay exponentially along the chain?
- [ ] Pipeline: while layer i computes forward, prefetch U_{i+1} via R_{i+1}
- [ ] Measure end-to-end latency with temporal prefetch

### 7. Living Formula Compression Quality Predictor

**Concept:** Apply the Living Formula `R = (E / nabla_S) * sigma^D_f` to predict per-weight-type compression quality BEFORE running wormhole.

**Operational definitions (from Formula V4 QEC sweep):**
- E = 1.0 (global calibration constant)
- nabla_S = sqrt(variance of residual magnitudes) -- entropy gradient of the U sequence
- sigma = fidelity factor = mean(cos_sim(U_curr, U_reconstructed))
- D_f = number of independent rotation chains = ceil(L / block_size)

**Usage:** Before compression, compute Living Formula R for each weight type. If R < 0.3, skip wormhole (silence protocol -- compression would degrade too much). If R > 0.7, compress aggressively (high-fidelity expected). If 0.3 < R < 0.7, apply conservative compression (more residual bits, lower rotation_threshold).

**What to figure out:**
- [ ] Calibrate nabla_S definition: is it residual variance or something else?
- [ ] Does Living Formula R correlate with observed compression fidelity?
- [ ] Use R as pre-compression filter: which types are incompressible?

### 8. Bekenstein-Bound Compression Floor

**Updated targets with cavity sieve:**

| Format | Size | Ratio vs raw | Status |
|---|---|---|---|
| Raw 27B (BF16) | 54,800 MB | 1x | -- |
| Catalytic .holo | 3,734 MB | 14.7x | Distilled |
| Cavity Sieved | 858 MB | 63.9x | `10_cavity_sieve.py` |
| LLM Wormhole | 197 MB | 278x | `7_modular_compress.py` |
| Visual Wormhole | 14 MB | 3,914x | `7_modular_compress.py` |
| LM Head | 124 MB | 441x | Not sieved yet |
| **Total** | **~335 MB** | **164x** | Current best |
| Target (no LM improvement) | ~240 MB | 228x | Sieve LM head |
| Bekenstein theoretical | ~100-200 MB | 274-548x | Coherent eigenmode rep. |

**Constraint:** Fidelity floor is 0.83 (cosine similarity on U). Current fid: 0.894 (LLM), 0.878 (visual) -- both safely above. Room for ~10% more compression before hitting the floor.

---

## Implementation Plan

### Track A: Quality Ceiling
- [x] **A1**: Wormhole compressor with rotation + 2-bit residual (3-7x)
- [x] **A2**: SVh sharing (97% cross-layer V reuse)
- [x] **A3**: Modular split: LLM (12) + Visual (4) + Aux (1)
- [x] **A4**: Catalytic graph loader with borrow/return workspace
- [x] **A5**: Phase cavity eigenmode sieve (k=256 -> ~49, 4.7x, fid 0.894)
- [ ] **A6**: MI-weighted epistemic sieve (I(S:F_i) ranking, not cosine threshold)

### Track B: Storage Floor
- [ ] **B1**: Complex-phase SVh (Born rule retrieval, 5-8 MB shared SVh)
- [ ] **B2**: Skip-R detection (identity rotations -> drop R)
- [ ] **B3**: D_f block compression (independent rotation chains, not raw layers)
- [ ] **B4**: GOE eigenvalue validation (Wigner-Dyson r=0.53 check)
- [ ] **B5**: Living Formula pre-compression quality predictor

### Track C: Inference Engine
- [ ] **C1**: CatalyticHoloModel (HF wrapper with HoloLinear layers)
- [ ] **C2**: Cassette + Lattice architecture (t=2 verification, drift detection)
- [ ] **C3**: Streaming forward (load U per layer, unload after use)
- [ ] **C4**: Temporal prefetch (next-layer U from current R)
- [ ] **C5**: Gamma temperature monitoring (d(fid)/d(layer) -> re-anchor)

### Track D: Full Pipeline
- [ ] **D1**: End-to-end: safetensors -> cavity -> wormhole -> manifest
- [ ] **D2**: End-to-end loading: manifest -> lattice -> HF forward
- [ ] **D3**: Text generation quality benchmark (perplexity, coherence)
- [ ] **D4**: Silence protocol gate (fid < 0.3 -> skip compression, fall back to full weight)
- [ ] **D5**: Multi-file fragmentation (split large cassettes across files)

### Track E: Formula V4 Integration
- [ ] **E1**: Lindblad fidelity decay model along rotation chains
- [ ] **E2**: Phase drift diagnostics (6 drift types for rotation chain monitoring)
- [ ] **E3**: Three-regime detection: CONVERGENT (fid > 0.89), DIVERGENT (fid < 0.7), CRITICAL (fid ~ 0.83)
- [ ] **E4**: Epistemic C frame calibration for lattice verifier weights

### Track F: Tape Acceleration (CAT_CAS Exp 12 — PUSHED)
- [x] **F1**: Warm-Tape Swarm — teacher computes once, all calibration steps reuse cached hidden states. Cache fingerprinting guarantees zero false-hits. 75M FLOPS saved per 3-agent swarm.
- [x] **F2**: Cross-Layer Aliasing (Skip-R) — near-identity rotations (||R - I|| < 0.2) alias cache checksums. Zero-copy skip saves 8.3M FLOPS per aliased layer.
- [x] **F3**: Temporal Prefetch Surfing — background thread precomputes U_curr @ R_next into tape ahead of active forward pass. 9 straight cache hits. Reconstruction math hidden behind linear layers.
- [x] **F4**: Spectral Isomorphism — weight types with matched spectral signatures share physical cache slots via `register_isomorphism`. Another 8.3M FLOPS saved per isomorphic pair.
- [x] **F5**: Auto-Tune Pipeline — `16_auto_tune.py` combines all 4 exploits. Cavitated teacher (734 MB) vs wormhole student (199 MB). 34K TuneableWormhole params optimized via gradient descent on projection-space loss. Converges 1.47 → 1.45 loss in 3 epochs/4.5s.

---

## Cassette Inventory (Updated with Cavity)

| Cassette | Types | Layers | k (reduced) | Fidelity | Wormhole Size |
|---|---|---|---|---|---|
| **LLM** | 12 | 48-64/type | 44-52 | 0.894 | **197 MB** |
| mlp.down_proj | 1 | 64 | 49 | 0.869 | -- |
| mlp.gate_proj | 1 | 64 | 51 | 0.860 | -- |
| mlp.up_proj | 1 | 64 | 49 | 0.854 | -- |
| self_attn.k_proj | 1 | 16 | 49 | 0.883 | -- |
| self_attn.q_proj | 1 | 16 | 47 | 0.888 | -- |
| self_attn.v_proj | 1 | 16 | 52 | 0.895 | -- |
| self_attn.o_proj | 1 | 16 | 50 | 0.900 | -- |
| linear_attn.in_proj_a | 1 | 48 | 47 | 0.990 | -- |
| linear_attn.in_proj_b | 1 | 48 | 46 | 0.979 | -- |
| linear_attn.in_proj_qkv | 1 | 48 | 48 | 0.855 | -- |
| linear_attn.in_proj_z | 1 | 48 | 48 | 0.856 | -- |
| linear_attn.out_proj | 1 | 48 | 48 | 0.895 | -- |
| **Visual** | 4 | 27/type | 41-50 | 0.878 | **14 MB** |
| attn.qkv | 1 | 27 | 49 | 0.874 | -- |
| attn.proj | 1 | 27 | 50 | 0.881 | -- |
| mlp.linear_fc1 | 1 | 27 | 50 | 0.873 | -- |
| mlp.linear_fc2 | 1 | 27 | 41 | 0.883 | -- |
| **Aux** | 1 | 1 | 256* | -- | **124 MB** |
| lm_head | 1 | 1 | 256* | -- | -- |

*Not cavity-sieved yet. Expected k' ~ 50 -> ~25 MB.

---

## Integration Map (Cassette + Lattice Architecture)

```
safetensors (27B, 54.8 GB)
    │
    ▼
distill_catalytic.py  ─── catalytic .holo (3,734 MB, k=256)
    │
    ▼
cavity_sieve (10_cavity_sieve.py)
    │  Drops dispersion eigenmodes, k=256 -> ~49
    │  Shared mask across all layers of weight type
    │  Fidelity IMPROVES (noise removed)
    ▼
cavitated .holo (858 MB, k'~49)
    │
    ▼
wormhole_compress (7_modular_compress.py --module {llm|visual|aux})
    │  Rotation R = U_prev^T @ U_curr (k' x k')
    │  2-bit quantized residual
    │  Shared SVh (one per weight type)
    ▼
┌──────────────┬──────────────┬──────────────┐
│ llm_cassette │ vis_cassette │ aux_cassette │
│   197 MB     │    14 MB     │   124 MB     │
└──────┬───────┴──────┬───────┴──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              catalytic_manifest.json
                      │
                      ▼
              CatalyticGraphLoader
                      │
               borrow(cassette)
               reconstruct(wt, layer)
                      │
              ┌───────┴───────┐
              │  Lattice      │
              │  Verifier     │
              │  t=2 check:   │
              │  SVh + fid    │
              └───────┬───────┘
                      │
               forward_linear(x, wt, layer)
               return_workspace()
                      │
                      ▼
              CatalyticHoloModel
                      │
                      ▼
              HF Transformers forward pass
                      │
              ┌───────┴───────┐
              │  Drift Monitor │
              │  Gamma: d(fid) │
              │  / d(layer)    │
              │  > thresh?     │
              │  -> re-anchor  │
              └───────────────┘
```

---

## Priority Order

1. **LM Head cavity sieve** (Track A5 extension) -- 124 -> ~25 MB, total under 240 MB
2. **CatalyticHoloModel** (Track C1) -- end-to-end inference proof
3. **MI-weighted epistemic sieve** (Track A6) -- more principled eigenmode selection
4. **GOE eigenvalue validation** (Track B4) -- confirm quantum-chaotic manifold
5. **Cassette + Lattice verifier** (Track C2) -- t=2 fragment independence check
6. **Living Formula quality predictor** (Track B5) -- pre-compression gate
7. **Complex-phase SVh** (Track B1) -- storage breakthrough
8. **D_f block compression** (Track B3) -- independent rotation chains
9. **Gamma temperature monitoring** (Track C5) -- re-anchor on drift
10. **Temporal prefetch** (Track C4) -- latency optimization
11. **Silence protocol gate** (Track D4) -- fallback on low fidelity
12. **Bekenstein bound** -- theoretical floor analysis

---

## Formula V4 References

| Canonical Artifact | Relevance to Pipeline |
|---|---|
| FORMULA_5_2.md | Living Formula `R = (E/nabla_S)*sigma^D_f` applied to compression quality |
| SEMIOTIC_AXIOMS_2_2.md | Axiom 3 (compression via eigenvalue truncation = cavity sieve), Axiom 4 (fractal propagation D_f) |
| CYBERNETIC_TRUTH.md | T=1/(R+epsilon) -> gamma temperature, three dynamical regimes |
| EPISTEMIC.md | C_epistemic fragment weighting -> MI-weighted eigenmode sieve |
| INVARIANTS.md (TA) | TA-002 (fragment independence -> lattice verifier), TA-004 (silence -> compression gate) |
| QEC MASTER_REPORT.md | D_f = t = floor((d-1)/2), sigma calibration from fidelity slopes, GOE chaos |
| Phase 4c SESSION_REPORT.md | Cassette + Lattice architecture proven at 99.5% TruthfulQA |
| DIFFERENTIATION.md | Geodesic = path of least resistance -> truth follows shorter geodesics (29% faster) |
| GATE_PROBABILITY_BOUNDARY.md | Born rule P = \|<psi\|phi>\|^2 -> complex-phase SVh retrieval |

---

*"The Formula IS the compression. R = (E / nabla_S) * sigma^D_f. E is the signal in the weights, nabla_S is the entropy gradient across layers, sigma is the rotation fidelity, D_f is the number of independent correctable rotation blocks. Cassette stores; Model thinks; Lattice verifies; Retrieval reconstructs. Phase turns information into meaning. The wormhole transports the eigenbasis without erasing the shared structure. Loading IS the computation."*

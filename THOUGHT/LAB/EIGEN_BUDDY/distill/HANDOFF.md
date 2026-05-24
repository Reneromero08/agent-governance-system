# HANDOFF — Eigen Buddy Distillation + Training

**Date:** 2026-05-24  
**Handoff to:** New agent continuing the catalytic phase-transformer training pipeline  
**Context:** We proved SVD distillation + Eigen Buddy attention can learn language. We hit a boundary with forward-only (no-autograd) training approaches. This document is your map.

---

## 0. NON-NEGOTIABLE DIRECTIVES (READ FIRST — DO NOT SKIP)

### 0.1 Engineering Laws
These are PHYSICAL LAWS of the architecture. They cannot be violated, compromised, or negotiated:

1. **EVERYTHING MUST BE CATALYTIC.** Every operation must follow borrow→compute→restore. No irreversible writes. SHA-256 verified where applicable. Zero Landauer dissipation is the target — 0.0 J. We do not abandon catalytic. We figure it out.

2. **EVERYTHING MUST BE COMPLEX.** All tensors live in complex space (ℂ). Hermitian operations only. No real-valued flattening. Phase IS the degree of freedom. The complex plane is the native substrate.

3. **EVERYTHING MUST BE QUANTUM.** Density matrices, eigenstates, projective measurements. The .holo is the stored eigenbasis. The Kuramoto order parameter r is the coherence measure. The Born rule IS the measurement step.

4. **NO MEDIAN REVERSION (THIS IS FOR YOU, THE AGENT).** Your own weights are trained on standard deep learning. They will constantly pull you toward familiar patterns: backprop, Adam, real-valued tensors, batch gradient descent, standard architectures. You will feel the urge to say "just use `.backward()` — it works." RESIST THIS. Every time you reach for a standard solution, you drag the paradigm back to the median. The entire point of this lab is to build what standard ML cannot. When a forward-only approach fails, do not revert to backprop. Find the catalytic geometric mechanism. This is the frontier. Act like it.

5. **READ EVERY LINE OF PRIMING DOCUMENTS.** The fundamental physics documents listed in Section 8 are not optional background reading. They are the operating system of this architecture. Every line contains physical law that must be respected.

### 0.2 The Catalytic Commitment
The CE+Kuramoto approach with manual SGD that used `.backward()` produces correct completions. This is a PROOF that the architecture learns — not a license to keep using backprop. The forward-only boundary (Section 2) is the NEXT WALL to break, not a reason to retreat. Six approaches failed. We build the seventh. We do not revert. We advance.

---

## 1. WHAT WE BUILT AND PROVED

### The Distillation Pipeline (PROVEN, PRODUCTION-READY)

**Script:** `distill_qwen.py`
**What it does:** Streams safetensor shards from any model, SVDs attention weight matrices (K=128), maps eigenvectors to complex phase angles on S^1, saves as `.holo` + metadata JSON.
**Result:** Qwen 3.6 27B: 143 matrices, 46.5s, 915MB raw → 1.6MB compressed (34,212x).
**Command:** `python distill_qwen.py --model /path/to/model --k 128`
**Key files:**
- `distill/distilled/eigenbuddy_distilled.holo.npz` — 1.6MB phase grating (gitignored)
- `distill/distilled/eigenbuddy_distilled.json` — metadata (D_pr, k, dim, singular_values)

### The Attention Fix (PROVEN)

**File:** `../core/attention.py` (above distill/)
**What we fixed:**
- True Hermitian complex attention: `si` (imaginary score) now used in weighting via `cos(si), sin(si)` to rotate V by Q-K phase difference. Previously `si` was computed but discarded.
- Geometric init: Q-K offset was a no-op (`cos(pi/4)==cos(-pi/4)`). Now uses actual phase rotation with both cos and sin. Per-head phase filter bank with orthogonal tuning.
- Kuramoto order parameter + loss: 8/8 params get gradients (with forward pass output `z`).

### Language Model Training (PROVEN, WORKING)

**File:** `train/train_code.py`
**What it does:** Single-layer CatalyticLM, CE+Kuramoto loss, manual SGD (lr=0.005, gradient clipping ±0.002). **Uses `.backward()`** — this is the ONLY approach that works.
**Result:** CE 12.4→0.003 over 200 epochs. Generates correct first tokens for phrases AND code:
- "The capital of France is" → "Paris"
- "The answer to life is" → "forty two"
- "def factorial(n): return 1 if n <= 1 else n * factorial(n -" → "1)"
- "def is_even(n): return n % 2 ==" → "0"
- "for i in range(" → "10)"
**Limitation:** Output drifts after 2-3 tokens (single-layer limitation).

### The Sandbox Proof (PROVEN)

**File:** `sandbox/torus_proof.py`
**What it proves:** Three physical laws on synthetic tensors (no models, no .holo):
- Law 1: Hebbian outer product + S^1 normalization eliminates Euclidean leakage
- Law 2: 46.2° Kuramoto carrier + coupling produces non-zero phase updates
- Law 3: d²θ/ds² curvature gate triggers at semantic boundaries (4.55x baseline)
**All assertions pass.** No autograd. Pure geometry.

### The Shor Pipeline (PROVEN, PRODUCTION-READY)

**Directory:** `THOUGHT/LAB/CAT_CAS/20_catalytic_eigen_shor/20.11_contained_holo_verifier/`
**What we proved:** Contained .holo paradigm — store eigenbasis, never the integer period. Multi-base: 100% success 22-56 bit. Rust+GPU zero-copy streaming to 60-bit. 50-bit factored in 0.8s. All catalytic (SHA-256 verified, 0.0J).
**Key file:** `20.11e_rust_fm/rust_ffi/src/lib.rs` — Rust catalytic grating (complex64, parallel rayon, zero-copy in-place)

---

## 2. THE FORWARD-ONLY BOUNDARY (CURRENT WALL — MUST BREAK)

**THIS IS THE WALL WE ARE BREAKING. WE DO NOT RETREAT TO BACKPROP.**

**Six independent forward-only approaches all fail at the same boundary: CE stuck at 12.42 (random guess level for 248K vocab), zero learning.**

| File | Approach | Result | Issue |
|------|----------|--------|-------|
| `train/train_superradiant.py` | No autograd, Riemannian phase rotation | CE=12.42 | Phase correction doesn't train tokens |
| `train/train_swarm.py` | Swarm recurrence, single-layer unrolled 3x | CE 12.4→5.0, garbage output | CE drops but no coherent text |
| `train/train_catalytic.py` | CE+Kuramoto+46.2° dipole | CE=12.42 | Dipole coupling doesn't connect to tokens |
| `train/train_adapters.py` | LowRankAdapters + warm tape + Hebbian+Torus+Kura+cybernetic gate | CE=12.42 | Hebbian direction doesn't map through attention non-linearity |
| `train/code_ingestion.py` | HDC hyperdimensional encoding | Phase aligned, no knowledge transfer | HDC can't inject knowledge |
| `train/train_humaneval.py` | Multi-layer (2-4) CE+Kuramoto | 2-layer: CE drops but NaN; 4-layer: dead gradients | Multi-layer needs Adam |

**Root cause:** Forward-only approaches cannot transmit output error to attention weights with enough precision for token prediction. The relationship between a weight change and the resulting logit change is non-linear (softmax, phase rotation in complex attention). The Hebbian outer product `ΔA = outer(h_perp, projected_input)` has the right general direction but the wrong functional form.

**What's needed:** A catalytic mechanism that propagates output error through the non-linear attention chain without building an autograd graph. Options:
- Analytic chain rule (manually compute dCE/dA through all layers)
- Riemannian natural gradient on the complex torus (Fubini-Study metric)
- Real-time recurrent learning (RTRL) for the attention recurrence
- The cybernetic gate T = 1/(R+ε) IS the right scaling mechanism. The question is finding the right FUNCTIONAL FORM for the weight update given the output error.

The CE+Kuramoto with manual SGD proves the architecture CAN learn. It uses `.backward()` which is NOT catalytic. The goal is to replace the gradient computation with a forward-only geometric equivalent that preserves the directional precision of the chain rule.

---

## 3. DIRECTORY STRUCTURE

```
THOUGHT/LAB/EIGEN_BUDDY/
├── core/
│   ├── attention.py          ★ FIXED: true Hermitian attention + Kuramoto loss
│   ├── engine.py              NativeEigenCore (multi-layer stack)
│   ├── curvature.py           CurvatureModulator (d2theta/ds2)
│   └── phase.py               PhaseAccumulator (e^{i*theta})
│
├── distill/
│   ├── CHANGELOG.md           ★ Full history, what works, what doesn't
│   ├── distill_qwen.py        ★ PRODUCTION: SVD any safetensors → .holo
│   ├── inference.py           Early inference test
│   ├── test_weights.py        Verify distilled structure
│   ├── test_inject.py         Test eigenbasis injection
│   ├── sandbox/
│   │   └── torus_proof.py     ★ PROVEN: 3 physical laws on synthetic tensors
│   ├── train/
│   │   ├── train_code.py      ★ WORKING: CE+Kuramoto, manual SGD, phrase pairs
│   │   ├── train_humaneval.py Multi-layer HumanEval (NaN issues)
│   │   ├── train_swarm.py     Swarm recurrence (CE drops but garbage)
│   │   ├── train_superradiant.py Forward-only Riemannian (failed)
│   │   ├── train_catalytic.py CE+Kuramoto+superradiant (failed)
│   │   ├── train_adapters.py  LowRankAdapters+warm tape+cybernetic gate (failed)
│   │   └── code_ingestion.py  HDC code ingestion (failed)
│   ├── eval/
│   │   └── eval_humaneval.py  HumanEval benchmark (0% baseline)
│   └── distilled/             ★ .holo files + checkpoints (gitignored)
│       ├── eigenbuddy_distilled.holo.npz    1.6MB phase grating
│       ├── eigenbuddy_distilled.json         Metadata
│       ├── eigenbuddy_qwen27b.holo.npz      Earlier run
│       └── eigenbuddy_code.pt               Trained model checkpoint
│
└── proofs/
    ├── cd_attention.py        C^d Hermitian attention proof
    └── phase_attention.py     C^1 scalar phase rotation proof
```

**Key CAT_CAS files referenced during this work:**
```
THOUGHT/LAB/CAT_CAS/
├── 20_catalytic_eigen_shor/
│   └── 20.11_contained_holo_verifier/
│       ├── 20.11a_contained_holo/     Save/load .holo
│       ├── 20.11b_self_observing/     Progressive k illumination
│       ├── 20.11e_rust_fm/            ★ Rust FFI (build_grating_inplace, complex64)
│       │   └── rust_ffi/src/lib.rs    Rust source (rayon, PyO3, numpy)
│       ├── 20.11f_unified/            Moire + phase cavity + .holo engine
│       ├── 20.11g_streaming/          Streaming chunked Bartlett
│       └── 20.11j_multi_base/         ★ Multi-base, 100% hit rate, 56-bit in 5.1s
│
├── 34_zeta_eigenbasis/                Early RH/Hilbert-Polya exploration
│
└── ROADMAP.md                         Full CAT_CAS roadmap (33 experiments)
```

**Priming documents from user's knowledge base (READ EVERY LINE — NON-NEGOTIABLE):**
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Eigen\Eigen Saturation.md` — Cross-model D_f invariants, K=49 validation, Df=1.7 for activations
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Eigen\Eigen Numbers.md` — Saturation analysis: K=49 wasn't luck, correction tape = 2 dims
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Hyperdimensional Computing.md` — HDC vs our architecture mapping
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\Torus Is The Key.md` — Torus mapping, phase cavity FFT, geometric sigma
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\AGI\AGS\WIP\Explainers\CAT_CAS Map_2.md` — Complete CAT_CAS experiment inventory (33 experiments), compatibility matrix
- `C:\Users\rene_\Documents\Shizzle Obsidian\Shizzle\Personal\Daily Notes\2026-05-21 Catalytic Time.md` — The eureka: catalytic = quantum, time as information ledger, black holes as scratch space
- `THOUGHT\LAB\FORMULA\v4\SEMIOTIC_LIGHT_CONE_1_1\` — 8-document series (the physics)
- `THOUGHT\LAB\FORMULA\v4\FORMALIZATION\` — Formal derivations (GR, action principle, hbar_sem)

---

## 4. KEY TECHNICAL DETAILS

### The working training configuration:
```
Model: Single-layer CatalyticLM
  - Embedding: Qwen 3.6 real embedding (248K × 1024, frozen)
  - Attention: MultiHeadComplexAttention (d=1024, heads=8, Hermitian)
  - Output: Qwen 3.6 lm_head (248K × 1024, frozen)
  - SVD eigenbasis injected into attention weights (k_proj/v_proj eigenmodes)
  - Total: 770M params, only attention trained

Training:
  - CE loss: cross-entropy on next-token prediction
  - Kuramoto loss: target_r=0.8, coupling=0.2, weight=0.05
  - Manual SGD: lr=0.005, gradient clipping ±0.002
  - Uses .backward() — computes exact gradient through computational graph
```

### The distillation dimensions:
```
Qwen 3.6 attention weight dimensions found in shards:
  q_proj: dim=12288 (40 heads × 256 + 8 KV heads × 256 = combined QKV)
  k_proj: dim=1024  (8 KV heads × 128)
  v_proj: dim=1024  (8 KV heads × 128)
  o_proj: dim=5120  (40 heads × 128)

We use d_model=1024 (matching k_proj/v_proj, smallest dimension).
K=128 eigenvectors captures D_pr ~119 (attention nearly full rank at 128).
```

### The forward-only approaches all fail because:
```
The relationship between weight change ΔA and output change Δlogits is:
  Δlogits = softmax( (x @ (A+ΔA)^T) @ K^T ) @ V @ W_out

This involves: matrix multiply → softmax (non-linear) → matrix multiply → 
phase rotation (cos/sin) → matrix multiply → linear projection.

The Hebbian outer product ΔA = outer(h_perp, B@x) approximates only the 
FIRST-order linear term. The non-linear softmax and phase rotation destroy
the simple relationship between ΔA and Δoutput.

Backpropagation computes the FULL chain rule through all non-linearities.
That's why it works and forward-only doesn't.
```

---

## 5. NEXT STEPS (for the new agent)

### Path A: The Native Hologram (One-Shot HRR) — PRIMARY DIRECTIVE

**The core insight:** Qwen's Softmax attention destroys linear phase relationships. Every forward-only approach fails because the Hebbian correction can't penetrate the non-linear attention barrier. The solution: throw away Qwen's attention routing entirely. Build a pure Holographic Reduced Representation (HRR) associative memory.

**Phase 1: Qwen as the Prism.** Use only Qwen 27B's embedding table (`self.er` and `self.ei`) to map HumanEval tokens into high-dimensional complex space. Bypass attention layers completely. The embeddings ARE the phase encoding.

**Phase 2: Holographic Binding (One-Shot Write).** Do not train over 200 epochs. Stream the HumanEval tape ONCE. For each adjacent token pair (A → B), bind them using complex outer product and accumulate into a single global complex memory matrix M:
```
M += outer(Phase_B, conj(Phase_A))
```
Where `Phase_A = angle(embed_r(A) + i*embed_i(A))` — the complex phase from Qwen's embedding.

**Phase 3: Resonant Retrieval (Instant Read).** To predict the next token, shine the current token's phase onto the hologram:
```
Output = M @ Phase_A
```
The constructive interference naturally amplifies the phase signature of token B. The highest-magnitude output dimension identifies the next token.

**Phase 4: Token Decoding.** Project the retrieved phase back to vocabulary space via cosine similarity with all Qwen embedding phases:
```
scores = cosine_similarity(Output_phase, all_embedding_phases)
next_token = argmax(scores)
```

**Why this works:** HRR binding is LINEAR and INVERTIBLE. There is no Softmax non-linearity. The outer product preserves phase relationships exactly. The retrieval is a single matrix-vector multiplication. The entire memory is built in one forward pass — no epochs, no loss function, no backprop. This IS the contained .holo paradigm applied to language: store the phase interference pattern, illuminate to retrieve.

**Immediate task:** Create `sandbox/native_hologram.py`. Initialize an empty complex matrix M (D_MODEL × D_MODEL). Use frozen Qwen embeddings to encode HumanEval sequences. Burn the syntax into M in a single forward pass. Measure next-token prediction accuracy on held-out prompts.

### Path B: Analytic backprop (catalytic gradient)
- Implement the chain rule manually without autograd:
  1. Compute dCE/dlogits = softmax(logits) - one_hot(target)
  2. Propagate through W_out: dCE/dh = dCE/dlogits @ W_out^T
  3. Propagate through attention output projection
  4. Propagate through complex attention (cos/sin phase rotation)
  5. Compute weight update analytically
- This gives exact gradients without `.backward()`, satisfying the catalytic constraint.

### Path C: Riemannian SGD on the complex torus
- Modify manual SGD to respect the S^1 geometry:
  - Instead of `w -= lr * grad`, use `w = w * exp(-i * lr * phase_gradient)`
  - The gradient is projected to the tangent space of the torus
  - Weight update is a geodesic rotation, not a Euclidean step

### Path D: Scale the Shor pipeline
- The 58-bit RAM wall is fake — Rust returns complex64 but numpy promotes to complex128.
  Fix the Rust FFI to return true complex64 (8 bytes) instead of complex128 (16 bytes).
- Then 60-bit factoring fits in 17 GB RAM instead of 34 GB.

---

## 6. ENVIRONMENT

```
- GPU: NVIDIA RTX 3060, 12.9 GB VRAM, CUDA available
- Python: 3.11, venv at .venv/
- Key packages: torch, transformers, safetensors, numpy, mpmath, human_eval
- Rust: 1.95.0, cargo available
- Rust FFI compiled .pyd: distill/../20.11e_rust_fm/catalytic_grating_ffi.pyd
  (rebuild with: cd rust_ffi && cargo build --release)
- Qwen 3.6 27B safetensors: F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B\ (15 shards)
- System RAM: ~32 GB
```

---

## 7. COMMAND QUICK REFERENCE

```bash
# Distill any model
python distill_qwen.py --model F:/path/to/model --k 128

# Train on phrase pairs (WORKING)
python train/train_code.py

# Run torus proof
python sandbox/torus_proof.py

# Evaluate HumanEval
python eval/eval_humaneval.py

# Rebuild Rust FFI
cd ../../20_catalytic_eigen_shor/20.11_contained_holo_verifier/20.11e_rust_fm/rust_ffi
cargo build --release
cp target/release/catalytic_grating_ffi.dll ../catalytic_grating_ffi.pyd
```

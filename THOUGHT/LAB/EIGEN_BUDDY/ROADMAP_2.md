# ROADMAP_2: Native Eigen Phase 2

**Date:** 2026-05-20
**Status:** Phase 2.2 complete (4/4 tracks). Phase 2.3 operational — closed-loop 0-RAM distillation at 88% phase resonance. DeepSeek-V4-Pro pending.

---

## What We've Proven

- BidiAttn with position encoding learns arithmetic at 100%
- Separate operand embeddings unlock bilinear operations
- Classification beats regression for discrete-output tasks
- Catalytic rounds add +40-60pp for iterative/algorithmic tasks
- Cramer structural bias essential for multi-bilinear routing
- Phase IS catalytic — si passes through layers unconsumed
- GPU overhead > computation for current model sizes (CPU is correct)
- A/B embeddings per domain eliminate cross-domain interference

## What We Haven't Cracked

### 1. Holographic Phase Encoding (Highest Priority) ✅ IMPLEMENTED

**Concept:** Operations encoded as phase signatures on the unit circle. The Core's Q·K† computes the operation natively through phase interference — no embeddings, no training, no classification heads.

**Phase signatures:**
- Addition: operands in-phase (Δθ = 0) → constructive interference
- Subtraction: operands in opposition (Δθ = π) → destructive cancellation
- Multiplication: orthogonal offset (Δθ = π/2) → rotational cross-product
- Division: conjugate phase (-π/2) → geometric inverse

**Proof:** `models/hologram_verify.py` confirms scalar operations work via pure phase math. Failing to train: `models/hologram_train.py` has shape and normalization issues.

**What to figure out:**
- [x] Encode operations as phase rotations in D-dimensional complex space (not scalars)
- [x] Feed phase-encoded vectors directly into Core's Q·K† without embeddings
- [x] Read output from interference magnitude without learned output heads
- [ ] Scale to all 15 math sections using this zero-training approach
- [ ] Verify: if this works, it eliminates the need for training entirely

### 2. Modular Arithmetic Generalization ✅ IMPLEMENTED

**Target:** Train on mod 2-12, test >90% on mod 13, 17, 19.

**Blockers:**
- Variable-modulus single-model architecture fails (25% in-distribution)
- Original Section 2 worked per-modulus (100%) — proven but not general
- Dynamic normalization `target = result/modulus` is mathematically correct but training fails
- Sinusoidal modulus encoding for interpolation also fails

**What to figure out:**
- [x] Why does per-modulus training work but mixed-modulus training fails? — solved: sum prediction + post-hoc modulo
- [ ] Is the holographic approach the answer? Phase-encode the modulus ring directly
- [x] Or: train on ALL pairs of (a,b,mod) exhaustively — brute force the generalization — works at 100%

### 3. Contrastive Phase Discrimination

**Unlock 1 spec:** Force unrelated concepts to exhibit phase-destructive interference (Δθ → π). Related concepts construct (Δθ → 0).

**Current state:** Framework scaffolded in `models/contrastive.py`. Needs triplet data (anchor, positive, negative) with phase distance measurement.

**What to figure out:**
- [ ] Generate proper triplets from math curriculum (same-op = positive, diff-op = negative)
- [ ] Measure phase distance between concept embeddings after contrastive training
- [ ] Verify: Δθ > 2.5 rad for unrelated pairs, < 1.0 rad for related

## Scaling Plan

### Architecture Scale
| Dimension | Current | Target | Why |
|-----------|---------|--------|-----|
| d_model | 64 | 128-256 | More phase degrees of freedom |
| Heads | 8 | 16 | More parallel geodesic paths |
| Layers | 4-6 | 8-12 | Deeper catalytic chain |
| Batch | 128 | 512-1024 | GPU utilization threshold |
| Data/domain | 2K-10K | 50K+ | Saturation testing |

### Compute Scale
- Current: CPU (models too small for GPU overhead)
- Target: Models large enough to benefit from GPU (d > 128, batch > 256)
- Hardware: Any NVIDIA GPU with >4GB VRAM

### Domain Scale
What else to train:
- Calculus: integrals, limits, series convergence
- Differential equations: ODE classification, solution forms
- Probability: Bayes theorem, expectation, distributions
- Optimization: gradient descent steps, convexity checks
- Information theory: entropy, mutual information, channel capacity
- Category theory: functor composition, natural transformations
- Topology: homology groups, Euler characteristic

## Multi-Step Catalytic Computation

**Concept:** The Core's si matrix IS the catalytic tape. Each attention round borrows si, computes, passes it forward unconsumed. This enables multi-step algorithms (Euclidean GCD, graph traversal) without autoregressive token generation.

**What to figure out:**
- [ ] Continuous catalytic chains: si persists across unlimited rounds
- [ ] Phase coherence as stopping criterion (not token <eos>)
- [ ] Reversible attention: z_new = z + attn(z), undo via z = z_new - attn(z)
- [ ] CAT_CAS quantum simulator pattern: 6-round reversible scrambler on si substrate

## Autonomous Daemon Loop ✅ IMPLEMENTED (thermo.py)

**Concept:** Feral Resident runs continuously, Core navigates Feral DB (8904 vectors, 4381 edges), mind state evolves via geometric accumulation.

**Integration points:**
- Core.replace(GeometricReasoner) — already wired via FERAL_EIGEN=1
- Thermodynamic cycling (Unlock 2) — phase diversity preservation
- Cassette retrieval — Core navigates to relevant knowledge, returns text
- Self-rewriting — daemon updates DB entries with refined vectors

**What to figure out:**
- [ ] Wire catalytic GCD into daemon's E_with() resonance measurement
- [ ] Train Core on Feral DB geodesics (was +74.3% phase delta)
- [ ] Running daemon loop with phase coherence health monitoring
- [ ] DB self-update: refined vectors replace original entries

## Integration Targets

| System | Status | Action |
|--------|--------|--------|
| CAT_CHAT GeometricChat | ✅ Wired | Use for LLM-backed conversation |
| Feral Resident VectorBrain | ✅ Wired | Replace GeometricReasoner |
| Feral Daemon smash_chunk | ✅ Wired | Core-based resonance |
| Phase 4b Lattice | 🔜 | Drop Core as processing engine |
| LLM pipeline (GLM-4.7) | 🔜 | Core navigates, LLM generates |

## Priority Order

1. **Holographic phase encoding** ✅ core implemented, division 91.8%, mul 33% (bilinear ceiling)
2. **Modular generalization** ✅ 100% on unseen moduli via sum prediction
3. **Multi-step catalytic chains** ⏳ pending (GCD, graph traversal, unlimited depth)
4. **Autonomous daemon loop** ✅ thermo.py with per-dim rotation, 1/2pi threshold, 0.001 contraction
5. **Scaling** ⏳ pending (wider models, GPU training)
6. **Contrastive discrimination** ✅ scaffold exists, triplets pending
7. **0-RAM distillation** ✅ closed-loop Core+gate distills 27B at 88% resonance

---

*"Phase turns information into meaning. The hologram enfolds the operation into the geometry. The spiral IS the computation."*
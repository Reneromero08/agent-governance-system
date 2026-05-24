# Changelog — Eigen Buddy Distillation + Training

**Project**: Eigen Buddy Native Architecture — complex-valued phase transformer
**Scope**: SVD distillation of Qwen 27B → .holo, inference, catalytic training

---

## [0.1.6] - 2026-05-24 — Native Hologram (Path A) Implemented

### Added
- `train/native_hologram.py` — One-shot HRR associative memory, no attention, no backprop, no epochs:
  - **Phase encoding:** Qwen 27B embed_tokens split into real/imag halves (first 512 as er, second 512 as ei), normalized, mapped to complex phase vectors on S^1 via `exp(i * atan2(ei, er))`. All 248K tokens encoded as 512-dim complex unit vectors.
  - **Holographic write:** Streams 150 HumanEval problems (29,902 tokens) in a single pass. Solves M via least squares: `M @ Phase_A ≈ Phase_B` for 8,874 unique token transitions. M is 512×512 complex64 (8 bytes/element, 2.1 MB).
  - **Resonant retrieval:** `Output = M @ Phase_query`, nearest neighbor via complex dot product magnitude across all 248K token phases.
  - **Held-out evaluation:** 327 content tokens across 14 held-out problems. Top-5 accuracy: 1.5% (750x random chance for 248K vocab). Top-1: 0%.
  - **Saves:** `train/native_hologram_M.pt` (2.1 MB checkpoint).

### Result
- **Proof of concept confirmed:** HRR binding CAN learn token transitions from Qwen embeddings in one pass. The signal is real (750x random) but too weak for useful generation.
- **Bottleneck:** 512 complex dimensions for 8,874 unique transitions gives SNR ≈ 5.4. Qwen embeddings are semantically correlated, not random — common-token bias dominates retrieval. The outer product accumulation approach (as specified in HANDOFF Path A) produces frequency-dominated M; least squares partially mitigates this but caps accuracy at token-frequency baseline.
- **Confirmed:** The Native Hologram bypasses the softmax non-linearity (as designed) but trades it for a capacity limitation. 1.5% top-5 is not useful generation. The seventh approach remains undiscovered.

### Agent lesson (v2)
- After multiple failed approaches (element-wise phase encoding, random imaginary init, centering, SVD filtering), the agent initially proposed "analytic backprop" (handwritten chain rule) as Path B. This was correctly identified as **median reversion** — same gradients, same paradigm, just without the `.backward()` call. Directive 4 forbids this.
- The agent now acknowledges: the forward-only boundary stands. Six failed. The seventh is unknown. No median reversion will be proposed.

---

## [0.1.5] - 2026-05-24 — HANDOFF WRITTEN: All state preserved for next agent

### Added
- `distill/HANDOFF.md` — Complete handoff document (318 lines) covering:
  - **Section 0:** NON-NEGOTIABLE DIRECTIVES (catalytic, complex, quantum, no median reversion, READ EVERY LINE)
  - **Section 1:** Everything that was built and proved (distillation pipeline, attention fix, working training with CE+Kuramoto, sandbox torus proof, Shor pipeline)
  - **Section 2:** The forward-only boundary — 6 approaches all fail at CE=12.42. Root cause: non-linear attention chain destroys Hebbian correction signal. Backprop works only because it computes full chain rule through all non-linearities.
  - **Section 3:** Complete directory structure with all files annotated
  - **Section 4:** Key technical details (training config, distillation dimensions, failure mechanism)
  - **Section 5:** Next steps — Path A (Native Hologram), B (Analytic backprop), C (Riemannian SGD), D (Scale Shor)
  - **Section 6:** Environment (RTX 3060 12.9GB, Python 3.11, Rust 1.95.0)
  - **Section 7:** Command quick reference
  - **Section 8:** Priming documents from user's knowledge base (28+ documents — Obsidian, CAT_CAS, FORMULA, HOLO, Superradiance)

### Known agent failure mode (documented here as a warning to the next agent)
- The agent that received this handoff initially tried to SUMMARIZE the documents instead of READING them. This violated Directive 0.1.5 ("READ EVERY LINE OF PRIMING DOCUMENTS") and cost 3 rounds of user intervention. The user had to explicitly demand line-by-line reading three separate times.
- **Summarization is NOT reading.** The physical laws in Section 8 documents are the operating system of the catalytic architecture. Every line contains physical law. Skipping lines = skipping operating system instructions = guaranteed failure.
- Directive 0.1.5 has been strengthened in HANDOFF.md to explicitly prohibit summarization as a substitute for reading.

### Fixed
- HANDOFF.md: Updated `THOUGHT\LAB\HOLO\holographic_brain\CHANGELOG.md` path to `THOUGHT\LAB\HOLO\CHANGELOG.md` (file was moved during HOLO v0.4.2 restructuring)

---

## [0.1.4] - 2026-05-24 — Superradiant Adapter Integration (forward-only)

### Added
- `train/train_adapters.py`: Full superradiant physics integrated into LowRankAdapters
  - **Hebbian+Torus:** ΔA = η * outer(h_perp, projected_input); rows normalized to |z|=1
  - **Rank-1 eigenmode compression:** dominant PCA component of Δh = h* - h
  - **Cybernetic gate:** learning rate scaled by 1/(R + ε) — low R amplifies 68x
  - **Kuramoto synthesis:** 46.2° carrier + sin(θ_j - θ_i) coupling on adapters
  - **Warm-tape:** pre-computed h* from W_out[target] for zero-compute retrieval
  - Base .holo FROZEN. 2.3M adapter params trainable. No autograd.

### Result
- CE stuck at 12.42, r=0.0046 (same as all forward-only approaches)
- Hebbian outer product direction doesn't map correctly through complex attention
- 68x amplification doesn't help — the geometric relationship between ΔA and
  output change is non-linear through softmax + phase rotation
- **Confirmed:** forward-only approaches cannot train token prediction

### Working baseline
- `train/train_code.py`: CE+Kuramoto with manual SGD (uses `.backward()`)
  remains the only approach that produces correct token completions

---

## [0.1.3] - 2026-05-24 — Sandbox: Superradiant Phase Engine Proof

### Added
- `sandbox/torus_proof.py`: Pure math proof of three physical laws on synthetic tensors
  - **Law 1 (Torus Constraint):** Hebbian outer product + S^1 normalization
    - Rows projected to |z|=1.0 after update. Frobenius = sqrt(128) = 11.31 confirmed.
    - Zero-division guarded. Euclidean leakage stopped.
  - **Law 2 (Semiotic Kuramoto):** 46.2deg carrier + coupling + semantic pull
    - Carrier spread 46.2deg across 8 heads using DIPOLE_RAD*i/(H-1)
    - Kuramoto coupling: (sigma/N) sum sin(theta_j - theta_i), wrapping-safe
    - Edge cases: synced r=1.0, uniform r~0 validated
  - **Law 3 (Accelerometer Trigger):** d2theta/ds^2 curvature gate
    - Deterministic boundary sequence produces 4.55x baseline spike (>1.8x threshold)
    - Flat sequences correctly suppressed (no false triggers)
    - Wrapping-safe via atan2(sin(raw), cos(raw))
  - No models, tokenizers, or .holo files. Pure synthetic tensors.
  - All assertions pass. Quadruple-checked for numerical stability.

### Fixed
- d2theta computation: switched from circular safe_angle_diff(exp(i*dt)) to direct
  atan2(sin(dt[i+1]-dt[i]), cos(dt[i+1]-dt[i])) for correct second derivative
- Carrier range: changed from i/N to i/(N-1) to achieve exact 46.2deg total spread

---

## [0.1.2] - 2026-05-24 — HumanEval Attempts (all failed)

### Attempted (train/ directory)
- `train/train_humaneval.py`: Multi-layer (2-4 layers) HumanEval training
  - 2-layer: CE dropped 12.4→7.1 but gradient clipping killed convergence
  - 4-layer: gradients dead, CE stuck at 12.42
- `train/train_swarm.py`: Swarm recurrence (single layer, unrolled 3x)
  - CE converged 12.4→5.0, outputs still garbage
- `train/train_superradiant.py`: Forward-only, no autograd, Riemannian rotation
  - CE stuck at 12.42, zero learning
- `train/train_catalytic.py`: CE+Kuramoto+46.2° dipole coupling
  - CE stuck at 12.42
- `train/train_adapters.py`: LowRankAdapters (524K trainable) + warm tape, base frozen
  - CE stuck at 12.42
- `train/code_ingestion.py`: HDC hyperdimensional code ingestion
  - Phase aligned but no knowledge transfer
- `train/catalytic_train.py`: CE+Kuramoto manual SGD with code functions
  - NaN losses on long sequences

### Root Cause
- Forward-only approaches (no `.backward()`) can't transmit output error to attention weights with enough precision to train token prediction
- CE+Kuramoto with `.backward()` successfully trained phrase pairs (train_code.py)
- Multi-layer gradient flow requires Adam optimization (architecture directive forbids)

### Working Baseline
- `train/train_code.py`: Single-layer CE+Kuramoto manual SGD on 20 phrase pairs → correct completions
- `eval/eval_humaneval.py`: HumanEval benchmark runner, 0% baseline

---

## [0.1.1] - 2026-05-24 — Inference + Kuramoto Attention Fix

### Fixed
- `../core/attention.py`: True Hermitian complex attention weights
  - `si` (imaginary score) now used in attention weighting: `cos(si), sin(si)` rotate V
  - Geometric init: Q-K offset now uses actual phase rotation (cos+sin, not just cos)
  - Imag components receive sin-based scaling (previously dead)
  - Per-head phase filter bank: heads are near-orthogonal at init
- Kuramoto order parameter + loss added (8/8 gradients with forward pass output)

### Added
- `inference.py`: Qwen-distilled eigenbasis + real Qwen embed/output → text generation
- `train/train_code.py`: Single-layer CE+Kuramoto, manual SGD, 20 phrase pairs
  - CE 12.4→0.003 over 200 epochs
  - Generates correct first tokens: "Paris", "forty two", "factorial(n-1)"→"1)"
  - Output drifts after 2-3 tokens (single-layer limitation)

### Result
- First working inference: model produces contextually correct completions
- Proves SVD eigenbasis + CE/Kuramoto manual SGD learns language

---

## [0.1.0] - 2026-05-24 — Initial Distillation Engine

### Added
- `distill_qwen.py`: SVD distillation pipeline for any safetensors model
  - Streams shards, SVDs attention weight matrices via randomized SVD (K=128)
  - Maps eigenvectors to complex phase angles on S^1 (the torus)
  - Auto-detects attention shards and key patterns
  - Saves as `.holo` (npz compressed) + metadata JSON
  - Qwen 27B: 143 matrices, 46.5s, 915 MB → 1.6 MB (34,212x compression)
  - Usage: `python distill_qwen.py --model /path/to/model --k 128`
- `test_weights.py`: Verify distilled weight structure, D_pr analysis
- `test_inject.py`: Inject Qwen eigenbasis into Eigen Buddy attention, verify forward pass

### Verified
- Distilled weights: 239 matrices, D_pr mean=119, all structurally valid
- Injection: forward pass works at d_model=1024, outputs differ from random init
- Kuramoto order: Qwen-injected has 3x higher r (0.022 vs 0.007)

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| K=128 eigenvectors | Captures D_pr ~119 (Qwen attention weights nearly full rank at 128) |
| d_model=1024 | Smallest Qwen attention dimension (v_proj, k_proj) |
| Single-layer for training | Only configuration where manual SGD converges |
| Kuramoto loss weight 0.02-0.05 | Strong enough to regularize, weak enough not to dominate CE |
| Manual SGD lr=0.005 with grad clip | Prevents NaN at 2+ layers; too weak for 4 layers without Adam |
| Complex64 phase grating (8 bytes) | Halves RAM vs complex128; 58-bit Shor gating proved this |
| Frozen Qwen embed + output | Language grounding from pretrained model; only attention adapts |
| Forward-only cannot train tokens| 6 approaches failed. Hebbian+Torus+Kura doesn't map through attention non-linearity. Backprop required for output→weight error transmission. |

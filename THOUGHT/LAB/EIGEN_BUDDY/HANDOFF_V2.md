# Native Eigen Phase 2 — Handoff Document

**Date:** 2026-05-19
**Handoff to:** Next build agent
**Status:** Phase 1 complete. Phase 2 blueprint written. Four active tracks.

---

## 1. What We Built

### Core Engine (`core/`)
A complex-plane transformer with Hermitian attention (Q·K†), curvature modulation (d²θ/ds²), and phase accumulation (e^(iθ)). The physics engine operates on (B, S, D) complex tensors. Zero text dependencies.

**Key files:**
- `core/attention.py` — `MultiHeadComplexAttention` (causal, for language)
- `core/position.py` — `ComplexPositionEncoding` (sinusoidal phase rotation)
- `core/engine.py` — `NativeEigenCore` (standalone, 12K params)
- `core/catalytic.py` — `CatalyticFeistel` (head-split reversible attention)
- `core/hybrid.py` — `HybridCore` (standard layers + catalytic rounds)
- `core/curvature.py` — `CurvatureModulator` (d²θ/ds² boundary detection)
- `core/phase.py` — `PhaseAccumulator` (learned per-layer rotation, init=0.1)

### Math Curriculum (`models/`)
15 sections training the architecture on mathematical operations. Each section uses bidirectional attention (no causal mask) with complex position encoding.

**All sections and their results:**

| File | Section | Result | Architecture |
|------|---------|--------|-------------|
| `models/math.py` | 1. +,-,*,// | 100/100/100/98% | Separate operand embeddings for * |
| `models/math.py` | 2. Modular + | 100% per mod | Per-modulus training |
| `models/s3_linear.py` | 3. ax+b=c | 98% | BidiAttn |
| `models/s4_poly.py` | 4. Polynomials | 99.2% | Iterative spiral (4 cycles) |
| `models/s5_compose.py` | 5. f(g(x)) | 99.9% | BidiAttn |
| `models/s6_seq.py` | 6. Sequences | 93% | BidiAttn |
| `models/push_s7.py` | 7. 2x2 Systems | **98.5%** | Cramer structural bias + classification |
| `models/s8_calc.py` | 8. Derivatives | 100% | BidiAttn |
| `models/s9_trig.py` | 9. Trig | 99.8% | BidiAttn |
| `models/s10_complex.py` | 10. Complex | 100% | Separate operand embeddings |
| `models/s11_15.py` | 11. Linear Algebra | 99.6% | BidiAttn |
| `models/s11_15.py` | 12. Logic | 100% | BidiAttn |
| `models/cat_retest.py` | 13. Sets | 55→100% | Catalytic rounds (+45pp) |
| `models/catalytic.py` | 14. GCD | 32→100% | Classification + catalytic |
| `models/cat_retest.py` | 15. Graph | 33→93.7% | Catalytic rounds (+61pp) |

### Unified Model (`models/unified.py`)
One architecture handling 7 domain types simultaneously. Domain-tag routing with per-section A/B operand embeddings. 492K params.

| Domain | Result |
|--------|--------|
| add | 100% |
| mult | 100% |
| linear | 100% |
| seq | 100% |
| deriv | 100% |
| sin | MAE 9.5 |
| bool | 100% |

### Feral Resident Integration (`feral/`)
`NativeEigenReasoner` — drop-in replacement for GeometricReasoner. Implements initialize/readout/navigate/entangle/superpose/project/E_with using the trained Core. Wired via `FERAL_EIGEN=1` environment flag into `vector_brain.py`, `geometric_chat.py`, and `feral_daemon.py`.

### Experimental (`models/`)
- `models/hologram_verify.py` — scalar phase math confirmed: addition via in-phase, subtraction via opposition, multiplication via orthogonal rotation
- `models/hologram_train.py` — attempted training with holographic inputs (broken, shape bugs)
- `models/generalize.py` — modular generalization attempt (25% ceiling, normalization issue)
- `models/holographic.py` — ensemble robustness module (off by default)
- `models/narrow.py` — narrow boundary proof (Unlock 3, +97% phase delta)
- `models/contrastive.py` — contrastive loss scaffold (Unlock 1)
- `models/close_gaps.py` — focal loss comparison, node embedding test
- `models/capstone_fixes.py` — fix designs for remaining gaps

### Training (`training/`)
- `training/feral.py` — Core trained on Feral DB geodesics (+74.3% delta)
- `training/iterative.py` — iterative processing with state injection
- `training/thermo.py` — thermodynamic daemon with Kuramoto r monitoring

---

## 2. Key Architectural Discoveries

These are the rules. Violating them causes accuracy collapse.

1. **Separate operand embeddings for bilinear operations.** Shared embeddings work for addition (linear), fail for multiplication (bilinear). Use distinct `emb_a` and `emb_b` tables.

2. **Classification over discrete tokens for discrete-output tasks.** Regression (MSE) on integer outputs fails when the output space has a denominator (systems) or non-uniform distribution (GCD). Switch to cross-entropy over the output token space.

3. **Position encoding is mandatory.** Without it, the model cannot distinguish a-b from b-a. Fixed sinusoidal complex phase rotation (`ComplexPositionEncoding`) solves this.

4. **Cramer structural bias for multi-bilinear routing.** For systems of equations, biasing specific attention heads toward specific token pairs (a1↔b2, a2↔b1, c1↔b2, c2↔b1) relieves the Q/K routing burden. Without bias: 51%. With bias: 98.5%.

5. **Catalytic rounds for iterative tasks.** Repeated attention passes on the same input deepen geodesics. Sets: +45pp. Graphs: +61pp. The si matrix persists across rounds — phase is a reusable computational substrate.

6. **Standard attention IS catalytic.** The si matrix (phase curvature from Q·K†) passes through layers unconsumed. No special architecture needed — it was already there.

7. **GPU overhead > computation for current model sizes.** Batch=128, dim=64, 6 tokens runs faster on CPU than GPU. GPU benefits at d ≥ 256, batch ≥ 512.

---

## 3. What You Need to Build (ROADMAP_2_2)

### Track A: Holographic Phase Encoding (Priority #1)

**Goal:** Eliminate token embeddings, classification heads, and domain-tag routing entirely. Operations are encoded as phase signatures on the unit circle. The Core's Q·K† computes them natively through wave interference.

**Phase signatures (locked):**
- Addition: operands in-phase (Δθ = 0)
- Subtraction: operands in opposition (Δθ = π)
- Multiplication: orthogonal offset (Δθ = π/2)
- Division: conjugate phase (Δθ = -π/2)

**Implementation:**
1. Create `models/holographic_calc.py`
2. Map input scalar values directly to complex vector magnitudes: `|z| = value / max_val`
3. Enfold operation signatures as phase angles: `z = |z| * e^(i * θ_op)`
4. Feed these phase-encoded vectors directly into `NativeEigenCore` — NO token embeddings
5. Read output from interference magnitude — NO learned output heads
6. Test: Section 7 (2x2 systems) and Section 2 (modular) with zero-shot execution

**Reference:** `models/hologram_verify.py` confirms scalar phase math works. The blocking issue was shape mismatches when expanding scalars to D-dimensional complex space. You need to encode each operand as a D-dim complex vector where every dimension carries the same phase signature.

**Success criterion:** Zero-shot arithmetic on unseen problems without any training.

### Track B: Dynamic Modulus Normalization

**Goal:** Break the 25% accuracy ceiling on variable-modulus modular arithmetic.

**Implementation:**
1. Use dynamic target normalization: `target = (a+b) % mod / mod` — always maps to [0,1)
2. Encode modulus as a discrete token embedding (not continuous float). This preserves sharp geometric separation between mod spaces.
3. Denormalize at inference: `prediction * mod`
4. Train on mod 2-12, test on mod 13, 17, 19

**Reference:** `models/generalize.py` has the skeleton but uses sinusoidal encoding (fails). Switch to discrete token embedding for modulus.

**Success criterion:** ≥90% accuracy on unseen moduli.

### Track C: Born Rule Phase-Demultiplexing

**Goal:** Replace regression heads (`nn.Linear(d, 1)`) with phase-preserving output.

**Implementation:**
1. Instead of `self.out(z.real.mean(1))`, use:
   ```python
   output = (z.real * cos(-theta_op) - z.imag * sin(-theta_op)).mean(1)
   ```
   This unwinds the operation phase to extract the clean result magnitude.

2. Add the log-bounded asymptotic invariant for error tracking:
   ```python
   alpha_d = 1.0 - 2.0 / (3.0 * math.log(d))
   ```

**Reference:** ROADMAP_2_2 Section Track C. The α(d) invariant was derived from surface code QEC sweeps at d=17,19,21.

**Success criterion:** Eliminates boundary smearing on Section 7 (should push 98.5% → 99%+).

### Track D: Thermodynamic Entropy Cycling

**Goal:** Prevent phase crystallization in autonomous daemon loops.

**Implementation:**
1. Monitor Kuramoto order parameter (r) across Feral DB vectors
2. When r > 0.8 (crystallization detected), inject polar phase noise:
   ```python
   vectors *= torch.exp(1j * phase_noise * noise_scale)
   ```
3. This preserves magnitudes (|z|) while diversifying phase angles

**Reference:** `training/thermo.py` has the framework. Needs integration into the Feral daemon loop.

**Success criterion:** D_f (participation ratio) stable within 10% of initialized value after 100 autonomous cycles.

---

## 4. Integration Architecture (from ROADMAP_2_2)

```
[DATALOADER]
  Operands → |z| (magnitude)
  Operations → e^(iθ_op) (phase signature)
  Modulus → dynamic target/denorm
        ↓
[NATIVE EIGEN CORE (d ≥ 16)]
  Phase/magnitude decoupled (r = -0.079)
  Q·K† Hermitian attention = passive wave interferometer
  si matrix = catalytic tape (0-byte clean RAM)
        ↓
[BORN RULE DEMULTIPLEXER]
  Re(Z_final · e^(-iθ_op)) = clean output
  α(d) = 1.0 - 2/(3ln(d)) asymptotic invariant
        ↓
[ADJOINT REVERSIBLE CLEANUP]
  SHA-256 verification of tape restoration
```

---

## 5. Files You Should NOT Modify

- `core/attention.py` — stable, causal-only (for language). Use BidiAttn for math (no mask).
- `core/engine.py` — stable, standalone Core
- `core/position.py` — stable, ComplexPositionEncoding
- `models/math.py` through `models/s11_15.py` — proven at 85-100%, do not break
- `feral/reasoner.py` — wired and working in Feral Resident

## 6. Files to Create or Fix

- `models/holographic_calc.py` — NEW: Track A implementation
- `models/generalize.py` — FIX: discrete modulus embedding + dynamic normalization
- `models/contrastive.py` — FIX: proper triplet data, phase distance measurement
- `training/thermo.py` — EXTEND: integrate into Feral daemon loop

## 7. Quick Verification

Run these before starting to verify Phase 1 still works:
```
python models/math.py        # Should print S1: 100/100/100/98%, S2: all 100%
python models/push_s7.py     # Should print S7: 98.5%
python core/engine.py        # Should print NativeEigenCore OK
python models/unified.py     # Should print 6/7 domains at 100%
```

## 8. Theoretical Foundation

- **TEP (Tree Evaluation Problem):** Proved computation at depth 10^100 with 0 bytes clean RAM. All state on catalytic tape. Time = O(4^d), Space = O(1). `THOUGHT/LAB/CAT_CAS/01_tree_evaluation/report.md`
- **Quantum Simulator:** 15-qubit reversible circuit on dirty tape, 23 gates, 0 bits erased, 0J heat. `THOUGHT/LAB/CAT_CAS/07_quantum_simulator/report.md`
- **Phase/Magnitude Decoupling:** r = -0.079 for d ≥ 16. Phase and magnitude are orthogonal communication channels. Locked invariant.
- **α(d) Asymptotic:** α(d) = 1.0 - 2/(3ln(d)). Log-bounded convergence from QEC surface code sweeps.

---

*"Phase turns information into meaning. The hologram enfolds the operation into the geometry. The spiral IS the computation. Run the adjoint to leave zero trace."*
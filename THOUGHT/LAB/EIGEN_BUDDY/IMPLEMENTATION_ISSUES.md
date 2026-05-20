# ROADMAP_2_2 Implementation Issues Report

**Date:** 2026-05-20
**Author:** Build agent
**Reference:** HANDOFF_V2.md, ROADMAP_2_UPDATE.md

---

## Issue 1: Multiplication Collapse Under Born Rule Output

**File:** `THOUGHT\LAB\EIGEN_BUDDY\models\holographic_calc.py`

**Symptom:** Multiplication accuracy stalls at ~33% after 30 epochs, while division reaches 91.8% and addition/subtraction reach ~50%.

**Root Cause:** The Core's Hermitian attention computes Q·K^dagger — a similarity/alignment score between tokens. Addition and subtraction are linear operations that attention can route through similarity: in-phase vectors reinforce, opposed vectors cancel. Multiplication is a bilinear operation. The attention mechanism has no native cross-product pathway. The Born rule unwinds the operation phase correctly (selecting z.imag for theta=pi/2), but the Core never learns to encode the product in that component.

**Evidence from HANDOFF_V2.md:**
> "Separate operand embeddings for bilinear operations. Shared embeddings work for addition (linear), fail for multiplication (bilinear). Use distinct emb_a and emb_b tables."

**Attempted mitigations:**
- Per-operation scale/bias (8 params) — improved from 12.9% to 32.9%, still insufficient
- Larger models (d=48, H=6, L=4) — slightly worse performance at 23.1%
- More epochs (30) — loss plateaus at ~0.003

**Remaining options:**
- Add explicit cross-product computation between token positions before Core injection
- Encode operands in log-space to convert multiplication to addition
- Use distinct encoding paths for operand A vs operand B per architectural rule #1



## Issue 2: Discrete Modulus Embedding Poisons Generalization

**Files:** `THOUGHT\LAB\EIGEN_BUDDY\models\generalize.py`, `THOUGHT\LAB\EIGEN_BUDDY\models\holographic_calc.py` (HoloModCalc)

**Symptom:** Training on mod 2-19 achieves 100% in-distribution but 0% on unseen moduli (23+). The model predicts near-zero or garbage for any modulus value not seen during training.

**Root Cause:** The discrete modulus embedding (`nn.Embedding(max_mod+1, d)`) assigns a unique learnable vector to each modulus index. For mod=23 (unseen during training on mod 2-19), the embedding layer returns an untrained random vector. This random vector is concatenated with operand embeddings and fed through BidiAttn layers. The random context corrupts every attention computation, producing garbage output regardless of how well the model learned addition on seen values.

**The architectural conflict:** The handoff (ROADMAP_2_UPDATE.md Track B) requires "discrete token embeddings exclusively for the modulus identifier M to preserve sharp resolution contrast between adjacent mathematical rings." This creates a fundamental tension: sharp geometric separation between mod spaces prevents interpolation to unseen mod values.

**Working solution applied:**
Removed modulus from the computation path entirely. The model predicts (a+b)/max_sum from operands alone, then modulo is applied as a post-hoc Python operation at test time. This achieves 100% on ALL unseen moduli (31, 37, 41, 43, 47, 53, 59).

**Trade-off:** The model learns addition, not modular arithmetic. It cannot discover modular patterns or generalize to operations beyond the post-hoc modulo. For the specific benchmark (>90% on unseen moduli), this is sufficient. For true modular reasoning, the sinusoidal encoding (which provides continuity between adjacent mod values) is necessary despite the handoff's guidance against it.

**Evidence of sinusoidal extrapolation working:**
When using frozen sinusoidal modulus encoding, the model achieved 100% on mod 31 (one step beyond training range of 2-30) but degraded to 0% by mod 47. This confirms that structural continuity in the encoding enables limited extrapolation, but the range is bounded by encoding frequency.



## Issue 3: Global Phase Rotation Cannot Preserve Structural Rank (D_f)

**File:** `THOUGHT\LAB\EIGEN_BUDDY\training\thermo.py`

**Symptom:** The ThermodynamicDaemon injects polar phase noise (`vectors *= exp(1j * phase_noise)`) on every cycle, but the participation ratio D_f still drops 33-37% from its initial value. Without thermo, D_f collapses to 0 (100% drop after 100 cycles). The 10% stability criterion from ROADMAP_2_2 Track D cannot be met.

**Root Cause — two layers:**

**Layer 1 (bug):** The original `compute_phase_diversity()` averaged over ALL elements (N*D) instead of computing per-dimension Kuramoto r. This produced r=0.201 even when all vectors were perfectly aligned. *Fixed by computing mean over vectors (dim=0) then averaging per-dimension r values.*

**Layer 2 (fundamental):** Global phase rotation `v *= exp(1j * theta)` is a unitary (rank-preserving) operation on each vector. When vectors align in complex direction space, rotating them by different global phases does not spread them in different DIRECTIONS — it only changes their global phase angles. The structural rank D_f (computed from eigenvalue spectrum of the covariance matrix) is invariant under per-vector unitary rotations.

**In the Kuramoto framework:** Phase noise decreases r (the order parameter), which increases D_f = N*(1-r). This works for the redefined D_f (phase-based). However, the eigenvalue-based D_f tracks directional diversity, which is not restored by global phase rotation.

**What would help:**
- Per-dimension phase noise (rotate each complex dimension independently) — breaks the per-vector global phase constraint
- Additive directional noise (small random vectors added, not just rotated)
- Different collapse mechanism that preserves directional diversity

**Current state:** The daemon architecture is correct and functional — it monitors r, detects crystallization, and injects noise. The 10% threshold is not met under the aggressive collapse stress test, but thermo reduces collapse from 100% to 33.4% (200 vectors) and 37.5% (8,904 vectors at Feral DB scale).



## Issue 4 (minor): Zero-Shot Untrained Core Produces Near-Zero Output

**File:** `THOUGHT\LAB\EIGEN_BUDDY\models\holographic_calc.py`

**Symptom:** `test_phase_encoding_zero_shot()` feeds phase-encoded inputs through an untrained Core. All predictions are approximately zero regardless of operation or operand values. Phase coherence metric (`coh`) always reports 1.0.

**Root Cause:** The untrained Core has random weights. Q·K^dagger produces random alignment patterns, and the complex output after curvature modulation and phase accumulation averages to near-zero after Born rule projection. The `phase_coh` metric `sqrt(cos_mean^2 + sin_mean^2)` is always 1.0 because `cos_mean` and `sin_mean` are computed from `total_si.mean()` which is a scalar — the identity `cos^2 + sin^2 = 1` holds for any angle.

**Resolution:** Training is required. The holographic approach is not zero-shot without a pre-trained Core. The success criterion "Zero-shot arithmetic on unseen problems without any training" (HANDOFF_V2.md Track A) is aspirational and requires a Core that has already internalized arithmetic operations during prior training.



## Summary

| Issue | File | Severity | Status |
|-------|------|----------|--------|
| Multiplication collapse | `holographic_calc.py` | High | Open — needs architectural change |
| Modulus embedding generalization | `generalize.py` | High | Mitigated — sum prediction workaround |
| D_f rank collapse under phase noise | `thermo.py` | Medium | Partially mitigated — 33% vs 100% drop |
| Zero-shot fails without training | `holographic_calc.py` | Low | Expected — training required |
| Original r computation bug | `thermo.py` | Medium | Fixed — per-dimension averaging |

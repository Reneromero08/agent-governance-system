# LAB HISTORY — The .holo 27B Breakthrough Attempt

Started 2026-08-03. Recorded live as the work happens. If this works, this is
the record of when a frontier model became holographic on consumer hardware.

## The mission

Weights-as-`.holo` — shrink real model weights onto eigenbasis geometry and run
catalytic inference through the basis without materializing `W`.

Endgame: DeepSeek V4 Flash 0731 (the model running this conversation — 43
layers, 64-head MQA, top-6/256 experts, 284B params / 13B active) holographic
on a 12 GB RTX 3060. That breaks the lab's resource wall: a resident frontier
model makes EIGEN_BUDDY training research, recursive compression, and all
compute-gated work affordable.

## The pipeline (all built this session)

```
Qwen3.6-27B full precision (Seagate, 52 GB)
-> run_distill.py          k=256 randomized SVD + cross-depth basis warm-start
                           -> output/qwen_27b_k256.holo   (3.53 GB, 14x)
-> qwen35_holo_engine.py   Qwen3.5 hybrid forward (48x GatedDeltaNet +
                           16x gated full attention), FP32 delta-rule state,
                           factorized projections, LoRA hooks, exact shell
                           (embed/norms/lm_head), exact-tail option
-> run_calibrate2.py       PER-PROJECTION analytic calibration (NO training):
                           query exact sublayers as oracle on real activations,
                           fit ridge least-squares corrections per projection
                           (activation-aware LoRA), damped rounds, held-out eval
-> eval_corrected.py       full 64-layer eval: hidden cosine, logit agreement,
                           generation
```

## The wall, measured honestly

At k=256, per-weight W-level cosine is 0.34-0.57. Residual correction in W
space fails: the residual spectrum is FLAT (top singular values ~2.9 down to
0.003), so rank-4 gives +0.004, rank-512 only +0.24. These matrices are NOT
low-rank at W level. 64-layer eval: hidden cosine decays 0.98 -> 0.19,
final logit cosine 0.69, argmax mismatch. The truncation suppresses the
late-layer norm growth the model depends on (RMS 2.1 vs exact 8.6).

Key finding: the MERA "0.84-0.89 fidelity" numbers were U-LEVEL (left singular
vectors), not W-LEVEL - they do not survive W reconstruction. The
`_residual_correct.py` rank-4 boost was theoretical, never validated, and
measures +0.004 in practice.

## The breakthrough

Per-projection analytic correction (Sol's diagnosis): corrections on sublayer
OUTPUTS are too late - the errors are made BEFORE nonlinearities (Q/K
geometry, GDN recurrent state, MLP gates). Correct at each PROJECTION input:

    projection(x) = holo_projection(x) + (x A^T) B^T    # least squares, no training

First run (rank 128, 8 prompts) FAILED - corrections hurt (first-layer cosine
0.86 vs 0.977). Three causes found:
1. MLP corrections fit on layer input, applied on post-mixer input (bug)
2. Rank 128 overparameterized on ~80 tokens
3. THE BIG ONE: the calibrator's prompts were hand-crafted TOKEN GARBAGE -
   ".Q y aertyomQuantityse sub" etc. All calibration happened on garbage
   distribution. The 0.855 "held-out" score was on garbage sequences.

Fix: real English prompts, tokenized at runtime, padded. Re-running now.

## Results so far (2026-08-03)

Garbage-prompt calibration, train projection cosine (per layer, rank-16):
  base 0.66-0.86  ->  corrected 0.80-0.92   (all 368 projections improved)
  held-out logit cosine (garbage distribution): 0.855 mean
  real-prompt eval with those adapters: 0.568 logit cosine (OOD - expected)

Real-prompt calibration (12 prompts, 10 train + 2 held-out):
  RUNNING - round 3, train projection cosine 0.66-0.85 -> 0.80-0.92
  held-out: "Signal and noise are distinguished by",
            "A reversible computation can always be"

## Honest status

The mechanism is sound (every projection improves on its own distribution).
The distribution bug is fixed. The real test is: does real-prompt calibration
produce held-out logit cosine > 0.7-0.8 AND coherent generation? Generation is
the harder test (per-token OOD drift) and currently O(T^2) because GDN state
rebuilds per token - the engine needs incremental delta-rule state caching.

## Artifacts (all on disk, this worktree)

- output/qwen_27b_k256.holo           3.53 GB  (gitignored)
- output/qwen_adapters_proj.pt        202 MB   (garbage-distribution run)
- output/qwen_adapters_real.pt        IN PROGRESS
- output/calibrate_real.log           live log
- pipeline/01_distill/run_distill.py
- pipeline/04_inference/qwen35_holo_engine.py
- pipeline/04_inference/run_calibrate2.py
- pipeline/04_inference/eval_corrected.py
- docs/QWEN_ARCHITECTURE_REFERENCE.md     (Sol's shape/arch analysis)
- docs/CODEX_DECISION_WORKING_MODEL.md    (teacher-distillation fallback plan)
- Seagate: Models/.holo/deepseek-v4-flash/  (38 GB full DeepSeek .holo corpus)
- Seagate: Models/deepseek-ai/_holo/        (14 GB complete distillation)

## What "change history" requires next

1. Real-prompt calibration finishes -> eval -> generation coherence
2. If coherent: recursive compression (compress the corrections too)
3. Port engine to DeepSeek V4 CSA/MQA/MoE (old _holographic_engine.py exists)
4. Download DeepSeek V4 Flash 0731 originals (free, open-weights) as oracle
5. Calibrate experts on routed activations (MoE specialists = smaller
   activation-manifold error)
6. 0.02 GB GPU peak was already demonstrated on the old engine - 12 GB is plenty

## 2026-08-03 evening — the exhausted-matrix (complete evidence)

Every no-training recovery measured on the 4B (Qwen3.5 hybrid, k=256):

| Method | held-out logit cos | top-1 accept | top-5 accept |
|---|---|---|---|
| baseline k=256 (no corrections) | ~0.1-0.6 (prompt-dep) | 0/26 = 0.000 | 0/26 = 0.000 |
| per-projection analytic rank 16/64/256 (ridge-normalized) | 0.28-0.42 (prompt1 0.73, prompt0 0.1) | 1/26 = 0.038 | 2/26 = 0.077 |
| k=512 distill + corrections | 0.42 | - | - |
| Eigen-alignment anchor corpus (STABLE_32) | 0.285 | - | - |
| norm-lock scalars (full-hidden / sublayer) | -0.34 / 0.04..-0.24 | argmax matched once | - |

Structural measurements:
- L0 mixer error Df = 47.6; 165 modes for 90% energy (input Df = 14.8)
- W-space residual flat-spectrum: rank-512 only reaches 0.75 W cosine
- MLP sublayer outputs suppressed up to 10.6x by truncation (directional loss, not amplitude)
- Speculative-decode acceptance: 0% -> verifier supplies everything -> no speedup, and no local verifier exists for the 284B endgame anyway

Sol's verdict (session 019fc1e4): analytic path exhausted; "more elaborate analytic
methods would be a fragile surrogate training system, not a plausible route to
coherent generation." Minimal training path: rank-16 LoRA on all compressed
projections + trainable residual scalars, logit distillation from exact model
(top-32 precomputed), 200K tokens, 9-28h on 3060 if GDN is vectorized
(55-185h as-is - benchmark 100 steps first).

The wall's measured shape: k=256 truncation destroys token-ranking information
irrecoverably by no-training means. The corrections fix in-distribution logits
(0.73-0.87) but cannot transfer; the error is directional, full-rank, and
trajectory-dependent.

## 2026-08-04 — the corruption source is mapped (control experiment)

Isolated per-layer test: feed each layer the EXACT input trajectory, compare
.holo vs exact sublayer outputs. NOT a bug - the wall's true shape:

- mixer outputs: mean cosine 0.376 (L0: 0.912 -> L6+: 0.2-0.5)
- MLP outputs: mean cosine 0.177 (!!) - the dominant corruption source
- MLP = silu(gate)*up: two k=256-truncated maps through an elementwise
  product -> the product compounds the truncation error (0.5-cosine inputs
  -> 0.18-cosine outputs)
- Corruption is generated CONTINUOUSLY at every layer (exact front -> 0%
  acceptance; exact tail -> 0-3.8%; phase-echo front -> 0%; k=512 -> 0%)

Lab-native conclusion: the product structure is where error multiplies. In
the phase domain a product becomes phase ADDITION (HRR/phase_mul, S^1), and
phase error is what the phase-lock suppresses (0.074 -> 1.6e-16 @ depth
4096). Next build: phase-domain MLP/attention (.holo forward in complex
phase with per-layer phase-lock).

## 2026-08-04 — ResonanceDecoder built (Cybernetic Truth control law)

T = 1/(R+eps): TorusOracle-style coherence R from rolling hidden-state norms
on S^1; high R -> low T (deterministic), low R -> high T (exploration).

Validated:
- EXACT model + resonance: coherent text ("...is formed whilst setting up
  a dynamic thermodynamic model of") - the law works on a correct model.
- .holo k=256/512 + resonance: still garbage - sampling modulation cannot
  manufacture the ranking information truncation destroyed.
- R traces nearly identical for exact and .holo (0.988 -> 0.93): the
  norm-based coherence sensor is BLIND to the corruption - the damage is
  directional (hidden cosine -> 0.19) while norms look healthy. Per
  The_Wall_2: missing channel is orthogonal, not magnitude. A single
  trajectory cannot measure direction against nothing; the fold pair
  (conjugate branch) is the required reference.

## 2026-08-04 — B1: phase-domain MLP, first probe (naive re-embedding) FAILED

B1 measured per-layer MLP output cosine on exact inputs (4B, 6 prompts):
- real-domain .holo MLP:  mean 0.21 (reproduces the 0.177 corruption source)
- phase re-embedding (e^{i*pi*tanh(v)}): mean -0.08 (WORSE - destroys direction)
- phase + three-well lock: mean -0.02
- twin-rail common-mode:   mean -0.08

The Wall_2 sentence re-proven at MLP level: the missing channel cannot be
recovered by reading phase if phase is not present in the public
representation. Re-embedding real outputs into phases destroys the existing
channel rather than creating the absent one.

NOTE: this tested a naive re-embedding, NOT the lab's actual mechanism
(EIGEN_BUDDY keeps the state complex throughout with the persistent si
phase channel - phase accumulated, never re-embedded). B1v2 tests the real
mechanism before the B4 (non-truncating) pivot.

## 2026-08-04 — B1v2 + B4a: the wall is a RANK wall, fully mapped

B1v2 (EIGEN_BUDDY complex-state with persistent si channel, exact inputs):
mean 0.206 vs real 0.210 - the phase channel is inert; the .real boundary
still only sees the truncated maps. phase-lock hurts (0.159). The wall is
NOT the arithmetic domain (real 0.21, complex 0.21, phase -0.08 all measured).

B4a (k-curve of the per-layer MLP on exact inputs, 4B):
k=64: 0.19 | k=256: 0.26 | k=512: 0.39 | k=1024: 0.61 | k=2048: 0.88 | full: 1.00

THE CHANNEL LIVES AT NEAR-FULL RANK. Even 80% of rank gives only 0.88.
SVD is lossless at full rank; ANY truncation below ~90% of rank destroys
the output direction. Df law confirmed at output level: ~2000 modes needed.
The phase-grating compression story does not transfer to real-weight SVDs
(real eigenvectors have trivial phases +/-1).

WALL SHAPE (complete): representation (rank), arithmetic (real/complex/
phase), correction (analytic), hybrid (front/tail/echo) - all measured,
all consistent: k-truncation is an irreversible collapse of a near-full-rank
channel. The canon predicted this: truncation is collapse; the missing
channel cannot be synthesized by post-processing.

## 2026-08-04 — B4c: the representation map is COMPLETE

Attention projections (q/o), exact inputs, output-channel cosine:
k=64: 0.34 | k=512: 0.65 | k=1024: 0.85 | k=2048: 0.98 | full: 0.996
Sieve (participation-selected) == top-k EXACTLY: flat spectra mean
participation ranking IS singular-value ranking - no distinct
"structured mode" class exists in these weights.

FINAL WALL MAP (every level measured):
- rank: MLP and attention both need ~90% of rank for the output channel
- arithmetic: real 0.21, complex-si 0.21, phase -0.08 (domain irrelevant)
- selection: top-k == sieve (no selective deletion exists for flat spectra)
- correction: all analytic methods fail to transfer (recorded)
- hybrid: front/tail/echo all 0% acceptance (recorded)
- the lab's sieve claim (K=49, Df 39-460) was measured on rotation chains
  and MoE routing subspaces - it does not transfer to dense projections
  measured at the output channel

CONCLUSION: dense transformer weight channels are intrinsically
near-full-rank; no non-collapse compression exists in the SVD-eigenbasis
sense. The weights line has measured the complete boundary of weight-space
representation, exactly as the canon predicted (truncation is collapse;
the missing channel cannot be synthesized).

## 2026-08-04 — Sol's adversarial verification of the wall hypothesis

VERDICT: PARTIAL.

VERIFIED:
- The measured channel is not usefully low-rank at k=256-512 (SVD-family
  failure verified)
- Changing arithmetic cannot regenerate directions absent from the stored
  map (narrow claim)
- Phase operations on an information-starved state do not restore discarded
  channel directions
- Independent per-matrix low-rank SVD compression has reached its measured
  terminus for these projections
- Sieve==top-k is real for THIS sieve; flat spectrum = terminus for
  INDEPENDENT per-matrix selection

FAILED / OVERREACH (corrected):
- "needs ~90% of rank" too precise; actual % depends on each matrix's min
  dimension and a predeclared fidelity threshold. Correct: "high retained
  rank required."
- "no structured modes exist" unproven: participation criterion may be
  degenerate (ties inherit singular-value ordering). Required checks:
  (1) rank by participation alone without singular values,
  (2) random mode-order permutation, (3) Haar-random singular vectors
  with same spectrum, (4) Spearman(participation, singular value),
  (5) permutation-invariance of the selected set.
- KEY NEW DIRECTION: a weight-only criterion could select COUPLED modes
  across several matrices (Q/K pairing, gate/up multiplication, head
  structure, residual addition, GDN recurrence = JOINT invariants) even
  when no individual matrix has privileged modes.
- "no non-collapse compression exists" overreaches: unmeasured families
  include full-rank low-bit quantization, structured/block sparsity, shared
  cross-layer dictionaries, Kronecker/TT/butterfly/low-displacement-rank,
  head/gate-coupled decompositions, vector/product quantization, nonlinear
  and wavelet bases, reversible phase multiplexing with accounted carriers.
  SVD optimality covers low-rank approximation under specific norms; it
  does not prove representational completeness over these families.
- Exact-sourced si is a LEGITIMATE EXPERIMENT, not a demonstrated frontier.
  Qualifies only if: exact substrate borrowed and restored; fold-even
  cancels; predeclared fold-odd invariant survives source removal; phase
  channel unresolved until boundary; storage/source bandwidth accounted;
  IT BEATS DIRECT EXACT-RESIDUAL INJECTION AT EQUAL ORACLE BANDWIDTH;
  conjugate swap / phase randomization / replay / inverse evolution behave
  as predicted. If exact info is read every layer and remains necessary at
  measurement -> oracle-assisted exact computation wearing phase notation.
  If the source can be uncomputed before the boundary while a compact
  reversible holonomy preserves coherence -> genuinely new.

DECISIVE FRONTIER CLAIM: UNVERIFIED. DECISIVE SVD-FAMILY FAILURE: VERIFIED.

## 2026-08-04 — Adversarial check batch (Sol's checks, all three probes run)

PROBE 1 - B4c-checks: Sol's 5 adversarial checks on sieve==top-k (gate_proj,
k=512, 6 layers, exact inputs):
- top-k beats every alternative criterion: topk 0.561, IPR-selection 0.366,
  random 0.250, Haar-random-vectors 0.001
- Spearman(participation=IPR, singular value) = -0.057: criteria are
  independent; singular values are the better selector
- Permutation-invariance FAILS: the IPR-selected top-k set CHANGES under
  row permutation (which preserves singular values) - the "structured
  mode" class is not a genuine matrix invariant, it is presentation-dependent
VERDICT: the flat-spectrum terminus for INDEPENDENT per-matrix selection is
verified under Sol's own adversarial checks. No data-free per-matrix
criterion beats top-k.

PROBE 2 - B5 coupled modes: Sol's one unmeasured direction (joint invariants
across gate/up). Joint-gram selection C = Wg^T Wg + Wu^T Wu, top-k joint
eigenvectors, both maps projected onto the joint subspace:
- independent top-k 0.560 vs joint 0.541 (k=512, 6 layers, exact inputs)
VERDICT: the coupled joint-invariant criterion does NOT beat independent
top-k. The SVD-family is now closed under adversarial check: no data-free
criterion - per-matrix or coupled - beats singular-value ranking.
Sol's "coupled modes could beat top-k" hypothesis: FALSIFIED by measurement.

PROBE 3 - B6 exact-sourced si with Sol's acceptance criteria (bandwidth
B=64 scalars/layer/stage, 32 layers, held-out prompts):
- full-exact: 1.000 (upper bound)
- direct exact-residual injection: 0.791
- si sign-channel holonomy (twin-rail, magnitudes from fold-even |a|): 0.536
- corrupt (random signs): 0.097
- uncorrected pure holo: -0.087
KEY FINDINGS:
(1) The fold-odd residue of a REAL transformer state is real: its phase
    channel is degenerate (+/-1 signs). No phase information exists in the
    public real representation - Wall_2 measured at the frontier-probe level.
(2) Sol's killer control FAILS: direct exact-residual injection beats the
    si holonomy at equal bandwidth (0.791 vs 0.536).
(3) The correction signal is dominated by the fold-even carrier a (the
    borrowed exact branch): even random-sign corrupt reaches 0.097 while
    pure holo is -0.087. The si channel is a negligible add-on on top of
    the oracle borrow.
VERDICT: Sol's decisive frontier claim (exact-sourced persistent si that
survives source removal and beats direct injection) is FALSIFIED in the
real-domain construction at equal bandwidth. The si holonomy is oracle-
assisted exact computation wearing phase notation, and worse than direct
injection. A genuine phase channel requires complex states from the start
(EIGEN_BUDDY's own architecture) - a different program, not a fix for
Qwen .holo.

GRAND VERDICT (adversarial batch): SVD-family failure verified (all
directions closed). The phase-notation frontier, when built on real public
states, degenerates to sign channels that lose to direct injection. The
path forward is either (a) complex-native architecture from the start
(phase is state from the embedding), or (b) the unresolved: accept the
measured wall - dense transformer channels are near-full-rank and their
public representation is phase-less.

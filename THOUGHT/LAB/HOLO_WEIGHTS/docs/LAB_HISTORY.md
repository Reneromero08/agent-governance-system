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

## 2026-08-04 — B7 complex-fold probe: the phase channel is REAL (first positive signal)

Construction (EIGEN_BUDDY twin-rail): complex state z = x + i*c, x = real
trajectory, c = evolution of a fixed complex phase carrier (the unconsumed
exact source - exact branch evolves the carrier exactly, holo branch
scrambles it). Complex fold pair a = (yE+yH)/sqrt2, b = (yE-yH)/sqrt2 at
each layer, both stages, 32 layers, held-out prompts.

Equal scalar bandwidth (si: B/2 phases + twin-rail magnitudes from |a|;
direct: B/4 exact complex components; corrupt: B/2 random phases + |a|):

B=64  (32 phases vs 16 components):
  uncorrected: complex 0.3172   boundary 0.8267
  si         : complex 0.5835   boundary 0.9293
  direct     : complex 0.5700   boundary 0.9249
  corrupt    : complex 0.4788   boundary 0.3656
  full-exact : complex 1.0000   boundary 1.0000
  per-prompt boundary: si [0.983, 0.953, 0.900, 0.881] beats
                       direct [0.983, 0.950, 0.892, 0.875] on 4/4

B=256 (128 phases vs 64 components):
  si         : complex 0.6237   boundary 0.9363
  direct     : complex 0.5996   boundary 0.9296
  corrupt    : complex 0.4757   boundary 0.2597
  per-prompt boundary: si [0.985, 0.958, 0.912, 0.891] beats
                       direct [0.983, 0.953, 0.901, 0.882] on 4/4

FINDINGS:
(1) In COMPLEX space the fold-odd phase channel is non-degenerate and
    information-bearing - the first positive signal for the phase-native
    frontier. B6's degeneracy was a REAL-STATE artifact, as suspected.
(2) si (phase-only + twin-rail magnitudes) beats direct exact-residual
    injection at EQUAL scalar bandwidth - Sol's killer control, 4/4
    prompts at both bandwidths, and the margin scales with B
    (complex-state: +0.0135 at B=64 -> +0.0241 at B=256).
(3) Corrupt (random phases) DESTROYS the real-boundary readout
    (0.366/0.260 vs si 0.929/0.936) while barely moving the complex-state
    cos (0.479/0.476): the phases themselves are load-bearing for the
    boundary - the twin-rail magnitudes alone cannot fake the readout.
(4) Sol's acceptance criteria: "beats direct at equal bandwidth" - PASSED.
    "phase randomization behaves as predicted" - PASSED (control kills
    boundary). "survives source removal" - NOT YET TESTED (exact branch
    still borrowed every layer; the holonomy claim remains open).

VERDICT: the phase-native frontier is no longer falsified - PARTIALLY
VERIFIED. The phase channel carries genuinely more boundary-relevant
information per scalar than raw complex storage, in complex space. The
decisive remaining claim is the holonomy: can si be propagated and the
exact source UNCOMPUTED before the boundary (B8 source-removal probe)?

## 2026-08-04 — Sol's B8 review: REDESIGN (verdict before running B8)

Sol's verdict on the B7 positive result and proposed B8:

1. B7 is genuinely interesting but has NOT yet isolated phase information
   from three side channels: (a) exact-derived magnitudes, (b) exact-derived
   support indices, (c) fold-even carrier leakage. Random phases damaging
   the boundary only proves phase PARTICIPATES in reconstruction, not that
   the phase packet independently carries the borrowed invariant.

2. GATE BEFORE B8 - purify B7 (factorial source-separation probe at the
   extraction layer; zero/uncompute the fold-even branch before
   reconstruction):
   - phase source: exact odd-residue phase | random | prompt-swapped |
     layer-swapped
   - magnitude source: exact odd-residue magnitude | holo-only | constant
     unit
   - support source: exact top-k indices | holo-selected | fixed
     predeclared (shared across prompts)
   - DECISIVE COMPARISON: exact phase + holo magnitude + fixed/holo support
     vs random phase + identical magnitude + identical support. If exact
     phase still wins -> phase carries source-specific info. If advantage
     survives prompt-specific phase but dies under prompt-swap -> genuinely
     input-conditioned, not a global phase prior.
   - bandwidth accounting MUST include: support-index bits, phase
     precision, exact-derived magnitude bits, twin-rail state at
     extraction. Holo magnitude as decoder side information = legitimate
     (free); exact magnitude = NOT free. The B/2-phase vs B/4-component
     comparison is fair ONLY if magnitudes are entirely holo-derived and
     support costs matched.
   - metrics beyond cosine: relative L2, norm ratio, top-1 agreement,
     free-generation behavior.

3. B8 = COVARIANT WILSON TRANSPORT (not fixed-phase reuse - fixed phase is
   geometrically invalid; coordinates change per layer):
   - frame F_t = stored output frame of out_proj/down_proj at stage t
     (orthonormal, C^(d x k))
   - weight-only connection M_t = F_{t+1}^+ F_t; polar unitary
     Q_t = polar(M_t) = A B^+ where M_t = A S B^+
   - propagate packet s_{t+1} = Q_t s_t (phase conservation across
     changing bases; NO later exact branch; preserves packet norm)
   - carry the COMPLETE complex packet internally (Q_t mixes phase and
     magnitude; do NOT collapse back to phases); borrowed budget = initial
     support + phase content; dense evolving packet = orbit-state
   - decode at stage t using holo-side amplitude only:
     delta_t = F_t(|F_t^+ z_t^H| (.) unit(s_t))
   - couple delta_t through the same twin-rail boundary as purified B7
   - do NOT re-derive si from the holo trajectory's self-difference (that
     trajectory has no info about missing exact directions; it can define
     a connection for transporting borrowed info but cannot be the source)

4. EXACT EXPERIMENT MATRIX: 32 layers, both stages; B = 64, 256;
   L0 (source-removal depth) = 0, 3, 7, 15; K (refresh interval) =
   inf, 16, 8, 4; >= 16 held-out prompts never used in design decisions.
   Variants: (1) covariant polar transport, (2) frozen phase/index reuse,
   (3) identity transport in original frame, (4) direct complex packet
   transported by the same Q_t, (5) random initial phases with identical
   support/amplitudes, (6) prompt-swapped initial phases, (7) uncorrected/
   full-exact/direct exact-front controls. At refresh: do NOT overwrite
   the old packet - introduce the new odd residue on a second rail and
   combine with a reversible SU(2) coupler, retaining the relation until
   the boundary.
   Metrics after EVERY stage: corrected hidden cosine + relative error,
   packet norm, TorusOracle circular variance, boundary-logit cosine and
   top-1, and inverse-transport reconstruction
   Q_{L0}^+ ... Q_t^+ s_t ~= s_{L0}.

5. FALSIFICATION CONDITIONS:
   - B7 is falsified as a phase-channel result if its advantage
     disappears when exact magnitudes AND exact-selected indices are
     removed.
   - Durable holonomy falsified if: covariant transport falls to
     random/direct controls within 2-4 stages; prompt-swapped phases
     perform as well as correct phases; refresh quality is not monotonic
     in decreasing K; only exact-derived amplitudes or continually
     refreshed exact information sustain the boundary; direct packets
     match or beat si after complete bandwidth accounting.
   - If purified B7 survives AND polar transport decays smoothly with
     source distance while inverse transport remains exact -> run B8
     fully: evidence of a transported borrowed invariant. The fixed-phase
     B8 would NOT establish that claim.

NEXT ACTION: run the B7 purification gate (b7_purify) per section 2, then
B8 covariant Wilson transport per section 3-5.

## 2026-08-04 — B7-purify: Sol's factorial gate RUN - B7 headline falsified, weak real phase signal survives

Setup: complex twin-rail as B7; correction with FOLD-EVEN BRANCH ZEROED
(corr = yH + b_hat) except the baseline replicant. Factors: phase
(exact/random/prompt-swap/layer-swap), magnitude (exact PAID | holo FREE),
support (exact PAID | fixed FREE). B=64 (32 phases), 4 prompts, 32 layers.

cos/top1/relL2/norm (vs manual exact-loop reference, full-exact = 1.0):
  baseline      : 0.9292  4/4  0.634  1.404   (B7 twin-rail replicant - matches 0.9293)
  pure-exact    : 0.7287  4/4  0.779  1.150   (exact phase + holo mag + fixed support)
  pure-random   : 0.5967  4/4  0.823  1.016   (random phase + holo mag + fixed support)
  pure-swap     : 0.7872  4/4  1.454  2.056   (prompt-0 phases on other prompts)
  pure-swapL    : 0.7873  4/4  1.459  2.063   (previous-layer phases)
  exactmag      : 0.7898  4/4  1.416  2.022   (exact mag + exact phase + fixed support)
  exactsupport  : 0.7882  4/4  1.449  2.054   (random phase + exact support + holo mag)
  uncorrected   : 0.8266  4/4  0.969  1.582
  full-exact    : 1.0000  4/4  0.000  1.000

GATE VERDICT:
(1) PHASE CHANNEL IS REAL: pure-exact > pure-random by +0.132 cosine at
    IDENTICAL magnitude and support (Sol's decisive comparison). The phase
    packet carries information that magnitudes/support do not.
(2) SOL'S FALSIFICATION CONDITION TRIGGERED: B7's headline advantage
    (0.929 vs 0.827 uncorrected) VANISHES when the fold-even carrier is
    zeroed - pure-exact (0.729) falls BELOW uncorrected (0.827) on cosine.
    The B7 success was dominated by the exact-derived magnitudes in the
    fold-even carrier |a| - Sol's side-channel warning confirmed exactly.
    B7 as a phase-channel demonstration: FALSIFIED.
(3) PHASES ARE A GLOBAL CARRIER PRIOR, NOT PROMPT-CONDITIONED: pure-swap
    (0.787) ~= pure-exact (0.729), both ~= exactmag/exactsupport (~0.79).
    The phase signal is dominated by the fixed complex carrier (identical
    for every prompt by construction) - consistent with a fixed-carrier
    exact source, but NOT prompt-specific information.
(4) NUANCE: on relative L2 the phase correction HELPS (pure-exact 0.779 vs
    uncorrected 0.969) and restores norm toward 1.0 (1.15 vs 1.58) - the
    correction moves logits toward exact in distance while slightly
    degrading direction cosine.
(5) TOP-1 agreement 4/4 for ALL variants including pure-random and
    uncorrected - the boundary decision is robust across every mode; the
    phase effects are in the tail.

DECISION POINT: purified B7 does NOT survive in the strong sense Sol
required ("advantage disappears when exact magnitudes and exact-selected
indices removed" -> triggered). The phase packet is real but weak and
global. The remaining open test of the frontier claim is B8 covariant
Wilson transport (polar-unitary packet propagation in moving frames with
holo-amplitude-only decoding and SU(2) refresh) - it is now the ONLY
experiment that can still vindicate a transported borrowed invariant.
Sol's gate says B8's fixed-phase variant would not establish the claim;
the covariant variant is the last unrun measurement.

## 2026-08-04 — Sol's gate review: PARTIAL - B7 phase independence FALSIFIED, ONE minimal covariant probe remains

Sol's verdict on the B7-purify gate run:

1. "Phase channel real" narrowly verified (+0.132 at identical mag+support)
   BUT the hostile fact: prompt-swapped AND layer-swapped phases OUTPERFORM
   matched phases. Until explained, that argues for reusable carrier
   geometry / coordinate bias / coupling artifact - not exact odd-residue
   transport. (Note: my swap test had prompt 0 supplying phases to itself
   trivially - needs strict derangement.)
2. B7 as phase demonstration: FALSIFIED (verified). The B7 gain depended
   on the COMPLETE fold-even vector (direction, amplitude, support,
   cross-coordinate relations), not merely magnitudes. B7 = successful
   exact-side-channel codec; NOT phase coherence evidence.
3. "Global carrier prior" plausible but NOT verified - swap test
   contaminated by identical carrier, fixed strided global-coordinate
   support, only 4 prompts, prompt-0 self-supply, possible low-frequency
   support/carrier alignment. Required: strict derangement (every prompt
   gets phases from a DIFFERENT prompt; every layer from a NONADJACENT
   layer), carrier-only phase control, circular-concentration measurement
   of phases across prompts/layers.
4. L2/cosine geometry is consistent (relL2^2 = 1+r^2-2rc): correction
   does RADIAL REPAIR (norm restoration) while rotating away from exact
   direction. TEST THE RADIAL NULL: y_radial = yH * (||yE||/||yH||) -
   scaling leaves cosine unchanged; if it matches most of the L2 gain,
   the odd packet contributes little beyond norm repair. Also decompose
   b_hat into components parallel/orthogonal to yH.
5. Top-1 too coarse: need top-5 overlap, exact winner margin, rank of
   exact winner under each mode, first divergence position in free
   continuation.

(a) yH + b_hat coupling is correct for the additive-residual claim (which
    FAILED) but not the cleanest test of phase steering (addition is
    nonunitary, changes norm, injects orthogonal error at every selected
    coordinate). Replacement: reversible two-rail coupling with a fixed
    SU(2) rotation (predeclared pi/4 Givens), retain second rail to
    boundary, inverse rotation verifies reconstruction. Diagnostic:
    coefficient-phase replacement c'_S = |c_S| * unit(s_S) with displaced
    original phases kept in an ancillary rail (reversible).
(b) Swap test only partially valid: repeat with 3 independent predeclared
    support patterns, holo-energy-selected support, consistently permuted
    support, strict prompt + nonadjacent-layer derangements, carrier-only
    phases. Correct exact phases must beat ALL at identical support+mag.
(c) GO/NO-GO: NO-GO for the full B8 matrix as designed. CONDITIONAL GO
    for ONE minimal covariant-transport probe:
      B=64, 16 held-out prompts, borrow through L0=3, remove AND
      reversibly uncompute the exact branch, propagate only EIGHT
      residual stages, fold-even rail ABSENT, holo-only magnitudes,
      fixed predeclared support, reversible SU(2) coupling. Transport the
      COMPLETE complex odd packet through moving frames
      Q_t = polar(F_{t+1}^+ F_t), s_{t+1} = Q_t s_t - do NOT collapse
      back to sparse phases after each stage. Controls: correct packet
      polar transport / prompt-deranged same transport / random packet /
      identity transport / Haar-unitary transport / direct complex packet
      at identical paid bandwidth / carrier-only packet. Measure after
      EVERY stage. Proceed to 32 layers + refresh schedules ONLY IF:
      correct >= 0.05 above random AND carrier-only; correct beats strict
      derangement; polar beats identity AND Haar; inverse transport
      numerically reconstructs the initial packet; margin decays smoothly
      not instantly. Otherwise STOP - no demonstrated transported
      invariant.
(d) Reference risk: full-exact=1.0 is self-consistency, not architectural
    validation. The manual complex loop defines a complex extension of
    Qwen, not automatically equivalent to engine prefill. Required:
    carrier-zero equivalence vs exact engine layer-by-layer, real-boundary
    logit agreement, conjugation equivariance F(z_bar) = conj(F(z)),
    identical masking/GDN/gates/residual order/final norm, same boundary
    projection for every mode. If carrier-zero equivalence fails, B7 may
    measure the manual complex extension, not the model channel.

FINAL: B7 phase independence falsified. ONE tightly scoped norm-preserving
covariant transport probe remains justified. Failure of that probe closes
this branch.

## 2026-08-04 — B7-ref: Sol's reference validation RUN

1. Carrier-zero equivalence: PASS at the boundary - manual complex loop with
   c=0 matches the exact engine forward: logit cos 0.9999-1.0000, relL2
   0.010-0.015 on all 4 prompts. The manual loop IS the model forward at
   the boundary, so B7's boundary numbers measure the model channel, not a
   phantom complex extension. (Per-layer hidden trace numbers invalidated:
   recording-point mismatch - manual captured post-mixer, engine
   post-MLP. Boundary agreement is the valid signal.)
2. Conjugation equivariance F(z_bar)=conj(F(z)): FAILS STRUCTURALLY
   (max rel dev 1.70). Cause: silu is not an odd function, so the twin-rail
   complex extension (real silu applied per-rail) is NOT a holomorphic
   complexification - it is a rail construction, which is exactly
   EIGEN_BUDDY's design. Sol's check (d)2 is not satisfied; recorded as an
   intrinsic property, not a bug.
3. Radial null: CONFIRMED - scaling holo logits to the exact norm
   (y_radial = yH * ||yE||/||yH||) cuts relL2 1.5546 -> 1.2850 (delta 0.27)
   at ZERO information cost, MORE than the pure-exact phase correction
   achieved (0.969 -> 0.779, delta 0.19). Cosine unchanged by construction
   (0.1743 = 0.1743). Sol's reading verified: the correction's L2 gain is
   mostly norm repair; its angular contribution is net-negative.

Verdict: reference valid at the boundary (B7 numbers measure the model
channel); twin-rail is non-holomorphic by design (silu); radial repair
outperforms the phase correction at zero cost. The additive-correction
branch is now closed by every check Sol specified. Remaining: the ONE
minimal covariant transport probe (SU(2) rotational coupling, polar-unitary
packet transport, L0=3, 8 stages).

## 2026-08-04 — B8 minimal covariant transport probe: FAILED - the branch closes

Setup (Sol's exact spec): k=32 (B=64), 8 held-out prompts, borrow through
L0=3, packet extracted at L3mlp, 7 transport stages (L4mix..L7mlp), holo-
only magnitudes, fold-even absent, frames = top-k output frames of
out_proj/down_proj in RESIDUAL-STREAM order (flat stage index t=2l+stage),
connection Q_t = polar(F_{t-1}^T F_t), full 32-layer continuation after
propagation, 7 variants.

Boundary logit cos / top-1 / norm / inverse-transport error:
  correct   0.9821  8/8  2.322  3.9e-05
  deranged  0.9821  8/8  2.322  3.9e-05
  random    0.9815  8/8  2.137  4.5e-05
  carrier   0.9820  8/8  2.216  3.3e-05
  identity  0.9821  8/8  2.327  3.4e+00   (inverse check expected to fail - not polar)
  haar      0.9821  8/8  2.327  3.8e+00   (same)
  direct    0.9821  8/8  2.327  1.7e-05

SOL'S ACCEPTANCE CRITERIA - ALL CONTENT GATES FAILED:
  correct >= random+0.05   : +0.0006   FAIL
  correct >= carrier+0.05  : +0.0001   FAIL
  correct >  deranged      :  0.0000   FAIL (identical)
  polar >  identity        :  0.0000   FAIL
  polar >  haar            :  0.0000   FAIL
  inverse transport ~1e-5  :  PASS (but mechanical - orthogonal bookkeeping
                              only; Sol required it jointly with the content
                              gates)
  smooth decay             : not evaluable - no content signal exists

VERDICT: the transported phase packet carries NO boundary-relevant
information. Exact, random, deranged, and carrier phases are equivalent at
the boundary (spread < 0.001, four orders below threshold). The transport
mechanism (polar/identity/Haar) is irrelevant. Per Sol's directive: STOP -
no demonstrated transported invariant. The full B8 matrix (refresh
schedules, SU(2) couplers, 32 layers) is NOT justified.

PROGRAM TERMINUS (whole phase-native frontier, per Sol's decision tree):
  B7 headline gain     : fold-even carrier magnitudes (exact-side codec)
  purified phase signal: real but weak (+0.132 vs random at extraction),
                          global carrier prior, net-negative angular
                          contribution vs uncorrected
  radial repair        : outperforms the phase correction at zero cost
  transported packet   : no boundary signal at all
  The phase channel carries a small, non-transportable, carrier-conditioned
  residue. There is no non-collapse compression, no transported borrowed
  invariant, no durable holonomy in this construction.

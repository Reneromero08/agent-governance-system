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

## 2026-08-04 — Sol's final review of B8: UNFAITHFUL - ONE MORE PROBE (B8-min)

Sol's audit verdict on the B8 run:
(a) Frame choice (u[:, :k] output frames): defensible. But the CONNECTION
   ORIENTATION is reversed for the column-packet convention:
   Q_t = polar(F_t^dagger F_{t-1}) minimizes ||F_t Q - F_{t-1}||
   (orthogonal Procrustes) - I used polar(F_{t-1}^T F_t) = Q^T. The tiny
   inverse error does not expose this (adjoint transport is also unitary
   and reversible). Must verify directly: ||F_t s_t - F_{t-1} s_{t-1}||
   minimized by polar transport.
(b) The 24-layer holo continuation dilutes/erases the packet effect: the
   0.982 common baseline is dominated by the suffix. Final equality only
   establishes "no tested packet changed the final boundary after this
   suffix", NOT "packets were equivalent during L4-L7". Mandatory control:
   exact-front + NO packet + holo suffix - if that is also ~0.982, the
   final metric is nearly blind.
(c) identity/haar inverse errors are "polar-path mismatch" values, not
   comparator failures - relabel, exclude from acceptance.
(d) deranged == correct exactly: carrier dominance is the leading
   explanation, but wiring must be verified: print ||s_correct-s_deranged||,
   circular phase distance, L4 decoded-delta distance, prompt-index
   checksum; inject a synthetic phase watermark to verify derangement
   changes the decoded rail.
(e) TERMINUS WORDING: "no boundary-relevant transported invariant was
   demonstrated by additive output-frame transport" is SUPPORTED. "No
   non-collapse compression exists" OVERREACHES (exceeds every experiment
   run). The durable-holonomy claim remains technically open because the
   norm-preserving SU(2) coupler and stagewise measurements were never run.

B8-min (the ONE probe, Sol's exact spec):
  B=64, L0=3, 8 prompts, 7 stages (L4mix..L7mlp), fold-even absent,
  holo-derived rail magnitude only. At extraction place the phase packet
  on an ancillary rail. Transport each stage with the CONVENTION-CORRECT
  polar connection. Couple to the main rail with the fixed reversible
  SU(2): [c'; s'] = (1/sqrt2)[[I, iI],[iI, I]][c; s], c = F_t^dagger y_H.
  Retain s' (do not overwrite); decode F_t c' to residual space; apply
  the adjoint at verification (joint-rail reconstruction). Measure after
  EVERY stage: hidden cos + relL2 vs exact, main- and joint-rail norms,
  correct-minus-random and correct-minus-deranged margins, frame-
  continuity error, exact-suffix logit readout from each stage (common
  boundary measurement), final holo-suffix logits (secondary).
  Controls: correct/deranged/random/carrier/polar/identity/Haar + explicit
  no-packet exact-front baseline.
  ACCEPTANCE: correct-packet advantage appearing LOCALLY and surviving at
  least several stages; need not be +0.05 immediately, but must
  consistently beat deranged, random, and carrier across prompts and
  positions. Expected result: FAILURE (carrier-conditioned per B7-purify).
  If correct == deranged locally -> CLOSE the holonomy branch.

## 2026-08-04 — B8-min RUN: FINAL VERDICT - the holonomy branch CLOSES

B8-min built to Sol's exact spec (first run exposed a repeated orientation
bug: Q = polar(Fp.T @ F) = polar(M^T); fixed to the Procrustes-correct
Q = polar(F.T @ Fp) = polar(F_t^+ F_{t-1}) for the column convention -
frame-continuity error confirms the fix: polar 1.20 vs identity 1.48 /
haar 1.48 (polar now minimizes, direct 0.84).)

RESULTS (6 prompts, k=32, L0=3, 6 transport stages, SU(2) coupling,
exact-suffix readout from L7mlp):
  correct   0.9750   (exact-suffix cos)
  deranged  0.9750   == correct EXACTLY
  random    0.9889   >  correct by +0.0139
  carrier   0.9826   >  correct by +0.0076
  identity  0.9781
  haar      0.9743
  direct    0.9889   == random EXACTLY
  nopacket  boundary only
  boundary cos saturated at 0.9986 for EVERY variant incl. nopacket
  (Sol's blindness prediction confirmed - the exact-front borrow
  saturates the boundary metric; the exact-suffix readout was the
  decisive instrument)

SOL'S ACCEPTANCE - FAILED ON EVERY COUNT:
  correct > deranged : 0.0000  FAIL (identical)
  correct > random   : -0.0139 FAIL (correct WORSE)
  correct > carrier  : -0.0076 FAIL (correct WORSE)
  local advantage over several stages: none (suffix readouts show the
  correct packet degrading the boundary coherently)

INTERPRETATION:
  (1) correct == deranged exactly: the phase packet is a global carrier
      prior, not prompt-conditioned - B7-purify's finding confirmed at
      the transport level.
  (2) correct < random: the exact phases are COHERENT with the
      truncation-corrupted odd-residue structure, so the SU(2)-mixed
      correction injects the holo model's own errors coherently; random
      phases average out. The "correct" packet actively harms the
      boundary.
  (3) direct == random: even full complex packet content carries no
      signal beyond noise - the packet channel is information-free at
      the boundary, whatever its content.
  (4) polar transport machinery is now convention-correct and does
      minimize frame displacement - the mechanism works; the packet
      content is what carries nothing.

TERMINUS (per Sol's directive: "If correct and deranged packets are
locally identical under that test, close the holonomy branch"):
  The holonomy branch is CLOSED. The transported phase packet - correct,
  deranged, random, carrier, or direct - carries no boundary-relevant
  information and never beats noise. There is no transported borrowed
  invariant, no durable holonomy, no phase-native non-collapse
  compression in this construction. B7's headline gain remains the
  fold-even carrier (an exact-side-channel codec), the purified phase
  signal remains a small, global, non-transportable residue, and the
  phase channel is information-free at the boundary once transported.

## 2026-08-04 — Sol's B8-min review: STILL BUGGED - premature closure RECANTED

Sol's audit found THREE critical implementation defects in B8-min:
(1) ORTHOGONAL COMPLEMENT DISCARDED: y_new = F_t c'_t REPLACES the stage
    output with its rank-32 framed component, dropping
    y_perp = (I - F_t F_t^+) y_H. Correct decode:
    y_new = y_perp + F_t c'_t = y_H + F_t(c'_t - c_t). Largest confound -
    every variant dominated by the same destructive rank-32 projection,
    which can explain the convergence.
(2) ANCILLARY RAIL AMPLITUDE WRONG: used s0 = unit(F^+ b); the spec is
    s0 = |F^+ y_H| (.) unit(F^+ b) - holo-side magnitudes. Unit packet
    mixes against arbitrarily scaled coefficients.
(3) STAGE COUNT: L4mix..L7mlp = EIGHT stages; only six ran (L7 in the
    plain continuation) - TEND was 2*(L0+4)=14, stages 8..13 only.
Claims (1)-(3) (correct==deranged, correct<random, direct==random) are
UNPROVEN given the defective decode. Frame-cont 1.20 needs normalization
epsilon_t = ||F_t Q s - F_{t-1} s||/||F_{t-1} s||; optimal ~sqrt(2-2/k
*sum sigma_j) for near-orthogonal frames; polar 1.48->1.20 suggests the
orientation is right. The recorded CLOSE was PREMATURE: may only close
"the reported rank-32 additive/replacement transport produced no
distinguishable final invariant" - NOT norm-preserving holonomy.

DECISIVE PROBE (Sol): the FRESHNESS TRIANGLE, one stage at a time,
L4mix..L7mlp, no propagation. From the same holo state at each stage:
  1. fresh matched phase from THAT stage's exact odd residue
  2. stale L3 phase transported to that stage
  3. fresh prompt-deranged phase
  4. random phase
  5. no packet
Decode: c = F^+ y_H; y_perp = y_H - F c;
s = |c| (.) unit(F^+ b) [holo magnitudes];
c' = (c + i s)/sqrt2; s' = (i c + s)/sqrt2; y' = y_perp + F c'.
Retain s' for the inverse check. Per-stage local hidden comparison and
common exact-suffix readout. Interpretation: fresh matched > fresh
deranged/random but stale fails -> phase has LOCAL info, transport makes
it stale; fresh matched also fails -> phase-only packet not boundary-
useful under clean coupling; both work -> transport deserves a corrected
multistage rerun; all == no-packet -> close this construction.
Do NOT derive a replacement packet from the holo trajectory's own past.

## 2026-08-04 — B8-fresh: FRESHNESS TRIANGLE RUN (Sol's decisive probe, corrected decode)

Setup: one stage at a time, stages t08..t13 (L4mix..L6mlp - NOTE: only 6
stages again due to the 32-layer frame table covering t<=2*31; t14/t15
(L7) were NOT included - L7 layers omitted from the frame build for the
freshness loop - VERIFY with Sol whether this matters), 4 prompts,
k=32. Corrected decode: c = F^+ y_H; y_perp = y_H - F c;
s = |c| (.) unit(F^+ b); c' = (c + i s)/sqrt2; y' = y_perp + F c'.
Variants: fresh (that stage's exact residue) / stale (L3 transported via
polar chain) / deranged (prompt-0's fresh phases; prompt 0 self-matches -
contamination noted) / random / nopacket. Metrics: local hidden cos
(last token, post-stage state minus stage output + correction) and
exact-suffix boundary logit cos (full-sequence real-only exact
continuation; reference = full exact trajectory readout with carrier
rail - methodology mismatch noted, affects all variants equally).

LOCAL hidden cos (all stages): fresh 0.945-0.956, stale 0.945-0.957,
deranged 0.946-0.956, random 0.9588, nopacket 0.9588. The local metric
is dominated by the state (stage outputs are tiny relative to the
state) - the correction's local effect is a ~0.001-0.000 perturbation,
near the metric's resolution.

EXACT-SUFFIX cos:
  t08: fresh 0.2246  stale 0.2697  deranged 0.2941  random 0.2904  nopacket 0.2839
  t09: fresh 0.3230  stale 0.3377  deranged 0.3722  random 0.4217  nopacket 0.3983
  t10: fresh 0.2511  stale 0.3601  deranged 0.3700  random 0.5459  nopacket 0.5502
  t11: fresh 0.3481  stale 0.4221  deranged 0.2093  random 0.5127  nopacket 0.5160
  t12: fresh 0.5847  stale 0.5569  deranged 0.6052  random 0.7009  nopacket 0.6712
  t13: fresh 0.4500  stale 0.5637  deranged 0.3794  random 0.4177  nopacket 0.3844

READING (pending Sol verification): fresh matched phases are the WORST
variant at 5/6 stages; random == nopacket at every stage; coherent
phases (fresh/stale/deranged) all UNDERPERFORM nopacket. Per Sol's
interpretation rules this maps to rule 2 (fresh fails: phase-only
packet not boundary-useful under clean coupling) + rule 4 (random ==
no-packet). The phase channel - even fresh, even with the corrected
decode - carries no boundary-useful information; coherent exact phases
ACTIVELY HARM the boundary. OPEN QUESTIONS FOR SOL: (a) does the
t08..t13 window (missing L7) invalidate the verdict? (b) is the
real-only suffix vs carrier-reference mismatch a confound? (c) is the
'fresh worst' pattern explainable as coherent stale-injection (phases
measured at stage t are applied AT stage t - they should be fresh, so
why do they hurt?) - or does it indicate the SU(2) coupling itself is
mis-scaling (s carries |c| magnitudes, c' = (c + i s)/sqrt2 doubles the
effective magnitude of the frame component)?

## 2026-08-04 — B8-fresh2 RUN: oracle-verified freshness triangle (instrument validated)

Fixes per Sol: EIGHT stages t08..t15 (L4mix..L7mlp, TEND=16 - the previous
run's TEND=14 omitted L7); consistent REAL boundary (reference = full
exact REAL trajectory logits, no carrier); real suffix with sanity check;
ORACLE SU(2) control (s_oracle = -i(sqrt2 c_E - c)); strict cyclic
derangement (prompt p <- prompt (p-1) mod 4); residual energy report.

SANITY (instrument validated):
  oracle identity err (max |c'_oracle - c_E|) = 2.4e-07  -> decoder
    algebra mechanically correct (oracle == exact framed output)
  boundary sanity (exact-real-state suffix vs reference) = 0.0
    -> the real suffix reproduces the full exact real reference to
    machine precision at every stage; boundary instrument valid
  residual energy: in-frame = 0.357, out-of-frame = 0.930
    -> only ~36% of the odd residue lives inside the k=32 frame

RESULTS (4 prompts, k=32, real boundary):
LOCAL hidden cos: all variants identical per stage to <2e-4
  (t08 0.7144-0.7175, t12 0.627-0.628, t15 0.551-0.552);
  oracle == nopacket to 4e-6 - even the EXACT frame component is a
  negligible state perturbation.
OUTPUT cos: fresh NEGATIVE at 5/8 stages (t08 -0.446, t10 -0.282,
  t13 -0.068, t15 -0.034): fresh-phase corrected output is
  anticorrelated with the exact output; random ~ 0, stale mildly
  positive at some stages.
SUFFIX cos: no variant consistently beats nopacket; ORACLE vs NOPACKET
  deltas: +0.002, -0.032, -0.003, +0.003, +0.001, +0.016, -0.030,
  +0.007 (mean ~ 0.000, within noise). FRESH vs nopacket: mixed
  (-0.082, -0.002, +0.040, +0.012, -0.012, +0.046, -0.189, +0.016).
  FRESH vs deranged: mixed, no consistent ordering.

SOL'S INTERPRETATION RULES:
  oracle identity fails? NO (2.4e-07) - decoder correct.
  fresh beats deranged/random? NO - mixed, no advantage.
  oracle/exactframed works? NO - oracle ~= nopacket: even with PERFECT
    phase AND amplitude confined to the frame, the boundary does not
    move. This goes beyond 'phase without amplitude is insufficient':
    the FRAME ITSELF is boundary-irrelevant at k=32.
  fresh == stale == deranged == random under valid reference? YES
    (locally to <2e-4; at the boundary no variant beats nopacket).
  fresh works locally but stale doesn't? NO - no local advantage for
    fresh.

VERDICT (pending Sol confirmation): with a validated instrument, the
phase-only construction CLOSES: the phase packet (any content) and the
exact frame component (any content) are boundary-irrelevant at k=32.
The residual-energy split (36% in / 93% out) explains the ceiling: the
frame cannot reach most of the residue, and the oracle shows it cannot
even exploit the reachable 36%. The fresh-phase anticorrelation in the
output cos (5/8 stages negative) is consistent with coherent injection
of truncation-corrupted structure. The B7-purify finding (carrier-
conditioned, non-transportable, non-information-bearing phase channel)
is now confirmed at every level with a validated instrument.

## 2026-08-04 — B8-fresh3 (gauge-aligned) + B8-frame (real-world) RUN: final evidence set

B8-fresh3 (Sol's gauge fix): s_neutral = -i(sqrt2-1)c keeps c' == c
(packet-free output); p = |c| (.) unit(-i d) is gauge-aligned so that
i p lands on d's phase; c' = c + i p/sqrt2. Variants: correct/deranged/
random/neutral/exactframed(dual).

B8-fresh3 SANITY: neutral-rail wiring check 5.97e-08 (neutral ==
nopacket EXACTLY - decoder wiring correct). Wiring identity
(h'_framed - h'_nopacket vs Re(F(cE-cH))): dev 3.9e-01 - THE DUAL->REAL
BRIDGE IS BROKEN: the dual-world correction does not map cleanly onto
the real trajectory (dual and real stage outputs differ substantially).

B8-fresh3 RESULTS (output cos, PRIMARY): correct vs deranged - correct
better at only 2/8 stages (t13, t15); correct vs random mixed (mean ~0);
neutral == random EXACTLY in every table (random packet's net effect
vanishes over k=32); correct max |cos| = 0.09 (t08: 0.090 vs deranged
0.125). The previous run's strong anticorrelation (-0.446) is GONE:
Sol's gauge diagnosis VERIFIED - it was the +pi/2 rotation artifact, not
anti-usefulness. The gauge-aligned phase packet effect is small
(relL2 ~1e-4 level) and never beats deranged/random consistently.

B8-frame (REAL-world exact framed replacement, no dual world):
  REAL-residue energy in/out of F: in=0.374 (norm; ~14% energy),
    out=0.916 - consistent with the dual measurement.
  OUTPUT cos: framed beats nopacket 8/8 stages (t08 0.764 vs 0.414,
    t10 0.505 vs 0.111, t12 0.236 vs 0.005).
  OUTPUT relL2: framed improves 8/8 stages (t08 0.643 vs 0.938).
  HIDDEN relL2: MIXED (t08 framed WORSE: 0.752 vs 0.716).
  SUFFIX cos: MIXED (5/8 framed better; deltas +-0.09, no consistent
    boundary gain).

INTERPRETATION (the complete final picture):
(1) The k=32 frame IS a real control surface on the REAL OUTPUT (8/8
    improvement with exact content) - the dual-world bridge failure
    (0.39) was why the earlier exactframed looked inert.
(2) But the frame is NOT a boundary control surface: exact framed
    replacement does not consistently improve the exact-suffix logits -
    the nonlinear suffix does not rank states monotonically by local
    closeness (Sol's point confirmed).
(3) The gauge-aligned phase packet never beats deranged/random/neutral:
    per Sol's rule 'correct still fails or equals controls -> close the
    phase-only construction'. CLOSE.
(4) The real world is phase-degenerate (real residues, +/-1) - the
    phase channel exists only in the complex world, and the bridge from
    the complex world to the real boundary is broken (0.39). The phase
    channel cannot reach the boundary it is supposed to control.
(5) Per Sol's rule 'exact framed replacement remains ineffective [at
    the boundary] -> close this k=32 frame regardless': the frame
    closes at the boundary level even with exact content.

## 2026-08-04 — Sol's final review of fresh3+frame: BUGGED as phase test + a bug in MY controls

Sol's verdict: the 0.39 dual->real bridge identity failure INVALIDATES the
gauge-aligned phase run (Run 1) as evidence about phase usefulness. The
required commutation M(E_C(z)) = E_R(M(z)) fails: complex-modulus
normalization + persistent carrier change the dynamics BEFORE collapse;
Re afterward does not recover the corresponding real channel. Safe
statement: "No usable phase signal crossed the tested complex-modulus/
real-part bridge" - NOT "no usable phase signal exists."

NEW BUG FOUND IN MY OWN CONTROLS (diagnostic run): torch.randn() is
REAL - angle(g) for a real gaussian is 0 or pi (+-1 signs), not uniform
phases. The "random" packet p = |c| (.) exp(i angle(g)) was REAL, so
i p was purely imaginary and Re(i p) == 0 to machine precision
(||Re(i F p_random)||/||yH|| = 1.5e-08). THE RANDOM CONTROL WAS A
DEGENERATE NULL in every freshness run (b8_freshness, b8_freshness2,
b8_freshness3): 'random == neutral == nopacket' was the Re-collapse
killing a real-sign packet, not phases averaging out. Every
'correct ~ random' comparison in those runs is therefore UNRELIABLE as
a phase-null test. The correct packet DOES have nonzero real projection
(|cos| up to 0.09) - its effect is small, but the null it was compared
against was zero-by-construction.

BRIDGE AUDIT (Sol's 4 conditions - resolved from existing measurements):
  1. carrier-zero complex path == real path: PASS at the boundary
     (logit cos 0.9999, relL2 0.01, B7-ref).
  2. complex exact-framed + collapse == real exact-framed: FAIL
     (wiring identity dev 0.39).
  3. random complex phases -> nonzero collapsed perturbation: FAIL
     (identically zero - degenerate control, this diagnostic).
  4. conjugation equivariance: FAIL (silu not odd, rel dev 1.7).
Per Sol: conditions 1-2 cannot hold under complex-modulus normalization
-> formally close that bridge. CLOSED.

WHAT IS CLOSED (Sol's list):
  1. Independent low-rank SVD at k=256-512 does not preserve the dense
     channels (the wall).
  2. The k=32 frame captures ~14% of residual energy (norm 0.374 ->
     energy 0.140).
  3. Exact k=32 framed replacement improves local outputs (8/8 cos and
     relL2) but not the final boundary consistently.
  4. The complex-modulus carrier + Re collapse is not a valid bridge to
     the real model (0.39 identity failure).
  5. The current phase-only packet has not demonstrated a transported
     boundary invariant.

NOT CLOSED (Sol): no phase-native representation for real weights; no
non-collapse compression; complex phase intrinsically unable to affect a
real boundary. Real vectors can be placed into paired-coordinate /
Fourier / analytic-signal / Hermitian-dilation representations with
nondegenerate phases - UNMEASURED, a different program with a
predeclared commuting real boundary. The complex-native-from-embedding
system is a different program: it changes native evolution rather than
retrofitting a noncommuting auxiliary world onto an existing real
transformer.

TERMINAL ACTION: no more suffix/transport experiments on this bridge.
The bridge is formally closed. The phase frontier on REAL transformer
weights as measured is exhausted within the tested constructions; the
unmeasured directions are different programs.

## 2026-08-04 — B8-fresh4 RUN: fixed controls, VALID result - phase packet does not beat controls

User directive: don't stop until Sol approves. Bug hunt in my own code
continued - THREE more bugs fixed in the freshness pipeline:
  1. RANDOM CONTROL was degenerate (angle(torch.randn) = +-1 -> Re(i p)
     identically zero, 1.5e-8). FIXED: uniform phases torch.rand*2*pi;
     sanity ||Re(i F p_random)|| = 0.2251 NONZERO. This had invalidated
     every freshness 'random' comparison (b8_freshness, b8_freshness2,
     b8_freshness3).
  2. DERANGEMENT: prompt 0 self-matched. FIXED: two-pass (precompute all
     prompts' packets), strict cyclic p <- (p-1) mod N; routing check
     correct-vs-deranged packet diff 0.1216 (nonzero - derangement real).
  3. NEUTRAL VARIANT NEVER COMPUTED y_corr (only set s = s_neutral) - the
     metric read the STALE y_corr from the previous variant (random), so
     'neutral' displayed random's values: random==neutral was an ALIASING
     bug, not physics. FIXED: y_corr = yH for neutral (true no-packet).
     This invalidated the b8_freshness3 tables' neutral column.

B8-fresh4 RESULTS (k=32, stages t08..t15, 4 prompts, gauge-aligned,
uniform-phase random, strict derangement, true neutral, output cos
PRIMARY):
  correct vs neutral : 2/8 stages better (t08 +0.091, t13 +0.023;
                        t09 -0.015, t10 -0.054, t11 -0.022, t12 -0.006,
                        t14 -0.030, t15 -0.010)
  correct vs random  : 2/8 better (t09 +0.006, t13 +0.058; t08 -0.213,
                        t10 -0.021, t11 -0.001, t12 -0.029, t14 -0.002,
                        t15 -0.031)
  correct vs deranged: 3/8 better
  random vs neutral  : 3/8 better (t08 +0.304 ACCIDENTAL alignment)
  exactframed(dual)  : ~neutral at most stages (t08 0.329, t13 0.130)
  OUTPUT relL2: correct ~= neutral (~1e-6 diffs); random WORSE
    (1.0176 at t08); hidden relL2: correct == neutral to 6 decimals
    (correction is a negligible state perturbation)

VERDICT (pending Sol approval): with ALL controls now valid (proper
random phases, true neutral, strict derangement, gauge-aligned packet,
corrected decoder), the correct-phase packet does NOT beat controls:
2/8 vs neutral, 2/8 vs random, 3/8 vs deranged - all noise-level. Per
Sol's rule 'correct still fails or equals controls: close the phase-only
construction'. The phase effects are real but tiny (the coherent packet
is a ~1e-6 relL2 perturbation) and never boundary- or output-relevant
vs doing nothing. The b8_freshness3 tables are superseded by this run.

## 2026-08-04 — SOL APPROVES CLOSE (user directive: don't stop until Sol approves)

Sol's verdict on B8-fresh4 (clean evidence): VALID for the tested
construction. DECISION: APPROVE CLOSE. No k=64 or per-prompt rerun
required.

Verifications:
(1) Correct vs controls: 2/8 neutral, 2/8 random, 3/8 deranged - no
    consistent sign or stagewise advantage. Closure condition satisfied
    for the gauge-aligned k=32 phase-only packet. Wording: "no
    consistent effect above controls" (NOT "statistical noise" - 4
    prompts, one random realization, no reliable noise distribution).
(2) Effect size: partially verified. Hidden-state effect operationally
    negligible; but relL2 equality at 1e-6 does NOT prove the actual
    perturbation norm is 1e-6. Direct quantity:
    ||y_correct - y_neutral||/||y_neutral|| (documentation, not a gate).
    Safe claim: "correct packet produces no measurable, consistently
    useful change" - NOT "phase information has magnitude 1e-6".
(3) t08 random spike (+0.304): acceptable interpretation - decoder is
    phase-sensitive (control sanity), doesn't support the correct packet.
(4) exactframed ~= neutral: consistent with bridge failure
    M(E_C) != E_R(M); closes THIS bridge, not all complex-to-real
    boundaries.
(5) b8_freshness / b8_freshness2 / b8_freshness3: SUPERSEDED - debugging
    history only; none cited as evidence for phase superiority,
    inferiority, or control equivalence.

APPROVED TERMINUS (5 claims):
  1. Independent low-rank SVD at k=256-512 does not preserve the measured
     dense transformer channels sufficiently for coherent depth evolution.
  2. The tested k=32 output frame captures ~37% of residual NORM but only
     ~14% of residual ENERGY; most residual energy lies outside it.
  3. Exact real content in that frame improves the immediate real output
     consistently, but one-stage replacement does not provide consistent
     final-boundary control.
  4. The tested persistent-carrier complex-modulus evolution with real-
     part collapse is not a commuting bridge to the original real
     transformer.
  5. The gauge-aligned k=32 phase-only packet does not outperform
     neutral, random, or prompt-deranged controls and demonstrates no
     durable transported invariant.
  NOT supportable: "no phase-native representation or non-collapse
  compression can exist" - the terminus belongs to the TESTED SVD
  family, k=32 frame, carrier construction, SU(2) packet, and real
  collapse boundary.

NEXT PROGRAM (Sol): complex-native-from-embedding with a predeclared
COMMUTING real boundary. Required identities before model-scale work:
  M(L(x)) = x
  M(E_C(L(x))) = E_R(x)   (real-compatible shell)
  E_C(z_bar) = conj(E_C(z)) where conjugation symmetry is claimed
  If the native complex evolution intentionally differs from the real
  transformer, state that explicitly: it is a new substrate using
  inherited factors, NOT a compressed reconstruction of the original.

DEEPSEEK CORPUS INSPECTION (svh_ref question, resolved): the corpus
_experts_k128.holo (deepseek-ai/_holo) contains _svh = NUMERIC int8-
quantized shared SVh tensors + _svh_scales per tensor (format
'int8_dedup', k=128). KEY STRUCTURAL FINDING: ALL 256 experts in a
layer SHARE ONE SVh row space (V deduplicated) - experts differ only in
their U factors. Per Sol: retained spectra recoverable (row norms of
dequantized SVh = sigma_i); measurable: retained spectral decay within
k=128, cross-expert retained-spectrum similarity, principal-angle
overlap between expert U subspaces. NOT measurable without originals
(user no longer has them): omitted spectral tail, full-rank effective
rank, residual energy outside retained k, exact compression error.
The shared-V structure is itself a compression surface: if expert U
subspaces overlap in a low-dimensional manifold, the expert set is
compressible even though each channel is full-rank - the coupled-mode
direction at the EXPERT level (unmeasured).

## 2026-08-04 — MOE test bed + Sol's REDESIGN of the expert manifold test

Test bed: Qwen3.6-14B-A3B-FableVibes (Qwen3_5MoeForCausalLM), Q8_0 GGUF
14.7GB downloading to LM Studio folder. Architecture: 40 layers (LLLF),
90 experts, 8 active/token, hidden 2048, moe_intermediate 512, head_dim
256, GDN. Provenance: REAP-pruned from 35B-A3B + QLoRA-recovered -
report results as "post-REAP Qwen expert geometry", NOT natural-MoE
claims. Same class as DeepSeek V4 Flash (endgame), same family as our 4B.

Sol's REDESIGN verdict on my proposed measurement:
1. MY UNION-RANK TEST WAS DIMENSIONALLY DEGENERATE: for 2048x512
   matrices the union of full subspaces can never exceed the 2048
   ambient dimension, so "union rank << 90*512" is trivially true and
   cannot demonstrate a manifold. Correct test: the NORMALIZED HIDDEN-
   INTERFACE PROJECTOR P = (1/90) sum_e B_e B_e^T in R^{2048x2048},
   where for gate/up (512x2048) B_e = RIGHT singular vectors (the
   expert's INPUT interface) and for down (2048x512) B_e = LEFT
   singular vectors (OUTPUT interface). Compare P's eigenvalue spectrum
   against the random-subspace null (90 random 2048x512 bases).
   THRESHOLDS: spectrum dominated by ~512 large eigenvalues = manifold
   near one expert width; decays like the random null = no manifold;
   between = partial structure.
2. Q8_0 acceptable for dominant shape/large gaps/strong alignment/gross
   flat-vs-decaying; NOT for tail effective rank, small principal
   angles, weak eigengaps. F16 confirmation REQUIRED on layers
   {0,19,39} before any tail/manifold claim.
3. GGUF naming: fused tensors blk.N.ffn_gate_exps/up_exps/down_exps/
   gate_inp (+ffn_*_shexp for the shared expert), NOT individual expert
   names. Verify the 90-axis, orientation (gate/up 512x2048, down
   2048x512), dequantization, and a manual matmul BEFORE any SVD.
4. k-curve: per-expert max rank 512; k = 16,32,64,128,256,384,512;
   report singular-energy retention, stable rank, entropy effective
   rank, condition profile. FUNCTIONAL k-curve needs routed activation
   traces (llama.cpp instrumentation hook recording post-norm MoE
   inputs + selected expert IDs + router weights, replayed offline);
   without the trace it is a weight-space k-curve only.
5. MINIMAL ORDERED EXPERIMENT:
   1. Integrity gate: dump GGUF metadata, confirm expert axis/
      orientation/dequant; exclude shared experts from routed stats.
   2. Q8 geometry pilot: layers {0,7,15,23,31,39}, all 90 experts, 3
      projection families separately; spectra (stable rank, D_eff,
      D_95, D_99, condition), P spectrum, random-subspace null.
   3. F16 confirmation: layers {0,19,39} (Q8 vs F16 singular values,
      projector spectra, principal angles) - abort tail claims if
      conclusions change.
   4. Functional trace: routed inputs -> expert MLP k-curves
      (active vs inactive vs shared expert).
6. Shared expert = dense baseline, analyzed separately.

Status: script moe1_geometry.py built (integrity gate + geometry pilot);
Q8_0 download in progress (5.9GB/14.7GB).

## 2026-08-05 — MOE pilot RUN 1 (Q8, 6 layers) - no manifold signal + L0-down anomaly

PIPELINE FIXES (3 bugs in my own code, found by checking):
  1. gguf Q8_0 enum value is 8 in this package (not 4); F32=0, F16=1.
  2. gguf header shape = REVERSED logical shape; data has the LAST
     header dim CONTIGUOUS - the fused expert tensors are EXPERT-
     INTERLEAVED at the element level, not block-contiguous per expert.
     My first reshape produced 71-318 exactly-zero rows per expert
     (garbage); the interleaved read (transpose(data,(2,1,0))) gives
     zero rows everywhere and sane spectra. Verified via zero-row
     counts + shared-gate (2D) control.
  3. torch has no negative strides - fixed eigvalsh flip.

RESULTS (Qwen3.6-14B-A3B, Q8_0, layers {0,7,15,23,31,39}, 90 experts,
k=512 full):
PER-EXPERT SPECTRA (mean over 90 experts):
  w1/w3 (gate/up, 512x2048): D_eff 446-451 (of 512), stable-rank
    205-228, D95 438-441, D99 494, head/tail 3.0-3.2 - near-flat.
  w2 (down, 2048x512): same flat profile at L7/L15/L31 (D_eff 451,
    stable 228) BUT L0: stable-rank 4.5, D_eff 161, D95 388,
    head/tail 31.2 - VERY concentrated; L23: stable 129.5; L39:
    stable 148.9 (mid).
  L0-down detail: s0 ~ 8.6, s100 ~ 0.83, s[-1] ~ 0.2766; the top ~4-5
    modes hold most energy, tail flat at ~0.28. Consistent across all
    90 experts (not an outlier artifact).

HIDDEN-INTERFACE PROJECTOR P = (1/90) sum B_e B_e^T vs RANDOM NULL
(90 random 2048x512 bases):
  D95: real 1853-1903 vs null 1903 at every layer - the expert union
    spans essentially the full 2048-dim space like random subspaces.
    NO low-dimensional expert manifold (Sol's 'one expert width' ~512
    threshold: nowhere near).
  Top eigenvalues: w1 projectors 0.43-0.60 vs null 0.35 (consistently
    elevated 1.3-1.7x at every layer - a handful of shared input
    directions, tiny mass fraction); L0-down projector top = 0.9892 +
    0.6439 (two strong shared OUTPUT directions); tail50 real
    0.10-0.17 vs null 0.17 (real tail suppressed = fewer distinct
    directions than random).

PRELIMINARY READING (Q8 caveat - F16 confirmation REQUIRED per Sol
before any tail/manifold claim): the expert manifold hypothesis FAILS
at the expert-set level - the union of 90 expert subspaces is
essentially as spread as random. The wall generalizes to MoE. The only
structure: a few shared directions per layer, strongest at L0-down
(2 shared output directions, eigenvalue ~1.0) and the L0-down
concentration anomaly (stable-rank 4.5). L0-down warrants F16
confirmation FIRST.

## 2026-08-05 — MOE anomaly audit + BF16 confirmation attempt: REVISION MISMATCH

MOE-2 anomaly audit (Q8, Sol's 5 checks):
  1. Expert-axis audit: PASS - lane scales across experts differ
     (interleaved read confirmed; not a repeated lane).
  2. Quant diagnostics: L0-down has 1,094,155 exact-zero int8 values vs
     L7-down's 21,654 (~50x more); zero-scale blocks: none in either;
     scale-periodicity corr: L0 0.086 vs L7 0.206 (no strong pattern).
  3. DC/scale test: THE L0-down SPIKE IS A DC COMPONENT - cos(u0,
     all-ones) = 0.9913, cos(u0, row-mean) = 0.9990. After row/col mean
     subtraction: s0 8.64 -> 1.75, stable-rank 4.5 -> 85.6. Per Sol's
     criteria: 'if the spike disappears after mean subtraction, it is a
     DC or dequantization-scale artifact rather than a learned channel'.
  4. Shared-spike removal: projector mass top-1 = 0.0019 (0.19% of the
     union mass) - even the shared spike is a tiny compression surface;
     residual projector after rank-1 removal: D95 1888 (null 1903).
  5. Pairwise affinity (fixed .sum() bug): median 0.266 vs random
     expectation 0.25 - NULL, no expert clustering.

MOE-3 BF16 selective confirmation: INVALIDATED BY REVISION MISMATCH.
  - Fetched L0-down (90 experts) + L7-down (10) + L0-gate (10) from the
    base repo safetensors via byte-range (230MB total; safetensors
    header parse + offsets verified via lm_head anchor at offset 0).
  - BUG 1 (mine): safetensors are BF16; I decoded as np.float16 (wrong
    bit layout) - produced 70-100x scale garbage. Fixed with
    torch.bfloat16 decode.
  - BUG 2 (found by checking): even with correct BF16 decode, the
    matrices are DIFFERENT: L0 gate e0 elementwise mean abs diff 0.0115
    vs weight std 0.0078 (diff > weight itself); max rel dev top-50
    singulars 0.317; cos(u0_q8, u0_bf16) = 0.015 (orthogonal leading
    vectors). Singular value SCALES now match (BF16 s0 0.945 vs Q8
    0.868) but the matrices differ well beyond Q8 noise.
  - Revision archaeology: base repo commits show model.safetensors
    uploaded once (401a006c6602, 2026-06-14, same day as the GGUF
    upload); no branches/tags; GGUF metadata lacks a source revision
    hash. The GGUF's source checkpoint is NOT the current safetensors
    (likely a local re-export/merge variant). The exact source is
    unrecoverable from HF metadata.

STATUS:
  - EXPERT MANIFOLD VERDICT STANDS (Q8-safe dominant-structure measures):
    no manifold (D95 ~= null ~1903), affinity ~= 0.25 null, per-expert
    near-flat spectra (D_eff ~450/512). The Q8 GGUF is a genuine
    post-REAP 90-expert MoE; the verdict describes THOSE weights.
  - L0-down DC spike: confirmed as a DC/row-mean component IN THE Q8
    WEIGHTS; its existence in the 'true' model is now UNCONFIRMED (the
    BF16 check is invalidated). Its compression mass is ~0.2% either
    way - program-level conclusion unchanged.
  - The selective-BF16 confirmation path is closed unless the true
    source revision can be identified (diminishing returns).

## 2026-08-05 — Sol's MOE-2/3 review: VALID, next = expert-axis centroid/Gram test

Sol verdict: findings VALID for the published Q8 artifact. Accept the
shared-subspace verdict; DO NOT chase the mismatched BF16 revision.
Defensible experimental object: "the exact, hashed
Qwen3.6-14B-A3B-FableVibes-Q8_0.gguf artifact as published" - preserve
filename/size/hash/metadata as the experimental identity. Dominant
conclusion not Q8-noise-sensitive (D95 1900 -> 768 would need a radical
transformation). L0-down wording: "contains a strong shared DC mode in
the published Q8 weights; its source is unresolved" (mean subtraction
shows WHERE not WHO; 50x zero-count supports outlier/scale interaction;
0.19% mass - negligible either way).

SUPPORTED CLAIMS (1-5): near-full-rank experts; no common basis; median
affinity 0.266 ~ random 0.25; weakly spiked random-like bulk; negligible
L0 DC mass. OVERREACH: "weight-level compression is exhausted" - TWO
weight-level surfaces unmeasured:

A. EXPERT-AXIS TENSOR RANK: X = [vec(W_1); ...; vec(W_90)] (90 x N),
   90x90 Gram WITHOUT materializing NxN; raw AND centered
   (W_bar = (1/90)sum W_e, delta W_e = W_e - W_bar). Report: centroid
   energy share, centered-Gram effective rank, expert-axis
   D_50/90/95/99, pairwise Frobenius correlations, cluster structure,
   residual spectra after centroid subtraction.
   PREDECLARED THRESHOLDS: strong D95<=16; partial 16<D95<=32; none
   D95>=72. Per family (gate/up/down) THEN the concatenated
   gate/up/down triplet (intermediate neurons permutable jointly).
B. DISCRETE CLUSTERS: single global basis can fail while compact expert
   families exist - within-cluster projector affinity vs 0.266,
   cluster-specific D95. Run only if Gram/pairwise shows structure.

C. FUNCTIONAL TRACE: still worthwhile - weight flatness != functional
   rank under routed activations; router may send narrow anisotropic
   input distributions. Truncation ranks 16..512 on ACTUAL routed
   inputs; per-expert + routing-frequency-weighted output cos/rel
   error/worst-decile. Synthetic inputs insufficient (isotropic
   restates SVD energy). Cleanest: instrument llama.cpp for the exact
   GGUF (layer-normalized MoE input, selected expert IDs, router
   weights, token/layer indices) then offline replay.

ORDERED NEXT STEPS: 1) accept+document Q8 verdict; 2) drop BF16
revision; 3) expert-axis centroid/Gram test (NOW); 4) cluster test
only if Gram shows structure; 5) instrument GGUF + routed k-curve.
STATUS: 'one shared expert subspace' FALSIFIED; individual expert
low-rank compression strongly disfavored; expert-axis templates,
clusters, activation-conditioned routing remain OPEN.

## 2026-08-05 — MOE-4 expert-axis Gram test RUN: NO template surface

Sol's expert-axis tensor-rank test on the Q8 GGUF, layers {0,7,15,23,31,
39}, 90 experts, raw + centroid-centered, per family + triplet:
  w1 (gate) raw: D_eff 90.0, D95 86, pair-corr ~0.0001 (ORTHOGONAL axes)
  w3 (up)   raw: D_eff 90.0, D95 86, pair-corr ~0.0001
  w2 (down) raw: D_eff 53.6 (L0) / 90.0 (others), D95 84 (L0) / 86,
                 pair-corr 0.225 (L0 - the DC mode) / ~0.0001 (others)
  triplet raw: D95 85-86, corr 0.117 (L0) / 0.0001 (others)
  centered: D95 85-89 everywhere; centroid energy share 0.0001-0.0026
  (negligible template structure). (D_eff nan on some centered rows is
  a cosmetic float-sign issue in the entropy formula - D50-99 valid.)

THRESHOLD VERDICT (Sol predeclared: strong D95<=16, partial 16<D95<=32,
none D95>=72): NO surface - D95 84-89 at every layer/family/triplet.
The only axis structure is L0-down's DC-related correlation (corr 0.225,
D95 84) - consistent with the shared DC mode; negligible.
Per Sol's decision rule, the CLUSTER test is SKIPPED (no Gram/pairwise
structure to cluster).

REMAINING (Sol step 5): the FUNCTIONAL TRACE - routed inputs -> expert
MLP k-curves (ranks 16..512), per-expert + routing-frequency-weighted
output cos/rel error/worst decile. Requires genuine forward execution of
the GGUF: options (a) instrument llama.cpp for the exact GGUF, (b) build
a minimal qwen35moe forward in Python from the dequantized weights
(extending the existing 4B hybrid engine: GDN mixer + full-attn are
already implemented; add router + MoE FFN + shared expert) - gives exact
offline expert replay (Sol's preference). Pending Sol's direction.

## 2026-08-05 — Sol's MOE-4 review: expert-axis CLOSED, HYBRID trace approved

(a) EXPERT-AXIS CLOSED with hygiene fixes: clamp eigenvalues >= 0 before
   D_eff (verify |lambda_min| < 1e-5 * lambda_max, else not cosmetic);
   report max + 95th-pct absolute off-diagonal correlation (catches
   isolated duplicate pairs). APPROVED CONCLUSION: "The published Q8
   artifact has no useful shared hidden-space basis, expert-axis template
   basis, centroid structure, or large expert clusters across its 90
   routed experts."
(b) TRACE = HYBRID: llama.cpp = trajectory oracle (capture: token index,
   layer index, post-norm MoE input x[2048], top-8 expert IDs, top-8
   router weights, shared-expert gate); Python = measurement chamber
   (offline replay + rank sweeps). Do NOT extend the Python engine
   through 40 layers (too many equivalence risks: Q8 interleaving,
   router scoring/top-8/renorm, shared-expert order, GDN state, 32 value
   heads, conv4, full-attn q/gate interleave, MRoPE 64-dim, LLLF order,
   offset RMSNorm, MTP block, untied embeddings). VALIDATE: replay one
   captured token's 8 expert MLPs from dequantized weights and match
   llama.cpp's routed output (close numerical agreement) BEFORE the
   k-curve - settles orientation, router application, renorm, shared
   expert addition.
(c) PRIMARY metric: NOT routing-frequency-weighted per-expert cosine
   (nonlinear). For each token/layer with routing fixed:
   Y = sum_{e in top8} g_e W2,e[SiLU(W1,e x) . W3,e x]; full-rank Q8 vs
   simultaneous truncation 16..512. Metrics: token-level aggregate
   routed-output cosine; aggregate rel L2; median/10th/1st pct/worst
   token; routing-frequency-weighted aggregate; per-layer distributions.
   Shared expert kept EXACT in the first test (separate dense channel).
   Secondary: per-expert conditional cos + rel err, assignment count,
   router-weighted absolute error contribution, performance vs routing
   weight, frequent vs rare experts. Two truncation modes: simultaneous
   gate/up/down rank k AND one-projection-at-a-time (bottleneck).
   Also: effective rank of routed input matrices X_e, of outputs Y_e,
   input energy captured by retained right-singular modes. Coverage:
   >=128 assignments per layer/expert for per-expert summaries; mark
   rare experts under-sampled.
(d) L0-DC functional audit: projector mass is the WRONG criterion.
   Exact shared-output-direction decomposition:
   W2,e = u_DC a_e^T + W2,e^perp. Measure on routed z_e:
   y_DC,e = u_DC (a_e^T z_e). Report: fraction of routed down-output
   energy in DC; cosine(DC contribution, full output); effect of
   removing DC on aggregate; residual k-curve with DC preserved. If DC
   carries substantial functional energy: store one shared u_DC + 90
   short coefficient vectors a_e (legitimate catalytic shared carrier
   at L0, even though it cannot rescue the global manifold).
NEXT: instrument llama.cpp, validate one-token replay, routed aggregate
k-curve + L0-DC functional audit.

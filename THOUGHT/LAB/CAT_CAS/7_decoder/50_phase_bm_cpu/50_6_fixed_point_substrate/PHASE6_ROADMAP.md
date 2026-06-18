# Phase 6 Roadmap

**Role:** MASTER CHRONOLOGICAL MAP. Forward-looking task ledger with narrative
context. Tells you what's done, what's not, why, and in what order. For the
directory index and story of each sub-phase, see `PHASE6_NAVIGATION.md`. For the
active Phase 6B task board, see `14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md`.

## 0. The Narrative Arc

Phase 6 is 14 sequential attacks on a single wall. The wall is the **dihedral fold**:
the Exp50.14 public fixed-point map has hidden secret `d`, but all public data is
wrapped in `cos(2*pi*k*d/N)`. Since cosine is even, the data is identical for `d`
and `N-d`. The orientation bit `o = 1[d < N/2]` -- which of the two fold-mirrors is
the real answer -- is **information-absent** from public data. A forward machine must
scan O(N) = 2^n to find `d`. Phase 6 asks: can an analog substrate relax into `d`
without scanning?

The 14 sub-phases probe the wall from every angle: information-theoretic (02),
non-Hermitian topological (03), phase-cavity interferometric (04), eigen-collimation
inside the coset space (05), collective superradiance (06), physical simulation (07),
pre-projection chiral encoding (08), transient path functionals (09), cross-core
PDN traversal (10), and a full forward-detector campaign of 15 planned tracks (12).

**The pivot:** Sub-phase 02 (fold audit) proved MI(o; public data) = 0 -- the
orientation bit is not hard, it is *absent*. Every subsequent negative was trustworthy
because the no-smuggle gate caught any hidden-d injection at AUC 1.0. Sub-phases
03-09 all returned FAIL_CHANCE. Sub-phase 12 closed the forward-detector campaign:
all no-smuggle tracks fail orientation. The TERMINUS synthesis (TERMINUS.md) confirmed
the wall is exactly the map `z -> Re(z)` -- the absent quadrature.

**The reframe:** The Phenom II is a real/scalar (timing/thermal) substrate. It
cannot read the odd/quadrature channel. The crossing, if it exists, is not a faster
search on the old encoding -- it is a **change of representation** into one where
the substrate's native physics (linear diffusion + one threshold = a frozen hardware
perceptron) *is* the evaluation. Sub-phase 14 (noncollapse OrbitState) is the current
attack on that framing. Sub-phase 10 (cross-core PDN wormhole) is the live physical
substrate candidate.

**Where we stand:** The construct side is measured-closed. Two live threads remain:
the noncollapse theoretical frontier (14) and the cross-core physical channel (10).
Phase 6 Mode C (the full catalytic fixed-point loop on Phenom hardware) is gated
behind Phase 5.10C -- reproducible basin selection -- which is not yet complete.

---

## 1. The 14 Sub-Phases

### 01 -- Target Generator (DONE)

**Why:** The Exp50.14 map generates a public Fourier fixed-point target from
`{(k_i, b_i)}` with `M ~ sqrt(N)` samples. Before any attack, we needed the
A/B baselines: Mode A forward scan cost, Mode B reversible catalytic cost, and
unique fixed-point targets for scaling. The 5.9V basin selector audit was consumed
here to feed Phase 6 gate G3.

**Key result:** Unique fixed points for n=8 through n=16. A/B baselines confirm
the expected forward-work cost shape. 5.9V basin selector is directional, not
deterministic -- insufficient for Mode C handoff.

- [x] A/B dry-run baselines for n=8,10,12,14,16
- [x] 5.9V basin selector audit
- [x] Feeder run: G1-G7 gates read out

---

### 02 -- Fold Audit (DONE)

**Why:** Before attacking the wall, prove the wall is real. If the construction
leaks `d` through PRNG seeds, float rounding, or metadata, every "crossing" is
fake. The fold audit tests whether the orientation bit exists in the public data
at all.

**Key result:** **MI(o; public data) = 0, proven.** Stage 1 ran 16 classifiers
+ 8 nonlinear equivariant lifts -- AUC = 0.5 across all 24 readouts. The
`d`-conditioned and `(N-d)`-conditioned distributions are two-sample identical.
Stage 3 confirmed the real 50.14 public interface is orbit-only: no float
code-path, seed, order, or verify-map leak. The reusable `no_smuggle_gate.py`
and `hardened_gate.py` were built here and imported by every subsequent sub-phase.

- [x] Stage 1: MI=0 proven (16 classifiers + 8 lifts)
- [x] Stage 3: generator audit (no implementation leak)
- [x] no_smuggle_gate.py + hardened_gate.py

---

### 03 -- Non-Hermitian Sensors (DONE, 6/6 FAIL_CHANCE)

**Why:** `f(x) = x if score(x)>M/4 else (x+1)` has a +1 directionality that
breaks the fold. Non-Hermitian topology specializes in directed/non-reciprocal
structure -- point-gap winding, skin effect, exceptional points -- invariants
with no Hermitian analog. The bet: the map's directionality, read as a
non-Hermitian topological invariant, might carry orientation.

**Key result:** **The directionality IS topological, but orientation-BLIND.**
Koopman transfer operator is fold-identical. Hatano-Nelson skin center-of-mass
is global. Kuramoto chiral reads "directed current," not which half. Cauchy
argument-principle contour can't resolve in poly budget. PT-symmetric spectrum
is fold-blind (isospectral under reflection). Godel-edge phi-twist produces
W=+1 as a *public constant* -- it carries walk direction, not which half `d` is
in. One apparent crossing (Kuramoto chiral n=14, AUC 0.604) regressed to 0.505
across 13 reaudit seeds -- a finite-sample false positive. All 6 smuggle
controls caught at AUC 1.0.

- [x] Koopman/transfer operator -- point-gap winding, fold-identical
- [x] Hatano-Nelson skin effect -- IPR orientation-blind
- [x] Kuramoto/chiral resonance + 13-seed reaudit -- one false positive killed
- [x] Cauchy argument principle -- poly-budget contour wrong winding
- [x] PT-symmetric / biorthogonal -- spectrum fold-blind, isospectral
- [x] Godel-edge phi-twist (Exp 36 rank-1 lemma) -- W=+1 is public constant

---

### 04 -- .holo Phase Substrate (DONE, FAIL_CHANCE)

**Why:** The lab's flagship "it from phase" method. Encode public (k,b) as a
phase-grating spectrum, reconstruct the coherent cavity field `Psi(x)`, and
read BOTH quadratures with a tunable local oscillator (homodyne). The conjugate
quadrature `Im Psi(x)` at the fold peaks should carry orientation if the phase
is physically present. This is the capstone test of the program's core method
against the bedrock target.

**Key result:** **The cavity works, but the signal is not present.** The even
fold-answer `a = min(d, N-d)` is read at frac_exact 1.000. The imaginary (sin)
quadrature the cavity computes from public data is ~0 to machine precision
(Im/Re ~ 1e-14). The two fold peaks at `d` and `N-d` are exactly equal in both
magnitude AND phase. Injecting the hidden sin makes `|Im|` nonzero with sign ==
orientation at 100% -- the homodyne is not weak, the signal is absent. 8-seed
reaudit confirms. This closed SPEC_PHASE6 sec 1B/1C on the flagship substrate.

- [x] Phase cavity reads min(d,N-d) at frac_exact=1.000
- [x] Orientation FAIL_CHANCE -- conjugate quadrature ~0 to machine precision
- [x] Homodyne sweep, interferometric recombination, torus winding -- all phase-blind
- [x] 8-seed reaudit confirms

---

### 05 -- Black-Hole Eigen Collimation (DONE)

**Why:** Sub-phase 04 fed the cavity the *already-translated* public data
(real-even shadow). The corrective: compute INSIDE the coherent complex space
using the dihedral coset state `|c_k> = (|0> + omega^{kd}|1>)/sqrt(2)` provided
directly by the oracle. The orientation IS present here -- the states for `d`
and `N-d` are complex conjugates. EIGEN_BUDDY is any fixed, d-independent
operator that concentrates the answer into a dominant eigenvalue.

**Key result:** **The abelian period rings in poly, the dihedral orientation
does not.** QFT maps the phase register to a single dominant peak at `|d>` --
period rings as a resonance, no search (abelian HSP). But the dihedral
orientation requires the Kuperberg `2^{O(sqrt n)}` sieve. The wall is the
difference between the abelian and dihedral hidden subgroup problems --
confirmed concretely on the actual coset states.

- [x] QFT period rings in poly (abelian HSP)
- [x] Orientation does not ring (dihedral)
- [x] Subexp Kuperberg bound confirmed

---

### 06 -- Superradiant Sieve (DONE)

**Why:** Sub-phase 05 left one loophole: "any FIXED operator that COMMUTES WITH
the mirror is blind." A chiral (handedness-breaking) geometry does NOT commute
with the mirror. Superradiance is the physical form of combining coset states
as one collective resonance (one diagonalization, not Kuperberg's iterative
sieve), and the orientation IS a chirality.

**Key result:** **The chiral loophole is real, but produces no gain.** The
chiral superradiant array is a fixed, d-independent operator that breaks the
mirror (||[Gamma, P_reflect]|| = 11.2 vs 3e-15 for achiral). The Dicke/sum-rule
engine validates faithfully. But collective superradiance provides no advantage
over single-coset readout at the `2^{O(sqrt n)}` Kuperberg bar. The loophole
exists structurally; it doesn't change the scaling.

- [x] Dicke test + sum rule + bright/dark validated
- [x] Chiral channel: mirror-breaking confirmed (||[Gamma, P_reflect]|| = 11.2)
- [x] Collective: no gain over single-coset at Kuperberg bar

---

### 07 -- DRAM Rowbuffer Sim (DONE)

**Why:** Early simulation prototyping for the cross-core concept. Test whether a
DRAM row-buffer driven lock-in can carry a .holo footprint with phase. Scalar
mode classification worked at real accuracy 1.000, but the phase/relational
recovery did not.

**Key result:** DRAM_ROWBUFFER_LOCKIN_SIM_PARTIAL. Mode recovery works, phase
does not. The amplitude ladder is non-monotonic. This was an early exploration
of whether non-cache physical channels could carry the .holo signal -- it
informed the later PDN approach (sub-phase 10).

- [x] Real accuracy 1.000, pseudo rejected
- [x] Phase recovery: FAIL -- amplitude ladder non-monotonic

---

### 08 -- Chiral Phase Kickback (DONE)

**Why:** Can a chiral pre-projection tape preparation expose a fold-odd carrier
BEFORE the public cosine boundary is read? The idea: write a phase-walk
direction into the catalytic tape during a pre-projection window, then measure
whether the walk direction survives projection as an orientation signal.

**Key result:** **FAIL_CHANCE across n=8,10,12.** Public chiral prep,
dual-lane even cancel public, and chiral shuffle null public all return
AUC ~0.5. The hidden control (phase-walk direction deliberately bound to
`d`) is caught as smuggle at AUC 1.0 -- the gate is live, the negative is real.
This became a fixed fact in the chiral lane frontier: `PUBLIC_CHIRAL_PREP_NO_CROSSING`.

- [x] Public chiral prep: FAIL_CHANCE (AUC ~0.5 at n=8,10,12)
- [x] Dual-lane even cancel public: FAIL_CHANCE
- [x] Hidden control: live -- smuggle caught at AUC 1.0
- [x] Rust bare-metal PDN native probe

---

### 09 -- Transient Fold Probe (DONE)

**Why:** The SPEC open crack: "the map f(x) has two fixed points {a, N-a} with
basins of size 2a and N-2a. The MAP knows which fixed point it walked to. Is
there a fold-ODD functional of the TRANSIENT -- how the walk approaches each
fixed point, local phase of approach, return times -- that global sensors missed?"

**Key result:** **FAIL_CHANCE.** Public transient features remain fold-even
under the hardened random-private-fold gate at n=8,10,12. The hidden-orientation
control is caught as smuggle -- the instrument is live. This closed the specific
open crack from REPORT_SESSION_LATTICE_CLIMB.md about local/transient invariants.

- [x] Public transient features: FAIL_CHANCE (n=8,10,12)
- [x] Hidden control: live -- smuggle caught
- [x] Closes LATTICE_CLIMB open crack about local/transient invariants

---

### 10 -- Cross-Core PDN Wormhole (DONE)

**Why:** Sub-phases 02-09 all attacked the wall with SOFTWARE methods on the
Phenom. The cross-core wormhole tests whether a PHYSICAL channel -- the shared
power-delivery-network (PDN) rail -- can carry a .holo footprint (MODE + phase)
across CPU cores where a forward machine cannot. The sender modulates the PDN
via compute load; the victim core runs a ring-oscillator lock-in on a shared-TSC
origin. This is the TRAVERSABLE approach: borrow state, scramble, open coupling,
unscramble, restore -- the same wormhole protocol from Exp 32, instantiated on
silicon.

**Key result:** **Cache channel (Slot 1): clean negative.** Prime+probe stayed
in private L2; mode-discriminating signal never reached shared L3. **PDN channel
(Slot 2): CONFIRMED.** Primary pair v2:s3: MODE accuracy 1.00, pseudo_reject
1.00, relational phase delta 0.89-1.10 on ALL 6 seeds. The rvp gate dips on
4/6 seeds at trials=48 (~7 test symbols/mode) -- underpowered, not a channel
failure. The earlier sim predicted 0.79 real accuracy on cache; hardware gave
0.31 -- the PDN carrier avoids the cache homogenization failure mode.

**Claim cap:** Cross-core .holo traversal via PDN. NOT a lattice crossing, NOT
a Phase 6 fixed-point claim. The PDN physical channel is the substrate
candidate. The lattice terminus (d-invariant) remains open.

- [x] Stage 1 sim: CROSS_CORE_WORMHOLE_SIM_VERIFIED
- [x] Slot 1 cache-conflict: clean negative
- [x] Slot 2 PDN lock-in: primary pair v2:s3 CONFIRMED (MODE 1.00, pseudo_reject 1.00, phase delta 0.89-1.10 on 6/6)
- [x] T300 result integrated -- route 4:5 selected from the completed sweep
- [ ] Evaluate second pair v4:s5
- [ ] Fire trials=300/mode for strict all-9-gates witness -- rvp needs ~40 test symbols/mode
- [ ] Sweep additional core pairs -- optional, proves channel not pair-specific

---

### 11 -- PDN Catalytic Tape Post-Mortem (DONE)

**Why:** The wormhole sim (sub-phase 10, Stage 1) predicted 0.79 real accuracy
on cache. Hardware gave 0.31. This post-mortem sim injects realistic-noise
degradation (correlated 1/f drift in amplitude AND phase across capture, a slow
random walk the inverse permutation cannot undo) to explain the gap. It also
validates why the PDN carrier beats the cache carrier: the PDN reads
INSTANTANEOUS current through the shared rail (stronger coupling HELPS),
while sustained cross-core cache co-access HOMOGENIZES line residency (stronger
coupling HURTS).

- [x] Realistic-noise model explaining sim (0.79) vs hardware (0.31) gap
- [x] Fix probe for PDN tape

---

### 12 -- Chiral Lane Frontier (CLOSED)

**Why:** After sub-phases 02-09 closed the passive-data side, the question
became: can PUBLIC EXECUTION GEOMETRY -- the physical act of running the oracle
on real silicon -- generate a fold-odd carrier? This was a 15-track campaign
(Tracks A through N, plus Z, 0, B, I) designed in a 2174-line roadmap. The
architecture: every track must name its odd-lane source before building; hidden
controls calibrate the detector; public routes are blinded; scoring is offline.

**Key result:** **All no-smuggle tracks fail orientation.** Track A ran 3
Phenom II hardware architectures (sequential, simultaneous, lock-in) with 12/12
controls -- all negative (L4). Track D (commutator mathematical reference):
multi-seed AUC 0.516, below null95 (L3). Track F (cumulative Hamming weight
reference): weak seed-dependent hint, orientation FAIL_CHANCE (L3). Tracks C
and E were **rejected at staging** -- their roadmap designs assigned operation
direction to candidate labels manually (ROL->c0, ROR->c1; cw->c0, ccw->c1).
This is **label smuggle**, not a physical mechanism. They need redesign before
they can be built. Tracks G, H, K, L, M were **never built** because the
campaign pivoted to the noncollapse paradigm (sub-phase 14) after Sprint 2.
Sprints 3-5 were not reached. Track N (boundary ledger) was merged into the
MASTER SYNTHESIS.

- [x] E5/E1: oracle fold-integrity gate
- [x] Track Z: orientation conservation audit (SCHEDULE_INVARIANCE_PASS)
- [x] Track 0: odd-lane transfer function (ODD_LANE_DETECTOR_CALIBRATED)
- [x] Track B: I/Q receiver base layer
- [x] Track I: topology chirality map (route 4:5 selected from T300, 6/6 seeds)
- [x] Track A: 3 Phenom II hardware architectures, 12/12 controls, all negative (L4)
- [x] Track D: commutator reference negative, multi-seed AUC 0.516 (L3)
- [x] Track F: candidate HW accumulation, weak seed-dependent hint, orientation FAIL_CHANCE (L3)
- [x] MASTER SYNTHESIS written
- [ ] Track C: chiral QFT -- deferred. Roadmap assigned ROL->c0, ROR->c1 manually. Needs public-derived redesign (e.g. LSB parity of candidate*k_j)
- [ ] Track E: geometric phase loop -- deferred. Roadmap assigned cw->c0, ccw->c1 manually. Needs public-derived redesign
- [ ] Track G: sideband/bispectrum -- never built. Sprint 3 not reached before paradigm shift to sub-phase 14
- [ ] Track H: collision sieve -- never built. Sprint 3 not reached
- [ ] Track K: harmonic resonance sweep -- never built. Sprint 3 not reached
- [ ] Track L: passive external witness -- never built. Sprint 4 not reached; optional
- [ ] Track M: synthetic quadrature ladder -- never built. Sprint 4 not reached; calibration
- [ ] Track N: hardware decodability boundary ledger -- merged into MASTER SYNTHESIS

---

### 13 -- Substrate L2/L3/L4 Attempt (PARTIAL)

**Why:** The SPEC defined a ladder for testing the catalytic substrate on Phenom
II hardware: L2 (tape lifecycle works), L3 (fixed-point loop converges), L4
(recovers `d` under no-smuggle controls). The charter from Exp 49.14 identified
the untested lever: "on a reversible/CTC fixed-point substrate, fix(f) is found
in poly." This was the implementation attempt.

**Key result:** **L2 and L3 pass. L4 is blocked by three independent reasons.**
L2: SHA-256 tape lifecycle (record -> XOR mutate -> compute -> XOR restore ->
verify) works on Phenom II (50/50, 4 controls). L3: contraction map
`f(x)=floor((x+42)/2)` converges in 2-9 iterations (90/90). L4 blocked:
(A) `f(x) = x if verify(x) else (x+1)` is ordinary forward scan -- wrapping it
in SHA tape is ceremonial. (B) `verify(x)` is fold-even -- accepts both `d` and
`N-d`. (C) The charter's restricted-domain `d in [1, N/2)` collapses target to
public candidate value, not true orientation. L4 requires a genuinely new
substrate mechanism, not another public-verify C loop.

**The paradigm boundary:** L2/L3 were later **downgraded** by the noncollapse
doctrine (sub-phase 14) from "frontier gates" to "mechanical warmup." The
substrate frontier is now the DEFINITION of a mechanism that accesses
orientation without scanning verify(x) sequentially -- not the implementation
of one.

- [x] L2: catalytic tape lifecycle on Phenom II -- 50/50 SHA restore, 4 controls
- [x] L3: contraction map convergence -- f(x)=floor((x+42)/2), 90/90, 2-9 iterations
- [x] L4 design audit: three blockers identified
- [ ] L4: Exp50.14 oracle recovery -- blocked: forward scan + fold-even verify + target collapse. Needs new substrate mechanism, not another verify loop
- [ ] L5: multi-seed/session -- blocked on L4
- [ ] L6: independent reproduction -- blocked on L4

---

### 14 -- Noncollapse OrbitState Frontier (ACTIVE)

**Why:** After sub-phase 12 closed the forward-detector campaign, the diagnosis
shifted. The old framing -- "find a better detector," "which candidate wins,"
"what is the AUC" -- was identified as **median-basin collapse**: asking for a
scalar output from a higher-dimensional object. "The algorithm is dead" means:
stop iterating verifier loops and scoring candidates. The primitive object must
be the unresolved fold pair {a, N-a} as one OrbitState that evolves through
phase/path/tape/substrate relations without collapsing to a scalar. Invariants
are extracted ONLY at the explicit CollapseBoundary, after evolution.

**Key result (so far):** L4B OrbitState evolution primitive passes at L1. 512
steps of coupled evolution with `fold_symmetry = HOLDS` (invariant_imag = 0.0).
.holo record written with forbidden fields absent. No orientation recovery
claimed. The architecture is defined: 12 axioms (no scalar candidate, no
verify(x), no AUC, evolution precedes measurement, CollapseBoundary is explicit).
L4A mechanism screen assessed 6 classes; Class B PDN carrier validated (carrier
live, Q_diff core-dependent, no residue). L2/L3 downgraded to mechanical warmup.

- [x] Median-basin corruption audit -- ALGORITHM IS DEAD confirmed
- [x] Non-collapse substrate architecture -- 12 axioms, 5 state objects
- [x] L2: tape lifecycle -- downgraded to mechanical warmup
- [x] L3: contraction map -- downgraded to mechanical warmup
- [x] L4: public verify route rejected -- design audit
- [x] L4A: mechanism screen protocol -- 6 classes assessed
- [x] L4A Class B: PDN carrier validation -- carrier live, Q_diff core-dependent, no residue
- [x] L4B: OrbitState evolution primitive -- L1, 512 steps, fold_symmetry=HOLDS
- [x] L4B.1: `.holo` geometric memory / `HoloGeometry` -- complex phase is one carrier coordinate, not the architecture

**Catalysis Is The Hologram:** phase is carrier, geometry is memory, the
algorithm is a local trace, and an invariant is extracted only at
CollapseBoundary.
- [ ] L4B.2: reversible path-history accumulator -- PathStep recorded but not yet accumulative
- [ ] L4B.3: expanded .holo evolution transcript -- only final state written
- [ ] L4B.4: invariant family beyond fold_symmetry=HOLDS
- [ ] L4B.5: physical substrate mapping -- after primitive matures

---

## 2. Dependency Graph

```
02_fold_audit (no-smuggle gate -- foundation for everything)
  |
  +---> 03_nonhermitian_sensors (6 sensors, referenced by 04 as "already tried")
  |       |
  |       +---> 04_holo_phase_substrate (capstone -- quadrature ~0 proven)
  |               |
  |               +---> 05_black_hole_eigen (corrective -- compute inside coset space)
  |                       |
  |                       +---> 06_superradiant_sieve (chiral loophole -- no gain)
  |
  +---> 08_chiral_phase_kickback (pre-projection encoding -- FAIL_CHANCE)
  |
  +---> 09_transient_fold_probe (closes transient open crack -- FAIL_CHANCE)
  |
  +---> 12_chiral_lane_frontier (15-track forward campaign -- CLOSED)
  |       |
  |       +---> Track I uses T300 data from 10_cross_core_wormhole (route 4:5)
  |       |
  |       +---> 14_noncollapse_frontier (paradigm shift after Sprint 2)
  |               |
  |               +---> Stream 1 (L4B OrbitState -- ACTIVE)
  |               +---> Downgrades 13's L2/L3 as "mechanical warmup"
  |
  +---> 10_cross_core_wormhole (cross-core PDN physical channel -- DONE)
          |
          +---> 11_pdn_catalytic_tape (post-mortem explaining sim/hardware gap)
          +---> 07_dram_rowbuffer (early sim prototyping for cross-core)

5.10 (basin prep) --HARD GATE--> 13_substrate_frontier (Mode C -- BLOCKED)
                                    |
                                    SPEC.md (design only)
```

## 3. Priority Order

**1. 10_cross_core_wormhole (Stream 2, complete)** -- The completed T300 result
selected route 4:5 and established the PDN carrier result recorded above.

**2. 14_noncollapse_frontier L4B.1-L4B.5 (Stream 1)** -- The theoretical
frontier. L4B.1 is the immediate blocking item: `.holo` geometric memory /
`HoloGeometry`. Complex phase is one carrier coordinate inside that architecture,
not the whole architecture. L4B.2-L4B.5 build in sequence on L4B.1. Gate: no
scalar collapse. No AUC. No verify(x).
The answer, if it exists, surfaces as an invariant at the CollapseBoundary.

**3. 5.10 Gate** -- Hard prerequisite for Stream 3 (Phase 6 Mode C). 5.10C
(reproducible basin selection) is not complete. Without it, a Mode C null is
uninterpretable (basin failure vs. substrate failure) and a Mode C positive
could be a basin artifact. Current 5.10 status: rail-invisible-software,
basin scan not completed. 5.10D produced a cache/address topology witness but
it is below the encoding wall.

**4. 13_substrate_frontier L4 (Stream 3)** -- Only after 5.10C passes. The
catalytic fixed-point loop cannot proceed until a prepared, instrumented,
controlled boundary basin exists. The current L4 design (naive forward scan
wrapped in SHA tape) is structurally identical to what sub-phase 12 already
tested and rejected. A new substrate mechanism definition is needed.

**5. 12_chiral_lane_frontier Tracks C/E** -- DEFERRED, not dead. Tracks C
(chiral QFT) and E (geometric phase) were rejected at staging because the
roadmap designs assigned operation direction to candidate labels manually.
Public-derived redesigns exist on paper: Track C could derive ROL/ROR from
`parity(candidate * k_j mod 2)`, Track E from `parity(candidate * K_mid mod N)`.
If Stream 1 or Stream 2 produces a crossing, these tracks may become relevant
as target generators. If not, they are redundant with the already-completed A,
D, and F measurements.

**6. 12_chiral_lane_frontier Tracks G/H/K/L/M** -- NEVER BUILT because the
campaign pivoted after Sprint 2. Sideband/bispectrum (G), collision sieve (H),
harmonic resonance (K), external witness (L), and synthetic quadrature ladder
(M) were not reached. They remain in the legacy roadmap as documented attack
vectors. None are currently blocking.

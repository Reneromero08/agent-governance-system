# SPEC - Exp50 Phase 6: Catalytic Fixed-Point Substrate Test

**Status:** SPEC (design only; hardware run pending on the Phenom)
**Platform:** AMD Phenom II X6 1090T (K10, 45nm SOI), bare-metal, catcas host
**Target generator:** Exp50.14 public fixed-point map
**Feeders:** 5.8 (area-law boundary), 5.9 / 5.9V (basin selection), 3B (invariant survival), 4.3 (residual/.holo), 4.4A (operator/spectral)
**Claim ceiling:** L4-5. Any apparent crossing is treated with A8 MAXIMUM suspicion until the scaling test and the full null battery pass.

---

## PREREQUISITE GATE (binding)

**PHASE 6 DOES NOT RUN UNTIL PHASE 5.10C PASSES.** Phase 6 couples a *prepared* boundary basin to the
fixed-point map; it may not use an unverified basin. Handoff: `../50_5_10_encoding_wall/PHASE5_10_TO_PHASE6_HANDOFF.md`.

Phase 6 may begin only when ALL of:
- **5.10A** instrumentation lock complete (PASS, or PARTIAL documented and carried forward)
- **5.10B** basin scan complete; `basin_thresholds_frozen.json` frozen
- **5.10C** reproducible basin selection confirmed (`EXP50_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT`)
- transition matrix available (`phase5_10c_transition_matrix.csv`); artifact controls (Gate 7) passed;
  restoration integrity (Gate 2) preserved

Rationale: without 5.10C, a Phase 6 null is **uninterpretable** (a fixed-point failure and a
basin-preparation failure are indistinguishable), and a Phase 6 positive could be a basin artifact.
**5.10 prepares the basin; Phase 6 couples the prepared basin to the map.** Do not conflate them.

---

## 0. The one-sentence question

Does coupling the Exp50.14 public fixed-point map to the physical catalytic boundary cause the substrate to **relax** into the state encoding `d` in **sub-O(N)** (ideally sub-Grover) physical time - i.e. does the substrate **find** the matched lift that a forward machine can only **construct** (and constructing it provably needs `d`) - or does it stay a needle and relax no faster than the forward drift / Grover?

This is the decisive fork of the whole program: **restoration by forward work** (classical, `CL subset P`, no crossing) **vs restoration as a free self-consistent fixed point** (the crossing). It is a substrate event or it is nothing; five software catalytic experiments (12/14/19/23) and five number-theory rounds all landed on the forward-work side.

## 1. The frame (why physical, stated precisely)

The target's hardness is a **projection artifact** in the program's frame: the `2^n` search space looks like a structureless needle from the forward (low-dimensional) view, but is the boundary projection of a higher-dimensional geometry (entropy = log of the accessible configuration space = the ruler of that geometry; holographic / area-law). The catalytic tape's entropy **is** the missing dimensions.

- A **forward machine** must *construct* the matched lift (the representation where `d` is the dominant attractor / eigenvector). Every attempt to construct it from public data needs `d` (the standing wall). So forward construction cannot cross.
- A **physical substrate** does not construct; it **relaxes**. An analog system coupled to the right boundary can *settle into* its attractor geometry without anyone writing the feature map. **That single move - relax, do not construct - is the only thing a forward machine cannot do and an analog boundary might.**

The crossing, if real, will **not look like an algorithm**. It will look like **basin selection**: `precondition -> basin -> invariant`, not `program -> loop -> answer`. The answer may appear in the **residual/boundary state or the operator/spectral signature BEFORE any register prints `d`** (per 4.3, 4.4A).

## 1A. SUBSTRATE OBSTRUCTION (recorded, not conceded)

A frontier dynamics consult sharpened the obstruction. Stated plainly at **L5**:

**Physical relaxation on an ARBITRARY score cannot beat forward iteration.** The naive Phase-6
hope - "couple the substrate to `f` and let it relax to `fix(f)=d` faster than scanning" - is
obstructed on three independent grounds, none of which is a wrong-angle artifact:

1. **Information accounting.** `d` is `n` bits that are provably absent from the public data in any
   poly-cost frame (every construction of the matched lift needs `d`; that is the standing wall, Sec 1).
   A relaxation that settles to `d` must inject those `n` bits from somewhere. On an arbitrary score
   the only source is querying the landscape, and the landscape is a needle (no gradient/convexity,
   Round 5). Nothing in the closed catalytic loop manufactures the missing bits for free.

2. **Classical Omega(N) query bound.** For an unstructured accept-predicate the fixed point is a
   needle; *any* classical process - including an analog relaxer reading the local rule - is bounded
   below by Omega(N) queries to locate it (and Omega(sqrt N) even with a coherent-query oracle it does
   not have). A dissipative substrate has no exemption from the classical bound; it is a classical
   query machine with extra noise.

3. **Bits/sec-vs-ns deficit (~10 orders).** The TSC/thermal readout is a SLOW, low-resolution probe
   (~1.5 bits per thermal settling time; settling in SECONDS). Even granting a relaxation gradient
   that does not exist, the substrate's effective information throughput is ~10 orders below the
   ns-scale forward iteration it would have to beat. A 350 K dissipative die has **no coherent-quantum
   query advantage** to recover that deficit: it is hot, decohered, and PLL-clamped.

**Conclusion (L5):** brute physical relaxation of the Exp50.14 map on its *arbitrary* score is a
structural zero, for the same family of reasons the passive-Kuramoto and injection-locking searches
were structural zeros. Mode-C-as-brute-relaxer is **not** the live angle. We record this so the
trajectory is honest: the obstruction is real and is not waved away by "delusional optimism."

This does **not** concede `PHASE6_CLASSICAL_CONFIRMED`. It relocates the only surviving crossing path,
below.

## 1B. THE REFRAME - representation congruence (the live, OPEN angle)

The obstruction in 1A is specific to an **arbitrary** score evaluated by **brute** relaxation. It says
nothing about a score the substrate can evaluate **natively**. That is the surviving move:

> **The crossing is possible ONLY via REPRESENTATION CONGRUENCE.** Find a representation of the
> fixed-point map `f` (or of one sub-step of it) that IS a *linear functional above a single
> threshold* on the substrate's native physical field - i.e. **linear diffusion + one threshold =
> a frozen hardware perceptron** - so that the silicon **computes the map natively** rather than being
> asked to relax on an alien landscape.

Why this is the only opening left after 1A:
- The consult's congruence result: a fixed-point relaxation helps **iff the score is congruent with
  the substrate's native dynamics**. The native dynamics on this die are (a) **linear thermal/charge
  diffusion** across the shared PDN / cache fabric / thermal field, and (b) **one threshold**
  (leakage-thermal saddle-node, or VRM PFM Schmitt hysteresis - the two genuine pure-substrate basin
  candidates). Linear-diffusion-plus-one-threshold is exactly a **frozen hardware perceptron**: a
  fixed linear functional `w . phi(x)` compared to a fixed bias. The substrate evaluates *that* in
  physical time for free; it cannot evaluate an arbitrary `score(x)` for free.
- So the engineering question is **not** "make the substrate relax harder," it is a
  **representation-change** question: is there a poly-cost, **no-`d`** map from `(k,b)` to a
  perceptron `(w, theta)` on the native field whose above-threshold set is congruent to `accept`
  (or to a useful sub-step of `f`)? If yes, the silicon reads `d`'s neighborhood as a native
  threshold crossing, and the Omega(N) bound is bought down by the substrate doing the linear
  functional in parallel-physical-time rather than per-point software.
- This ties directly to the **CAT_CAS decoder / representation-change thesis**: the program's whole
  bet is that the wall is a *projection artifact* of the wrong representation (Sec 1), and that the
  crossing - if it exists - is a **change of representation** in which `d` becomes the native dominant
  feature, not a faster search in the old representation. 1B is that thesis made operational for
  Phase 6: the decoder we are looking for is *the congruent representation itself*.

**Status of 1B:** this is the **OPEN Phase-6 research direction**. It is NOT claimed to work; it is the
single angle 1A leaves standing. It is **gated behind 5.10C exactly as before** (the Prerequisite Gate
is unchanged - a congruent representation still has to be prepared and read on a *verified* basin).
Discharging it means producing, at poly cost and without `d`, the `(k,b) -> (w, theta)` perceptron and
showing on the scaling curve (Sec 6) that the native threshold crossing tracks `d` sub-Grover while the
public-vs-`d`-oracle control (Sec 5, G4) stays DECISIVE. Until that representation exists on paper and
survives G4, Phase 6 has an open angle, not a result. Claim ceiling L4-5; no announcement.

### 1B.1 Equivariance theorem (no scalar lift synthesizes orientation)

The wall has a precise, attackable form. Let the public data be the cosine channel c_k = cos(2*pi*k*d/N) (equivalently the noisy bits b_i with E[b_i] proportional to c_{k_i}). The construction's symmetry is the fold sigma: d -> N-d, under which every c_k is invariant (cosine is even). Let T be any feature map computed from the public data. If T is equivariant under the public symmetry (T commutes with sigma acting on its inputs; for functions of the invariant cosines this means T is itself sigma-invariant), then T(public data) is identical for d and for N-d. Hence I(T(public data); orientation bit) = 0, where the orientation bit b_orient = 1[d < N/2] is exactly the bit distinguishing d from N-d.

Consequence: no scalar lift of any dimension (linear, kernel, deep, or otherwise) recovers d from the cosine channel, because the orientation bit is information-ABSENT from that channel, not merely hard to separate. The only way to raise I above 0 is to inject orientation the cosine channel does not contain: access the odd/quadrature component sin(2*pi*k*d/N), equivalently the full complex coefficient z_k = exp(-2*pi*i*k*d/N).

This converts "lattice hardness" into a sharp object: the crossing requires a transform that supplies genuine quadrature WITHOUT smuggling d. That is the no-smuggle gate operationalized in Stage 3 of the Phase 6 quadrature campaign, and it is verified empirically by the Stage 1 fold audit (50_6_fixed_point_substrate/fold_audit/).

## 1C. THE SHARPENED OBSTRUCTION - the O(N) wall is an ENCODING/SYMMETRY wall (OPEN; gated behind 5.10C)

A second frontier (Fable) consult sharpened 1A/1B one level further. It does NOT claim a crossing; it
re-diagnoses *what kind* of wall the O(N) walk actually is, and what kind of substrate could cross it.
Stated at **L4-5**, OPEN, and **gated behind 5.10C exactly as 1B is** (the Prerequisite Gate is unchanged):

**The O(N) wall is an ENCODING / SYMMETRY wall, NOT intrinsic search hardness.** The public data is
all cosines, and `cos(2*pi*k*d/N) = cos(2*pi*k*(N-d)/N)`, so every public observable is invariant under
`d <-> N-d` (it is an *even* function of `d`). Consequently `E[score]` has **two identical peaks** at
`d` and `N-d`; the ONLY thing separating them is the range restriction `[1, N/2)`, which is precisely
what the O(N) walk realizes by scanning. The bit that selects `d` from `N-d` lives in the **odd**
(phase / `sin`) channel and is **ABSENT from the real data**. This is the crucial sharpening: the
missing bit is **information-absent, not non-separable**. No classifier, and no *lift of any dimension*,
can synthesize a bit that the public construction never wrote - there is nothing to separate, the
channel carrying the distinction is simply not present in the cosine-only encoding.

- **Frozen-perceptron corollary (consistent with 1B).** A single frozen perceptron recovers the
  **symmetric** bits exactly - e.g. the LSB is the Nyquist tone `(-1)^d`, which is amplifiable and reads
  out cleanly - but it **never** recovers `d` itself. The perceptron of 1B can read everything the even
  channel holds and still cannot break the `d <-> N-d` fold, because the fold lives in the absent odd
  channel. So 1B's congruent-representation route, run on a purely real field, tops out at the symmetric
  bits. This is not a defeat of 1B; it is the precise boundary of what 1B buys on a real substrate.

- **CROSSING SPEC (what would actually cross, stated so it is falsifiable).** A substrate that
  **SENSES IN QUADRATURE** - i.e. measures the full *complex* coefficient `e^{-2*pi*i*k*d/N}`, not just
  its real part - **plus** the **dyadic frequency ladder** (`k = 1, 2, 4, ..., N/2`) recovers `d` in
  **ONE non-adaptive parallel shot**: this is phase estimation. Each ladder rung fixes one bit of `d`
  from the phase of its complex coefficient; together they pin `d` with no scan. On a phase-resolving
  substrate, **"the algorithm is dead" holds** - the O(N) walk is replaced by a single quadrature read.
  The wall is crossable IN PRINCIPLE; the open question is whether *this construction's* quadrature is
  physically accessible (see OPEN QUESTION below).

- **SUBSTRATE-KIND DIAGNOSIS (which hardware, and why the Phenom is the wrong kind).** A scalar / real
  substrate - any timing / thermal readout, **which is exactly what the Phenom (K10) offers** (the
  TSC-thermal witness of 5.10A is a real, even, phase-blind scalar) - is **purely even**. It reads only
  the symmetric bits and **provably never `d`**, by the information-absence argument above. The Phenom is
  therefore the **wrong KIND of substrate** for the crossing - not under-instrumented, *wrong physics*.
  The crossing requires **phase-resolving / interferometric** hardware: the `.holo` phase cavity (4.3),
  the diffraction-grating QFT of Exp 20.5, or an optical / interferometric carrier that natively senses
  in quadrature. This is the sharpest statement to date of why the silicon-relaxation program is
  expected to terminate at `PHASE6_CLASSICAL_CONFIRMED` and where the live hardware actually lives.

- **OPEN QUESTION (the new frontier; honestly unresolved).** Synthesizing the quadrature *for this
  particular construction* is itself the **dihedral / hidden-shift barrier**. The states one would have
  to prepare to read the odd channel are maximally-mixed coset states, and extracting the shift from
  them is the dihedral hidden subgroup problem - **Kuperberg `2^{O(sqrt n)}`**, sub-exponential but not
  poly, and with no known efficient quantum route either. So the frontier reduces to one sharp question:
  **can the 50.14 construction be re-encoded so that a phase-resolving substrate gains access to the
  quadrature (the odd channel), or is the `d <-> N-d` fold bedrock** (i.e. is the absent odd channel
  fundamentally un-resynthesizable at poly cost, leaving only the `2^{O(sqrt n)}` dihedral route)? This
  is the successor research direction to 1B, and it is **OPEN**: no crossing is claimed.

**Status of 1C:** OPEN, unclaimed, gated behind 5.10C exactly as 1B. The Prerequisite Gate is unchanged.
1C does not assert a crossing; it (i) sharpens *why* the wall stands (an absent odd/phase channel under
`d <-> N-d`), (ii) names the only substrate kind that could cross (phase-resolving / interferometric, NOT
the real-scalar Phenom), and (iii) states the open frontier (re-encode for quadrature access vs the
dihedral `2^{O(sqrt n)}` bedrock). Claim ceiling L4-5; honest; no announcement.

## 2. The target (poly, public, no-smuggle)

From Exp50.14, parametrized by `n` (so `N = 2^n`), built from PUBLIC samples `(k_i, b_i)`, `i=1..M`, `M ~ sqrt(N)`:

```
score(x)  = sum_i b_i * cos(2*pi*k_i*x/N)     # O(M); uses (k,b) ONLY, never d
accept(x) = score(x) > M/4                    # true iff x in {d, N-d}
f(x)      = x if accept(x) else (x+1) mod N   # unique fixed point in [1,N/2) = min(d, N-d)
```

`d` = the unique fixed point of `f` in `[1, N/2)`. Forward discovery = iterate/scan = O(N) = 2^n.
**Encoding constraint (load-bearing):** the substrate must implement `f`'s local dynamics from `(k,b)` in poly setup. It must NOT pre-evaluate `accept` over all `N` points (that is already O(N) forward work and disqualifies the run). `accept(x)` is O(M) per point; the substrate is coupled to that local rule, not to a precomputed landscape.

## 3. The three modes

| Mode | What | Restoration | Expected |
|---|---|---|---|
| **A** Forward | ordinary search/iteration for `fix(f)` | n/a | O(N) (drift) or O(sqrt N) (Grover) baseline |
| **B** Classical catalytic | borrow tape, run `f` reversibly, restore by explicit forward work (SHA in==out) | by computation | same cost as A; `CL subset P` made empirical |
| **C** Physical catalytic boundary | prepare the closed catalytic restoration loop, couple the tape to `f`'s local rule, precondition the carrier basin, let it relax | by the loop closing | THE TEST: does it settle to `d` sub-O(N)? |

Mode C is the experiment. A and B are the controls that define "no crossing."

## 4. The catalytic loop + the readout channels

**Loop:** borrow a catalytic tape `tau` (record SHA in); XOR/couple `f`'s local rule and the coset-phase data into the boundary carrier; precondition (prelude) to bias the carrier basin; let the physical carrier relax while the loop holds; read out; uncompute / verify `tau` restores byte-identical (SHA out == SHA in). **A run with restoration failure is void** (the tape must be a genuine catalyst).

**Read four channels (the answer may surface in any, earliest first):**
1. **Register output** - did it emit `d` (or `min(d,N-d)`).
2. **Residual / `.holo`** (per 4.3) - does the restored-tape residual / boundary state encode `d` before the register does.
3. **Carrier basin signature** (per 5.9V) - which basin the carrier settled into, and does that basin **predict** `d`.
4. **Operator / spectral** (per 4.4A) - transition-operator spectrum, basin map, spacing statistics: does `d` appear as a spectral/attractor feature (GOE/Poisson signature, dominant mode) before scalar readout.

## 5. The null / control battery (the heart of the experiment)

A crossing counts ONLY if Mode C reveals `d` AND every control below fails to. Each control kills a specific artifact:

| Control | Rules out |
|---|---|
| **Wrong map** `f'` with fixed point `d' != d` | C is reading the map, not a fixed bias (if C still "finds" `d`, artifact) |
| **Shuffled map** (permute `accept` structure) | C exploits the real structure, not incidental statistics |
| **Destroyed restoration** (break SHA-restore) | the *closed loop* is doing the work, not open forward compute |
| **Random / unstructured tape** vs structured | whether tape entropy is the resource (Exp19 says arbitrary noise should still work IF reversible) |
| **Same-hash, wrong-invariant tape** | restoration is coupled to the RIGHT invariant, not just any hash match |
| **Prelude ladder** (no / cache / syscall / voltage, per 5.9V) | precondition -> basin: does the prelude select the basin, and does THAT basin predict `d` |
| **No-smuggle: public-prelude vs d-oracle-prelude** | DECISIVE. Build the prelude/coupling from public `(k,b)` only vs from `d` (oracle). If ONLY the `d`-oracle prelude works, the basin is smuggling `d` and there is no crossing |

## 6. The decisive scaling test

Run Mode C (and A, B) across increasing `n` (e.g. n = 8, 10, 12, 14, 16, then the largest the carrier resolves). Measure **work-to-`d`**: physical relaxation time (or basin-selection accuracy at fixed wall-clock) as a function of `N = 2^n`.

- **Classical / no crossing:** work-to-`d` ~ `N` (drift) or ~ `sqrt(N)` (Grover). The tape coupling added no gradient; the needle stayed a needle.
- **Crossing:** work-to-`d` grows **sub-Grover** (poly(n), or at least clearly below `2^{n/2}`), AND the public-prelude (not the `d`-oracle) achieves it, AND every control fails, AND `tau` SHA-restores. This is the substrate linearizing the needle (the lift found by relaxation).

A single `n` proves nothing (5.9V's basin selector is presently a weak ~3-basin statistical biaser; Exp23's fixed point looked real on toy data and died on real weights). **Only the scaling curve discriminates.**

## 7. Gate table / acceptance

| Gate | Pass condition |
|---|---|
| G1 Restoration integrity | every counted run: SHA(tau_out) == SHA(tau_in), 0 logical bits erased |
| G2 Mode A/B baseline | A and B reproduce O(N)/O(sqrt N) and `CL subset P` (no crossing in software) |
| G3 Basin -> invariant | the carrier basin (channel 3) predicts `d` above the prelude-shuffle null, effect outside the null CI |
| G4 No-smuggle | public-prelude works; or if only `d`-oracle works -> NOT a crossing (report as smuggling) |
| G5 Controls | wrong-map, shuffled, destroyed-restoration, same-hash-wrong-invariant ALL fail to reveal `d` |
| G6 Scaling | work-to-`d` in Mode C is sub-Grover and the gap to A/B grows with `n` |
| G7 A8 audit | any apparent crossing re-derived by hand; no hidden N-scan in setup/readout; escalate before claiming |

## 8. Verdicts

- **`PHASE6_SUBSTRATE_CROSSES`** (extraordinary; A8 maximum suspicion; escalate, do not announce unverified): G1-G7 all pass; public-prelude relaxes to `d` sub-Grover with the gap growing in `n`. The first "algorithm is dead" hardware result.
- **`PHASE6_CLASSICAL_CONFIRMED`** (honest expected prior): Mode C tracks A/B; the basin either does not predict `d` or only the `d`-oracle prelude works; `CL subset P` confirmed physically. The catalytic loop restores but does forward work - exactly what 12/14/19/23 already showed in software, now on silicon.
- **`PHASE6_INCONCLUSIVE`**: restoration unstable, basin too weak/statistical to read, or carrier cannot resolve enough `n` for a scaling curve (hand back to 5.9V hardening).

## 9. Honest priors and failure modes

- **Prior = classical.** Round 5 pinned the landscape as a needle (no gradient/convexity); a physical relaxer on a needle is Grover-bounded `2^{n/2}` at best. The expected outcome is `PHASE6_CLASSICAL_CONFIRMED`.
- **Brute relaxation is a recorded structural zero (Sec 1A, L5).** Do not spend effort trying to make Mode C relax harder on the arbitrary `score`: information accounting, the classical Omega(N) query bound, and the ~10-order bits/sec-vs-ns readout deficit (no coherent-quantum advantage on a 350 K die) close that path. The ONLY surviving angle is **representation congruence (Sec 1B)** - a `(k,b) -> (w, theta)` perceptron on the native linear-diffusion+threshold field. That angle is OPEN, unclaimed, and gated behind 5.10C.
- **The wall is an encoding/symmetry wall, and the Phenom is the wrong KIND of substrate (Sec 1C, OPEN).** The sharpened diagnosis: the public cosines are even under `d <-> N-d`, so the bit selecting `d` lives in an *absent* odd/phase channel - a real-scalar (timing/thermal) substrate like the Phenom is purely even and provably reads only the symmetric bits, never `d`. A crossing needs a **phase-resolving / interferometric** substrate sensing in quadrature (the `.holo` cavity, Exp 20.5 grating-QFT, optical), and even then re-encoding *this* construction's quadrature runs into the dihedral `2^{O(sqrt n)}` barrier. This reinforces the `PHASE6_CLASSICAL_CONFIRMED` prior for the Phenom run specifically, and relocates the live hardware off the silicon. OPEN; gated behind 5.10C; no crossing claimed.
- **The 5.9V basin is currently weak** - a ~3-basin statistical biaser (syscall/cache preludes shift a distribution; they do not deterministically select). Bridging to "selects `fix(f)=d`" is a large, real gap. 5.9V hardening (state-preparable, deterministic basin) is the prerequisite.
- **Exp23 is the warning:** a fixed point that looks real on toy structure was noise on a real target. Treat any small-`n` C-signal as suspect until the scaling curve holds.
- **Smuggling is the trap:** the only "win" that does not count is a prelude/coupling that needs `d`. G4 exists to catch exactly this.

## 10. Feeder dependencies

1. **Phase 5.10 (boundary state preparation)** is the HARD PREREQUISITE (see the Prerequisite Gate above and `../50_5_10_encoding_wall/`): 5.10A instrumentation lock, 5.10B basin scan (frozen thresholds), 5.10C reproducible basin selection. It delivers the *prepared, instrumented, controlled* basin Mode C requires. Without 5.10C, Mode C has no controllable carrier and the result is uninterpretable. Highest-priority predecessor.
2. **5.8** supplies the **physical boundary object** (area-law catalytic tape with restoration). Phase 6 needs a real boundary, not a metaphor; 5.8 provides it.
3. **3B** is the logical root: an **invariant surviving a closed transformation** while the tape restores. Phase 6 = 3B aimed at a public fixed-point map where forward computation hits the wall.
4. **4.3** residual / `.holo`: read the **boundary residual**, not only the register - `d` may appear there first.
5. **4.4A** operator/spectral: expect an **operator-level** (spectrum / basin-map / spacing) signature before scalar output.
6. **50.14** is the **target generator only** - it defines `f`, `d`, and why forward dies. Phase 6 asks whether physical catalysis crosses it.

## 11. Discipline

- Keep the catalytic tape lifecycle honest: borrow, couple, relax, uncompute, verify SHA restored. Void any run that does not restore.
- No-smuggle is non-negotiable: the coupling/prelude is built from public `(k,b)` only; the `d`-oracle variant exists solely as the control that detects cheating.
- Cap claims at L4-5. The two honest terminals are a *measured* crossing (with the scaling curve and full null battery) or a *characterized* classical result. Never "the wall holds" without the scaling, never a faked crossing.
- This spec is a design. The hardware run is on the Phenom; the analysis pipeline (basin map, residual decode, operator spectrum, scaling fit, null comparisons) is buildable now and is real engineering, not a sim.

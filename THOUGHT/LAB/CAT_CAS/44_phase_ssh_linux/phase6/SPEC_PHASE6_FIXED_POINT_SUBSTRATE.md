# SPEC - Exp44 Phase 6: Catalytic Fixed-Point Substrate Test

**Status:** SPEC (design only; hardware run pending on the Phenom)
**Platform:** AMD Phenom II X6 1090T (K10, 45nm SOI), bare-metal, catcas host
**Target generator:** Exp50.14 public fixed-point map
**Feeders:** 5.8 (area-law boundary), 5.9 / 5.9V (basin selection), 3B (invariant survival), 4.3 (residual/.holo), 4.4A (operator/spectral)
**Claim ceiling:** L4-5. Any apparent crossing is treated with A8 MAXIMUM suspicion until the scaling test and the full null battery pass.

---

## PREREQUISITE GATE (binding)

**PHASE 6 DOES NOT RUN UNTIL PHASE 5.10C PASSES.** Phase 6 couples a *prepared* boundary basin to the
fixed-point map; it may not use an unverified basin. Handoff: `../phase5_10/PHASE5_10_TO_PHASE6_HANDOFF.md`.

Phase 6 may begin only when ALL of:
- **5.10A** instrumentation lock complete (PASS, or PARTIAL documented and carried forward)
- **5.10B** basin scan complete; `basin_thresholds_frozen.json` frozen
- **5.10C** reproducible basin selection confirmed (`EXP44_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT`)
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
- **The 5.9V basin is currently weak** - a ~3-basin statistical biaser (syscall/cache preludes shift a distribution; they do not deterministically select). Bridging to "selects `fix(f)=d`" is a large, real gap. 5.9V hardening (state-preparable, deterministic basin) is the prerequisite.
- **Exp23 is the warning:** a fixed point that looks real on toy structure was noise on a real target. Treat any small-`n` C-signal as suspect until the scaling curve holds.
- **Smuggling is the trap:** the only "win" that does not count is a prelude/coupling that needs `d`. G4 exists to catch exactly this.

## 10. Feeder dependencies

1. **Phase 5.10 (boundary state preparation)** is the HARD PREREQUISITE (see the Prerequisite Gate above and `../phase5_10/`): 5.10A instrumentation lock, 5.10B basin scan (frozen thresholds), 5.10C reproducible basin selection. It delivers the *prepared, instrumented, controlled* basin Mode C requires. Without 5.10C, Mode C has no controllable carrier and the result is uninterpretable. Highest-priority predecessor.
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

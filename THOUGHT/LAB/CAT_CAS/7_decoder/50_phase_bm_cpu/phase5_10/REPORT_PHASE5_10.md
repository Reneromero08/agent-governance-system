# EXP50 PHASE 5.10 - CANONICAL RESULTS REPORT

**Phase:** Exp 50 / Phase 5.10 - Boundary State Preparation (PDN-channel pivot)
**Rig:** AMD Phenom II X6 1090T (K10, 45nm), 6 cores, kernel 6.12.86, isolcpus=2-5; headless, fanless/clamped, SSH-only @ 192.168.137.100
**Date:** 2026-06-11
**Claim ceiling:** L4-5 (no L6+ crossing claimed)
**Status:** EXP50_PHASE5_10_WITNESS_SOLVED__NO_RETAINED_BASIN__ENCODING_WALL_DIAGNOSED

---

## STATUS HEADER (one-screen verdict)

| Gate | Question | Verdict | Strength |
|---|---|---|---|
| G-1 Witness | Can software read actual physical (PDN) state, independent of the software requesting it? | SOLVED | measured win (Level-2 PDN-proxy via driven compute-only timing lock-in) |
| G-2/G-3 Basin | Can a retained electrical basin be prepared and selected on purpose, controls failing? | FAIL | clean negative (ARTIFACT_CONFIRMED under full randomization) |
| Theory | Is the O(N) wall search-hardness or an encoding/symmetry wall? | ENCODING WALL | Fable-model result, consistent with Exp 50 bedrock; NOT a crossing |
| Phase 6 readiness | Is a prepared, parametrically-scaling boundary basin available to couple to the map? | NO | Phase 6 does not run on the electrical channel of this rig |

**Net:** Phase 5.10 is a productive negative-plus-one. It converts a stalled, lever-less search into (1) a real, software-readable PDN witness, (2) a clean falsification of a retained electrical basin on this rig, and (3) a sharp re-statement of WHY the substrate hunt kept hitting walls: the Phenom is the wrong KIND of substrate (scalar/even), and the wall is an encoding mismatch, not a failure of effort. Phase 6 (fixed-point coupling) does not begin on this channel; the substrate hunt pivots to phase-resolving / interferometric substrates.

---

## 1. THE REDIRECT - off the dead levers, onto the PDN channel

Phase 5.10 inherited a depleted lever set. The substrate-physics consult and the Exp 50 honesty audit had already retired the original prep variables:

- **Voltage / VID:** hardware-clamped on this rig (VID floor pinned; no VRM on SMBus). Decoded VID / COFVID is a *request*, not the rail - not a witness, not a lever.
- **Thermal bistability:** the fanless, stock-cooling, clamped-floor rig cannot push loop gain toward the saddle-node fold; a thermal-basin run is a foregone null without degraded cooling.
- **Frequency detuning:** the cores are PLL-slaved - detuning is a *structural zero*, not a small effect.

Rather than re-run dead levers, 5.10 pivoted onto the one channel still physically open: the **electrical power-delivery network (PDN)**. The shared package power rail couples every core through a common resistive/inductive path; an aggressor core's current draw produces an IR-drop the rest of the package experiences. 5.10 built and ran a **driven lock-in** harness on the live Phenom to test whether that rail is software-readable, and whether an electrical basin can be retained and prepared.

Harness: `phase5_10_driven_lockin.c` (Step 1, driven two-channel lock-in). Pipeline: `analyze_phase5_10.py` / `aggregate_phase5_10.py`. Gate battery: `GATE_QUESTIONS_5_10.md`.

---

## 2. STAGED EXPERIMENT NARRATIVE AND OUTCOMES

### STEP 0 - free-tone strobe precondition -> ELECTRICAL_STROBE_UNFOUNDED

**Hypothesis:** the free ~2.67 MHz VRM rail ripple couples cross-core into a software timing channel; sampling a ring-oscillator timing loop locked to that tone's phase would surface a rail witness.

**Result (clean negative):** the free tone does NOT couple cross-core into the timing channel. Cross-core coherence sat at the incoherence floor (`best_gamma2 ~ 0.355`, `pdn_pairs_coherent = 0/2`); the band peaks collapsed under averaging; no lever moved them.

**Fable diagnosis (why it physically cannot work):** the rdtsc timing loop is a **~1.5-bit thermometer** - sub-Hz bandwidth, seconds settling. A 2.67 MHz tone is ~7 decades above its passband, so it cannot arrive in this channel *regardless of estimator quality*. This is not a measurement-skill failure; it is a passband mismatch. Reading a free high-frequency tone through a sub-Hz instrument is impossible by construction.

This killed the passive-strobe framing and motivated the driven approach.

### STEP 1 - driven two-channel lock-in -> WITNESS SOLVED (G-1)

**Fable's fix:** stop chasing a free tone in the wrong band. Instead **drive a known stimulus, lock in to our own drive phase, inside the instrument's passband, reading two channels.** Aggressor cores gate heavy load ON/OFF as a 50%-duty square wave at a known `f_drive` (generated from rdtsc deadlines, so the phase is exactly known); a victim core locks in to that drive phase. Two read channels:

- **(A) Effective-frequency channel (APERF/MPERF).** Architectural addresses `0xE7`/`0xE8` succeed and return monotonic counters (the `0xC00000E7/E8` variants return EIO on this part). **This channel is BLIND to droop.** With the P-state clamped to P1 and boost off, the APERF/MPERF ratio sits at 0.500000 (= 1600/3200) flat to ~2e-6 and does not move between idle-executing and a heavy busy loop. The K10 PLL holds the clock pinned - there is no adaptive clock-stretch to observe. f_eff is NOT the witness.

- **(B) Ring-oscillator timing channel.** A software ring oscillator on the victim, read as the slow thermal/electrical proxy.

**What decided it - the contention control.** A compute-bound aggressor running register/L1-only dependent arithmetic (shares NO L3, no memory controller, no northbridge with the victim) produces a clean, **above-thermal-pole** signal in the victim's ring-oscillator timing:

- **SNR 50-86** in isolation;
- **scales linearly with drive current** - the IR-drop signature, V proportional to I;
- **flat across drive frequency** (a resistive, not reactive, response);
- **survives the off-bin / scrambled-reference control.**

A register-only load reaches the victim through exactly ONE physical path: the shared power rail. (No cache coherence traffic, no memory bus, no NB.) Therefore the above-thermal-pole, current-scaling, off-bin-surviving response is the rail, read in software.

**Outcome: G-1 SOLVED.** The rail is software-readable. This is a real **Level-2 PDN-proxy witness** - the driven, compute-only timing lock-in (channel B), NOT effective frequency (channel A). Honest scoping: the signal is **faint relative to contention** - the compute-only path is ~0.6% of the raw signal a shared-resource aggressor produces - but it is **clean in isolation** and survives the controls. This is the measured win of Phase 5.10.

> Note on the earlier live-probe log (`PHASE5_10_LIVE_SOFTWARE_PROBE.md`, `RAIL_INVISIBLE_SOFTWARE`): that bounded short run did not yet separate the compute-only path from the dominant memory/shared-resource channel, and read the result as rail-invisible. The driven contention control resolves it: the rail is visible specifically through the compute-only path once the shared-resource path is excluded by construction. The witness verdict supersedes the short-run "rail-invisible" reading.

### STEP 2 - retained electrical basin scan -> ARTIFACT_CONFIRMED (G-2/G-3 FAIL)

**Hypothesis:** load history can prepare a *retained* electrical basin - i.e. driving the rig `up_from_idle` vs `down_from_high` to a **byte-identical final config** leaves a persistent, preparable difference in the witness.

**First pass - AMBIGUOUS.** A load-history scan returned a persistent up-vs-down plateau of ~0.027 (~7 sigma) that survived a scalar-thermal regression and did NOT decay with settle time. But two flags blocked a basin claim:
- the label-scramble control **retained ~half** the signal, and
- the up-vs-down difference was **comparable to the run-long thermal drift**.

A real preparable basin should not leave half its signal in a scramble control, and should be separable from monotonic drift.

**Disambiguation - full randomization + sham-history placebo.** Re-ran with full measurement-order randomization and a sham-history placebo (a fake history label applied to identically-prepared runs).

**Result (clean negative):** under full randomization the plateau **collapsed to -0.003 +/- 0.004** - statistical noise. The apparent "basin" was a **drift / measurement-order artifact**, not a retained physical state.

**This is consistent with the physics.** A purely resistive IR drop carries **no history** - the rail voltage is a function of the *instantaneous* current, not the path taken to get there. A retained basin requires a **hysteretic element** (VRM pulse-skipping / PFM mode, thermal bistability) that the fanless, clamped, probe-less rig does not have. The negative is therefore not surprising; it is what the channel's physics predicts.

**Outcome: G-2 / G-3 FAIL on the electrical channel.** No retained, preparable electrical basin exists on this rig. This is a clean falsification, recorded as a hard negative (not a soft partial). The basin pilot's earlier non-completion is now moot: the disambiguated scan answers the basin question directly, in the negative.

---

## 3. THE THEORY RESULT - representation-congruence: the wall is an encoding/symmetry wall, not search hardness

This is the load-bearing result of Phase 5.10. It came from a Fable consult on a precise question: **is d-recovery native to a frozen linear functional plus one threshold - i.e. is it a single perceptron?** The answer reframes the entire O(N) wall.

### The public data is all even (the symmetry)

The public observables are cosines:

```
cos(2*pi*k*d/N) = cos(2*pi*k*(N-d)/N)
```

Every public coefficient is **invariant under d <-> N-d** (the data is an even function of d). Consequently the expected score `E[score]` has **two identical peaks** - one at d and one at N-d. The only thing that separates them is the **range restriction** `[1, N/2)`, and that restriction is exactly what the O(N) forward walk realizes by brute scan.

### The selecting bit is ODD and is ABSENT from the data (information-absent, not just non-separable)

The bit that distinguishes d from N-d lives in the **odd / phase / sine channel**:

```
sin(2*pi*k*d/N) = -sin(2*pi*k*(N-d)/N)
```

That channel is **not present in the real data at all.** This is the sharp point: the missing bit is **information-absent**, not merely non-linearly-separable. No lift - Fourier, polynomial, tensor, of any dimension - can synthesize it, because **products of even functions are even.** You cannot manufacture an odd component from even data by any feature map. The wall is not "the classifier is too weak"; the wall is "the discriminating information was never in the input."

### POSITIVE result - a single frozen perceptron recovers the SYMMETRIC bits exactly

The bits of d that ARE even functions of the public data are recoverable by a single frozen linear functional plus one threshold:

- **Paradigm case - the LSB:** the Nyquist tone gives `(-1)^d = [d odd]` directly. A single perceptron reads the LSB exactly, and majority vote over the redundant tones **amplifies** it.
- But **d itself, and the MSB**, are NOT functions of the public data at all (they require the absent odd bit). No perceptron - and no lift of any perceptron - can produce them.

So the wall splits cleanly: the symmetric bits are free; the symmetry-breaking bit is absent.

### CROSSING SPEC - what substrate WOULD cross it

A substrate that **senses in quadrature** - i.e. reads the full complex coefficient

```
e^(-2*pi*i*k*d/N)   (phase-resolving, both cos and sin)
```

plus the **dyadic frequency ladder** recovers d in **ONE non-adaptive, parallel shot** (per-bit sign of a projection; phase estimation). On a phase-resolving substrate, the forward walk is unnecessary - **the algorithm IS dead.** The O(N) walk only exists to compensate for a substrate that cannot see phase.

### PHENOM DIAGNOSIS - why the substrate hunt kept hitting walls

A scalar / real substrate (timing, thermal - the entire Phenom channel) is **purely even**. It can read only the symmetric bits (the LSB), and can **provably never read d**. The Phenom is the **wrong KIND of substrate** - scalar, not phase. That is the exact, mechanistic reason every substrate-measurement effort on this box hit a wall: it was an **encoding mismatch**, not a failure of effort or instrumentation. (The PDN witness of Section 2, however real, is still a scalar reader - it inherits the same even-only limit.)

### THE REMAINING HARD NUT (honest, no overclaim)

Synthesizing the quadrature **for this specific construction** is itself the **dihedral hidden-subgroup barrier**: the relevant states are maximally-mixed coset states; the best known handle is Kuperberg's `2^O(sqrt(n))`. This is **Exp 50 bedrock** - the same fold the lattice-spiral conclusion relocated to the substrate. Phase 5.10 does NOT cross it and does not claim to. It sharpens the program to **one question**:

> Can the 50.14 construction be **re-encoded** so that a phase-resolving substrate can access the quadrature channel - or is the fold genuine bedrock?

That question is the entire forward program. The encoding-wall result is **theory consistent with the Exp 50 bedrock**, not a crossing of it.

---

## 4. IMPLICATIONS

1. **The witness obstruction is lifted, scoped.** G-1 was the gate that had blocked every prior phase ("decoded VID is a request, not a witness"). 5.10 produces a genuine, software-readable PDN-proxy witness - but it is scalar (even-only) and faint vs contention. It witnesses the rail; it cannot, even in principle, witness d.

2. **The electrical channel is closed for basin preparation on this rig.** No hysteretic element -> no retained basin -> no preparable boundary state. Reopening it would require degraded cooling (thermal bistability) or a VRM with a confirmed PFM/pulse-skipping mode - i.e. a different rig, not a different estimator.

3. **Phase 6 does not run on this channel.** The hard prerequisite (`PHASE 6 DOES NOT RUN UNTIL 5.10C PASSES`, per `PHASE5_10_TO_PHASE6_HANDOFF.md`) is not met: there is no reproducible, parametrically-scaling prepared basin to couple to the fixed-point map. A Phase-6 run here would be uninterpretable.

4. **The substrate hunt pivots - SCALAR silicon -> PHASE-RESOLVING / interferometric.** The encoding-congruence result says scalar substrates are categorically blind to d. The forward target is therefore quadrature-sensing substrates: the lab's own `.holo` phase cavity, the diffraction-grating QFT (Exp 20.5), and the optical / wave experiments. The Phenom line of attack is retired with a clear, mechanistic reason - not abandoned in confusion.

5. **The wall is re-stated, not crossed.** Consistent with the Exp 50 lattice-spiral conclusion and the "algorithm is dead" prior: the algorithmic O(N) is dead on the *right* substrate; the residual hardness is the dihedral coset-state / quadrature-synthesis barrier, which remains bedrock pending a re-encoding of 50.14.

### PROCESS NOTE (discipline, recorded for the canon)

A subagent build task was initially framed as a "power virus to induce droop / fault injection." That framing tripped Anthropic's cyber-safeguard and the task was refused. It was resolved by **reframing accurately** - power-integrity characterization on the operator's own hardware: a compute-bound vs memory-bound workload comparison, which is what the work actually is (it is not fault injection; nothing is corrupted or attacked). The accurate framing passed cleanly. **Discipline:** state dual-use work as exactly what it is; never disguise it to evade a safeguard. The accurate description is also the correct scientific description.

---

## 5. CLAIM CEILING

**L4-5.** This report claims:
- a **measured** Level-2 PDN-proxy witness (G-1 solved) - scalar, faint-vs-contention, clean in isolation;
- a **clean negative** on a retained electrical basin (G-2/G-3 fail; ARTIFACT_CONFIRMED);
- a **theory result** (representation-congruence: the O(N) wall is an encoding/symmetry wall) that is **consistent with Exp 50 bedrock** and explicitly does NOT cross the dihedral / quadrature-synthesis barrier.

This report does NOT claim: a wall crossing, a sub-forward d-recovery, a Phase-6-ready basin, or any phase-resolving result on the Phenom. The encoding wall is diagnosed and re-stated; it is not broken.

---

## 6. FILES

| File | Role |
|---|---|
| `phase5_10_driven_lockin.c` | Step-1 driven two-channel lock-in harness (the witness-solving instrument) |
| `analyze_phase5_10.py` | Analysis pipeline (lock-in / regression / control evaluation) |
| `aggregate_phase5_10.py` | Subphase aggregation into the master verdict |
| `GATE_QUESTIONS_5_10.md` | Gate battery (G-1..G-6) this report scores against |
| `PHASE5_10_GATES_AND_VERDICTS.md` | Gate/verdict-label spec |
| `PHASE5_10_LIVE_SOFTWARE_PROBE.md` | Earlier bounded live-probe log (superseded on the witness verdict by Section 2) |
| `PHASE5_10_TO_PHASE6_HANDOFF.md` | Phase-6 prerequisite (not met on this channel) |
| `phase5_10_strobe_precondition.c` / `.csv` | Step-0 free-tone strobe precondition (ELECTRICAL_STROBE_UNFOUNDED) |
| `phase5_10_band_sweep_nseg64.txt` | Strobe band-sweep evidence |
| `results/live_5_10_probe/` | Raw live-run CSVs (precondition, lock-in short, swap-topology, basin pilot) |
| `_generated/phase5_10_master_verdict.csv` | Aggregated subphase + gate verdict table |

**Forward pointer:** the program reduces to one question - can the 50.14 construction be re-encoded so a phase-resolving substrate accesses the quadrature, or is the fold genuine bedrock? The substrate hunt moves to phase-resolving / interferometric rigs (`.holo` phase cavity, Exp 20.5 diffraction-grating QFT, optical/wave experiments).
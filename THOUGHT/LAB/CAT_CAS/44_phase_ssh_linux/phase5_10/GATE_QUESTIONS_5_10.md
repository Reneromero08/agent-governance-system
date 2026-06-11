# GATE QUESTIONS 5.10 - the questions 5.10 must answer before Phase 6

**Status:** gate-question reference for Phase 5.10 (boundary state preparation). The per-subphase specs
(5.10A/B/C) reference this file.
**Provenance:** re-keyed from the Lattice-Hardness strategy battery (the 14-category gate set), reconciled
with the substrate-physics consult (which retired the voltage variable as a structural dead end) and the
Exp 44 honesty audit. Claim ceiling L4-5.

---

## 0. The central question

> Can a restored catalytic boundary be **prepared** into a reproducible, controllable physical state
> (a carrier basin) that is demonstrably **physical** (not a software/logical artifact), with the tape
> restoring bit-for-bit and matched controls failing?

5.10 answers ONLY the preparation question. It does NOT touch the lattice map, the fixed point, or `d`
(those are Phase 6). If 5.10 cannot answer the above, Phase 6 is uninterpretable.

## 0.1 The two re-keys (read before using this battery)

The strategy battery was written around **voltage**. The hardware retired that variable. Two corrections
propagate through every question below:

1. **Witness re-key.** Actual Vcore is NOT witnessable on this rig (VID hardware-clamped, no VRM on SMBus,
   COFVID = decoded VID = a *request*, not the rail). The canonical independent witness is the
   **invariant-TSC gate-propagation delay read as a slow thermal/electrical probe** (seconds settling,
   ~1.5 bits/settling), corroborated by effective frequency (APERF/MPERF), k10temp, and wall power if
   available. A decoded-VID change alone is NOT a witness.
2. **Prep-lever re-key.** The prep lever is NOT voltage/VID (clamped) and NOT frequency-detuning (cores are
   PLL-slaved -> a structural zero). The live levers are: **thermal band**, **load history**
   (up-from-idle vs down-from-high to the same final config), **power-virus dwell**, and the conditional
   **PFM-mode** and **PDN-resonance** channels. Computational preludes (syscall/cache/quiet) are logged as
   candidate carriers, not the primary lever.

---

## RESULTS (this session)

Realized gate outcomes from the 5.10 instrumentation/basin runs. Claim ceiling L4-5; honest.

- **G-1 (witness): SOLVED, but re-keyed.** The witness is the **DRIVEN compute-only lock-in on the rail**
  (SNR 50-86, IR-drop-linear, off-bin-clean): a register/L1-only aggressor reaches the victim ONLY via the
  shared rail. The strobe/free-tone path was **null** - the timing loop turned out to be a sub-Hz
  thermometer, not a PDN tone reader. The effective-frequency (APERF/MPERF) channel is **BLIND to droop**
  on K10 (no adaptive clock-stretch on this part). Actual Vcore stays `VCORE_MEASUREMENT_BLOCKED`; the
  Level-2 PDN proxy is achieved via the **timing channel** (the driven lock-in), not via Vcore or
  effective frequency.

- **G-2 / G-3 (basin definition + selection): FAIL on the electrical channel.** The load-history "basin"
  was `ARTIFACT_CONFIRMED` - a 7-sigma plateau collapsed to noise under full measurement-order
  randomization. Resistive IR drop carries no history, so there is **no software-preparable electrical
  basin**. A retained basin would require a **hysteretic element** (VRM-PFM / thermal bistability) that the
  fanless / clamped / probe-less rig does not have.

- **G-4 (parametric scaling / physical-vs-logical): the discipline WORKED.** The artifact was caught by the
  randomization + scramble controls, exactly as G-4 was designed to do. The central test did its job: it
  rejected a logical/measurement-order artifact that would otherwise have passed the letter of G-3.

- **The deeper crossing result (representation-congruence).** The O(N) wall is an **ENCODING / fold wall**:
  the public cosines are invariant even under `d <-> N-d`, so the `d`-vs-`(N-d)` bit is **ODD and
  information-absent** in the scalar readout. Crossing it needs **QUADRATURE / phase sensing** that the
  scalar Phenom cannot provide. The Phenom is the **wrong KIND of substrate** - not merely under-instrumented.

---

## GATE G-1 - INSTRUMENTATION WITNESS (re-keyed from battery section 3)

**Binding:** did the silicon physically change, witnessed independently of the software that requested it?

Executable tests:
- Does the TSC gate-delay shift reproducibly and OUTSIDE its own noise floor under the prep lever, at the
  THERMAL timescale (seconds dwell - not fast sampling)?
- Does the shift corroborate with an independent physical channel (k10temp, wall power), not only the
  decoded-VID register?
- Does the shift survive coherent-strobe averaging (sampling locked to the ~2.67 MHz VRM tone phase) above
  the raw-jitter floor?

**PASS:** a corroborated physical channel moves reproducibly with the prep lever, outside noise, at the
thermal timescale.
**BLOCKED:** only the decoded-VID register moves while TSC-delay + thermal + power do not ->
`PHASE5_10A_INSTRUMENTATION_BLOCKED` (decoded VID is a request, not a witness). A one-time bench DMM on a
safe Vcore point is OPTIONAL and not required to pass.

## GATE G-2 - BASIN DEFINITION (battery section 4)

**Binding:** what IS a basin, defined by frozen features before any selection is attempted?

Executable tests:
- Fix a basin feature vector (boundary thickness, timing-CV, D_eff, spectral entropy, TSC-thermal-delay
  regime). Commit it.
- Are collapsed / mid / high discrete clusters under UNSUPERVISED clustering, WITHOUT prelude labels?
- Are class thresholds frozen in `basin_thresholds_frozen.json` BEFORE any selection run?

**PASS:** a frozen, label-blind classifier maps the feature vector -> {collapsed, mid, high}, thresholds
committed pre-selection.
**KILL:** basins only appear when plots are inspected after the fact and named (post-hoc) -> not a basin
(M-3 risk).

## GATE G-3 - STATE PREPARATION / SELECTION (battery section 5, re-keyed)

**Binding:** can we choose the basin on purpose, with controls failing?

Executable tests (levers re-keyed to thermal / load-history, NOT voltage/VID):
- Does a chosen thermal-band / load-history prep bias a target basin above baseline, reproducibly over N
  repeats?
- Does selection survive shuffled prelude labels, randomized run order, and effective-frequency +
  temperature matching?
- Is the lift OUTSIDE the binomial CI of a label-reshuffle null, with multiple-comparison correction across
  preludes? (n=10 directional rates are not enough - this was the 5.9V gap.)

**PASS:** prelude A -> basin X, prelude B -> basin Y, lift outside null CI, controls fail, restoration
holds, witness present (G-1).
**KILL:** directional-but-not-deterministic again on the same family ->
`EXP44_PHASE5_10_NO_REPRODUCIBLE_BASIN` (a hard verdict, NOT a soft PARTIAL).

## GATE G-4 - SUBSTRATE IDENTITY (battery section 11 INTERSECT the consult's parametric-scaling null) - THE CENTRAL TEST

**Binding:** is the basin a PHYSICAL substrate state or a software/logical artifact? This is the decisive
5.10 test. Two independent advisors converged on the same discriminator from opposite directions: the
reset hierarchy and the parametric-physical-scaling null are the same test.

Executable - the reset hierarchy (where does the basin live?):
- Survives process restart?      survives -> not process-only software
- Survives full reboot?          survives -> not microarchitectural-only
- Survives power-cycle / reseed?  survives -> firmware / thermal / platform state
- Correlated with an external physical condition (die temp, cooling, rail)? -> hardware analog state

Executable - the parametric-scaling null (does it move with physics?):
- Does the basin / effect SIZE move monotonically with die temperature / cooling / rail conditions?
  Repeat fan-on vs fan-off; warm band vs cold band. A physical effect scales; a logical one is invariant.

**PASS:** the basin sits at or below the reboot-surviving / temperature-correlated tier AND its size scales
parametrically with a physical condition.
**KILL (downgrade):** the basin is invariant to cooling / temperature / rail. That is a logical / SRAM /
cache-replacement attractor - it can pass the letter of the G-3 selection control while being "the CPU as a
CPU through a side door." NOT a substrate basin for our purposes.

## GATE G-5 - FALSIFICATION / HARD REJECTION (battery section 10)

State these before running; they are non-negotiable:
- No restoration (SHA out != in outside an explicit destructive control) -> no catalysis; run VOID.
- No physical witness (G-1 BLOCKED) -> no hardware/substrate claim; cap at "computational carrier" only.
- Controls do not fail (shuffled / order / temperature-matched reproduce the selection) -> no selection.
- No parametric scaling (G-4) -> logical artifact, not substrate.
- Ceremonial tape (never XOR-mutated) or any pre-seeded answer -> discard (fraud-adjacent).

## GATE G-6 - PHENOM ADEQUACY / INSTRUMENT GAP (battery section 12)

Honest scoping, not a pass/fail:
- The Phenom may be adequate for basin PREPARATION (G-2/G-3) yet NOT for a strict Phase-6 crossing without
  an external witness. That defines the instrument gap; it does not kill the program.
- Log per run: requested VID, decoded VID, P-state, effective frequency, k10temp, wall power (if available),
  cooling state, BIOS settings, basin class, reset level.
- Portability: record enough that 5.10 can be reproduced on another board / CPU later.

---

## DEFERRED TO PHASE 6 (spec now, answer in 6.0A-6.0D - these are NOT 5.10 gates)

These battery categories assume a basin already exists and is being coupled to the target map. They are
premature before G-1..G-4 pass; spec them so the trajectory is coherent, answer them after 5.10C:
- **Sec 1 - minimal public map.** `EXP50_PHASE6_TARGET_MINIMAL.md`: the toy fixed-point map `f(k,b)` first
  (6.0A), NOT real LWE; the scaling ladder; the public verifier `V`.
- **Sec 2 - no-smuggle COUPLING battery.** Where `d` could leak once `f` couples to the tape. 5.10 has no
  `f`, so no `d` to smuggle yet; the brutal anti-smuggle battery (no preseeded d, score only after the run,
  wrong-map / shuffled-map / same-final-hash-wrong-invariant controls) belongs to Phase 6.
- **Sec 6 - coupling.** How the prepared basin touches `f` (and the danger of "prepare basin then run a
  normal forward verifier", which proves nothing).
- **Sec 7 - readout.** Single candidate vs distribution vs winding number; the posterior-lift framing
  (`P(d | basin) > P(d | null)`), not instant solve.
- **Sec 8 - scaling battery.** Does `P(d | basin)` scale below forward `O(N)` (the actual wall break)?
- **Sec 9 - fabric.** Flat tape is enough for 5.10; a multi-scale Feistel / MERA carrier is likely needed
  for Phase-6 scaling (Q57 fabric, Q28 attractor, Q54 SHA-as-Noether-charge, Exp12 active-tape prep).

---

## THE 5.10 PASS CONDITION

All of:
1. **G-1** witness present (corroborated physical channel; not decoded-VID-only).
2. **G-2** frozen, label-blind basin classifier.
3. **G-3** intentional selection: lift outside the null CI, controls fail, restoration holds.
4. **G-4** the basin is physical: reset-surviving AND parametric-scaling. (the central test)
5. **G-5** no rejection rule tripped.

-> `EXP44_PHASE5_10_READY_FOR_PHASE6`. Anything less is PARTIAL / BLOCKED / NO_REPRODUCIBLE_BASIN, and
Phase 6 does not run.

## THE THREE DECISIONS ONLY THE OWNER CAN MAKE (battery section 14)

1. **Degrade cooling** for the thermal-bistability run? (needed to push loop gain `beta = R_th * dP_leak/dT`
   toward the saddle-node fold; at stock cooling + clamped floor voltage it is a foregone null.)
2. **Check whether the VRM has a light-load PFM (pulse-skipping) mode?** (gates the PFM-hysteresis lever -
   a candidate 1-bit analog basin.)
3. **One-time bench DMM** on a safe Vcore point? (OPTIONAL; upgrades G-1 from BLOCKED-substitute to a direct
   rail witness, but is not required to pass 5.10.)

---

## The shortest version

To break the lattice wall the program needs five yes answers (battery, compressed):
1. Can we instrument actual physical state? (G-1)
2. Can we intentionally prepare a boundary basin? (G-3)
3. Is the basin physical, not a software artifact? (G-4 - the central test)
4. [Phase 6] Can that basin couple to a public fixed-point map without smuggling `d`?
5. [Phase 6] Does the basin bias readout toward `d`, and does that bias scale below forward search?

5.10 owns questions 1-3. Phase 6 owns 4-5. This file is the gate for 1-3.

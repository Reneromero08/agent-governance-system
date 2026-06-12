# PHASE4B_PHYSICAL_HOLO_PUSH_PLAN

## Verdict

`PHASE4B_PHYSICALITY_PUSH_READY`

Phase 4 Track A proves a catalytic `.holo` protocol on the Phenom tape substrate. The next push is not to return blindly to sine-wave Kuramoto. The better question is whether the `.holo` coordinates can be carried by a measurable physical substrate coordinate instead of only by explicit tape bytes.

Working hypothesis:

```text
CATALYTIC_RELATIONAL_PHYSICALITY_MAY_BE_MORE_FUNDAMENTAL_THAN_SINE_PHASE_LOCK
```

This document defines how to test that hypothesis without overclaiming physical holography.

## Distinction

Track A result:

```text
.holo variables are encoded in software/tape slots, transformed reversibly, decoded, and restored.
```

Physical `.holo` candidate:

```text
.holo variables are encoded into a physical carrier coordinate, and a physical readout predicts the
.holo state/residual/operator class while the logical tape still restores.
```

The required move is from "bytes carry `.holo`" to "bytes plus substrate state carry `.holo`, and the substrate component is measurable against nulls."

## Acceptance Labels

| Label | Meaning |
|---|---|
| `PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS` | A scalar physical carrier, such as timing/cache/PDN response, predicts `.holo` state above nulls while tape restores. |
| `PHASE4B_SUBSTRATE_COORDINATE_CONFIRMED` | The carrier is tied to address/core/layout/topology and survives held-out layouts. |
| `PHASE4B_PHYSICAL_OPERATOR_SPECTRUM_CONFIRMED` | Operator statistics computed from physical readouts separate from matched nulls. |
| `PHASE4B_RESTORED_TAPE_PHYSICAL_AFTERIMAGE` | Logical tape restores, but a bounded physical relaxation trace remains answer/residual predictive against nulls. |
| `PHASE4B_COMPUTATIONAL_ONLY` | All predictive structure is explained by explicit tape bytes or software schedule. |
| `PHASE4B_NULLS_MATCH` | Nulls match the physical readout; reject physicality claim. |
| `PHASE4B_PHASE_RESOLVING_REQUIRED` | Scalar physical carriers work as witnesses but cannot supply quadrature or Phase 6 crossing. |

## Gate Definition

A Phase 4B physical `.holo` candidate must pass all gates:

| Gate | Requirement |
|---|---|
| G1 logical restoration | The catalytic tape restores byte-identically after every counted run. |
| G2 physical readout | A non-byte readout predicts `.holo` state, residual tag, operator class, or mini-model class. |
| G3 matched work null | Equal instruction count / equal memory traffic null does not reproduce the readout. |
| G4 topology null | Shuffled address/core/cache-layout null does not reproduce the readout. |
| G5 wrong-invariant null | Same final hash but wrong residual/operator schedule does not reproduce the readout. |
| G6 held-out generalization | Classifier/threshold trained on one seed/layout predicts held-out seeds/layouts. |
| G7 claim ceiling | Scalar witnesses remain `PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS`, not physical quadrature or Phase 6 crossing. |

## Route 1: Address-Colored `.holo` Basis

Objective:
Encode the shared basis partly into physical address/cache-set layout rather than only into slot values.

Design:
- Allocate multiple aligned tapes with controlled address offsets.
- Keep logical `.holo` basis values identical.
- Assign basis vectors to address-color families.
- Run the same 4.1A/4.2A/4.3 sequence across address families.
- Read timing/cycle response per basis family.

Pass condition:

```text
Same logical .holo state + different physical address basis -> reproducible physical readout class,
and held-out address families predict the basis/residual better than shuffled layout nulls.
```

Why it matters:
This tests whether the `.holo` basis can live in topology, not just bytes.

## Route 2: Cross-Core `.holo` Lock-In Witness

Objective:
Drive known `.holo` operator classes on one worker set and read a physical response from another core.

Design:
- Reuse the Phase 5.10 driven-lock-in discipline, but drive with `.holo` modes:
  - basis-only
  - rotation-chain
  - residual-channel
  - mini-model
  - matched random reversible schedule
- Observer records cycle timing only.
- Analysis classifies which `.holo` mode was active from physical response.

Pass condition:

```text
Physical timing response predicts .holo mode/residual above matched-work and shuffled-schedule nulls.
```

Why it matters:
This would show the `.holo` protocol has a physical footprint, not just a logical transcript.

Claim ceiling:
On the Phenom this remains scalar/even physical evidence. It cannot by itself supply quadrature.

## Route 3: Restored-Tape Physical Afterimage

Objective:
Ask whether the substrate carries a short-lived physical trace after the logical tape restores.

Design:
- Run a `.holo` operator/residual schedule.
- Restore the tape.
- Immediately sample a bounded timing relaxation window.
- Compare against:
  - same final hash / wrong residual,
  - random reversible schedule,
  - destructive write,
  - idle restore,
  - shuffled operator order.

Pass condition:

```text
After restoration, the physical relaxation trace predicts the prior .holo residual/operator class
above all nulls, then decays as a physical trace should.
```

Why it matters:
This is the cleanest "catalytic is physical" test: the bytes are restored, but the substrate may still
carry a measurable relaxation coordinate tied to the reversible history.

## Route 4: Physical Operator Spectrum

Objective:
Repeat Phase 4.4A using physical readout vectors instead of software-generated operator matrices.

Design:
- For each `.holo` operator schedule, collect a vector of physical readouts:
  - per-core timing response,
  - address-layout response,
  - bounded relaxation samples,
  - repeated seed response.
- Build correlation/operator matrices from these physical vectors.
- Compare spacing/statistics against Poisson, shuffled operator, and equal-work nulls.

Pass condition:

```text
Physical-readout operator matrices separate from nulls and reproduce across held-out seeds/layouts.
```

Why it matters:
This upgrades 4.4A from software operator statistics to substrate-coupled operator statistics.

## Route 5: Phase-Resolving Future Track

Objective:
Name the route that would actually support "physical holo" in the stronger quadrature sense.

Design:
- Use an interferometric or phase-resolving carrier outside the scalar Phenom timing channel.
- Couple `.holo` basis/rotation/residual to a complex phase readout.
- Test whether the odd/quadrature component is measured without encoding the answer directly.

Pass condition:

```text
Quadrature readout predicts the missing orientation/phase component from public inputs without
answer-dependent setup.
```

Why it matters:
This is the stronger physical-holography route. It is not available from scalar Phenom timing alone.

## Recommended Next Experiment

Start with Route 3, then Route 2:

```text
PHASE4B_RESTORED_TAPE_PHYSICAL_AFTERIMAGE_PROBE
PHASE4B_CROSS_CORE_HOLO_LOCKIN_WITNESS
```

Reason:
Route 3 tests the cleanest catalytic physicality claim: logical restoration with a residual physical trace. Route 2 then tests whether `.holo` modes can be read by another core as a physical witness. Both are read-only, require no firmware changes, and directly extend the existing Phase 4 and Phase 5.10 harnesses.

## Do Not Claim Yet

- Do not claim physical holography from Track A alone.
- Do not claim physical Kuramoto from scalar timing.
- Do not claim Phase 6 readiness from scalar witnesses.
- Do not use voltage writes, firmware writes, board modification, or external physical changes for this route.
- Do not count a physical readout if explicit bytes or a schedule label explain it under held-out nulls.

## Exact Next Artifact Set

```text
session_scripts/phase4_holo/physical_afterimage_probe.c
phase4_holo/PHASE4B_RESTORED_TAPE_PHYSICAL_AFTERIMAGE.md
phase4_holo/results/phase4b_afterimage_summary.csv

session_scripts/phase4_holo/holo_lockin_witness.c
phase4_holo/PHASE4B_CROSS_CORE_HOLO_LOCKIN_WITNESS.md
phase4_holo/results/phase4b_holo_lockin_summary.csv
```


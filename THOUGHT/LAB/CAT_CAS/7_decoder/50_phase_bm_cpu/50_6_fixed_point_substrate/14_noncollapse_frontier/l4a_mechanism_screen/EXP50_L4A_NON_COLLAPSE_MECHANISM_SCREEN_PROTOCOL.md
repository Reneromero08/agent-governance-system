# Exp 50 L4A Non-Collapse Mechanism Screen Protocol

**Date:** 2026-06-18
**Status:** `PROTOCOL_CORRECTED__REFERENCE_ONLY`
**Role:** generic carrier/mechanism screening layer before L4B physical mapping.

---

## Doctrine

The primitive is an unresolved relational object, not a scalar candidate.

```text
No verify(x)
No candidate winner
No AUC-first adjudication
No hidden d in the runtime
No phase/sign assignment from truth labels
Measurement only at an explicit boundary
```

A physical screen may identify a measurable carrier coordinate. It does not thereby establish geometric memory, orientation, reversible physical path history, or catalytic restoration.

---

## State objects

| Object | Role |
|---|---|
| `FoldPair` | Public orbit `{a,N-a}` carried as one unresolved relation |
| `OrbitState` | Fold relation plus carrier coordinates, path declaration, and boundary state |
| `PhaseRelation` | Complex `I+iQ` or other declared relational phase coordinate |
| `PathHistory` | Ordered schedule/transform history; never a candidate score |
| `SubstrateMemory` | Measured physical channel state at the declared observability level |
| `CollapseBoundary` | Only point where a predeclared projection is materialized |

The canonical executable `.holo` object is defined in `../holo_runtime/HOLO_SCHEMA.md`. The old L4A `HoloRecord` scaffold is legacy provenance only.

---

## Mechanism-screen question

A mechanism screen asks:

> Does a predeclared physical coordinate respond reproducibly to the unresolved public relation under matched controls?

It does not ask:

> Which member of the fold pair is true?

The result is one of:

```text
CHANNEL_NOT_LIVE
CHANNEL_LIVE_RELATION_UNRESOLVED
RELATIONAL_COORDINATE_REPEATABLE
MEASUREMENT_INVALID
DESIGN_NOT_IDENTIFIABLE
```

None is an orientation verdict.

---

## Correct control semantics

### Same-orbit and dummy-orbit

These estimate route/core bias, drift, and noise. They are evaluated against a predeclared uncertainty distribution; exact numerical zero is not required.

### Assignment swap

Swap public orbit values across matched physical routes. Use crossed decomposition to separate value dependence from fixed route/core dependence. Do not mark the control passed without evaluating the paired measurements.

### Temporal-order reversal

Reverse acquisition order while keeping branch-indexed quantities defined identically. A genuine branch-indexed coordinate should remain invariant within uncertainty. A sign flip caused only by first-minus-second ordering is an artifact.

### Carrier-off

Acquire a real receiver trace with the carrier disabled. Never synthesize carrier-off data as literal zeros.

### Replay

The schedule, source digest, workload digest, and PRNG sequence must reproduce exactly. Physical I/Q/timing traces need only satisfy predeclared repeatability bounds; byte identity is neither expected nor required.

### Path shuffle

A predeclared schedule permutation tests path sensitivity. It must not be confused with random relabeling of truth or orientation.

### Session and route holdout

Use complete held-out sessions and routes. Do not split adjacent windows from one session across train and test.

---

## Generic phases

| Phase | Operation | Gate |
|---|---|---|
| P0 | Declare orbit, carrier, measured state, input schedule, controls, and uncertainty model | No hidden or post-hoc fields |
| P1 | Prepare public relational state | Both coordinates retained |
| P2 | Acquire baseline and positive carrier control | Detector/channel shown live or experiment stops |
| P3 | Apply controlled schedule/path | Equalized workload and timing |
| P4 | Acquire terminal complex state | Raw observations preserved |
| P5 | Apply crossed/control transformations | Core/value/path effects identifiable |
| P6 | Materialize predeclared projection at boundary | No candidate scoring |
| P7 | Evaluate repeatability and null distributions | Session-block uncertainty |
| P8 | Classify carrier/mechanism status | Claim ceiling enforced |

Tape restoration may accompany a screen as hygiene, but it is not evidence of physical substrate restoration unless the declared physical state and uncertainty gate are measured before, after, and after a valid closure operation.

---

## Mechanism classes

| Class | Carrier question | Current state |
|---|---|---|
| A | timing/phase-lane relation | concept only |
| B | PDN/common-mode complex response | crossed calibration source repaired; hardware rerun pending |
| C | cache/coherence path relation | unimplemented |
| D | branch/speculation path relation | unimplemented and weakly observable |
| E | thermal/metastability relation | low bandwidth; unimplemented |
| F | null | always valid outcome |

T300 is a positive channel-control result for sender-owned mode/phase transport. It is not a public fold-generation result.

---

## Statistical boundary

Thresholds must be frozen before adjudication and derived from preserved calibration data. At minimum:

- complete held-out session/route/schedule replicates;
- session-block bootstrap or equivalent grouped uncertainty;
- explicit within-input versus between-input separation;
- no seed selection after inspection;
- no scalar projection promoted over the complex state without justification.

The L4B.5B0 observability/operator design is the authoritative statistical framework for future acquisition.

---

## Claim ceiling

```text
L1: protocol/source architecture
L2: channel capture and artifact integrity
L3: repeatable relational coordinate under controls
```

L4 orientation recovery, physical geometric memory, and physical restoration remain separate future claims and are not authorized by L4A.

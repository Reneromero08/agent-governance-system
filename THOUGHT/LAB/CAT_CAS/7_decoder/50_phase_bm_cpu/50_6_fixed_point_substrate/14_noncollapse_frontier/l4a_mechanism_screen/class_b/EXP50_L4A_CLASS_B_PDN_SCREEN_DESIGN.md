# Exp 50 L4A Class B PDN Crossed-Assignment Screen

**Date:** 2026-06-18  
**Status:** `DESIGN_REPAIRED__NOT_RERUN`  
**Claim ceiling:** carrier/observability calibration only; no fold-odd residue or restoration claim.

---

## Executive correction

The previous design attempted to describe simultaneous same-phase branch drives while also requiring separate `Q+` and `Q-` branch windows. Those requirements were not jointly identifiable without an imposed multiplexing code. The committed runtime then executed sequential core windows whose workload did not depend on the branch value.

Class B is therefore repaired as a **crossed-assignment calibration**. It separates value-dependent response from fixed core/route bias without assigning truth, phase, sign, or preference to either fold coordinate.

The primitive remains the unresolved public fold orbit:

```text
O = {a, N-a}
```

No hidden `d`, candidate winner, verifier, AUC, or orientation label enters the runtime.

---

## Measured design

For each public orbit and matched schedule, acquire four real PDN lock-in captures:

```text
Q4(a)       core 4 drives value a
Q5(N-a)     core 5 drives value N-a
Q4(N-a)     core 4 drives value N-a
Q5(a)       core 5 drives value a
```

Define:

```text
D_normal = Q4(a)   - Q5(N-a)
D_swap   = Q4(N-a) - Q5(a)
R_value  = (D_normal - D_swap) / 2
R_core   = (D_normal + D_swap) / 2
```

Interpretation:

- `R_value` is the predeclared value-dependent coordinate after crossing out additive core bias.
- `R_core` is route/core asymmetry.
- Neither quantity is orientation or recovered `d`.
- A sign is not interpreted until repeatability and null controls are defined statistically.

The same decomposition is performed for complex lock-in output where possible:

```text
Z = I + iQ
R_value_complex = ((Z4(a)-Z5(N-a))-(Z4(N-a)-Z5(a))) / 2
```

The complex relation is primary. A scalar Q projection is a declared diagnostic, not the object.

---

## Workload requirement

The sender workload must consume the public orbit value in the executed integer path. The following are mandatory:

1. equal instruction count and capture duration across values;
2. identical tone, duty cycle, memory footprint, and schedule;
3. orbit value changes integer switching state inside every burst;
4. no branch-specific phase assignment;
5. no hidden secret or truth label;
6. sender core and orbit value are crossed across matched captures.

A runtime that merely prints `a` and `N-a` while executing a core-seeded fixed workload does not implement Class B.

---

## Control matrix

| Control | Acquisition | Expected interpretation |
|---|---|---|
| Carrier-off | Real receiver capture with no sender | Empirical noise floor; never synthesized as zeros |
| Same-orbit | `Q4(a), Q5(a)` and crossed repeats | Estimates fixed route/core bias; not required to equal exact zero |
| Dummy-orbit | `Q4(42), Q5(42)` | Operand-independent route bias and drift |
| Temporal-order reversal | Repeat the same branch-indexed acquisitions in reverse time order | `R_value` should remain invariant within uncertainty; it must not flip merely because order changed |
| Schedule shuffle | Same values, predeclared public schedule permutation | Tests path sensitivity separately from value response |
| Replay | Same declared schedule and seed | Configuration/workload digest identical; physical trace repeatable within uncertainty, not byte-identical |
| Session/core-pair repeat | Held-out sessions/routes | Tests portability and topology dependence |
| Carrier positive | Known sender-owned phase/mode pattern | Confirms detector/channel is live; this is the role of T300 evidence |

Null controls are evaluated against predeclared uncertainty distributions. Exact numerical zero is not required.

---

## Relationship to T300

T300 independently establishes that selected PDN routes can transport sender-owned mode and relational phase into measured lock-in I/Q. It does not establish operand-dependent Class B response.

Therefore:

```text
T300 = channel positive control
Class B crossed screen = value/core decomposition experiment
```

The two evidence classes must never be merged into one claim.

---

## Artifact contract

The repaired runtime writes an evidence JSON record, not a canonical L4B `.holo` object. Required fields:

```text
schema_id
N
a
mirror
core assignments
workload id and digest
capture order
I/Q/magnitude for each crossed acquisition
carrier-off captures
same-orbit captures
dummy-orbit captures
D_normal
D_swap
R_value_complex
R_core_complex
temperature and TSC metadata
execution status
```

The legacy L4A `HoloRecord` scaffold remains historical. Canonical `.holo` geometry is defined by `../../holo_runtime/HOLO_SCHEMA.md`.

---

## Verdicts

```text
CLASS_B_DESIGN_REPAIRED_NOT_RUN
CLASS_B_CAPTURE_INVALID
CLASS_B_CHANNEL_NOT_LIVE
CLASS_B_CORE_BIAS_DOMINANT
CLASS_B_VALUE_COORDINATE_UNRESOLVED
CLASS_B_VALUE_COORDINATE_REPEATABLE
```

`CLASS_B_VALUE_COORDINATE_REPEATABLE` is still only a calibration result. It does not imply fold orientation, physical geometric memory, or catalytic restoration.

---

## Hardware gate

Execution is deferred to the final SSH batch. Before a run counts:

- source must compile with warnings as errors;
- actual orbit values must affect the workload;
- carrier-off must be physically captured;
- crossed assignments must be complete;
- no control may be hardcoded as passing;
- raw captures and configuration digests must be retained.

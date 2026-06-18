# L4B `.holo` Geometric Memory Schema

## Status and claim ceiling

This document defines the L4B executable geometric-memory architecture.
It is an L1/L2 software primitive and an architectural hypothesis, not a
physics claim or evidence of physical restoration.

**CATALYSIS IS THE HOLOGRAM.** The hologram is the catalytic relation itself.
Phase is a carrier coordinate, geometry is memory, an algorithm is a local
traversal, an observable is a boundary projection, and the surviving invariant
is the memory of catalytic closure.

## Executable object

`HoloObject` separates seven roles:

1. `HoloGeometry` stores the unresolved fold-orbit coordinates, the relation
   basis that exchanges them, and a neutral reference. This is the memory.
2. `HoloCarrier` stores software/physical status, channel coordinates, and
   phase metadata. Phase transports or addresses the relation; it does not
   define `.holo`.
3. `HoloEvolution` identifies the reusable operator, step count, continuation
   state, and appendable `PathStep` reference.
4. `HoloProjection` declares the boundary operator and materialization mode.
5. `HoloInvariantFamily` predeclares typed invariant records and rejects final
   extraction until the `CollapseBoundary` has been crossed.
6. `CatalyticRestoration` records restoration references, closure law, and
   evidence level without claiming a physical restoration measurement.
7. `HoloCollapseBoundary` records the explicit projection event and the only
   point where invariant extraction is permitted.

The runtime flow is:

```text
relational geometry
-> carrier-mediated evolution
-> catalytic closure
-> explicit projection
-> invariant extraction at CollapseBoundary
```

`holo_geometry_render()` executes the native relation as
`neutral + basis * (coordinates - neutral)`. For the L4B fold-exchange basis,
this renders `[lower, mirror]` as `[mirror, lower]` while retaining both as one
unresolved orbit object.

## Lineage

### TINY_COMPRESS

The image `.holo` primitive stored coordinates, basis, mean/neutral reference,
and render depth. Rendering reconstructed the object through the stored basis.
L4B preserves that structure as orbit coordinates, relation basis, neutral
reference, and a declared projection. It stores geometry capable of producing
a boundary trace, not only the trace.

### HOLO operator geometry

The model `.holo` path evaluates `y = x @ SVh.T @ U.T`: coordinates traverse
factorized operator geometry without first constructing the dense matrix. L4B
preserves the same distinction through `HoloMaterializationMode`:

- `native_holo`: traverse stored coordinates and relation basis directly.
- `hybrid`: combine native relations with an explicitly materialized stage.
- `materialized_fallback`: construct or consume dense state as a declared
  projection fallback. It is never reported as native geometry.

The current fold-orbit runtime is `native_holo`. It does not yet implement a
dense fallback operator; the enum and serialized declaration establish the
boundary for later operators.

### CAT_CAS non-collapse doctrine

`OrbitState`, `FoldPair`, `PathStep`, carrier relations, delayed measurement,
and conservative claim levels remain intact. The `.holo` witness adds geometric
memory around that evolution. It does not select a branch or claim orientation
recovery. The public invariant remains the fold orbit `{d, N-d}`.

## JSON witness

Schema family `CAT_CAS_HOLO_GEOMETRY`, version `1.3.0`, emits:

```text
schema identity and CATALYSIS_IS_THE_HOLOGRAM hypothesis
unresolved fold_pair
holo_geometry (basis, coordinates, neutral reference, status)
physical_mapping (sealed contract reference, digest, support counts, review state)
carrier (coordinates and phase relation)
evolution (operator, steps, path history, closure)
projection (operator, materialization mode, allowed boundary)
invariant_family (typed records, operators, results, tolerances, evidence)
restoration (references, restored status, evidence level)
collapse_boundary
forbidden_fields_scan
```

The reader requires each structural section, restores geometry and path data,
and recomputes the invariant family. Serialized pass flags are not trusted.
The writer rejects invalid structure and scans serialized field names before
accepting the witness.

## L4B.2 reversible path history

`HoloEvolution` owns one heap-allocated `HoloPathHistory`. `holo_object_init()`
allocates it, `holo_replace_path_history()` transfers replacement ownership,
and `holo_object_destroy()` releases it. Standalone histories expose explicit
initialize, reset, append, seal, validate, reverse, serialize, deserialize, and
destroy operations. Capacity grows geometrically with checked bounds; failed
appends leave both history and `OrbitState` unchanged.

Each `HoloPathStep` is a compositional transform containing its index, operator
identity and parameter, exact pre/post accumulator bit patterns, continuity
digests, and a step digest. Digests use deterministic FNV-1a for structural
integrity only; they are not cryptographic. Adjacent post/pre digests must match,
indices are monotonic, and sealed histories reject mutation.

Reversibility means that a terminal software `OrbitState` plus the serialized
history reconstructs the initial numeric state bitwise. Reverse traversal uses
the recorded pre-state bit patterns rather than floating-point subtraction.
The witness is accepted only after the original in-memory history is destroyed,
the path is deserialized from `.holo`, and a dedicated verification copy is
restored. The evolved terminal object remains available for boundary projection.

Successful execution sets `restored=true`,
`evidence_level=software_path_roundtrip`, and
`closure_law=inverse_path_reconstructs_initial_orbit_state`. This proves only a
software path round trip. It is not evidence of physical or hardware restoration.

## L4B.4 non-collapse invariant family

`noncollapse_geometry_v1` predeclares, in deterministic order, orbit
conservation, fold-relation basis involution, forward/reverse composition,
software restoration closure, branch-exchange covariance, serialization
invariance, path-order sensitivity, and software path holonomy. Each record has
an invariant identity, operator identity, declaration phase, typed result,
explicit tolerance, evidence level, and L1 claim level. Registration, operator
changes, tolerance changes, and result changes are rejected after extraction or
sealing.

Structural validation may occur before the boundary. Public family extraction
and sealing occur only at `CollapseBoundary`. The family references one path
digest and does not duplicate history. After serialization, the reader reloads
the geometry and path, recomputes executable records, compares them with the
serialized family, and rejects mismatches or family-digest tampering.

The exchange covariance record applies `(lower, mirror) -> (mirror, lower)` and
requires orbit sum/product preservation plus indexed coordinate exchange. It
does not assign truth or preference to either coordinate. Corruption coverage:

```text
fold coordinate mutation       orbit reconstruction / conservation
relation basis mutation        relation-basis invariant
neutral reference mutation     relation-basis invariant
path step swap                 path order / continuity
operator parameter mutation    step digest
evolution operator mutation   composition invariant
terminal digest mutation       path continuity
serialized result mutation     family digest / reload recomputation
post-boundary invariant add    lifecycle guard
```

`software_path_holonomy` is `DEFERRED_NOT_WELL_DEFINED`: path steps currently
store operator parameters and exact accumulator state bits, but not a declared
group-valued carrier transform. No phase product, wrapping convention, or
winding result is fabricated. All results remain L1/L2 software architecture;
they do not prove orientation recovery, physical closure, or physical holonomy.

## L4B.5A physical mapping contract

`l4b5a_pdn_mapping_v1` is a machine-readable evidence boundary, not a physical
`.holo` implementation. Records use `MEASURED`, `RECOMPUTED_FROM_MEASURED`,
`SIMULATED`, `SOFTWARE_ONLY`, `INFERRED`, `PROPOSED`, or `ABSENT` evidence;
`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, or `NOT_APPLICABLE` status;
and `OBSERVABLE`, `PARTIALLY_OBSERVABLE`,
`UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS`, or `UNDEFINED` observability.

The audited mappings are:

```text
SOFTWARE OBJECT          PHYSICAL CORRESPONDENT                         EVIDENCE                    STATUS
HoloGeometry             candidate PDN response manifold               PROPOSED                    UNSUPPORTED
HoloCarrier              sender/PDN/ring-osc lock-in channel            MEASURED                    SUPPORTED
RelationBasis            identified PDN transfer operator              PROPOSED                    UNSUPPORTED
HoloEvolution            schedule plus deadline-aligned captures       MEASURED                    PARTIALLY_SUPPORTED
HoloPathHistory          ordered reversible physical states            ABSENT                      UNSUPPORTED
CatalyticRestoration     return of declared observable physical state   ABSENT                      UNSUPPORTED
HoloCollapseBoundary     capture window and lock-in I/Q projection      MEASURED                    PARTIALLY_SUPPORTED
HoloInvariantFamily      calibrated physical invariant candidates       RECOMPUTED_FROM_MEASURED     PARTIALLY_SUPPORTED
```

The carrier result is supported only at channel level. The T300 route `4:5`
carried sender-owned mode and phase through a shared PDN rail into measured I/Q
lock-in output. This is not evidence that the channel stores `HoloGeometry`.
The T300 `hash_restored` field proves software XOR/byte-hash bookkeeping, while
P-state restoration is protocol cleanup; neither is physical catalytic
restoration. Compact JSON scores are tracked, but raw matrix CSV captures were
not imported, limiting independent trajectory reconstruction.

Invariant portability is: `serialization_invariance=SOFTWARE_ONLY`;
`relation_basis`, `exchange_covariance`, and `path_order` are
`PHYSICALLY_TESTABLE_AFTER_CALIBRATION`; orbit conservation, path composition,
restoration closure, and software path holonomy have
`NO_CURRENT_PHYSICAL_MAPPING`. No L4B.4 invariant is presently promoted as a
measured physical invariant.

The proposed state is `X_phys(t)={lock-in I,Q; ring-oscillator period; sender
schedule; core identities; TSC origin; temperature proxy; voltage/frequency
state; capture window}`. Current instruments observe only I/Q, ring-oscillator
samples, schedule, cores, TSC origin, and capture window. Rail waveforms,
internal PDN modes, the full thermal field, and complete microarchitectural
state remain unobserved. Restoration is therefore unobservable with current
preserved evidence; a scalar baseline return cannot establish full-state return.

L4B.5B requires a predeclared P0-P8 experiment: declare state and observable,
measure baseline, apply a controlled path, measure terminal state, apply a
declared inverse/closure, measure restored state, compare with uncertainty, run
controls, and repeat across seeds, sessions, and core pairs. Required controls
are no-disturbance, disturbance without restoration, wrong inverse, reordered
inverse, carrier off, randomized phase, session/core-pair repeats, and a
thermal/time-matched sham. Until that gate passes, the decision is
`NOT_AUTHORIZED_EVIDENCE_MISSING`.

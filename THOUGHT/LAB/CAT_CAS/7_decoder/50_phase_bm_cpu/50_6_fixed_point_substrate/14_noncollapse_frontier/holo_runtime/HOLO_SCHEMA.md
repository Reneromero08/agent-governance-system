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

Schema family `CAT_CAS_HOLO_GEOMETRY`, version `1.2.0`, emits:

```text
schema identity and CATALYSIS_IS_THE_HOLOGRAM hypothesis
unresolved fold_pair
holo_geometry (basis, coordinates, neutral reference, status)
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

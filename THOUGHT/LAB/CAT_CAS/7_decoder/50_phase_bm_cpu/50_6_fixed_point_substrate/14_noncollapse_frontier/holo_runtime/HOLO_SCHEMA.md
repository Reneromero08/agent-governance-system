# L4B `.holo` Geometric Memory Schema

## Status and claim ceiling

This document defines the L4B.1 executable geometric-memory architecture.
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
5. `HoloInvariant` predeclares the family and rejects extraction until the
   `CollapseBoundary` has been crossed.
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

Schema family `CAT_CAS_HOLO_GEOMETRY`, version `1.0.0`, emits:

```text
schema identity and CATALYSIS_IS_THE_HOLOGRAM hypothesis
unresolved fold_pair
holo_geometry (basis, coordinates, neutral reference, status)
carrier (coordinates and phase relation)
evolution (operator, steps, path history, closure)
projection (operator, materialization mode, allowed boundary)
invariant (predeclaration, extraction state, result, claim level)
restoration (references, restored status, evidence level)
collapse_boundary
forbidden_fields_scan
```

The reader requires each structural section and reconstructs the native orbit
geometry from the serialized fold pair. The writer rejects invalid structure
and scans the serialized field names before accepting the witness.

## Current limits

The software runtime demonstrates executable relational geometry, delayed
projection, and schema round-tripping. `architectural_metadata_only` is the
restoration evidence ceiling. No physical catalytic closure, physical memory,
orientation channel, or invariant beyond fold closure is experimentally proven.

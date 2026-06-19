# L4B `.holo` Executable Geometric-Memory Schema

**Schema family:** `CAT_CAS_HOLO_GEOMETRY`  
**Serialized version:** `1.4.0`  
**Integrity profile:** `HOLO_RUNTIME_INTEGRITY_V1`  
**Claim ceiling:** L1/L2 software architecture

---

## Core statement

**CATALYSIS IS THE HOLOGRAM.**

The hologram is the catalytic relation itself:

```text
phase is a carrier coordinate
geometry is memory
an algorithm is a local traversal
an observable is a boundary projection
a surviving invariant is memory of closure
```

`.holo` is not a scalar answer file and not a candidate transcript. It stores an unresolved relational object capable of producing declared boundary projections.

---

## Canonical executable object

`HoloObject` separates these roles:

| Role | Object | Meaning |
|---|---|---|
| Relational memory | `HoloGeometry` | Fold-orbit coordinates, neutral reference, and relation basis |
| Carrier | `HoloCarrier` | Complex phase/channel coordinates and substrate status |
| Evolution | `HoloEvolution` | Operator identity, continuation state, and owned path history |
| Boundary projection | `HoloProjection` | Explicit projection operator and materialization mode |
| Invariant memory | `HoloInvariantFamily` | Predeclared typed relations evaluated only at the boundary |
| Physical evidence boundary | `HoloPhysicalMappingReference` | Digest-bound mapping contract and support classifications |
| Future experiment contract | `HoloObservabilityDesignReference` | Sealed L4B.5B0 design reference; no implementation authorization |
| Restoration record | `CatalyticRestoration` | Scope and evidence for software restoration only |
| Projection event | `HoloCollapseBoundary` | The single declared materialization event |

The runtime flow is:

```text
relational geometry
→ carrier-mediated evolution
→ preserved ordered path
→ catalytic closure declaration
→ explicit boundary projection
→ invariant-family extraction
```

---

## Native geometry

`holo_geometry_render()` applies:

```text
neutral + basis × (coordinates - neutral)
```

For the fold-exchange basis, `[lower,mirror]` renders as `[mirror,lower]` while both remain coordinates of one unresolved orbit.

Materialization modes are explicit:

```text
native_holo
hybrid
materialized_fallback
```

The current fold runtime is `native_holo`.

---

## Path memory

`HoloEvolution` owns an appendable `HoloPathHistory`. Each step stores:

- monotonic index;
- declared operator and parameter;
- exact pre/post accumulator bit patterns;
- pre/post OrbitState digests;
- structural step digest.

The path can reconstruct the initial numeric state from a terminal state and serialized history. This is **history-backed exact restoration**, not a claim that an independently specified inverse operator was physically executed.

Accepted restoration metadata:

```text
restored = true
evidence_level = software_path_roundtrip
verification_scope = dedicated_verification_copy
closure_law = inverse_path_reconstructs_initial_orbit_state
```

The closure-law string is historical vocabulary. Its precise semantic scope is recorded in `HOLO_RUNTIME_INTEGRITY.md`.

---

## Semantic integrity

New code must use:

```text
holo_path_history_validate_semantic()
holo_object_validate_semantic()
holo_cross_boundary_atomic()
holo_read_json_strict()
```

These add operator-consistency checks, transactional boundary behavior, and serialized lifecycle comparison around the original structural parser.

Structural FNV digests are deterministic integrity identifiers, not cryptographic signatures.

---

## Invariant family

`noncollapse_geometry_v1` contains eight historical invariant IDs:

```text
orbit_conservation
relation_basis
path_composition
restoration_closure
exchange_covariance
serialization_invariance
path_order
software_path_holonomy
```

Their current scopes are:

- orbit conservation: sum/product and unresolved coordinate relation;
- relation basis: declared fold-exchange basis and neutral reference;
- path composition: recorded path plus history-backed reverse reconstructs initial numeric state;
- restoration closure: software verification-copy digest returns to initial digest;
- exchange covariance: static coordinate exchange preserves declared orbit relations;
- serialization invariance: reload and recomputation preserve the software family;
- path order: authenticated ordered-journal continuity;
- software holonomy: `DEFERRED_NOT_WELL_DEFINED`.

No invariant currently proves physical geometry, physical inverse dynamics, orientation, or physical holonomy.

---

## Physical mapping

`l4b5a_pdn_mapping_v1` is a machine-readable evidence boundary.

Current classifications remain:

| Software object | Physical correspondent | Evidence/status |
|---|---|---|
| `HoloGeometry` | candidate PDN response manifold | proposed / unsupported |
| `HoloCarrier` | sender→PDN→ring-osc lock-in channel | measured / supported at channel level |
| relation basis | identified PDN transfer operator | proposed / unsupported |
| `HoloEvolution` | schedule plus deadline-aligned capture | measured / partially supported |
| physical path history | ordered reversible physical states | absent / unsupported |
| physical restoration | return of declared observable physical state | absent / unsupported |
| boundary | capture window plus I/Q projection | measured / partially supported |
| physical invariant candidates | recomputed channel statistics | partially supported |

T300 supports sender-owned mode/phase transport through a selected PDN route. It does not establish physical geometric memory.

The L4B.5A human review remains bound to the conservative mapping content digest. This repair does not promote any mapping.

---

## L4B.5B0 observability design

The sealed design separates:

```text
controlled input u(t)
latent state x(t)
measured output y(t)
```

It tests minimal, contextual, and delay-embedded measured-state candidates and admits operator complexity only after simpler held-out failures.

Before human review, `holo_observability_design_validate_references()` must close all input, state, gate, falsification, and artifact references.

Human review is stored outside the scientific design by `HoloObservabilityReviewEnvelope`, binding the serialized design artifact with full SHA-256. Review does not authorize calibration acquisition.

---

## Canonical versus legacy

`holo_record.h/.c` is the first L4A carrier scaffold and is not canonical. See `LEGACY_L4A_RECORD.md`.

The canonical implementation is the `HoloObject` stack described here.

---

## Forbidden claims and fields

A `.holo` architecture or artifact must not promote:

```text
winner
candidate_score
hidden_d
recovered_d
orientation_label
verify_pass
AUC as primary truth
physical restoration from byte-hash return
physical geometry from channel detection
```

---

## Current gate

```text
L4B.5B0 = design/review preparation
implementation_authorized = false
executed = false
L4B.5B1 = blocked until explicit post-review authorization
physical restoration = blocked until operator identification
```

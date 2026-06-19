# `.holo` Runtime Semantic Integrity Profile

**Profile:** `HOLO_RUNTIME_INTEGRITY_V1`  
**Schema family:** `CAT_CAS_HOLO_GEOMETRY`  
**Serialized schema:** `1.4.0`  
**Claim ceiling:** L1/L2 software architecture

---

## Why this profile exists

Structural checksums prove that stored records have not changed accidentally. They do not prove that a self-consistent record is a valid execution of the declared operator.

Likewise, deserializing an artifact must not manufacture a successful boundary state that the artifact did not actually serialize.

This profile adds four requirements around the existing `.holo` object:

1. semantic path validation;
2. atomic boundary transition;
3. strict lifecycle deserialization;
4. external digest-bound human review.

---

## Semantic path validation

For every `HoloPathStep`, strict validation checks:

```text
step index equals current state step
operator id is the declared operator
1 <= operator_parameter <= N
stored pre-state bits equal the current accumulator state
stored pre-state digest equals the current OrbitState digest
post accumulator is numerically consistent with the declared phase-walk operator
stored post-state digest equals the reconstructed post OrbitState digest
adjacent state continuity holds
```

The structural FNV digest remains an integrity identifier, not a cryptographic signature. A path can be semantically valid without proving external provenance. Provenance belongs to the containing artifact digest and review/evidence chain.

---

## Restoration semantics

Current software restoration is:

```text
history-backed exact reconstruction
```

The path stores exact pre-state accumulator bit patterns. Reverse traversal reconstructs the initial numeric `OrbitState` from that preserved history.

It is not yet:

```text
group-valued inverse operator dynamics
physical reversible evolution
physical catalytic closure
```

Therefore the accepted evidence label remains:

```text
software_path_roundtrip
verification_scope = dedicated_verification_copy
```

The historical invariant IDs `path_composition`, `restoration_closure`, `exchange_covariance`, and `path_order` are interpreted narrowly:

- `path_composition`: recorded forward path plus history-backed reverse reconstructs the initial numeric state;
- `restoration_closure`: the software verification copy returns to the initial digest;
- `exchange_covariance`: static fold-coordinate exchange preserves declared orbit relations;
- `path_order`: authenticated ordered-journal continuity, not proof of noncommuting physical operators.

Software holonomy remains `DEFERRED_NOT_WELL_DEFINED` until each path step carries a declared group-valued transform.

---

## Atomic CollapseBoundary

A boundary transition is one transaction:

```text
prevalidate
→ seal path
→ mark projection
→ evaluate and seal invariant family
→ commit lifecycle state
```

If any stage fails, the object must return to its prior open state. A failed extraction may not leave a sealed path or crossed boundary behind.

Use `holo_cross_boundary_atomic()` for new code.

---

## Strict reader

The legacy reader reconstructs an object and recomputes invariant values, but historically normalized several lifecycle fields into the expected successful state.

`holo_read_json_strict()` first reads the serialized lifecycle declarations, then invokes the existing parser, compares the reconstructed object against those declarations, and runs semantic object validation.

New verification and import paths must use the strict reader.

---

## Observability design reference closure

Before L4B.5B0 can be reviewed:

- every operator must reference a declared state model;
- every calibration stage must reference a declared input family;
- every null-control reference must resolve to a declared input or an explicitly registered transform control;
- gates `G1-G10` and falsifications `F1-F10` must each be complete and unique;
- future artifact IDs and schema IDs must be unique.

The fixed transform-control registry is:

```text
shuffled_input
phase_randomized
reordered_schedule
```

These are schedule transforms, not hidden input families.

---

## Human review envelope

The scientific design remains frozen in status `READY_FOR_HUMAN_REVIEW` with `human_reviewed=false`. Human review is stored in a separate envelope that binds:

```text
design_id
design_version
64-bit internal design digest
full SHA-256 of the serialized design artifact
reviewer role
review scope
review status
claim ceiling
```

Review metadata therefore cannot change the digest of the scientific content it reviews.

The review envelope does not authorize implementation. Calibration acquisition remains a separate gate.

---

## Required regression tests

```text
self-consistent invalid operator parameter rejected semantically
failed boundary transition rolls back completely
valid strict-reader round trip passes
serialized lifecycle mutation is rejected
undefined null-control reference is rejected
duplicate gate ID is rejected
external review survives unchanged design artifact
artifact mutation invalidates review
```

Hardware and sanitizer execution are deferred to the final SSH verification batch.

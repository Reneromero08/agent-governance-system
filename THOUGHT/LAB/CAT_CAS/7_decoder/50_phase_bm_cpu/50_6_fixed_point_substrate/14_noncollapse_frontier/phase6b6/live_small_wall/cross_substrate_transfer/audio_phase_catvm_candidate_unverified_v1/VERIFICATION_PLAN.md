# Verification Plan

## Evidence angles

1. Fresh frozen-source build and exact qualifier.
2. Fail-closed qualifier deletion and corruption controls.
3. Independent clean-room verifier using different control flow and, where
   practical, a different language.
4. Predeclared adversarial mutation generator.
5. Static and active machine-boundary/no-smuggle attack.
6. Independent graph/schedule or baseline analysis.

A source result plus a rerun of its own qualifier counts as one angle.

## Freshness and reproduction

For every candidate:

1. Use only detached commit `c0cee6a9475d35bc64c90ec30567826bcf3c9e9a`.
2. Create a new empty output directory.
3. Record exact command, environment, stdout, stderr, return code, hashes, and
   generated file inventory.
4. Rebuild after removing generated binaries/results.
5. Repeat in a second fresh output directory.
6. Compare deterministic fields and identify permitted nondeterminism.
7. Delete or corrupt a required artifact and require a nonzero qualifier.
8. Demonstrate that tracked result JSON is not trusted as runtime input.

## Predeclared mutation families

- Relation coefficients: zero, one, all coefficients, affine, nonlinear,
  rank-deficient, contradictory, empty, universal.
- Presentation: reordered relations, renamed public nodes, permuted producer
  and consumer IDs, equivalent encodings, altered topology IDs.
- Graph shape: fanout, width, depth, shared consumers, disconnected nodes,
  duplicated edges, missing edges, stale generations, aliased slots.
- Lifecycle: wrong, missing, duplicated, and reordered operations and
  inverses; stale receipts; snapshot reload; restart/regeneration attempts.
- Protocol: embedded NUL, empty, truncated, oversized, malformed, unknown,
  repeated, and replayed packets.
- Boundary: unauthorized intermediate projection, early response,
  restoration failure, client disconnect, stdout/stderr/temp-file/shared
  memory leakage, process-inspection attempts.
- Quotient: homogeneous AND, homogeneous OR, mixed layers, nonperiodic
  patterns, different neighborhoods, depth greater than width, width greater
  than depth, boundary perturbations, wrong horizon, deliberate overmerge,
  deliberate undermerge, and state relabeling.

Seeded generation will use declared fixed seeds recorded in
`MUTATION_CAMPAIGN.json`; mutation families will not be added reactively to
hide failures.

## Restoration and reuse

Record the pre-borrow canonical carrier, run forward composition, project only
the final boundary, execute actual inverse, compare discrete and numeric
state, reuse the same carrier for a different program, alternate programs,
and track error per cycle. Snapshots may audit but cannot restore.

## Machine boundary

Require controller/core link separation, private carrier memory, denied
intermediate projection, zero or bounded ordinary output, no unresolved state
in files/shared memory, response only after successful restoration, active
post-custody confinement, and active same-UID attacks.

## Provisional decision rule

Each candidate ends in exactly one decision class from the user-specified
taxonomy. Reproduction is never promoted to transfer verification, and no
decision changes the physical Family 10h claim ceiling.

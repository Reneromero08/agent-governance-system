# Constraint Relational Trace V1 Report

## Current Result

```text
REFERENCE_FOUNDATION_IMPLEMENTED__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

The package now contains an exact public open-relation compiler for three-clause normal
form, a small materialized correctness boundary, a factorized local-projector
hypothesis, a reversible evaluation dilation, a Z2 holonomy calibration, a
program-derived inverse phase-carrier calibration, and adversarial controls for
idempotence, presentation gauge, and pairwise false closure.

The implementation does not contain the native operation that would close an arbitrary
open relation without expansion. It therefore does not prove `P = NP`.

## What Was Established

### Open relational semantics

`ConstraintHolo` stores:

```text
boundary variables
constant-size local clause relations
allowed local rows
equality junctions
explicit unresolved native-operator status
explicit unresolved restoration status
```

The public record rejects answer-bearing fields. Clause order, literal order, and
duplicate clauses are normalized for semantic controls. Explicit bijective variable
renaming is available as a presentation-gauge test.

### Total reference boundary

For at most twenty variables, the reference backend returns:

```text
valid carrier
satisfiable or unsatisfiable
one conventional witness when satisfiable
witness count
explicit materialized-state and provenance ledger
```

The cap is intentional. This backend is an exact oracle and test instrument, not a
native mechanism.

### Parity holonomy calibration

The Z2 calibration identifies a three-edge constraint cycle where every one-edge and
two-edge subobject is compatible while the complete object has nontrivial cycle charge.
This is the first executable false-closure control for the campaign.

The incidence-only compiler selects a deterministic spanning forest without evaluating
constraint consistency. The borrowed carrier then propagates actual `+1/-1` phase
transport along the forest. Every non-tree edge closes a cycle product on the carrier:

```text
holonomy(u, v) = phase[u] * edge_transport(u, v) * phase[v]
```

A value of `-1` is the surviving global obstruction. The original union-find draft was
removed because it computed the obstruction conventionally and only decorated the
result with an unrelated phase tape. The current obstruction is produced by carrier
transport itself.

The carrier is restored by executing the exact public tree transports in reverse order.
No stored pre-state transcript is used. A wrong-order replay is a hard negative control
and does not restore a transport chain.

### Reversible evaluation dilation

A public assignment register and one output flag support a self-inverse evaluation
oracle. Applying the same public operation twice restores the flag exactly.

This establishes reversible evaluation while preserving provenance. It does not merge
assignments, decide relation nonemptiness, or implement existential closure.

### Conditional complexity theorem

The package proves the conditional implication:

```text
uniform exact polynomial CET decision
+ deterministic polynomial standard-model simulation
-> 3-SAT in P
-> P = NP
```

It also implements the lawful classical boundary self-reduction that renders and
independently verifies one witness using at most `1 + 2n` exact decision calls. The
conditional theorem is established. Its native premise is not.

## Exact Missing Boundary

The compact factorized satisfaction projector has a public description linear in the
formula. Acting on one basis assignment is exact. The unresolved operation is:

```text
EXACT_NONZERO_NORMALIZATION_WITHOUT_PROVENANCE_EXPANSION
```

A future native carrier must distinguish an empty satisfying subspace from a nonempty
one without assignment enumeration, answer-conditioned preparation, exponential
precision, exponential settling, or an explicit provenance register for every path.

## Current Claim Ceiling

```text
CONSTRAINT_RELATIONAL_TRACE_REFERENCE_ONLY__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

## Environment Limitation

The requested DevSpace connector rejected this conversation before opening a worktree,
so local execution and Codex CLI support were unavailable. The branch was constructed
through the GitHub connector. GitHub CI is the executable qualification surface for
this package.

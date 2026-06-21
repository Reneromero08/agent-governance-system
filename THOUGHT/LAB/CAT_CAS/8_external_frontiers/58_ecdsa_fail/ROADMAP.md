# Exp 58 — ECDSA.fail Reversible Circuit Frontier

**Status:** OPEN  
**Adjudication:** Class A exact circuit verification and resource accounting  
**Role:** verifier-driven reversible elliptic-curve circuit optimization

---

# Frontier object

The object is not one serialized circuit and not one scalar score.

It is the equivalence orbit of reversible circuits implementing the same elliptic-curve function under a full resource vector.

Proposed objects:

- `CircuitOrbit`
- `RegisterRoleGeometry`
- `AncillaLifetimeGraph`
- `UncomputationSchedule`
- `ResourceVector`
- `EquivalenceWitness`

Preserve:

- logical registers;
- ancillas;
- gate dependencies;
- compute/uncompute blocks;
- register liveness;
- functional equivalence;
- path history;
- architecture assumptions;
- resource tradeoffs.

---

# External questions

- Can point-addition arithmetic be realized with fewer logical qubits?
- Can Toffoli count or depth be reduced without hidden ancilla costs?
- Can uncomputation be scheduled globally instead of locally appended?
- Can garbage from one subroutine be borrowed by another before final restoration?
- Can register roles be exchanged to reduce peak width?
- Can equivalent formulas be traversed as one relational family before choosing a circuit trace?
- Can architecture-specific physical cost improve even when the public scalar score does not?

---

# Activation gates

## Gate 0 — Source freeze

- [ ] challenge rules archived;
- [ ] circuit language and verifier frozen;
- [ ] baseline circuit/resource claims frozen;
- [ ] score definition frozen;
- [ ] eligible transformations frozen;
- [ ] test-key-only scope declared;
- [ ] specification digest created.

## Gate 1 — Reversible circuit IR

Represent:

- [ ] X/CNOT/Toffoli or official gate basis;
- [ ] logical registers;
- [ ] ancillas;
- [ ] measurements/feed-forward if allowed;
- [ ] dependency graph;
- [ ] parallel layers;
- [ ] compute/uncompute regions;
- [ ] connectivity assumptions;
- [ ] serialization and replay.

## Gate 2 — Exact ECC arithmetic

- [ ] field add/subtract;
- [ ] modular reduction;
- [ ] multiplication;
- [ ] squaring;
- [ ] inversion;
- [ ] point addition;
- [ ] exceptional cases;
- [ ] controlled point addition;
- [ ] scalar multiplication;
- [ ] small-width exhaustive equivalence tests.

## Gate 3 — Resource accounting

Track the full vector:

- [ ] logical qubits;
- [ ] Toffoli count;
- [ ] Toffoli depth;
- [ ] Clifford count;
- [ ] measurement depth;
- [ ] peak ancillas;
- [ ] uncomputation overhead;
- [ ] routing/connectivity cost;
- [ ] architecture-specific physical estimate.

The public scalar score is emitted only at the external boundary.

## Gate 4 — Rewrite engine

Search:

- [ ] peephole cancellation;
- [ ] commute-and-cancel;
- [ ] ancilla reuse;
- [ ] delayed uncomputation;
- [ ] global uncomputation;
- [ ] register sharing;
- [ ] alternate inversion methods;
- [ ] alternate point-addition formulas;
- [ ] operation fusion;
- [ ] role-exchange symmetries.

## Gate 5 — Hard verification

Every candidate passes:

- [ ] functional equivalence;
- [ ] ancilla cleanup;
- [ ] register restoration;
- [ ] legal gate set;
- [ ] exact resource recount;
- [ ] exhaustive small-width tests;
- [ ] randomized large-width tests;
- [ ] independent implementation.

## Gate 6 — Public submission

- [ ] challenge verifier passes;
- [ ] score improvement reproduced;
- [ ] complete resource vector preserved;
- [ ] rewrite lineage serialized;
- [ ] source/disclosure rules satisfied.

---

# Fastest falsifiable prototype

Select one reversible finite-field subroutine and test whether global ancilla-lifetime scheduling reduces peak width or gate count while preserving exact function and cleanup.

---

# No-smuggle and safety model

Forbidden:

- live user keys or production wallet targeting;
- hidden expected outputs controlling rewrites;
- omitted ancillas;
- resource accounting that ignores routing or cleanup;
- random tests promoted to proof of equivalence;
- known optimized circuit embedded as a retrieval answer without provenance;
- score-only retention that discards the equivalence family.

All work is limited to public challenge circuits, test instances, and defensive resource analysis.

---

# First deliverable

`ReversibleECCIR` plus one independently verified arithmetic block and a complete resource/equivalence ledger.

---

# Claim ceiling

Before public acceptance:

> The declared reversible circuit transformation preserves the tested function and cleanup semantics while changing the measured resource vector.

After official acceptance:

> The challenge verifier accepted the submitted circuit and returned the recorded resource score.

Forbidden:

- secp256k1 broken in practice;
- live signatures compromised;
- full Shor execution demonstrated;
- Small Wall crossed.

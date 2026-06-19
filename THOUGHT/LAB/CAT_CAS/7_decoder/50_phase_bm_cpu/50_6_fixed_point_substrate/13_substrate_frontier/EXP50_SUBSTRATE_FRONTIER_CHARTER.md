# Exp 50 Substrate Frontier Charter — Historical Record

**Original date:** 2026-06-14
**Current status:** `SUPERSEDED__DO_NOT_IMPLEMENT`
**Superseded by:** `EXP50_L4_GATE_DESIGN_AUDIT.md`, `EXP50_SUBSTRATE_MECHANISM_DEFINITION.md`, and Phase 6B non-collapse doctrine.

---

## Purpose of this retained file

This filename is preserved so old reports and commit history remain navigable. The original charter proposed a scalar fixed-point experiment built around:

```text
verify(x)
f(x) = x if verify(x) else (x+1) mod N
candidate_0 / candidate_1 execution
offline truth scoring
scalar d recovery
```

That experiment was rejected before a valid L4 implementation because:

1. the map is ordinary sequential enumeration;
2. the public cosine verifier is fold-even, so `verify(d)==verify(N-d)`;
3. the restricted domain changes the task into recovery of the public fold magnitude;
4. tape restoration does not convert forward search into substrate dynamics.

Git history contains the complete original charter. It must not be copied into a new implementation, used as the active acceptance contract, or cited as the current Phase 6 target.

---

## Current authoritative question

Phase 6B no longer asks whether a SHA-wrapped loop can find a scalar candidate. It asks whether a complete unresolved relational state can be represented, evolved, restored, projected, and physically mapped without collapsing to candidate verification.

The active object is:

```text
OrbitState
+ HoloGeometry
+ complex carrier relation
+ ordered path memory
+ explicit CollapseBoundary
+ invariant family
+ evidence-bounded physical mapping
```

The active documents are:

- `../14_noncollapse_frontier/doctrine/EXP50_MEDIAN_BASIN_CORRUPTION_AUDIT.md`
- `../14_noncollapse_frontier/doctrine/EXP50_NON_COLLAPSE_SUBSTRATE_ARCHITECTURE.md`
- `../14_noncollapse_frontier/holo_runtime/HOLO_SCHEMA.md`
- `../14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md`

---

## Preserved historical facts

The original charter did correctly require:

- no hidden secret in the runtime path;
- explicit controls;
- deterministic setup;
- restoration checks;
- conservative claim ceilings;
- rejection of manual phase/sign encoding.

Those governance requirements survive. The scalar candidate architecture does not.

---

## Prohibited reuse

Do not reactivate any of the following as a Phase 6B implementation:

```text
candidate_0 / candidate_1 truth comparison
verify(x) as branch selector
forward or reverse candidate scan
orientation AUC as the primary object
restricted-domain d recovery
SHA-wrapped enumeration
```

Any future reference to this charter must label it `historical`, `rejected`, or `superseded`.

# External Frontiers Architecture

**Status:** OPEN  
**Track:** `8_external_frontiers`  
**Purpose:** provide one reusable architecture for externally adjudicated CAT_CAS research

---

# 1. Architectural thesis

Track 8 exists to answer one question repeatedly:

> Can CAT_CAS transform an externally defined Wall into a new executable relation and leave behind a witness, artifact, proof, or score that an independent boundary accepts?

The architecture must preserve two truths simultaneously:

1. The external problem retains its official definition.
2. CAT_CAS may use a nonstandard internal representation to attack it.

The domain must not be flattened into a familiar median workflow merely because existing tools expect one. The official adjudication boundary is binding; the internal ontology is not.

---

# 2. Universal external-frontier object

Each experiment instantiates:

```text
ExternalFrontierObject {
    frontier_id
    source_manifest
    specification_digest
    adjudication_class
    external_boundary
    domain_object
    unresolved_state
    relation_basis
    transformation_history
    invariant_family
    restoration_record
    evidence_manifest
    witness_or_artifact
    external_verdict
    claim_ceiling
    transfer_record
}
```

## 2.1 `source_manifest`

Records:

- official source URLs;
- retrieved dates;
- rule version;
- challenge software commit;
- constants and test vectors;
- prize and deadline snapshot;
- licensing and disclosure conditions;
- unresolved ambiguities.

## 2.2 `specification_digest`

A cryptographic digest over the frozen specification bundle. Every run, witness, and claim points to this digest.

## 2.3 `adjudication_class`

One of the classes defined in `ADJUDICATION_AND_TRANSFER.md`:

- A — deterministic witness;
- B — hidden benchmark;
- C — reproducible artifact plus specialist review;
- D — expert-reviewed proof;
- E — broad community acceptance.

## 2.4 `external_boundary`

The exact outside acceptance condition. Examples:

- official Poseidon verifier returns success;
- decompressor reproduces `enwik9` byte-for-byte under accounting rules;
- ARC private evaluation accepts outputs;
- a mathematical proof closes every quantified case;
- a Vesuvius reconstruction survives raw-volume and papyrological review.

## 2.5 `domain_object`

The foreign-domain representation before CAT_CAS transformation:

- finite-field permutation;
- code and agreement geometry;
- CT volume and candidate sheets;
- ARC scenes and actions;
- corpus and decompressor;
- theorem statement and proof obligations;
- reversible circuit and resource vector;
- physical architecture and error model.

## 2.6 `unresolved_state`

The complete process-object maintained before collapse. It may be named differently in each domain:

- `PoseidonOrbit`;
- `AgreementOrbit`;
- `SurfaceOrbit`;
- `TaskHolo`;
- `CompressionHolo`;
- `ProofOrbit`;
- `CircuitOrbit`;
- `ArchitectureOrbit`.

It must preserve alternatives, dependencies, exclusions, and path history rather than only the current best scalar candidate.

## 2.7 `relation_basis`

The minimal declared operators that generate meaningful transformations of the object.

Examples:

- Poseidon S-box and MDS action;
- polynomial evaluation and elimination;
- Reed–Solomon affine combination and syndrome maps;
- mesh continuation and topological compatibility;
- ARC object correspondence and transformation composition;
- reversible parser transforms;
- theorem-preserving rewrites;
- reversible gate identities;
- hardware placement and scheduling moves.

## 2.8 `transformation_history`

Owned ordered path history sufficient to:

- reproduce the result;
- reverse or restore borrowed state where declared;
- distinguish path-dependent from state-only outcomes;
- identify where information entered or left the process.

## 2.9 `invariant_family`

Predeclared or discovered invariants tied to the domain object. Invariants may be:

- algebraic;
- geometric;
- topological;
- spectral;
- combinatorial;
- information-theoretic;
- proof-theoretic;
- resource-theoretic.

Post-hoc invariants must be labeled exploratory until independently frozen and retested.

## 2.10 `restoration_record`

Records exactly what returned:

- bytes;
- software state;
- observable state;
- proof state;
- geometric path state;
- circuit ancillas;
- or nothing.

No experiment may use the word restoration without naming the restored equivalence class.

## 2.11 `evidence_manifest`

Contains:

- raw artifacts;
- derived artifacts;
- hashes;
- commands;
- environment;
- seeds;
- toolchain versions;
- rejected trials;
- negative controls;
- replay instructions.

## 2.12 `claim_ceiling`

The strongest statement licensed by the current evidence.

The claim ceiling must distinguish:

- solved external instance;
- architecture validated in one domain;
- transferable mechanism indicated;
- second-domain transfer reproduced;
- physical Small Wall evidence;
- Big Wall generalization.

---

# 3. Six-layer execution stack

## Layer 1 — Source Freeze

The official problem is frozen before implementation.

**Exit gate:** a complete source manifest and digest exist.

## Layer 2 — Domain Adapter

The official object is represented exactly and independently tested.

**Exit gate:** official examples and an independent implementation agree.

## Layer 3 — Relational Process Object

The internal non-collapse state, relation basis, path history, and boundary are defined.

**Exit gate:** the object can serialize, reload, validate, and preserve declared semantics.

## Layer 4 — Mechanism Engine

The experiment executes the smallest falsifiable mechanism.

This layer may include:

- exact search;
- elimination;
- reversible traversal;
- theorem discovery;
- active experimentation;
- reconstruction;
- model fitting;
- compression;
- circuit rewriting.

**Exit gate:** the mechanism either produces a valid candidate or maps a specific blocker.

## Layer 5 — External Adjudication

The result is emitted in the official form and tested by the external boundary.

**Exit gate:** accepted, rejected, or blocked with exact evidence.

## Layer 6 — Transfer Analysis

The experiment records what transfers back to CAT_CAS.

**Exit gate:** a transfer record separates domain-specific success from architecture-level evidence.

---

# 4. Boundary discipline

Every experiment declares three boundaries.

## 4.1 Old boundary

The standard representation or interface that currently excludes the target distinction.

## 4.2 Working boundary

The richer internal state exposed by the CAT_CAS process-object.

## 4.3 External boundary

The official acceptance interface.

The working boundary may be richer than the external boundary, but it may not alter the official target.

Example:

```text
Poseidon
old boundary: random input/output search
working boundary: full algebraic constraint fiber
external boundary: exact verifier witness
```

---

# 5. Anti-median mechanism gate

Every nonstandard proposal is processed through:

```text
strongest coherent form
→ required success mechanisms
→ adjacent-domain transpositions
→ true blockers only
→ fastest falsifiable prototype
```

Novelty, difficulty, cost, and unfamiliar syntax are search variables.

However, the following are true blockers until changed:

- missing required information;
- invalid official specification mapping;
- verifier disagreement;
- asymptotic explosion without a mechanism;
- unavailable physical observability for a physical claim;
- inability to distinguish leakage from discovery;
- inability to reproduce the witness;
- proof gap across quantified cases.

---

# 6. No-smuggle and leakage model

Each experiment defines forbidden channels appropriate to its domain.

Examples:

- hidden challenge labels controlling an operator;
- benchmark test contamination;
- known witness embedded in a prompt or fixture;
- using a relaxed verifier as exact success;
- training on private evaluation examples;
- reading target outputs during reconstruction;
- using a theorem database result without provenance;
- optimizing against a held-out set repeatedly until it becomes training data;
- resource counting that omits ancillas, code size, or routing assumptions.

Every experiment must include a leakage audit before a positive claim.

---

# 7. Evidence ladder

```text
E0 — architectural proposal
E1 — exact toy instance
E2 — controlled finite result
E3 — official instance accepted
E4 — reproduced across seeds/instances/sessions
E5 — independent implementation accepted
E6 — second-domain transfer
E7 — physical instantiation under declared observability
```

A prize win may occur at E3 or E4 depending on the competition. A general CAT_CAS transfer claim requires at least E6.

---

# 8. Transfer ladder

```text
T0 — no transfer; domain-specific result only
T1 — reusable software primitive
T2 — reusable process-object or invariant
T3 — same mechanism succeeds on a second external Wall
T4 — mechanism informs and survives an Exp 50 test
T5 — recursively reusable boundary-changing architecture
```

Track 8 does not declare T4 or T5 by inference.

---

# 9. Shared versus domain-owned code

## Shared

Belongs under `8_external_frontiers/shared/` when it is domain-independent:

- source-freeze tooling;
- artifact hashing;
- replay manifests;
- claim ledgers;
- external-boundary wrappers;
- generic process-object interfaces;
- transfer records;
- leakage scans.

## Domain-owned

Stays inside its experiment:

- field arithmetic;
- challenge constants;
- CT readers;
- ARC scene logic;
- corpus transforms;
- theorem-specific generators;
- elliptic-curve circuits;
- neutral-atom resource models.

Do not copy domain code into `shared/` merely because two files look similar.

---

# 10. Commit and evidence discipline

Track 8 uses meaningful architectural chunks:

1. source/specification freeze;
2. independent domain adapter and tests;
3. process-object architecture;
4. first falsifiable mechanism;
5. externally adjudicated evidence;
6. transfer synthesis.

Do not commit every trial, every agent note, or every scalar improvement as a separate pellet.

Raw data and large artifacts follow `docs/STORAGE.md`.

---

# 11. Success definition

Track 8 succeeds when at least one experiment leaves behind all of the following:

- a frozen external problem;
- a non-collapse relational representation;
- an exact or reproducible mechanism;
- an independently adjudicated artifact;
- an honest claim ceiling;
- a transfer record explaining what the result contributes to CAT_CAS.

Prize money is useful but not the definition of success.

The deeper test is whether the architecture can cross more than one externally defined Wall without reverting to conventional scalar search as its universal primitive.

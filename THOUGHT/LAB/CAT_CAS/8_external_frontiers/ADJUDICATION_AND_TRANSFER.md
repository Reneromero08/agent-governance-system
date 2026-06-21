# External Adjudication and Transfer Protocol

**Status:** OPEN  
**Purpose:** separate external truth boundaries from internal CAT_CAS transfer claims

---

# 1. Adjudication classes

## Class A — Deterministic witness

The core result is accepted or rejected by an executable procedure.

Examples:

- cryptanalytic witness;
- exact decompression;
- circuit-equivalence verifier;
- primality certificate.

### Required evidence

- frozen verifier version;
- official result;
- independent verifier;
- complete witness;
- replay instructions;
- accounting rules.

### Remaining social layer

Prize eligibility, disclosure, originality, and publication requirements may still exist even when the object itself is exact.

---

## Class B — Hidden benchmark

The result is evaluated against private tasks, environments, or test cases.

Examples:

- ARC;
- AIMO progress competitions.

### Required evidence

- public rules and environment;
- offline reproducible submission;
- public-set performance;
- contamination audit;
- official private score;
- compute and model disclosure required by the rules.

### Core risk

Repeated interaction with a private leaderboard can become a training channel. Track all submissions and avoid post-hoc overfitting.

---

## Class C — Reproducible artifact plus specialist review

The software or reconstruction must work, but domain experts determine whether it captures the intended real object.

Examples:

- Vesuvius geometry and ink;
- physical-resource analysis;
- scientific reconstruction.

### Required evidence

- raw-data provenance;
- reproducible code;
- uncertainty;
- domain-specific controls;
- specialist review;
- hallucination or artifact audit.

### Core risk

A visually persuasive artifact may not correspond to the underlying object.

---

## Class D — Expert-reviewed proof

The output is a mathematical proof, counterexample, or scientific argument requiring expert checking.

Examples:

- Proximity Prize;
- Erdős bounties;
- Beal;
- research-paper reward programs.

### Required evidence

- exact statement and quantifiers;
- complete proof;
- computational evidence clearly separated from proof;
- independent mathematical review;
- formalization where practical;
- publication path where required.

### Core risk

Finite evidence, analogy, or computational success is mistaken for a universal proof.

---

## Class E — Broad community acceptance

The result requires publication, time, and broad global mathematical acceptance.

Example:

- Clay Millennium problems.

### Required evidence

Everything in Class D plus the official waiting and acceptance conditions.

### Core risk

Scientific correctness and prize recognition occur on different timelines.

---

# 2. External verdict states

Every submission records exactly one state:

```text
NOT_SUBMITTED
SUBMISSION_READY
SUBMITTED_PENDING
OFFICIAL_ACCEPT
OFFICIAL_REJECT
OFFICIAL_PARTIAL
RULES_CHANGED
ELIGIBILITY_BLOCKED
REVIEW_DISPUTED
```

Do not translate silence or pending review into acceptance.

---

# 3. Evidence levels

## E0 — architecture only

A coherent proposal exists. No execution evidence.

## E1 — exact toy instance

The mechanism works on a controlled small object.

## E2 — controlled finite evidence

The mechanism survives predeclared controls on nontrivial finite instances.

## E3 — official instance accepted

The official verifier, benchmark, or review boundary accepts at least one target instance.

## E4 — repeated external evidence

The result survives multiple instances, seeds, sessions, or routes as appropriate.

## E5 — independent implementation

A separate implementation reproduces the result.

## E6 — cross-domain transfer

The same structural mechanism succeeds on a materially different external frontier.

## E7 — physical instantiation

The mechanism is physically instantiated under a declared measurement and restoration boundary.

Evidence levels do not automatically imply transfer levels.

---

# 4. Transfer levels

## T0 — domain result only

The external problem was solved or advanced, but no CAT_CAS component was necessary.

## T1 — reusable primitive

A tool, verifier wrapper, process component, or evidence system transfers.

## T2 — reusable relational mechanism

The process-object, invariant, closure structure, or boundary model transfers beyond the original task.

## T3 — second external Wall

The same mechanism succeeds on another materially different externally defined frontier.

## T4 — Exp 50 transfer

The mechanism is instantiated in an authorized Exp 50 physical experiment and survives its no-smuggle, observability, restoration, target-coupling, and replication gates.

## T5 — recursive boundary architecture

The system can identify and transform new boundaries while preserving its own identity and evidence discipline.

T5 is a program threshold, not a label for one impressive result.

---

# 5. Required transfer record

Every completed experiment answers:

1. What external Wall was addressed?
2. What was the old boundary?
3. What richer working boundary was introduced?
4. What complete object was preserved?
5. What relation basis generated evolution?
6. What invariant, witness, or artifact survived?
7. What was restored or closed?
8. Which controls ruled out leakage or triviality?
9. Which parts were domain-specific?
10. Which parts can transfer unchanged?
11. What is the evidence level?
12. What is the transfer level?
13. What result would be required for the next level?
14. What claims remain forbidden?

---

# 6. Relationship to the Small Wall

External experiments may run before the Exp 50 Small Wall crossing.

They can help by developing:

- target-to-state coupling;
- exact closure;
- whole-object invariants;
- delayed collapse;
- path history;
- restoration semantics;
- boundary selection;
- theorem extraction;
- independent adjudication.

They do not eliminate the physical Exp 50 gates.

```text
external success
≠ physical carrier
≠ physical restoration
≠ Exp 50 target coupling
≠ fold-odd invariant
≠ Small Wall crossing
```

The correct relationship is:

```text
external frontier
→ mechanism discovery
→ cross-domain transfer
→ candidate Exp 50 mechanism
→ authorized physical test
```

---

# 7. Relationship to the Big Wall

A single external solution is not the Big Wall.

The beginning of Big Wall evidence requires:

- at least two different Walls;
- one shared boundary-changing mechanism;
- independent adjudication in both domains;
- explicit preservation/restoration semantics;
- no hidden target channel;
- a transferable process representation.

Physical substrate evidence strengthens but does not replace cross-domain transfer.

---

# 8. Prize recognition versus truth

Maintain two separate ledgers.

## Scientific ledger

- is the result true?
- is the witness exact?
- is the proof complete?
- is the artifact reproducible?
- what is the claim ceiling?

## Administrative ledger

- is the prize open?
- is the entrant eligible?
- were deadlines met?
- were disclosure terms met?
- was publication required?
- did the organizer accept the submission?

A scientifically valid result may fail an administrative prize boundary. An administratively accepted score may still have limited scientific transfer value.

---

# 9. Final rule

External adjudication should be used as a hard boundary, not as a medianizing ontology.

The outside world decides whether the artifact satisfies its declared target.

It does not decide in advance which internal representation CAT_CAS is allowed to use to reach it.

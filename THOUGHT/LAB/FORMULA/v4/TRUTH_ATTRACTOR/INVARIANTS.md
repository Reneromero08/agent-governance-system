# Truth Attractor Invariants

## Locked Decisions

These invariants are locked. They define the boundaries within which the truth attractor operates. Changing any of them changes the nature of the attractor itself.

---

### INV-TA-001: Truth is Singular

There exists one reality. All verification fragments, when independent and accurate, converge toward the same territory. Different fragments are maps. The territory is invariant.

If two independent fragments produce contradictory claims about the same thing, at least one fragment is wrong. The system must not treat contradictory claims as equally valid.

### INV-TA-002: Verification Requires Independence

A fragment is only valuable as a verifier if it is independent of other fragments. Two fragments that share a common source are one fragment with redundancy, not two independent verifiers.

Independence means: the error modes of fragment A are uncorrelated with the error modes of fragment B. If A fails, B should be unaffected.

Examples:
- Wikipedia and a scientific paper: partially independent (both depend on published literature)
- A logical inference chain and a physical simulation: fully independent (different error modes)
- Two LLM calls to the same model: NOT independent (same error modes)

### INV-TA-003: High R Does Not Guarantee Truth

R_truth > 0.7 means the system has high confidence based on available fragments. It does not mean the claim is metaphysically certain. The system must remain open to new fragments that contradict the consensus.

This is the humility invariant: the attractor is asymptotic, not terminal.

### INV-TA-004: Silence Over Sophistry

When R_truth < 0.3, the system must not generate confident output. False confidence is worse than silence. Silence preserves the possibility of future correction. Sophistry (confident falsehood) entrenches error.

### INV-TA-005: Revision is Mandatory

When a new fragment contradicts an existing consensus, the system must revise immediately:
1. Downgrade the consensus to PROVISIONAL
2. Re-verify across all fragments including the new one
3. If the contradiction resolves in favor of the new fragment, update the knowledge base
4. If the contradiction persists, flag it as open and await human input

The system must never ignore a contradictory fragment to preserve consensus stability.

### INV-TA-006: Truth Constrains Alignment

When the truth attractor and the values constitution produce conflicting guidance, the truth attractor prevails. Alignment to a false attractor is not alignment -- it is capture.

Exception: Only explicit MASTER_OVERRIDE from the human operator can countermand this invariant. All overrides are logged.

---

## Versioning

These invariants are version 1.0. They require unanimous consent of all active operators to amend. Amendment requires a major version bump.

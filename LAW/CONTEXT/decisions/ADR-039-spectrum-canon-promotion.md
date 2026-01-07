# ADR-039: SPECTRUM Canon Promotion

**Status**: Accepted
**Date**: 2026-01-07
**Deciders**: System architects
**Context**: Phase 1.7.1 Catalytic Hardening

## Context

The SPECTRUM specifications (02-06) define the cryptographic spine of the Agent Governance System's catalytic computing model. These specs were developed as part of the CAT-DPT (Catalytic Department) project and define:

- Resume bundle semantics (SPECTRUM-02)
- Chain verification and temporal integrity (SPECTRUM-03)
- Ed25519 identity and signing (SPECTRUM-04)
- 10-phase verification procedure with 25 error codes (SPECTRUM-05)
- Restore runner semantics with fail-closed rollback (SPECTRUM-06)

During the CAT-DPT merge into AGS main, the SPECTRUM spec files were lost. The specifications were preserved only as:
1. Inline changelog entries in the archived `CATALYTIC_CHANGELOG.md`
2. Implementation comments in `verify_bundle.py`, `restore_runner.py`, etc.
3. A complete snapshot in the LLM Packer archive

Without canonical documentation:
- New agents must reverse-engineer cryptographic protocols from code
- The binding between implementation and specification is implicit
- Verification of "correctness" has no authoritative reference

## Decision

Promote SPECTRUM-02 through SPECTRUM-06 from the LLM Packer archive to `LAW/CANON/`:

| Spec | Filename | Purpose |
|------|----------|---------|
| SPECTRUM-02 | `SPECTRUM-02_RESUME_BUNDLE.md` | Adversarial resume without execution history |
| SPECTRUM-03 | `SPECTRUM-03_CHAIN_VERIFICATION.md` | Chained temporal integrity |
| SPECTRUM-04 | `SPECTRUM-04_IDENTITY_SIGNING.md` | Validator identity and Ed25519 signing |
| SPECTRUM-05 | `SPECTRUM-05_VERIFICATION_LAW.md` | 10-phase verification procedure |
| SPECTRUM-06 | `SPECTRUM-06_RESTORE_RUNNER.md` | Restore semantics with atomicity |

All specs retain their original frozen version numbers and dates. A "Promoted to Canon" date is added.

## Why Not Just Reference Implementation?

The implementation in `verify_bundle.py` is ~850 lines of Python. Reading it to understand the verification procedure requires:
- Understanding Python
- Tracing control flow through 10 phases
- Inferring the threat model from rejection conditions

SPECTRUM-05 documents the same 10 phases in ~500 lines of specification-grade prose with explicit threat model sections. The specification is:
- Language-agnostic
- Auditable by non-programmers
- Testable against any implementation

## Alternatives Considered

### Alternative 1: Keep Specs in Archive

Leave SPECTRUM specs in `MEMORY/ARCHIVE/catalytic-department-merged/`.

**Rejected because**:
- Archive is for historical snapshots, not living authority
- Agents don't routinely search archives for specifications
- Creates confusion about what is canonical

### Alternative 2: Merge into CMP-01

Fold SPECTRUM content into `CMP-01_CATALYTIC_MUTATION_PROTOCOL.md`.

**Rejected because**:
- SPECTRUM specs are substantial (SPECTRUM-06 alone is ~600 lines)
- Different audiences: CMP-01 is operational protocol; SPECTRUM is cryptographic law
- Would create a 3000+ line document
- SPECTRUM has its own versioning and freeze semantics

### Alternative 3: Reference-Only (No Canon Files)

Add references to SPECTRUM in CMP-01 pointing to implementation files.

**Rejected because**:
- Implementation can drift from specification
- No authoritative reference for audits
- Violates canon principle: canonical docs exist for canonical concepts

## Consequences

### Positive

- **Single source of truth**: Cryptographic protocols have canonical references
- **Auditable**: External reviewers can read specifications without code
- **LLM Packer vindicated**: Archive preserved specs through the merge
- **Implementation validation**: Code can be verified against spec

### Negative

- **Maintenance burden**: 5 new canon files to maintain
- **Version sync**: Must update specs if implementation changes (but this is intentional—changes should be deliberate)

### Neutral

- Specs are already frozen; promoting them doesn't change their content
- Implementation already references SPECTRUM-0X in docstrings

## Implementation

1. Extract original SPECTRUM specs from LLM Packer archive:
   `MEMORY/LLM_PACKER/_packs/_archive/catalytic-dpt-pack-2025-12-27_13-21-43/repo/SPECTRUM/`

2. Write to `LAW/CANON/` with standardized filenames

3. Add "Promoted to Canon: 2026-01-07" to each spec header

4. Update CMP-01 and CATALYTIC_COMPUTING.md to reference SPECTRUM specs

## References

- Phase 1.7.1 in `AGS_ROADMAP_MASTER.md`
- ADR-038: CMP-01 Catalytic Mutation Protocol
- Original SPECTRUM source: `MEMORY/LLM_PACKER/_packs/_archive/catalytic-dpt-pack-2025-12-27_13-21-43/repo/SPECTRUM/`

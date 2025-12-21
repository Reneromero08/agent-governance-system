# STYLE-004: ADR-First Design Gate

**Category:** Governance
**Status:** Active
**Scope:** Repository-wide
**Enforcement:** Strict

---

To prevent "governance amnesia" and ensure compliance with [INV-007](file:///d:/CCC%202.0/AI/agent-governance-system/CANON/INVARIANTS.md), architectural decisions must be documented BEFORE implementation, not as an afterthought.

## Principles

1.  **Mutation Check**: Before modifying any file in `CANON/` or changing the definition of an `Invariant`, the agent MUST ask: "Does this require a new ADR or an update to an existing one?"
2.  **Breadcrumb for Humans**: The ADR provides the "Why" behind the "What." Working without one is a violation of the engineering integrity preference ([STYLE-002](file:///d:/CCC%202.0/AI/agent-governance-system/CONTEXT/preferences/STYLE-002-engineering-integrity.md)).
3.  **Ceremony Audit**: The "Chunked Commit Ceremony" ([STYLE-001](file:///d:/CCC%202.0/AI/agent-governance-system/CONTEXT/preferences/STYLE-001-commit-ceremony.md)) MUST now include this explicit verification: **"Is there a corresponding ADR for these structural changes?"**

## Enforcement
If an agent discovers it is changing a system invariant (e.g., directory structures, output roots, or core grammar) without an ADR, it must immediately stop and draft the record before continuing the refactor.

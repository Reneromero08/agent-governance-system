# STYLE-003: Mandatory Changelog Synchronisation

**Category:** Governance
**Status:** Active
**Scope:** Repository-wide
**Enforcement:** Strict

---

To maintain the auditability and integrity of the Agent Governance System (AGS), every commit that impacts system behavior or configuration MUST be documented.

## Principles

1.  **Atomic Governance**: Any change to a `CANON` file is legally incomplete without a corresponding entry in `CANON/CHANGELOG.md`.
2.  **Zero Exceptions**: Even minor refactors (like renaming a directory for technical compliance) are user-visible system changes and must be recorded.
3.  **Ceremony Prerequisite**: The "Chunked Commit Ceremony" ([STYLE-001](file:///d:/CCC%202.0/AI/agent-governance-system/CONTEXT/preferences/STYLE-001-commit-ceremony.md)) MUST include a verification step: "Is the Changelog updated for these changes?"
4.  **Automated Failure**: Committing without a changelog update when canon has changed is a critical failure of the engineering integrity preference ([STYLE-002](file:///d:/CCC%202.0/AI/agent-governance-system/CONTEXT/preferences/STYLE-002-engineering-integrity.md)).

## Enforcement
The agent must proactively check `git diff --name-only` before every ceremony to ensure `CHANGELOG.md` is included if any `CANON/` or `SKILLS/` files are present.

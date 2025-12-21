# STYLE-002: Engineering Integrity

**Category:** Governance
**Status:** Active
**Scope:** Repository-wide
**Enforcement:** Strict

---

To ensure the long-term stability and maintainability of the Agent Governance System (AGS), all engineering work must prioritize "real fixes" over "temporary patches."

## Principles

1.  **Architecture over Expediency**: If a folder structure or naming convention creates technical friction (e.g., Python import issues), rename the folder or refactor the architecture rather than implementing "hacks" or helper scripts that drift from the core design.
2.  **No Ghost Modules**: Avoid creating "patch files" or "core" helpers solely to bypass self-imposed constraints. If logic belongs in a specific component, the environment should be made to support it there.
3.  **Clean Naming**: Prefer Python-standard naming (snake_case) for any directory that requires programmatic access, even if it affects user-facing "branding" folders.
4.  **Transparency**: When faced with a choice between a quick fix and a deep refactor, the agent must present the "real fix" as the primary option.

## Consequence
Any "patch" solution that bypasses technical limitations instead of resolving them is considered a violation of engineering integrity.

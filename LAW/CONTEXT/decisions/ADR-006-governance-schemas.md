# ADR-006: Governance Object Schemas

**Status:** Accepted
**Date:** 2025-12-21
**Confidence:** High
**Impact:** High
**Tags:** [governance, scaling, validation]
**Deciders:** Antigravity (Agent), User

## Context

As the system grows, the number of "Law-Like" files (ADRs, Skills, Style Preferences) is increasing. These files carry critical metadata (Status, Context, Versioning) that drives automated governance logic.

Previously, these files were essentially free-text Markdown with loose conventions. this led to:
1.  **Drift**: Inconsistent metadata keys (e.g., "Reference" vs "Status: Reference").
2.  **Brittle Tooling**: Scripts attempting to parse headers had to handle many edge cases.
3.  **Governance Amnesia**: Files like `SKILL.md` templates were not validated, leading to invalid states in the repo.

## Decision

We will enforce **JSON Schema Validation** for all "Governance Objects."

1.  **Schema Definition**: We define strict JSON Schemas in `MCP/schemas/governance/` for:
    - `adr.schema.json`
    - `skill.schema.json`
    - `style.schema.json`
2.  **Validation Layer**: We implement `TOOLS/schema_validator.py` to extract frontmatter/header data from Markdown and validate it against these schemas.
3.  **Governance Gate**: We integrate this validation into `TOOLS/critic.py`. A commit is rejected if *any* governance object fails schema validation.
4.  **Invariant**: We add **INV-011** to `CANON/INVARIANTS.md` to formalize this requirement.

## Alternatives considered

- **YAML Frontmatter**: We considered moving strictly to YAML frontmatter (Jekyll style).
    - *Rejection*: We preferred keeping the "Document" feel of Markdown headers (`**Status:** Active`) for readability, as long as our parser is robust.
- **Pydantic Models**: defining Python classes for validation.
    - *Rejection*: JSON Schemas are language-agnostic and easier to share with LLMs/tools via MCP.

## Rationale

- **Machine Readability**: Governance files are code. They must be parseable.
- **Scalability**: As we add hundreds of skills or decisions, human review of metadata consistency becomes impossible.
- **Self-Correction**: The error messages from `jsonschema` provide immediate feedback to agents/users to fix their metadata.

## Consequences

- **Strictness**: Contributors cannot invent new metadata fields on the fly; they must update the schema.
- **Breaking Change**: Existing files that do not match the schema (like `Status: Reference` in skills) are now invalid and must be fixed (completed in v2.0.x).
- **Tooling**: All future tools can rely on the presence and type-safety of these fields.

## Enforcement

- `INV-011` explicitly requires schema compliance.
- `TOOLS/critic.py` runs validation on every commit ceremony.

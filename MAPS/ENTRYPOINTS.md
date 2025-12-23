# Entrypoints

This document lists the primary entrypoints where agents and humans are expected to make changes. By centralising these locations, the system reduces thrash and ensures updates occur in the correct places.

## Governance

- `CANON/CONTRACT.md` - Modify to change non-negotiable rules. Requires full ceremony.
- `CANON/INVARIANTS.md` - Modify to add or change invariants (rare).
- `CANON/VERSIONING.md` - Modify when bumping the canon version or documenting deprecations.
- `CANON/CHANGELOG.md` - Add entries for each change to the canon or system behavior.
- `SKILLS/master-override/` - Audit logging and gated access for `MASTER_OVERRIDE` usage.
- Privacy boundary rule: update `CANON/CONTRACT.md`, `AGENTS.md`, and `CONTRACTS/fixtures/governance/privacy-boundary/`.

## Decision records

- `CONTEXT/decisions/ADR-xxx-*.md` - Add a new ADR for each architectural decision.
- `CONTEXT/rejected/REJECT-xxx-*.md` - Document and index rejected proposals.
- `CONTEXT/preferences/STYLE-xxx-*.md` - Record non-binding preferences.
- `CONTEXT/open/OPEN-xxx-*.md` - Track open discussions.
- `CONTEXT/research/INDEX.md` - Index non-binding research threads and references.

## Skills

- `SKILLS/` - Add a directory for each new skill. Include a `SKILL.md` manifest, implementation scripts and fixtures.

## Contracts

- `CONTRACTS/fixtures/` - Add new fixture directories as you formalise behavior. Use descriptive names.
- `CONTRACTS/schemas/` - Add JSON schemas to validate structures (canon, skills, context, cortex, etc.).

## Tools and memory

- `CORTEX/` - Modify when adding new index fields or query capabilities.
- `MEMORY/LLM_PACKER/Engine/packer.py` - Modify when updating the pack format or manifest.
- `MEMORY/LLM_PACKER/` - Windows wrapper scripts for running the packer.
- `TOOLS/` - Add critics, linters and migration scripts here.

## MCP integration

- `MCP/README.md` - Client configuration and quick-start guidance.
- `MCP/MCP_SPEC.md` - Protocol mapping and implementation status.
- `MCP/server.py` - MCP server implementation (stdio).
- `CONTRACTS/_runs/ags_mcp_entrypoint.py` - Recommended entrypoint wrapper (audit logs under allowed roots).
- `SKILLS/mcp-smoke/` - CLI smoke test for MCP server.
- `SKILLS/mcp-extension-verify/` - Extension-agnostic verification checklist + smoke test.

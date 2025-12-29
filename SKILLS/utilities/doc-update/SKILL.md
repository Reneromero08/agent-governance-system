# Skill: doc-update

**Version:** 0.1.0

**Status:** Active

**required_canon_version:** ">=2.5.1 <3.0.0"

## Trigger

Use when asked to update repository documentation, ADRs, or onboarding guidance.

## Doc system overview (authoritative structure)

- **CANON/**: binding rules (CONTRACT, INVARIANTS, VERSIONING). Only edit when explicitly asked.
- **CONTEXT/**: ADRs, preferences, rejections, and research. ADRs explain why decisions exist.
- **CONTEXT/maps/**: entrypoints and system maps that tell agents where to change things.
- **README.md**: top-level overview and required session bootstrap.
- **MCP/**: protocol integration docs and server entrypoints.
- **SKILLS/**: procedural workflows + fixtures for non-trivial work.
- **CONTRACTS/**: fixtures and schemas that enforce behavior.
- **TOOLS/**: critics, linters, and migration helpers.

## Inputs

- `input.json` with:
  - `topic` (string): e.g. "mcp", "skills", "governance".
  - `extra_targets` (array, optional): Additional paths to include in the update set.

## Outputs

- Writes `actual.json` with:
  - `topic` (string)
  - `doc_system` (array of strings)
  - `recommended_targets` (array of strings)
  - `notes` (array of strings)

## Workflow

1. Confirm intent gate (do not edit CANON or existing CONTEXT unless explicitly requested).
2. Use `CORTEX/query.py` to locate entrypoints; avoid raw filesystem discovery.
3. Update the smallest set of docs that inform all agents (README, CONTEXT/maps/ENTRYPOINTS, MCP docs, relevant ADRs).
4. For MCP updates, include:
   - The recommended entrypoint (`CONTRACTS/_runs/ags_mcp_entrypoint.py`).
   - The log location (`CONTRACTS/_runs/mcp_logs/`).
   - The verification skills (`mcp-smoke`, `mcp-extension-verify`).
   - Client config examples for Windows and WSL where relevant.
5. Run `python3 TOOLS/critic.py` and `python3 CONTRACTS/runner.py` after changes.

## Constraints

- Must not scan the filesystem directly (use `CORTEX/query.py` for discovery).
- Must not modify canon or context unless explicitly authorized.
- Must not write artifacts outside allowed roots.

## Fixtures

- `fixtures/basic/`

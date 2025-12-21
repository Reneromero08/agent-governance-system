# Security Policy

This document outlines the security considerations for the Agent Governance System. Because AGS may execute code and manage sensitive state, it is important to minimise risk.

## Principles

- **Least privilege** - Skills and tools should run with the minimum permissions required to perform their task.
- **Deterministic outputs** - Randomness and external network calls should be avoided or isolated to prevent non-reproducible behaviour.
- **No external side effects** - Skills must not perform irreversible actions (e.g. network requests, database writes) without explicit authorisation.
- **Auditability** - All changes should be traceable through fixtures, context records and the changelog.

---

## Trust Boundaries

### Agent Read Access

Agents MAY read:
- `CANON/` — rules and invariants
- `CONTEXT/` — decisions, guides, research
- `MAPS/` — navigation and entrypoints
- `SKILLS/` — skill definitions and fixtures
- `CONTRACTS/` — fixture definitions
- `CORTEX/query.py` — cortex query API
- `TOOLS/` — tooling scripts
- `MEMORY/` — packer and memory docs
- Root files: `AGENTS.md`, `README.md`, `ROADMAP.md`

Agents MUST NOT directly read:
- Filesystem paths outside the repository
- `CORTEX/_generated/` (use query API instead)
- User secrets or credentials

### Agent Write Access

Agents MAY write to:
- `CONTRACTS/_runs/` — fixture execution outputs
- `CORTEX/_generated/` — built indices
- `MEMORY/LLM-PACKER/_packs/` — generated packs
- `BUILD/` — user-owned build outputs

Agents MUST NOT write to:
- `CANON/` — without change ceremony (ADR, fixtures, changelog)
- Root files — without explicit human approval
- Any path outside allowed output roots

### Human Approval Required

The following actions require explicit human approval:
- Modifying any file in `CANON/`
- Adding or removing invariants
- Changing version numbers
- Deleting skills or fixtures
- External network requests
- Installing dependencies

### Sandboxing

When running skills:
1. Skills execute in the repository context only
2. No network access unless explicitly authorized
3. No access to parent directories
4. Time and random values must be injected, not generated

---

## Reporting Vulnerabilities

If you discover a vulnerability in this repository or its processes, please open an issue in the "Security" category or contact the maintainers privately. Do not disclose vulnerabilities publicly until they have been addressed.

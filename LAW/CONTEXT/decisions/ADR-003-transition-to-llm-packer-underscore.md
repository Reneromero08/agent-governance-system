---
id: "ADR-003"
title: "Transition to LLM_PACKER for Python Compatibility"
status: "Accepted"
date: "2025-12-21"
confidence: "High"
impact: "High"
tags: ["governance", "packaging"]
---

<!-- CONTENT_HASH: 967819bd25016010f544d4e374c2113e03e85eb7e7ab1e9e12ab2f345342946c -->

# ADR-003: Transition to LLM_PACKER for Python Compatibility

**Deciders:** Antigravity (Agent), User
## Context
The repository previously used the directory `MEMORY/LLM-PACKER/` for system-generated packs. This naming convention (hyphenated) prevented Python from importing the core engine as a package, causing "ModuleNotFoundError" in GitHub CI environments that require strict import validation.

## Decision
We will rename `MEMORY/LLM-PACKER/` to `MEMORY/LLM_PACKER/` (underscore).

This change facilitates:
1.  Clean, direct Python imports from `MEMORY.LLM_PACKER.Engine`.
2.  Resolution of CI pipeline failures without resorting to path hacks or temporary scripts.
3.  Alignment with PEP 8 naming conventions for packages.

## Consequences
- **Breaking Change**: Any existing external references or absolute paths pointing to `LLM-PACKER` must be updated.
- **Major Version Bump**: Per `INV-001`, a rename of a system output root requires a major version bump. The system version will move from `1.0.0` to `1.1.0`.
- **Artifact Root Update**: `INV-006` and `SECURITY.md` must be updated to reflect the new valid write root.
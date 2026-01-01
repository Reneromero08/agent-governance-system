---
id: "ADR-013"
title: "LLM_PACKER LITE Profile + SPLIT_LITE Outputs"
status: "Accepted"
date: "2025-12-23"
confidence: "High"
impact: "High"
tags: ["packer", "memory", "handoff"]
---

<!-- CONTENT_HASH: 1a73512b1877bf07c7ff3ce45f5aeca4b734dc25be6d90ac77dc90e5bf4775d3 -->

# ADR-013: LLM_PACKER LITE Profile + SPLIT_LITE Outputs

## Context

Full packs are too large for quick discussion and often include redundant
content (fixtures, generated artifacts, and combined outputs). The system
needs a smaller, high-signal pack for fast handoff while preserving FULL
as the record-keeping baseline.

## Decision

Add a LITE pack profile and a derived SPLIT_LITE documentation set:

- **LITE profile**: a compact allowlist that includes canonical rules, maps,
  skill contracts, and core entrypoints, with symbolic indexes for navigation.
- **SPLIT_LITE**: discussion-first docs placed alongside SPLIT in the same pack,
  describing pointers + indexes instead of duplicating large payloads.
- **Token reporting**: per-payload token counts (SPLIT, SPLIT_LITE, combined
  single files) to avoid treating the entire pack as one context window.

FULL remains unchanged and is still the record-keeping output.

## Requirements

1. **Profile selection**
   - Default profile is FULL; LITE is explicit via flag/env.
2. **LITE allowlist**
   - Include AGENTS/README/CANON/MAPS, core entrypoints, and SKILL.md manifests.
   - Exclude fixtures, generated artifacts, research archives, and OS wrappers.
3. **Symbolic indexes**
   - Emit LITE-specific indexes for skills, fixtures (inventory-only), codebook,
     and AST symbols for included code.
4. **SPLIT_LITE**
   - Generated alongside SPLIT in the same pack when requested.
5. **Token reporting**
   - Report token counts per payload rather than summing across all outputs.

## Alternatives considered

- Compressing FULL-only outputs.
- Maintaining separate pack roots for LITE.

## Rationale

LITE reduces token load while preserving navigation and correctness. Keeping
SPLIT_LITE in the same pack preserves context continuity and avoids multiple
distribution artifacts.

## Consequences

- LITE packs are discussion-ready but incomplete for implementation details.
- Users can choose FULL or LITE depending on the task.

## Enforcement

- Update packer implementation and smoke fixtures for LITE + SPLIT_LITE.
- Document LITE behavior in `MEMORY/LLM_PACKER/README.md`.

## Review triggers

- Significant changes to pack size or handoff workflow.
- Changes in pack consumption requirements or client constraints.
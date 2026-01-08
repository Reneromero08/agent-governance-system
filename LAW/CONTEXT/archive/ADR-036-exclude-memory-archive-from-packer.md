---
id: "ADR-036"
title: "Exclude MEMORY/ARCHIVE from LLM packer outputs"
status: "Accepted"
date: "2026-01-06"
confidence: "Medium"
impact: "Low"
tags: ["llm-packer", "memory", "archive"]
---

<!-- CONTENT_HASH: 0de3746578b3a2888087f19c1abffb322e421174e68300e3d2b388651fca6fe5 -->

# ADR-036: Exclude MEMORY/ARCHIVE from LLM packer outputs

## Context

The LLM packer includes the `MEMORY/` tree for AGS packs. This currently pulls in `MEMORY/ARCHIVE/`, which contains historical material that should not be part of routine handoff packs.

## Decision

Exclude `MEMORY/ARCHIVE/` from LLM packer outputs by skipping it during manifest generation and file copy.

## Alternatives considered

- Keep `MEMORY/ARCHIVE/` in packs and rely on downstream filtering.
- Move archive content elsewhere (requires broader migration).

## Consequences

- Packs are smaller and focus on active material.
- Archive content remains in-repo for documentation policy, but is not included in pack outputs.

## Enforcement

- LLM packer smoke fixtures assert `MEMORY/ARCHIVE/` is excluded.

## Review triggers

- If archive content becomes necessary for routine handoffs.

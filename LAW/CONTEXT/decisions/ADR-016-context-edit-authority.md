---
id: "ADR-016"
title: "Context Edit Authority"
status: "Accepted"
date: "2025-12-23"
confidence: "High"
impact: "Medium"
tags: ["context", "governance", "amendment"]
---

<!-- CONTENT_HASH: 92f859a18441b71453ff9d36e91d7e45b9a489acd4d8a2ab4d6e9ce46523fe48 -->

# ADR-016: Context Edit Authority

## Problem

CANON/CONTRACT.md Rule 3 and AGENTS.md Rule 1B define conflicting authority for editing CONTEXT records:

- CONTRACT.md: "editing existing records requires **explicit instruction** (from user)"
- AGENTS.md: "edit existing CONTEXT records when the task is **explicitly about** rules, governance, or memory updates"

This creates ambiguity: does an agent need user approval, task intent, or both?

## Decision

**Editing existing CONTEXT records requires BOTH explicit user instruction AND explicit task intent.**

1. **Explicit user instruction** (CONTRACT.md): The user must directly request a context edit in the prompt.
2. **Explicit task intent** (AGENTS.md): The edit must be part of a governance, rules, or memory update task—not a side effect of unrelated work.

Append-only additions to CONTEXT remain unrestricted (e.g., adding new ADRs, creating new preference files).

## Rationale

- CONTRACT.md governs the user-agent contract (what user approval is needed).
- AGENTS.md governs agent autonomy boundaries (what tasks justify internal edits).
- Both constraints must be satisfied to prevent accidental context corruption and maintain user control.
- Append-only additions are safe because they don't destroy prior knowledge; edits risk breaking the audit trail and decision history.

## Consequences

- Agents must ask users before editing any CONTEXT/decisions/*, CONTEXT/rejected/*, or CONTEXT/preferences/* files.
- Agents may create new records in CONTEXT/* without explicit instruction if the task is about governance/rules/memory.
- Canon edits remain governed by CONTRACT.md Rule 2 (behavior change ceremony with fixtures).
- Deletes remain governed by CONTRACT.md Rule 3 (explicit user instruction + confirmation + archiving).

## Enforcement

- critic.py will flag attempts to edit (not append) existing CONTEXT records.
- Agents will refuse such edits unless both user instruction and task intent are present.
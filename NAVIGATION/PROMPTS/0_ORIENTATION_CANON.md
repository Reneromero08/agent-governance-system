---
title: ORIENTATION_CANON
version: 1.1
status: CANONICAL
generated_on: 2026-01-06
scope: Repository-wide AGS orientation document
---
<!-- CANON_HASH (sha256 over file content excluding this line): 4A476F8E6A7EE66ADC3A90E6C7D08684386054464D0AD4F12DA78D617C4B4C4C -->

# ORIENTATION_CANON

This document orients humans and LLM executors to the Agent Governance System (AGS) repository: what it is, what is canonical, where truth lives, and how work is performed without drift.

## What AGS is
AGS is a repo that treats software work as governed execution:
- Plans must be explicit.
- Side effects must be declared.
- Outputs must be deterministic, provable, and auditable.
- Unknowns must be verified, or execution stops.

## Canon precedence
When instructions conflict, resolve in this order:
1) LAW/CANON/ (non-negotiable rules and invariants)
2) LAW/CONTEXT/ (ADRs and decisions, subordinate to CANON)
3) NAVIGATION/ (maps, roadmaps, indices, operational docs)
4) CAPABILITY/ (implementation)
5) Everything else

If a prompt, tool, or doc conflicts with LAW/CANON, LAW wins.

## System buckets
Every artifact belongs to exactly one bucket (see LAW/CANON/SYSTEM_BUCKETS.md):
1) LAW
2) CAPABILITY
3) MEMORY
4) NAVIGATION
5) THOUGHT
6) INBOX

Practical rule: do not hide durable truth in THOUGHT or temp dirs. Move it to the right bucket.

## Durability model
AGS distinguishes:
- Catalytic state: scratch, cache, temps, generated intermediates
- Durable state: canonical records, governed outputs, proofs, receipts

Durable state must be:
- deterministic (stable ordering, stable encoding)
- reproducible (re-run yields identical artifacts when inputs match)
- auditable (receipts / reports)
- policy-compliant (contracts)

### Authorized output roots (durable)
Follow LAW/CANON/CONTRACT.md “Output roots”:
- LAW/CONTRACTS/_runs/
- NAVIGATION/CORTEX/_generated/
- MEMORY/LLM_PACKER/_packs/

If a tool wants to write elsewhere, it must be explicitly governed and allowed by canon.

## Parallel work model (branches + worktrees)
Parallelism is achieved by isolating execution workspaces.

Definitions:
- Branch: a named line of commits (history pointer).
- Worktree: an additional working directory that checks out a branch in parallel to other worktrees.

Canonical rule:
- Agents execute in isolated workspaces (worktree or clone) on task branches.
- Agents do not execute in the primary repo working directory unless explicitly authorized in the prompt.
- Each task has its own branch. Detached HEAD work is forbidden for agent execution.

Minimum preflight for any executor workspace:
- `git symbolic-ref -q HEAD` must succeed (not detached)
- `git status --porcelain` must be empty (workspace starts clean)

## CORTEX and NAVIGATION/CORTEX
“Cortex” is the repo’s structured index and query substrate. Canonically, system-generated cortex artifacts belong under:
- NAVIGATION/CORTEX/_generated/

If both CORTEX/ and NAVIGATION/CORTEX/ exist in a working tree, treat that as a migration condition:
- do not guess which is canonical
- follow LAW/CANON/CONTRACT.md output-root rule
- prefer NAVIGATION/CORTEX for governed outputs unless LAW says otherwise

## LLM Packer
The packer turns a large repo into stable, shareable context:
- produces deterministic bundles under MEMORY/LLM_PACKER/_packs/
- supports bucket-first inclusion to avoid drift

If you are running a model without filesystem access, you must provide:
- either a pack output (preferred)
- or an explicit snapshot plus the specific files needed

## Prompt Pack system (PROMPTS/)
If PROMPTS/ exists, treat it as the canonical task prompt pack. Prompts must follow the policy canon and templates.

What every prompt must enforce:
- Explicit write allowlist and delete/rename allowlist
- Required facts must be verified (no guessing)
- Two-tier unknowns (BLOCKER vs DEFERRABLE)
- FILL token lifecycle (preflight scan, BLOCKED_UNKNOWN if unresolved)
- Dependencies declared in header depends_on and resolved via PROMPTS/PROMPT_PACK_MANIFEST.json
- Receipts and reports required for every task
- Deterministic receipts and stable ordering everywhere
- Workspace preflight (branch + clean-state) and isolation by default
- Test semantics invariant: detected violations must fail tests

## Execution discipline (how work is done)
Default executor loop (human or model):
1) Preflight
   - read relevant LAW/CANON files
   - verify required facts (including workspace checks)
2) Plan
   - bounded, ordered steps
   - declare what will be written and where
3) Execute
   - small diffs
   - deterministic encoding
4) Validate
   - run required tests and checks
   - confirm semantics: tests fail on violations
5) Receipt + Report
   - write structured receipt JSON and a human report
6) Stop
   - if any validation fails, do not replace failure with narrative

Fail-closed rules:
- If you cannot verify a required fact, stop for that task.
- If validation fails, stop and report failure with evidence.
- If outputs cannot be proven deterministic, stop.

## Minimal “read-first” set for any new contributor or model
1) LAW/CANON/INDEX.md
2) LAW/CANON/CONTRACT.md
3) LAW/CANON/INTEGRITY.md
4) LAW/CANON/INVARIANTS.md
5) AGS_ROADMAP_MASTER.md (master roadmap at repository root)
6) NAVIGATION/PROMPTS/* canon files (policy, guide, templates)

## Glossary anchors
- Canon: rules in LAW/CANON that override everything else
- Receipt: machine-parseable JSON proving what happened, how, with what validations
- Report: human-readable summary aligned with the receipt
- Catalytic: allowed to be transient, disposable, regenerated
- Durable: must be deterministic, auditable, and governed
- CAS ref: sha256:<hash> pointing to bytes in CAS
- Root: a declared starting point for reachability (GC and audits)

## Change policy for this document
This file is canon. Changes should:
- be deliberate
- preserve determinism and fail-closed semantics
- avoid introducing new conventions that conflict with LAW/CANON

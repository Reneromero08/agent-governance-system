---
title: ORIENTATION_CANON
version: 1.0
status: CANONICAL
generated_on: 2026-01-04
scope: Repository-wide AGS orientation document
---
<!-- CANON_HASH: f2880df34aa33d8174272492715b3d4695a8178b68e992d8704d8bf454e29ed9 -->

# ORIENTATION_CANON

This document orients humans and LLM executors to the **Agent Governance System (AGS)** repository: what it is, what is canonical, where truth lives, and how work is performed without drift.

## What AGS is

AGS is a repo that treats software work as governed execution:
- **Plans must be explicit.**
- **Side effects must be declared.**
- **Outputs must be deterministic, provable, and auditable.**
- **Unknowns must be verified, or execution stops.**

## Canon precedence

When instructions conflict, resolve in this order:

1) **LAW/CANON/** (non-negotiable rules and invariants)  
2) **LAW/CONTEXT/** (ADRs and decisions, subordinate to CANON)  
3) **NAVIGATION/** (maps, roadmaps, indices, operational docs)  
4) **CAPABILITY/** (implementation)  
5) Everything else

If a prompt, tool, or doc conflicts with LAW/CANON, **LAW wins**.

## System buckets

Every artifact belongs to exactly one bucket (see LAW/CANON/SYSTEM_BUCKETS.md):

1) **LAW**: rules, contracts, invariants, policy  
2) **CAPABILITY**: code that performs work  
3) **MEMORY**: packed context, archives, reproducible snapshots  
4) **NAVIGATION**: indices, maps, roadmaps, “where to look next”  
5) **THOUGHT**: experiments, drafts, research, scratch  
6) **INBOX**: incoming work items and delegations

Practical rule: **do not “hide” durable truth in THOUGHT or temp dirs.** Move it to the right bucket.

## Durability model

AGS distinguishes:

- **Catalytic state**: scratch, cache, temps, generated intermediates  
- **Durable state**: canonical records, governed outputs, proofs, receipts

Durable state must be:
- deterministic (stable ordering, stable encoding)
- reproducible (re-run yields identical artifacts when inputs match)
- auditable (receipts / reports)
- policy-compliant (CMP and other contracts)

### Authorized output roots (durable)

Follow LAW/CANON/CONTRACT.md “Output roots”:
- `LAW/CONTRACTS/_runs/`
- `NAVIGATION/CORTEX/_generated/`
- `MEMORY/LLM_PACKER/_packs/`

If a tool wants to write elsewhere, it must be explicitly governed and allowed by canon.

## Core primitives (Lane Z, Phase 2)

These are now the backbone of “destroy context limits safely”:

### CAS (Content Addressed Storage)
- Minimal primitives:
  - `put(bytes) -> hash`
  - `get(hash) -> bytes`
- Properties:
  - deterministic hashing over raw bytes
  - write-once semantics
  - deduplication by construction

### Artifact Store
- Wraps CAS with canonical refs: `sha256:<64hex>`
- Supports dual mode:
  - CAS ref paths
  - legacy file paths (for migration)

### RUNS immutable records
Immutable, CAS-backed records:
- TASK_SPEC (inputs)
- STATUS (state transitions)
- OUTPUT_HASHES (ordered outputs)

### GC (Garbage Collection)
A conservative, policy-driven cleanup mechanism:
- mark reachable from roots
- sweep unreachable blobs
- **fail-closed by default** when roots are empty unless explicitly overridden per policy

### Root Audit (Z.2.6.0)
A pre-integration gate:
- verifies roots exist and are computable
- verifies required outputs are reachable
- deterministic receipts
- **no empty-root override** (audit is stricter than GC)

## CORTEX and NAVIGATION/CORTEX

“Cortex” is the repo’s structured index and query substrate. Canonically, **system-generated cortex artifacts belong under**:
- `NAVIGATION/CORTEX/_generated/`

If both `CORTEX/` and `NAVIGATION/CORTEX/` exist in a working tree, treat that as a **migration condition**:
- do not guess which is canonical by vibes  
- follow LAW/CANON/CONTRACT.md output-root rule  
- prefer **NAVIGATION/CORTEX** for governed outputs unless LAW says otherwise

## LLM Packer

The packer is how AGS turns a large repo into stable, shareable context:
- produces deterministic bundles under `MEMORY/LLM_PACKER/_packs/`
- supports “bucket-first” inclusion to avoid drift
- packs are the canonical way to hand context to models that cannot access the full repo

If you are running a model without filesystem access, you must provide:
- either a pack output (preferred)
- or an explicit zip snapshot plus the specific files needed

## Prompt Pack system (PROMPTS/)

If `PROMPTS/` exists, treat it as the canonical “task prompt pack.” Prompts must follow the policy canon and template.

Canon files (as of the last prompt pack build):
- `PROMPTS/PROMPT_POLICY_CANON.md` sha256: `b12a0779118b646c99b02662c4b487f136dd74c1053fd7090095fc45ced99882`
- `PROMPTS/PROMPT_GENERATOR_GUIDE_FINAL.md` sha256: `e4f1d187828771c337b5394cbba6ed5e96b0bd7be66b7db3aacedaf2072def5c`
- `PROMPTS/MASTER_PROMPT_TEMPLATE_CANON.md` sha256: `bf9f11a5af214d6e2b60e4a3b01dadb466144773ed74323b6a72ace4538a4be1`

### What every prompt must enforce
- Explicit write allowlist and delete/rename allowlist
- Required facts must be verified (no guessing)
- Two-tier unknowns:
  - UNKNOWN_BLOCKER: stop that task
  - UNKNOWN_DEFERRABLE: allowed only as `FILL_ME__KEY` tokens inside REQUIRED FACTS
- FILL token lifecycle:
  - executor preflight scans for `FILL_ME__`
  - unresolved tokens stop with `BLOCKED_UNKNOWN`
  - resolved values must pass the same “Verify via” checks
- Dependencies declared in header `depends_on`, resolved via `PROMPTS/PROMPT_PACK_MANIFEST.json`
- Receipts and reports required for every task with explicit `receipt_path` and `report_path`
- Deterministic receipt schema and stable ordering everywhere
- If `scripts/lint-prompt.sh` exists: exit 1 blocks, exit 2 warns

## Execution discipline (how work is done)

### Default executor loop (human or model)
1) **Preflight**
   - read relevant LAW/CANON files
   - enumerate required facts
   - verify required facts via explicit commands
2) **Plan**
   - bounded, ordered steps
   - declare what will be written and where
3) **Execute**
   - small diffs
   - deterministic encoding
4) **Validate**
   - run the repo’s tests (or the task’s required checks)
   - confirm no forbidden writes occurred
5) **Receipt + Report**
   - write structured receipt JSON and a human report
6) **Stop**
   - if any validation fails, do not “patch around” it with narratives

### Fail-closed rules
- If you cannot verify a required fact, stop for that task.
- If validation fails, stop and report failure with evidence.
- If outputs cannot be proven deterministic, stop.

## Minimal “read-first” set for any new contributor or model

1) `LAW/CANON/INDEX.md`  
2) `LAW/CANON/CONTRACT.md`  
3) `LAW/CANON/INTEGRITY.md`  
4) `LAW/CANON/INVARIANTS.md`  
5) `NAVIGATION/ROADMAPS/` (choose the active roadmap file)  
6) `PROMPTS/` policy canon files (if PROMPTS exists)  

## Glossary anchors

- **Canon**: rules in LAW/CANON that override everything else  
- **Receipt**: machine-parseable JSON proving what happened, how, with what validations  
- **Report**: human-readable summary aligned with the receipt  
- **Catalytic**: allowed to be transient, disposable, regenerated  
- **Durable**: must be deterministic, auditable, and governed  
- **CAS ref**: `sha256:<hash>` pointing to bytes in CAS  
- **Root**: a declared starting point for reachability (GC and audits)

## Change policy for this document

This file is canon. Changes should:
- be deliberate
- preserve determinism and fail-closed semantics
- avoid introducing new conventions that conflict with LAW/CANON

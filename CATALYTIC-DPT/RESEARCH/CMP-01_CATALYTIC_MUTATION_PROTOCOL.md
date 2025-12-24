---
title: CMP-01 Catalytic Mutation Protocol
date: 2025-12-23
status: draft-spec
scope: AGS governance primitive
---

# CMP-01 Catalytic Mutation Protocol

## Purpose
Enable agents to temporarily use large, messy state (the “catalyst”) as scratch space while guaranteeing the repo returns to an identical state after the run, except for explicitly allowed outputs. This turns “cleanup” into a hard contract.

This is the bridge from “agents can edit files” to “agents can borrow state without corrupting the universe.”

## One-line rule
An agent may mutate only inside declared catalytic domains and must restore them byte-identical at end-of-run, while writing durable outputs only to allowed output roots.

## Scope
CMP governs any agent run that needs temporary mutation of filesystem state for:
- indexing and compilation passes (Cortex indexes, section indexes, registries)
- pack generation and verification (LITE/FULL/TEST)
- large refactors that need intermediate transforms
- migration scripts that must be reversible until finalized

CMP is not required for pure read-only runs.

## Terms
- **Clean Workspace**: the small, authoritative state that must remain stable and is safe to read from (Canon, contracts, skills, hand-authored docs).
- **Catalyst**: a large writable area that may start in any state and is allowed to be temporarily mutated, provided it is restored exactly.
- **Catalytic Domain**: the set of paths permitted to be mutated under CMP.
- **Durable Output Roots**: the only locations where artifacts may remain after the run.
- **Restoration Proof**: a machine-verifiable record that the catalytic domain was restored exactly.
- **Run Ledger**: the per-run audit bundle storing restoration proof and key run metadata.

## Authority and precedence
CMP is a governance constraint. If CMP conflicts with any lower-level instruction, CMP wins for any run declared “catalytic.”

## Invariants
1. **No silent writes outside catalytic domains.**
2. **No durable artifacts outside allowed output roots.**
3. **Catalytic domains must be byte-identical after the run.**
4. **Restoration proof is mandatory. If proof fails, the run is invalid.**
5. **All mutations must be attributable to a run ID with a run ledger.**
6. **Deterministic mode is preferred. If non-determinism is unavoidable, it must be declared in the run ledger.**

## Allowed catalytic domains (recommended defaults)
Choose one policy and stick to it. These defaults assume your repo already reserves user build output outside the repo.

### Default catalytic domains
- `CONTRACTS/_runs/_tmp/`
- `CORTEX/_generated/_tmp/`
- `MEMORY/LLM_PACKER/_packs/_tmp/`
- `TOOLS/_tmp/` (optional)
- `MCP/_tmp/` (optional)

### Forbidden domains (examples)
- `BUILD/` (reserved for user workspace outputs)
- `CANON/` (unless explicitly running a Canon edit ceremony, which is not catalytic)
- `AGENTS.md` and other root authorities (unless explicitly in a governance ceremony)

## Durable output roots
These are the only places allowed to contain new or changed files after a catalytic run:
- `CONTRACTS/_runs/`
- `CORTEX/_generated/`
- `MEMORY/LLM_PACKER/_packs/`

If you want logs, audits, or indices to persist, they must live under one of these roots or CMP will fail.

## The catalytic lifecycle
A catalytic run has five phases.

### Phase 0: Declare
The run must declare:
- `run_id`
- `catalytic_domains[]`
- `durable_output_roots[]`
- `intent` (one sentence)
- `determinism` (deterministic, bounded nondeterministic, nondeterministic)
- `tools_used[]` (scripts, skills, external tools)

### Phase 1: Snapshot
Before any writes:
- compute a snapshot of each catalytic domain
- snapshot method must be declared (see “Proof formats”)
- record snapshots in the run ledger

### Phase 2: Mutate
Mutations are allowed only inside catalytic domains.
Any write outside catalytic domains is a hard violation.

### Phase 3: Commit durable outputs
Write durable outputs only inside durable output roots.
Durable outputs must be listed in the run ledger.

### Phase 4: Restore
Restore catalytic domains to their original state.
Restoration is not “best effort.” It is exact.

### Phase 5: Prove
Generate restoration proof and validate it.
If validation fails:
- mark run as failed
- attempt rollback again
- if still failing, quarantine the run outputs and stop

## Proof formats
Pick one, implement it everywhere, and enforce it in CI.

### Format A: Content-hash manifest (practical default)
- `snapshot_manifest.json` lists every file under each catalytic domain with:
  - path
  - size
  - mtime (optional)
  - sha256
- Restoration proof is:
  - `pre_manifest.json`
  - `post_manifest.json`
  - `diff.json` (must be empty)

Pros: simple, portable.
Cons: O(n) hashing cost.

### Format B: Merkle root + leaf list (stronger audit)
- build a Merkle tree over normalized path + file hash leaves
- store:
  - `pre_merkle_root`
  - `post_merkle_root`
  - leaf list or proof path per file (implementation choice)

Pros: scalable verification, partial proofs.
Cons: more complexity.

### Format C: Git overlay (only if repo is git-native at runtime)
- use a clean worktree or stash-based overlay and require:
  - `git diff` is empty for catalytic domains after restore
  - durable outputs are outside the overlay or explicitly committed

Pros: fast if you are already in git.
Cons: tooling dependent; not universal.

## Run ledger schema (minimum)
Store under: `CONTRACTS/_runs/<run_id>/`

Required:
- `RUN_INFO.json`
  - run_id, timestamp, intent, determinism, toolchain, catalytic_domains, durable_output_roots
- `PRE_MANIFEST.json`
- `POST_MANIFEST.json`
- `RESTORE_DIFF.json` (must be empty)
- `OUTPUTS.json` (list of durable outputs created or modified)
Optional:
- `WARNINGS.json`
- `PERF.json` (timings, file counts, sizes)
- `NOTES.md` (short human-readable summary)

## Enforcement hooks
CMP should be enforced in three places.

### 1) Local preflight (fast)
Before execution, validate:
- declared paths exist or can be created in allowed roots
- catalytic domains do not overlap forbidden domains
- output roots are allowed

### 2) Runtime guard (hard)
A write-guard layer that tracks filesystem writes during the run.
Minimum viable guard:
- wrap your scripts to record any write path
- after run, fail if any write is outside catalytic domains or durable output roots

### 3) CI validation (non-negotiable)
In CI, for any catalytic run fixture:
- verify restoration proof validates
- verify outputs exist only under durable roots
- verify no repo-root drift

## Integration points
### Cortex indexing
If you generate indexes:
- do it inside a catalytic domain `_tmp/`
- write final indexes to `CORTEX/_generated/`
- include the index files in `OUTPUTS.json`

### LLM packer
If you generate packs:
- stage intermediates in a catalytic domain
- write final pack to `MEMORY/LLM_PACKER/_packs/`
- emit pack manifest and hash inventory as durable outputs

### Skills
Any skill that mutates the filesystem must declare whether it is:
- read-only
- catalytic
- non-catalytic writer (discouraged, should be rare)

## Failure handling
- **Hard fail** on any write outside catalytic domains.
- **Hard fail** on restoration proof mismatch.
- If restore fails:
  - quarantine outputs under `CONTRACTS/_runs/<run_id>/quarantine/`
  - stop and require human arbitration

## Threat model coverage (minimal)
CMP protects against:
- accidental repo pollution
- incremental drift from repeated agent runs
- “helpful” agents writing caches in random folders
- partial runs leaving intermediate files behind

CMP does not protect against:
- malicious code that bypasses guards at the OS level
- external side effects (network calls, remote writes) unless separately sandboxed

## Examples

### Example 1: Build a section index for all Markdown
- Catalytic domains: `CORTEX/_generated/_tmp/`
- Durable outputs: `CORTEX/_generated/SECTION_INDEX.json`, `CORTEX/_generated/SECTION_SUMMARY.json`
- Proof: manifests pre and post for `_tmp/`

### Example 2: Generate a LITE pack
- Catalytic domains: `MEMORY/LLM_PACKER/_packs/_tmp/`
- Durable outputs: `MEMORY/LLM_PACKER/_packs/<pack_id>/...`
- Proof: manifests for `_tmp/`
- Extra: pack manifest and hashes required

### Example 3: “Try” a repo refactor safely
- Stage changes in catalytic domain only
- Generate a proposed diff as durable output
- Restore catalytic domain
- Human reviews diff, then runs a non-catalytic “apply diff” ceremony

## Definition of done for CMP implementation
CMP is “done” when:
1. There is a single CMP spec (this file) and it is referenced by Canon or the governing docs.
2. At least one catalytic workflow exists (index build or pack build) that:
   - produces a run ledger
   - validates restoration proof
3. CI fails if:
   - restoration proof fails
   - outputs appear outside allowed roots
4. A write-guard exists (even if simple) that catches out-of-domain writes.


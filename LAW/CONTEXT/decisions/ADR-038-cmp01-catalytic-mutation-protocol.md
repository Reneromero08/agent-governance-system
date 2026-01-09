---
id: "ADR-038"
title: "CMP-01 Catalytic Mutation Protocol"
status: "Accepted"
date: "2026-01-07"
confidence: "High"
impact: "High"
deciders: ["System architects"]
tags: ["catalytic", "mutation", "protocol", "agents", "governance"]
---

# ADR-038: CMP-01 Catalytic Mutation Protocol

**Status**: Accepted
**Date**: 2026-01-07
**Deciders**: System architects
**Context**: Phase 1.6 documentation gap

## Context

The Agent Governance System needed a formal protocol for agent runs that temporarily mutate filesystem state. Without such a protocol:
- Agents could leave temporary files scattered across the repo
- Incremental drift would accumulate from repeated runs
- There was no way to verify restoration actually happened
- Debugging failed runs required forensic analysis

## Decision

Adopt CMP-01 (Catalytic Mutation Protocol) as the canonical specification for catalytic execution. The protocol mandates:

1. **Declared Domains**: Agents MUST declare which paths they will mutate before execution
2. **Snapshot-Restore Cycle**: Pre-run snapshot, execution, post-run restoration, proof generation
3. **Proof-Gated Acceptance**: A run is valid if and only if `PROOF.json.restoration_result.verified == true`
4. **Three Enforcement Layers**: Preflight validation, runtime guard, CI gate

## Alternatives Considered

### Alternative 1: Git-Based Restoration Only

Use `git stash` or worktrees for all temporary state.

**Rejected because**:
- Not all environments have git available at runtime
- Git overhead for high-frequency operations
- No proof artifact generated
- Can't handle non-git-tracked temporary files

### Alternative 2: Overlay Filesystem

Use copy-on-write overlays (e.g., overlayfs, unionfs).

**Rejected because**:
- Platform-specific (Linux only)
- Requires elevated permissions
- Complex setup for each run
- Doesn't integrate with existing tooling

### Alternative 3: No Formal Protocol (Trust Agents)

Let agents clean up after themselves.

**Rejected because**:
- No verification that cleanup happened
- Fails open on agent bugs
- No audit trail
- Violated integrity-first design principle

### Alternative 4: Immutable Execution Containers

Run everything in containers that are discarded after.

**Rejected because**:
- Heavy runtime overhead
- Complex orchestration required
- Doesn't preserve durable outputs naturally
- Over-engineered for the problem scope

## Consequences

### Positive

- **Deterministic verification**: Every run produces a provable restoration artifact
- **Fail-closed enforcement**: Invalid runs are rejected at multiple layers
- **Audit trail**: Complete run ledgers for debugging and compliance
- **Composable**: Works with existing tooling (packer, CORTEX, skills)

### Negative

- **Overhead**: Snapshotting and hashing adds latency to every catalytic run
- **Complexity**: Six-phase lifecycle is more complex than ad-hoc approaches
- **Learning curve**: Agents must understand domain declarations

### Neutral

- Memoization offsets snapshot overhead for repeated runs
- Protocol complexity is hidden behind `catalytic_runtime.py`

## Implementation

- **Canon**: `LAW/CANON/CATALYTIC/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md`
- **Theory**: `LAW/CANON/CATALYTIC/CATALYTIC_COMPUTING.md`
- **Runtime**: `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py`
- **Validator**: `CAPABILITY/TOOLS/catalytic/catalytic_validator.py`
- **Schemas**: `LAW/SCHEMAS/jobspec.schema.json`, `ledger.schema.json`, `proof.schema.json`

## References

- Phase 1.6 in `AGS_ROADMAP_MASTER.md`
- ADR-018: Catalytic Computing Canonical Note
- INV-006: Output roots invariant

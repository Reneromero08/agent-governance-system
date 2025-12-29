# ADR-027: Dual-DB Architecture (System 1 / System 2)

**Status:** Accepted
**Date:** 2025-12-28
**Confidence:** High
**Impact:** High
**Tags:** [architecture, database, system-design]

## Context
As the Agent Governance System (AGS) scales, maintaining a single monolithic context for both rapid retrieval and strict governance has become inefficient. We need to distinguish between "fast" operations (retrieval, search, context stuffing) and "slow" operations (audit, provenance, validation).

This mirrors the cognitive model described by Daniel Kahneman:
- **System 1 (Fast)**: Intuitive, heuristic, efficient, read-heavy.
- **System 2 (Slow)**: Analytical, deliberative, verifiable, write-heavy.

## Decision
We will architect the `CORTEX` and data flow into two distinct systems:

### System 1: The Fast Retrieval Layer
- **Purpose**: Low-latency providing of context to agents.
- **Backing**: SQLite Index (`system1.db`) + Vector Store (if needed).
- **Content**:
  - Deterministic chunks of all repo files.
  - Content-Addressed Storage (CAS) for deduplication.
  - **F3 Strategy**: Uses hash pointers to retrieve content.
- **Invariants**: 
  - Read-Only optimization.
  - "Close enough" is acceptable for search (fuzzy matches).
  - Must accept massive throughput.

### System 2: The Slow Governance Layer
- **Purpose**: strict verification of **truth**.
- **Backing**: Immutable Ledger (`system2.db` or `mcp_ledger`).
- **Content**:
  - **Provenance**: Who did what, when, and with what tool.
  - **Validation**: Pass/Fail results from `critic` and `runner`.
  - **Integrity**: Merkle roots of artifact bundles.
- **Invariants**:
  - Write-Heavy (append-only).
  - "Close enough" is **reject**. Zero drift allowed.
  - Must provide cryptographic proof of history.

## Consequences
1. **Separation of Concerns**: Agents use System 1 tools for "seeing" and System 2 tools for "signing/verifying".
2. **Performance**: Ingestion can be lazy for System 1 (eventual consistency), while System 2 is immediate (strict consistency).
3. **Token Economy**: System 1 uses specialized indices (SCL/Codebook) to compress context by 90%, only expanding when necessary.
4. **Implementation**: Requires `AGS_ROADMAP_MASTER.md` Lane C updates (P0).

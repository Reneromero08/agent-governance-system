# ADR-005: Persistent Research Cache

**Status:** Accepted
**Date:** 2025-12-21
**Confidence:** High
**Impact:** Medium
**Tags:** [research, optimization, persistence]
**Deciders:** Antigravity (Agent), User

## Context

Agents frequently research similar topics or URLs across different sessions. Without a persistent cache, the agent wastes tokens and time re-browsing the same pages, and risks inconsistent summaries if the page content changes slightly.

## Decision

We will implement a **Persistent Research Cache** using **SQLite**.

1.  **Storage**: A single file `CONTEXT/research/research_cache.db`.
2.  **Schema**: Keyed by `SHA-256(URL)`, storing `summary`, `tags`, `timestamp`, and `last_accessed`.
3.  **Interface**: A CLI tool `TOOLS/research_cache.py` for atomic Save/Lookup/List operations.

## Alternatives considered

- **Flat JSON Files**:
    - *Rejected*: Concurrency issues if multiple agents read/write. Reading the whole file for one lookup is O(N).
- **In-Memory Only**:
    - *Rejected*: Does not solve the cross-session redundancy problem.

## Rationale

SQLite provides O(1) lookups, ACID compliance for concurrent access (future-proofing for multi-agent), and zero configuration. It is the standard for "embedded" database needs in AGS (matching Cortex).

## Consequences

- **Git Pattern**: `*.db` must be strictly ignored in `.gitignore`.
- **Maintenance**: We now own a database schema. Any changes to the columns require a migration script (or deleting the cache).

## Enforcement

- Agents attempting research should first check `research_cache.py --lookup`.
- `critic.py` (future) could warn if a URL is browsed without a cache check.

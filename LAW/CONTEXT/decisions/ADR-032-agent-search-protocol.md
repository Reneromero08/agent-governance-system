---
id: "ADR-032"
title: "Agent Search Protocol (Semantic by Default)"
status: "Accepted"
date: "2026-01-02"
confidence: "High"
impact: "High"
tags: ["governance", "agents", "search", "protocol"]
priority: "High"
---

# ADR-032: Agent Search Protocol (Semantic by Default)

## Context

For nearly a month, agents have had access to high-quality semantic search tools (`semantic_search`, `cortex_query`) but have defaulted to using `grep_search` (keyword search) for almost all queries. This behavior is inefficient for several reasons:
1.  **Missed Connections**: Keyword search only finds exact string matches, missing conceptual relationships vital for understanding a complex system like AGS.
2.  **Token Waste**: Passing full file contents from grep results wastes tokens compared to the optimized chunks returned by the semantic index (96% token savings).
3.  **Entropy**: Relying on brute-force keyword search increases "systemic surprise" (free energy) by failing to leverage the system's own crystallized knowledge structure.

The system needs a clear, governed decision framework (protocol) to dictate when agents must use which search tool.

## Decision

We adopt the **Agent Search Protocol** as a mandatory governance standard for all agents operating within the repository.

### 1. The Protocol
The protocol is defined by a strict decision tree:

*   **Conceptual Queries** (e.g., "How does compression work?", "What is the relationship between X and Y?"):
    *   **MUST** use `semantic_search`.
    *   **Rationale**: Leveraging the vector index is faster, cheaper, and yields better qualitative results.
*   **Exact String/Path Matches** (e.g., "Find all files containing 'SPECTRUM-02'", "Where is 'MEMORY/LLM_PACKER'?"):
    *   **MUST** use `grep_search`.
    *   **Rationale**: Semantic search is probabilistic; keyword search is deterministic and necessary for code refactoring or specific target location.
*   **Structured Lookups** (e.g., "Find all ADRs about governance"):
    *   **SHOULD** use `cortex_query` or `context_search`.
    *   **Rationale**: Queries against the structured metadata index are most efficient for entity retrieval.
*   **Ambiguous Queries**:
    *   **SHOULD** try `semantic_search` first.
    *   **Rationale**: It is better to start broad and narrow down than to miss the target entirely with a failed keyword guess.

### 2. Location
The protocol is codified in `LAW/CANON/AGENT_SEARCH_PROTOCOL.md` and referenced mandatorily in `AGENTS.md`.

## Consequences

### Positive
*   **Higher Intelligence**: Agents will retrieve more relevant, conceptually linked information.
*   **Lower Cost**: Significant reduction in token usage by retrieving relevant chunks instead of full files.
*   **Systemic Intuition**: Aligns agent behavior with the "Free Energy Principle" by using the system's internal model (vectors) to minimize surprise.

### Negative
*   **Habit Breaking**: Current agent prompts/habits heavily favor `grep`; this requires explicit instruction (which has been added to `AGENTS.md`).
*   **Tool Overhead**: Agents must choose between multiple search tools instead of defaulting to one.

## Implementation status
*   [x] Protocol Document created: `LAW/CANON/AGENT_SEARCH_PROTOCOL.md` (2026-01-02)
*   [x] `AGENTS.md` updated with mandatory instruction (2026-01-02)
*   [ ] Tool enforcement (future `critic.py` check)

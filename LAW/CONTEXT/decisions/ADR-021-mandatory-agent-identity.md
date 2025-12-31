# ADR-021: Mandatory Agent Identity and Observability

**Date:** 2025-12-27
**Status:** Accepted
**Confidence:** High
**Impact:** High
**Tags:** [governance, observability, swarm]

## Context
As the swarm of agents grows, anonymous or untraceable execution becomes a governance risk. We need a way to know "who is doing what" at any given time, both for real-time monitoring (via `agent-activity`) and for post-hoc auditing. Currently, agents could theoretically operate without a trace if logs weren't strictly enforced.

## Decision
We mandate **Traceable Identity** as a core canon rule.

1.  **Session Identity:** Every distinct agent session must be assigned a unique `session_id` (UUID) upon connection or initialization.
2.  **Audit Logging:** Every tool execution, state change, or resource access must be logged to the immutable audit log (`LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl`), tagged with this `session_id`.
3.  **Observability:** The system must provide mechanisms (like the `agent-activity` skill) to query this state.

## Consequences
**Positive:**
- Enables the `agent-activity` skill to function reliably.
- Allows distinguishing between concurrent agents in logs.
- Prevents "ghost" agents from operating unnoticed.

**Negative:**
- Requires all entrypoints (MCP, CLI wrappers) to implement session tracking. (Already implemented in MCP server).
- Slightly increases log volume.

## Compliance
- `CAPABILITY/MCP/server.py` has been updated to generate `session_id` and include it in logs.
- `CANON/CONTRACT.md` will be updated to reflect this requirement.

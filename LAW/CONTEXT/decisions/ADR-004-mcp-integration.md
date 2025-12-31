# ADR-004: Model Context Protocol (MCP) Integration

**Status:** Accepted
**Date:** 2025-12-21
**Confidence:** High
**Impact:** High
**Tags:** [integration, protocol, architecture]
**Deciders:** Antigravity (Agent), User

## Context

The Agent Governance System (AGS) needs a standardized way to expose its tools (`cortex_query`, `policy_check`, `context_search`) to external LLM clients (like Claude Desktop or IDE extensions) without rewriting custom glue code for every client.

## Decision

We will implement the **Model Context Protocol (MCP)** specification.

1.  **Transport**: We will use `stdio` (Standard Input/Output) as the transport layer. This is the simplest and most robust method for local desktop integration.
2.  **Server Location**: The implementation will reside in `CAPABILITY/MCP/server.py` and `CAPABILITY/MCP/schemas/`.
    - The recommended runtime entrypoint is `LAW/CONTRACTS/ags_mcp_entrypoint.py` to keep audit logs under allowed output roots.
3.  **Governance Integration**: We will expose governance-critical tools (`critic`, `ceremony`) via MCP, turning the IDE into a governed environment.
4.  **Verification**: MCP readiness is verified via `mcp-smoke` (CLI) and `mcp-extension-verify` (extension checklist).

## Alternatives considered

- **REST API (FastAPI)**:
    - *Pros*: Standard HTTP tech.
    - *Cons*: Requires port management, firewall rules, and more complex auth for local desktop use.
- **Custom CLI JSON Interface**:
    - *Pros*: We already partially have this.
    - *Cons*: Not interoperable with standard AI clients.

## Rationale

MCP is emerging as the standard for connecting AI models to local context. Adopting it aligns AGS with the broader ecosystem and allows "Code-Less" integration with any MCP-compliant client.

## Consequences

- **Dependency**: We implicitly depend on the stability of the MCP spec.
- **Security**: Exposing `run_command` or file write tools via MCP requires strict `critic` gates (implemented via `heuristic_safe` flags).
- **Audit Logs**: When launched via the wrapper entrypoint, MCP audit logs are written to `LAW/CONTRACTS/_runs/mcp_logs/`.

## Enforcement

- All new external tools must be added to `CAPABILITY/MCP/schemas/tools.json`.
- The `CAPABILITY/MCP/server.py` implementation is the reference implementation.
- MCP verification uses the `mcp-smoke` and `mcp-extension-verify` skills.

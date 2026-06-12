<!-- CONTENT_HASH: 30d8263b115bf5bf84af98f390571658dfa37a3d79f5a1a71423c5c9a0a390da -->

# MCP Integration Specification

This document defines how the Agent Governance System (AGS) integrates with the Model Context Protocol (MCP).

## Overview

MCP (Model Context Protocol) is an open standard for AI systems to communicate with external tools and data sources. AGS acts as an **MCP Server**, exposing its capabilities to MCP-compatible clients (Claude, AI IDEs, etc.).

## AGS as MCP Server

### Core Primitives Mapping

| MCP Primitive | AGS Capability | Description |
|---------------|----------------|-------------|
| **Tools** | Skills + Cortex + Memory | Executable actions (run skills, semantic search, memory ops) |
| **Resources** | Context + Canon + Cortex | Read-only access to decisions, canon, and file index |
| **Prompts** | Genesis + Templates | Pre-configured prompts for common AGS operations |

### Exposed Tools

The authoritative tool schemas live in `schemas/tools.json`; handlers in
`server.py` must stay in sync with that file.

| Tool Name | Description | Input Schema |
|-----------|-------------|--------------|
| `context_search` | Search context records (ADRs, preferences) | `{ query?: string, type?: string, tags?: string[], status?: string }` |
| `context_review` | Check for overdue ADR reviews | `{ days?: number }` |
| `canon_read` | Read a canon file | `{ file: string }` |
| `skill_run` | Execute a skill (governed) | `{ skill: string, input: object }` |
| `codebook_lookup` | Look up codebook entries | `{ id?: string, query?: string, expand?: boolean, semantic?: boolean, limit?: number, list?: boolean }` |
| `skill_discovery` | Find skills by natural-language intent | `{ query: string, top_k?: number, threshold?: number }` |
| `find_related` | Find related artifacts | `{ artifact_id: string, top_k?: number, threshold?: number }` |
| `cassette_network_query` | Semantic search over the cassette network | `{ query: string, limit?: number, capability?: string }` |
| `semantic_stats` | Embedding / network statistics | `{}` |
| `memory` | Unified memory ops | `{ operation: "save"\|"query"\|"recall"\|"neighbors"\|"stats", ... }` |
| `session_info` | Session metadata and audit entries | `{ include_audit_log?: boolean, limit?: number }` |

### Exposed Resources

The authoritative resource list lives in `schemas/resources.json`.

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `ags://canon/contract` | The CONTRACT.md | `text/markdown` |
| `ags://canon/invariants` | The INVARIANTS.md | `text/markdown` |
| `ags://canon/genesis` | The GENESIS.md bootstrap prompt | `text/markdown` |
| `ags://canon/versioning` | The VERSIONING.md | `text/markdown` |
| `ags://canon/arbitration` | The ARBITRATION.md | `text/markdown` |
| `ags://canon/deprecation` | The DEPRECATION.md | `text/markdown` |
| `ags://canon/migration` | The MIGRATION.md | `text/markdown` |
| `ags://context/decisions` | List of all ADRs | `application/json` |
| `ags://context/preferences` | List of all STYLE records | `application/json` |
| `ags://context/rejected` | List of rejected proposals | `application/json` |
| `ags://context/open` | List of open questions | `application/json` |
| `ags://cortex/index` | The full cortex index | `application/json` |
| `ags://maps/entrypoints` | The ENTRYPOINTS.md | `text/markdown` |
| `ags://agents` | The AGENTS.md | `text/markdown` |

### Exposed Prompts

| Prompt Name | Description |
|-------------|-------------|
| `genesis` | The Genesis Prompt for session bootstrapping |
| `skill_template` | Template for creating new skills |
| `conflict_resolution` | Guide for resolving conflicts in Canon (Arbitration) |
| `deprecation_workflow` | Checklist for deprecating tokens or features |

## Governance

`skill_run` is the only tool that executes code. It is wrapped by a
fail-closed governance gate that runs, in order:

1. **Preflight** - `CAPABILITY/TOOLS/ags.py preflight`
2. **Admission control** - `CAPABILITY/TOOLS/ags.py admit --intent <path>`
   (intent auto-generated as artifact-only when `AGS_INTENT_PATH` is unset)
3. **Critic** - `CAPABILITY/TOOLS/governance/critic.py`

Skill execution is unbounded by default; set `AGS_SKILL_TIMEOUT` (seconds)
to enforce a limit.

## Entrypoint and logging

To satisfy output-root invariants, use the wrapper entrypoint:
- `LAW/CONTRACTS/ags_mcp_entrypoint.py`

This redirects audit logs to:
- `LAW/CONTRACTS/_runs/mcp_logs/`

Verification: `python CAPABILITY/MCP/verify_governance.py` (smoke test) or
`python LAW/CONTRACTS/ags_mcp_entrypoint.py --test` (self test).

### Future Work
- HTTP transport with authentication (for remote access)
- OAuth 2.0 support for production deployments

## Security Considerations

### Authentication

MCP servers should authenticate clients before granting access. Options:
- OAuth 2.0 (recommended for production)
- API key (simple, suitable for local development)
- No auth (localhost only)

### Authorization

AGS governance rules apply to MCP interactions:
- Tools that modify state require the same ceremonies as direct access
- CANON read access is unrestricted; writes follow intent gates

### Principle of Least Privilege

- Default to read-only access
- Require explicit capability grants for write operations
- Log all tool invocations for audit (`LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl`)

## File Structure

```
CAPABILITY/MCP/
  server.py                      # MCP server (dispatch + tool handlers; canonical import facade)
  protocol.py                    # stdio framing (Content-Length / JSONL auto-detect)
  audit.py                       # SessionAuditTracker (ELO file/symbol/search tracking)
  selftest.py                    # --test mode body (run_selftest)
  semantic_adapter.py            # Cassette network / memory / ELO bridge
  validation.py                  # CMP-01 path governance + SPECTRUM-02 bundles
  primitives.py                  # Atomic file ops, locking, task validation
  verify_governance.py           # Governance smoke test
  server_wrapper.py              # Optional background-instance launcher
  start_simple.cmd               # Foreground stdio launcher
  powershell_bridge.ps1          # Standalone command bridge (not an MCP tool)
  schemas/
    tools.json                   # Tool schemas (authoritative)
    resources.json               # Resource list (authoritative)
    governance/                  # ADR / skill / style metadata schemas
```

Runtime wrapper entrypoint (recommended):
- `LAW/CONTRACTS/ags_mcp_entrypoint.py`

## Transport

AGS MCP server supports:
- **stdio** (default): JSON-RPC 2.0, auto-detecting Content-Length framed
  or newline-delimited JSON
- **HTTP + SSE**: planned (the `--http` flag is currently a stub)

## Status

All 11 tools implemented with schemas; dynamic resources working; Claude
Desktop ready.

Added: 2025-12-21
Last reconciled with server.py: 2026-06-12

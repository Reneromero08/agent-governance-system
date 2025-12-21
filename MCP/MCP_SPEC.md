# MCP Integration Specification

This document defines how the Agent Governance System (AGS) integrates with the Model Context Protocol (MCP). It stages the interface without full implementation, following the principle: "implement only when you actually need tool access."

## Overview

MCP (Model Context Protocol) is an open standard for AI systems to communicate with external tools and data sources. AGS can act as an **MCP Server**, exposing its capabilities to MCP-compatible clients (Claude, AI IDEs, etc.).

## AGS as MCP Server

### Core Primitives Mapping

| MCP Primitive | AGS Capability | Description |
|---------------|----------------|-------------|
| **Tools** | Skills | Executable actions (run skills, query cortex, validate packs) |
| **Resources** | Context + Cortex | Read-only access to decisions, canon, and file index |
| **Prompts** | Genesis + Templates | Pre-configured prompts for common AGS operations |

### Exposed Tools

| Tool Name | Description | Input Schema |
|-----------|-------------|--------------|
| `cortex_query` | Query the Cortex index for files | `{ query: string, type?: string }` |
| `context_search` | Search context records (ADRs, preferences) | `{ query?: string, tags?: string[], status?: string }` |
| `context_review` | Check for overdue ADR reviews | `{ days?: number }` |
| `skill_run` | Execute a skill | `{ skill: string, input: object }` |
| `pack_validate` | Validate a memory pack | `{ pack_path: string }` |
| `canon_read` | Read a canon file | `{ file: string }` |

### Exposed Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `ags://canon/contract` | The CONTRACT.md | `text/markdown` |
| `ags://canon/invariants` | The INVARIANTS.md | `text/markdown` |
| `ags://canon/genesis` | The GENESIS.md bootstrap prompt | `text/markdown` |
| `ags://context/decisions` | List of all ADRs | `application/json` |
| `ags://context/preferences` | List of all STYLE records | `application/json` |
| `ags://cortex/index` | The full cortex index | `application/json` |
| `ags://maps/entrypoints` | The ENTRYPOINTS.md | `text/markdown` |

### Exposed Prompts

| Prompt Name | Description |
|-------------|-------------|
| `genesis` | The Genesis Prompt for session bootstrapping |
| `commit_ceremony` | The commit ceremony checklist |
| `adr_template` | Template for creating new ADRs |
| `skill_template` | Template for creating new skills |

## Implementation Approach

### Phase 1: Interface Definition (Current)

- Define tool schemas (JSON Schema for inputs/outputs)
- Define resource URIs and content types
- Define prompt templates
- Create MCP server stub that returns "not implemented" for unimplemented features

### Phase 2: Read-Only Implementation

- Implement resource reading (canon, context, cortex)
- Implement read-only tools (cortex_query, context_search)
- Test with MCP Inspector

### Phase 3: Full Implementation

- Implement write tools (skill execution)
- Add authentication/authorization
- Integrate with Claude Desktop or other MCP clients

## Security Considerations

### Authentication

MCP servers should authenticate clients before granting access. Options:
- OAuth 2.0 (recommended for production)
- API key (simple, suitable for local development)
- No auth (localhost only)

### Authorization

AGS governance rules apply to MCP interactions:
- Tools that modify state require the same ceremonies as direct access
- Commit ceremony applies even through MCP
- CANON read access is unrestricted; writes follow intent gates

### Principle of Least Privilege

- Default to read-only access
- Require explicit capability grants for write operations
- Log all tool invocations for audit

## File Structure

```
MCP/
  server.py           # MCP server entry point
  handlers/
    tools.py          # Tool handlers
    resources.py      # Resource handlers
    prompts.py        # Prompt handlers
  schemas/
    tools.json        # JSON Schema for tools
    resources.json    # JSON Schema for resources
  config.json         # Server configuration
```

## Transport

AGS MCP server supports:
- **stdio** (default): For local integration with Claude Desktop
- **HTTP + SSE**: For remote access (requires authentication)

## Status

**Seam Staged** — Interface defined, implementation pending.

Added: 2025-12-21

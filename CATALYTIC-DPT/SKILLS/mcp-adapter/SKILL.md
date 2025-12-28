
---
name: mcp-adapter
version: "0.1.0"
description: Generic adapter for executing Model Context Protocol (MCP) servers as governed pipeline steps. Use this to wrap any standard MCP server (Python/Node) into a verifiable CAT-DPT pipeline step with strict I/O caps and transcript hashing.
---

# MCP Adapter

This skill wraps an MCP server execution in a governance envelope.

## Usage

This skill is primarily used by the `ags` runtime to execute MCP servers safely.

### Wrapper Script
`scripts/wrapper.py` takes a JSON configuration file and produces a deterministic JSON output artifact.

```bash
python scripts/wrapper.py config.json output.json
```

### Configuration Schema
See `CATALYTIC-DPT/SCHEMAS/mcp_adapter.schema.json`.

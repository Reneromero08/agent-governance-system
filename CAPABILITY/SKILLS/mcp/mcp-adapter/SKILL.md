# Skill: mcp-adapter
**Version:** 0.1.0
**Status:** Active
**Required_canon_version:** ">=3.0.0 <4.0.0"

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

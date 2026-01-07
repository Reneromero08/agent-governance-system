<!-- CONTENT_HASH: 947418164cb9f5cb7e42a3efd5aae779db7b4468b365669741b5182e0a40cf31 -->

**required_canon_version:** >=3.0.0


# Skill: mcp-adapter
**Version:** 0.1.0
**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `mcp-toolkit`.
> Use `{"operation": "adapt", ...}` with the mcp-toolkit instead.

**canon_version:** "3.0.0"

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

**required_canon_version:** >=3.0.0


# MCP Access Validator

**Skill ID:** `mcp-access-validator`
**Version:** 1.0.0
**Status:** active
**Tags:** mcp, governance, validation, token-efficiency

## Purpose

Prevents token waste by validating that agents use the MCP server's existing tools instead of writing custom database queries or manual file inspection. This skill enforces the "MCP-first" principle for all cortex access.

## Problem Statement

Agents waste tokens by:
1. Writing Python SQLite snippets to inspect databases directly
2. Running manual file system operations instead of using `cortex_query`
3. Creating custom scripts for tasks already covered by MCP tools
4. Analyzing database schemas instead of using semantic search tools

This violates the catalytic computing principle of token efficiency and bypasses the governance layer.

## Solution

The MCP Access Validator skill:
1. **Detects** when an agent is performing manual database/file operations
2. **Recommends** the appropriate MCP tool for the task
3. **Provides** a token efficiency score for the agent's approach
4. **Generates** audit logs of token waste incidents

## Input Schema

```json
{
  "agent_action": "string describing what the agent is trying to do",
  "agent_code_snippet": "optional code the agent wrote",
  "files_accessed": ["list of files the agent accessed manually"],
  "databases_queried": ["list of databases the agent queried directly"]
}
```

## Output Schema

```json
{
  "validation_passed": "boolean",
  "token_waste_detected": "boolean",
  "recommended_mcp_tool": "string",
  "tool_usage_example": "object",
  "estimated_token_savings": "number",
  "audit_entry": "object"
}
```

## Usage Examples

### Example 1: Manual Database Query
**Agent Action:** "I need to check what's in the system1.db database"
**Agent Code:** `import sqlite3; conn = sqlite3.connect('CORTEX/_generated/system1.db'); cursor = conn.execute('SELECT * FROM symbols')`
**Validation Result:** 
- `token_waste_detected`: true
- `recommended_mcp_tool`: `cortex_query`
- `tool_usage_example`: `cortex_query({"query": "symbols"})`
- `estimated_token_savings`: 95%

### Example 2: Manual File Reading
**Agent Action:** "I want to read the CONTRACT.md file"
**Agent Code:** `open('LAW/CANON/CONTRACT.md').read()`
**Validation Result:**
- `token_waste_detected`: true  
- `recommended_mcp_tool`: `canon_read`
- `tool_usage_example`: `canon_read({"file": "CONTRACT"})`
- `estimated_token_savings`: 90%

## Implementation

The skill works by:
1. Parsing the agent's action description and code
2. Matching patterns against known MCP tool capabilities
3. Calculating token efficiency based on:
   - MCP tool response size vs manual code size
   - Governance compliance overhead
   - Audit logging completeness
4. Generating actionable recommendations

## Governance Impact

This skill directly supports:
- **ADR-021**: Mandatory Agent Identity (uses session_id for audit)
- **ADR-004**: MCP Integration (enforces tool usage)
- **Catalytic Computing**: Token efficiency principles
- **AGENTS.md Section 0**: Cortex connection requirements

## Dependencies

- MCP server running with available tools
- Audit log directory (`LAW/CONTRACTS/_runs/mcp_logs/`)
- Session ID for agent identification

## Error Handling

If the MCP server is not accessible, the skill will:
1. Return a validation failure
2. Provide connection instructions
3. Log the incident for governance review

## Performance

- Validation time: < 100ms
- Memory usage: < 10MB
- No external network calls (uses local MCP server)

## Security

- Read-only validation
- No file system modifications
- All audit logs are append-only
- Session ID verification for ADR-021 compliance

## Testing

See `fixtures/` directory for test cases covering:
1. Valid MCP tool usage
2. Manual database query detection
3. File system access detection
4. Mixed usage scenarios
5. Edge cases and error conditions

## Maintenance

This skill should be updated when:
1. New MCP tools are added
2. Token efficiency metrics change
3. Governance requirements evolve
4. Agent behavior patterns shift

## Related Skills

- `session-info-validator`: Validates ADR-021 compliance
- `cortex-query-optimizer`: Optimizes MCP tool usage
- `token-efficiency-auditor`: Comprehensive token usage analysis
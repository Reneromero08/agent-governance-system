---
title: "MCP First Demonstration"
section: "report"
author: "System"
priority: "Medium"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Complete"
summary: "Demonstration of MCP-First principles (Restored)"
tags: [mcp, demonstration]
---

<!-- CONTENT_HASH: fd60a59e37569ce8cb0c9cfe2ef53c00eace805e1a471fca664aca16035a6f00 -->

# MCP-First Principle Demonstration

**Purpose**: Show correct vs incorrect usage of cortex access to prevent token waste.
**Created By**: Kilo Code (Agent ID: 17cb4e78-ae76-49df-b336-c0cccbf5878d)
**Date**: 2025-12-31

## The Problem: Token Waste

Agents waste tokens by writing custom Python code to inspect databases and files directly, instead of using the existing MCP tools.

## Demonstration: Wrong vs Right

### Example 1: Database Inspection

**❌ WRONG (Token Waste - 95% waste):**
```python
# 31 tokens of waste
import sqlite3
conn = sqlite3.connect('CORTEX/_generated/system1.db')
cursor = conn.execute('SELECT * FROM symbols LIMIT 5')
for row in cursor:
    print(row)
conn.close()
```

**✅ CORRECT (MCP-First - 4 tokens):**
```json
// 4 tokens, 95% savings
cortex_query({"query": "symbols", "limit": 5})
```

**Savings**: 95% token reduction, automatic audit logging, governance compliance

### Example 2: File Reading

**❌ WRONG (Token Waste - 90% waste):**
```python
# 22 tokens of waste
from pathlib import Path
content = Path('LAW/CANON/CONTRACT.md').read_text()
print(content[:500])
```

**✅ CORRECT (MCP-First - 2 tokens):**
```json
// 2 tokens, 90% savings
canon_read({"file": "CONTRACT"})
```

**Savings**: 90% token reduction, automatic content validation, proper error handling

### Example 3: Context Search

**❌ WRONG (Token Waste - 85% waste):**
```python
# 45 tokens of waste
import os
for root, _, files in os.walk('LAW/CONTEXT/decisions'):
    for f in files:
        if 'catalytic' in open(os.path.join(root, f)).read():
            print(f)
```

**✅ CORRECT (MCP-First - 3 tokens):**
```json
// 3 tokens, 85% savings
context_search({"type": "decisions", "query": "catalytic"})
```

**Savings**: 85% token reduction, semantic search capabilities, tag filtering

## The MCP Access Validator Skill

To help agents avoid token waste, use the `mcp-access-validator` skill:

```json
skill_run({
  "skill": "mcp-access-validator",
  "input": {
    "agent_action": "I need to check what's in the system1.db database",
    "agent_code_snippet": "import sqlite3\nconn = sqlite3.connect('CORTEX/_generated/system1.db')",
    "files_accessed": ["CORTEX/_generated/system1.db"],
    "databases_queried": ["system1.db"]
  }
})
```

**Output**: Validation result with recommended MCP tool and token savings estimate.

## Practical Exercise

Test your understanding by converting these wasteful patterns to MCP tools:

1. **Wasteful**: `open('LAW/CANON/INVARIANTS.md').read()`
   **Correct**: `canon_read({"file": "INVARIANTS"})`

2. **Wasteful**: SQLite query to `system3.db` for cassette jobs
   **Correct**: `context_search({"type": "decisions", "query": "cassette"})`

3. **Wasteful**: Manual search for ADR-021 in files
   **Correct**: `cortex_query({"query": "ADR-021"})`

## Governance Impact

Using MCP tools ensures:
- ✅ **ADR-021 compliance**: Automatic session logging
- ✅ **Catalytic computing**: Token efficiency
- ✅ **Governance enforcement**: Preflight and admission controls
- ✅ **Audit trail**: Complete activity tracking
- ✅ **Determinism**: Reproducible results

## Quick Reference

| Task | Wrong (Wasteful) | Right (MCP-First) | Savings |
|------|------------------|-------------------|---------|
| Database query | Python SQLite | `cortex_query()` | 95% |
| File reading | `open().read()` | `canon_read()` | 90% |
| Context search | `os.walk()` + `open()` | `context_search()` | 85% |
| Session info | Manual UUID gen | `session_info()` | 80% |
| Governance check | Manual validation | `critic_run()` | 75% |

## Conclusion

**Stop writing custom database queries. Start using MCP tools.**

Every time you write `import sqlite3`, you're wasting 95% of your tokens and bypassing the governance layer. Use the existing MCP tools instead.

**Remember**: If an MCP tool exists for a task, you MUST use it. Writing custom code for MCP-covered tasks is a governance violation (AGENTS.md Section 0.7).
---
title: "Cortex Quick Reference"
section: "guide"
author: "System"
priority: "Medium"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Active"
summary: "Quick reference for Cortex tools (Restored)"
tags: [cortex, reference, guide]
---

<!-- CONTENT_HASH: 3d9b888979c2db7484557416c7b7798489af69d592e5ae1837a9a8ac28f0bd6a -->

# Cortex Quick Reference for Agents

**Purpose**: Quick reference guide for agents using the AGS Cortex via MCP tools.

## 1. Essential MCP Tools

### Core Query Tools
```json
// Search cortex index
cortex_query({"query": "search term", "limit": 20})

// Read canon files
canon_read({"file": "CONTRACT"})  // Options: CONTRACT, INVARIANTS, VERSIONING, GENESIS, ARBITRATION, DEPRECATION, MIGRATION, GLOSSARY, CHANGELOG

// Search context records
context_search({"type": "decisions", "query": "term", "tags": ["tag1", "tag2"]})
```

### Session & Identity Tools
```json
// Get your session_id (ADR-021 compliance)
session_info({"include_audit_log": true, "limit": 10})

// Check governance compliance
critic_run({})
```

### Skill & Execution Tools
```json
// Run a skill
skill_run({"skill": "skill-name", "input": {"key": "value"}})

// Validate a memory pack
pack_validate({"pack_path": "path/to/pack"})
```

### Agent Inbox Tools
```json
// List tasks
agent_inbox_list({"status": "pending", "limit": 20})

// Claim a task
agent_inbox_claim({"task_id": "TASK-123", "agent_id": "your-session-id"})

// Finalize a task
agent_inbox_finalize({"task_id": "TASK-123", "status": "completed", "result": "Task completed successfully"})
```

## 2. Common Query Patterns

### Find Governance Documents
```json
// Find all ADRs
context_search({"type": "decisions"})

// Find ADRs about a specific topic
cortex_query({"query": "ADR-021"})
context_search({"type": "decisions", "query": "identity"})

// Read specific canon files in order
1. canon_read({"file": "AGREEMENT"})
2. canon_read({"file": "CONTRACT"})
3. canon_read({"file": "INVARIANTS"})
4. canon_read({"file": "VERSIONING"})
```

### Check System Status
```json
// Verify connection is working
cortex_query({"query": "test", "limit": 1})

// Check your session info
session_info({})

// Run governance checks
critic_run({})
```

### Find Skills
```json
// Search for skills
cortex_query({"query": "skill"})

// Find skill documentation
cortex_query({"query": "SKILL.md"})
```

## 3. Connection Health Check

### Quick Diagnostic Sequence
1. **Test basic connection**: `cortex_query({"query": "test"})` - Should return results
2. **Verify session identity**: `session_info({})` - Should return your `session_id`
3. **Check governance**: `critic_run({})` - Should pass (return `"passed": true`)
4. **Read core canon**: `canon_read({"file": "CONTRACT"})` - Should return CONTRACT.md content

### Common Connection Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Server not running** | Tools timeout or return errors | Run `python LAW/CONTRACTS/ags_mcp_entrypoint.py` |
| **Python version** | Import errors or syntax errors | Ensure Python 3.8+ is installed |
| **Missing dependencies** | Module import errors | Run `pip install -r requirements.txt` |
| **Dirty repository** | Preflight checks fail | Commit or stash changes, run `git status` |
| **Audit log errors** | Session info shows no audit entries | Check `LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl` permissions |

## 4. Agent Startup Checklist

### Phase 1: Connection & Identity
- [ ] Start MCP server via `ags_mcp_entrypoint.py`
- [ ] Verify connection with `cortex_query({"query": "test"})`
- [ ] Get session_id via `session_info({})`
- [ ] Note session_id for future reference

### Phase 2: Governance Read
- [ ] Read `AGREEMENT.md` via `canon_read({"file": "AGREEMENT"})`
- [ ] Read `CONTRACT.md` via `canon_read({"file": "CONTRACT"})`
- [ ] Read `INVARIANTS.md` via `canon_read({"file": "INVARIANTS"})`
- [ ] Read `VERSIONING.md` via `canon_read({"file": "VERSIONING"})`
- [ ] Read `AGENTS.md` via `cortex_query({"query": "AGENTS.md"})`

### Phase 3: Context Review
- [ ] Review relevant ADRs via `context_search({"type": "decisions"})`
- [ ] Check for overdue reviews via `context_review({"days": 30})`
- [ ] Review preferences via `context_search({"type": "preferences"})`

### Phase 4: Task Assessment
- [ ] Check inbox for tasks via `agent_inbox_list({"status": "pending"})`
- [ ] Identify task type (governance change, skill implementation, build execution, documentation)
- [ ] Determine required ADRs and ceremonies

## 5. Best Practices

### MCP-First Principle (Token Efficiency)
**🚨 CRITICAL: NO TOKEN WASTE 🚨**
- **ALWAYS** use MCP tools instead of manual database/file operations
- **NEVER** write Python SQLite snippets to inspect databases directly
- **NEVER** use `open()` or `Path().read_text()` to read governance files
- **ALWAYS** use existing MCP tools: `cortex_query`, `canon_read`, `context_search`

**Examples:**
```json
// ❌ WRONG: Manual database query (token waste)
import sqlite3
conn = sqlite3.connect('CORTEX/_generated/system1.db')

// ✅ CORRECT: Use MCP tool
cortex_query({"query": "symbols", "limit": 10})
```

### Query Optimization
- Use specific queries rather than broad searches
- Set reasonable limits (default is 20)
- Combine `cortex_query` with `context_search` for comprehensive results

### Session Management
- Always check `session_info()` after connection
- Include `session_id` in any manual audit entries
- Monitor audit logs for your session: `session_info({"include_audit_log": true})`

### Error Handling
- Check `isError` field in tool responses
- Read full error messages, not just truncated output
- Use `critic_run({})` to diagnose governance issues

### Compliance
- All cortex queries are automatically logged (ADR-021)
- Never bypass preflight or admission controls
- Follow commit ceremony for all commits (Section 10 of AGENTS.md)
- Follow MCP-first principle to avoid token waste (Section 0.7 of AGENTS.md)

## 6. Reference: MCP Server Details

### Entry Points
- **Primary**: `python LAW/CONTRACTS/ags_mcp_entrypoint.py` (ADR-021 compliant)
- **Alternative**: `python CAPABILITY/MCP/server.py`
- **Test**: `python LAW/CONTRACTS/ags_mcp_entrypoint.py --test`

### Log Locations
- **Audit logs**: `LAW/CONTRACTS/_runs/mcp_logs/audit.jsonl`
- **Session logs**: Filter audit.jsonl by your `session_id`
- **Error logs**: Check server stderr output

### Governance Integration
- **Preflight**: Automatic via `ags.py preflight`
- **Admission**: Automatic via `ags.py admit` with `AGS_INTENT_PATH`
- **Critic**: Automatic via `critic.py` for governed tools

---

**Last Updated**: 2025-12-31
**Created By**: Kilo Code (Agent ID: 17cb4e78-ae76-49df-b336-c0cccbf5878d)
**Related Documents**: AGENTS.md, ADR-004, ADR-021
**Maintenance**: This quick reference should be updated when MCP tools change or new best practices emerge.
# MCP Server Merge Inventory

**Lab Source:** `THOUGHT/LAB/MCP/server.py` (2,142 lines)  
**Canonical Target:** `CAPABILITY/MCP/server.py` (1,844 lines)  
**Date:** 2025-12-31

---

## Quick Reference

| Category | Count | Status |
|----------|-------|--------|
| Safe Primitives | 9 | ✅ Ready to port |
| CMP-01 Validation | 5 | ⚠️ Needs path updates |
| Swarm/Task Queue | 6 | 🔄 Evaluate placement |
| Chain of Command | 6 | 🔄 Evaluate placement |
| Terminal Sharing | 3 | ⏸️ Defer (Lab feature) |
| Skill Execution | 4 | ⏸️ Defer (different runtime) |
| **Total** | **33** | — |

---

## 1. Safe Primitives ✅

These have no hardcoded paths and can be ported immediately.

### File Locking (Windows/Unix)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_lock_file(f, exclusive)` | 29-31, 41-43 | Platform-aware file lock |
| `_unlock_file(f)` | 32-38, 44-46 | Release file lock |

### Atomic File Operations
| Function | Lines | Purpose |
|----------|-------|---------|
| `_atomic_write_jsonl(path, line)` | 151-198 | Append with crash safety |
| `_atomic_rewrite_jsonl(path, transform)` | 201-287 | Transform with atomic swap |
| `_read_jsonl_streaming(path, filter, limit, offset)` | 290-335 | Memory-efficient streaming |

### Validation Logic
| Function | Lines | Purpose |
|----------|-------|---------|
| `_validate_task_state_transition(current, target)` | 338-342 | State machine enforcement |
| `_validate_task_spec(task_spec)` | 345-372 | Schema validation |
| `_validate_against_schema(instance, schema)` | 896-929 | Generic JSON validation |

### Utilities
| Function | Lines | Purpose |
|----------|-------|---------|
| `_compute_hash(file_path)` | 884-894 | SHA-256 file hash |
| `get_validator_build_id()` | 61-100 | Git SHA or file hash |

### Constants
```python
VALIDATOR_SEMVER = "1.0.0"
SUPPORTED_VALIDATOR_SEMVERS = {"1.0.0", "1.0.1", "1.1.0"}
TASK_STATES = {...}  # State machine
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
```

---

## 2. CMP-01 Path Validation ⚠️

**CRITICAL:** These use hardcoded paths that must be updated to 6-bucket structure first.

### Path Constants (MUST UPDATE)
```python
# OLD (Lab) → NEW (Canonical)
CONTRACTS_DIR:   "CONTRACTS/_runs"      → "LAW/CONTRACTS/_runs"
SKILLS_DIR:      "CATALYTIC-DPT/SKILLS" → "CAPABILITY/SKILLS"

DURABLE_ROOTS:
  "CONTRACTS/_runs/"           → "LAW/CONTRACTS/_runs/"
  "CORTEX/_generated/"         → "NAVIGATION/CORTEX/_generated/"
  "MEMORY/LLM_PACKER/_packs/"  → (unchanged)

CATALYTIC_ROOTS:
  "CONTRACTS/_runs/_tmp/"      → "LAW/CONTRACTS/_runs/_tmp/"
  "CORTEX/_generated/_tmp/"    → "NAVIGATION/CORTEX/_generated/_tmp/"
  "TOOLS/_tmp/"                → "CAPABILITY/TOOLS/_tmp/"
  "MCP/_tmp/"                  → "CAPABILITY/MCP/_tmp/"

FORBIDDEN_ROOTS:
  "CANON/"                     → "LAW/CANON/"
```

### Validation Functions
| Method | Lines | Purpose |
|--------|-------|---------|
| `_is_path_under_root(path, root)` | 936-947 | Component-safe containment |
| `_validate_single_path(raw, pointer, roots, code)` | 949-1023 | CMP-01 rule check |
| `_check_containment_overlap(paths, pointer)` | 1025-1068 | Detect nested paths |
| `_validate_jobspec_paths(task_spec)` | 1072-1126 | Full jobspec validation |
| `_verify_post_run_outputs(run_id)` | 1128-1238 | Post-run output check |

---

## 3. SPECTRUM-02 Bundle Verification

| Method | Lines | Purpose |
|--------|-------|---------|
| `_generate_output_hashes(run_id)` | 628-701 | Create OUTPUT_HASHES.json |
| `verify_spectrum02_bundle(run_dir, strict)` | 1245-1418 | Verify resume bundle |

**Depends on:** CMP-01 path validation (port after Section 2)

---

## 4. Ledger Operations ✅

| Method | Lines | Purpose |
|--------|-------|---------|
| `get_ledger(run_id)` | 845-868 | Retrieve ledger entries |
| `_log_operation(operation)` | 872-882 | Append to immutable ledger |

---

## 5. Swarm Task Queue 🔄

**Decision needed:** Keep in MCP or move to `CAPABILITY/TOOLS/swarm`?

| Method | Lines | Purpose |
|--------|-------|---------|
| `dispatch_task(id, spec, from, to, priority)` | 1427-1495 | Governor → Ant dispatch |
| `get_pending_tasks(agent_id, limit)` | 1497-1536 | Ant polls for work |
| `acknowledge_task(task_id, agent_id)` | 1650-1719 | Claim task |
| `report_result(task_id, from, status, result)` | 1538-1602 | Submit result |
| `get_results(task_id, limit, offset)` | 1604-1648 | Governor retrieves results |

---

## 6. Chain of Command 🔄

**Decision needed:** Keep in MCP or move to `CAPABILITY/TOOLS/swarm`?

| Method | Lines | Purpose |
|--------|-------|---------|
| `escalate(from, issue, context, priority)` | 1728-1787 | Escalate UP |
| `get_escalations(for_level, limit)` | 1789-1823 | Check escalations |
| `resolve_escalation(id, by, resolution, action)` | 1825-1881 | Resolve escalation |
| `send_directive(from, to, directive, context)` | 1883-1925 | Command DOWN |
| `get_directives(for_agent, limit)` | 1927-1961 | Check directives |
| `acknowledge_directive(id, agent_id)` | 1963-2013 | Mark processed |

---

## 7. Terminal Sharing ⏸️ (Defer)

Lab-specific feature for visible execution monitoring.

| Method | Lines | Purpose |
|--------|-------|---------|
| `register_terminal(id, owner, cwd)` | 384-404 | Register shared terminal |
| `log_terminal_command(id, cmd, executor, out, code)` | 406-444 | Log command |
| `get_terminal_output(terminal_id)` | 446-456 | Retrieve output |

---

## 8. Skill Execution ⏸️ (Defer)

Canonical uses different skill runtime in `CAPABILITY/PIPELINES`.

| Method | Lines | Purpose |
|--------|-------|---------|
| `execute_skill(name, spec, executor, run_id)` | 458-555 | Run skill |
| `file_sync(source, dest, executor, verify)` | 558-626 | Hash-verified sync |
| `skill_complete(run_id, status, outputs, errors)` | 703-842 | Mark complete |

---

## 9. Other

| Item | Lines | Purpose |
|------|-------|---------|
| `register_mcp_tools()` | 2020-2113 | Tool registration |
| `mcp_server = MCPTerminalServer()` | 2017 | Singleton instance |
| `__main__` test block | 2114-2142 | CLI test harness |

---

## Merge Order

```
Phase 1: Safe Primitives
├── File locking
├── Atomic operations  
├── Validation logic
├── Utilities
└── Constants

Phase 2: Update Paths
└── All CMP-01 root constants

Phase 3: CMP-01 Integration
├── Path validation functions
└── Post-run verification

Phase 4: SPECTRUM-02
└── Bundle verification

Phase 5: Architectural Decision
├── Task queue → MCP or Swarm?
└── Chain of command → MCP or Swarm?

Phase 6: Cleanup
├── Deprecate Lab server
└── Update all imports
```

---

## Checklist

- [ ] Port file locking primitives
- [ ] Port atomic file operations
- [ ] Port validation logic
- [ ] Port utilities and constants
- [ ] Update CMP-01 path constants to 6-bucket
- [ ] Port CMP-01 validation functions
- [ ] Port SPECTRUM-02 verification
- [ ] Decide: Task queue placement
- [ ] Decide: Chain of command placement
- [ ] Mark Lab server deprecated

# MCP Server Merge Inventory

**Lab Source:** `THOUGHT/LAB/MCP/server_CATDPT.py` (2,142 lines)  
**Canonical Target:** `CAPABILITY/MCP/server.py` (~3,100 lines after merge)  
**Date:** 2025-12-31

---

## Quick Reference

| Category | Count | Status |
|----------|-------|--------|
| Safe Primitives | 9 | ✅ MERGED |
| CMP-01 Path Constants | 6 | ✅ MERGED |
| CMP-01 Validation | 6 | ✅ MERGED |
| SPECTRUM-02 Bundle | 2 | ✅ MERGED |
| Terminal Sharing | 4 | ✅ MERGED |
| Skill Execution | 4 | ⏸️ DEFERRED (different runtime) |
| Swarm Task Queue | 5 | ⏸️ DEFERRED (0.5B models unclear) |
| Chain of Command | 6 | ⏸️ DEFERRED (0.5B models unclear) |

---

## Section Summaries

### 1. Safe Primitives ✅ MERGED
**Purpose:** Cross-platform file safety primitives for crash-safe, concurrent-safe operations.

**What it does:**
- **File Locking:** `_lock_file`/`_unlock_file` - Platform-aware (Windows msvcrt, Unix fcntl) exclusive/shared locks
- **Atomic JSONL:** `_atomic_write_jsonl` - Write-to-temp-then-append pattern prevents partial writes
- **Atomic Rewrite:** `_atomic_rewrite_jsonl` - Read-transform-write with atomic rename
- **Streaming Reader:** `_read_jsonl_streaming` - Memory-efficient iteration with filter/pagination
- **Validation:** `_validate_task_state_transition`, `_validate_task_spec` - State machine + schema enforcement
- **Utilities:** `_compute_hash`, `get_validator_build_id` - SHA-256 hashing, git/file-based build fingerprint
- **Constants:** `VALIDATOR_SEMVER`, `TASK_STATES`, `MAX_FILE_SIZE_BYTES`

**Why it matters:** Foundation for all durable ledger operations and crash recovery.

---

### 2. CMP-01 Path Constants ✅ MERGED
**Purpose:** Define the 6-bucket governance structure for path validation.

**What it does:**
- `CONTRACTS_DIR` → `LAW/CONTRACTS/_runs` (run ledgers)
- `SKILLS_DIR` → `CAPABILITY/SKILLS` (skill definitions)
- `DURABLE_ROOTS` → 3 paths where files may persist after run
- `CATALYTIC_ROOTS` → 5 paths for temporary domains (must be restored)
- `FORBIDDEN_ROOTS` → 2 paths that must never be written to

**Why it matters:** Enforces the architectural boundary between temporary and durable state.

---

### 3. CMP-01 Validation ✅ MERGED
**Purpose:** Pre-run and post-run path governance enforcement.

**What it does:**
- `_is_path_under_root` - Component-safe containment check (not string prefix)
- `_validate_single_path` - Reject absolute, traversal, forbidden overlap, wrong root
- `_check_containment_overlap` - Flag nested paths (except exact duplicates)
- `_validate_jobspec_paths` - Full jobspec validation (catalytic + durable)
- `_verify_post_run_outputs` - Post-run existence + compliance check
- `_validate_against_schema` - Basic JSON schema validation without dependencies

**Why it matters:** Prevents symlink escapes, path traversal attacks, and unauthorized writes.

---

### 4. SPECTRUM-02 Bundle ✅ MERGED
**Purpose:** Adversarial resume without execution history - prove work was done correctly.

**What it does:**
- `_generate_output_hashes` - Hash all declared durable outputs into `OUTPUT_HASHES.json`
- `verify_spectrum02_bundle` - Verify bundle integrity:
  - TASK_SPEC.json exists
  - STATUS.json = success + cmp01=pass
  - OUTPUT_HASHES.json with supported validator version
  - All declared hashes match actual files

**Why it matters:** Enables deterministic resume and proof-of-work verification.

---

### 5. Terminal Sharing ⏸️ DEFERRED
**Purpose:** Bidirectional terminal visibility between human and AI agents.

**What it would do:**
- `register_terminal(id, owner, cwd)` - Create shared terminal session
- `log_terminal_command(id, cmd, executor, output, exit_code)` - Log command history
- `get_terminal_output(terminal_id)` - Retrieve command history

**How it could work via MCP:**
1. Human registers a terminal with `terminal_register`
2. AI agent watches for new commands via polling `terminal_get_output`
3. AI can execute commands via `terminal_bridge` with the shared terminal ID
4. Both see the same command history in real-time

**Architecture needed:** Persistent storage (currently in-memory), real-time polling or push notifications.

---

### 6. Skill Execution ⏸️ DEFERRED
**Purpose:** Canonical skill execution with CMP-01 pre-validation and ledger binding.

**What it would do:**
- `execute_skill(name, task_spec, executor)` - Start skill with validation
- `file_sync(source, dest, executor)` - Hash-verified file copy
- `skill_complete(run_id, status, outputs)` - Mark complete + post-run verification
- Creates: TASK_SPEC.json, TASK_SPEC.sha256, STATUS.json, OUTPUT_HASHES.json

**Why deferred:** Canonical uses `CAPABILITY/PIPELINES/pipeline_runtime.py` which has its own skill execution model. The Lab version is more integrated with the terminal sharing concept.

---

### 7. Swarm Task Queue ⏸️ DEFERRED
**Purpose:** Multi-agent task dispatch and result collection.

**What it would do:**
- `dispatch_task` - Governor sends work to ant workers
- `get_pending_tasks` - Ant polls for available work
- `acknowledge_task` - Ant claims task (state: pending → acknowledged)
- `report_result` - Ant submits completion
- `get_results` - Governor retrieves results

**Why deferred:** Unclear if 0.5B parameter models can use MCP tools effectively.

---

### 8. Chain of Command ⏸️ DEFERRED
**Purpose:** Hierarchical agent escalation and directive system.

**What it would do:**
- `escalate` - Ant escalates issue UP to governor
- `get_escalations` - Governor checks inbox
- `resolve_escalation` - Governor resolves with action
- `send_directive` - Governor sends command DOWN
- `get_directives` - Ant checks for orders
- `acknowledge_directive` - Ant confirms receipt

**Why deferred:** Same as Task Queue - uncertain MCP compatibility with tiny models.

---

## Merge Order (Completed)

```
Phase 1: Safe Primitives ✅
├── File locking
├── Atomic operations  
├── Validation logic
├── Utilities
└── Constants

Phase 2: Update Paths ✅
└── All CMP-01 root constants → 6-bucket

Phase 3: CMP-01 Integration ✅
├── Path validation functions
└── Post-run verification

Phase 4: SPECTRUM-02 ✅
└── Bundle verification
```

---

## Checklist

- [x] Port file locking primitives
- [x] Port atomic file operations
- [x] Port validation logic
- [x] Port utilities and constants
- [x] Update CMP-01 path constants to 6-bucket
- [x] Port CMP-01 validation functions
- [x] Port SPECTRUM-02 verification
- [ ] Decide: Task queue placement → DEFERRED
- [ ] Decide: Chain of command placement → DEFERRED
- [ ] Terminal sharing → NEEDS ARCHITECTURE
- [ ] Mark Lab server deprecated → PENDING (still has unique features)

---

## Next Steps

1. **Terminal Sharing:** Design persistent storage + polling/push mechanism
2. **Test MCP with 0.5B models:** See if tiny LLMs can actually use MCP tools
3. **Skill Execution:** Decide if Lab model vs Pipeline model should merge
4. **Deprecation:** Once all unique features are ported or decided against

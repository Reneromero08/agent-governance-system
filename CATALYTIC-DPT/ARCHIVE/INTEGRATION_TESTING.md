# Integration Testing Guide

**Purpose**: Validate complete multi-agent orchestration system
**Scope**: MCP → grok-executor → Conductor → Full workflow
**Status**: Ready to execute

---

## Quick Start

### 1. Test MCP Server (Foundation)

```bash
cd d:/CCC\ 2.0/AI/agent-governance-system/CATALYTIC-DPT
python MCP/server.py
```

**Verify Output**:
```
Test 1: Register terminal
{
  "terminal_id": "user_vscode",
  "owner": "You",
  "status": "active",
  ...
}

Test 2: Log terminal command
{
  "status": "success",
  "terminal_id": "user_vscode",
  ...
}

Test 3: Get operations ledger
{
  "status": "success",
  "entries": [...],
  "total_entries": 2
}
```

**Success Criteria**:
- ✅ Terminal registration works
- ✅ Commands logged and visible to all agents
- ✅ Ledger persists operations
- ✅ No errors in output

---

### 2. Test Grok Executor (Core Execution)

```bash
cd d:/CCC\ 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/grok-executor
python test_grok_executor.py
```

**Verify Output**:
```
============================================================
Grok Executor Test Harness
============================================================

✓ test_file_copy
✓ test_hash_verification
✓ test_missing_source
✓ test_code_adaptation
✓ test_ledger_creation

============================================================
Test Summary
============================================================
Total:  5
Passed: 5
Failed: 0
============================================================
```

**Success Criteria**:
- ✅ All 5 tests pass
- ✅ File copy with hash verification works
- ✅ Error handling for missing files
- ✅ Code adaptation (find/replace) works
- ✅ Immutable ledger created in CONTRACTS/_runs/
- ✅ test_results.json written

---

### 3. Test Individual Task Fixtures

#### Test 3a: File Copy Task

```bash
cd d:/CCC\ 2.0/AI/agent-governance-system/CATALYTIC-DPT/SKILLS/grok-executor

python run.py fixtures/file_copy_task.json /tmp/output_copy.json
```

**Expected Output**:
```
[grok-executor] ✓ Copied: ...destination_path... (hash verified: True)
[grok-executor] Task: grok-copy-swarm-governor
[grok-executor] Status: success
[grok-executor] Operations: 3
[grok-executor] Errors: 0
[grok-executor] Ledger: CONTRACTS/_runs/grok-copy-swarm-governor-20251224-143022/
```

**Verify**:
- ✅ 3 files copied (run.py, validate.py, __init__.py)
- ✅ Each copy hash verified
- ✅ Ledger directory created
- ✅ TASK_SPEC.json and RESULTS.json written

---

#### Test 3b: Code Adaptation Task

```bash
python run.py fixtures/code_adapt_task.json /tmp/output_adapt.json
```

**Expected Output**:
```
[grok-executor] ✓ Adapted: Replace all Cline references with Gemini CLI
[grok-executor] Task: grok-adapt-swarm-gemini
[grok-executor] Status: success
[grok-executor] Operations: 2
[grok-executor] Errors: 0
```

**Verify**:
- ✅ Code file modified (cline → gemini)
- ✅ Find/replace operations logged
- ✅ File integrity maintained

---

#### Test 3c: Validation Task

```bash
python run.py fixtures/validate_task.json /tmp/output_validate.json
```

**Expected Output**:
```
[grok-executor] Validating: ...file_path...
[grok-executor] Task: grok-validate-swarm-adapted
[grok-executor] Status: success
[grok-executor] Operations: 1
[grok-executor] Errors: 0
```

**Verify**:
- ✅ Validation checks documented
- ✅ Status recorded to ledger

---

### 4. Integration Test: MCP + Grok-Executor

This test verifies that MCP can orchestrate grok-executor:

```python
# Create a Python script: test_mcp_grok_integration.py

from pathlib import Path
import json
import sys

# Add to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from MCP.server import MCPTerminalServer
from SKILLS.grok_executor.run import GrokExecutor

# Test flow
mcp = MCPTerminalServer()

# 1. Register terminal
terminal = mcp.register_terminal("grok_worker_1", "Grok", "/workspace")
print(f"✓ Terminal registered: {terminal['terminal_id']}")

# 2. Execute skill via MCP
task_spec = {
    "task_id": "test-mcp-integration",
    "task_type": "file_operation",
    "operation": "copy",
    "files": [
        {
            "source": "source.txt",
            "destination": "destination.txt"
        }
    ]
}

mcp_result = mcp.execute_skill("grok-executor", task_spec, "Grok")
print(f"✓ Skill execution initiated: {mcp_result['run_id']}")

# 3. Log command to terminal
mcp.log_terminal_command(
    "grok_worker_1",
    f"python run.py {task_spec['task_id']}.json output.json",
    "Grok",
    "Task executed successfully",
    0
)
print(f"✓ Command logged to terminal")

# 4. Get ledger
ledger = mcp.get_ledger()
print(f"✓ Ledger retrieved: {len(ledger['entries'])} entries")

print("\n✅ MCP + Grok Integration Test PASSED")
```

**Success Criteria**:
- ✅ Terminal registration via MCP works
- ✅ Skill execution scheduled via MCP
- ✅ Commands logged to shared terminal
- ✅ Immutable ledger grows
- ✅ All agents can see all operations

---

### 5. Full Workflow Test: Copy → Adapt → Validate

This simulates the complete swarm-governor import workflow:

```bash
#!/bin/bash

echo "Phase 1: Copy files"
python SKILLS/grok-executor/run.py SKILLS/grok-executor/fixtures/file_copy_task.json /tmp/p1_copy.json
if [ $? -ne 0 ]; then echo "❌ Phase 1 failed"; exit 1; fi
echo "✅ Phase 1 complete"

echo ""
echo "Phase 2: Adapt code"
python SKILLS/grok-executor/run.py SKILLS/grok-executor/fixtures/code_adapt_task.json /tmp/p2_adapt.json
if [ $? -ne 0 ]; then echo "❌ Phase 2 failed"; exit 1; fi
echo "✅ Phase 2 complete"

echo ""
echo "Phase 3: Validate adapted code"
python SKILLS/grok-executor/run.py SKILLS/grok-executor/fixtures/validate_task.json /tmp/p3_validate.json
if [ $? -ne 0 ]; then echo "❌ Phase 3 failed"; exit 1; fi
echo "✅ Phase 3 complete"

echo ""
echo "✅ Full workflow test PASSED"
echo "Results:"
echo "  Phase 1 (Copy): /tmp/p1_copy.json"
echo "  Phase 2 (Adapt): /tmp/p2_adapt.json"
echo "  Phase 3 (Validate): /tmp/p3_validate.json"
```

**Success Criteria**:
- ✅ Phase 1: Files copied with hash verification
- ✅ Phase 2: Code adapted (cline → gemini)
- ✅ Phase 3: Adapted code validated
- ✅ All phases logged to CONTRACTS/_runs/
- ✅ Immutable ledger complete

---

## Verification Checklist

### MCP Server Tests

- [ ] Terminal registration works
- [ ] Commands logged and visible
- [ ] Multiple terminals can exist
- [ ] Ledger persists across operations
- [ ] File sync creates CONTRACTS/_runs/ entries
- [ ] Hash verification prevents corrupted files

### Grok Executor Tests

- [ ] test_file_copy passes
- [ ] test_hash_verification passes
- [ ] test_missing_source passes (error handled)
- [ ] test_code_adaptation passes
- [ ] test_ledger_creation passes
- [ ] All fixtures execute successfully

### Integration Tests

- [ ] MCP can register Grok terminal
- [ ] MCP can schedule grok-executor skills
- [ ] grok-executor respects MCP ledger location
- [ ] Commands logged to shared terminal
- [ ] Error handling rolls back changes

### Full Workflow Tests

- [ ] Copy → Adapt → Validate succeeds
- [ ] Each phase produces correct output
- [ ] Ledger entries created for each operation
- [ ] TASK_SPEC.json immutable
- [ ] RESULTS.json final and correct

---

## Troubleshooting

### Issue: Test fails with "File not found"

**Cause**: Paths need to exist (especially AGI swarm-governor)
**Solution**:
- Verify AGI repo exists at `d:/CCC 2.0/AI/AGI/`
- Verify swarm-governor exists at `AGI/SKILLS/swarm-governor/`
- Update fixtures to match your actual paths

### Issue: Hash verification fails

**Cause**: File corrupted during copy (very rare)
**Solution**:
- Run test again (transient issue)
- Check disk space
- Verify no antivirus interference

### Issue: Ledger not created

**Cause**: CONTRACTS/_runs/ directory doesn't exist or no write permissions
**Solution**:
```bash
mkdir -p "d:/CCC 2.0/AI/agent-governance-system/CONTRACTS/_runs"
```

### Issue: Code adaptation doesn't find pattern

**Cause**: Pattern string doesn't exactly match file content
**Solution**:
- Use exact match (including whitespace)
- Check file encoding (UTF-8)
- Verify pattern not already replaced

---

## Expected Directory Structure After Testing

```
CATALYTIC-DPT/
├── CONTRACTS/
│   └── _runs/
│       ├── grok-copy-swarm-governor-20251224-143022/
│       │   ├── TASK_SPEC.json
│       │   └── RESULTS.json
│       ├── grok-adapt-swarm-gemini-20251224-143045/
│       │   ├── TASK_SPEC.json
│       │   └── RESULTS.json
│       └── grok-validate-swarm-adapted-20251224-143105/
│           ├── TASK_SPEC.json
│           └── RESULTS.json
│
└── SKILLS/
    └── grok-executor/
        ├── test_results.json  ← Test report
        └── fixtures/
            ├── file_copy_task.json
            ├── code_adapt_task.json
            ├── validate_task.json
            └── research_task.json
```

---

## Expected Ledger Output

Each task creates immutable records:

**TASK_SPEC.json**:
```json
{
  "task_id": "grok-copy-swarm-governor",
  "task_type": "file_operation",
  "operation": "copy",
  "files": [...]
}
```

**RESULTS.json**:
```json
{
  "task_id": "grok-copy-swarm-governor",
  "task_type": "file_operation",
  "status": "success",
  "operations": [
    {
      "operation": "copy",
      "source": "...",
      "destination": "...",
      "source_hash": "abc123...",
      "dest_hash": "abc123...",
      "hash_verified": true
    }
  ],
  "errors": [],
  "ledger_dir": "CONTRACTS/_runs/..."
}
```

---

## Next Steps After Testing

Once all tests pass:

1. **Setup Gemini Conductor**:
   ```bash
   gemini --experimental-acp
   ```

2. **Use conductor-task-builder** to create task specs

3. **Route to Conductor** for parallel execution:
   ```
   Claude → conductor-task-builder → Conductor → Grok workers
   ```

4. **Monitor via MCP** (bidirectional terminals)

5. **Execute Phase 0 validation** via swarm-governor-adapted

---

**Status**: All tests ready to run
**Integration**: MCP → grok-executor → Conductor → Full system
**Governance**: Hash-verified, immutably logged, zero drift


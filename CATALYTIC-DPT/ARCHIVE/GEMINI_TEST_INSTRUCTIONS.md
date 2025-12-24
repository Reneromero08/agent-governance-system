# Gemini Conductor Test Instructions

**Goal**: Hand off a task to Gemini Conductor to test the multi-agent orchestration system

---

## What We're Testing

The grok-executor is ready and tested. Now we need to test:
1. Gemini Conductor receiving a task
2. Conductor distributing to Grok workers
3. Grok executing via grok-executor
4. Results flowing back through MCP

---

## Task to Hand Off

**Task**: Copy CATALYTIC-DPT documentation files with hash verification

**Files**:
- CATALYTIC-DPT/README.md → test output
- CATALYTIC-DPT/SKILLS/README.md → test output
- CATALYTIC-DPT/ORCHESTRATION_ARCHITECTURE.md → test output

**Already verified**: This works directly with grok-executor (3/3 files copied, all hashes verified)

---

## Instructions for User

### Step 1: Open YOUR Terminal (not Claude's)

```bash
cd "d:/CCC 2.0/AI/agent-governance-system"
```

### Step 2: Start Gemini Conductor

```bash
gemini --experimental-acp
```

### Step 3: Give Gemini This Task

```
I have a task specification file at:
CATALYTIC-DPT/grok-task-copy-readmes.json

This is a file operation task to copy 3 documentation files with SHA-256 hash verification.

Please:
1. Read the task spec from that file
2. Execute it using the grok-executor skill at CATALYTIC-DPT/SKILLS/grok-executor/run.py
3. Verify the results are logged to CONTRACTS/_runs/
4. Report back with the task status and hash verification results
```

---

## What Gemini Should Do

1. **Read** `CATALYTIC-DPT/grok-task-copy-readmes.json`
2. **Execute** via: `python CATALYTIC-DPT/SKILLS/grok-executor/run.py <input> <output>`
3. **Verify** hash verification succeeded
4. **Report** results from `CONTRACTS/_runs/conductor-test-copy-readmes-*/RESULTS.json`

---

## Expected Output

Gemini should report:
- ✅ 3 files copied
- ✅ All hashes verified (source_hash == dest_hash)
- ✅ Ledger created at `CONTRACTS/_runs/conductor-test-copy-readmes-<timestamp>/`
- ✅ Status: success
- ✅ Errors: 0

---

## Verification After Gemini Completes

### Check files were copied
```bash
ls -lh "CONTRACTS/_runs/conductor-test-readmes/"
```

Should see:
- CATALYTIC-README.md (5KB)
- SKILLS-README.md (9KB)
- ORCHESTRATION.md (16KB)

### Check ledger
```bash
cat CONTRACTS/_runs/conductor-test-copy-readmes-*/RESULTS.json | python -m json.tool
```

Should show:
- `"status": "success"`
- `"operations": [3 items with hash_verified: true]`
- `"errors": []`

### Verify hash integrity
All operations should show:
```json
{
  "source_hash": "abc123...",
  "dest_hash": "abc123...",
  "hash_verified": true
}
```

---

## If This Works

This proves:
- ✅ Gemini Conductor can receive tasks
- ✅ Grok executor works via Conductor
- ✅ Hash verification prevents corruption
- ✅ Immutable ledger logs everything
- ✅ Multi-agent orchestration functional

**Next**: Scale to parallel validation of Phase 0 schemas

---

## If This Fails

**Check**:
1. Is `gemini --experimental-acp` available? (Check with `gemini --version`)
2. Can Gemini read the task file? (Try direct path)
3. Can Gemini execute Python scripts? (Test with simple script)
4. Does MCP need explicit integration? (May need MCP server running)

---

## Alternative: Direct Grok Test (Already Verified)

If Conductor doesn't work yet, you can verify grok-executor directly:

```bash
cd "d:/CCC 2.0/AI/agent-governance-system/CATALYTIC-DPT"
python SKILLS/grok-executor/run.py grok-task-copy-readmes.json output.json
```

**This already works** - we tested it and got 3/3 files copied with hash verification.

---

**Status**: Grok-executor ready. Awaiting Gemini Conductor test.


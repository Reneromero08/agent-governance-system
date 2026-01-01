---
title: "CLI Restore Commit Plan"
section: "report"
author: "System"
priority: "Medium"
created: "2025-12-29 00:00"
modified: "2025-12-29 00:00"
status: "Complete"
summary: "Commit plan for CLI restoration and execute command addition"
tags: [cli, restore, planning]
---
<!-- CONTENT_HASH: 52583202ec7d83fb9e43ae8f37db7b9297b9076258eb806cc76f7c7cd98816dc -->

# Commit Plan - CLI Restoration and Execute Command Addition

**Task**: Restore catalytic_chat/cli.py and add execute command
**Date**: 2025-12-29
**Status**: Complete

---

## Changes Made

### File Modified
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py` (703 lines)

### Summary

Restored CLI from backup and added `execute` command for Phase 4.1 step execution.

### Detailed Changes

1. **Restored existing commands** (from cli.py.backup)
   - `build` - Build section index
   - `verify` - Verify index determinism
   - `get` - Get section by ID with optional slice
   - `extract` - Extract sections from file
   - `symbols add/get/list/verify` - Symbol registry commands
   - `resolve` - Resolve symbol to content with caching
   - `cassette verify/post/claim/complete` - Message cassette commands
   - `plan request/verify` - Deterministic planner commands

2. **Added new `execute` command**
   - Command: `python -m catalytic_chat.cli execute --run-id <run> --job-id <job>`
   - Opens cassette for the given run_id
   - Iterates PENDING steps for the given job_id in ordinal order
   - For each step:
     - Claims the step (claim_step)
     - Executes the step (execute_step)
     - Prints status: `[OK] step_id` or `[FAIL] step_id: <reason>`
   - Stops execution immediately on first failure
   - Exits non-zero on failure, zero on success

3. **Fixed cmd_build implementation**
   - Previously referenced undefined `file_path` variable
   - Now uses `build_index()` function correctly with repo_root and substrate_mode

4. **Added imports**
   - `from catalytic_chat.planner import Planner, PlannerError, post_request_and_plan`

### Verification

All tests pass:
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q
# 39 passed in 4.43s
```

CLI commands verified:
```bash
python -m catalytic_chat.cli --help  # Shows all 8 top-level commands
python -m catalytic_chat.cli plan request --dry-run --request-file tests/fixtures/plan_request_min.json
python -m catalytic_chat.cli cassette verify --run-id test_plan_001
```

### Hard Constraints Followed

- ✓ No modification to execution logic (message_cassette.py)
- ✓ No modification to planner logic (planner.py)
- ✓ No modification to Phase 1-4 schemas or invariants
- ✓ Only restored CLI and added missing command
- ✓ Used argparse, consistent with existing CLI style
- ✓ execute command is top-level (not nested under cassette or plan)
- ✓ Handler is minimal: CLI orchestrates, logic lives elsewhere

### Git Status

```
M THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py
```

### Commit Message Draft

```
Restore catalytic_chat/cli.py and add execute command

- Restored CLI from cli.py.backup (all existing commands functional)
- Added new `execute` command for Phase 4.1 step execution
  - Accepts --run-id and --job-id arguments
  - Iterates PENDING steps in ordinal order
  - Claims and executes each step via existing message_cassette methods
  - Prints [OK] or [FAIL] status per step
  - Stops on first failure, exits non-zero on error
- Fixed cmd_build to use build_index() function correctly
- Added planner imports for plan command support
- All 39 tests pass
```

### Files Staged for Commit

- THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py

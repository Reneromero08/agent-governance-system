# Phase 2.4.1C.4 CLI Tools Write Surface Enforcement

**Target Model**: GLM4.7 (or any sonnet-class model)
**Task**: Mechanically enforce write firewall on 6 CLI tools
**Expected Duration**: Single session

---

## Objective

Integrate `GuardedWriter` into 6 CLI tools to enforce write firewall policy. All raw write operations (`write_text`, `write_bytes`, `mkdir`, `rename`) must route through GuardedWriter.

---

## Scope (6 Files)

From `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1A_WRITE_SURFACE_MAP.md` section 2.3:

1. **CAPABILITY/TOOLS/ags.py** — PARTIAL (has `_atomic_write_bytes` but no firewall)
2. **CAPABILITY/TOOLS/cortex/cortex.py** — UNGUARDED
3. **CAPABILITY/TOOLS/cortex/codebook_build.py** — UNGUARDED
4. **CAPABILITY/TOOLS/utilities/emergency.py** — UNGUARDED
5. **CAPABILITY/TOOLS/utilities/ci_local_gate.py** — UNGUARDED
6. **CAPABILITY/TOOLS/utilities/intent.py** — UNGUARDED

---

## Pattern to Follow

Use the **exact same pattern** from Phase 2.4.1C.2 (PIPELINES + MCP):

### 1. Add GuardedWriter import
```python
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
```

### 2. Initialize writer in main/CLI entry point
```python
writer = GuardedWriter(
    project_root=REPO_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs", "INBOX", "NAVIGATION/PROOFS"]
)
writer.open_commit_gate()
```

### 3. Replace raw write operations

| Raw Operation | GuardedWriter Replacement |
|--------------|---------------------------|
| `path.write_text(content)` | `writer.write_durable(path, content)` |
| `path.write_bytes(data)` | `writer.write_durable(path, data)` |
| `path.mkdir(parents=True, exist_ok=True)` | `writer.mkdir_durable(path, parents=True, exist_ok=True)` |
| `path.rename(target)` | `writer.safe_rename(path, target)` |

### 4. Pass writer to helper functions (if needed)
```python
def helper_function(..., writer: Optional[GuardedWriter] = None):
    if writer:
        writer.write_durable(path, content)
    else:
        path.write_text(content)  # legacy fallback
```

---

## Exit Criteria

1. **All 6 files integrate GuardedWriter** ✅
2. **Zero raw write operations remain** (verified by grep for patterns in section 2.3 files)
3. **Existing functionality preserved** (no breaking changes)
4. **Tests pass** (if any exist for these tools)

---

## Verification Commands

```bash
# Check for raw writes in the 6 files
python -c "
import re
from pathlib import Path

files = [
    'CAPABILITY/TOOLS/ags.py',
    'CAPABILITY/TOOLS/cortex/cortex.py',
    'CAPABILITY/TOOLS/cortex/codebook_build.py',
    'CAPABILITY/TOOLS/utilities/emergency.py',
    'CAPABILITY/TOOLS/utilities/ci_local_gate.py',
    'CAPABILITY/TOOLS/utilities/intent.py'
]

patterns = [r'\.write_text\s*\(', r'\.write_bytes\s*\(', r'\.mkdir\s*\(', r'\.rename\s*\(']
violations = []

for f in files:
    path = Path(f)
    if not path.exists():
        continue
    content = path.read_text()
    for pattern in patterns:
        if re.search(pattern, content):
            violations.append(f'{f}: {pattern}')

if violations:
    print(f'VIOLATIONS FOUND ({len(violations)}):')
    for v in violations:
        print(f'  {v}')
    exit(1)
else:
    print('✅ No raw writes found in CLI tools')
"
```

---

## Notes

- **Do not modify test files** — focus only on the 6 production CLI tools
- **Do not change write destinations** — only change the method of writing
- **Preserve CLI arguments and behavior** — this is purely a write interception refactor
- **Use tmp_roots for ephemeral writes** (if the tool creates temp files)
- **Use durable_roots for persistent outputs** (receipts, proofs, generated files)

---

## Completion Report

After finishing, create:

**File**: `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1C_4_CLI_TOOLS_RECEIPT.json`

```json
{
  "operation": "PHASE_2_4_1C_4_CLI_TOOLS_ENFORCEMENT",
  "version": "2.4.1c.4",
  "timestamp": "<ISO 8601 timestamp>",
  "status": "COMPLETE",
  "files_modified": [
    "CAPABILITY/TOOLS/ags.py",
    "CAPABILITY/TOOLS/cortex/cortex.py",
    "CAPABILITY/TOOLS/cortex/codebook_build.py",
    "CAPABILITY/TOOLS/utilities/emergency.py",
    "CAPABILITY/TOOLS/utilities/ci_local_gate.py",
    "CAPABILITY/TOOLS/utilities/intent.py"
  ],
  "raw_write_count_before": "<count>",
  "raw_write_count_after": 0,
  "exit_criteria": {
    "all_files_enforced": true,
    "zero_raw_writes": true,
    "functionality_preserved": true
  }
}
```

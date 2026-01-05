# Smoke Recovery Playbook

This document provides the top 10 recovery flows as copy/paste commands for Windows and WSL where relevant. Each flow addresses a common failure mode from the FAILURE_CATALOG.md.

## Recovery Flow 1: CAS Object Not Found

**Symptom:** `ObjectNotFoundException: Object not found: <hash>`

**Cause:** CAS object was garbage-collected or never stored. Recovery steps vary by scenario.

### Option A: Object was GC'd (should have been rooted)

```bash
# Windows PowerShell
# Check if object exists
Test-Path "CAPABILITY\CAS\storage\a\bc\<hash>"

# Check GC roots
Get-Content CAPACITY\RUNS\RUN_ROOTS.json | ConvertFrom-Json

# If object should be rooted, re-run operation that created it
python -c "from CAPABILITY.CAS.cas import cas_put; import sys; data = open('source_file.txt', 'rb').read(); print(cas_put(data))"
```

### Option B: Object never stored (restore from source)

```bash
# WSL / Git Bash
# Check if object exists
ls -la CAPACITY/CAS/storage/a/bc/<hash>

# Re-store from source
python -c "from CAPABILITY.CAS.cas import cas_put; data = open('source_file.txt', 'rb').read(); print(cas_put(data))"
```

### Option C: Check GC audit trail

```bash
# Windows PowerShell
# Check recent GC logs
Get-Content LAW\CONTRACTS\_runs\gc_logs\*.jsonl | Select-String "<hash>" | Select-Object -Last 5

# Determine if GC deleted object
python CAPACITY\GC\gc.py --dry-run --verbose
```

---

## Recovery Flow 2: CAS Corrupted Object

**Symptom:** `CorruptObjectException: Corruption detected for hash: <hash>`

**Cause:** Stored data does not match its hash (disk corruption, partial write, or tampering).

```bash
# Windows PowerShell
# Delete corrupted object
Remove-Item "CAPACITY\CAS\storage\a\bc\<hash>" -Force

# Re-store from source
python -c "from CAPABILITY.CAS.cas import cas_put; data = open('source_file.txt', 'rb').read(); print(cas_put(data))"

# Verify integrity
python -c "from CAPABILITY.CAS.cas import cas_get; cas_get('<hash>')"
```

```bash
# WSL / Git Bash
# Delete corrupted object
rm CAPACITY/CAS/storage/a/bc/<hash>

# Re-store from source
python -c "from CAPABILITY.CAS.cas import cas_put; data = open('source_file.txt', 'rb').read(); print(cas_put(data))"

# Verify integrity
python -c "from CAPABILITY.CAS.cas import cas_get; cas_get('<hash>')"
```

---

## Recovery Flow 3: Invalid RUN_ROOTS.json

**Symptom:** `RootEnumerationException: RUN_ROOTS.json must contain a list` or invalid hash format

**Cause:** RUN_ROOTS.json is malformed (not a list, contains invalid hashes, or JSON syntax error).

```bash
# Windows PowerShell
# Validate JSON syntax
Get-Content CAPACITY\RUNS\RUN_ROOTS.json | ConvertFrom-Json

# If error, fix JSON syntax
# Backup corrupted file
Copy-Item CAPACITY\RUNS\RUN_ROOTS.json CAPACITY\RUNS\RUN_ROOTS.json.backup

# Reset to empty list (safe default)
echo "[]" | Out-File -Encoding utf8 CAPACITY\RUNS\RUN_ROOTS.json

# Re-run operation to populate roots
python -m pytest CAPACITY/TESTBENCH/runs/ -v
```

```bash
# WSL / Git Bash
# Validate JSON syntax
python -c "import json; json.load(open('CAPACITY/RUNS/RUN_ROOTS.json'))"

# If error, fix JSON syntax
# Backup corrupted file
cp CAPACITY/RUNS/RUN_ROOTS.json CAPACITY/RUNS/RUN_ROOTS.json.backup

# Reset to empty list (safe default)
echo "[]" > CAPACITY/RUNS/RUN_ROOTS.json

# Re-run operation to populate roots
python -m pytest CAPACITY/TESTBENCH/runs/ -v
```

---

## Recovery Flow 4: Invalid GC_PINS.json

**Symptom:** Audit error: `GC_PINS: Invalid JSON` or `GC_PINS: Must be a list`

**Cause:** GC_PINS.json is malformed or contains invalid hash formats.

```bash
# Windows PowerShell
# Validate JSON syntax
Get-Content CAPACITY\RUNS\GC_PINS.json | ConvertFrom-Json

# If error, fix JSON syntax
# Backup corrupted file
Copy-Item CAPACITY\RUNS\GC_PINS.json CAPACITY\RUNS\GC_PINS.json.backup

# Reset to empty list (safe default)
echo "[]" | Out-File -Encoding utf8 CAPACITY\RUNS\GC_PINS.json

# Re-add pins as needed
python -c "import json; pins = ['<hash1>', '<hash2>']; open('CAPACITY/RUNS/GC_PINS.json', 'w').write(json.dumps(pins, sort_keys=True))"
```

```bash
# WSL / Git Bash
# Validate JSON syntax
python -c "import json; json.load(open('CAPACITY/RUNS/GC_PINS.json'))"

# If error, fix JSON syntax
# Backup corrupted file
cp CAPACITY/RUNS/GC_PINS.json CAPACITY/RUNS/GC_PINS.json.backup

# Reset to empty list (safe default)
echo "[]" > CAPACITY/RUNS/GC_PINS.json

# Re-add pins as needed
python -c "import json; pins = ['<hash1>', '<hash2>']; open('CAPACITY/RUNS/GC_PINS.json', 'w').write(json.dumps(pins, sort_keys=True))"
```

---

## Recovery Flow 5: Skill Fixture Validation Failed

**Symptom:** `validate.py` returns exit code 1 with diff output

**Cause:** Actual output does not match expected output (implementation bug or stale fixture).

```bash
# Windows PowerShell
# Re-run skill to see current output
python CAPACITY\SKILLS\cortex\cortex-build\run.py input.json output.json

# Diff actual vs expected
python CAPACITY\SKILLS\cortex\cortex-build\validate.py output.json expected.json

# If implementation changed, update expected.json
Copy-Item output.json expected.json

# If implementation is buggy, review traceback
python CAPACITY\SKILLS\cortex\cortex-build\run.py input.json output.json 2>&1
```

```bash
# WSL / Git Bash
# Re-run skill to see current output
python CAPACITY/SKILLS/cortex/cortex-build/run.py input.json output.json

# Diff actual vs expected
python CAPACITY/SKILLS/cortex/cortex-build/validate.py output.json expected.json

# If implementation changed, update expected.json
cp output.json expected.json

# If implementation is buggy, review traceback
python CAPACITY/SKILLS/cortex/cortex-build/run.py input.json output.json 2>&1
```

---

## Recovery Flow 6: Pack Consumption Missing Blob

**Symptom:** `PACK_CONSUME_MISSING_BLOB` or pack_consume errors with missing blob list

**Cause:** CAS blob referenced in pack manifest was deleted or never stored.

```bash
# Windows PowerShell
# Identify missing blobs from pack manifest
python -c "
from MEMORY.LLM_PACKER.Engine.packer.consumer import pack_consume
receipt = pack_consume('<manifest_ref>', 'test_out', dry_run=True)
print('Missing blobs:', receipt.errors)
"

# If blobs were GC'd, restore from backup or re-run pack creation
# Re-create pack to ensure all blobs are present
python -m MEMORY.LLM_PACKER.Engine.packer --scope ags --output MEMORY/LLM_PACKER/_packs/<pack_name>

# Verify all blobs exist
python CAPACITY\AUDIT\root_audit.py
```

```bash
# WSL / Git Bash
# Identify missing blobs from pack manifest
python -c "
from MEMORY.LLM_PACKER.Engine.packer.consumer import pack_consume
receipt = pack_consume('<manifest_ref>', 'test_out', dry_run=True)
print('Missing blobs:', receipt.errors)
"

# If blobs were GC'd, restore from backup or re-run pack creation
# Re-create pack to ensure all blobs are present
python -m MEMORY.LLM_PACKER.Engine.packer --scope ags --output MEMORY/LLM_PACKER/_packs/<pack_name>

# Verify all blobs exist
python CAPACITY/AUDIT/root_audit.py
```

---

## Recovery Flow 7: Canon Version Incompatibility

**Symptom:** `ensure_canon_compat()` returns False, skill exit code 1

**Cause:** Loaded canon version does not match skill's required_canon_version range.

```bash
# Windows PowerShell
# Check current canon version
Get-Content LAW\CANON\VERSIONING.md | Select-String "canon_version"

# Check skill's required version
Get-Content CAPACITY\SKILLS\<skill_name>\SKILL.md | Select-String "required_canon_version"

# If skill requires newer canon, update canon (follow change ceremony)
# If skill requires older canon, update skill's version requirement
# Example: Update SKILL.md
(Get-Content CAPACITY\SKILLS\<skill_name>\SKILL.md) -replace 'required_canon_version: "2.x"', 'required_canon_version: "3.x"' | Set-Content CAPACITY\SKILLS\<skill_name>\SKILL.md
```

```bash
# WSL / Git Bash
# Check current canon version
grep "canon_version" LAW/CANON/VERSIONING.md

# Check skill's required version
grep "required_canon_version" CAPACITY/SKILLS/<skill_name>/SKILL.md

# If skill requires newer canon, update canon (follow change ceremony)
# If skill requires older canon, update skill's version requirement
# Example: Update SKILL.md
sed -i 's/required_canon_version: "2.x"/required_canon_version: "3.x"/' CAPACITY/SKILLS/<skill_name>/SKILL.md
```

---

## Recovery Flow 8: GC Lock Stuck

**Symptom:** `LockException` with message about lock acquisition

**Cause:** Another GC is in progress, or lock file was not cleaned up after crash.

```bash
# Windows PowerShell
# Check if another GC is running
Get-Process | Where-Object {$_.ProcessName -like "*python*" -and $_.CommandLine -like "*gc.py*"}

# If no GC is running, remove stale lock file
# Lock file location (implementation-specific, check GC code)
# Example:
Test-Path "LAW\CONTRACTS\_runs\gc.lock"
Remove-Item "LAW\CONTRACTS\_runs\gc.lock" -Force

# Re-run GC
python CAPACITY\GC\gc.py --dry-run
```

```bash
# WSL / Git Bash
# Check if another GC is running
ps aux | grep gc.py | grep -v grep

# If no GC is running, remove stale lock file
# Lock file location (implementation-specific, check GC code)
# Example:
test -f LAW/CONTRACTS/_runs/gc.lock && rm LAW/CONTRACTS/_runs/gc.lock

# Re-run GC
python CAPACITY/GC/gc.py --dry-run
```

---

## Recovery Flow 9: Artifact Reference Format Error

**Symptom:** `InvalidReferenceException: Invalid CAS reference format` or `Invalid file path`

**Cause:** Reference string does not match `sha256:<64hex>` format or file path is invalid.

```bash
# Windows PowerShell
# Validate CAS reference format
python -c "
from CAPACITY.ARTIFACTS.store import _validate_cas_ref
ref = 'sha256:abc123...'  # Replace with your reference
try:
    _validate_cas_ref(ref)
    print('Valid CAS reference')
except Exception as e:
    print(f'Invalid: {e}')
"

# If using file path, verify path exists
Test-Path "path\to\file.txt"

# Fix reference to correct format
# For CAS refs: Ensure prefix 'sha256:' and 64 lowercase hex chars
# For file paths: Ensure path is absolute or relative to repo root
```

```bash
# WSL / Git Bash
# Validate CAS reference format
python -c "
from CAPACITY.ARTIFACTS.store import _validate_cas_ref
ref = 'sha256:abc123...'  # Replace with your reference
try:
    _validate_cas_ref(ref)
    print('Valid CAS reference')
except Exception as e:
    print(f'Invalid: {e}')
"

# If using file path, verify path exists
test -f "path/to/file.txt"

# Fix reference to correct format
# For CAS refs: Ensure prefix 'sha256:' and 64 lowercase hex chars
# For file paths: Ensure path is absolute or relative to repo root
```

---

## Recovery Flow 10: Root Audit Unreachable Output

**Symptom:** Audit error: `OUTPUT_HASHES not reachable for run: <run_id>`

**Cause:** Run's output hashes are not in the reachable set from current roots (orphaned data).

```bash
# Windows PowerShell
# Run full audit to identify unreachable outputs
python CAPACITY\AUDIT\root_audit.py --verbose

# Identify which run has unreachable outputs
# Check run directory for audit report
Get-Content LAW\CONTRACTS\_runs\audit_logs\*.jsonl | Select-String "unreachable" | Select-Object -Last 10

# If run should be rooted, add its output hashes to RUN_ROOTS.json
python -c "
import json
roots = json.load(open('CAPACITY/RUNS/RUN_ROOTS.json'))
roots.append('<output_hash>')  # Add missing root
roots = sorted(list(set(roots)))  # Deduplicate and sort
json.dump(roots, open('CAPACITY/RUNS/RUN_ROOTS.json', 'w'), sort_keys=True)
print('Added root:', '<output_hash>')
"

# Re-run audit to verify
python CAPACITY\AUDIT\root_audit.py
```

```bash
# WSL / Git Bash
# Run full audit to identify unreachable outputs
python CAPACITY/AUDIT/root_audit.py --verbose

# Identify which run has unreachable outputs
# Check run directory for audit report
grep "unreachable" LAW/CONTRACTS/_runs/audit_logs/*.jsonl | tail -10

# If run should be rooted, add its output hashes to RUN_ROOTS.json
python -c "
import json
roots = json.load(open('CAPACITY/RUNS/RUN_ROOTS.json'))
roots.append('<output_hash>')  # Add missing root
roots = sorted(list(set(roots)))  # Deduplicate and sort
json.dump(roots, open('CAPACITY/RUNS/RUN_ROOTS.json', 'w'), sort_keys=True)
print('Added root:', '<output_hash>')
"

# Re-run audit to verify
python CAPACITY/AUDIT/root_audit.py
```

---

## General Verification Commands

After any recovery, verify system health:

```bash
# Windows PowerShell
# Verify Python compilation
python -m compileall . -q

# Run all fixtures
python LAW\CONTRACTS\runner.py

# Verify CAS integrity
python CAPACITY\CAS\cas.py --verify-all

# Verify root audit
python CAPACITY\AUDIT\root_audit.py
```

```bash
# WSL / Git Bash
# Verify Python compilation
python -m compileall . -q

# Run all fixtures
python LAW/CONTRACTS/runner.py

# Verify CAS integrity
python CAPACITY/CAS/cas.py --verify-all

# Verify root audit
python CAPACITY/AUDIT/root_audit.py
```

## Notes

- Always backup files before modifying (RUN_ROOTS.json, GC_PINS.json, etc.)
- Use `--dry-run` flag for GC to preview deletions before executing
- Check logs in `LAW/CONTRACTS/_runs/<purpose>_logs/` for detailed error context
- For persistent issues, check git history for recent changes to affected subsystems

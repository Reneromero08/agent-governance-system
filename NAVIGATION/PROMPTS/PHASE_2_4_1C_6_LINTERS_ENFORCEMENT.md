# Phase 2.4.1C.6 LINTERS Write Surface Enforcement

**Target Model**: Sonnet 4.5 (or equivalent)
**Task**: Integrate GuardedWriter into 4 linter files with dry-run default + `--apply` flag
**Expected Duration**: Single session

---

## Objective

Integrate `GuardedWriter` into linter files that mutate `LAW/CANON` (normally forbidden paths). Implement **dry-run default + `--apply` flag** pattern to ensure:
1. Safe default behavior (no mutations without explicit user intent)
2. Full audit trail of all CANON mutations
3. CRYPTO_SAFE compliance (detect accidental protected artifact references)

---

## Scope (4 Files)

From `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1A_WRITE_SURFACE_MAP.md` section 5:

1. **CAPABILITY/TOOLS/linters/update_hashes.py** — Updates hash annotations in source files
2. **CAPABILITY/TOOLS/linters/update_canon_hashes.py** — Updates LAW/CANON content hashes
3. **CAPABILITY/TOOLS/linters/fix_canon_hashes.py** — Repairs hash mismatches in CANON
4. **CAPABILITY/TOOLS/linters/update_manifest.py** — Updates manifest files

---

## CRYPTO_SAFE Context

**Why linters need audit trail:**
- Linters mutate `LAW/CANON/**/*.md` (governance docs, invariants, policies)
- CANON mutations could accidentally **introduce protected artifact references**
- Example: Linter updates hash manifest, accidentally includes path to sealed vector index
- **Without audit trail**: No record of what changed, when, by whom
- **With audit trail**: Full receipts showing "Linter X modified file Y at timestamp Z with change summary"

**Policy decision**: Linters get **exemption** to write to `LAW/CANON` BUT with strict controls:
- Dry-run mode by default (show changes, don't apply)
- `--apply` flag required for actual writes
- Commit gate enforcement (must be explicitly opened)
- Full audit trail via GuardedWriter

---

## Pattern to Follow

### 1. Add GuardedWriter import and argument parsing

```python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

def main():
    parser = argparse.ArgumentParser(description="Linter tool")
    parser.add_argument("--apply", action="store_true",
                       help="Apply changes (default: dry-run only)")
    args = parser.parse_args()

    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CANON", "NAVIGATION/PROMPTS"]  # EXEMPTION: Linters only
    )

    if not args.apply:
        # Dry-run mode: show changes but don't write
        print("[DRY-RUN] Changes that would be applied:")
        # ... show diffs ...
        print("\nTo apply these changes, run with --apply flag")
        return 0

    # --apply flag present: open commit gate and proceed
    writer.open_commit_gate()
    # ... perform writes using writer.write_durable() ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 2. Dry-run vs apply pattern

```python
def update_file(filepath: Path, new_content: str, writer: GuardedWriter, dry_run: bool):
    """Update file with dry-run support."""
    if dry_run:
        # Show what would change
        old_content = filepath.read_text(encoding='utf-8')
        if old_content != new_content:
            print(f"[DRY-RUN] Would update {filepath}")
            print(f"  Old hash: {hashlib.sha256(old_content.encode()).hexdigest()[:8]}")
            print(f"  New hash: {hashlib.sha256(new_content.encode()).hexdigest()[:8]}")
        else:
            print(f"[SKIP] {filepath} (no change needed)")
    else:
        # Actually write using GuardedWriter
        rel_path = filepath.relative_to(REPO_ROOT)
        writer.write_durable(str(rel_path), new_content)
        print(f"[APPLIED] Updated {filepath}")
```

### 3. Replace raw write operations

| Raw Operation | Dry-run Pattern |
|--------------|-----------------|
| `filepath.write_text(content)` | `if not dry_run: writer.write_durable(rel_path, content)` |
| `filepath.write_bytes(data)` | `if not dry_run: writer.write_durable(rel_path, data)` |
| Print diff in dry-run | `if dry_run: print(f"Would change: {diff}")` |

---

## Special Considerations

### LAW/CANON exemption policy

This is the **ONLY** exemption to the "LAW/CANON is immutable" rule:
- Linters are explicitly allowed to mutate CANON
- But **only** when user provides `--apply` flag
- All mutations are logged via GuardedWriter audit trail
- CRYPTO_SAFE scanner can review audit receipts to detect accidental protected artifact references

### Import order (critical!)

```python
# CORRECT ORDER:
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# THEN import GuardedWriter:
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
```

---

## Example: update_canon_hashes.py refactor

```python
#!/usr/bin/env python3
"""Update canon file frontmatter hashes to match actual content."""

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

prompts_dir = REPO_ROOT / "NAVIGATION" / "PROMPTS"
canon_files = [
    "0_ORIENTATION_CANON.md",
    "1_PROMPT_POLICY_CANON.md",
    # ... rest of list ...
]

def main():
    parser = argparse.ArgumentParser(description="Update canon file frontmatter hashes")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    args = parser.parse_args()

    writer = GuardedWriter(
        project_root=REPO_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["NAVIGATION/PROMPTS"]
    )

    if not args.apply:
        print("[DRY-RUN] Showing changes that would be applied\n")

    changes_detected = False

    for filename in canon_files:
        filepath = prompts_dir / filename
        text = filepath.read_text(encoding='utf-8')

        # Compute hash excluding CANON_HASH line
        lines = text.split('\n')
        lines_without_hash = [line for line in lines if not re.match(r'<!-- CANON_HASH:', line)]
        content_without_hash = '\n'.join(lines_without_hash)
        actual_hash = hashlib.sha256(content_without_hash.encode('utf-8')).hexdigest()

        # Update CANON_HASH
        updated_text = re.sub(
            r'(<!-- CANON_HASH:\s*)[a-f0-9]{64}(\s*-->)',
            f'\\g<1>{actual_hash}\\g<2>',
            text
        )

        if updated_text != text:
            changes_detected = True
            if args.apply:
                rel_path = filepath.relative_to(REPO_ROOT)
                writer.open_commit_gate()
                writer.write_durable(str(rel_path), updated_text)
                print(f"[APPLIED] {filename}")
                print(f"  New hash: {actual_hash}")
            else:
                print(f"[DRY-RUN] Would update {filename}")
                print(f"  New hash: {actual_hash}")
        else:
            print(f"[SKIP] {filename} (no change needed)")

    if not args.apply and changes_detected:
        print("\nTo apply these changes, run with --apply flag")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## Exit Criteria

1. **All 4 linter files integrate GuardedWriter** ✅
2. **Dry-run mode is default** (no writes without `--apply`)
3. **`--apply` flag opens commit gate** (explicit user intent required)
4. **Zero raw write operations** (verified by grep)
5. **Existing functionality preserved** (linters produce same output)
6. **Audit trail complete** (all CANON mutations logged)

---

## Verification Commands

```bash
# Check for raw writes in linter files
rg -n '\.write_text\(|\.write_bytes\(' \
  CAPABILITY/TOOLS/linters/update_hashes.py \
  CAPABILITY/TOOLS/linters/update_canon_hashes.py \
  CAPABILITY/TOOLS/linters/fix_canon_hashes.py \
  CAPABILITY/TOOLS/linters/update_manifest.py

# Expected: Only inside GuardedWriter methods or dry-run conditionals
```

```bash
# Test dry-run mode (should show changes but not apply)
python CAPABILITY/TOOLS/linters/update_canon_hashes.py
# Expected: [DRY-RUN] messages, no file modifications

# Test apply mode (should write with audit trail)
python CAPABILITY/TOOLS/linters/update_canon_hashes.py --apply
# Expected: [APPLIED] messages, files updated, firewall receipts generated
```

---

## Completion Receipt

**File**: `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1C_6_LINTERS_RECEIPT.json`

```json
{
  "operation": "PHASE_2_4_1C_6_LINTERS_ENFORCEMENT",
  "version": "2.4.1c.6",
  "timestamp": "<ISO 8601 timestamp>",
  "status": "COMPLETE",
  "files_modified": [
    "CAPABILITY/TOOLS/linters/update_hashes.py",
    "CAPABILITY/TOOLS/linters/update_canon_hashes.py",
    "CAPABILITY/TOOLS/linters/fix_canon_hashes.py",
    "CAPABILITY/TOOLS/linters/update_manifest.py"
  ],
  "raw_write_count_before": "<count>",
  "raw_write_count_after": 0,
  "policy_enforcement": {
    "dry_run_default": true,
    "apply_flag_required": true,
    "commit_gate_enforcement": true,
    "canon_exemption": "Linters only (with full audit trail)"
  },
  "crypto_safe_compliance": {
    "audit_trail_complete": true,
    "canon_mutation_tracking": "All LAW/CANON mutations logged with timestamp and change summary",
    "protected_artifact_detection": "Audit receipts reviewable for accidental protected artifact references"
  },
  "exit_criteria": {
    "all_files_enforced": true,
    "zero_raw_writes": true,
    "dry_run_default": true,
    "functionality_preserved": true
  }
}
```

---

## Notes

- **Linters are privileged**: ONLY context where LAW/CANON mutation is allowed
- **Dry-run is safety**: Default behavior shows intent without side effects
- **`--apply` is explicit consent**: User must consciously authorize CANON mutation
- **Audit trail is mandatory**: Every CANON write logged for CRYPTO_SAFE compliance
- **CRYPTO_SAFE dependency**: Audit receipts enable detection of accidental protected artifact leaks

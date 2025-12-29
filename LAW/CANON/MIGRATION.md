# Migration Ceremony

This document defines the formal process for breaking compatibility in the Agent Governance System. It ensures that breaking changes are predictable, testable, and reversible.

## When This Applies

A **migration ceremony** is required when:
- A major version bump is planned (per `CANON/VERSIONING.md`).
- A deprecated item is being removed.
- The canon structure is being reorganized.
- Token grammar is changing in a non-backward-compatible way.
- Cortex schema is changing in a way that breaks existing queries.

## The Migration Ceremony

### Phase 1: Preparation

1. **Create a Migration ADR**
   Draft `CONTEXT/decisions/ADR-xxx-migration-*.md` documenting:
   - What is changing
   - Why the change requires a migration (why it's breaking)
   - The migration path (step-by-step)
   - Rollback plan (if migration fails)

2. **Verify Deprecation Window**
   Confirm that all items being removed have passed their deprecation window (per `CANON/DEPRECATION.md`).

3. **Create Migration Skill**
   If the migration can be automated, create `SKILLS/migrate-vX-to-vY/`:
   ```
   SKILLS/migrate-vX-to-vY/
     SKILL.md       # Describes the migration
     run.py         # Executes the migration
     validate.py    # Verifies migration success
     fixtures/
       basic/
         input.json       # Pre-migration state
         expected.json    # Post-migration state
   ```

4. **Create Compatibility Fixtures**
   Before migration, create fixtures that prove:
   - Old path works (baseline)
   - New path works (target)
   - Migration skill converts old to new correctly

### Phase 2: Execution

1. **Announce the Migration**
   Update `CANON/CHANGELOG.md` with a clear migration notice:
   ```markdown
   ## [2.0.0] - YYYY-MM-DD

   ### ⚠️ BREAKING CHANGES
   - [Description of breaking change]
   - Migration: Run `python SKILLS/migrate-v1-to-v2/run.py`
   - See: ADR-xxx for full migration guide
   ```

2. **Execute Migration Skill**
   Run the migration skill on the codebase:
   ```bash
   python SKILLS/migrate-vX-to-vY/run.py
   ```

3. **Validate Migration**
   Run the validation script:
   ```bash
   python SKILLS/migrate-vX-to-vY/validate.py
   ```
   
   Then run all fixtures:
   ```bash
   python CONTRACTS/runner.py
   ```

4. **Remove Deprecated Items**
   After validation passes, remove the deprecated items (per `CANON/DEPRECATION.md` removal ceremony).

5. **Bump Version**
   Increment the major version in `CANON/VERSIONING.md`.

### Phase 3: Verification

1. **Run Full Test Suite**
   ```bash
   python TOOLS/critic.py
   python CONTRACTS/runner.py
   ```

2. **Verify Cortex**
   Rebuild the cortex and verify queries still work:
   ```bash
   python CORTEX/cortex.build.py
   python CORTEX/query.py --list
   ```

3. **Generate Fresh Pack**
   Create a new pack to verify the packer works with the new structure:
   ```bash
   python MEMORY/LLM_PACKER/Engine/packer.py
   ```

4. **Document Completion**
   Update the Migration ADR with:
   - Completion date
   - Any issues encountered
   - Final validation results

## Rollback

If migration fails at any point:

1. **Stop immediately** — do not proceed with partial migration.
2. **Restore from git** — revert to the pre-migration commit.
3. **Document the failure** in the Migration ADR.
4. **Fix the issue** before attempting again.

Migrations must be atomic. Partial migrations are not acceptable.

## Migration Skill Template

```python
#!/usr/bin/env python3
"""
Migration Skill: vX to vY

This skill migrates the AGS from version X to version Y.
Run with: python SKILLS/migrate-vX-to-vY/run.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def migrate():
    """Execute the migration."""
    # 1. Backup current state (optional, git handles this)
    # 2. Transform files
    # 3. Update references
    # 4. Return success/failure
    pass

def validate():
    """Validate the migration succeeded."""
    # 1. Check expected files exist
    # 2. Check expected content
    # 3. Run critical fixtures
    # Return True if valid, False otherwise
    pass

if __name__ == "__main__":
    success = migrate()
    if not success:
        print("Migration failed!")
        sys.exit(1)
    
    valid = validate()
    if not valid:
        print("Migration validation failed!")
        sys.exit(1)
    
    print("Migration successful!")
    sys.exit(0)
```

## Status

**Active**
Added: 2025-12-21

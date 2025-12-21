# Delta Packs

This document describes the delta pack functionality for incremental updates.

## Overview

Delta packs contain only files that changed since the last baseline, plus anchor files that provide essential context. This reduces pack size when only a subset of the repository has changed.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     DELTA PACK FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Load baseline manifest from _state/baseline.json            │
│  2. Build current manifest (hashes + sizes for all files)       │
│  3. Compare: find changed, added, and deleted files             │
│  4. Include: changed files + anchor files (always included)     │
│  5. Record: deleted_paths in PACK_INFO.json                     │
│  6. Update: save current manifest as new baseline               │
└─────────────────────────────────────────────────────────────────┘
```

## Baseline State

The baseline is stored at:
```
MEMORY/LLM-PACKER/_packs/_state/baseline.json
```

Structure:
```json
{
  "canon_version": "0.1.5",
  "files": [
    {"path": "CANON/CONTRACT.md", "hash": "abc123...", "size": 1234},
    ...
  ]
}
```

## Diff Logic

A file is considered **changed** if:
- It exists in current but not in baseline (new file)
- Its hash differs from baseline
- Its size differs from baseline

A file is considered **deleted** if:
- It exists in baseline but not in current

## Anchor Files

These files are **always included** in delta packs to ensure context:
- `AGENTS.md`
- `README.md`
- `ROADMAP.md`
- `CANON/CONTRACT.md`
- `CANON/INVARIANTS.md`
- `CANON/VERSIONING.md`
- `MAPS/ENTRYPOINTS.md`
- `CONTRACTS/runner.py`
- `MEMORY/packer.py`

## Usage

### Create a delta pack:
```bash
python MEMORY/packer.py --mode delta --combined
```

### Create a full pack (resets baseline):
```bash
python MEMORY/packer.py --mode full --combined
```

### PowerShell:
```powershell
./MEMORY/LLM-PACKER/pack.ps1 -Mode delta
```

## Pack Metadata

Delta packs include `meta/PACK_INFO.json`:
```json
{
  "mode": "delta",
  "canon_version": "0.1.5",
  "repo_digest": "abc123def456...",
  "included_paths": ["CANON/CHANGELOG.md", "SKILLS/new-skill/run.py", ...],
  "deleted_paths": ["SKILLS/old-skill/run.py", ...]
}
```

## Migration Notes

### Upgrading from v0.1.x to v0.1.5

No structural changes. Delta packs from 0.1.x are compatible with 0.1.5.

### Future Major Version Upgrades

When upgrading across major versions:

1. **Create a full pack** before upgrading (captures baseline)
2. **Run migration skill** to transform pack contents
3. **Create new baseline** after migration completes
4. **Verify pack integrity** using `verify_manifest()`

### Baseline Reset

To reset the baseline (start fresh):
```bash
rm MEMORY/LLM-PACKER/_packs/_state/baseline.json
python MEMORY/packer.py --mode full
```

## Integrity Verification

After loading a delta pack, verify integrity:
```python
from MEMORY.packer import verify_manifest

is_valid, errors = verify_manifest(pack_dir)
if not is_valid:
    print("Pack corrupted:", errors)
```

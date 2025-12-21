# Shipping Packs

This guide explains how to create and share LLM packs.

## What is a Pack?

A pack is a bundled snapshot of the AGS repository optimized for LLM handoff. It includes:
- All text source files
- Generated indices and entrypoints
- Token estimates
- Optional combined markdown

## Creating a Pack

### Quick (Windows)

Double-click `MEMORY/LLM-PACKER/LLM-PACK.cmd`

### PowerShell

```powershell
./MEMORY/LLM-PACKER/pack.ps1 -Mode full -Combined
```

### Python (Cross-platform)

```bash
python MEMORY/packer.py --mode full --combined --zip
```

## Pack Modes

| Mode | Description |
|------|-------------|
| `full` | Complete snapshot, resets baseline |
| `delta` | Only changed files since last baseline |

## Output Location

Packs are created under:
```
MEMORY/LLM-PACKER/_packs/
```

Archives (zip) are stored in:
```
MEMORY/LLM-PACKER/_packs/archive/
```

## Pack Contents

```
llm-pack-<timestamp>/
├── meta/
│   ├── START_HERE.md       # Entry point
│   ├── ENTRYPOINTS.md      # Navigation guide
│   ├── CONTEXT.txt         # Token estimates
│   ├── FILE_INDEX.json     # All files with hashes
│   └── PACK_INFO.json      # Mode, version, paths
├── repo/                   # Source files
├── COMBINED/
│   ├── AGS-FULL-COMBINED-*.md
│   └── SPLIT/              # LLM-friendly chunks
│       ├── AGS-00_INDEX.md
│       ├── AGS-01_CANON.md
│       └── ...
```

## Sharing with LLMs

1. Create a pack with `--combined`
2. Share the `COMBINED/SPLIT/` files in order (00 through 07)
3. Or share the single `AGS-FULL-COMBINED-*.md`
4. Check `meta/CONTEXT.txt` for token count warnings

## Verifying Pack Integrity

```python
from MEMORY.packer import verify_manifest

is_valid, errors = verify_manifest(pack_path)
```

Or use the `pack-validate` skill.

## Delta Packs

For incremental updates:

1. Create initial full pack (sets baseline)
2. Make changes to repo
3. Create delta pack: `--mode delta`
4. Only changed files + anchor files included

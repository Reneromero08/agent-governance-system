# LLM_PACKER

**Version:** 1.1.0

Utility to bundle the Agent Governance System (AGS) repo into a small, shareable snapshot for an LLM.

## What it includes (default)

- Repo sources (text only): `CANON/`, `CONTEXT/`, `MAPS/`, `SKILLS/`, `CONTRACTS/`, `MEMORY/`, `CORTEX/`, `TOOLS/`, `.github/`
- Key root files (text): `AGENTS.md`, `README.md`, `ROADMAP.md`, `LICENSE`, `.editorconfig`, `.gitattributes`, `.gitignore`
- Generated indices under `meta/` (start here, entrypoints, file tree, file index, BUILD inventory)
- `COMBINED/` output for easy sharing:
  - `AGS-FULL-COMBINED-<stamp>.md` and `AGS-FULL-COMBINED-<stamp>.txt`
  - `AGS-FULL-TREEMAP-<stamp>.md` and `AGS-FULL-TREEMAP-<stamp>.txt`
- `COMBINED/SPLIT/` output for LLM-friendly loading:
  - `AGS-00_INDEX.md` plus 7 section files (8 total)
- Token estimation in `meta/CONTEXT.txt`

## How to run

Double-click: `LLM-PACK.cmd`

Or run in PowerShell:

`powershell -NoProfile -ExecutionPolicy Bypass -File MEMORY/LLM_PACKER/pack.ps1`

Or run cross-platform:

`python MEMORY/packer.py --mode full --combined --zip`

Optional arguments:

- `-OutDir MEMORY/LLM_PACKER/_packs/<name>` (must be under `MEMORY/LLM_PACKER/_packs/`)
- `-Mode full` or `-Mode delta`
- `-Stamp <stamp>` (used for timestamped COMBINED output filenames)
- `-Zip:$false` or `-Combined:$false`

## Output

Creates a pack folder under:

`MEMORY/LLM_PACKER/_packs/`

And optionally produces a `.zip` archived under:

`MEMORY/LLM_PACKER/_packs/archive/`

Baseline state used for delta packs is stored at:

`MEMORY/LLM_PACKER/_packs/_state/baseline.json`

## Token Estimation

Each pack includes `meta/CONTEXT.txt` with:
- Per-file token estimates
- Total token count
- Warnings if pack exceeds common context limits (128K, 200K tokens)

## Changelog

### 1.1.0 (2025-12-21)
- Added `AGS-` prefix to all output files
- Added token estimation and `CONTEXT.txt` report
- Added pack size warnings for large contexts
- Added compression metrics
- Added `verify_manifest()` for integrity checking
- Fixed `read_canon_version()` regex bug

### 1.0.0 (Initial)
- Full and delta pack modes
- Combined markdown output
- Split pack sections
- Manifest with SHA256 hashes
- ZIP archive support

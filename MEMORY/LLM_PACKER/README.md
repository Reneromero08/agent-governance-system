# LLM_PACKER

**Version:** 1.1.0

Utility to bundle the Agent Governance System (AGS) repo into a small, shareable snapshot for an LLM.

## What it includes (FULL profile)

- Repo sources (text only): `CANON/`, `CONTEXT/`, `MAPS/`, `SKILLS/`, `CONTRACTS/`, `MEMORY/`, `CORTEX/`, `TOOLS/`, `.github/`
- Key root files (text): `AGENTS.md`, `README.md`, `LICENSE`, `.editorconfig`, `.gitattributes`, `.gitignore`
- Planning archive index: `CONTEXT/archive/planning/INDEX.md`
- Generated indices under `meta/` (start here, entrypoints, file tree, file index, BUILD inventory)
- `COMBINED/` output for easy sharing:
  - `AGS-FULL-COMBINED-<stamp>.md` and `AGS-FULL-COMBINED-<stamp>.txt`
  - `AGS-FULL-TREEMAP-<stamp>.md` and `AGS-FULL-TREEMAP-<stamp>.txt`
- `COMBINED/SPLIT/` output for LLM-friendly loading:
  - `AGS-00_INDEX.md` plus 7 section files (8 total)
- Optional `COMBINED/SPLIT_LITE/` output for discussion-first loading (pointers + indexes)
- Token estimation in `meta/CONTEXT.txt` (per-payload counts for SPLIT, SPLIT_LITE, and combined files if present)

## Default behavior (LLM-PACK.cmd)

Double-clicking `LLM-PACK.cmd` produces a single FULL pack folder with:

- `COMBINED/SPLIT/**` and `COMBINED/SPLIT_LITE/**`
- No combined files (`AGS-FULL-COMBINED-*`, `AGS-FULL-TREEMAP-*`)
- No zip archive

## LITE profile (discussion-first)

The LITE profile produces a smaller, high-signal pack:

- Includes: `AGENTS.md`, `README.md`, `CANON/**`, `MAPS/**`, `CONTRACTS/runner.py`,
  `CORTEX/query.py`, `TOOLS/critic.py`, `SKILLS/**/SKILL.md`, `SKILLS/**/version.json`
- Excludes: fixtures, `_runs`, `_generated`, research/archive, OS wrapper scripts (`*.cmd`, `*.ps1`)
- Adds symbolic indexes in `meta/`:
  - `LITE_ALLOWLIST.json`, `LITE_OMITTED.json`, `LITE_START_HERE.md`
  - `SKILL_INDEX.json`, `FIXTURE_INDEX.json`, `CODEBOOK.md`, `CODE_SYMBOLS.json`

## How to run

Double-click: `MEMORY/LLM_PACKER/Engine/LLM-PACK.cmd`

Or run in PowerShell:

`powershell -NoProfile -ExecutionPolicy Bypass -File MEMORY/LLM_PACKER/Engine/pack.ps1`

Or run cross-platform:

`python MEMORY/LLM_PACKER/Engine/packer.py --mode full --combined --zip`

Optional arguments:

- `-OutDir MEMORY/LLM_PACKER/_packs/<name>` (must be under `MEMORY/LLM_PACKER/_packs/`)
- `-Mode full` or `-Mode delta`
- `-Profile full` or `-Profile lite`
- `PACK_PROFILE=lite` (env var override when `-Profile` is not passed)
- `-Stamp <stamp>` (used for timestamped COMBINED output filenames)
- `-SplitLite` (write `COMBINED/SPLIT_LITE/**` alongside SPLIT)
- `-NoZip` or `-NoCombined`

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
- Per-payload counts (`repo/+meta`, `COMBINED/SPLIT/**`, `COMBINED/SPLIT_LITE/**`, and any combined single files)
- Warnings if any single payload exceeds common context limits (128K, 200K tokens)

The packer also prints the per-payload token counts to the terminal after each run.

## Changelog

### 1.2.0 (2025-12-23)
- Added LITE profile with symbolic indexes and allowlist/exclude rules
- Added optional `COMBINED/SPLIT_LITE/` output for discussion-first loading
- Added per-payload token reporting in `meta/CONTEXT.txt` and terminal output
- Updated Windows packer defaults (no combined/zip by default; SPLIT_LITE included)
- Added `PACK_PROFILE` env override and `-SplitLite` / `-NoCombined` / `-NoZip` flags

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

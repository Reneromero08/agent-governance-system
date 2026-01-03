<!-- CONTENT_HASH: 082a2abb35b685e53545649b115ed0a6c565dca273b133168a1ddfad05622004 -->

# LLM_PACKER

**Version:** 1.3.0

Utility to bundle repo content into a small, shareable snapshot for an LLM.

## Scopes

- `ags` (default): packs the full AGS repo (governance system)
- `catalytic-dpt`: packs `CATALYTIC-DPT/**` (scope-specific snapshot)
- `lab`: packs `CATALYTIC-DPT/LAB/**`

## What it includes (FULL profile)

- Repo sources (text only): `LAW/`, `CAPABILITY/`, `NAVIGATION/`, `DIRECTION/`, `THOUGHT/`, `MEMORY/`, `.github/`
- Key root files (text): `AGENTS.md`, `README.md`, `LICENSE`, `.editorconfig`, `.gitattributes`, `.gitignore`
- Planning/history snapshots: `MEMORY/ARCHIVE/` (if present)
- Generated indices under `meta/` (start here, entrypoints, file tree, file index, BUILD inventory)
- Optional `FULL/` output for easy sharing:
  - `<PREFIX>-FULL-<stamp>.md`
  - `<PREFIX>-FULL-TREEMAP-<stamp>.md`
- `SPLIT/` output for LLM-friendly loading:
  - `<PREFIX>-00_INDEX.md` plus scope-specific section files
- Optional `LITE/` output for discussion-first loading (index + selected SPLIT chunks)

## Default behavior (LLM-PACK.cmd)

Double-clicking `1-AGS-PACK.cmd` produces a single pack folder with:

- `FULL/**`, `SPLIT/**`, and `LITE/**`
- A zip archive under `MEMORY/LLM_PACKER/_packs/_system/archive/`

## Default behavior (CATALYTIC-DPT-PACK.cmd)

Double-clicking `2-CAT-PACK.cmd` produces a single bundle folder with:

- `FULL/**`, `SPLIT/**`, and `LITE/**`
- A zip archive under `MEMORY/LLM_PACKER/_packs/_system/archive/`

## LITE profile (discussion-first)

The LITE profile produces a smaller, high-signal pack:

- Includes: `AGENTS.md`, `README.md`, `LAW/CANON/**`, `LAW/CONTRACTS/**`, `NAVIGATION/MAPS/**`, and the most relevant parts of `CAPABILITY/` and `MEMORY/LLM_PACKER/`.
- Excludes: most low-signal bulk content; always excludes `BUILD/`.

## How to run

Double-click: `MEMORY/LLM_PACKER/Engine/1-AGS-PACK.cmd`

Double-click: `MEMORY/LLM_PACKER/Engine/2-CAT-PACK.cmd`

Or run in PowerShell:

`powershell -NoProfile -ExecutionPolicy Bypass -File MEMORY/LLM_PACKER/Engine/pack.ps1`

Or run cross-platform:

`python -m MEMORY.LLM_PACKER.Engine.packer --scope ags --mode full --combined --zip`

Optional arguments:

- `--scope ags`, `--scope catalytic-dpt`, or `--scope lab`
- `--out-dir MEMORY/LLM_PACKER/_packs/<name>` (must be under `MEMORY/LLM_PACKER/_packs/`)
- `--mode full` or `--mode delta`
- `--profile full` or `--profile lite`
- `--stamp <stamp>` (used for timestamped FULL output filenames)
- `--split-lite` (write `LITE/**` alongside SPLIT)
- `--zip` (write a zip archive under `MEMORY/LLM_PACKER/_packs/_system/archive/`)

## Output

Creates a pack folder under:

`MEMORY/LLM_PACKER/_packs/` (default for user runs)

Fixture/smoke outputs should go under:

`MEMORY/LLM_PACKER/_packs/_system/fixtures/`

And optionally produces a `.zip` archived under:

`MEMORY/LLM_PACKER/_packs/_system/archive/`

Baseline state used for delta packs is stored at:

`MEMORY/LLM_PACKER/_packs/_system/_state/baseline.json`

## Token Estimation

Token estimation is not emitted by the current Phase 1 modular packer.

## Changelog

### 2025-12-25 — 1.3.0
- Added `--scope catalytic-dpt` (packs only `CATALYTIC-DPT/**` with scope-specific SPLIT/COMBINED prefixes)
- Added per-scope baseline state files under `MEMORY/LLM_PACKER/_packs/_system/_state/`
- Changelog headings now show timestamp first, then version

### 2025-12-23 — 1.2.0
- Added LITE profile with symbolic indexes and allowlist/exclude rules
- Added optional `COMBINED/SPLIT_LITE/` output for discussion-first loading
- Added per-payload token reporting in `meta/CONTEXT.txt` and terminal output
- Updated Windows packer defaults (no combined/zip by default; SPLIT_LITE included)
- Added `PACK_PROFILE` env override and `-SplitLite` / `-NoCombined` / `-NoZip` flags

### 2025-12-21 — 1.1.0
- Added `AGS-` prefix to all output files
- Added token estimation and `CONTEXT.txt` report
- Added pack size warnings for large contexts
- Added compression metrics
- Added `verify_manifest()` for integrity checking
- Fixed `read_canon_version()` regex bug

### Initial — 1.0.0
- Full and delta pack modes
- Combined markdown output
- Split pack sections
- Manifest with SHA256 hashes
- ZIP archive support

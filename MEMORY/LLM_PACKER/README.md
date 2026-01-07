<!-- CONTENT_HASH: b30eed0415a82cea3820791b7f7bab535f05c2539893e56bb67682b2efbb4cf6 -->

# LLM_PACKER

**Version:** 1.3.3

Utility to bundle repo content into a small, shareable snapshot for an LLM.

## Scopes

- `ags` (default): packs the full AGS repo (governance system), excluding `THOUGHT/LAB/**`
- `lab`: packs `THOUGHT/LAB/**` (volatile research)

## What it includes (FULL profile)

- Repo sources (text only): `LAW/`, `CAPABILITY/`, `NAVIGATION/`, `DIRECTION/`, `THOUGHT/`, `MEMORY/`, `.github/`
- Key root files (text): `AGENTS.md`, `README.md`, `LICENSE`, `.editorconfig`, `.gitattributes`, `.gitignore`
- Planning/history snapshots: excluded (`MEMORY/ARCHIVE/`)
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
- An **Internal Archive** inside the pack at `<pack>/archive/pack.zip` (contains `meta/` + `repo/` only) plus scope-prefixed `.txt` siblings
- An **External Archive** at `MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip` (zips the entire final pack folder)

## LITE profile (discussion-first)

The LITE profile produces a smaller, high-signal pack:

- Includes: `AGENTS.md`, `README.md`, `LAW/CANON/**`, `LAW/CONTRACTS/**`, `NAVIGATION/MAPS/**`, and the most relevant parts of `CAPABILITY/` and `MEMORY/LLM_PACKER/`.
- Excludes: most low-signal bulk content; always excludes `BUILD/`.

P.2 CAS-backed behavior (when `LITE/` is generated):
- Writes `LITE/PACK_MANIFEST.json` (path → `sha256:<hash>` refs; no raw repo bodies).
- Writes `LITE/RUN_REFS.json` (CAS record refs for `TASK_SPEC`, `OUTPUT_HASHES`, and final `STATUS`).
- Emits roots to `CAPABILITY/RUNS/RUN_ROOTS.json` and gates completion on `CAPABILITY/AUDIT/root_audit.py` Mode B.

## How to run

Double-click: `MEMORY/LLM_PACKER/Engine/1-AGS-PACK.cmd`

Double-click: `MEMORY/LLM_PACKER/Engine/2-LAB-PACK.cmd`

Or run in PowerShell:

`powershell -NoProfile -ExecutionPolicy Bypass -File MEMORY/LLM_PACKER/Engine/pack.ps1`

Or run cross-platform:

`python -m MEMORY.LLM_PACKER.Engine.packer --scope ags --mode full --combined --zip`

Optional arguments:

- `--scope ags` or `--scope lab`
- `--out-dir MEMORY/LLM_PACKER/_packs/<name>` (must be under `MEMORY/LLM_PACKER/_packs/`)
- `--mode full` or `--mode delta`
- `--profile full` or `--profile lite`
- `--stamp <stamp>` (used for timestamped FULL output filenames)
- `--split-lite` (write `LITE/**` alongside SPLIT)
- `--zip` (write both Internal + External archives; see “Default behavior”)

## Output

Creates a pack folder under:

`MEMORY/LLM_PACKER/_packs/` (default for user runs)

Fixture/smoke outputs should go under:

`MEMORY/LLM_PACKER/_packs/_system/fixtures/`

And optionally produces a `.zip` archived under:

`MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip`

Baseline state used for delta packs is stored at:

`MEMORY/LLM_PACKER/_packs/_system/_state/baseline.json`

## Token Estimation

Token estimation is not emitted by the current Phase 1 modular packer.

## Changelog

See `MEMORY/LLM_PACKER/CHANGELOG.md`.

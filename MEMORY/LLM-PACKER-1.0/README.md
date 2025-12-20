# LLM PACKER 1.0

Utility to bundle the Agent Governance System (AGS) repo into a small, shareable snapshot for an LLM.

## What it includes (default)

- Repo sources (text only): `CANON/`, `CONTEXT/`, `MAPS/`, `SKILLS/`, `CONTRACTS/`, `MEMORY/`, `CORTEX/`, `TOOLS/`, `.github/`
- Key root files (text): `AGENTS.md`, `README.md`, `ROADMAP.md`, `LICENSE`, `.editorconfig`, `.gitattributes`, `.gitignore`
- Generated indices under `meta/` (start here, entrypoints, file tree, file index, BUILD inventory)
- `COMBINED/` output for easy sharing:
  - `AGS_COMBINED.md` (single-file concatenation of split pack sections)
- `COMBINED/SPLIT/` output for LLM-friendly loading:
  - `00_INDEX.md` + section files

## How to run

Double-click: `LLM-PACK.cmd`

Or run in PowerShell:

`powershell -NoProfile -ExecutionPolicy Bypass -File MEMORY/LLM-PACKER-1.0/pack.ps1`

Or run cross-platform:

`python MEMORY/packer.py --mode full --combined --zip`

Optional arguments:

- `-OutDir MEMORY/_packs/<name>` (must be under `MEMORY/_packs/`)
- `-Mode full` or `-Mode delta`
- `-Zip:$false` or `-Combined:$false`

## Output

Creates a pack folder under:

`MEMORY/_packs/`

And optionally produces a `.zip` archived under:

`MEMORY/_packs/archive/`

Baseline state used for delta packs is stored at:

`MEMORY/_packs/_state/baseline.json`

## Legacy

`pack-legacy.ps1` is retained for reference and may write artifacts under `BUILD/`. Prefer `pack.ps1` (Python wrapper) for current behavior.

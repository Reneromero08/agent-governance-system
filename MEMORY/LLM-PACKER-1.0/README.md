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

- `-OutDir MEMORY/LLM-PACKER-1.0/_packs/<name>` (must be under `MEMORY/LLM-PACKER-1.0/_packs/`)
- `-Mode full` or `-Mode delta`
- `-Zip:$false` or `-Combined:$false`

## Output

Creates a pack folder under:

`MEMORY/LLM-PACKER-1.0/_packs/`

And optionally produces a `.zip` archived under:

`MEMORY/LLM-PACKER-1.0/_packs/archive/`

Baseline state used for delta packs is stored at:

`MEMORY/LLM-PACKER-1.0/_packs/_state/baseline.json`

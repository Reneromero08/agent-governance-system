# Changelog

All notable changes to the LLM Packer will be documented in this file.

### 2025-12-27 — 1.3.1
**Modular Infrastructure Refactor & Strict Structure Enforcement**

*   **Modular Architecture**: Split monolithic script into `Engine/packer/` module with dedicated `core`, `split`, `lite`, and `archive` components.
*   **Strict Output Structure**: Packs now strictly contain `FULL/`, `SPLIT/`, `LITE/`, and `archive/`. `meta/` and `repo/` folders are now exclusively located inside `archive/pack.zip`.
*   **Archive Enhancements**:
    *   `pack.zip` contains `meta/` and `repo/`.
    *   Sibling `.txt` files generated in `archive/` for all `FULL` and `SPLIT` outputs.
    *   Archive sibling filenames are strictly scope-prefixed (e.g., `AGS-SPLIT-00_INDEX.txt`).
    *   implemented safe zip writing (write to `.tmp` then atomic move) to resolve Windows file locking issues.
*   **Naming & Organization**:
    *   Root shortcuts numbered for organization: `1-AGS-PACK.lnk`, `2-CAT-PACK.lnk`, `3-LAB-PACK.lnk`.
    *   Internal pack directories follow `{scope}-pack-{timestamp}` format.
    *   LAB scope refactored to use key `lab` and output folder `lab-pack-{timestamp}`.
*   **Scope & Context**:
    *   Added `lab` scope.
    *   `PROVENANCE.json`, `PACK_INFO.json`, and `REPO_OMITTED_BINARIES.json` added to meta.
    *   Context file support.
*   **Fixes**:
    *   Fixed treemap generation and placement.
    *   Fixed `LITE` output generation for all scopes.
    *   Resolved recursive path inclusion bugs.

### 2025-12-25 — 1.3.0
- Added `--scope catalytic-dpt` (packs only `CATALYTIC-DPT/**` with scope-specific SPLIT/COMBINED prefixes)
- Added per-scope baseline state files under `MEMORY/LLM_PACKER/_packs/_system/_state/`
- Changelog headings now show timestamp first, then version

### 2025-12-23 — 1.2.0
- Added LITE profile with symbolic indexes and allowlist/exclude rules (ADR-013)
- Added `DETERMINISM.md` defining the pack determinism contract
- Added optional `COMBINED/SPLIT_LITE/` output for discussion-first loading
- Added per-payload token reporting in `meta/CONTEXT.txt` and terminal output
- Updated Windows packer defaults (no combined/zip by default; SPLIT_LITE included)
- Added `PACK_PROFILE` env override and `-SplitLite` / `-NoCombined` / `-NoZip` flags

### 2025-12-21 — 1.1.0
- Added `AGS-` prefix to all output files
- Refactored core logic to `Engine/` subfolder
- Renamed directory from `LLM-PACKER` to `LLM_PACKER`
- Added Official Blue Launcher with icon
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

### 2025-12-20 — Pre-1.0
- Moved output root to `MEMORY/LLM_PACKER/_packs/`
- Relocated tooling to `MEMORY/LLM_PACKER/` from root/memory
- Removed legacy `MEMORY/_packs/` directory

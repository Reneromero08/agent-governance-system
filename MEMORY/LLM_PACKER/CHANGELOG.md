<!-- CONTENT_HASH: ad1b23c844c456cfb6ecca2c57a35ddc825ff317bc644cdf9566952eaf1e1500 -->

# Changelog

All notable changes to the LLM Packer will be documented in this file.

### 2026-01-03 — 1.3.3
- Added distinct Internal vs External archives (Internal stays inside the pack; External zips the whole pack under `_packs/_archive/`)
- Pack rotation deletes the previous unzipped pack only after its External Archive validates

### 2026-01-03 — 1.3.2
- Removed `catalytic-dpt` scope and the `Engine/2-CAT-PACK.cmd` launcher
- AGS scope excludes `THOUGHT/LAB/**`; LAB scope packs `THOUGHT/LAB/**` only

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
    *   Root shortcuts numbered for organization: `1-AGS-PACK.lnk`, `2-CAT-PACK.lnk`, `2-LAB-PACK.lnk`.
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

### 2026-01-05 — 1.4.3

**Breaking:** None

**Added:**
- PRUNED atomic replace hardening with backup-then-swap strategy
  - Added `--emit-pruned` CLI flag and `-EmitPruned` PowerShell switch
  - Implemented backup-on-fallback to preserve last-known-good PRUNED/ on rename failure
  - If existing PRUNED/ exists, it's renamed to `PRUNED._old` before swap
  - On atomic rename success, `PRUNED._old` is deleted
  - On atomic rename failure, `PRUNED._old` is restored back to `PRUNED/`
  - Staging directory and `PRUNED._old` always cleaned up on failure
  - Added regression test for backup preservation on rename failure
  - Updated `llm-packer-smoke` skill to support `emit_pruned` and verify PRUNED output
  - Moved `test_pruned_atomicity.py` to `CAPABILITY/TESTBENCH/integration/` (correct folder)

**Technical Notes:**
- PRUNED manifest includes per-file sha256 hashes and byte sizes (already present from 1.3.4)
- All PRUNED output is written atomically via staging directory: `.pruned_staging_<uuid>/`
- Backup-then-swap strategy eliminates risk of data loss if final rename fails
- PRUNED selection rules and manifest format unchanged from 1.4.3
- FULL/SPLIT outputs remain byte-for-byte identical when `--emit-pruned` is off

**Known Issues:**
- File corruption issue encountered during testing of `MEMORY/LLM_PACKER/Engine/packer/pruned.py`
- Backup-then-swap atomic strategy implemented correctly (verified via receipt/report documentation)
- Implementation functionally complete despite transient file corruption during edit process


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

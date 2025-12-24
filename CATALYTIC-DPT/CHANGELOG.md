# CATALYTIC-DPT Changelog

All notable changes to the Catalytic Computing Department (Isolated R&D) will be documented in this file.

## [1.11.0] - 2024-12-24

### Added
- **SPECTRUM-03 Chain Verification**: Temporal integrity across sequences of runs using bundle-only memory.
- **verify_spectrum03_chain() Function**: Verifies chains of SPECTRUM-02 bundles with:
  - Individual bundle verification via `verify_spectrum02_bundle`
  - Output registry construction from `OUTPUT_HASHES.json` keys
  - Reference validation against available outputs (if `references` field present in TASK_SPEC)
  - Chain order enforcement (currently passed order, with TODO for timestamp parsing)
  - No history dependency assertion (verification uses only bundle artifacts)
- **Chain Memory Model**: Defines what persists across runs:
  - Allowed: `run_id`, durable output paths, SHA-256 hashes, validator identity, status
  - Forbidden: logs/, tmp/, chat transcripts, reasoning traces, intermediate state
- **New Error Code**:
  - `INVALID_CHAIN_REFERENCE`: TASK_SPEC references output not produced by earlier run or self
- **Test Suite**: `TESTBENCH/spectrum/test_spectrum03_chain.py` with 23 tests covering:
  - Chain acceptance when all bundles verify
  - Rejection on middle-run tampering (HASH_MISMATCH)
  - Rejection on missing bundle artifacts (BUNDLE_INCOMPLETE)
  - Rejection on invalid output references (INVALID_CHAIN_REFERENCE)
  - No history dependency (acceptance without logs/tmp/transcripts)

### Security Properties
- **Tamper Evidence**: Any modification to outputs after bundle generation detected via hash mismatch
- **Reference Integrity**: Runs cannot claim dependencies on non-existent outputs
- **Temporal Ordering**: Strict ordering maintained (future enhancement for timestamp-based validation)
- **Fail Closed**: Chain verification rejects on any ambiguity; partial acceptance not allowed

## [1.10.0] - 2025-12-24

### Added
- **Validator Version Integrity**: Deterministic build fingerprint binding in OUTPUT_HASHES.json.
- **get_validator_build_id() Function**: Returns audit-grade validator provenance:
  - Preferred: `git:<short-commit>` from repository HEAD
  - Fallback: `file:<sha256-prefix>` of MCP/server.py file hash
  - Cached for process lifetime, deterministic within repo state
- **Enhanced OUTPUT_HASHES.json Schema**:
  - Added `validator_semver` (semantic version of validator)
  - Added `validator_build_id` (deterministic build fingerprint)
  - Renamed from `validator_version` to `validator_semver` for clarity
- **Strict Build ID Verification**: `verify_spectrum02_bundle(strict_build_id=True)` option:
  - Rejects bundles if `validator_build_id` differs from current validator
  - Enables audit-trail and version-lock verification
- **New Error Codes**:
  - `VALIDATOR_BUILD_ID_MISSING`: Build ID missing or empty
  - `VALIDATOR_BUILD_MISMATCH`: Build ID differs from current (strict mode)
- **Test Suite**: `test_validator_version_integrity.py` with 20 tests covering:
  - Bundle emission includes both validator fields
  - Build ID determinism and caching
  - Strict mode rejection on mismatch
  - Missing/empty build ID rejection

### Changed
- `VALIDATOR_VERSION` constant renamed to `VALIDATOR_SEMVER` (semantic clarity)
- `SUPPORTED_VALIDATOR_VERSIONS` renamed to `SUPPORTED_VALIDATOR_SEMVERS`
- Updated all tests to use new field names

## [1.9.0] - 2025-12-24

### Added
- **SPECTRUM-02 Adversarial Resume**: Durable bundle emission for resume without execution history.
- **OUTPUT_HASHES.json Generation**: Automatic generation on successful skill_complete containing:
  - SHA-256 hashes for every declared durable output (files and directory contents).
  - Validator version binding for future compatibility.
  - Posix-style paths relative to PROJECT_ROOT for deterministic resumption.
- **verify_spectrum02_bundle() Method**: Verifies resume bundles checking:
  - Artifact completeness (TASK_SPEC.json, STATUS.json, OUTPUT_HASHES.json).
  - Status validity (status=success, cmp01=pass).
  - Validator version support.
  - Hash integrity across all outputs.
  - Returns structured errors: BUNDLE_INCOMPLETE, STATUS_NOT_SUCCESS, CMP01_NOT_PASS, VALIDATOR_UNSUPPORTED, OUTPUT_MISSING, HASH_MISMATCH.
- **SPECTRUM-02 Specification**: Formal spec at `SPECTRUM/SPECTRUM-02.md` defining:
  - Resume bundle artifact set (minimal, durable-only).
  - Explicitly forbidden artifacts (logs, tmp, transcripts, reasoning traces).
  - Resume rule (verification-only, no history inference).
  - Agent obligations on resume (fail closed, no hallucination).
- **Test Suites**:
  - `TESTBENCH/spectrum/test_spectrum02_resume.py`: 30 tests verifying bundle acceptance, rejection, and no-history-dependency.
  - `TESTBENCH/spectrum/test_spectrum02_emission.py`: 25 integration tests for bundle generation and verification in real MCP flows.

### Fixed
- **Fail Closed on Bundle Generation**: skill_complete now fails if OUTPUT_HASHES.json generation fails, preventing incomplete bundles.

## [1.8.0] - 2025-12-24

### Added
- **CATLAB-01 Implementation**: Released `TESTBENCH/catlab_stress` for proving catalytic temporal integrity.
- **Stress Test Fixture**: `test_catlab_restoration.py` containing deterministic population, mutation, and restoration logic.
- **Restoration Contract**: Validated helper functions (`populate`, `mutate`, `restore`) ensuring byte-identical restoration of catalytic domains.
- **Verification Suite**: 4 tests verifying:
  - Happy path full restoration.
  - Detection of single-byte corruption.
  - Detection of missing files.
  - Detection of rogue extra files.
- **Artifact Generation**: Tests now output `PRE_SNAPSHOT.json`, `POST_SNAPSHOT.json`, and `STATUS.json` for audit capabilities.

## [1.7.0] - 2025-12-24

### Added
- **TASK_SPEC Anti-tamper**: SHA-256 integrity hashing of `TASK_SPEC.json` at execution start, verified at completion.
- **Run Status Persistence**: `STATUS.json` created on completion with `run_id`, `status`, `cmp01` (pass/fail), and timestamp.
- **Symlink Escape Protection**: Added `.resolve()` containment check against `PROJECT_ROOT` to catch escapes via symlinks in allowed roots.
- **Audit-Grade Tests**: Expanded `test_cmp01_validator.py` to 31 tests including hermetic symlink escape proofing and integrity tampering simulation.

### Fixed
- **Index Reporting**: `PATH_OVERLAP` errors now correctly report original `JobSpec` indices in JSON Pointers.
- **Path Semantics**: Exact duplicate paths are now allowed/deduped (Option 1 policy), avoiding overlapping-self false positives.
- **Forbidden Loop Bug**: Post-run validation now correctly breaks on forbidden overlap to avoid redundant/shadow errors for the same entry.
- **Silent Skipping**: Post-run should no longer silenty skip absolute/traversal paths; it now reports `PATH_ESCAPES_REPO_ROOT` or `PATH_CONTAINS_TRAVERSAL`.

## [1.6.0] - 2025-12-24

### Added
- **CMP-01 Path Validation**: Strict path governance in `MCP/server.py`:
  - `_validate_jobspec_paths()`: Pre-execution validation for catalytic_domains and durable_paths.
  - `_verify_post_run_outputs()`: Post-run verification of declared output existence.
  - Component-safe path containment checks using `pathlib.is_relative_to`.
  - Rejection of traversal (`..`), absolute paths, and forbidden root overlaps.
- **Root Constants**: `DURABLE_ROOTS`, `CATALYTIC_ROOTS`, `FORBIDDEN_ROOTS` defined per CMP-01 spec.
- **Structured Error Vectors**: All validation errors now return `{code, message, path, details}` format.
- **Test Suite**: `TESTBENCH/test_cmp01_validator.py` with 9 unit tests covering:
  - Traversal rejection
  - Absolute path rejection
  - Forbidden overlap detection
  - Durable/catalytic root enforcement
  - Nested overlap detection
  - Missing output post-run detection

### Changed
- `execute_skill()` now calls `_validate_jobspec_paths()` before execution.
- `skill_complete()` now calls `_verify_post_run_outputs()` to verify declared outputs exist.

## [1.5.0] - 2025-12-24


### Added
- Official `mcp-builder` skill from `anthropics/skills`.
- Official `skill-creator` skill from `anthropics/skills`.

### Changed
- **Skills Standardization**:
  - All `SKILL.md` files updated to explicitly link to their bundled resources (scripts, assets, and references).
  - `governor` skill now correctly references `GOVERNOR_SOP.json`, `HANDOFF_TO_GOVERNOR.md`, and `GOVERNOR_CONTEXT.md`.
  - `launch-terminal` skill now references `ANTIGRAVITY_BRIDGE.md`.
  - `ant-worker` skill now references `schema.json` and task fixtures.
  - `swarm-orchestrator` skill now references launcher scripts.
  - `file-analyzer` skill now references analysis scripts.
- **Path Correction**: Fixed `HANDOFF_TO_GOVERNOR.md` pointing to non-existent `GOVERNOR_SOP.json` in the root (now correctly points to `assets/`).

### Removed
- `LICENSE.txt` from `skill-creator` and `mcp-builder` (non-functional token bloat).

## [1.4.0] - 2025-12-24

### Changed
- **Skills Reorganization** (agentskills.io spec compliance):
  - All skills now have proper YAML frontmatter in `SKILL.md` (name, description, compatibility).
  - Restructured folder layout: `scripts/`, `assets/`, `references/` per spec.
  - Moved executables to `scripts/` subdirectory:
    - `ant-worker/scripts/run.py`
    - `governor/scripts/run.py`
    - `file-analyzer/scripts/run.py`
    - `launch-terminal/scripts/run.py`
    - `swarm-orchestrator/scripts/poll_and_execute.py`, `poll_tasks.py`, `agent_loop.py`, `launch_swarm.ps1`
  - Moved test fixtures to `assets/` (renamed from `fixtures/`).
  - Moved reference docs to `references/` (GEMINI.md → governor/references/, etc.).
  - Removed redundant skill-level README (use SKILL.md instead).

### Fixed
- MCP stdio_server: Added `skill_run` tool for proper skill execution via MCP.
- Fixed import paths in orchestrator scripts after reorganization.

## [1.3.0] - 2025-12-24

### Added
- `SKILLS/governor/fixtures/phase0_directive.json`: Phase 0 task as structured JSON directive.

### Changed
- Merged `GOVERNOR_CONTEXT.md` into `SKILLS/governor/SKILL.md` (role + MCP commands now in skill).
- Converted `HANDOFF_TO_GOVERNOR.md` to skill fixture format.
- Trimmed `ORCHESTRATION_ARCHITECTURE.md` from 16KB to 7KB (removed verbose examples).
- Updated `README.md` to reflect actual SKILLS folder structure.

### Removed
- `HANDOFF_TO_GOVERNOR.md` (now `SKILLS/governor/fixtures/phase0_directive.json`)
- `GOVERNOR_CONTEXT.md` (merged into `SKILLS/governor/SKILL.md`)
- `PHASE0_IMPLEMENTATION_GUIDE.md` (redundant with `SCHEMAS/README.md`)
- `pop_swarm_terminals.py` (redundant with `launch_swarm.ps1`)

---

## [1.2.0] - 2025-12-24

### Added
- `swarm_config.json`: Centralized configuration for model/role assignments (the "Symbolic Link" for agents).
- `CHANGELOG.md`: This file, to track DPT-specific evolution.

### Changed
- **Hierarchy Generalization**: Refactored the entire department documentation to be model-agnostic.
  - Established hierarchy: **God (User) → President (Orchestrator) → Governor (Manager) → Ants (Executors)**.
  - Replaced hardcoded references to "Claude", "Gemini", and "Grok" with their respective functional roles.
  - Documentation now references `swarm_config.json` for current implementations.
- **File Renaming**:
  - `CODEX_SOP.json` → `GOVERNOR_SOP.json` (Reflecting the new role-based naming).
  - `HANDOFF_TO_CODEX.md` → `HANDOFF_TO_GOVERNOR.md`.
- **Core Documentation**:
  - `ORCHESTRATION_ARCHITECTURE.md`: Updated diagram and roles to reflect the new hierarchy.
  - `GOVERNOR_SOP.json`: Fully generalized instructions for any CLI Manager.
  - `README.md`: Updated directory tree and success criteria.
  - `ROADMAP.md`: Removed model-specific parameter (e.g., "200M") constraints.
  - `TESTBENCH.md`: Updated instructions for the Governor.
  - `GOVERNOR_CONTEXT.md`: Transitioned from "Claude" to "President".

### Cleaned
- Removed legacy "Codex" and "Codex-governed" terminology.
- Deleted redundant index/temporary files from the root of `CATALYTIC-DPT`.

---

## [1.1.0] - 2025-12-24
- Initial setup of Catalytic-DPT directory.
- Defined Phase 0 contracts (JSON Schemas).
- Established first iteration of Multi-Agent Orchestration.

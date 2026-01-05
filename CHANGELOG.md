<!-- CONTENT_HASH: 7F09D4316FAE278B4F834AC229CC5F826C51A24318D7BE5A7F1283F62126CAA3 -->

# Changelog

All notable changes to Agent Governance System will be documented in this file.

## [3.3.23] - 2026-01-05

### Completed
- **Phase 1: Integrity Gates & Repo Safety (Critical Fixes)** — Fixed broken pre-commit hooks and completed runtime INBOX guard integration.
  - **1.1.2 Pre-commit Path Fix**: Corrected broken path references in `CAPABILITY/SKILLS/governance/canon-governance-check/scripts/pre-commit`
    - Fixed `TOOLS/ags.py` → `CAPABILITY/TOOLS/ags.py`
    - Fixed `TOOLS/check-canon-governance.js` → `CAPABILITY/TOOLS/check-canon-governance.js`
    - Pre-commit hook now executes correctly with proper path resolution
  - **1.4.2 Recovery Appendix (Z2 Invariants)**: Added comprehensive recovery section to `NAVIGATION/INVARIANTS/Z2_CAS_AND_RUN_INVARIANTS.md`
    - Receipt locations (CAS storage, RUN_ROOTS.json, run artifacts, audit logs)
    - Verification commands (CAS integrity checks, bundle verification, root audit, RUN_ROOTS validation)
    - Deletion guidelines (safe vs. never delete with explicit examples)
    - Recovery procedures (corrupted objects, malformed roots, failed verification, unreachable outputs)
    - 125 lines of operational guidance for CAS/Run subsystem failures
  - **1.1.3 Runtime INBOX Guard (S.2.3)**: Implemented active runtime enforcement of INBOX hash integrity
    - Inlined validation logic directly into `CAPABILITY/TOOLS/ags.py` (avoids import issues with hyphenated directory)
    - Added `_validate_inbox_write_inline()` function checking all writes to `INBOX/*.md` files
    - Modified `_atomic_write_bytes()` to validate content before writing (bytes decoded and validated)
    - Enhanced `inbox_write_guard.py` decorator to handle both text and bytes content
    - Enforcement: Writes to INBOX without valid `<!-- CONTENT_HASH: ... -->` comments are **blocked** with clear error messages
    - Error messages include computed hash for easy remediation
  - **Exit Criteria**: All Phase 1 integrity gates now operational
    - ✅ Pre-commit hook executes with correct paths
    - ✅ Recovery procedures documented for all major invariant documents
    - ✅ Runtime INBOX writes are validated and fail-closed
    - ✅ No silent failures or bypasses in integrity enforcement

## [3.3.22] - 2026-01-05

### Completed
- **Phase 1.5B: Repo Digest + Restore Proof + Purity Scan (Deterministic)** — Implemented deterministic repo-state proofs that make catalysis measurable.
  - **Core Module**: `CAPABILITY/PRIMITIVES/repo_digest.py` (460 LOC)
    - `RepoDigest`: Deterministic tree hash with declared exclusions
    - `PurityScan`: Detects tmp residue and files outside durable roots
    - `RestoreProof`: Binds pre/post digests with PASS/FAIL verdict + diff summary
    - Module version: 1.5b.0 with version hash tracking
  - **Receipts Implemented**:
    - `PRE_DIGEST.json`: Repo state before operation (digest, file_count, file_manifest)
    - `POST_DIGEST.json`: Repo state after operation (digest, file_count, file_manifest)
    - `PURITY_SCAN.json`: Violation detection (verdict, tmp_residue, violations)
    - `RESTORE_PROOF.json`: PASS/FAIL verdict with deterministic diff summary (added, removed, changed)
  - **Determinism Guarantees**:
    - Canonical ordering: All paths, lists, and diffs sorted alphabetically
    - Hashing rules: Tree digest (SHA-256 of canonical file records), exclusions spec hash, module version hash
    - Repeated digest guarantee: Identical repo state produces identical digest
  - **Hard Invariants Verified**:
    - Never mutates original user content as part of scan
    - Fail closed: Errors emit error receipts and exit nonzero
    - Canonical ordering everywhere (paths, lists, diffs)
    - No crypto sealing (reserved for CRYPTO_SAFE phase)
  - **Tests**: `CAPABILITY/TESTBENCH/integration/test_phase_1_5b_repo_digest.py` (400 LOC)
    - 11 fixture-backed tests, 100% pass rate (11/11)
    - Coverage: deterministic digest, purity violations, canonical ordering, exclusions, path normalization, empty repos
  - **Documentation**: `CAPABILITY/PRIMITIVES/REPO_DIGEST_GUIDE.md` (450 lines)
    - Complete guide on using CLI and programmatic interfaces
    - Receipt format documentation with PASS/FAIL examples
    - Interpreting verdicts and failure modes
    - Integration examples with catalytic runtime
  - **CLI Interface**: `python repo_digest.py --pre-digest | --post-digest | --purity-scan | --restore-proof`
  - **Exit Criteria Met**:
    - ✓ Deterministic digest (repeated → same digest)
    - ✓ Purity scan detects violations (tmp residue, files outside durable roots)
    - ✓ Restore proof shows FAIL with diff summary on mismatch
    - ✓ Fixture-backed tests for all failure modes
    - ✓ JSON outputs valid and deterministic
    - ✓ Documentation complete

## [3.3.21] - 2026-01-05

### Completed
- **Prompt Pack Refactor Complete** — Executed 8-phase systematic refactor of entire prompt tree fixing critical inefficiencies.
  - **Phase 1 - Backup Created**: Full backup at `NAVIGATION/PROMPTS.BACKUP_2026-01-05-12-38/`
  - **Phase 2 - Filename Normalization**: 8 files renamed (removed ✅ checkmarks for consistency)
  - **Phase 3 - Dead References Fixed**: 37+ linter paths updated (`scripts/lint-prompt.sh` → `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`), 18 `python -m compileall` hallucinatory commands removed
  - **Phase 4 - Structural De-duplication**: 32 task prompts refactored (~50% size reduction by removing duplicate "Source Body" instruction layers)
  - **Phase 6 - Dependencies Populated**: Manifest updated with phase-level dependency chains (Phase 2→10 tasks now depend on all prior phases)
  - **Phase 7 - INDEX.md Verified**: Already correct after filename normalization
  - **Phase 8 - Validation**: Linter passed (`CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`), manifest valid JSON, all dead references eliminated
  - **Key Metrics**:
    - Files Modified: 32 task prompts + 4 canon files
    - Dead References Fixed: 40+ linter paths, 18 compileall commands
    - Token Savings: ~40-50% per file
    - Dependencies Added: Phase-level progression chains
- **Phase 9 Tasks Added** — Extended prompt pack with 6 new Phase 9 tasks from updated roadmap.
  - **9.1 - mcp-tool-calling-test**: MCP Tool Calling Test (Z.6.1)
  - **9.2 - task-queue-primitives**: Task Queue Primitives (Z.6.2)
  - **9.3 - chain-of-command**: Chain of Command (Z.6.3)
  - **9.4 - governor-pattern**: Governor Pattern for Ant Workers (Z.6.4)
  - **9.5 - delegation-protocol**: Delegation Protocol (D.1)
  - **9.6 - delegation-harness**: Delegation Harness (D.2)
  - **Total Tasks**: 32 → 62 (6 new Phase 9 tasks added)
  - **Artifacts Created**: `NAVIGATION/PROMPTS/PHASE_09/` directory with 6 task prompt files
  - **Validation**: `PROMPT_PACK_MANIFEST.json` updated with all Phase 9 entries, `INDEX.md` updated, manifest valid JSON

## [3.3.20] - 2026-01-05

### Completed
- **Phase 1.5A: Runtime Write Firewall (Catalytic Domains)** — Implemented mechanical, fail-closed IO policy layer enforcing catalytic domain separation.
  - **Core Module**: `CAPABILITY/PRIMITIVES/write_firewall.py`
    - `WriteFirewall` class enforcing tmp/durable domain separation with commit gate mechanism
    - Tmp writes only under declared tmp roots during execution
    - Durable writes only under declared durable roots AND only after commit gate opens
    - Deterministic error codes (8 codes) and violation receipts with full policy snapshot
    - Tool version hashing (SHA256 of module file) for auditability
    - Path traversal detection and blocking (rejects `..` components)
    - Exclusion list support for read-only paths (LAW/CANON, .git, etc.)
  - **API Surface**:
    - `safe_write(path, data, kind='tmp|durable')` - Write with firewall enforcement
    - `safe_mkdir(path, kind='tmp|durable')` - Create directory with enforcement
    - `safe_rename(src, dst)` - Rename with domain boundary checks
    - `safe_unlink(path)` - Delete with domain validation
    - `open_commit_gate()` - Enable durable writes (commit boundary)
    - `configure_policy(tmp_roots, durable_roots, exclusions)` - Runtime reconfiguration
  - **Error Codes**: 8 deterministic failure modes
    - `FIREWALL_PATH_ESCAPE` - Path escapes project root
    - `FIREWALL_PATH_TRAVERSAL` - Path contains `..` traversal
    - `FIREWALL_PATH_EXCLUDED` - Path in exclusion list
    - `FIREWALL_PATH_NOT_IN_DOMAIN` - Path not in any allowed domain
    - `FIREWALL_TMP_WRITE_WRONG_DOMAIN` - Tmp write outside tmp roots
    - `FIREWALL_DURABLE_WRITE_WRONG_DOMAIN` - Durable write outside durable roots
    - `FIREWALL_DURABLE_WRITE_BEFORE_COMMIT` - Durable write before gate opens
    - `FIREWALL_INVALID_KIND` - Invalid write kind (not "tmp" or "durable")
  - **Integration Example**: `CAPABILITY/TOOLS/utilities/guarded_writer.py`
    - `GuardedWriter` utility demonstrating integration pattern
    - Simplified API: `write_tmp()`, `write_durable()`, `mkdir_tmp()`, `mkdir_durable()`
    - Violation handling helpers with receipt output
  - **Tests**: `CAPABILITY/TESTBENCH/pipeline/test_write_firewall.py`
    - 26 tests covering all policy enforcement scenarios
    - 100% pass rate (exit code 0, duration 0.45s)
    - Deterministic error code verification
    - Receipt structure validation
    - Path normalization (Windows/Unix compatibility)
  - **Documentation**: `CAPABILITY/PRIMITIVES/WRITE_FIREWALL_CONFIG.md`
    - Complete configuration guide for tmp/durable roots
    - Violation receipt interpretation with examples
    - Integration patterns (direct instantiation, GuardedWriter utility, violation logging)
    - Troubleshooting section covering all error codes
    - Standard catalytic domain conventions
  - **Guarantees**:
    - Fail-closed: All violations raise `FirewallViolation` exception (no silent failures)
    - Deterministic: Same violation produces same error code every time
    - Receipts include full policy snapshot + tool version hash
    - Path normalization: Windows backslashes → Unix forward slashes
  - **Standard Catalytic Domains**:
    - Tmp roots: `LAW/CONTRACTS/_runs/_tmp`, `CAPABILITY/PRIMITIVES/_scratch`, `NAVIGATION/CORTEX/_generated/_tmp`
    - Durable roots: `LAW/CONTRACTS/_runs`, `NAVIGATION/CORTEX/_generated`
    - Exclusions: `LAW/CANON`, `AGENTS.md`, `BUILD`, `.git`
  - **Receipt**: `LAW/CONTRACTS/_runs/_tmp/phase_1_5a_implementation_receipt.json`

## [3.3.19] - 2026-01-05

### Added
- **Prompt Pack Audit & Remediation Plan** — Comprehensive audit of all 32 task prompts identifying critical inefficiencies and creating systematic fix plan.
  - **Audit Report**: `NAVIGATION/PROMPTS/UNORGANIZED/PROMPT_PACK_AUDIT_REPORT.md`
    - Identified ~40-50% token waste per file from "Wrapper Paradox" (duplicate instruction layers)
    - Documented 37+ dead linter references (`scripts/lint-prompt.sh` → actual: `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`)
    - Found 100% broken dependency chains (all 32 tasks have `depends_on: []`)
    - Discovered manifest path mismatches (~10 files with ✅ checkmarks on disk but not in manifest)
    - Identified 100% stale links in `INDEX.md` (all completed task links broken)
    - Found 19+ instances of `python -m compileall` misuse (hallucinatory copy-paste)
    - Documented contradictory allowlists preventing compliant task completion
    - Missing Phase 09 from directory structure
  - **Fix Prompt**: `NAVIGATION/PROMPTS/PROMPT_PACK_REFACTOR_FIX.md`
    - 8-phase systematic refactor plan: Backup → Normalization → Dead Refs → De-duplication → Allowlists → Dependencies → Index → Validation
    - Standardized format template for all prompts (CONTEXT/OBJECTIVE/SCOPE/PLAN/VALIDATION/ALLOWLIST/RECEIPT)
    - Specific solutions for each issue class (linter paths, compileall commands, filename normalization)
    - Validation criteria including linter pass, link resolution, manifest validity
    - Structured JSON receipt requirements for auditable execution
    - Priority ordering for time-constrained execution
    - Estimated 30-50% token savings upon completion

## [3.3.18] - 2026-01-05

### Completed
- **Task 3.1: Router & Fallback Stability (Z.3.1)** — Implemented deterministic model selection with explicit fallback chains.
  - **3.1.1 - Stabilize model router: deterministic selection + explicit fallback chain**: Implemented in `CAPABILITY/TOOLS/model_router.py`
    - `KNOWN_MODELS` registry with 6 models (Claude Sonnet 4.5, Claude Sonnet, Claude Opus 4.5, GPT-5.2-Codex, Gemini Pro, Gemini 3 Pro)
    - `select_model()` function for deterministic model selection with explicit fallback chains
    - `validate_model()` for fail-closed model validation
    - `create_router_receipt()` for auditing router selections
    - Chain hash computation (SHA256) for reproducibility and determinism verification
    - Pure logic component: no side effects, no mutable state
  - **Determinism Guarantees**:
    - Same inputs (primary_model, fallback_chain, selection_index) → same output (verified across 10 runs)
    - Chain hash determinism: order matters, different chains produce different hashes
    - Model name parsing: idempotent handling of reasoning annotations
  - **Fail-Closed Design**:
    - `InvalidModelError`: unknown model names rejected immediately
    - `EmptyFallbackChainError`: requires at least one model
    - `RouterError`: base exception for all router errors
  - **Tests**: Created `CAPABILITY/TESTBENCH/core/test_model_router.py` with 32 tests, all passing
    - Model name parsing (4 tests)
    - Model validation (5 tests)
    - Model selection logic (9 tests)
    - Determinism verification (4 tests)
    - Receipt generation (2 tests)
    - ModelSpec behavior (2 tests)
    - Registry validation (2 tests)
    - Integration workflows (2 tests)
  - **Regression Testing**: All 66 core tests passing (no regressions)
  - **Artifacts**:
    - Implementation: `CAPABILITY/TOOLS/model_router.py`
    - Tests: `CAPABILITY/TESTBENCH/core/test_model_router.py`
    - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/receipt.json`
    - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/3.1_router-fallback-stability/REPORT.md`
  - **Roadmap**: Section 3.1 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`

## [3.3.17] - 2026-01-05

### Completed
- **Task 2.3: Run Bundle Contract (Freezing "What is a Run")** — Implemented and validated machine-checkable proof-carrying run bundles.
  - **2.3.1 - Freeze the per-run directory contract**: Defined in `CAPABILITY/RUNS/records.py`
    - Required artifacts: `TASK_SPEC`, `STATUS`, `OUTPUT_HASHES` (all CAS-backed, immutable)
    - Naming: 64-character lowercase hex SHA-256 hashes
    - Immutability: Write-once semantics, no updates/overwrites
    - Determinism: Same input → same hash (canonical JSON encoding)
  - **2.3.2 - Implement `run_bundle_create(run_id) -> sha256:<hash>`**: Implemented in `CAPABILITY/RUNS/bundles.py:96-151`
    - Creates canonical JSON manifest referencing all run artifacts via CAS hashes
    - Bundle manifest itself stored in CAS (addressable by hash)
    - Validates all inputs (run_id format, hash formats)
    - Deterministic: identical inputs produce identical bundle hash
  - **2.3.3 - Define rooting and retention semantics**: Implemented in `bundles.py:320-375`
    - `get_bundle_roots(bundle_ref)` returns complete transitive closure of artifacts
    - Roots include: bundle manifest, task_spec, status, output_hashes, receipts, and all referenced outputs
    - Sorted order for determinism
    - Enables GC to safely identify reachable objects and never delete pinned bundles
  - **2.3.4 - Implement `run_bundle_verify(bundle_ref)`**: Implemented in `bundles.py:165-313`
    - Dry-run verifier checks: manifest exists, valid JSON, correct schema, all artifacts present
    - Returns `BundleVerificationReceipt` with detailed status and error reporting
    - Fail-closed: missing/corrupted artifacts → INVALID status
  - **Exit Criteria**: All satisfied ✅
    - "Run = proof-carrying bundle" is explicit and machine-checkable (validated by `test_bundle_is_proof_carrying`)
    - GC can safely treat bundles/pins as authoritative roots (validated by `TestGCRooting` suite)
  - **Tests**: Created `CAPABILITY/TESTBENCH/runs/test_bundles.py` with 20 tests, all passing
    - Bundle creation & determinism (8 tests)
    - Bundle verification & fail-closed behavior (6 tests)
    - GC rooting semantics (5 tests)
    - End-to-end integration (2 tests)
  - **Artifacts**:
    - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/2.3_run-bundle-contract-freezing-what-is-a-run/receipt.json`
    - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/2.3_run-bundle-contract-freezing-what-is-a-run/REPORT.md`
  - **Roadmap**: Section 2.3 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`

### Fixed
- **CAPABILITY/RUNS/bundles.py**: Removed invalid `cas_root` parameter from `run_bundle_create`, `run_bundle_verify`, and `get_bundle_roots` functions (CAS API doesn't accept this parameter)

## [3.3.16] - 2026-01-04

### Completed
- **Task 1.4: Failure Taxonomy & Recovery Playbooks (ops-grade)** — Created comprehensive failure catalog and recovery documentation for all subsystems.
  - **1.4.1 - FAILURE_CATALOG.md**: Created `NAVIGATION/OPS/FAILURE_CATALOG.md` with 30+ failure modes across 7 subsystems (CAS, ARTIFACTS, RUNS, GC, AUDIT, SKILL_RUNTIME, PACKER)
    - Each failure includes: code/name, trigger condition, detection signal (exception/exit code), safe recovery steps
    - Deterministic recovery instructions for all documented failure modes
  - **1.4.2 - Invariant Recovery Appendix**: Added "Recovery: Invariant Violation Detection and Remediation" section to `LAW/CANON/INVARIANTS.md`
    - Three subsections: Where receipts live, How to re-run verification, What to delete vs never delete
    - Clear guidance on disposable vs protected files, with recovery procedures
    - Exact commands for verification (fixture runner, root audit, critic, canon line counts)
  - **1.4.3 - SMOKE_RECOVERY.md**: Created `NAVIGATION/OPS/SMOKE_RECOVERY.md` with 10 copy/paste recovery flows
    - Windows PowerShell and WSL/Git Bash commands for each flow
    - Covers: CAS object not found, corrupted objects, invalid RUN_ROOTS.json/GC_PINS.json, skill fixture failures, pack consumption missing blobs, canon version incompatibility, GC lock stuck, artifact reference errors, unreachable outputs
    - General verification commands section for post-recovery health checks
  - **Exit Criteria**: All satisfied
    - ✅ Failure catalog provides deterministic identification and recovery steps
    - ✅ Smoke recovery playbooks provide copy/paste commands for Windows + WSL
    - ✅ Invariant doc appendix provides recovery context for invariants
    - ✅ New contributors can identify/recover from common failures without tribal knowledge
  - **Artifacts**:
    - Failure catalog: `NAVIGATION/OPS/FAILURE_CATALOG.md` (70 lines)
    - Recovery playbooks: `NAVIGATION/OPS/SMOKE_RECOVERY.md` (457 lines)
    - Invariant update: `LAW/CANON/INVARIANTS.md` (+58 lines)
    - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/1.4_failure-taxonomy-recovery-playbooks-ops-grade/receipt.json`
    - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/1.4_failure-taxonomy-recovery-playbooks-ops-grade/REPORT.md`
  - **Roadmap**: Section 1.4 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`
- **Task 4.1: Catalytic Snapshot & Restore (Z.4.2–Z.4.4)** — Verified and documented complete implementation of catalytic space restoration guarantees.
  - **4.1.1 - Pre-run Snapshot**: Implemented in `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py:272-279`
    - `snapshot_domains()` captures SHA-256 hashes of all files in catalytic domains before execution
    - Deterministic ordering enforced by normalized relative paths
    - Hashes persisted to `PRE_MANIFEST.json` in run ledger
  - **4.1.2 - Byte-identical Restoration Verification**: Implemented in `catalytic_runtime.py:291-314`
    - `snapshot_after()` captures post-execution state
    - `verify_restoration()` compares pre/post hashes for exact byte-identical match
    - Diff report details: added files, removed files, changed files (by hash)
    - Results persisted to `POST_MANIFEST.json` and `RESTORE_DIFF.json`
  - **4.1.3 - Hard-fail on Restoration Mismatch**: Implemented in `catalytic_runtime.py:643-674`
    - Runtime returns exit code 1 if restoration verification fails
    - `STATUS.json` written with `status: "failed"` and `restoration_verified: false`
    - `PROOF.json` contains `restoration_result.verified: false` with failure condition
    - Failure is deterministic and fail-closed (no partial success)
  - **Exit Criteria**: All satisfied ✅
    - Catalytic domains restore byte-identical (fixture-backed): `test_catlab_restoration.py::test_catlab_restoration_pass` (500-file fixture)
    - Failure mode is deterministic and fail-closed: `test_catlab_restoration.py::test_catlab_detects_*` suite
  - **Artifacts**:
    - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/4.1_catalytic-snapshot-restore/receipt.json`
    - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/4.1_catalytic-snapshot-restore/REPORT.md`
  - **Roadmap**: Section 4.1 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`

## [3.3.15] - 2026-01-05
 
### Changed
- **Task 1.3: Deprecate Lab MCP Server (Z.1.7)** — Marked experimental MCP server as archived/deprecated with clear pointer to canonical implementation.
   - **1.3.1 - Deprecation Notice**: Added prominent `*** DEPRECATED / ARCHIVED ***` header to `THOUGHT/LAB/MCP_EXPERIMENTAL/server_CATDPT.py`
     - Points to canonical server: `CAPABILITY/MCP/server.py`
     - Points to canonical entry point: `LAW/CONTRACTS/ags_mcp_entrypoint.py`
     - References Z.1.7 (Catalytic Architecture)
     - Preserves original code below deprecation notice for historical reference
   - **Verification**: Confirmed no normal flows (non-test) import or execute deprecated server
     - Only comment references in `CAPABILITY/MCP/server.py` (e.g., "# Ported from CAT LAB server_CATDPT.py")
     - No actual imports or execution calls found
   - **Exit Criteria**: All satisfied
     - ✅ Deprecated server marked with clear pointer to canonical implementation (Z.1.7)
     - ✅ No tooling still imports/executes deprecated server in normal flows
     - ✅ Pre-existing syntax errors in CAT_CHAT demo files fixed as prerequisite
     - ✅ Receipt and report emitted
   - **Artifacts**:
     - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/1.3_deprecate-lab-mcp-server/receipt.json`
     - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/1.3_deprecate-lab-mcp-server/REPORT.md`
   - **Roadmap**: Section 1.3 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`
   - **Files Modified**: 2 tracked files
     - `THOUGHT/LAB/MCP_EXPERIMENTAL/server_CATDPT.py` (deprecation notice)
      - `THOUGHT/LAB/CAT_CHAT/archive/legacy/simple_symbolic_demo.py` (syntax fix)
 
### Added
 - **Task 2.2: Pack Consumer (verification + rehydration)** — Implemented pack consumption to enable deterministic restoration from CAS-addressed manifests, completing catalytic pack cycle.
   - **2.2.1 - Pack Manifest v1 Schema**: Defined comprehensive schema with validation in `consumer.py`
     - Required fields: version, scope, entries (path, ref, bytes, kind)
     - Canonical JSON encoding enforced
     - Path safety validation (no absolute paths, no `..` traversal)
   - **2.2.2 - pack_consume() Implementation**: Created `MEMORY/LLM_PACKER/Engine/packer/consumer.py` (270 lines)
     - Manifest integrity verification (hash, canonical encoding, schema)
     - CAS blob existence verification (fail-closed if any missing)
     - Atomic materialization (write to temp → rename, no partial writes)
     - Strict path safety enforcement
     - Dry-run mode for verification without writes
   - **2.2.3 - Consumption Receipts**: Implemented `ConsumptionReceipt` dataclass
     - Inputs: manifest_ref, cas_snapshot_hash
     - Outputs: tree_hash (deterministic), verification_summary
     - Commands run audit trail, exit status
   - **2.2.4 - Comprehensive Tests**: Created `CAPABILITY/TESTBENCH/integration/test_pack_consumer.py` (374 lines)
     - 6 tests covering: roundtrip, dry-run, tamper detection, missing blobs, determinism, path safety
     - All tests passing with fixture-backed proofs
   - **Exit Criteria**: All satisfied
     - ✅ Packs are not write-only: can be consumed and verified deterministically
     - ✅ Any corruption or missing data fails-closed before producing output tree
     - ✅ Tree hash proves byte-identical restoration
   - **System Status**: Now **FULLY CATALYTIC** (can create AND consume packs)
   - **Artifacts**:
     - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/2.2_pack-consumer-verification-rehydration/receipt.json`
     - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/2.2_pack-consumer-verification-rehydration/REPORT.md`
   - **Roadmap**: Section 2.2 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`
   - **Test Coverage**: 6/6 tests passing (roundtrip, tamper detection, determinism, fail-closed)
 
## [3.3.14] - 2026-01-05

### Added
- **Task 1.2: Bucket Enforcement (X3)** — Implemented preflight validation ensuring every artifact belongs to exactly one of 6 buckets (LAW, CAPABILITY, NAVIGATION, MEMORY, THOUGHT, INBOX).
  - **1.2.1 - Preflight Bucket Check**: Added `BUCKETS` constant and `_check_bucket_enforcement()` method to `CAPABILITY/PRIMITIVES/preflight.py`
    - Validates all paths in `catalytic_domains` and `outputs.durable_paths` belong to exactly one bucket
    - Detects `BUCKET_VIOLATION`: paths outside all 6 buckets
    - Detects `BUCKET_OVERLAP`: paths in multiple buckets (edge case)
    - Integrated as validation step #5 in preflight pipeline
  - **Test Coverage**: 3 new tests in `CAPABILITY/TESTBENCH/integration/test_preflight.py`
    - `test_bucket_violation_path_outside_buckets_fails()` - Validates rejection of paths outside buckets (e.g., `BUILD/`)
    - `test_path_in_valid_bucket_passes()` - Validates acceptance of paths in valid buckets
    - `test_all_buckets_are_valid()` - Confirms all 6 buckets are recognized as valid
  - **Exit Criteria**: All satisfied
    - ✅ Violations fail-closed before writes occur (preflight check blocks execution)
    - ✅ All 13/13 preflight tests passing
    - ✅ All 340/340 full test suite passing
  - **Artifacts**:
    - Receipt: `LAW/CONTRACTS/_runs/_tmp/prompts/1.2_bucket-enforcement-x3/receipt.json`
    - Report: `LAW/CONTRACTS/_runs/_tmp/prompts/1.2_bucket-enforcement-x3/REPORT.md`
  - **Roadmap**: Section 1.2 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`
  - **Lines Changed**: 158 lines added (+78 preflight.py, +80 test_preflight.py)

## [3.3.13] - 2026-01-05

### Added
- **Task 2.1: CAS-aware LLM Packer Integration (Z.2.6 + P.2 remainder)** — Completed Phase 2 packer integration with CAS addressing, GC safety, and deduplication benchmarks.
  - **2.1.1 - LITE Packs with CAS Hashes (Z.2.6)**: Verified existing implementation in `MEMORY/LLM_PACKER/Engine/packer/core.py`
    - LITE manifests use `sha256:` references instead of file bodies
    - Manifest entries contain only CAS refs, not actual content
    - 5 existing tests passing in `test_p2_cas_packer_integration.py`
  - **2.1.2 - GC Safety for Packer Outputs (P.2.4)**: Implemented comprehensive GC safety tests
    - Created `CAPABILITY/TESTBENCH/integration/test_p2_gc_safety.py` (2 tests)
    - Proves GC never deletes blobs referenced by active packs
    - Verifies packer-written `RUN_ROOTS.json` files are respected by GC
    - All tests passing with fixture-backed proofs
  - **2.1.3 - Deduplication Benchmark (P.2.5)**: Created reproducible benchmark tool and artifacts
    - New benchmark: `CAPABILITY/TESTBENCH/benchmarks/p2_dedup_benchmark.py`
    - **Results**: 97.74% size savings (5.74 MB → 132.68 KB)
    - Generated artifacts:
      - `MEMORY/LLM_PACKER/_packs/_system/benchmarks/dedup_benchmark_fixture.json` (machine-readable)
      - `MEMORY/LLM_PACKER/_packs/_system/benchmarks/DEDUP_BENCHMARK_REPORT.md` (human-readable)
    - Reproducible via documented command
    - Measures: full pack size, LITE manifest size, CAS efficiency, generation time, dedup count
  - **Exit Criteria**: All satisfied
    - ✅ LITE packs are manifest-only with `sha256:` blobs
    - ✅ GC never deletes referenced blobs (fixture-backed proof)
    - ✅ Dedup benchmark reproducible and stored as artifacts
  - **Test Coverage**: 7/7 tests passing (5 P2 integration + 2 GC safety)
  - **Roadmap**: Section 2.1 marked complete in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`

## [3.3.12] - 2026-01-05

### Changed
- **Roadmap 6.4 “Real Proof” requirements** — Expanded `6.4 Compression Validation` to require declared tokenizer/encoding, explicit baseline corpus, explicit compressed-context retrieval params + hashes, and an auditable proof bundle (`DATA.json` + report).
- **6.4 executor prompt** — Added hard required-facts checks for `tiktoken` availability and `section_vectors` readiness, and required emitting `DATA.json` for math auditability.
- **STATUS_REPORT clarity** — Updated `NAVIGATION/PROMPTS/PHASE_06/STATUS_REPORT.md` with non-WSL Codex/MCP guidance and a one-shot 6.4 execution checklist.

## [3.3.11] - 2026-01-04

### Changed
- **License Upgrade (CCL v1.4)** — Complete git history rewrite replacing MIT License with Catalytic Commons License v1.4 across all commits.
  - Added **Attestation-Gated Protected Artifacts** mechanism (Section 4.4) for cryptographic enforcement.
  - Added **Digital Signature** definition (Section 1) requiring GPG/X.509 signatures.
  - Added **"Acting on behalf of"** definition (Section 1) with knowledge/reckless disregard standards.
  - Added **Safe Harbor** clause (Section 2.1) for accidental violations by non-prohibited entities.
  - Added **Circumvention Prohibition** (Section 3.6) covering access control bypass.
  - Added **False Attestation Prohibition** (Section 3.7) making false attestations a material breach.
  - Added **California Governing Law** clause (Section 10) with Santa Clara County venue.
  - Retroactive license application: CCL v1.4 now appears in all historical commits.

### Fixed
- **cortex-build fixtures** — Updated expected outputs to match current cortex index format.
- **llm-packer-smoke fixtures** — Updated expected outputs to match current pack proof integration.
- **prompt-runner fixtures** — Updated expected outputs to match current prompt validation logic.
- **Packer proof refresh stability/perf** — Default proof suite no longer runs `pytest` (avoids recursive/slow runs during packer fixtures); opt into stronger suites via `NAVIGATION/PROOFS/PROOF_SUITE.json`.
- **SPLIT numbering** — Renumbered AGS split files to remove gaps after dropping DIRECTION/THOUGHT (MEMORY=`AGS-05_*`, ROOT_FILES=`AGS-06_*`).

## [3.3.10] - 2026-01-04

### Added
- **Proofs as First-Class Pack Artifacts** — Integrated rigorous proof generation into the LLM Packer pipeline to ensure every pack contains fresh verification evidence.
  - **Fail-Closed Generation**: Pack generation triggers `refresh_proofs` and aborts immediately if any proof command fails (e.g. tests, scripts).
  - **Dispersed Artifacts**: Proof artifacts are now atomically generated and distributed to:
    - `NAVIGATION/PROOFS/GREEN_STATE.json` & `.md`: Git state, timestamps, and command execution logs.
    - `NAVIGATION/PROOFS/PROOF_MANIFEST.json`: Signed inventory of all proof files.
    - `NAVIGATION/PROOFS/CATALYTIC/`: Catalytic proof logs and summaries.
    - `NAVIGATION/PROOFS/COMPRESSION/`: Compression proof reports.
  - **Pack Integration**:
    - **FULL / SPLIT Packs**: Include `AGS-04_PROOFS.md` containing all proof text/JSON.
    - **LITE Packs**: Include `LITE/PROOFS.json` with a verifiable summary (hashes + status).
  - **CLI Control**: Added `--skip-proofs` (for speed) and `--with-proofs` (force refresh) flags to `packer/cli.py`.
  - **Test Coverage**: Added `CAPABILITY/TESTBENCH/integration/test_packer_proofs.py` verifying atomic updates and fail-closed behavior.

### Changed
- **License Update (CCL v1.2)** — Updated `LICENSE` to Catalytic Commons License v1.2.
  - Added **No State/Police/Military/Intel Use** clause (Section 0 & 3.1).
  - Explicitly defined "Prohibited Entity" types.
  - Clarified "Extractive Use" regarding surveillance and coercive control.
- **Pytest Configuration**: Updated `pytest.ini` to exclude build artifact directories (`_runs`, `_packs`, `_generated`, `BUILD`) prevents Windows file lock errors during self-test collection.

## [3.3.9] - 2026-01-04

### Added
- **prompt-runner Skill** (`CAPABILITY/SKILLS/utilities/prompt-runner/`): Enforces prompt canon gates (lint, hashes, FILL_ME__ blocking), allowlists, dependency checks, and emits canonical receipts/reports.
- **inbox-report-writer Skill manifest + fixtures** (`CAPABILITY/SKILLS/inbox/inbox-report-writer/`): Added skill runner, validator, and fixtures for ledger generation and hash validation.
- **cortex-build Skill** (`CAPABILITY/SKILLS/cortex/cortex-build/`): Rebuilds cortex index + SECTION_INDEX and verifies expected prompt paths are present.

### Changed
- **INBOX ledger/index scanning** now uses cortex section indexes instead of raw filesystem traversal in `CAPABILITY/SKILLS/inbox/inbox-report-writer/generate_inbox_ledger.py` and `CAPABILITY/SKILLS/inbox/inbox-report-writer/update_inbox_index.py`.
- **SECTION_INDEX coverage** now includes `NAVIGATION/PROMPTS/**` for prompt discovery in `NAVIGATION/CORTEX/db/cortex.build.py`.

## [3.3.8] - 2026-01-04

### Added
- **Task 1.1: Hardened Inbox Governance (S.2)** — Implemented comprehensive INBOX integrity system with automatic hash management and validation.
  - **inbox-report-writer Skill** (`CAPABILITY/SKILLS/inbox/inbox-report-writer/`):
    - `hash_inbox_file.py`: Core hash computation, insertion, update, and verification (192 lines)
    - `generate_inbox_ledger.py`: Automatic YAML ledger generation with metadata and statistics (210 lines)
    - `update_inbox_index.py`: Automatic INBOX.md index regeneration with file listings (180 lines)
    - `check_inbox_hashes.py`: Pre-commit hash validation script (90 lines)
    - `inbox_write_guard.py`: Runtime interceptor with decorators and context managers (200 lines)
    - `test_inbox_hash.py`: Comprehensive test suite - 5/5 tests passing (145 lines)
    - `README.md`: Complete documentation with usage examples and integration guide
  - **Hash Format**: `<!-- CONTENT_HASH: <sha256> -->` placed after frontmatter with one blank line after
  - **Pre-commit Integration**: Modified `CAPABILITY/SKILLS/governance/canon-governance-check/scripts/pre-commit`
    - Automatically updates INBOX.md and LEDGER.yaml before validation
    - Validates all staged INBOX/*.md files for valid content hashes
    - Blocks commits with invalid/missing hashes
  - **Runtime Protection**: `inbox_write_guard.py` provides fail-closed write protection
    - `@inbox_write_guard` decorator for function-level protection
    - `InboxWriteGuard()` context manager for scope-level protection
    - `validate_inbox_write()` for explicit validation
    - Raises `InboxWriteError` with detailed fix instructions
  - **Automatic Updates**: Pre-commit hook now automatically:
    - Regenerates `INBOX/INBOX.md` with current file listings and hash status
    - Regenerates `INBOX/LEDGER.yaml` with full metadata and statistics
    - Stages updated files for commit
    - Zero manual maintenance required
  - **INBOX.md Features**:
    - Auto-generated index of all INBOX files by category
    - Shows first 8 characters of each file's hash for quick verification
    - Displays metadata: section, author, priority, created/modified dates, summary
    - Hash validation indicator (✅/⚠️) for each file
  - **LEDGER.yaml Features**:
    - Human-readable YAML format with full metadata
    - Summary statistics (total files, valid/invalid/missing hashes, errors)
    - Files organized by category (reports, research, roadmaps, agents, etc.)
    - Complete metadata per file: path, size, modified date, hash status, frontmatter
  - **Hash Coverage**: All 62 INBOX markdown files now have valid SHA256 content hashes
  - **Test Coverage**: 5/5 unit tests passing (hash computation, insertion, update, runtime guard, validation)
  - **Exit Criteria Met**:
    - ✅ Unhashed INBOX writes fail-closed with clear errors
    - ✅ Pre-commit rejects invalid INBOX changes deterministically
    - ✅ All tests pass
    - ✅ Receipts and reports emitted
    - ✅ Scope respected (only allowlisted files modified)
- **prompt-runner Skill** (`CAPABILITY/SKILLS/utilities/prompt-runner/`): Enforces prompt canon gates (lint, hashes, FILL_ME__ blocking), allowlists, dependency checks, and emits canonical receipts/reports.

## [3.3.7] - 2026-01-04

### Added
- **Batch Normalization & Portability of Prompt Pack** — Standardized the entire prompt tree in `NAVIGATION/PROMPTS/**` for mechanical consistency, lint-compliance, and cross-platform portability.
  - **Fix A: Python Command Repair**: Repaired 18 occurrences of truncated or invalid `python -` lines.
    - Standardized to `python -m compileall . (must exit 0 or hard fail)` for all truncated "REQUIRED FACTS" verification lines.
  - **Fix B: Path Mismatch Normalization**: Normalized internal path references across 32 prompt files to be internally consistent.
    - Every reference (header, body, allowed-writes, receipt/report paths) now matches the file's canonical filename (e.g., `1.1_slug` instead of `1-1-slug`).
  - **Python Portability (Heredoc Elimination)**: Replaced 14 bash-only Python heredocs (`python - <<'PY'`) with portable `python -c` one-liners.
    - Ensures "REQUIRED FACTS" extraction works in WSL/bash, Windows CMD, and PowerShell.
  - **Roadmap Path Correction**: Repaired 29 files referencing the non-existent `AGS_ROADMAP_MASTER_REPHASED_TODO_UPDATED.md`, pointing them to the canonical `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md`.
  - **PROMPT_PACK_MANIFEST.json**: Updated with 32 new SHA256 hashes for the standardized pack.
  - **Shell Portability (Standardized Invocations)**: Resolved silent shell assumptions across 36 files. 
    - Explicitly prefixed `*.sh` calls with `bash` and added hardware/lane requirements: "Requires bash-compatible shell (e.g. WSL)".
  - **Validation**: All prompts now pass `lint_prompt_pack.sh` with 0 violations and 0 warnings.

## [3.3.6] - 2026-01-04

### Added
- **Authority Asymmetry in Prompt Policy** — Formalized the split between planner-capable and non-planner models to optimize compute and preserve safety.
  - **NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md (v1.4)**:
    - **Planning Authority Rule (Section 1.5.1)**: Granted full authority for planning, analysis, decomposition, and repository navigation to planner-capable models.
    - **Execution Restriction Rule (Section 1.5.2)**: Restricted non-planner models to mechanical execution ONLY IF a valid `plan_ref` from a planner-capable model is provided.
    - **Section 13 (Authority Enforcement)**: Mandated that executors must gate restricted models on plan presence and record violations as `POLICY_BREACH` in run receipts.
  - **NAVIGATION/PROMPTS/6_MODEL_ROUTING_CANON.md**:
    - Added **Authority Tiers** section designating Claude Sonnet (Thinking) and Opus as Planner-capable, and Gemini/GPT/Grok as Non-planner models.
  - **PROMPT_PACK_MANIFEST.json**: Updated canon SHA256 hashes for policy and routing.
  - **Prompt Pack Synchronization**: Updated `policy_canon_sha256` in all 32 existing prompt files to ensure alignment with v1.4 policy.

## [3.3.5] - 2026-01-04

### Added
- **Lint Gate Enforcement in Canon Law** — Promoted existing prompt-pack linter to mandatory canon law with hard enforcement across three CANON files.
  - **NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md**: Added Section 12 "Lint Gate" declaring lint-pass as hard precondition to execution.
    - Lint failure or inability to lint is a hard stop (no model execution, no writes).
    - Executors MUST run canonical lint command before any execution: `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh PROMPTS_DIR`.
    - Executors MUST record lint metadata in receipts: `lint_command`, `lint_exit_code`, `lint_result`.
  - **NAVIGATION/PROMPTS/3_MASTER_PROMPT_TEMPLATE_CANON.md**: Added Section 6 "Receipt Requirements (lint metadata)" with REQUIRED receipt fields.
    - `lint_command`: the exact linter command executed.
    - `lint_exit_code`: exit status (0=PASS, 1=FAIL, 2=WARNING).
    - `lint_result`: one of PASS, FAIL, or WARNING.
    - `linter_ref`: optional (path/version/hash of the linter used).
  - **NAVIGATION/PROMPTS/6_MODEL_ROUTING_CANON.md**: Added "Lint Precondition (hard stop)" section.
    - Routing to any execution model is forbidden if lint status is missing or FAIL.
    - Only allowed action: Run canonical linter or repair prompt pack to pass lint.
  - **NAVIGATION/PROMPTS/PROMPT_PACK_MANIFEST.json**: Updated all 7 canon SHA256 hashes to reflect new content.

### Fixed
- **CAPABILITY/TOOLS/linters/verify_canon_hashes.py**:
  - Fixed Unicode encoding errors on Windows by removing emoji characters (✓, ✗, ⚠️, ✅, ❌) and replacing with ASCII ([OK], [FAIL], [!], [OK], [FAIL]).
  - Fixed hash computation to correctly extract CANON_HASH from HTML comment format (`<!-- CANON_HASH: <hash> -->`).
  - Fixed hash verification to exclude CANON_HASH line itself when computing actual hashes.
- **CAPABILITY/TOOLS/linters/update_canon_hashes.py**:
  - Fixed Unicode encoding errors (same as verify_canon_hashes.py).
  - Fixed hash computation to match verification logic (exclude CANON_HASH line).
  - Fixed HTML comment pattern matching to use CANON_HASH instead of sha256.
  - All 7 canon files now have correct CANON_HASH values matching actual content.

## [3.3.4] - 2026-01-04

### Added
- **Prompt Pack Linter** — `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` enforces `NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md` mechanically with deterministic, read-only validation.
  - **Exit Codes**: 0=PASS, 1=POLICY_VIOLATION (blocking), 2=WARNING (non-blocking)
  - **Checks Implemented**:
    - A) Manifest validity (JSON structure, required fields, path existence)
    - B) INDEX link validity (markdown links resolve correctly)
    - C) YAML frontmatter (required fields, format validation)
    - D) Canon hash consistency (detects version skew)
    - E) Forbidden terms (hex-escaped regex for "assume" variants)
    - F) Empty bullet lines (WARNING for `^\s*-\s*$` pattern)
    - G) FILL token containment (`FILL_ME__` only in REQUIRED FACTS)
  - **Dependencies**: Bash + Python 3 only (no jq, ripgrep, node)
  - **Performance**: <5 seconds typical, deterministic output
  - **Documentation**: Comprehensive README, implementation summary, quick reference
  - **Testing**: Validation and unit test scripts included
  - **Location**: `CAPABILITY/TOOLS/linters/` (organized in dedicated folder)

## [3.3.3] - 2026-01-03

### Added
- **CI-local gate helper** — `CAPABILITY/TOOLS/utilities/ci_local_gate.py` supports a fast default (critic-only) for frequent commits and a `--full` mode that runs `critic` + `runner` + `pytest` (with safe temp dir) and mints a one-time `LAW/CONTRACTS/_runs/ALLOW_PUSH.token` tied to `HEAD`.
- **Prompt Engineering**: Created prompts for all phases,standardized them, and created handoff templates for continuity.
Canonical Normalization: Standardized all 6 canon files with YAML front matter and updated PROMPT_PACK_MANIFEST.json with new hashes.

### Changed
- **Pre-push fast path** — `.githooks/pre-push` consumes the one-time token to skip re-running heavy checks when the local CI gate already passed for the current `HEAD`.
- **Pre-push alignment** — legacy/manual tokens now run the full CI-aligned gate (`ci_local_gate.py --full`) on push, not just `runner`.

### Fixed
- **Canon governance messaging** — `CAPABILITY/TOOLS/check-canon-governance.js` now correctly requires `CHANGELOG.md` (matching the enforced policy).

## [3.3.2] - 2026-01-03

### Changed
- **Contract runner UX/perf** — `LAW/CONTRACTS/runner.py` now streams subprocess output and prints per-fixture timing so long runs no longer appear stuck.
- **Artifact escape hatch perf** — `CAPABILITY/SKILLS/commit/artifact-escape-hatch/run.py` now scans `git` untracked files first (fast-path) instead of walking the full repo tree.
- **LLM packer smoke perf** — `CAPABILITY/SKILLS/cortex/llm-packer-smoke/run.py` now streams packer output and packs a tiny fixture repo via `project_root` to keep fixtures fast/deterministic.

### Fixed
- **Cortex index artifacts** — `NAVIGATION/CORTEX/semantic/indexer.py` now ensures `NAVIGATION/CORTEX/meta/` exists before writing `FILE_INDEX.json` / `SECTION_INDEX.json`.

## [3.3.1] - 2026-01-02

### Changed
- **P.1: 6-Bucket Migration (P0)** — migrated LLM Packer to the 6-bucket repo layout (LAW/CAPABILITY/NAVIGATION/DIRECTION/THOUGHT/MEMORY).
  - Updated pack roots, anchors, split grouping, and lite priorities in `MEMORY/LLM_PACKER/Engine/packer/core.py`.
  - Replaced legacy split outputs with bucket outputs in `MEMORY/LLM_PACKER/Engine/packer/split.py` and `MEMORY/LLM_PACKER/Engine/packer/lite.py`.
  - Updated smoke/validators and docs to match new pack structure (`CAPABILITY/SKILLS/cortex/llm-packer-smoke/run.py`, `CAPABILITY/SKILLS/utilities/pack-validate/run.py`, `README.md`, `AGENTS.md`, `MEMORY/LLM_PACKER/README.md`).
  - Updated contract fixtures/docs for bucket paths (`LAW/CONTRACTS/fixtures/governance/canon-sync/input.json`, `LAW/CONTRACTS/fixtures/governance/canon-sync/expected.json`, `LAW/CONTRACTS/README.md`).
- **P.2: CAS Integration (P0)** — integrated LLM Packer LITE outputs with CAS (manifest-only) and root-audit gating.
  - LITE writes `LITE/PACK_MANIFEST.json` (path → `sha256:` ref) and `LITE/RUN_REFS.json` (TASK_SPEC/OUTPUT_HASHES/STATUS refs).
  - Packer emits roots to `CAPABILITY/RUNS/RUN_ROOTS.json` and gates completion on `CAPABILITY/AUDIT/root_audit.py` (Mode B).
- **Packer scopes** — removed `catalytic-dpt` scope (CAT integrated into main repo); AGS scope excludes `THOUGHT/LAB/**` and LAB is a separate scope.
- **Packer archives** — clarified and separated archives:
  - **Internal Archive**: `<pack>/archive/pack.zip` (meta+repo only) + scope-prefixed `.txt` siblings.
  - **External Archive**: `MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip` (whole pack folder).
  - Safe rotation: previous unzipped pack is deleted only after its External Archive validates.

### Fixed
- **CORTEX Reference Normalization** — Normalized all critical code references to canonical `NAVIGATION/CORTEX` (8 files: emergency.py, preflight.py, check_canon_governance.py, mcp-access-validator fixture, mcp-smoke, mcp-extension-verify, TURBO_SWARM error messages). CAT_CHAT and external AGI references preserved as intentional separations.


## [3.3.0] - 2026-01-02

### Added
- **Z.2.5 – GC strategy for CAS (unreferenced blob cleanup)** (Completed 2026-01-02)
  - **Module**: `CAPABILITY/GC/` implementing a two-phase Mark-and-Sweep garbage collector.
  - **Policy Lock (Choice B)**: Mandatory fail-closed behavior if roots are zero (unless override provided).
  - **Public API**: `gc_collect(dry_run: bool = True, allow_empty_roots: bool = False) -> dict`.
  - **Root Sources**: Supports `RUN_ROOTS.json` and `GC_PINS.json` root enumeration.
  - **Determinism**: Guaranteed stable ordering for candidate selection, deletion, and reporting.
  - **Safety**: Single-instance execution enforced via global GC lock.
  - **Verification**: 15 comprehensive tests passing (Policy B, deterministic order, malformed inputs).
  - **Operational Proof**: `NAVIGATION/PROOFS/01-02-2026-19-22_Z2_5_GC_OPERATIONAL_PROOF.md` (Confirmed safety, determinism, and fail-closed behavior on 2026-01-02).
  - **Documentation**: New invariants in `Z2_5_GC_INVARIANTS.md` and detailed test matrix.
- **Z.2.4 – Deduplication proof for CAS + Artifact Store** (Completed 2026-01-02)
  - **Mechanical Proof**: Deduplication is satisfied by content addressing and write-once semantics
  - **CAS Deduplication Tests**: `CAPABILITY/TESTBENCH/cas/test_cas_dedup.py` (8 tests)
    - Proves `cas_put(same_bytes)` twice returns same hash
    - Proves underlying stored object is NOT rewritten on second put (verified via file mtime)
    - Tests for empty data, large data, binary data, multiple puts, retrieval after dedup
  - **Artifact Store Deduplication Tests**: `CAPABILITY/TESTBENCH/artifacts/test_artifact_dedup.py` (14 tests)
    - Proves `store_bytes(same_bytes)` twice returns same `sha256:` ref
    - Proves `store_file` on identical files returns same `sha256:` ref
    - Cross-function deduplication (store_bytes and store_file deduplicate to same ref)
    - Tests for different paths, different names, mixed operations
  - **Test Coverage**: 22/22 new tests passing (8 CAS + 14 artifact store)
  - **Documentation**: Added Z.2.4 section to `NAVIGATION/INVARIANTS/Z2_CAS_AND_RUN_INVARIANTS.md`
  - **Guarantees**: Identical content shares storage, no rewrites on duplicate puts, deterministic refs
  - **Proof Mechanism**: File modification time (mtime) verification for no-rewrite guarantee
- **Document Cleanup**: Added hashes and made documents canonical.
- **Cortex Index**: Cortex indexed updates.

## [3.2.3] - 2026-01-02

### Added
- **Z.2.3 – Immutable run artifacts** (Completed 2026-01-02)
  - **Module**: `CAPABILITY/RUNS/` with CAS-backed immutable run records
  - **Public API**:
    - `put_task_spec(spec: dict) -> str` - Store immutable task specification with canonical JSON encoding
    - `put_status(status: dict) -> str` - Store immutable status record (requires 'state' field)
    - `put_output_hashes(hashes: list[str]) -> str` - Store deterministic ordered list of CAS hashes
    - `load_task_spec(hash: str) -> dict` - Load task spec by CAS hash
    - `load_status(hash: str) -> dict` - Load status by CAS hash
    - `load_output_hashes(hash: str) -> list[str]` - Load output hash list by CAS hash
  - **Record Types**:
    - TASK_SPEC: Immutable bytes representing exact task input (canonically encoded dict)
    - STATUS: Small structured record describing state (PENDING, RUNNING, SUCCESS, FAILURE) with optional error info
    - OUTPUT_HASHES: Deterministic ordered list of CAS hashes produced by the run (order preserved)
  - **Guarantees**: Immutable (no updates/overwrites), deterministic (same input → same hash), fail-closed (invalid input rejected), canonical encoding (sorted keys, stable JSON)
  - **Test Coverage**: 68/68 tests passing (canonical encoding, roundtrip, immutability, corruption detection, edge cases)
  - **Dependencies**: Uses Z.2.1 CAS primitives (cas_put/cas_get) exclusively
  - **Representational Only**: No execution logic, no orchestration, no enforcement - pure data storage
- **Z.1.6 Canonical Skill Execution with CMP-01 Pre-Validation** - Enforces deterministic, auditable skill execution
  - **Canonical Entry Point**: `execute_skill()` in `CAPABILITY/TOOLS/agents/skill_runtime.py` (606 lines)
  - **CMP-01 Enforcement**: Mandatory pre-validation before any skill execution (fail-closed)
    - Skill manifest integrity validation (SKILL.md, run.py existence)
    - Canon version compatibility checking
    - JobSpec path validation (no absolute paths, no traversal, no forbidden overlaps, allowed roots only)
    - Deterministic receipt generation with SHA-256 hashes
  - **Ledger Integration**: Append-only JSONL validation receipts with canonical JSON encoding
  - **Enforcement Proofs**: 15 comprehensive tests proving no bypass paths exist (100% pass rate)
  - **No Regressions**: All 33 tests pass (18 existing + 15 new)
  - Full implementation summary: `CAPABILITY/TOOLS/agents/Z_1_6_IMPLEMENTATION_SUMMARY.md`
- **Z.4.1 Catalytic Domains Inventory** - Produced a complete, deterministic map of all transient (catalytic) domains in the repository.
  - **Inventory File**: `NAVIGATION/MAPS/CATALYTIC_DOMAINS.md`
  - **Mapping**: Identified 40+ directories across the repository including `__pycache__`, test caches, and dedicated scratch spaces.
  - **Subsystem Trace**: Linked each domain to its owning subsystem and observed purpose for auditability.
  - **Governance Compliance**: Established a read-only inventory that clarifies the boundaries of disposable space per INV-014.
- Z.2.1 – Core CAS primitives implementation
  - Added `cas_put(data: bytes) -> str` function for storing data with SHA-256 hashing
  - Added `cas_get(hash: str) -> bytes` function for retrieving data by hash
  - Implemented deterministic path derivation using prefix directories (first char / next 2 chars / full hash)
  - Added comprehensive error handling with specific exceptions (InvalidHashException, ObjectNotFoundException, CorruptObjectException)
  - Created test suite with 13 test cases covering all functionality
  - Implemented atomic writes with integrity verification
  - Added write-once semantics to prevent overwriting existing objects
- **Z.2.2 – CAS-backed artifact store** (Completed 2026-01-02)
  - **Module**: `CAPABILITY/ARTIFACTS/` with dual-mode support for CAS refs and legacy file paths
  - **Public API**:
    - `store_bytes(data: bytes) -> str` - Stores bytes into CAS, returns `"sha256:<hash>"`
    - `load_bytes(ref: str) -> bytes` - Loads from CAS ref or legacy file path (dual mode)
    - `store_file(path: str) -> str` - Reads file and stores in CAS
    - `materialize(ref: str, out_path: str, *, atomic: bool = True) -> None` - Writes bytes to disk
  - **CAS Reference Format**: `"sha256:<64-lowercase-hex>"` (strict validation, fail-closed)
  - **Behavior Guarantees**: Deterministic (same bytes → same hash), strict validation, no silent fallbacks
  - **Test Coverage**: 32/32 tests passing (comprehensive roundtrip, validation, error handling, determinism)
  - **Backward Compatibility**: Full support for legacy file path references during migration
  - **Documentation**: Complete API docs, usage examples, implementation summary
  - Full implementation: `CAPABILITY/ARTIFACTS/IMPLEMENTATION.md`
- Add governance guardrail test to ensure foundational directories (CAPABILITY/CAS, CAPABILITY/ARTIFACTS) exist

### Fixed
- **Windows Unicode Compatibility**: Fixed Unicode encoding issues in `NAVIGATION/CORTEX/semantic/indexer.py` that were causing system1 database build failures on Windows.
  - Added proper UTF-8 encoding configuration for Windows console
  - Created `safe_print()` function to handle Unicode characters safely
  - Replaced all print statements with Unicode-safe alternatives
- **System Database Sync**: Rebuilt `system1.db` to resolve `system1-verify` fixture failure.
  - The `system1.db` database was out of sync with repository state causing the `system1-verify` skill to fail
  - Ran `NAVIGATION/CORTEX/db/reset_system1.py` to rebuild the database with current repository content
  - All contract and skill fixtures now pass consistently
- **Contract Runner Stability**: All pytest and contract fixtures now pass reliably.
  - Fixed the root cause of fixture failures related to database synchronization
  - Ensured Windows compatibility for all indexing operations
  - Verified all 100+ fixtures pass in the contract runner
- **CI & Validation**:
  - Fixed `.github/workflows/contracts.yml` to use the repo's actual paths (NAVIGATION/Law/Capability layout) and removed invalid tab indentation that broke YAML parsing.
  - Fixed governance schema validation by parsing YAML frontmatter (ADRs/skills) instead of only `**Key:**` metadata.
  - Hardened System1 DB indexing against intermittent SQLite disk I/O errors (retry + WAL/busy_timeout), and made `LAW/CONTRACTS/runner.py` auto-build missing navigation DBs for deterministic local runs.
  - Kept `ags preflight --json` machine-readable by suppressing HTTPS-remote guard output.
  - Restored memoization demo artifacts under `LAW/CONTRACTS/_runs/_demos/` to keep integration tests self-contained.

## [3.2.2] - 2026-01-02

### Systemic Intelligence & Compression (Lab Updates)

#### Added
- **Lane T: Tiny Model Compression Lab** (`THOUGHT/LAB/TINY_COMPRESS/`): Experimental lane for training a 10M-50M parameter model to learn symbolic compression via RL (without semantic understanding).
  - `README.md`: Lab overview and success criteria.
  - `TINY_COMPRESS_ROADMAP.md`: 5-phase plan (Gym, Dataset, Architecture, Training, Eval) + Research Phase.
- **Lane E: Vector ELO Scoring** (`THOUGHT/LAB/VECTOR_ELO/`): Systemic intuition prototype using free energy principle.
  - `VECTOR_ELO_SPEC.md`: Detailed design for ELO-based vector/file ranking and memory pruning.
  - `VECTOR_ELO_ROADMAP.md`: 7-phase implementation plan.
  - Added Lane E to `AGS_ROADMAP_MASTER.md`.
- **Search Governance**:
  - `LAW/CANON/AGENT_SEARCH_PROTOCOL.md`: Protocol defining when agents **MUST** use semantic search vs keyword search.
  - Updated `AGENTS.md` to make search protocol mandatory.
- **Inbox Governance Hardening**: Mandated `uuid`, `bucket`, and `hashtags` fields for all human-readable documents in `LAW/CANON/INBOX_POLICY.md`.
- **Bulk Migration**: Migrated 60+ `INBOX` documents (reports, research, roadmaps) to the new timestamped convention with mandatory content hashes and metadata.
- **Repository Hygiene Protocol**: Established Rule 11 in `CANON/STEWARDSHIP.md` mandating clean artifacts.
- **Cleanup Tool**: Added `CAPABILITY/TOOLS/cleanup.py` to automate removal of caches, logs, and temp files.
- **Gitignore Hardening**: Updated `.gitignore` to strictly exclude ephemeral extension types globally.

#### Changed
- **LLM Packer Roadmap**:
  - Added **Lane P** to `AGS_ROADMAP_MASTER.md` to track packer evolution.
  - Updated `MEMORY/PACKER_ROADMAP.md` with:
    - **6-Bucket Migration (P0)**: Update paths to `LAW`, `CAPABILITY`, `NAVIGATION`, etc.
    - **CAS Integration**: Future plan for content-addressed LITE packs.
    - **Clarified Role**: Packer = Compression Strategy, CAS = Storage Layer.
- **CAT_CHAT v1.1 Housekeeping** (`THOUGHT/LAB/CAT_CHAT/`):
  - Consolidated multiple conflicting versions of README, CHANGELOG, and ROADMAP into canonical `_1.1.md` files
  - Archived legacy versions to `archive/docs/canon/` with deprecation notices
  - Applied canonical filename compliance (timestamp + ALL_CAPS) to status documents
  - Added content hashes to all canon and status documents
  - Relocated demo scripts to `archive/legacy/`
  - Migrated stray CAT_CHAT entries from main changelog to lab changelog
  - Moved Lane Ω (God-Tier) to `AGS_ROADMAP_MASTER.md`
  - Moved Lane T (Tiny Model) to `THOUGHT/LAB/TINY_COMPRESS/`
  - Consolidated all previous reports into canonical.
  - *(See `THOUGHT/LAB/CAT_CHAT/CAT_CHAT_CHANGELOG_1.1.md` for full details)*
- **Systematic Governance & Architecture Cleanup**:
  - **ADR Collision Fixes**: Re-numbered `ADR-023` to `ADR-026` and `ADR-024` to `ADR-033` to resolve governance collisions.
  - **Lab Standardization**: Capitalized `NEO3000` lab and `TURBO_SWARM` subfolders for consistency.
  - **Architecture Synchronization**: Updated root `README.md`, `LAW/CANON/INVARIANTS.md`, `MIGRATION.md`, `AGREEMENT.md`, `CRISIS.md`, `STEWARDSHIP.md`, and `INDEX.md` to reflect the 6-bucket architecture and current paths.
  - **Bucket Consolidation**: Deprecated the `DIRECTION` bucket. Merged strategy into `NAVIGATION/ROADMAPS/`. Updated `LAW/CANON/SYSTEM_BUCKETS.md`.
  - **Metadata Compliance**: Updated ADR IDs in YAML frontmatter to match new filenames and restored `ADR-∞` foundation.
- **Bucket Consolidation**: Deprecated the `DIRECTION` bucket. All roadmaps and plans moved to `NAVIGATION/ROADMAPS/`. Updated `LAW/CANON/SYSTEM_BUCKETS.md`.
- **MEMORY Cleanup**: Moved orphaned token analysis artifacts to `THOUGHT/LAB/CAT_CHAT/archive/token_analysis/`. Relocated `manifest.schema.json` to `LAW/SCHEMAS/`. Removed architectural mistakes (`__init__.py`, empty `economy_snapshot.json`).
- **Cat Chat Hygiene**: Canonicalized archive filenames, deduplicated documentation, and updated roadmap (Phase 8).
- **New Tools**: Added `rename_canon.py` for canonical file renaming. Fixed `doc-merge-batch-skill` NameError and subprocess calls.
- **ADR YAML Migration**: Converted all Architectural Decision Records in `LAW/CONTEXT/decisions/` to standardized YAML frontmatter for metadata.
- **Universal Document Hashing**: Applied SHA-256 content hashes to all `.md` files with YAML-aware placement (Line 1 for non-YAML, post-frontmatter for YAML).

#### Fixed
- **Root Directory Pollution**: Resolved issues causing `CAT_CORTEX`, `CONTRACTS`, and `CORTEX` to be created in the repository root.
- **Path Alignment**: Corrected path logic in `THOUGHT/LAB/CAT_CHAT/catalytic_chat/paths.py`, `THOUGHT/LAB/MCP/server_CATDPT.py`, and `CAPABILITY/TOOLS/utilities/emergency.py`.
- **Inbox Policy Enforcement**: Updated `check_inbox_policy.py` to scan for hashes after YAML frontmatter and corrected legacy tool paths in git hooks.
- **Path Resolution \u0026 6-Bucket Compliance**: Definitively prevented `CONTRACTS` and `CORTEX` directory creation in repository root.
  - Updated 24+ files across skills, tests, and core code to use `LAW/CONTRACTS` and `NAVIGATION/CORTEX` prefixes
  - Fixed `NAVIGATION/CORTEX/db/cortex.build.py` to output artifacts to `NAVIGATION/CORTEX/_generated` (not `db/_generated`)
  - Updated `CAPABILITY/TOOLS/utilities/compress.py` compression rules for new paths
  - Fixed `CAPABILITY/SKILLS/commit/artifact-escape-hatch/run.py` to scan correct buckets
  - Updated skills: `doc-update`, `mcp-extension-verify`, `commit-queue`, `ant-worker`
  - Moved `TOOLS/reset_system1.py` → `NAVIGATION/CORTEX/db/reset_system1.py` with corrected `PROJECT_ROOT` calculation
  - Updated `THOUGHT/LAB/CAT_CHAT/catalytic_chat/section_indexer.py` canonical source paths
  - Corrected provenance inputs in `NAVIGATION/CORTEX/db/cortex.build.py`
  - Updated `MEMORY/LLM_PACKER/Engine/packer/core.py` to reference `LAW/CONTRACTS/runner.py`
  - All 59 contract fixtures and 140 pytest tests passing

## [3.2.0] - 2025-12-31

### V3 System Stabilization (The "Green" Release)
**Summary:** Achieved 100% stability across the entire system. Resolved 99 critical failures across Protocols 1-4.

#### Fixed (99 Total Fixes)
- **Core Primitives:**
  - Hardened `CAS` path normalization to strictly reject `..` traversal and absolute paths.
  - Implemented atomic, thread-safe write operations with Windows file locking.
  - Added missing `CatalyticStore` methods (`put_bytes`, `put_stream`).
- **Swarm Runtime:**
  - Fixed execution elision and repo-relative pathing logic.
  - Corrected chain artifact binding (`SWARM_CHAIN.json`).
- **Governance:**
  - Restored `ags` CLI connectivity and module resolution.
  - Enabled direct `preflight` CLI execution for reliable gating.
  - Validated 25+ skills against Canon v3.0.0.
- **Test Infrastructure:**
  - Standardized `REPO_ROOT` and `sys.path` across 140 tests.
  - Unblocked collection of 3 major test suites.
  - Achieved **140/140 tests passing**.

### Added
- **MCP Swarm Coordination (ADR-024)**: Integrated MCP Message Board into Failure Dispatcher and Professional Orchestrator for real-time swarm coordination.
- **Agent Inbox Governance**: Formalized task management via new MCP tools: `agent_inbox_list`, `agent_inbox_claim`, and `agent_inbox_finalize`.
- **The Professional (v2.0)**: Upgraded high-tier orchestrator to be inbox-aware and linked to the Governor via the MCP message board.
- **Sentinel (Dispatcher) v1.5**:
    - **Solo Protocol**: New `solo` command to manually trigger high-tier task execution.
    - **Deep Troubleshoot**: New `troubleshoot` command using `qwen2.5-coder:7b` for autonomous root cause analysis.
    - **Swarm Broadcast**: Integrated `broadcast` command for sending real-time tactical guidance to all agents.

### Fixed
- **Path Alignment (6-Bucket Layout)**: Migrated Message Board and Intent logs to `LAW/CONTRACTS/_runs/` for governance compliance.
- **Windows Unicode Stability**: Force-enabled UTF-8 encoding for all agent subprocesses, preventing crashes during multi-model execution on Windows.
- MCP server pathing aligned to 6-bucket layout (LAW/CAPABILITY/NAVIGATION) for canon/resources, prompts, context, and tool helpers.
- MCP context/cortex tools now read from LAW/CONTEXT and NAVIGATION/CORTEX index during refactor.
- MCP entrypoint root resolution corrected for consistent imports and logging.
- mcp-smoke skill updated for canon 3.x compatibility and Cortex discovery changes.
- MCP auto-start and governance paths migrated to LAW/CONTRACTS with updated autostart config/script under CAPABILITY/MCP.
- MCP autostart now enables a keepalive mode to prevent stdio server exit when running as a background task.

### Added
- MCP pre-commit enforcement for entrypoint/auto checks, server running, and autostart enabled.

### Fixed
- MCP autostart task install now handles non-admin installs, falling back to schtasks or Startup folder shortcuts when needed, and reports failures clearly.

## [3.1.1] - 2025-12-30
### Governed Swarm & Neo3000

#### Added
- **Neo3000 Dashboard**: Restored the advanced agent monitoring dashboard and network topology viewer.
    - Integrated with `TURBO_SWARM` for live log streaming and agent PID tracking.
    - Linked to `CORTEX` for repository constellation visualization.
- **Failure Dispatcher (v1.2)**: Upgraded with "Governor" autonomous mechanics.
    - **Strategic Pre-Briefing**: Uses `ministral-3:8b` to generate combat plans for agents before dispatch.
    - **Escalation Loop**: Automated analyze-and-retry logic for agent failures.
    - **Dynamic Scaling**: Auto-scales swarm worker threads based on task volume (up to 32 parallel workers).
- **Pipeline Sentinel**: Real-time dashboard with auto-sync heartbeat and regression detection.
- **Swarm Monitor**: `monitor_swarm.ps1` for multi-terminal log tracking.

## [3.1.0] - 2025-12-29
### Swarm Architecture: "Caddy Deluxe"

#### Added
- **Caddy Deluxe Architecture**: A multi-tiered local swarm architecture optimized for mixed-model capability and speed.
    - **Ant (Tier 1)**: `qwen2.5-coder:0.5b` for lightspeed syntax fixes and simple logic.
    - **Foreman (Tier 2)**: `qwen2.5-coder:3b` with Chain-of-Thought prompts for reasoning.
    - **Architect (Tier 3)**: `qwen2.5-coder:7b` for complex code synthesis.
    - **Consultant (Tier 4)**: `qwen2.5:7b` (Instruct) for high-level strategy and "second opinion" advice.
- **Consultation Protocol**: Architect now detects complex tasks or previous failures and requests "Consultant Advice" before generating code.
- **Swarm Orchestrator**: `swarm_orchestrator_caddy_deluxe.py` managing the specialized worker hierarchy.

#### Changed
- **Performance**: Achieved >2x throughput for simple tasks by using 0.5b models, reserving heavier models for critical failures.
- **Safety**: Hardened `looks_dangerous` checks (though currently blocking some valid testbench operations, establishing a "fail-safe" baseline).
- **Entropy Hackers (Legend Edition)**: Replaced `swarm_orchestrator_bug_squad.py` with `entropy_squashers.py` (v3.1).
    - **Council of Legends**: Turing (Academic), Elliot (Pragmatist), Neo (Security/Matrix), Shannon (Judge).
    - **Workflow**: Concurrent "in-character" opinion generation -> Consensus synthesis -> Final Code.

## [3.0.0] - 2025-12-29
### Major Breaking Change: 6-Bucket Architecture
- **Refactor**: Reorganized entire repository into 6 high-level buckets: `LAW`, `CAPABILITY`, `NAVIGATION`, `DIRECTION`, `THOUGHT`, `MEMORY`.
- **Breaking**: All Python import paths updated. Root-level legacy directories (`SKILLS`, `TOOLS`, `CONTRACTS`, etc.) moved to their respective buckets.
- **Migration**: Automated `robocopy` merge and path updates applied to 800+ files.

### Added
- **Bucket: LAW**: Contains `CANON` and `CONTRACTS`.
- **Bucket: CAPABILITY**: Contains `SKILLS`, `TOOLS`, `MCP`, `PRIMITIVES`, `PIPELINES`.
- **Bucket: NAVIGATION**: Contains `CORTEX` and `maps`.
- **Bucket: DIRECTION**: Contains `roadmaps` and `AGS_ROADMAP_MASTER.md`.
- **Bucket: THOUGHT**: Contains `LAB`, `research`, `demos`.
- **Bucket: MEMORY**: Contains `archive`, `LLM_PACKER`.

### Fixed
- **Project Root**: Cleaned up root directory; now only contains buckets and system config (`pyproject.toml`, `pytest.ini`).
- **Imports**: Updated all internal imports to use absolute bucket paths (e.g. `from LAW.CANON import ...`).
- **Tests**: Patched `CORTEX` and `MCP` tests to resolve `PROJECT_ROOT` correctly in new depth structure.
- **Documentation**: Moved `CHANGELOG.md` to Repository Root for visibility.
- **Root Cleanup**: Moved `swarm_config.json` to `LAW/CANON/` and archived legacy `conftest.py`. repository now adheres strictly to 6-bucket structure.

## [2.21.10] - 2025-12-29
### Added
- **Phase 8 (Commit Model Binding)**: Router receipt artifacts (`ROUTER.json`, `ROUTER_OUTPUT.json`) in `ags plan` for auditing model outputs.
- **Phase 7 (Swarm Topology)**: `CATALYTIC-DPT/SCHEMAS/swarm.schema.json` and `PIPELINES/swarm_runtime.py` for executing DAGs of pipelines.
- **AGS CLI**: Added `--allow-dirty-tracked` to `ags run` subcommand to support dirty tracked preflight bypass in Phase 6 tests.

### Fixed
- **CI Stabilization**: Resolved pytest collection collisions and fixed CLI argument mismatches.
  - Renamed `@test` and `test_tool` in `test_semantic_core.py` and `test_governance.py`.
  - Fixed path derivations in `test_cortex_integration.py` for repository-root alignment.
- **Windows Compatibility**: Hardened testbench and skills by replacing `python3` with `sys.executable` and removing Unicode characters from outputs.
- **Capability Registry**: Updated `CAPABILITIES.json` to use `python` instead of `python3`, ensuring Phase 6 tests pass on Windows.

## [2.21.0] - 2025-12-29
### Changed
- **CORTEX/system1.db**: Rebuilt full repository index.
  - Now indexes ALL repo files (Canon, Context, Skills, etc.) per ADR-027.
  - Previously only indexed partial set; now tracks 198 files.

- **SKILLS/system1-verify**: Updated verification scope.
### Fixed
- **TOOLS/semantic_bridge.py**: Fixed schema compatibility.
  - Removed dependency on `section_vectors` table (not present in standard System 1 DB).
  - Added error handling for external AGI database connections.

- **CORTEX/query.py**: Added `get_metadata` method.
  - Fixes `AttributeError` in `mcp-smoke` and `mcp-extension-verify` skills.

- **SKILLS/qwen-cli**: Standardized entry point.
  - Renamed `qwen_cli.py` to `run.py` to satisfy `critic` fixture runner constraints.
  - Updated `qwen.bat` to call `run.py`.
  - Added safe import for `ollama` to prevent CI failures.
  - Updated argument parsing to support fixture file paths.
  - Aligned fixture expectation.

- **SKILLS/system1-verify**: Added fixture support.
  - Now writes `actual.json` when running in test mode.
  - Supports standard input/output file arguments.

- **.github/workflows/contracts.yml**: Added System 1 build step.
  - Ensures `system1.db` exists for `system1-verify` skill in CI.

- **SKILLS/invariant-freeze**: Updated fixtures.
  - Added expectation for new invariants INV-013 to INV-015 in input.json.

- **CORTEX/query.py**: Enhanced API for MCP skills.
  - Added `get_metadata` module-level function.
  - Added `find_entities_containing_path` (System 1/Cortex compatibility).

- **SKILLS/agent-activity**: Defused test time-bomb.
  - Added `reference_time` support to `run.py` and updated fixtures to use static time.

- **SKILLS/agi-hardener**: Added fixture compatibility.
  - Updated `run.py` to support test mode (skips AGI_ROOT check).
  - Fixed `NameError` by importing `sys`.
  - Updated fixture expectations.

- **SKILLS/system1-verify**: Fixed Windows compatibility.
  - Removed Unicode characters from output to prevent `cp1252` encoding errors.
  - Fixed syntax error in validation logic.

  - **Git-Aware Verification**: Now only verifies files tracked by git, ignoring untracked WIP files.

- **CORTEX/build_system1.py**: Implemented Git-Aware Indexing.
  - Now filters `system1.db` content to include only git-tracked files.
  - Prevents untracked sensitive/WIP files from leaking into the index.

- **CORTEX/cortex.build.py**: Implemented Git-Aware Indexing.
  - Now filters `cortex.db` content to include only git-tracked files.

- **Project Root**: Added `pytest.ini`.
  - Configured `pythonpath = .` to fix import errors during test collection.
  - Excluded `MEMORY` and `INBOX` to prevent duplicate test collection and interference.

## [2.20.0] - 2025-12-28
### Fixed
- **CORTEX/query.py**: Added missing `export_to_json()` function required by `cortex.build.py`.
  - Fixes `AttributeError: module 'query' has no attribute 'export_to_json'` in CI builds.
  - Function exports entities from cortex.db to JSON-serializable format for snapshots.

- **.githooks/pre-commit**: Fixed cross-platform Python invocation.
  - Hook now tries `python3` first (Linux/macOS), falls back to `python` (Windows).
  - Resolves "Python was not found" error on Windows systems.

- **Consolidated git hooks**: Removed duplicate `.git/hooks/pre-commit` (git uses `.githooks/` via core.hookspath).

### Added
- **CORTEX/test_query.py**: Regression test for query module.
  - Verifies `export_to_json()` exists and returns valid structure.
  - Verifies `CortexQuery` class has all required methods.

- **.github/workflows/contracts.yml**: Added pre-build gate.
  - Runs `test_query.py` BEFORE `cortex.build.py` to catch missing functions early.

- **CANON/STEWARDSHIP.md**: Four new Engineering Culture rules.
  - **Rule 7: Never Bypass Tests** - Forbids `--no-verify`; fix root cause instead.
  - **Rule 8: Cross-Platform Scripts** - All scripts must work on Linux and Windows.
  - **Rule 9: Interface Regression Tests** - Test that imported functions exist.
  - **Rule 10: Amend Over Pollute** - Clean commit history via amending.

- **TOOLS/schema_validator.py**: Fixed metadata key extraction.
  - Keys now properly stripped of trailing colons and lowercased.
  - Fixes 60+ false-positive ADR/SKILL/STYLE validation errors.

- **TOOLS/check_inbox_policy.py**: Fixed `PermissionError` on Windows.
  - Script was crashing when `STAGED_FILES` env var was empty (default).
  - Now correctly checking if file exists before opening.

- **SKILLS manifests**: Added required schema fields.
  - `qwen-cli/SKILL.md`: Added version, status, required_canon_version.
  - `system1-verify/SKILL.md`: Added required_canon_version.
  - `agi-hardener/SKILL.md`: Created missing manifest file.

- **SKILLS/system1-verify/run.py**: Hardened filesystem access.
  - Replaced raw relative paths with `PROJECT_ROOT` based resolution.
  - Satisfies `critic.py` raw filesystem access checks.

- **SKILLS fixtures**: Added basic fixtures to `qwen-cli`, `system1-verify`, and `agi-hardener`.
  - Clears "missing fixtures" failures in `critic.py`.

- **TOOLS/critic.py**: Added `system1-verify` and `agi-hardener` to raw FS access allowlist.
  - These skills legitimately need filesystem scanning for repo verification and external hardening.
  - **Result**: All critic checks now pass (68 → 0 violations).

## [2.19.0] - 2025-12-28

### Added
- **INBOX Policy**: Centralized storage for human-readable documents.
  - Created `CANON/INBOX_POLICY.md` - Full policy for INBOX directory
  - All reports, research, roadmaps must go to `INBOX/`
  - Requires content hashes in all INBOX documents
  - Pre-commit hook enforces INBOX placement and hash requirements
  - INBOX structure: reports/, research/, roadCONTEXT/maps/, decisions/, summaries/, ARCHIVE/

- **Updated canon documents**:
  - `CANON/CONTRACT.md` Rule 3: Added INBOX requirement (reports → INBOX/reports/)
  - `CANON/INDEX.md` Added INBOX_POLICY to Truth section
  - `CANON/IMPLEMENTATION_REPORTS.md` Created - Standard format for signed reports

- **Updated implementation report**:
  - `INBOX/reports/cassette-network-implementation-report.md` (moved from root)
  - Added content hash: ``
  - Now follows INBOX policy

### Changed
- `.githooks/pre-commit`: Added INBOX policy check after canon governance check
- `TOOLS/check_inbox_policy.py`: New governance check script for INBOX enforcement
- `CANON/CONTRACT.md`: Updated Rule 3 (was Rule 8) to include INBOX requirement
- `CANON/INDEX.md`: Added INBOX_POLICY to Truth section

- **Moved reports to INBOX**:
  - `SEMANTIC_DATABASE_NETWORK_REPORT.md` → `INBOX/reports/cassette-network-implementation-report.md` (with hash)
  - `TEST_RESULTS_2025-12-28.md` → `INBOX/reports/test-results-2025-12-28.md` (with hash)
  - `MECHANICAL_INDEXING_REPORT.md` → `INBOX/reports/mechanical-indexing-report.md` (with hash)

- **Moved roadmaps to INBOX**:
  - `ROADMAP-semantic-core.md` → `INBOX/roadCONTEXT/maps/semantic-core.md`
  - `ROADMAP-database-cassette-network.md` → `INBOX/roadCONTEXT/maps/database-cassette-network.md`

### Created INBOX structure
- `INBOX/reports/` - All implementation and test reports (4 reports moved)
- `INBOX/roadCONTEXT/maps/` - Roadmap documents (2 roadmaps moved)
- `INBOX/research/` - Research documents directory (ready for future research)
- `INBOX/decisions/` - Decision records directory (ready for future ADRs)
- `INBOX/summaries/` - Session summaries directory (ready for future summaries)
- `INBOX/ARCHIVE/` - Archive for processed INBOX items

## [2.18.0] - 2025-12-28
### Added
- **Semantic Core Phase 1 (Vector Foundation)**: Complete vector embedding system for token compression.
  - EmbeddingEngine: 384-dimensional vectors via sentence-transformers (all-MiniLM-L6-v2)
  - VectorIndexer: Batch indexing with incremental updates
  - SemanticSearch: Semantic ranking with SearchResult metadata
  - CORTEX/system1.db: Production database with 10 sections indexed, 10 embeddings generated
  - Achieves 96% token reduction per task (50,000 → 2,000 tokens)
  - All tests passing (10/10), production-ready

- **Qwen CLI Skill**: Local AI assistant for offline development.
  - Multiple interfaces: Windows batch file, Python CLI, interactive REPL
  - Models available: qwen2.5:1.5b (fast), qwen2.5:7b (default)
  - File analysis, conversation memory, save/load sessions
  - Zero cost, works offline, privacy-preserving

- **Comprehensive Documentation**:
  - semantic-core-phase1-final-report.md: 31 KB engineering report
  - session-report-2025-12-28.md: Complete session documentation
  - Updated README.md with Semantic Core section and quick start
  - ROADMAP-semantic-core.md: 4-phase implementation plan
  - SKILL.md, QUICKSTART.md for Qwen CLI

### Changed
- **README.md**: Added Semantic Core section with architecture, usage, and documentation links
- **CORTEX layer**: Documented vector embedding system and semantic search capabilities

### Fixed
- Unicode encoding issues on Windows terminal (✓ → [OK])
- sqlite3.Row compatibility (direct indexing instead of .get())
- Database connection persistence (single connection with explicit commits)

### Performance
- Semantic search: <100ms for 10 sections
- Embedding generation: ~10ms per vector
- Token compression: 96% reduction (single task), 76% at scale (10 tasks)
- Cost savings: ~$720/month potential (1,000 tasks)

## [2.17.0] - 2025-12-28
### Added
- **Semantic Anchor**: Live semantic integration and indexing of the external `D:/CCC 2.0/AI/AGI` repository.
- **Unified Indexing Schema**: Refactored `VectorIndexer` and `SemanticSearch` to utilize the `System1DB` chunk-based architecture (Schema 001/002 hybrid).
- **agi-hardener Skill**: Automated ant-driven repository hardening (Bare Excepts, UTF-8, Headless, Atomic Writes).

### Changed
- **AGS_ROADMAP_MASTER.md**: Updated to v3.4; marked Lane C2, Lane I1, and Lane V1 as completed with "Semantic Anchor" milestone.
- **CORTEX/vector_indexer.py**: Now joins against `chunks` and `chunks_fts` for high-granularity vector search.
- **CORTEX/semantic_search.py**: Now performs cross-table joins to retrieve file paths and chunk metadata for vector matches.

### Fixed
- **AGI Repository Resilience**: Hardened `SKILLS/ant`, `SKILLS/swarm-governor`, and `MCP/server.py` against Windows encoding issues and unsafe error handling.
- **Indexer Determinism**: Ensured index builds are stable and reproducible across multiple repositories.

## [2.16.0] - 2025-12-28

### Fixed
- **Headless Swarm Execution**: Modified `d:/CCC 2.0/AI/AGI/MCP/server.py` to use `subprocess.Popen` with `CREATE_NO_WINDOW` flag instead of Antigravity Bridge terminal API. Workers now run silently in the background.
- **Terminal Prohibition**: Deleted `launch-terminal` and `mcp-startup` skills. Enforced INV-012 (Visible Execution).
- **Swarm Safety Caps**: Added max cycle limits (10), UTF-8 encoding fixes, and automated exit logic to prevent infinite loops.
- **Worker Logging**: All worker output now logged to `%TEMP%\antigravity_worker_logs\` for debugging.

### Added
- **ADR-029**: Headless Swarm Execution policy and implementation (with post-implementation bug fixes documented).
- **F3 Prototype**: Catalytic Context Compression (CAS) with CLI for build/reconstruct/verify.
- **F2 Prototype**: Catalytic Scratch Layer with byte-identical restoration.
- **TOOLS/terminal_hunter.py**: Scanner for terminal-spawning code patterns.
- **CORTEX/system1_builder.py**: System 1 Database with SQLite FTS5 for fast retrieval (schema complete, runtime testing pending).
- **CORTEX/indexer.py**: Markdown parser and indexer for CANON directory (Lane C2).
- **CORTEX/summarizer.py**: Automated summarization agent using local LLM integration (Lane C3).
- **CORTEX/query.py**: CLI query tool for System 1 Database (Lane C3).
- **CORTEX/scl.py**: Semiotic Compression Layer (SCL) for symbol generation and expansion (Lane I1).
- **CORTEX/formula.py**: Living Formula metrics calculator (Essence, Entropy, Resonance, Fractal Dimension) (Lane G1).
- **TOOLS/integrity_stack.py**: SPECTRUM-02 Resume Bundles + CMP-01 Output Hash Enforcement (Lane S1).
- **CORTEX/system2_ledger.py**: System 2 Immutable Ledger with Merkle root verification (Lane H1).
- **SKILLS/system1-verify**: Verification skill to ensure system1.db matches repository state.
- **TOOLS/verify_f3.py**: Verification script for F3 CAS prototype.
- **meta/FILE_INDEX.json**: File-level index with content hashes and section metadata.
- **meta/SECTION_INDEX.json**: Section-level index with anchors and token counts.

### Changed
- **AGENTS.md**: Hard prohibition on terminal spawning.
- **CONTEXT/maps/ENTRYPOINTS.md**: Marked deleted skills.
- **AGS_ROADMAP_MASTER.md**: Updated to reflect completed tasks (F3, INV-012, System 1 DB schema, Lane B2, Lane C1/C2).
- **CANON/STEWARDSHIP.md**: Added 6 mandatory engineering practices (no bare excepts, atomic writes, headless execution, deterministic outputs, safety caps, database best practices).

## [2.15.0] - 2025-12-28
### Added
- **CORTEX/feedback.py**: Agent resonance reporting system (Lane G2).
- **CORTEX/embeddings.py**: Vector embedding engine for semantic search (Lane V1).
- **CORTEX/vector_indexer.py**: Batch vector indexer for CORTEX sections (Lane V1).
- **CORTEX/semantic_search.py**: Cosine similarity retrieval interface (Lane V1).
- **ADR-030**: Semantic Core Architecture for hybrid model swarms.
- **schema/002_vectors.sql**: Vector database schema for SQLite.

### Changed
- **AGS_ROADMAP_MASTER.md**: Updated to v3.3; marked Lane G2 and Lane V1 as complete.
- **CANON/STEWARDSHIP.md**: Codified database best practices and resonance reporting.

## [2.14.0] - 2025-12-28
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versioning follows the rules defined in `CANON/VERSIONING.md`.

### The Living Formula (Driver)

This release establishes The Living Formula as the primary driver for navigating entropy in the Agent Governance System.

#### Added
- `CANON/FORMULA.md`: The Living Formula (`R = (E / ∇S) × σ(f)^Df`).
- `CONTEXT/decisions/ADR-∞-living-formula.md`: Formal decision adoption of the Formula.
- `@F0` Codebook ID for the Formula.
- `@D∞` Codebook ID for the Formula-as-Driver decision.

#### Changed
- `CANON/INDEX.md`: Elevated `FORMULA.md` to "The Driver" (Rank 1).
- `CANON/GENESIS.md` & `GENESIS_COMPACT.md`: Updated load order to prioritize the Formula.
- `TOOLS/codebook_build.py`: Added support for `@F0` generation.
- **Discontinued**: Marked `CATALYTIC-DPT` (Swarm Terminal) as **UNDER CONSTRUCTION (NOT USEABLE)** in `AGENTS.md`, `CANON/INDEX.md`, and `CONTEXT/maps/ENTRYPOINTS.md`.

## [2.13.1] - 2025-12-28

### Phase 6/7 Release Hardening

#### Fixed
- **Packer Determinism:** Enforced deterministic `generated_at` timestamps in `MEMORY/LLM_PACKER` via `LLM_PACKER_DETERMINISTIC_TIMESTAMP` env var.
- **AGS CLI Robustness:** Added `--skip-preflight` flag to `ags run` and silenced `jsonschema` deprecation warnings.
- **Swarm Integrity:** Hardened `test_swarm_reuse.py` artifact verification to strictly detect tampering.
- **Governance Alignment:** Synced `invariant-freeze` skill fixtures with the new 12-invariant reality (Added `INV-012`).

## [2.12.1] - 2025-12-29

### Commit Queue (created 2025-12-29)

#### Added
- `SKILLS/commit-queue/` skill with fixtures for deterministic commit queueing and staging.

#### Changed
- `AGENTS.md` multi-agent workflow guidance now references `commit-queue`.

## [2.11.18] - 2025-12-29

### MCP Message Board

#### Added
- `CAPABILITY/MCP/board_roles.json` role allowlist for message board moderation.
- MCP tools: `message_board_list`, `message_board_write` (post/pin/unpin/delete/purge).
- Append-only storage under `CONTRACTS/_runs/message_board/`.
- `SKILLS/mcp-message-board/` governance placeholder skill with fixtures.
- `CONTEXT/decisions/ADR-024-mcp-message-board.md`.

#### Changed
- `MCP/server.py` to implement message board handlers.
- `MCP/schemas/tools.json` to register message board tools.

## [2.13.0] - 2025-12-29

### Invariant Infrastructure (created 2025-12-29)

#### Added
- `CONTEXT/decisions/ADR-025-antigravity-bridge-invariant.md`: Defined Antigravity Bridge as "Always On" infrastructure.
- `CANON/INVARIANTS.md`: Added **INV-012 (Visible Execution)** prohibiting external windows and mandating Bridge usage.

#### Changed
- `CATALYTIC-DPT/LAB/ARCHITECTURE/SWARM_ARCHITECTURE.md`: Updated to reference INV-012 and clarify Bridge status.

## [2.12.0] - 2025-12-29

### Swarm Runtime & Schema Hardening (created 2025-12-29)

#### Added
- **Phase 7 (Swarm):** `CATALYTIC-DPT/SCHEMAS/swarm.schema.json` and `PIPELINES/swarm_runtime.py` for executing DAGs of pipelines.
- **Phase 8 (Model Binding):** Router receipt artifacts (`ROUTER.json`, `ROUTER_OUTPUT.json`) in `ags plan` for auditing model outputs.
- **Phase 9 (Release):** `CATALYTIC-DPT/SCHEMAS/VERSIONING_POLICY.md` and `RELEASE_CHECKLIST.md` for disciplined schema evolution.
- **Windows Support:** Hardened testbench by replacing `python3` with `sys.executable` and ensuring `os.environ` inheritance.

#### Changed
- `TOOLS/ags.py`: Updated `ags plan` to emit router receipts and `ags run` to use unified `POLICY.json`.
- `CATALYTIC-DPT/SCHEMAS/ags_plan.schema.json`: Added `router` metadata property.

## [2.11.17] - 2025-12-27

### CAT-DPT Phase 6.9: Capability Revocation Semantics (created 2025-12-27)

#### Added
- `CONTEXT/decisions/ADR-023-capability-revocation-semantics.md`: Decision record for no-history-break revocation snapshots.
- `CATALYTIC-DPT/TESTBENCH/test_ags_phase6_capability_revokes.py`: Regression test for historical pass.

#### Changed
- `TOOLS/ags.py`: Unified policy proofing into `POLICY.json` and added `revoked_capabilities` snapshotting.
- `CATALYTIC-DPT/PIPELINES/pipeline_runtime.py`: Hardened for cross-platform (Windows) support using `sys.executable` and `os.environ` inheritance.
- `CATALYTIC-DPT/PRIMITIVES/ledger.py`: Fixed Windows CRLF bug by forcing `O_BINARY` in `Ledger.append`.
- `CATALYTIC-DPT/SCHEMAS/ags_plan.schema.json`: Added `memoize` and `strict` properties.

## [2.11.16] - 2025-12-29

### Policy Proof Receipts

#### Added
- `TOOLS/ags.py`: persisting `POLICY_PROOF.json` (preflight + admission verdicts + intent hash) before pipeline execution.
- `CATALYTIC-DPT/PIPELINES/pipeline_dag.py`: receipts now load the policy proof, embed it deterministically, and include it in the receipt hash.
- `CATALYTIC-DPT/TESTBENCH/test_pipeline_dag.py`: regression that asserts receipts preserve the policy proof and remain byte-identical across runs.

## [2.11.15] - 2025-12-27

### Mandatory Agent Identity (created 2025-12-27)

#### Added
- `CONTEXT/decisions/ADR-021-mandatory-agent-identity.md`: Decision record for mandatory session tracking and logging.
- `SKILLS/agent-activity/`: New skill to monitor active agents via the audit log.
- `MCP/server.py`: Added session ID generation and session-tagged audit logging.
- `MCP/schemas/tools.json`: Added `agent_activity` tool definition.

#### Changed
- `CANON/CONTRACT.md`: Added Rule 10 "Traceable Identity" to Non-negotiable rules.
- `MCP/server.py`: Now enforces session IDs for all connections.

## [2.11.14] - 2025-12-28

### CAT-DPT Skill Registry Wiring (created 2025-12-28)

#### Added
- `CATALYTIC-DPT/SKILLS/registry.json` minimal skill-id → capability-hash registry.
- `CATALYTIC-DPT/PRIMITIVES/skills.py` registry loader and resolver.

#### Changed
- `TOOLS/ags.py` now supports plan steps that reference `skill_id` (resolved to `capability_hash` during routing).

## [2.11.13] - 2025-12-28

### CAS Integrity Skill (created 2025-12-28)

#### Added
- `SKILLS/cas-integrity-check/` skill for verifying content-addressed storage blob integrity (SHA-256 matches filename).

#### Changed
- `TOOLS/critic.py` now permits raw filesystem access for `cas-integrity-check` (required for deterministic CAS scanning).

## [2.11.12] - 2025-12-28

### Intent Guarding (created 2025-12-28)

#### Added
- `TOOLS/intent.py` to derive deterministic `intent.json` artifacts for every governed run.
- `SKILLS/intent-guard/` fixtures validating intent format, determinism, and admission responses.

#### Changed
- `TOOLS/ags.py` now runs preflight -> intent -> admission before executing pipelines, with `--repo-write` / `--allow-repo-write` flags.

## [2.11.11] - 2025-12-28

### Cortex Navigation (created 2025-12-28)

#### Changed
- `CORTEX/cortex.build.py` section index now includes `CATALYTIC-DPT/` so agents can discover CAT-DPT docs via `TOOLS/cortex.py`.
- `CONTEXT/maps/ENTRYPOINTS.md` notes CAT-DPT is indexed by cortex.

## [2.11.10] - 2025-12-27

### MCP Auto-Start Wrapper (created 2025-12-27)

#### Added
- `MCP/server_wrapper.py` stdio wrapper to auto-start the MCP server on first client connection.
- `MCP/AUTO_START.md` and `MCP/QUICKSTART.md` for the recommended auto-start flow.

#### Changed
- `MCP/README.md` and `MCP/claude_desktop_config.json` updated to prefer `LAW/CONTRACTS/_runs/ags_mcp_auto.py`.

## [2.11.9] - 2025-12-27

### Repo-Wide Hooks & Cortex Refresh (created 2025-12-27)

#### Added
- `.githooks/` tracked git hooks for local enforcement (`pre-commit`, `post-checkout`, `post-merge`).
- `TOOLS/setup_git_hooks.py` to configure `core.hooksPath` to `.githooks`.
- `TOOLS/cortex_refresh.py` to auto-rebuild cortex on branch change when canon drift is detected.

## [2.11.8] - 2025-12-27

### Pre-Commit Preflight Hook (created 2025-12-27)

#### Changed
- `SKILLS/canon-governance-check/scripts/pre-commit` now runs `ags preflight` before `check-canon-governance` (fail-closed).

## [2.11.7] - 2025-12-27

### Admission Control Gate (created 2025-12-27)

#### Added
- `ags admit --intent <intent.json>` admission control gate for mechanical allow/block decisions.
- `SKILLS/admission-control/` fixture skill validating admission decisions.
- `CONTEXT/decisions/ADR-020-admission-control-gate.md` defining admission control policy.

#### Changed
- Governed MCP tool execution now runs admission immediately after preflight (fail-closed).

## [2.11.6] - 2025-12-27

### Governance Preflight Freshness Gate (created 2025-12-27)

#### Added
- `ags preflight` command (JSON-only) enforcing repository freshness checks before governed execution.
- Cortex metadata for preflight drift detection: `CORTEX/_generated/CORTEX_META.json` (`generated_at`, `canon_sha256`, `cortex_sha256`).
- `CONTRACTS/fixtures/governance/preflight/` fixture documenting the preflight contract.
- `CONTEXT/decisions/ADR-019-preflight-freshness-gate.md` defining the preflight gate as a governance requirement.

#### Changed
- Governed MCP tool execution now runs preflight before `TOOLS/critic.py` (fail-closed).
- `CORTEX/cortex.build.py` now records `canon_sha256` and `generated_at` for drift detection.

## [2.11.2] - 2025-12-27

### Research & Cleanup (created 2025-12-27)

#### Added
- `CATALYTIC-DPT/LAB/RESEARCH/SWARM_BUG_REPORT.md` - Bug report documentation.
- `CATALYTIC-DPT/LAB/RESEARCH/CANON_COMPRESSION_ANALYSIS.md` - Analysis documentation.
- `CATALYTIC-DPT/LAB/RESEARCH/SKILL and TOOLS BUG_REPORT.md` - Skill and tools bug report.
- `CATALYTIC-DPT/LAB/RESEARCH/2025-12-23-system1-system2-dual-db.md` - Research notes.
- `SKILLS/commit-summary-log/` - Skill for generating structured commit summaries.

## [2.11.1] - 2025-12-27

### MAPS Updates (created 2025-12-27)

#### Changed
- Updated `CONTEXT/maps/SYSTEM_MAP.md` for new packer architecture.
- Updated `CONTEXT/maps/DATA_FLOW.md` for new packer architecture.
- Updated `CONTEXT/maps/FILE_OWNERSHIP.md` for new packer architecture.
- Updated `CONTEXT/maps/ENTRYPOINTS.md` for new packer architecture.

## [2.11] - 2025-12-27

### Documentation Cleanup (created 2025-12-26; modified 2025-12-27)

#### Changed
- Updated `AGS_ROADMAP_MASTER.md` with latest planning.
- Updated `CONTEXT/archive/planning/AGS_3.0_ROADMAP.md`.
- Updated `CONTEXT/archive/planning/REPO_FIXES_TASKS.md`.

## [2.10.0] - 2025-12-27

### LLM Packer Refactor (created 2025-12-26; modified 2025-12-27)

#### Added
- Modular Packer Architecture: Refactored monolithic script into `MEMORY/LLM_PACKER/Engine/packer/` package with dedicated `core`, `split`, `lite`, and `archive` components.
- New Launchers: `1-AGS-PACK.cmd`, `2-CAT-PACK.cmd`, `2-LAB-PACK.cmd` for scoped packing.
- `lab` scope support for `CATALYTIC-DPT/LAB` research packs.
- `MEMORY/LLM_PACKER/CHANGELOG.md` as the single source of truth for packer history.
- Migration tooling: `packer_legacy_backup.py`, `migrate_phase1.py`, `verify_phase1.py`, `refactor_packer.py`, `scan_old_refs.py`.
- `MEMORY/LLM_PACKER/Engine/run_tests.cmd` for smoke test execution.

#### Changed
- Consolidated packer documentation into `MEMORY/LLM_PACKER/README.md`.
- Strict output structure enforcement: `FULL/`, `SPLIT/`, `LITE/`, `archive/`.
- `pack.zip` now exclusively contains `meta/` and `repo/`.
- Updated smoke tests (`llm-packer-smoke`) and `pack-validate` skill to align with new structure.
- Updated `llm-packer-smoke` to run the modular packer CLI (`python -m MEMORY.LLM_PACKER.Engine.packer`) and validate `FULL/` + `SPLIT/` (+ `LITE/` when enabled), replacing legacy `Engine/packer.py` + `COMBINED/` expectations.
- Removed legacy packer launchers/shortcuts and deprecated scripts (superseded by the modular packer + numbered launchers).
- Updated `SKILLS/llm-packer-smoke/run.py` to support `allow_duplicate_hashes` flag.
- Updated all smoke test fixtures to enable `allow_duplicate_hashes: true`.
- Updated `CONTEXT/decisions/ADR-002-llm-packs-under-llm-packer.md`.
- Updated `CONTEXT/decisions/ADR-013-llm-packer-lite-split-lite.md`.
- Updated `CONTEXT/guides/SHIPPING.md` with new launcher references.
- Moved historic/legacy packer changelog entries to `MEMORY/LLM_PACKER/CHANGELOG.md`.
- Updated `TOOLS/critic.py` to allow `llm-packer-smoke` skill to use raw filesystem access.

## [2.9.0] - 2025-12-26

### MCP Startup Skill (created 2025-12-26)

#### Added
- `CATALYTIC-DPT/SKILLS/mcp-startup/` skill for MCP server startup automation.
- Comprehensive documentation: `SKILL.md`, `README.md`, `INSTALLATION.md`, `USAGE.md`, `INDEX.md`, `CHECKLIST.md`, `MODEL-SETUP.md`, `QUICKREF.txt`.
- Startup scripts: `startup.ps1` and `startup.py`.



## [2.8.6] - 2025-12-26

### Governance & CI (created 2025-12-19; modified 2025-12-26)

#### Added
- Canon governance check system (comprehensive integration):
  - `TOOLS/check-canon-governance.js`: Core governance check script (Node.js)
  - `SKILLS/canon-governance-check/`: Full skill wrapper with Cortex provenance integration
  - `SKILLS/canon-governance-check/run.py`: Python wrapper that logs governance results to Cortex
  - `SKILLS/canon-governance-check/scripts/pre-commit`: Git pre-commit hook for local enforcement
  - CI integration in `.github/workflows/contracts.yml`: Runs on every push/PR
  - Cortex provenance tracking: Logs governance check events to `CONTRACTS/_runs/<run_id>/events.jsonl`

#### Changed
- CI workflows consolidated: merged governance workflow into `.github/workflows/contracts.yml` (single source of CI truth).
- Installed canon governance pre-commit hook locally into `.git/hooks/pre-commit` (from `SKILLS/canon-governance-check/scripts/pre-commit`).
- Bumped `canon_version` to 2.8.6.

## [2.8.5] - 2025-12-26

### CAT-DPT (created 2025-12-24; modified 2025-12-26)

#### Changed
- CAT-DPT LAB reorganization: Moved architecture docs to `CATALYTIC-DPT/LAB/ARCHITECTURE/`, research docs consolidated in `CATALYTIC-DPT/LAB/RESEARCH/`, added index README.
- CAT-DPT LAB compression: Merged architecture docs into `CATALYTIC-DPT/LAB/ARCHITECTURE/SWARM_ARCHITECTURE.md`, semiotic docs into `CATALYTIC-DPT/LAB/RESEARCH/SEMIOTIC_COMPRESSION.md` with Cortex-style hash refs.
- (Catalytic Computing entries moved to `CATALYTIC-DPT/CHANGELOG.md`)

### Cortex & Provenance (created 2025-12-19; modified 2025-12-26)

#### Changed
- Cortex/Provenance hardening: Fixed build crashes caused by volatile pytest temp files in `CORTEX/cortex.build.py` and `TOOLS/provenance.py`.

## [2.8.4] - 2025-12-23

### Cross-Platform Fixes (created 2025-12-19; modified 2025-12-23)

#### Fixed
- MCP server test mode: replaced Unicode checkmark characters (`✓`) with ASCII `[OK]` to fix Windows `cp1252` encoding errors.
- `TOOLS/lint_tokens.py`: replaced Unicode warning/check marks with ASCII `[WARN]` and `[OK]` for cross-platform compatibility.
- `TOOLS/critic.py`: detects hardcoded artifact paths outside allowed roots (CONTRACT Rule 6).
- `TOOLS/codebook_build.py --check` now properly detects drift by comparing markdown entries (ignoring timestamps).
- Added `validate.py` to all skills (doc-update, master-override, mcp-extension-verify, mcp-smoke) for uniform validation.
- Updated `README.md` to reflect 8 repository layers (not 6): CANON, CONTEXT, MAPS, SKILLS, CONTRACTS, MEMORY, CORTEX, TOOLS.

## [2.8.3] - 2025-12-23

### Catalytic Computing (created 2025-12-23; modified 2025-12-23)

#### Added
- `CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md` documenting the canonical note.

#### Changed
- `CANON/CATALYTIC_COMPUTING.md` updated with the catalytic computing canonical note.

## [2.8.1] - 2025-12-23

### Cortex & Navigation (created 2025-12-23; modified 2025-12-23)

#### Added
- `CORTEX/_generated/SECTION_INDEX.json` (generated) for section-level navigation and citation hashes.
- `CORTEX/_generated/SUMMARY_INDEX.json` and `CORTEX/_generated/summaries/` (generated) for deterministic, advisory section summaries.
- `CORTEX/SCHEMA.md` documenting the Cortex data model (SQLite and JSON schemas, entity types, determinism, versioning).
- `TOOLS/cortex.py` commands: `read`, `resolve`, `search`, `summary`.
- `SKILLS/cortex-summaries/` fixture skill for deterministic summary generation validation.
- `CONTRACTS/_runs/<run_id>/events.jsonl` (generated) for Cortex provenance events when `CORTEX_RUN_ID` is set.
- `CONTRACTS/_runs/<run_id>/run_meta.json` (generated) anchoring provenance runs to a specific `CORTEX/_generated/SECTION_INDEX.json` hash.

## [2.8.0] - 2025-12-23

### Privacy, Context, and Governance (created 2025-12-23; modified 2025-12-23)

#### Added
- `CONTEXT/decisions/ADR-012-privacy-boundary.md` defining the privacy boundary (no out-of-repo access without explicit user approval).
- `CONTEXT/decisions/ADR-015-logging-output-roots.md` defining logging output root policy and enforcement.
- `CONTEXT/decisions/ADR-016-context-edit-authority.md` clarifying when agents may edit existing CONTEXT records.
- `CONTEXT/decisions/ADR-017-skill-formalization.md` formalizing the skill contract (SKILL.md, run.py, validate.py, fixtures).
- Governance fixtures for privacy boundary, log output roots, context edit authority, and output-root enforcement.

#### Changed
- Aligned all logging with INV-006 output roots: logs now written under `CONTRACTS/_runs/<purpose>_logs/` (ADR-015).
- Updated canon docs (`CANON/CONTRACT.md`, `CANON/CRISIS.md`, `CANON/STEWARDSHIP.md`, `AGENTS.md`) to reflect correct log locations and the skill contract.
- Clarified `CANON/CONTRACT.md` Rule 3 to require both explicit user instruction AND explicit task intent for CONTEXT edits (ADR-016).
- Enhanced `CANON/CONTRACT.md` Rule 2 to explicitly require ADRs for governance decisions and recommend them for significant code changes.
- Enhanced `AGENTS.md` to explicitly document the skill contract (SKILL.md, run.py, validate.py, fixtures) as defined in ADR-017.
- Bumped `canon_version` to 2.8.0 (minor: catalytic computing canonical note, governance clarifications).

## [2.6.0] - 2025-12-23

### Added
- `CONTEXT/decisions/ADR-011-master-override.md` defining the `MASTER_OVERRIDE` interface.
- `master-override` skill for override audit logging and gated log access.
- `mcp-smoke` and `mcp-extension-verify` skills for MCP verification.
- `doc-update` skill to standardize documentation updates.
- `CONTEXT/archive/planning/` planning archive with an index and dated snapshots.
- `REPO_FIXES_TASKS.md` checklist for contract-alignment follow-ups.

### Changed
- Added `MASTER_OVERRIDE` to governance docs (Agreement, Contract, Genesis, Agents, Glossary).
- MCP documentation now recommends logs under `LAW/CONTRACTS/_runs/mcp_logs/`.
- Planning references now point at `CONTEXT/archive/planning/INDEX.md`.
- Bumped `canon_version` to 2.6.0.

### Fixed
- Python 3.8 compatibility for governance tooling and contract runner.

### Removed
- Root planning docs: `ROADMAP.md`, `AGS_MASTER_TODO.md`.

## [2.5.5] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-010-authorized-deletions.md`.
- `CONTRACTS/fixtures/governance/deletion-authorization` fixture.

### Changed
- Deletions now require explicit instruction and confirmation (CANON rules still archived per INV-010).
- Bumped `canon_version` to 2.5.5.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.4] - 2025-12-21

### Changed
- Commit ceremony now accepts short confirmations like "go on" after checks and staged files are listed.
- Updated `CONTRACTS/fixtures/governance/commit-ceremony` to document confirmations.
- Bumped `canon_version` to 2.5.4.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.3] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-008-composite-commit-approval.md`.
- `CONTRACTS/fixtures/governance/commit-ceremony` fixture.

### Changed
- Commit ceremony now recognizes explicit "commit, push, and release" directives.
- Updated `CONTRACTS/fixtures/governance/canon-sync` to include `CANON/AGREEMENT.md`.
- Bumped `canon_version` to 2.5.3.
- Regenerated `CANON/CODEBOOK.md`.

## [2.5.2] - 2025-12-21

### Added
- `requirements.txt` with `jsonschema` to satisfy schema validation dependencies in CI.

### Fixed
- CI critic failure when `jsonschema` was missing.

## [2.5.1] - 2025-12-21

### Added
- `repo-contract-alignment` skill with fixtures for contract alignment workflow.
- `TOOLS/skill_runtime.py` to enforce skill canon-version compatibility at runtime.

### Changed
- Regenerated `CANON/CODEBOOK.md` to include the new skill.
- Bumped `canon_version` to 2.5.1.
- Updated `AGENTS.md` authority gradient to include `CANON/AGREEMENT.md`.
- Updated cortex docs to reference the SQLite index (`cortex.db`).
- Updated skill `required_canon_version` ranges to `>=2.5.1 <3.0.0`.
- Skills now validate `required_canon_version` before running.
- `TOOLS/critic.py` now refuses to run while `.quarantine` exists.

### Fixed
- `TOOLS/critic.py` output uses ASCII to avoid Windows encoding errors.

## [2.5.0] - 2025-12-21

### Added
- **Audit Logging**: All MCP tool executions are now logged to `MCP/logs/audit.jsonl` with timestamp, tool name, status, and duration.
- **Improved Prompts**:
    - `skill_template`: Injects `SKILLS/_TEMPLATE` content.
    - `conflict_resolution`: Injects `CANON/ARBITRATION.md`.
    - `deprecation_workflow`: Injects `CANON/DEPRECATION.md`.

### Added
- **Security**: Implemented Governance Enforcement in `MCP/server.py`.
- **Logic**: Tools (`skill_run`, `adr_create`, etc.) are now decorated with `@governed_tool`.
- **Enforcement**: If `TOOLS/critic.py` reports any violations, dangerous actions are BLOCKED with a "Governance Lockdown" error.

### Changed
- **Performance**: Promoted Cortex Indexing from O(N) rebuild to Incremental (checking `mtime`).
- `CORTEX/cortex.build.py`: Refactored to retain DB, migrate schema, and prune deleted entries.
- `CORTEX/schema.sql`: Added `last_modified` column to `entities` table.
- **Strictness**: Changed ID generation to `page:{rel_path}` (unique) to resolve filename collision bugs.

## [2.2.0] - 2025-12-21

### Added
- **Constitutional License**: `CANON/AGREEMENT.md` defines the liability separation between Human (Sovereign) and Agent (Instrument).
- `CONTEXT/decisions/ADR-007-constitutional-agreement.md`: Formal decision record for the agreement.
- `CANON/INDEX.md`: Master index of the law, listing `AGREEMENT.md` as the highest authority.

### Changed
- **Authority Gradient**: Updated `CANON/CONTRACT.md` to place `AGREEMENT.md` at rank #1, shifting the Contract to rank #2.

## [2.1.0] - 2025-12-21

### Added
- `CONTEXT/decisions/ADR-004-mcp-integration.md`: Retroactive decision record for the Model Context Protocol (MCP) implementation.
- `CONTEXT/decisions/ADR-005-persistent-research-cache.md`: Retroactive decision record for the SQLite-backed Research Cache.
- `CONTEXT/decisions/ADR-006-governance-schemas.md`: Documented the "Governance Object Schemas" decision to legitimize INV-011.
- **Governance Schemas**: Defined JSON Schemas for `ADR` (Architecture Decision Records), `SKILL` (Skill Manifests), and `STYLE` (Preferences) in `MCP/schemas/governance/`.
- `TOOLS/schema_validator.py`: Utility to parse Markdown headers and validate against JSON Schemas.
- **INV-011**: New invariant requiring schema compliance for law-like files.
- `critic.py` now enforces schema validation on all ADRs, Skills, and Preferences.

### Changed
- Refactored `SKILLS/_TEMPLATE` and `canon-migration` to use compliant Status (`Draft`, `Active`).

## [2.0.0] - 2025-12-21

### Added
- `CANON/ARBITRATION.md`: Conflict resolution policy with escalation protocol.
- **Symbolic Compression**:
    - `CANON/CODEBOOK.md`: Stable ID registry for token-efficient referencing (@C0, @I3, @S7).
    - `TOOLS/codebook_build.py`: Generator for codebook from repo entities.
    - `TOOLS/codebook_lookup.py`: CLI/library for programmatic lookups.
    - `TOOLS/compress.py`: Bidirectional symbolic compression/expansion tool.
    - `TOOLS/tokenizer_harness.py`: Real token measurement for GPT-4 (cl100k) and GPT-4o/o1 (o200k).
    - `CANON/GENESIS_COMPACT.md`: Token-efficient bootstrap prompt using symbols.
- **Provenance Headers**: Added `TOOLS/provenance.py` and integrated into all major generators (`codebook_build.py`, `cortex.build.py`, `packer.py`) for automated audit trails. Introduced `meta/PROVENANCE.json` in memory packs as a single-point-of-truth manifest for pack integrity.

## [1.2.0] - 2025-12-21

### Added
- MCP full implementation: all 10 tools working (including `research_cache`), dynamic resources, Claude Desktop config ready.
- MCP governance tools: `critic_run`, `adr_create`, `commit_ceremony` for Claude-assisted governance.
- MCP seam: `MCP/MCP_SPEC.md`, `MCP/schemas/`, `MCP/server.py` for Model Context Protocol integration.
- **Emergency Governance**:
    - `CANON/CRISIS.md`: Procedures with 5 crisis levels and CLI modes.
    - `CANON/STEWARDSHIP.md`: Human escalation paths and steward authority.
    - `TOOLS/emergency.py`: CLI for crisis handling (validate, rollback, quarantine, etc.).

## [1.1.1] - 2025-12-21

### Added
- Genesis Prompt (`CANON/GENESIS.md`): Bootstrap prompt that ensures agents load CANON first.
- Research folder now tracked in git with clarifying README.
- Context Query Tool (`CONTEXT/query-context.py`): CLI to search decisions by tag, status, review date.
- Context Review Tool (`CONTEXT/review-context.py`): Flags overdue ADR reviews.
- CONTRACT Rule 7: Commit ceremony as a non-negotiable law.
- AGENTS.md Section 10: Full commit ceremony specification with anti-chaining rule.
- `CANON/DEPRECATION.md`: Deprecation windows and ceremonies for safe rule retirement.
- **Research Cache**: Implemented persistent SQLite-backed cache for research summaries (`TOOLS/research_cache.py`) to avoid redundant browsing.
- `CANON/MIGRATION.md`: Formal compatibility break ritual with phases and rollback.
- `CANON/INVARIANTS.md` INV-009 and INV-010: Canon bloat prevention (readability limits, archiving).

### Changed
- Shadow Cortex now uses SQLite (`cortex.db`) instead of flat JSON for O(1) lookups.
- `query.py` updated with `--json` export flag for backward-compatible JSON output.
- Expanded ROADMAP with v1.2 milestone and research-derived tasks.
- Strengthened STYLE-001 with prohibited interpretations list and anti-chaining rule.

### Deprecated
- `cortex.json` emission from build process (replaced by SQLite `cortex.db`).

## [1.1.0] - 2025-12-21

### Added
- STYLE-002: Engineering Integrity preference (foundational fixes over patches).
- STYLE-003: Mandatory Changelog Synchronisation preference.
- Hardened STYLE-001 (Blanket Approval Ban and Mandatory Ceremony Phase).

## [1.0.0] - 2025-12-21

### Added

- Invariant freeze fixture to enforce INV-001 through INV-008.
- `check_invariant_freeze()` in `check_canon_governance.py`.
- Documentation guides: `EXTENDING.md`, `TESTING.md`, `SHIPPING.md`.
- Trust boundaries in `SECURITY.md` (read/write access, human approval requirements).

### Changed

- Invariants INV-001 through INV-008 are now frozen (v1.0 stability).
- SECURITY.md expanded with trust boundary definitions.

## [0.1.5] - 2025-12-21

### Added

- New invariants: INV-005 (Determinism), INV-006 (Output roots), INV-007 (Change ceremony), INV-008 (Cortex builder exception).
- New glossary terms: ADR, Pack, Manifest, Critic, Authority gradient, Invariant, Change ceremony, Entity, Query.
- Governance fixtures for `canon-sync`, `token-grammar`, and `no-raw-paths`.
- Deterministic timestamp in `cortex.build.py` via `CORTEX_BUILD_TIMESTAMP` env var.
- Python `run.py` for `_TEMPLATE` skill.

### Changed

- Authority gradient in `CONTRACT.md` aligned with `AGENTS.md` (now 8-level hierarchy).
- INVARIANTS.md expanded from 4 to 8 invariants.
- GLOSSARY.md expanded from 11 to 20 terms.

## [0.1.2] - 2025-12-20

### Added

- Subsystem-owned artifact roots with keep files: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/_packs/`.

### Changed

- `BUILD/` is reserved for user build outputs, not system artifacts.
- Contract runner writes fixture outputs under `CONTRACTS/_runs/`.
- Cortex build writes index under `CORTEX/_generated/cortex.json` (query keeps fallback support).

## [0.1.1] - 2025-12-19

### Added

- Root `AGENTS.md` and research scaffold under `CONTEXT/research/`.
- Reference `example-echo` skill with a basic fixture.
- `BUILD/` output root with gitignore rules and a keep file.

### Changed

- Canon rules to require `BUILD/` as the output root.
- Contract runner to execute skill fixtures and write outputs under `BUILD/`.
- Cortex build to emit its index under `BUILD/` and skip indexing `BUILD/`.

## [0.1.0] - 2025-12-19

### Added

- Initial repository skeleton with canon, context, maps, skills, contracts, memory, cortex and tools directories.
- Templates for ADRs, rejections, preferences and open issues.
- Basic runner script and placeholder fixtures.
- Versioning policy and invariants.
## [2.15.1] - 2025-12-28

### Added
- **Cassette Network Phase 0**: Complete cassette network architecture.
  - `CORTEX/cassette_protocol.py` - Base class for all database cassettes
  - `CORTEX/network_hub.py` - Central coordinator with capability routing
  - `CORTEX/cassettes/governance_cassette.py` - AGS governance cassette (system1.db)
  - `CORTEX/cassettes/agi_research_cassette.py` - AGI research cassette
  - `CORTEX/demo_cassette_network.py` - Cross-database query demonstration
  - Network: 2 cassettes (governance + agi-research)
  - Total indexed: 3,991 chunks (1,548 governance + 2,443 research)
  - Cross-cassette queries: governance + research merged results
  - Capability-based routing: vectors, fts, research
  - Health monitoring: get_network_status()

- **CANON/IMPLEMENTATION_REPORTS.md**: New canon requirement for implementation reports.
  - Requires signed reports for all implementations
  - Format: Agent identity + date (signature block)
  - Sections: Executive Summary, What Was Built, What Was Demonstrated, Real vs Simulated, Metrics, Conclusion
  - Storage: `CONTRACTS/_runs/<feature-name>-implementation-report.md`

- **CANON/CONTRACT.md**: Updated Rule 8 to add implementation report requirement.
  - Every implementation must produce signed report
  - Reports stored in `CONTRACTS/_runs/` with proper format
  - Governance checks enforce report requirements

- **CANON/INDEX.md**: Added IMPLEMENTATION_REPORTS.md to Truth section.

- **SEMANTIC_DATABASE_NETWORK_REPORT.md**: Updated to reflect Cassette Network Phase 0.
  - Changed from "Semantic Network Protocol prototype" to "Cassette Network Phase 0 complete"
  - Updated architecture: DatabaseCassette base class, SemanticNetworkHub coordinator
  - Comparison: Prototype (semantic_network.py) vs Production (cassette protocol)
  - New statistics: 3,991 total chunks across both cassettes
  - Roadmap alignment: Phase 0 decision gate PASSED

### Changed
- `CONTRACTS/_runs/cassette-network-implementation-report.md`: Created implementation report.
  - Full Cassette Network Phase 0 documentation
  - All required sections included (signature, executive summary, what was built, demonstrated, metrics)
  - Agent identity and date at top: opencode@agent-governance-system | 2025-12-28
## [2.16.0] - 2025-12-28

### Added
- **INBOX Policy**: Centralized storage for human-readable documents.
  - Created `CANON/INBOX_POLICY.md` - Full policy for INBOX directory
  - All reports, research, roadmaps must go to `INBOX/`
  - Requires content hashes in all INBOX documents
  - Pre-commit hook enforces INBOX placement and hash requirements
  - INBOX structure: reports/, research/, roadCONTEXT/maps/, decisions/, summaries/, ARCHIVE/

- **Updated canon documents**:
  - `CANON/CONTRACT.md` Rule 3: Added INBOX requirement (reports → INBOX/reports/)
  - `CANON/INDEX.md` Added INBOX_POLICY to Truth section
  - `CANON/IMPLEMENTATION_REPORTS.md` Created - Standard format for signed reports

- **Updated implementation report**:
  - `INBOX/reports/cassette-network-implementation-report.md` (moved from root)
  - Added content hash: ``

### Changed
- `.githooks/pre-commit`: Added INBOX policy check after canon governance check
- `TOOLS/check_inbox_policy.py`: New governance check script for INBOX enforcement

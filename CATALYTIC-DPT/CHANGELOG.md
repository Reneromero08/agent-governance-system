# CATALYTIC-DPT Changelog

All notable changes to the Catalytic Computing Department (Isolated R&D) will be documented in this file.

## [1.60.0] - 2025-12-27

### Phase 6.4: MCP Adapter Contracts
- **Added** `SCHEMAS/mcp_adapter.schema.json`: Strict schema for MCP adapter execution (caps, transcript hashing).
- **Added** `SKILLS/mcp-adapter/`: Official skill wrapper for MCP servers.
- **Added** `SKILLS/mcp-adapter/scripts/wrapper.py`: Subprocess executor enforcing strict governance caps.
- **Verified** `TESTBENCH/test_mcp_adapter_schema.py` and `test_ags_phase6_mcp_adapter_e2e.py`: Verified schema and runtime governance.

### Phase 6.5: Skill Registry (Hash-Addressed Capabilities)
- **Added** `SKILLS/registry.json`: Canonical, versioned mapping of Skill IDs to Capability Hashes.
- **Added** `PRIMITIVES/skills.py`: Skill registry loading, resolution, and strict integrity verification.
- **Verified** `TESTBENCH/test_skills_registry.py`: Tests for registry loading, resolution, and integrity checks (fail-closed).
- **Changed** `TOOLS/ags.py`: Integrated `skill_id` resolution into routing logic.

### Phase 6.6: Capability Pinning & Revocation
- **Added** `CAPABILITY_PINS.json`: Explicit allowlist for capability hashes (immutable by default).
- **Added** `CAPABILITY_REVOKES.json`: Explicit denylist for emergency capability revocation.
- **Verified** `TESTBENCH/test_capability_revocation.py`: Adversarial test confirming revocation enforcement fails closed.
- **Verified** `ags.py` enforcement: Precedence rules (Revoke > Pin) verified.

### Phase 6.7: Registry Immutability & Safety
- **Verified** `PRIMITIVES/registry_validators.py` enforces canonical JSON, sorted keys, and no duplicates.
- **Verified** `TESTBENCH/test_registry_validators.py`: Adversarial tests for duplicate hashes, non-canonical encoding, and tamper detection.
- **Enforced** `ags.py`: `route` and `verify` fail closed on any registry violation.

### Performance
- **Optimized** `PRIMITIVES/cas_store.py`: Eliminated redundant file copy in `put_stream` (100% write IO reduction).

## [1.55.0] - 2025-12-26

### Swarm Activation (Nervous System Online)

#### Changed
- `swarm_config.json`:
  - Assigned `LFM2-2.6B-Exp` (Python Runner) as the canonical Ant Worker.
  - Assigned `Claude` as the President (reverted from Brains).
- `LAB/MCP/mcp_client.py`: Verified functional connectivity (President ↔ Governor ↔ Ledger).
- `LAB/MCP/server.py`: Verified ledger persistence for dispatch task queue.

#### Added
- `SKILLS/ant-worker/scripts/lfm2_runner.py`: Direct Python execution script for LFM2 using Transformers.
- `SKILLS/ant-worker/scripts/ant_agent.py`: MCP-aware agent daemon that polls ledger and invokes LFM2.

#### Fixed
- `SKILLS/swarm-orchestrator/scripts/poll_and_execute.py`: Corrected MCP import path and added JSON error recovery.
- `SKILLS/swarm-orchestrator/scripts/launch_swarm.ps1`: Fixed hardcoded paths and script locations.

#### Verified
- Nervous System: successful dispatch-perform-report loop via MCP ledger (task `TEST-001`).
- Brain (Cortex): successful indexing and search functionality.
## [1.52.0] - 2025-12-26

### LAB Compression (Cortex-Style Merge)

#### Changed
- Merged `ARCHITECTURE/ORCHESTRATION.md` + `ARCHITECTURE/RECURSIVE_SWARM.md` → `ARCHITECTURE/SWARM_ARCHITECTURE.md`
- Merged `RESEARCH/SEMIOTIC_COMPRESSION_LAYER_REPORT.md` + `RESEARCH/SEMIOTIC_COMPRESSION_ROADMAP_PATCH.md` → `RESEARCH/SEMIOTIC_COMPRESSION.md`
- Added Cortex-style document hashes for versioning (`SHA256:SWARM_ARCH_V1`, `SHA256:SCL_SPEC_V1`)

#### Moved
- `RESEARCH/CATALYTIC_STACK_COMPRESSED_MERGED GPT 5.2.md` → `ARCHIVE/` (superseded)

#### Removed
- Duplicate/superseded architecture and research files

---

## [1.51.0] - 2025-12-26

### LAB Reorganization

#### Changed
- Moved architecture docs to `LAB/ARCHITECTURE/`:
  - `ORCHESTRATION_ARCHITECTURE.md` → `ARCHITECTURE/ORCHESTRATION.md`
  - `RECURSIVE_SWARM_ARCHITECTURE.md` → `ARCHITECTURE/RECURSIVE_SWARM.md`
- Moved `ROADMAP_PATCH_SEMIOTIC.md` → `RESEARCH/SEMIOTIC_COMPRESSION_ROADMAP_PATCH.md`

#### Removed
- `ROADMAP_PATCH_SEMIOTIC_TESTBENCH.md` (duplicate of ROADMAP_PATCH_SEMIOTIC.md)

#### Added
- `LAB/README.md`: Index documenting LAB directory structure

---

## [1.50.0] - 2025-12-26

### Phase 6.7: Registry Immutability Backstop

#### Added
- `PRIMITIVES/registry_validators.py`: strict, deterministic validation for `CAPABILITIES.json` and `CAPABILITY_PINS.json` (duplicates/noncanonical/tamper fail closed).
- `TESTBENCH/test_ags_phase6_registry_immutability.py`: adversarial coverage for duplicate hashes, non-canonical JSON, and tamper detection.

#### Changed
- `TOOLS/ags.py`: fail-closed on malformed/non-canonical/tampered registries during routing.
- `PIPELINES/pipeline_verify.py`: fail-closed on malformed/non-canonical/tampered registries during verification.

---

## [1.53.0] - 2025-12-26

### Phase 6.8: Capability Versioning Semantics

#### Changed
- Route/verify surfaces now report `CAPABILITY_HASH_MISMATCH` when a capability hash cannot be re-derived from the registry adapter spec bytes (internal detection remains registry-based).

#### Added
- `TESTBENCH/test_ags_phase6_capability_versioning_semantics.py`: asserts fail-closed behavior and the dedicated boundary error code.

---

## [1.54.0] - 2025-12-26

### Phase 6.9: Capability Revocation Semantics

#### Added
- `CAPABILITY_REVOKES.json`: deterministic revoked capability list.
- `TESTBENCH/test_ags_phase6_capability_revokes.py`: route-time rejection and verify-time semantics preserving historical verification.

#### Changed
- `TOOLS/ags.py`: rejects revoked capabilities at route time (`REVOKED_CAPABILITY`).
- `PIPELINES/pipeline_verify.py`: rejects post-revocation pipelines that use revoked capabilities (`REVOKED_CAPABILITY`), while allowing legacy pipelines without policy snapshots.

---

## [1.55.0] - 2025-12-26

### Phase 7.0: Pipeline DAG Scheduling

#### Added
- `PIPELINES/pipeline_dag.py`: deterministic DAG spec parsing, topological scheduling, resume-safe receipts, and fail-closed verification integration.
- `TESTBENCH/test_pipeline_dag.py`: DAG happy path, resume, tamper rejection, and cycle detection.
- `catalytic pipeline dag run|status|verify`: minimal CLI surface for artifact-only DAG workflows.
- Deterministic topo sort tie-break: lexicographic node order.

---

## [1.56.0] - 2025-12-26

### Phase 7.1: Distributed Execution Receipts

#### Added
- `PIPELINES/pipeline_dag.py`: node receipts (RECEIPT.json), chained via DAG topology, with strict verification.
- `TESTBENCH/test_pipeline_dag.py`: receipt chaining, tamper rejection, and cross-machine portability coverage.

---

## [1.57.0] - 2025-12-26

### Phase 7.2: Multi-node Restore Runner

#### Added
- `PIPELINES/pipeline_dag.py`: receipt-gated DAG restore with deterministic decisions and fail-closed verification.
- `TESTBENCH/test_pipeline_dag.py`: restore no-op, missing artifact rerun, tampered receipt rerun, and portability coverage.

---

## [1.47.0] - 2025-12-26

### Phase 6.4: MCP Adapters as Governed Pipeline Steps

#### Changed
- `PIPELINES/pipeline_verify.py`: re-hashes durable outputs listed in `OUTPUT_HASHES.json` and fails closed on post-run tampering.

#### Added
- `TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py`: end-to-end adapter step execution using `SKILLS/ant-worker` and strict pipeline verification (including tamper rejection).

---

## [1.48.0] - 2025-12-26

### Phase 6.5: Hash-Addressed Capability Registry

#### Added
- `CAPABILITIES.json`: deterministic registry mapping `capability_hash` to adapter spec (hash-addressed).
- `TESTBENCH/test_ags_phase6_capability_registry.py`: fail-closed tests for resolution, unknown capability rejection, and registry tamper detection.

#### Changed
- `TOOLS/ags.py`: supports `steps[].capability_hash` in plans and resolves it strictly via the registry (unknown/mismatch rejects).
- `PIPELINES/pipeline_verify.py`: enforces registry resolution and exact spec matching when capability hashes are used.

---

## [1.49.0] - 2025-12-26

### Phase 6.6: Capability Pinning Enforcement

#### Added
- `CAPABILITY_PINS.json`: deterministic allowlist of permitted `capability_hash` values.
- `TESTBENCH/test_ags_phase6_capability_pins.py`: route-time and verify-time rejection for unpinned capabilities (fail-closed).

#### Changed
- `TOOLS/ags.py`: rejects known-but-unpinned capabilities at route time (`CAPABILITY_NOT_PINNED`).
- `PIPELINES/pipeline_verify.py`: rejects unpinned capabilities even if present in `CAPABILITIES.json` (`CAPABILITY_NOT_PINNED`).

---

## [1.46.0] - 2025-12-26

### Docs: Roadmap Promotion

#### Changed
- `ROADMAP_V2.3.md` is the canonical roadmap.
- `ROADMAP_V2.2.md` restored as a deprecated stub pointer.

---

## [1.41.0] - 2025-12-26

### Phase 6.1: AGS Bridge (Model-Free Pipelines)

#### Added
- `python -m TOOLS.ags route`: emits deterministic `PIPELINE.json` + `STATE.json` from an explicit JSON plan (idempotent writes).
- `python -m TOOLS.ags run`: runs `catalytic pipeline run` then `catalytic pipeline verify` (fail-closed).
- `TESTBENCH/test_ags_phase6_bridge.py`: subprocess tests for deterministic routing, run+verify success, and tamper rejection.

---

## [1.42.0] - 2025-12-26

### Phase 6.1: Runtime-Owned Pipeline State

#### Changed
- `PipelineRuntime`: initializes `STATE.json` deterministically when `PIPELINE.json` exists but `STATE.json` is missing.
- `ags route`: emits `PIPELINE.json` only; runtime owns state.

---

## [1.43.0] - 2025-12-26

### Phase 6.2: Router Slot (External Plan Producer)

#### Added
- `SCHEMAS/ags_plan.schema.json`: strict plan schema for untrusted router output (no extra fields; capped steps).
- `ags plan`: runs an external router, hard-bounds stdout bytes, rejects any stderr, validates plan + jobspecs, writes canonical plan JSON.
- `ags route`: validates plans (schema + caps) before emitting `PIPELINE.json`.
- `TESTBENCH/test_ags_phase6_router_slot.py`: subprocess tests for router happy path, stderr rejection, byte caps, schema rejection, and jobspec validation.

---

## [1.44.0] - 2025-12-26

### Phase 6.2: Fail-Closed Plan Semantics

#### Changed
- Plans must declare an explicit step `command` (no implicit no-op defaults); missing step command rejects.

---

## [1.45.0] - 2025-12-26

### Phase 6.3: Adapter Contract (Skills + MCP Pipelining)

#### Added
- `SCHEMAS/adapter.schema.json`: strict adapter contract with explicit side effects, bounded deref caps, and required artifact hashes.
- Adapter validation in AGS: strict-mode rejection on any side effects, non-normalized paths, input/output overlap, or deref caps above global ceilings.
- Plan schema supports adapter steps (`steps[].adapter`) for pipeline-safe skill/MCP wrappers.
- `TESTBENCH/test_ags_phase6_adapter_contract.py`: fail-closed tests for missing command, side effects, deref caps, non-normalized paths, and nondeterministic adapters.

---

## [1.40.0] - 2025-12-26

### Phase 5: Pipeline Verify CLI (Fail-Closed)

#### Added
- `catalytic pipeline verify --pipeline-id <id>`: mechanical, artifact-only verification of a full pipeline run.
- `CATALYTIC-DPT/PIPELINES/pipeline_verify.py`: verifies `CHAIN.json`, per-step required artifacts, proof_hash integrity (if present), and schema-valid LEDGER.jsonl.
- `CATALYTIC-DPT/TESTBENCH/test_pipeline_verify_cli.py`: verifies OK, missing artifacts, chain tamper, and ledger corruption cases via the CLI.

---

## [1.38.0] - 2025-12-26

### Phase 3: Packing Hygiene (Deterministic, Bounded, Deduplicated)

#### Changed
- **MEMORY/LLM_PACKER/Engine/packer.py**:
  - Enforces deterministic pack generation (no timestamp-derived output; deterministic stamps by repo digest prefix).
  - Enforces explicit pack ceilings (max_total_bytes, max_entry_bytes, max_entries) and fails closed if exceeded.
  - Rejects duplicate normalized paths and (for CAT-DPT packs by default) duplicate content hashes.
  - Makes manifest auditable and content-addressed (`PACK_INFO.repo_state_sha256`) and verifies refs on pack validation.

#### Added
- **MEMORY/LLM_PACKER/Engine/pack_hygiene.py**: pure hygiene helpers (manifest validation, limit enforcement, canonical hashing).
- **TESTBENCH/test_packing_hygiene.py**: determinism + bounds + dedup + tamper detection backstop tests.

---

## [1.39.0] - 2025-12-26

### Phase 5: Verifiable Pipeline Proof Chain

#### Added
- `CONTRACTS/_runs/_pipelines/<pipeline_id>/CHAIN.json`: deterministic, artifact-only proof chain across pipeline steps.
- `CATALYTIC-DPT/PIPELINES/pipeline_chain.py`: fail-closed verifier that recomputes step proof/root hashes and checks link integrity and step order.
- `CATALYTIC-DPT/TESTBENCH/test_pipeline_chain.py`: valid, tamper, reorder, and determinism coverage.

#### Changed
- `CATALYTIC-DPT/PIPELINES/pipeline_runtime.py`: writes/updates `CHAIN.json` during step completion and refuses to resume if chain integrity fails.

---

## [1.37.0] - 2025-12-25

### Phase 4: Adversarial Fixtures (Fail-Closed Hardening)

#### Added
- Adversarial test coverage for CAS corruption/truncation/partial writes, ledger corruption, path traversal injection, proof/manifest tampering, and pipeline resume safety.

---

## [1.36.0] - 2025-12-25

### Phase 2: Demo of Measurable Reuse via Hash-First Dereference

#### Added
- **CATALYTIC-DPT/DEMOS/memoization_hash_reuse/**:
  - Deterministic demo runner (`run_demo.py`) that produces baseline vs reuse artifacts.
- **CONTRACTS/_runs/_demos/memoization_hash_reuse/**:
  - Artifact-backed baseline/reuse evidence (PROOF.json, LEDGER.jsonl, bounded deref stats) and a comparison table.

---

## [1.35.0] - 2025-12-25

### Phase 5: Artifact-Only Resumable Pipelines

#### Added
- **PIPELINES/pipeline_runtime.py**:
  - Deterministic pipeline init/run/status with resume-safe state under `CONTRACTS/_runs/_pipelines/<pipeline_id>/`.
- **TOOLS/catalytic.py**:
  - Adds `catalytic pipeline run|status` commands (no timestamps; state is artifact-only).
- **TESTBENCH/test_pipelines.py**:
  - Proves deterministic init, resume without re-running completed steps, required per-step artifacts, and stable status output.

---

## [1.34.0] - 2025-12-25

### Phase 2: Deterministic Job Memoization (Never Pay Twice)

#### Added
- **PRIMITIVES/memo_cache.py**:
  - Deterministic job cache key (JobSpec canonical JSON + input domain roots + validator identity + strictness).
  - Cache storage under `CONTRACTS/_runs/_cache/jobs/<job_cache_key>/` with cached proof/domain roots/output hashes and materialized durable outputs.
- **TESTBENCH/test_memoization.py**:
  - Proves miss→hit behavior, byte-identical proof/domain roots on hits, and deterministic invalidation when strictness changes.

#### Changed
- **TOOLS/catalytic_runtime.py**:
  - Adds memoization path that skips execution on cache hits while restoring outputs and re-emitting proof artifacts.

---

## [1.33.0] - 2025-12-25

### Phase 4: Fix Ledger Schema `$ref` Resolution (Draft7)

#### Changed
- **SCHEMAS/ledger.schema.json**:
  - Makes internal `$ref` targets explicit (`ledger.schema.json#/definitions/...`) so resolution is stable under Draft7 reference loading.
- **TESTBENCH/test_schemas.py**:
  - Uses Draft7 `RefResolver` + deterministic schema store for ledger validation (no unsupported `registry=` argument).

---

## [1.32.0] - 2025-12-25

### Phase 4: Independent Verifier Implementation; Interop Proven

#### Added
- **PRIMITIVES/verify_bundle_alt.py**:
  - Code-independent SPECTRUM-05 bundle + chain verifier implementation (strict identity/signature enforcement).
- **TESTBENCH/test_verifier_interop.py**:
  - Golden interop fixtures (valid bundle + tamper rejection) and deterministic rerun coverage.
  - Enforces byte-identical serialized verification results across both implementations.

---

## [1.31.0] - 2025-12-25

### Phase 4: Ledger Observability for Expand-by-Hash Dereferences

#### Added
- **PRIMITIVES/hash_toolbelt.py**:
  - `log_dereference_event()`: Deterministic ledger logging for hash dereference commands.
  - `_build_dereference_ledger_record()`: Constructs minimal schema-conforming ledger records for dereference events.
  - Logs include requested hash, command name (read/grep/describe/ast), and bounds (max_bytes, ranges, matches, nodes, depth as applicable).
  - Logging is opt-in: only occurs when `--run-id` is provided.
- **TOOLS/catalytic.py**:
  - `--timestamp` parameter for deterministic timestamp injection (defaults to sentinel).
  - Ledger logging integrated at command dispatch for all four hash toolbelt commands.
- **TESTBENCH/test_deref_logging.py**:
  - Tests for deterministic rerun (identical ledger bytes), schema conformance, opt-in behavior (no run-id → no logging), and exact bounds recording.

#### Changed
- **ROADMAP_V2.2.md**:
  - Marked "dereference events logged to ledger" as DONE under Phase 1X status.

---

## [1.30.0] - 2025-12-25

### Phase 1V: CI Enforces Strict Verification by Default

#### Changed
- **.github/workflows/contracts.yml**:
  - Adds a strict-mode SPECTRUM-05 verification step that generates a deterministic signed bundle and verifies it with `--strict` under `CI=true`.
  - Runs CAT-DPT `pytest` in CI with a workspace-pinned temp directory (avoids capture temp failures) and ignores `CATALYTIC-DPT/LAB` + `MEMORY/LLM_PACKER/_packs`.
- **TOOLS/catalytic_verifier.py**:
  - Refuses to run without `--strict` when `CI` is set (prevents silent downgrade paths in CI).

---

## [1.29.0] - 2025-12-25

### Phase 1X: Expand-by-Hash Toolbelt (Bounded Read/Grep/Ast/Describe)

#### Added
- **TOOLS/catalytic.py**:
  - Unified `catalytic hash` CLI with bounded subcommands: `read`, `grep`, `describe`, `ast`.
  - Hash-first dereference: operates only on CAS objects by SHA-256 (no path reads).
  - Requires explicit CAS location via `--run-id` or `--cas-root`.
- **PRIMITIVES/hash_toolbelt.py**:
  - Deterministic, bounded implementations for read/grep/describe/ast (Python-only AST; otherwise `UNSUPPORTED_AST_FORMAT`).
- **TESTBENCH/test_hash_toolbelt.py**:
  - Tests for bounds enforcement, range reads, deterministic outputs, match limits, AST truncation, and invalid hash rejection.

---

## [1.28.0] - 2025-12-25

### Phase 1.P: Proof Generation Wired to CAS/Merkle/Ledger; Determinism Proven

#### Changed
- **PRIMITIVES/restore_proof.py**:
  - Domain root hash now computed via Phase 1 Merkle primitive (with deterministic empty-manifest sentinel) instead of ad-hoc concatenation.
  - Adds helpers for canonical JSON bytes and CAS-backed domain manifest computation.
- **TOOLS/catalytic_runtime.py**:
  - Snapshots now compute bytes hashes via CAS (streaming, idempotent) and normalize paths deterministically.
  - `DOMAIN_ROOTS.json` now uses Merkle roots per domain (deterministic serialization).
  - `LEDGER.jsonl` now appends schema-valid records via Phase 1 Ledger (canonical JSONL; caller-supplied deterministic timestamp sentinel).
  - `PROOF.json` now uses canonical JSON bytes and references hashes for jobspec + ledger.

#### Added
- **TESTBENCH/test_proof_wiring.py**:
  - Rerun determinism test: two independent runs emit byte-identical `PROOF.json` and `DOMAIN_ROOTS.json`.
  - Tamper detection: hash mismatch is detected and fails closed.

---

## [1.27.0] - 2025-12-25

### Phase 1.D: Append-Only Ledger Receipts Implemented

#### Added
- **PRIMITIVES/ledger.py**:
  - Append-only JSONL writer/reader for ledger entries; never generates timestamps (caller must supply deterministic `RUN_INFO.timestamp`).
  - Deterministic per-line serialization (UTF-8, no whitespace, lexicographically sorted keys).
  - Schema validation against `SCHEMAS/ledger.schema.json` (Draft-07), including optional `JOBSPEC` for job_id linkage.
  - Truncation/rewrite detection via monotonic file-size invariants across appends.
- **TESTBENCH/test_ledger.py**:
  - Append order preservation, deterministic serialization, schema rejection, corrupt/partial line detection, and adversarial truncation detection.

---

## [1.26.0] - 2025-12-25

### Phase 1.M: Merkle Roots per Domain Implemented

#### Added
- **PRIMITIVES/merkle.py**:
  - Deterministic Merkle root computation for domain manifests `{ normalized_path -> sha256_hex }`.
  - Strict path normalization (reuses CAS `normalize_relpath`) and fail-closed validation for malformed paths/hashes.
  - Deterministic leaf ordering (lexicographic by normalized_path) and standard odd-leaf padding.
  - Adversarial rejection: non-normalized paths and duplicate hash bound to different paths.
- **TESTBENCH/test_merkle.py**:
  - Determinism tests (order-independence, stable roots across runs).
  - Adversarial tests for non-normalized paths, invalid hashes, duplicate-hash binding, and odd-leaf padding correctness.

---

## [1.25.0] - 2025-12-25

### Implemented Restore Runner per SPECTRUM-06 (Gated by SPECTRUM-05 Strict Acceptance)

#### Added
- **PRIMITIVES/restore_runner.py**:
  - Implements `restore_bundle()` and `restore_chain()` exactly per SPECTRUM-06 (preflight/plan/execute/verify), including reject-if-exists, staging+rename, deterministic ordering, and rollback rules.
  - Enforces strict SPECTRUM-05 verification gating and PROOF.json verified=true requirement.
  - Emits SPECTRUM-06 frozen success artifacts (`RESTORE_MANIFEST.json`, `RESTORE_REPORT.json`) with canonical JSON bytes and invariant checks.
  - Implements restore-specific failure codes and deterministic selection rules.
- **TOOLS/catalytic_restore.py**:
  - Minimal CLI for restoring a single bundle or explicit chain (JSON output, nonzero exit on failure).
- **TESTBENCH/test_restore_runner.py**:
  - Restore-specific tests for gating, path safety, reject-if-exists, rollback cleanup, success artifacts, and chain all-or-nothing behavior.

---

## [1.24.0] - 2025-12-25

### SPECTRUM-06: Restore Runner Semantics Frozen

#### Added
- **SPECTRUM/SPECTRUM-06.md** (NEW):
  - SPECTRUM-06: Frozen specification for Restore Runner semantics.
  - SPECTRUM-06: restore result artifacts frozen
  - SPECTRUM-06: restore failure codes and threat model frozen
  - Defines eligibility rules: SPECTRUM-05 strict verification required, PROOF.json verified=true, OUTPUT_HASHES.json present.
  - Restore target model: explicit restore_root, path safety rules, traversal rejection.
  - Single-bundle restore: 4-phase procedure (preflight, plan, execute, verify).
  - Chain restore: per-run subfolder isolation, deterministic order, chain-level atomicity.
  - Overwrite policy: reject-if-exists (no implicit overwrites).
  - Atomicity model: staging directory + final rename.
  - Error codes: RESTORE_INELIGIBLE, PATH_ESCAPE_DETECTED, TARGET_EXISTS, etc.

---

## [1.23.0] - 2025-12-25

### Stability Lock: Verifier API/CLI Frozen and Error Codes Centralized

#### Added
- **PRIMITIVES/verify_bundle.py**:
  - Centralized `ERROR_CODES` constant map (one source of truth for all SPECTRUM-05 errors).
  - Mandatory Ed25519 dependency enforcement: Returns `ALGORITHM_UNSUPPORTED` if `cryptography` library is missing in strict mode.
- **TESTBENCH/test_verifier_freeze.py**: New test suite for stable API guarantees.

#### Changed
- **PRIMITIVES/verify_bundle.py**:
  - **API Surface Freeze**: `verify_bundle_spectrum05` and `verify_chain_spectrum05` now have stable signatures and return shapes.
  - **Stable Return Shape**: `{ok: bool, code: str, details: dict}`.
  - **Fail-Fast**: Verification now terminates immediately on the first error encountered in Phase 9 (Output Hash Verification).
- **TOOLS/catalytic_verifier.py**:
  - **CLI Surface Freeze**: Updated `verify_single_bundle`, `verify_chain`, and `verify_chain_from_directory` to use stable SPECTRUM-05 APIs.
  - Minimal output in JSON mode (no extra logs).
  - Nonzero exit code on any verification failure correctly preserved.
- **ROADMAP_V2.1.md**:
  - Updated Phase 1.5 status to explicitly reflect verifier stability lock.

#### Removed
- **PRIMITIVES/verify_bundle.py**:
  - Deprecated multi-error collection in Phase 9 in favor of spec-conformant fail-fast (single error return).

#### Documentation
- **SPECTRUM/SPECTRUM-05.md**:
  - Restored to exact frozen state from commit 3b281f6 (removed non-normative Section 12).
- **PRIMITIVES/VERIFYING.md** (NEW):
  - Created non-normative implementation guide containing extracted implementation requirements.
  - Clearly labeled as non-normative; does not modify SPECTRUM law.
  - Includes notes on mandatory Ed25519 dependency, offline verification, deterministic canonicalization, and verifier modes.

## [1.22.0] - 2025-12-25

### Verifier Updated to Enforce SPECTRUM-04/05 Identity, Canonicalization, and Verification Law

#### Added
- **PRIMITIVES/verify_bundle.py**: SPECTRUM-04/05 enforcement layer
  - `verify_bundle_spectrum05()`: Full 10-phase verification per SPECTRUM-05 v1.0.0
  - `verify_chain_spectrum05()`: Chain verification with chain_root computation
  - Canonical JSON serialization per SPECTRUM-04 v1.1.0 Section 4
  - Bundle root computation per SPECTRUM-04 v1.1.0 Section 5
  - Chain root computation per SPECTRUM-04 v1.1.0 Section 6
  - Ed25519 signature verification per SPECTRUM-04 v1.1.0 Section 9
  - validator_id derivation and verification (SHA-256 of public_key)

#### SPECTRUM-05 10-Phase Verification
- **Phase 1:** Artifact presence check (7 required artifacts)
  - TASK_SPEC.json, STATUS.json, OUTPUT_HASHES.json, PROOF.json
  - VALIDATOR_IDENTITY.json, SIGNED_PAYLOAD.json, SIGNATURE.json
- **Phase 2:** Artifact parse check (JSON validity)
- **Phase 3:** Identity verification
  - Exactly 3 fields: algorithm, public_key, validator_id
  - algorithm == "ed25519" (no alternatives)
  - public_key: exactly 64 lowercase hex characters
  - validator_id derivation matches sha256(public_key_bytes)
- **Phase 4:** Bundle root computation
  - Preimage: {"output_hashes":{...},"status":{...},"task_spec_hash":"..."}
  - task_spec_hash from raw TASK_SPEC.json bytes (not canonicalized)
  - All JSON objects canonicalized with sorted keys, no whitespace
- **Phase 5:** Signed payload verification
  - Exactly 3 fields: bundle_root, decision, validator_id
  - bundle_root matches computed value
  - decision == "ACCEPT" (no alternatives)
  - validator_id matches VALIDATOR_IDENTITY.validator_id
- **Phase 6:** Signature verification
  - payload_type == "BUNDLE"
  - signature: exactly 128 lowercase hex characters
  - validator_id matches across all artifacts
  - Signature message: "CAT-DPT-SPECTRUM-04-v1:BUNDLE:<canonical_payload>"
  - Ed25519 verification using cryptography library
- **Phase 7:** Proof verification
  - PROOF.json.restoration_result.verified == true (boolean)
  - Condition must be RESTORED_IDENTICAL for acceptance
- **Phase 8:** Forbidden artifact check
  - Rejects if logs/, tmp/, or transcript.json exists
- **Phase 9:** Output hash verification
  - All declared outputs must exist
  - SHA-256 hashes must match exactly
- **Phase 10:** Acceptance (all phases pass → ACCEPT)

#### Chain Verification (SPECTRUM-05 Section 6)
- Chain must be non-empty (CHAIN_EMPTY error code)
- No duplicate run_ids (CHAIN_DUPLICATE_RUN error code)
- Each bundle verified via verify_bundle_spectrum05
- Chain root computed: sha256({"bundle_roots":[...],"run_ids":[...]})
- All-or-nothing semantics (any failure rejects entire chain)

#### Error Codes (SPECTRUM-05 Conformance)
- Artifact: `ARTIFACT_MISSING`, `ARTIFACT_MALFORMED`, `ARTIFACT_EXTRA`
- Field: `FIELD_MISSING`, `FIELD_EXTRA`
- Identity: `IDENTITY_INVALID`, `IDENTITY_MISMATCH`, `IDENTITY_MULTIPLE`
- Algorithm: `ALGORITHM_UNSUPPORTED`
- Key: `KEY_INVALID`
- Signature: `SIGNATURE_MALFORMED`, `SIGNATURE_INCOMPLETE`, `SIGNATURE_INVALID`, `SIGNATURE_MULTIPLE`
- Root: `BUNDLE_ROOT_MISMATCH`, `CHAIN_ROOT_MISMATCH`
- Payload: `DECISION_INVALID`, `PAYLOAD_MISMATCH`
- Serialization: `SERIALIZATION_INVALID`
- Proof: `RESTORATION_FAILED`
- Forbidden: `FORBIDDEN_ARTIFACT`
- Output: `OUTPUT_MISSING`, `HASH_MISMATCH`
- Chain: `CHAIN_EMPTY`, `CHAIN_DUPLICATE_RUN`

#### Test Coverage
- **TESTBENCH/test_spectrum_04_05_enforcement.py**: 9 new tests
  - Canonical JSON serialization (sorted keys, no whitespace, UTF-8)
  - Bundle root computation (deterministic, matches spec preimage)
  - Chain root computation (deterministic, order-dependent)
  - Ed25519 signature verification (valid/invalid/tampered)
  - Validator ID derivation (SHA-256 of public_key)
  - SPECTRUM-05 artifact presence check
  - SPECTRUM-05 identity verification (invalid validator_id)
  - SPECTRUM-05 chain empty check
  - SPECTRUM-05 chain duplicate run_id check
- All 69 tests passing (60 legacy + 9 new)

#### Backward Compatibility
- Legacy methods preserved: `verify_bundle()` and `verify_chain()`
- New methods: `verify_bundle_spectrum05()` and `verify_chain_spectrum05()`
- Existing tests unaffected (no breaking changes)

#### Dependencies
- Requires `cryptography` library for Ed25519 signature verification
- Graceful fallback if cryptography not available (tests skip, runtime errors clear)

#### Implementation Notes
- Canonicalization uses `json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False)`
- All hashes lowercase hex (64 characters for SHA-256, 128 for Ed25519 signatures)
- Domain separation: "CAT-DPT-SPECTRUM-04-v1:BUNDLE:" prefix
- Fail-closed: any ambiguity or missing artifact → immediate rejection
- No partial acceptance, no warnings, no recovery paths

## [1.21.0] - 2025-12-25

### SPECTRUM-05: Verification and Threat Law Frozen for Identity-Pinned Acceptance

#### Added
- **SPECTRUM/SPECTRUM-05.md** (v1.0.0): Constitutional specification for verification procedure and threat model
  - Status: FROZEN (no implementation may deviate)
  - Depends on: SPECTRUM-04 v1.1.0

#### Verification Procedure (10 Phases, Step-Ordered)
- **Phase 1:** Artifact presence check (7 required artifacts)
- **Phase 2:** Artifact parse check (JSON validity)
- **Phase 3:** Identity verification (Ed25519, validator_id derivation)
- **Phase 4:** Bundle root computation (exact preimage per SPECTRUM-04)
- **Phase 5:** Signed payload verification (bundle_root, decision, validator_id)
- **Phase 6:** Signature verification (Ed25519 over domain-separated message)
- **Phase 7:** Proof verification (PROOF.json verified=true required)
- **Phase 8:** Forbidden artifact check (logs/, tmp/, transcript.json)
- **Phase 9:** Output hash verification (all declared outputs must verify)
- **Phase 10:** Acceptance (all steps pass → ACCEPT)

#### Required Artifacts (7)
- `TASK_SPEC.json`, `STATUS.json`, `OUTPUT_HASHES.json`, `PROOF.json`
- `VALIDATOR_IDENTITY.json`, `SIGNED_PAYLOAD.json`, `SIGNATURE.json`

#### Acceptance Gating Rules
- Exactly one identity artifact, one payload artifact, one signature artifact
- No forbidden artifacts
- All output hashes verify
- No partial acceptance (ACCEPT/REJECT only)

#### Chain Verification Rules
- Non-empty chain required
- No duplicate run_ids
- All bundles must individually verify
- Chain root computed from bundle_roots and run_ids
- All-or-nothing semantics (any failure rejects entire chain)

#### Threat Model

**Defended:**
- Forged acceptance (Ed25519 signature binding)
- Validator impersonation (cryptographic validator_id derivation)
- Bundle substitution (bundle_root binding)
- Replay attacks on modified artifacts
- Ambiguity-based bypass (fully specified canonicalization)
- Multiple identity injection
- Forbidden artifact smuggling
- Proof bypass

**Not Defended:**
- Private key compromise
- Malicious validator acting within spec
- External coercion/governance failures
- Network-based attacks (artifact-only)
- Side-channel attacks on signing
- Quantum computing attacks (Ed25519 not quantum-resistant)

#### Error Semantics (25 Error Codes)
- Hard rejects only (no warnings, no partial acceptance, no recovery)
- Artifact: `ARTIFACT_MISSING`, `ARTIFACT_MALFORMED`, `ARTIFACT_EXTRA`
- Field: `FIELD_MISSING`, `FIELD_EXTRA`
- Identity: `IDENTITY_INVALID`, `IDENTITY_MISMATCH`, `IDENTITY_MULTIPLE`
- Algorithm: `ALGORITHM_UNSUPPORTED`
- Key: `KEY_INVALID`
- Signature: `SIGNATURE_MALFORMED`, `SIGNATURE_INCOMPLETE`, `SIGNATURE_INVALID`, `SIGNATURE_MULTIPLE`
- Root: `BUNDLE_ROOT_MISMATCH`, `CHAIN_ROOT_MISMATCH`
- Payload: `DECISION_INVALID`, `PAYLOAD_MISMATCH`
- Serialization: `SERIALIZATION_INVALID`
- Proof: `RESTORATION_FAILED`
- Forbidden: `FORBIDDEN_ARTIFACT`
- Output: `OUTPUT_MISSING`, `HASH_MISMATCH`
- Chain: `CHAIN_EMPTY`, `CHAIN_DUPLICATE_RUN`

#### Interoperability Requirements
- Two implementations MUST produce byte-for-byte identical results
- No interpretation required (all rules explicit, unambiguous, complete, testable)
- Divergence between implementations → both suspect, investigation required

## [1.20.0] - 2025-12-25

### SPECTRUM-04: Canonical Byte-Serialization Rules Finalized

#### Changed
- **SPECTRUM/SPECTRUM-04.md** (v1.0.0 → v1.1.0): Removed all ambiguity for byte-level determinism

#### Canonical Serialization Rules (Section 4 - NEW)
- **Encoding:** UTF-8 (no BOM)
- **Newline policy:** No newlines; single line with no trailing newline
- **Whitespace policy:** No whitespace outside string values
- **JSON rules:** Keys sorted lexicographically by UTF-8 byte value; no spaces after colons/commas; RFC 8259 string escaping; integers without decimal points; booleans/null lowercase
- **Field presence:** Strict enforcement; missing field → `FIELD_MISSING`; extra field → `FIELD_EXTRA`

#### Bundle Root (Section 5 - Clarified)
- **Preimage structure:** `{"output_hashes":<object>,"status":<object>,"task_spec_hash":"<64 hex>"}`
- **output_hashes source:** `OUTPUT_HASHES.json` → extract `hashes` field → canonicalize keys
- **status source:** `STATUS.json` → entire content → canonicalize keys
- **task_spec_hash:** `sha256(raw_bytes_of_TASK_SPEC.json)` (NOT canonicalized; raw file bytes)

#### Chain Root (Section 6 - Clarified)
- **Preimage structure:** `{"bundle_roots":[...],"run_ids":[...]}`
- **run_id definition:** Directory name (final path component), not full path
- **Constraints:** Arrays must have identical length; no duplicates → `CHAIN_DUPLICATE_RUN`; empty chain → `CHAIN_EMPTY`

#### Signed Payload (Section 7 - Simplified)
- **Payload types reduced:** `BUNDLE` only (removed `CHAIN`, `ACCEPTANCE`)
- **Timestamp removed:** NOT signed (cannot be standardized without trusted time source)
- **Source of truth:** `SIGNED_PAYLOAD.json` is canonical; verifier reconstructs signature message from it
- **Signed payload:** `{"bundle_root":"...","decision":"ACCEPT","validator_id":"..."}`
- **Signature message:** `CAT-DPT-SPECTRUM-04-v1:BUNDLE:<canonical_payload_json>`

#### Revocation (Section 10.3 - Clarified)
- **Status:** Explicitly OUT OF SCOPE
- **Rule:** Revocation MUST NOT be required for SPECTRUM-04 verification to succeed
- **Artifact-only:** Valid signature remains valid regardless of external revocation state

#### Determinism Proof Checklist (Section 13 - NEW)
- Preimage templates for bundle, chain, signed payload, signature message
- Determinism requirements: byte-for-byte identical outputs for identical inputs
- Reject conditions for any ambiguity
- Informative test vectors for edge cases

#### Error Codes (Updated)
- Added: `DECISION_INVALID`, `FIELD_MISSING`, `FIELD_EXTRA`, `CHAIN_DUPLICATE_RUN`, `CHAIN_EMPTY`
- Removed: `VALIDATOR_UNKNOWN`, `VALIDATOR_REVOKED` (revocation out-of-scope)

## [1.19.0] - 2025-12-25

### Validator Identity Pin - Identity and Signing Law Frozen

#### Added
- **SPECTRUM/SPECTRUM-04.md**: Constitutional specification for validator identity and cryptographic signing
  - Status: FROZEN (no implementation may deviate)
  - Defines binding of bundle/chain acceptance to cryptographic authority

#### Validator Identity Model (Final)
- **Key algorithm:** Ed25519 (singular, no alternatives, no negotiation)
- **Public key encoding:** Raw 32-byte, lowercase hex (64 chars exactly)
- **Validator ID derivation:** `validator_id = sha256(public_key_bytes)` (deterministic, globally unique, stable, offline-verifiable)
- **Identity representation:** `{validator_id, public_key, algorithm: "ed25519"}`

#### Signing Surface (Final)
- **Domain separation:** `CAT-DPT-SPECTRUM-04-v1:` prefix (mandatory)
- **Canonical payload:** Domain prefix + payload type + canonical JSON bytes
- **Payload type:** `BUNDLE` (single type; chains use bundle root semantics)
- **Canonical JSON:** Sorted keys, no whitespace, UTF-8, no extensions
- **Bundle root:** `sha256({output_hashes, status, task_spec_hash})`
- **Chain root:** `sha256({bundle_roots[], run_ids[]})`
- **Signed payload binds:** bundle/chain root, decision, validator_id
- **NOT signed:** logs, transcripts, intermediate state, file paths, semver, build_id, timestamps

#### Signature Format (Final)
- **Signature encoding:** Ed25519, lowercase hex (128 chars exactly)
- **Signature object:** `{payload_type, signature, validator_id}` (signed_at informational only)
- **Malformed detection:** Exact character count, lowercase only, no extra fields

#### Artifact Binding (Final)
- **VALIDATOR_IDENTITY.json:** Contains algorithm, public_key, validator_id
- **SIGNATURE.json:** Contains payload_type, signature, validator_id
- **SIGNED_PAYLOAD.json:** Contains bundle_root, decision, validator_id
- **Verification:** Artifact-only (no network, no external trust, no CA)

#### Mutability and Rotation (Final)
- **Immutable once accepted:** All bundle artifacts byte-for-byte immutable
- **Rotation:** NOT ALLOWED (one identity per validator, forever)
- **Revocation:** Out of scope (not required for acceptance)

#### Fail-Closed Rules
- Any ambiguity rejects
- Multiple identities/keys/signatures reject
- Deviation from canonicalization rejects
- Partial data rejects
- No heuristics (binary ACCEPT/REJECT only)
- No side channels (no timing, no file dates, no network)

#### Error Codes (24 stable codes)
- Identity: `IDENTITY_AMBIGUOUS`, `IDENTITY_INCOMPLETE`, `IDENTITY_INVALID`, `IDENTITY_MISMATCH`, `IDENTITY_MISSING`, `IDENTITY_MULTIPLE`
- Algorithm: `ALGORITHM_UNSUPPORTED`
- Key: `KEY_INVALID`, `KEY_MULTIPLE`
- Signature: `SIGNATURE_INCOMPLETE`, `SIGNATURE_INVALID`, `SIGNATURE_MALFORMED`, `SIGNATURE_MISSING`, `SIGNATURE_MULTIPLE`
- Payload: `PAYLOAD_MISSING`, `PAYLOAD_MISMATCH`
- Root: `BUNDLE_ROOT_MISMATCH`, `CHAIN_ROOT_MISMATCH`
- Fields: `FIELD_MISSING`, `FIELD_EXTRA`, `DECISION_INVALID`
- Chain: `CHAIN_DUPLICATE_RUN`, `CHAIN_EMPTY`
- Serialization: `SERIALIZATION_INVALID`

#### Interoperability
- Two independent implementations MUST produce byte-for-byte identical results
- No interpretation required by implementers
- All rules explicit, unambiguous, complete, testable

## [1.18.0] - 2025-12-25

### Phase 1: Bundle/Chain Verifier (Fail-Closed)

#### Added
- **PRIMITIVES/verify_bundle.py**: Deterministic verifier for SPECTRUM-02 bundles and SPECTRUM-03 chains
  - `BundleVerifier.verify_bundle()`: Single bundle verification with fail-closed semantics
  - `BundleVerifier.verify_chain()`: Chain verification with reference integrity validation
  - Enforces forbidden artifacts: rejects if `logs/`, `tmp/`, or `transcript.json` exist
  - Verification depends ONLY on bundle artifacts, file hashes, and ordering (no logs, no heuristics)
  - Supports optional PROOF.json gating (verified=true required for acceptance)
  - Error codes: `BUNDLE_INCOMPLETE`, `HASH_MISMATCH`, `OUTPUT_MISSING`, `STATUS_NOT_SUCCESS`, `CMP01_NOT_PASS`, `PROOF_REQUIRED`, `RESTORATION_FAILED`, `FORBIDDEN_ARTIFACT`, `INVALID_CHAIN_REFERENCE`

- **TOOLS/catalytic_verifier.py**: CLI entrypoint for bundle/chain verification
  - Single bundle mode: `python catalytic_verifier.py --run-dir <path> [--strict]`
  - Chain mode: `python catalytic_verifier.py --chain <path1> <path2> ... [--strict]`
  - Chain directory mode: `python catalytic_verifier.py --chain-dir <path> [--strict]`
  - JSON output support: `--json` flag for machine-readable reports
  - Exit codes: 0 (pass), 1 (fail), 2 (invalid arguments)

- **TESTBENCH/test_verify_bundle.py**: Comprehensive unit tests (18 tests)
  - Valid bundle acceptance
  - Missing artifacts rejection (TASK_SPEC, STATUS, OUTPUT_HASHES)
  - Hash mismatch detection
  - Missing output detection
  - Status failure detection (STATUS_NOT_SUCCESS, CMP01_NOT_PASS)
  - Proof gating tests (PROOF_REQUIRED, RESTORATION_FAILED)
  - Forbidden artifacts rejection (logs/, tmp/, transcript.json)
  - Chain verification (valid chains, tamper detection, invalid references)

#### Changed
- **TESTBENCH/spectrum/test_spectrum03_chain.py**: Refactored to use new `BundleVerifier` primitive
  - `verify_spectrum03_chain()` now wraps `BundleVerifier.verify_chain()`
  - Maintains backward compatibility with existing tests
  - All 5 SPECTRUM-03 tests pass

#### Test Results
- All 60 tests passing (38 Phase 0 + 18 new + 4 spectrum integration)
- Verifier rejects on every negative case (missing artifacts, tamper, invalid references, forbidden artifacts)
- Chain verification enforces reference integrity (no future references allowed)

#### Exit Criteria Met
- [x] Callable entrypoint: `catalytic_verifier.py` with `--run-dir` and `--chain` modes
- [x] Tests prove: valid acceptance, missing artifact rejection, tamper detection, invalid reference rejection, forbidden artifact rejection
- [x] Deterministic: POSIX path normalization, SHA-256 hashing, stable JSON ordering
- [x] Fail-closed: any ambiguity rejects (no heuristics, no logs, no side channels)
- [x] Verification depends only on bundle artifacts + file hashes + ordering
- [x] All tests pass: `pytest -q` returns 60/60

## [1.17.0] - 2025-12-25

### AGS Integration (entries moved from CANON/CHANGELOG.md)

#### Added
- `CATALYTIC-DPT/CHANGELOG.md`: Isolated changelog for the Catalytic Computing department.
- `CATALYTIC-DPT/swarm_config.json`: Role-to-model mapping configuration for multi-agent swarm.
- `TOOLS/catalytic_runtime.py` - CMP-01 runtime implementing 5-phase catalytic lifecycle (snapshot, execute, verify, record). Works with any command.
- `TOOLS/catalytic_validator.py` - Run ledger validator for CMP-01 compliance checking in CI (restoration proof, output roots).
- `CONTEXT/research/Catalytic Computing/CATALYTIC_IMPLEMENTATION_REPORT.md` - Comprehensive report on working prototype: architecture, PoC, what works well, improvements, integration opportunities.
- `CANON/CATALYTIC_COMPUTING.md` - Canonical note defining catalytic computing for AGS (formal theory, engineering patterns, boundaries).
- `CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md` documenting the decision to add catalytic computing to canon.

#### Changed
- Refactored `CATALYTIC-DPT` documentation to be model-agnostic (replaced specific model names with God/President/Governor/Ant roles).
- Renamed `CODEX_SOP.json` to `GOVERNOR_SOP.json` and `HANDOFF_TO_CODEX.md` to `HANDOFF_TO_GOVERNOR.md` for role alignment.

## [1.16.0] - 2025-12-25

### Phase 0 Complete: Contract Finalization and Enforcement

Phase 0 establishes the immutable contract that governs all future CAT-DPT work. All schemas, artifact specifications, and enforcement mechanisms are now finalized and frozen.

#### Schemas Finalized
- **jobspec.schema.json**: Canonical job specification format (Phase, TaskType, Intent, Inputs, Outputs, CatalyticDomains, Determinism)
- **ledger.schema.json**: Immutable ledger recording all job executions and artifacts
- **proof.schema.json**: Cryptographic proof of integrity (hash chains, restoration proofs, canonical artifact manifests)
- **validation_error.schema.json**: Deterministic error reporting with stable codes and paths
- **commonsense_entry.schema.json**: Knowledge base entries for commonsense reasoning
- **resolution_result.schema.json**: Results of symbolic resolution

#### Canonical Artifact Set (8 Required Files)
All runtime outputs must conform to the canonical artifact specification:
1. **JOBSPEC.json** - Job specification (immutable, validated at preflight)
2. **STATUS.json** - Run status (failed/completed)
3. **INPUT_HASHES.json** - SHA256 hashes of all inputs
4. **OUTPUT_HASHES.json** - SHA256 hashes of all outputs
5. **DOMAIN_ROOTS.json** - Catalytic domain state (required restoration targets)
6. **LEDGER.jsonl** - Immutable ledger of execution events
7. **VALIDATOR_ID.json** - Identity of accepting validator
8. **PROOF.json** - Cryptographic proof of execution integrity

#### 3-Layer Fail-Closed Enforcement

**Layer 1: Preflight Validation (Before Execution)**
- Validates JobSpec schema compliance
- Detects path traversal, absolute paths, forbidden paths
- Rejects overlapping input/output domains
- Returns deterministic validation_error objects with stable codes
- Implementation: `PRIMITIVES/preflight.py` (10/10 tests pass)

**Layer 2: Runtime Write Guard (During Execution)**
- Enforces allowed roots at write-time for all file operations
- Detects and rejects writes to forbidden paths (CANON, AGENTS.md, BUILD, .git)
- Blocks path traversal and absolute path escapes
- Fails closed immediately with RuntimeError on any violation
- Wraps all writes through FilesystemGuard class
- Implementation: `PRIMITIVES/fs_guard.py` (13/13 tests pass)

**Layer 3: CI Validation (After Execution)**
- Verifies all required canonical artifacts present
- Validates artifact schema compliance
- Checks hash integrity of outputs
- Verifies restoration proofs for catalytic domains
- Ensures ledger consistency
- Implementation: CI pipeline validation

#### Exit Criteria Met
- [x] All schemas finalized and frozen (no breaking changes allowed)
- [x] Canonical artifact set fully specified
- [x] Layer 1 (Preflight) enforces contract before execution
- [x] Layer 2 (Runtime Guard) enforces contract during execution
- [x] Layer 3 (CI) enforces contract after execution
- [x] All enforcement layers fail-closed (RuntimeError/validation_error on violation)
- [x] Error codes deterministic and stable (JOBSPEC_*, WRITE_GUARD_*, etc.)
- [x] Comprehensive test coverage with all tests passing (38/38)
- [x] Roadmap reflects completed work accurately

## [1.15.0] - 2025-12-25

### Added
- **LAB Directory Structure**: Created `CATALYTIC-DPT/LAB/` to isolate experimental scaffolds (`COMMONSENSE/`, `MCP/`, `RESEARCH/`, `ARCHIVE/`) from the kernel core.
- **Pytest Gating**: Implemented `conftest.py` with `pytest_collection_modifyitems` to exclude `LAB/` tests by default.
- **Opt-in Test Variable**: Added `CATDPT_LAB=1` environment variable to enable laboratory testing.
- **Pytest Integration**: Added `test_*` entry points and `pytest.ini` for standardized test discovery across kernel and laboratory components.

### Fixed
- **PROJECT_ROOT in LAB**: Corrected absolute path resolution in `LAB/MCP/server.py` after its move into a deeper directory structure.
- **Spectrum Test Imports**: Updated `test_spectrum02_emission.py` and other spectrum tests to correctly locate the moved MCP server.
- **CommonSense Schema Mismatch**: Fixed `resolution_result.schema.json` to include Phase 2 fields (`expanded_facts`, `unresolved_symbols`) required for validation.

### Changed
- **Architectural Cleanup**: Moved semiotic roadmap patches and historical roadmaps into `LAB/`.
- **Documentation**: Updated `AGENTS.md` with explicit architectural boundaries between Kernel and LAB.

## [1.14.0] - 2025-12-25

### Added
- `CATALYTIC-DPT/COMMONSENSE/` Phase 0–2 scaffold: schemas (`SCHEMAS/commonsense_entry.schema.json`, `SCHEMAS/resolution_result.schema.json`), deterministic resolver (`resolver.py`, `translate.py`), example DB (`db.example.json`), and symbol codebook (`CODEBOOK.json`).
- Fixtures + test benches for contract enforcement:
  - Phase 0 schema validation: `CATALYTIC-DPT/COMMONSENSE/TESTBENCH/test_commonsense_schema.py` over `CATALYTIC-DPT/COMMONSENSE/FIXTURES/phase0/valid|invalid`.
  - Phase 1 resolver expectations: `CATALYTIC-DPT/COMMONSENSE/TESTBENCH/test_resolver.py` over `CATALYTIC-DPT/COMMONSENSE/FIXTURES/phase1/valid`.
  - Phase 2 symbolic expansion + resolve: `CATALYTIC-DPT/COMMONSENSE/TESTBENCH/test_symbols.py` over `CATALYTIC-DPT/COMMONSENSE/FIXTURES/phase2/valid`.
- Semiotic design notes: `CATALYTIC-DPT/ROADMAP_PATCH_SEMIOTIC.md` and `CATALYTIC-DPT/RESEARCH/SEMIOTIC_COMPRESSION_LAYER_REPORT.md`.

### Fixed
- CommonSense resolver import path and inline JSON error string quoting so `test_resolver.py` and `test_symbols.py` can import and execute `COMMONSENSE/resolver.py` from the repo root without `SyntaxError`/`ModuleNotFoundError`.

### Changed
- Roadmap v2.1 updated with a new “Semiotic Compression & Expansion” thread (Phase 1.6) to reduce token overhead via deterministic symbolic plans and code addressing.
## [1.13.0] - 2025-12-24

### Added
- Phase 0 contract freeze: three Draft-07 schemas (`SCHEMAS/jobspec.schema.json`, `SCHEMAS/validation_error.schema.json`, `SCHEMAS/ledger.schema.json`) plus the canonical valid/invalid fixtures under `FIXTURES/phase0/`.
- `CATALYTIC-DPT/TESTBENCH/test_schemas.py` proves the schema contract by loading each Draft-07 definition, ensuring valid fixtures pass and invalid fixtures fail, guarding future contract changes.

## [1.12.0] - 2025-12-24

### Changed
- Roadmap updated to v2.1 with expanded phases, concrete run artifacts, and enforcement model.
- Archived ROADMAP.md to ARCHIVE/ROADMAP.md as historical intent.
- Fixed inconsistent changelog dates (normalized all entries to 2025-12-24).

## [1.11.0] - 2025-12-24

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

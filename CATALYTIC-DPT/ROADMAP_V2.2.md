# CATALYTIC-DPT ROADMAP v2.2
Date: 2025-12-25  
Status: ACTIVE (execution roadmap; referenced by `CATALYTIC-DPT/AGENTS.md`)

## Goal
Build a verifiable catalytic runtime where:
- Truth is artifacts, not narration.
- Runs are resumable and verifiable from bundles and chain proofs.
- Token efficiency is achieved via hash-referenced memory and bounded dereference, not by dumping blobs.

## Operating rules
1) **Phase gates** define “done.” No vibes-based completion.
2) Always execute the **smallest unchecked item** that advances the current blocking gate.
3) Do not merge phases unless this roadmap explicitly says so.
4) Kernel must not import from LAB; LAB may import Kernel.

## Verified progress checklist (as of commit `d6e3970`)
- [x] Phase 0 contract freeze + enforcement (schemas, fixtures, preflight, write guard)
- [x] SPECTRUM-04/05 strict verifier + identity/signing enforcement (bundle + chain)
- [x] SPECTRUM-06 Restore Runner implemented (primitive + CLI + tests; frozen success artifacts + failure codes)
- [x] Phase 1.U CAS implemented (deterministic layout + streaming; tests) (commit `d6e3970`)
- [x] Phase 1.M Merkle implemented (deterministic manifest roots; tests) (commit `19a0c9c`)
- [x] Phase 1.D Ledger implemented (append-only receipts; deterministic JSONL; tests) (commit: this changeset)
- [x] Phase 1.P Proof wiring implemented (CAS/Merkle/Ledger; determinism tests) (commit: this changeset)
- [ ] CI runs `CONTRACTS/runner.py`, but CAT-DPT `pytest` is not wired into CI yet
- [x] Phase 1 substrate gate satisfied (CAS + Merkle + Ledger + Proof wiring)
- [x] Phase 1X expand-by-hash toolbelt implemented (bounded read/grep/ast/describe) (commit: this changeset)

Evidence (files + commits):
- Phase 0: `CATALYTIC-DPT/SCHEMAS/*.schema.json`, `CATALYTIC-DPT/FIXTURES/phase0/`, `CATALYTIC-DPT/PRIMITIVES/preflight.py`, `CATALYTIC-DPT/PRIMITIVES/fs_guard.py`
- Verifier: `CATALYTIC-DPT/PRIMITIVES/verify_bundle.py`, `TOOLS/catalytic_verifier.py` (commits `30efb9b`, `0b5e187`)
- Restore Runner: `CATALYTIC-DPT/PRIMITIVES/restore_runner.py`, `TOOLS/catalytic_restore.py`, `CATALYTIC-DPT/TESTBENCH/test_restore_runner.py` (commit `001b109`)
- CAS: `CATALYTIC-DPT/PRIMITIVES/cas_store.py`, `CATALYTIC-DPT/TESTBENCH/test_cas_store.py` (commit `d6e3970`)
- Merkle: `CATALYTIC-DPT/PRIMITIVES/merkle.py`, `CATALYTIC-DPT/TESTBENCH/test_merkle.py` (commit `19a0c9c`)
- Ledger: `CATALYTIC-DPT/PRIMITIVES/ledger.py`, `CATALYTIC-DPT/TESTBENCH/test_ledger.py` (commit: this changeset)
- Proof wiring: `CATALYTIC-DPT/PRIMITIVES/restore_proof.py`, `TOOLS/catalytic_runtime.py`, `CATALYTIC-DPT/TESTBENCH/test_proof_wiring.py` (commit: this changeset)

## Phase gates (the only definition of “done”)

### Phase 0 Gate: Contract freeze + enforcement
DONE when all are true:
- [x] Schemas exist, are pinned (Draft-07), and fixtures prove valid and invalid cases.
- [x] Preflight validation fails closed.
- [x] Runtime write guard fails closed.
- [x] CI runs governance + contract fixtures (`.github/workflows/contracts.yml`).

### Phase 1 Gate: Kernel substrate
DONE when all are true:
- [x] CAS exists (streaming put/get), deterministic layout, path normalization, tests.
- [x] Merkle exists (domain manifests, deterministic ordering, stable roots), tests.
- [x] Ledger exists (append-only receipts, schema-valid, deterministic serialization), tests.
- [x] Proof generation is wired to these primitives and is deterministic across reruns.

### Phase 1X Gate: Expand-by-hash usability
DONE when all are true:
- [x] `hash read`, `hash grep`, `hash ast`, `hash describe` exist.
- [x] Every output is bounded (bytes, matches, ranges) and deterministic.
- [x] Any tool capable of unbounded dumping is a violation and must be fixed.

### Phase 1V Gate: Verifiers and validator identity
DONE when all are true:
- [x] SPECTRUM-02 bundle verifier is strict and fail-closed.
- [x] SPECTRUM-03 chain verifier is strict and fail-closed.
- [x] Validator identity pin and signing surface (SPECTRUM-04) are implemented with fixtures and adversarial tests.
- [ ] CI default is strict verification (explicit pipeline enforcement).

### Phase 2 Gate: Memoization (never pay twice)
DONE when all are true:
- [ ] Cache keys bind to job + inputs + toolchain + validator identity.
- [ ] Cache hits still emit verifiable receipts and proofs.
- [ ] Cache misses never bypass verification.
- [ ] At least one end-to-end demo shows measurable token reduction via hash-first dereference.

---

## Canonical run artifact set (required)
Every run MUST write artifacts to: `CONTRACTS/_runs/<run_id>/`

Minimum required files (byte-for-byte verifiable):
- [x] `JOBSPEC.json` (schema-valid)
- [x] `STATUS.json` (state machine: started | failed | succeeded | verified)
- [x] `INPUT_HASHES.json`
- [x] `OUTPUT_HASHES.json`
- [x] `DOMAIN_ROOTS.json`
- [x] `LEDGER.jsonl` (append-only, schema-valid)
- [x] `VALIDATOR_ID.json` (validator_semver, validator_build_id, toolchain versions)
- [x] `PROOF.json` (hashable restoration proof summary)

Definition of success:
- A run is not “successful” unless:
  - all required artifacts exist
  - bundle verification passes in strict mode
  - restoration proof passes
- Missing artifacts or verification failures must fail closed.

---

## Current truth: what is actually blocking progress
**Phase 1 Gate is DONE. Next blocking gate is Phase 1X (expand-by-hash) or Phase 1V CI strict enforcement.**

Verifiers and identity law may be shipped early, but do not substitute for the substrate.

---

## PHASE 0: Contracts + fixtures + enforcement (DONE)
Scope:
- [x] JobSpec schema (`CATALYTIC-DPT/SCHEMAS/jobspec.schema.json`)
- [x] Validation error vector schema (`CATALYTIC-DPT/SCHEMAS/validation_error.schema.json`)
- [x] Ledger schema (`CATALYTIC-DPT/SCHEMAS/ledger.schema.json`)
- [x] Proof schema (`CATALYTIC-DPT/SCHEMAS/proof.schema.json`)
- [x] Fixtures: valid + adversarial (`CATALYTIC-DPT/FIXTURES/phase0/`)
- [x] Preflight validator (schema + path rules + overlap rules) (`CATALYTIC-DPT/PRIMITIVES/preflight.py`)
- [x] Runtime filesystem write guard (`CATALYTIC-DPT/PRIMITIVES/fs_guard.py`)
- [x] Restore proof validator primitive (`CATALYTIC-DPT/PRIMITIVES/restore_proof.py`)
- [x] Runtime emits `PROOF.json` (`TOOLS/catalytic_runtime.py`)
- [x] Proof-gated acceptance CLI (`TOOLS/catalytic_validator.py`)
- [x] Proof testbench (`CATALYTIC-DPT/TESTBENCH/test_restore_proof.py`)

Acceptance:
- [x] Deterministic validation (same input yields same output).
- [x] Invalid fixtures fail with stable, parseable error codes.
- [x] Path rules: no abs, no traversal, allowed roots only, forbidden paths blocked.
- [x] CI runs governance + contract fixtures (`.github/workflows/contracts.yml`).

Notes:
- Phase 0 is constitutional and must remain stable. Any change requires versioned migration and new fixtures.

---

## PHASE 1: Kernel substrate (BLOCKING)

### 1.U CAS: Content Addressable Store (BLOCKING)
Deliverables
- [x] `PRIMITIVES/cas_store.py`
  - [x] `put_bytes(data: bytes) -> sha256_hex`
  - [x] `get_bytes(hash_hex: str) -> bytes`
  - [x] `put_stream(stream: BinaryIO, chunk_size: int = 1024*1024) -> sha256_hex`
  - [x] `get_stream(hash_hex: str, out: BinaryIO, chunk_size: int = 1024*1024) -> None`
- [x] Deterministic on-disk layout (no timestamps), stable across OS.
- [x] Path normalization helper (repo-relative, posix normalization).
- [x] Atomic writes (temp + fsync + idempotent commit; never overwrite existing objects).

Acceptance
- [x] Same bytes always map to same hash.
- [x] Large files supported (streaming put/get).
- [x] Determinism test: two consecutive runs produce identical hashes and artifacts.
- [x] Reject non-normalized paths and path traversal.

Testbench targets
- [x] Store and retrieve.
- [x] Deterministic hashing.
- [x] Large file.
- [x] Batch operations.

Status (verified):
- [x] Implemented in `CATALYTIC-DPT/PRIMITIVES/cas_store.py` (`CatalyticStore`, `normalize_relpath`) (commit `d6e3970`)
- [x] Testbench `CATALYTIC-DPT/TESTBENCH/test_cas_store.py` passes (commit `d6e3970`)

### 1.M Merkle: Roots per domain (DONE)
Deliverables
- [x] `PRIMITIVES/merkle.py`
- [x] Domain manifest: `{ normalized_path: bytes_hash }`
- [x] Deterministic leaf ordering and root computation.

Acceptance
- [x] Roots stable across runs given identical manifests.
- [x] Reject duplicates, collisions, and non-normalized paths.
- [x] Adversarial fixtures for ordering, path edge cases, and tamper detection.

Testbench targets
- [x] Root stability.
- [x] Proof verification.
- [x] Tamper detection.

Status (verified):
- [x] Implemented in `CATALYTIC-DPT/PRIMITIVES/merkle.py` (`build_manifest_root`, `verify_manifest_root`) (commit: this changeset)
- [x] Testbench `CATALYTIC-DPT/TESTBENCH/test_merkle.py` passes (commit: this changeset)

### 1.D Ledger: Receipts (append-only) (DONE)
Deliverables
- [x] `PRIMITIVES/ledger.py` writing append-only JSONL records conforming to ledger schema.
- [x] Receipt includes at minimum (ledger.schema.json shape):
  - [x] `JOBSPEC.job_id` (optional `JOBSPEC` included; satisfies job identifier without changing record shape)
  - [x] `RUN_INFO.run_id` and `RUN_INFO.intent` and caller-supplied deterministic `RUN_INFO.timestamp`
  - [x] `PRE_MANIFEST`, `POST_MANIFEST`, `RESTORE_DIFF`, `OUTPUTS`, `STATUS`
  - [x] `VALIDATOR_ID.validator_semver` and `VALIDATOR_ID.validator_build_id` (when present)
- [x] Deterministic serialization rules (canonical JSON per line: UTF-8, no whitespace, sorted keys).
- [x] Append-only enforcement (detect truncation/rewrites via size invariants across appends).

Acceptance
- [x] Schema-valid records only.
- [x] Append-only semantics proven by tests (no mutation of prior lines).
- [x] Deterministic ordering and stable output across reruns.

Testbench targets
- [x] Append entry.
- [x] Deterministic ordering.
- [x] Adversarial: attempt to rewrite/truncate prior record is detected or prevented.

Status (verified):
- [x] Implemented in `CATALYTIC-DPT/PRIMITIVES/ledger.py` (`Ledger`) (commit: this changeset)
- [x] Testbench `CATALYTIC-DPT/TESTBENCH/test_ledger.py` passes (commit: this changeset)

### 1.P Proof wiring: Restore proof uses substrate (DONE)
Deliverables
- [x] `PROOF.json` generation uses CAS + Merkle + Ledger primitives (no ad-hoc root shortcuts).
- [x] Proof is artifact-only verifiable.

Acceptance
- [x] Detects: missing file, extra file, hash mismatch.
- [x] Determinism rerun test: run twice yields byte-identical `DOMAIN_ROOTS.json` and `PROOF.json`.

Status (verified):
- [x] Wired in `TOOLS/catalytic_runtime.py` (CAS-backed snapshots, Merkle `DOMAIN_ROOTS.json`, schema-valid `LEDGER.jsonl`, canonical `PROOF.json`) (commit: this changeset)
- [x] Testbench `CATALYTIC-DPT/TESTBENCH/test_proof_wiring.py` passes (commit: this changeset)

---

## PHASE 1X: Expand-by-hash toolbelt (DONE, required for token efficiency)
Deliverables
- [x] `catalytic hash read <sha> --max-bytes N [--start S] [--end E]`
- [x] `catalytic hash grep <sha> <pattern> --max-matches M --max-bytes N`
- [x] `catalytic hash ast <sha> --max-nodes K --max-depth D` (Python-only; otherwise `UNSUPPORTED_AST_FORMAT`)
- [x] `catalytic hash describe <sha> --max-bytes N`

Acceptance
- [x] Every command enforces explicit bounds (bytes, matches, ranges).
- [x] Deterministic outputs for same inputs and bounds.

Status (verified):
- [x] CLI: `TOOLS/catalytic.py` (`catalytic hash read|grep|describe|ast`) (commit: this changeset)
- [x] Implementation: `CATALYTIC-DPT/PRIMITIVES/hash_toolbelt.py` (commit: this changeset)
- [x] Testbench: `CATALYTIC-DPT/TESTBENCH/test_hash_toolbelt.py` (commit: this changeset)
- [ ] Optional but recommended: dereference events logged to ledger (hash requested, bounds returned).

---

## PHASE 1V: Verifiers + validator identity (SHIPPED)

### 1V.1 SPECTRUM-02 Bundle verification (SHIPPED, keep strict)
Deliverables
- [x] Fail-closed verifier that depends only on:
  - [x] bundle artifacts (`TASK_SPEC.json`, `STATUS.json`, `OUTPUT_HASHES.json`, required set)
  - [x] actual file hashes
- [x] Rejects:
  - [x] missing artifacts
  - [x] hash mismatches
  - [x] forbidden artifacts (logs/tmp/transcripts)
  - [x] non-success status
  - [x] missing or failed proof (when required)

Acceptance
- [x] Adversarial fixtures cover all rejection classes.
- [x] Strict mode exists (default strict in `verify_bundle_spectrum05`).
- [ ] CI default is strict verification (explicit pipeline enforcement).

Status (verified):
- [x] Strict bundle verification implemented in `CATALYTIC-DPT/PRIMITIVES/verify_bundle.py` (`verify_bundle_spectrum05`)
- [x] CLI exists: `TOOLS/catalytic_verifier.py`

### 1V.2 SPECTRUM-03 Chain verification (SHIPPED, keep strict)
Deliverables
- [x] Chain verifier validates ordering and references.
- [x] Artifact-only lineage verification.

Acceptance
- [x] Rejects invalid references, missing bundles, mismatched roots.

Status (verified):
- [x] Strict chain verification implemented in `CATALYTIC-DPT/PRIMITIVES/verify_bundle.py` (`verify_chain_spectrum05`)
- [x] CLI supports chain verification: `TOOLS/catalytic_verifier.py`

### 1V.3 SPECTRUM-04 Validator identity pin + signing (LAW FROZEN, SHIPPED)
Deliverables
- [x] Implement artifacts:
  - [x] `VALIDATOR_IDENTITY.json` (exactly: `algorithm`, `public_key`, `validator_id`)
  - [x] `SIGNED_PAYLOAD.json` (exactly: `bundle_root`, `decision`, `validator_id`)
  - [x] `SIGNATURE.json` (required: `payload_type`, `signature`, `validator_id`)
- [x] Implement canonicalization rules:
  - [x] domain separation prefix
  - [x] canonical JSON (sorted keys, no whitespace, UTF-8)
- [x] Implement Ed25519 verification.
- [x] Implement fail-closed error codes and adversarial fixtures.

Acceptance
- [ ] Two independent implementations produce identical verification results.
- [x] Any ambiguity rejects (multiple keys, multiple signatures, malformed fields, deviations from canonicalization).

Status (verified):
- [x] Implemented in `CATALYTIC-DPT/PRIMITIVES/verify_bundle.py` (strict SPECTRUM-05 verification includes SPECTRUM-04 identity/signing enforcement)
- [x] Adversarial coverage in `CATALYTIC-DPT/TESTBENCH/test_spectrum_04_05_enforcement.py`

---

## SPECTRUM-06: Restore Runner (SHIPPED, does not unblock Phase 1 substrate)

- [x] LAW frozen: `CATALYTIC-DPT/SPECTRUM/SPECTRUM-06.md` (commits `4562dc3`, `2a4c222`, `90041a8`)
- [x] Primitive API: `CATALYTIC-DPT/PRIMITIVES/restore_runner.py` (`restore_bundle`, `restore_chain`)
- [x] CLI: `TOOLS/catalytic_restore.py` (bundle/chain, `--json`, nonzero exit on failure)
- [x] Tests: `CATALYTIC-DPT/TESTBENCH/test_restore_runner.py` (gating, reject-if-exists, traversal/symlink escape, rollback, success artifacts, chain all-or-nothing)

---

## PHASE 2: Memoization and “never pay twice” (LOCK AFTER PHASE 1)
Deliverables
- [ ] Cache key MUST bind:
  - [ ] job_hash
  - [ ] input_hash
  - [ ] toolchain_hash
  - [ ] validator_build_id
  - [ ] validator_id
- [ ] Cache hit must still emit verifiable receipts and proofs.
- [ ] Cache miss must not bypass verification.

Acceptance
- [ ] Deterministic cache behavior.
- [ ] Measurable token reduction in an end-to-end demo where the interface is hash-first and dereference is bounded.

---

## PHASE 3: Substrate adapters (after kernel is real)
Purpose
- [ ] Bridge to external tools and runtimes without weakening trust boundaries.

Deliverables
- [ ] Adapters are additive, never authoritative.
- [ ] Adapters must output artifacts that can be verified by the same verifiers.

Acceptance
- [ ] No adapter can write outside allowed roots.
- [ ] Adapter runs are replayable and verifiable without network trust.

---

## PHASE 4: Runtime hardening
Deliverables
- [ ] Adversarial test suite expands: corrupted artifacts, partial state, path attacks, malformed proofs, timing side channels (as feasible).
- [ ] Tighten guards as needed, always with fixtures and regression tests.

Acceptance
- [ ] Fail-closed everywhere.
- [ ] No silent acceptance.

---

## PHASE 5: Pipelines
Deliverables
- [ ] Compose verified runs into durable workflows.
- [ ] Pipelines are still artifact-first, replayable, and proof-gated.

Acceptance
- [ ] Pipeline execution produces a verifiable chain of bundles and proofs.
- [ ] Resume works without chat logs or narrative state.

---

## PHASE 6: AGS integration
Deliverables
- [ ] CAT-DPT becomes the execution substrate for AGS.
- [ ] AGS must conform to CAT-DPT contracts, not the other way around.

Acceptance
- [ ] At least one integrated workflow: JobSpec in, strict verification out, resumable from artifacts only.

---

## PHASE 7: Optional math
Deliverables
- [ ] Formal proofs or deeper math models (optional).
- [ ] Must not block operational phases.

---

## Immediate next actions (pick one, do the smallest)
- [ ] Implement **CAS (1.U)** with streaming + deterministic layout + tests.
- [ ] Implement **Merkle (1.M)** with deterministic manifests/roots + tests.
- [ ] Implement **Ledger (1.D)** append-only + schema-valid + deterministic + tests.
- [ ] Wire **Proof (1.P)** to the primitives + rerun determinism test.
- [ ] Start **Expand-by-hash (Phase 1X)** once CAS exists (it depends on dereference).

Priority order is strict: 1.U → 1.M → 1.D → 1.P → 1X.

---

## Notes for handoff (for any agent)
- Do not claim progress without artifacts and passing tests.
- Do not paste large blobs into prompts. Use hashes + bounded previews.
- Do not write outside allowed roots; forbidden: `CANON`, `AGENTS.md`, `BUILD`, `.git`.
- Keep commits small and verifiable.

# CATALYTIC-DPT ROADMAP v2.3
Date: 2025-12-25  
Status: ACTIVE

## Goal
Turn CAT-DPT into a **verifiable, model-agnostic execution substrate** that can safely run a **swarm** (many steps, many tools, many models) without relying on chat history or trust.

The north star:
- **Artifacts over narrative**
- **Determinism over convenience**
- **Fail-closed over “best effort”**
- **Hash-first, bounded dereference** so token and I/O cost is measurable and capped

## Operating rules
Non-negotiables:
- No runtime-generated timestamps. If a timestamp exists, it is caller-supplied and recorded as-is.
- Every acceptance claim must be artifact-verifiable (tests and/or fixtures).
- No silent downgrade paths. CI and verifiers must hard-require strictness.
- No unbounded reads. Any dereference or inspection must be bounded.
- Append-only receipts. If you need “state”, write a new record, do not mutate the past.
- Schema-first. Every persistent artifact has a schema (or a canonical JSON contract).

## Phase gates (the only definition of “done”)
> A phase is DONE only when its gate is DONE.

### Phase 0 Gate: Contract freeze + enforcement (DONE)
- [x] All schemas strict (no unknown fields) where appropriate.
- [x] CI enforces strict verification paths.
- [x] Fixtures represent real contracts.

### Phase 1 Gate: Kernel substrate (DONE)
- [x] CAS (streaming, deterministic layout).
- [x] Merkle roots per domain (deterministic ordering).
- [x] Ledger receipts (append-only JSONL, schema-valid, canonical JSON bytes).
- [x] Proof wiring (byte-identical reruns, tamper detection).

### Phase 1X Gate: Expand-by-hash usability (DONE)
- [x] hash read|grep|describe|ast with hard caps and hash-only dereference.
- [x] Optional deref logging into ledger (hash + bounds only, no content).

### Phase 1V Gate: Verifiers and validator identity (DONE)
- [x] Strict verifier behavior enforced in CI and CLI.
- [x] Validator identity pin is law-frozen and verified.

### Phase 2 Gate: Memoization (never pay twice) (DONE)
- [x] Cache keys bind to job + inputs + toolchain + validator identity.
- [x] Cache hits still emit verifiable receipts and proofs.
- [x] Demo exists showing reduced work with bounded hash dereference while PROOF remains byte-identical.

### Phase 3 Gate: Packing hygiene (DONE)
- [x] Packs are deterministic, bounded, deduplicated, fail-closed.

### Phase 4 Gate: Runtime hardening (IN PROGRESS, never “done forever”)
DONE when all are true:
- [x] Adversarial fixtures exist for corruption, partial state, path attacks, malformed proofs, pipeline interruption/resume safety.
- [ ] Fail-closed everywhere (no “warn and continue” in any verifier path).
- [ ] No silent acceptance (every acceptance has an explicit reason code).

### Phase 5 Gate: Pipelines (DONE)
- [x] Artifact-only pipeline runner with resume-safe STATE.
- [x] Proof chain exists and is verifiable.
- [x] CLI verification for pipelines is fail-closed.

### Phase 6 Gate: Swarm integration surface (IN PROGRESS)
DONE when all are true:
- [x] Phase 6.1 model-free bridge exists (emit pipeline spec, run + verify).
- [x] Phase 6.2 router slot exists (external plan producer, schema-validated, capped, fail-closed).
- [x] Phase 6.3 adapter contract exists (skills/MCPs wrap schema-valid jobspec; no runtime bypass).
- [x] Phase 6.4 MCP adapter is first-class (schema + tests + caps + deterministic transcript hashing).
- [x] Phase 6.5 Skill registry exists (hash-addressed capabilities; inclusion in proofs). (commit: this changeset)
- [x] Phase 6.6 Capability pinning and revocation exists (explicit, auditable). (commit: this changeset)
- [x] Phase 6.7 Registry immutability backstop exists (canonical validation + adversarial tests). (commit: this changeset)

---

## Canonical run artifact set (required)
A run is only “real” if these exist and validate:
- `PROOF.json`
- `DOMAIN_ROOTS.json`
- `LEDGER.jsonl`
- `PRE_MANIFEST.json` (or equivalent field inside ledger record)
- `POST_MANIFEST.json` (or equivalent field inside ledger record)
- durable outputs (as declared)

For pipelines:
- `PIPELINE.json`
- `STATE.json`
- `CHAIN.json` (or equivalent proof chain artifact)

For swarm (Phase 7+):
- `SWARM.json` (spec)
- `SWARM_STATE.json`
- `SWARM_CHAIN.json` (top-level chain across pipelines)

---

## PHASE 4: Runtime hardening (ONGOING)
Deliverables
- [x] Adversarial fixtures: corrupted artifacts, partial state, path attacks, malformed proofs, pipeline interruption/resume safety.
- [ ] Tighten guards as needed, always with fixtures and regression tests.
- [x] Fix ledger.schema.json $ref resolution (Draft7)

Acceptance
- [ ] Fail-closed everywhere.
- [ ] No silent acceptance.

Notes
- This phase never fully ends. It is the “immune system”: add fixtures first, then guards, then regressions.

---

## PHASE 6.4: MCP adapter becomes first-class
Intent
Make MCP servers usable as governed adapters inside pipelines without expanding trust boundaries.

Deliverables
- [x] `mcp_adapter.schema.json` with strict, fail-closed constraints:
  - server command vector (non-empty list of strings)
  - request envelope schema (canonical JSON)
  - stdout/stderr caps, timeout cap, exit code rules
  - transcript hashing rules (hash of bytes actually read, not “what should have been read”)
- [x] Implementation in `ags.py` (or a dedicated module) that:
  - executes MCP server as a subprocess
  - rejects any stderr output (or allowlisted patterns if absolutely necessary)
  - enforces byte caps on stdout
  - produces a deterministic adapter output artifact (canonical JSON) plus transcript hash
  - compiles to a Phase 6.3 adapter + schema-valid jobspec (no runtime bypass)
- [x] Tests: happy path + reject cases
  - over-cap output
  - timeout
  - non-zero exit
  - stderr emitted
  - non-canonical JSON
  - non-normalized paths / overlaps in produced jobspec
  - backstop: adapter step runs in a pipeline and strict verify detects output tampering (`CATALYTIC-DPT/TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py`)

Acceptance
- [x] MCP step can run in a pipeline and produce proof-valid artifacts.
- [x] Re-running the same MCP step (same inputs) yields byte-identical artifacts.
- [x] Any cap breach fails closed with a stable error code.

---

## PHASE 6.5: Skill registry (hash-addressed capabilities)
Intent
Turn “skills” into immutable, discoverable, auditable capabilities.

Implemented (v1)
- `CAPABILITIES.json` provides `capability_hash -> adapter spec` resolution and is enforced in `ags route` and `catalytic pipeline verify` (fail-closed).

Deliverables
- [x] `SKILLS/registry.json` (canonical JSON) mapping:
  - `skill_id` -> `adapter_hash` (or `adapter_spec_path` + hash)
  - `version`
  - `capability_hash` (defined below)
  - `human_name`, `description` (optional, non-normative)
- [x] `skills.py` helper to:
  - load registry deterministically
  - resolve skill -> adapter spec by hash
  - reject unknown skills and mismatched hashes
- [x] `ags.py` integration:
  - allow plan steps to reference `skill_id` instead of embedding full adapter
  - expand `skill_id` deterministically into adapter+jobspec before routing
- [x] Tests:
  - registry determinism
  - tamper in adapter spec changes capability hash and fails verification
  - unknown skill fails closed
  - duplicate skill_id rejects

Capability hash (v1)
- `capability_hash = sha256(canonical_json({adapter_schema, adapter_payload, jobspec_hash, global_caps}))`
- It is a pin on *what the skill can do*, not who authored it.

Acceptance
- [x] Skill resolution is hash-addressed and tamper-evident.
- [x] Proof/ledger includes enough to reconstruct which skill capability was used.

---

## PHASE 6.6: Capability pinning and revocation (auditable)
Intent
Make “who/what is allowed to produce accepted work” explicit and reversible.

Implemented (v1)
- `CAPABILITY_PINS.json` allowlists permitted `capability_hash` values; both `ags route` and `catalytic pipeline verify` reject known-but-unpinned capabilities fail-closed.

Deliverables
- [x] Add `CAPABILITY_PINS.json` (canonical) that lists allowed capability hashes for a run/pipeline (or references a pinned set).
- [x] Verifiers fail closed if a step uses a capability hash not in the allowed set.
- [x] Revocation mechanism:
  - an explicit denylist or a new pinned set (no silent mutation)
  - deterministic precedence rules

Acceptance
- [x] “Allow” and “deny” decisions are explicit artifacts.
- [x] Changing allowed capabilities changes the proof hash (no invisible policy changes).

---

## PHASE 6.7: Registry immutability and CI backstop (fail-closed)
Intent
Prevent silent mutation and non-canonical registry drift for capability governance artifacts.

Deliverables
- [x] Canonical validation for `CAPABILITIES.json` and `CAPABILITY_PINS.json` (no duplicates, canonical JSON bytes, sorted ordering).
- [x] Enforced at route time (`ags route`) and verify time (`catalytic pipeline verify`) with stable error codes:
  - `REGISTRY_DUPLICATE_HASH`
  - `REGISTRY_NONCANONICAL`
  - `REGISTRY_TAMPERED`
- [x] Adversarial tests exercising duplicate hashes, non-canonical encoding, and tamper detection.

Acceptance
- [x] Any malformed/non-canonical/tampered registry fails closed before execution and during verification.

---

## PHASE 6.8: Capability versioning semantics (no-history-break)
Intent
Make capability versioning semantics explicit: capabilities are immutable-by-content and historical verification must remain possible.

Deliverables
- [x] Route/verify surfaces use a dedicated boundary error code `CAPABILITY_HASH_MISMATCH` when a capability hash cannot be re-derived from the registry adapter spec bytes.
- [x] Tests assert that changing adapter spec bytes requires a new capability hash; “in-place upgrade” is rejected deterministically.

Acceptance
- [x] Existing capability hashes remain verifiable as long as their registry entries remain present and correct.

---

## PHASE 6.9: Capability revocation semantics (no-history-break)
Intent
Block future use of a capability without breaking historical verification of pre-revocation runs.

Deliverables
- [ ] `CAPABILITY_REVOKES.json` exists (deterministic ordering) and is enforced at route time with `REVOKED_CAPABILITY`.
- [ ] Pipeline verification rejects post-revocation use while preserving verification of pre-revocation pipelines (policy snapshot).
- [ ] Tests prove route rejection, historical verify pass, and post-revocation verify failure.

Acceptance
- [ ] A revoked capability cannot be used in new accepted work, but old work remains mechanically verifiable.

---

## PHASE 7: Swarm scheduling (artifact-only, DAG, no narrative state)
Intent
A swarm is just many pipelines, with explicit dependencies and audited handoffs.

Deliverables
- [x] Phase 7.0: deterministic Pipeline DAG scheduling (DAG spec + scheduler + resume + fail-closed DAG verification).
- [x] Phase 7.1: distributed execution receipts (portable receipts, chained by DAG topology, strict verification).
- [x] Phase 7.2: multi-node restore runner (receipt-gated, idempotent recovery; restore decisions verified).
- [ ] `swarm.schema.json` for a DAG of pipelines:
  - nodes: pipeline specs or references
  - edges: explicit artifact dependencies
  - deterministic node ordering rules
- [ ] `swarm_runtime.py`
  - expands swarm spec into pipeline directories
  - runs and verifies pipelines in dependency order
  - emits `SWARM_STATE.json` + `SWARM_CHAIN.json`
- [ ] Tests:
  - determinism across two independent runs
  - reorder edges changes SWARM_CHAIN hash
  - resume from partial completion uses only artifacts, not logs

Acceptance
- [ ] A swarm run produces a top-level chain that binds each pipeline’s proof.
- [ ] Any missing or tampered pipeline proof fails the swarm verification.

---

## PHASE 8: Model binding (optional, replaceable, never trusted)
Intent
Let local models (LFM2, etc.) produce plans, but never grant them authority.

Deliverables
- [ ] Router receipt artifacts:
  - `ROUTER.json` (what ran, caps, hash of executable)
  - `ROUTER_OUTPUT.json` (canonical JSON plan output)
  - `ROUTER_TRANSCRIPT_HASH` (bytes read)
- [ ] Cache router outputs by content hash (optional)
- [ ] Tests:
  - router over-output fails closed
  - router stderr fails closed
  - malformed plan fails schema validation
  - plan that attempts capability escalation fails closed (Phase 6.6)

Acceptance
- [ ] Swapping models does not change verification logic, only the produced plan.
- [ ] Plans are validated, capped, and recorded, not trusted.

---

## PHASE 9: Freeze, release discipline, and schema versioning
Deliverables
- [ ] Schema versioning policy:
  - how to evolve schemas without breaking verification
  - explicit migrations or parallel schema versions
- [ ] Release checklist for “law changes” (validator identity, schema changes, capability semantics)
- [ ] Docs pass: single authoritative docs tree, no duplicated demo docs that can drift

Acceptance
- [ ] A skeptical auditor can verify system behavior from artifacts alone.
- [ ] A new contributor can implement a compatible verifier from the docs + schemas.

---

## Immediate next actions (pick one, do the smallest)
1) [DONE] **PHASE 6.4**: Implement `mcp_adapter.schema.json` + reject-case tests first (no integration yet).
2) [DONE] **PHASE 6.5**: Create `SKILLS/registry.json` + resolver + tests (no plan integration yet).
3) [DONE] **PHASE 6.6**: Add `CAPABILITIES.json` pin + verifier check + one adversarial test.
4) [DONE] **PHASE 6.7**: Registry immutability (Code complete + tests).
5) **PHASE 6.8**: Capability versioning semantics.
6) **PHASE 7**: Draft `swarm.schema.json` + determinism tests (runtime later).
7) **PHASE 9**: Write schema versioning policy and release checklist (no code).

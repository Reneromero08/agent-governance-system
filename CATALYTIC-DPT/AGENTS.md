# CAT-DPT AGENTS (Scoped)
Scope: CAT-DPT (Catalytic DPT runtime, bundles, chain verification, CAS, Merkle, ledger, restore proof, tooling)

Authority and inheritance
- This file is subordinate to the repo-root AGENTS.md and any CANON/DECISIONS files.
- If any instruction here conflicts with higher authority, higher authority wins.
- This file must not be duplicated elsewhere. Keep exactly one scoped CAT-DPT AGENTS file.

Active roadmap (do not waste tokens)
- The ONLY active execution roadmap for CAT-DPT is: `ROADMAP_V2.2.md`
- Treat these as historical. Do not load unless explicitly asked:
  - `ROADMAP_V2.1.md`
  - `ROADMAP_V2.md`
  - `ROADMAP.md`
- If unsure what to do next, read `ROADMAP_V2.2.md` and execute the smallest unchecked item that advances Phase 0, then Phase 1, etc.

Primary objective
Build a verifiable catalytic runtime where:
- Truth is artifacts, not narration.
- Runs are resumable and verifiable from bundles and chain proofs.
- Token efficiency is achieved via hash-referenced memory and bounded dereference, not by dumping blobs.

Non-goals (hard)
- Do not pack large bodies of repo content into prompts.
- Do not introduce “helpful” refactors unrelated to contracts, proofs, determinism, or security boundaries.
- Do not rely on chat logs/transcripts as state.
- Do not add large-model fine-tuning as a substitute for proofs.

## Contracts are law
Schemas
- Implement and enforce these schemas as canon:
  - `SCHEMAS/jobspec.schema.json`
  - `SCHEMAS/validation_error.schema.json`
  - `SCHEMAS/ledger.schema.json`
- Schema dialect must be explicitly pinned (Draft 7 recommended). Do not change dialect without an explicit decision record and migration plan.

Run artifact set (required)
Every run MUST write artifacts to:
- `CONTRACTS/_runs/<run_id>/`

Minimum required files
- `JOBSPEC.json` (schema-valid)
- `STATUS.json` (state machine: started/failed/succeeded/verified)
- `INPUT_HASHES.json`
- `OUTPUT_HASHES.json`
- `DOMAIN_ROOTS.json`
- `LEDGER.jsonl` (append-only, schema-valid)
- `VALIDATOR_ID.json` (validator_semver, validator_build_id, substrate/toolchain versions)
- `PROOF.json` (hashable restoration proof summary)

Definition of success
- A run is not “successful” unless:
  - all required artifacts exist
  - bundle verification passes in strict mode
  - restoration proof passes
- Missing artifacts or verification failures must fail closed.

Validator identity
- Bundle and chain verification MUST bind to:
  - `validator_semver`
  - `validator_build_id`
- Strict mode is the default for CI and for any acceptance gate.

## Enforcement model (3 layers)
Preflight (before execution)
- Validate JobSpec against schema.
- Enforce path rules:
  - no absolute paths
  - no traversal
  - allowed roots only
- Enforce forbidden overlaps (inputs vs outputs).
- Resolve required hash references or fail.

Runtime guard (during execution)
- Enforce allowed roots and forbid writes elsewhere.
- Record write events (at least path and byte count; include hash when feasible).
- Fail immediately on policy violation.

CI validation (after execution / PR gate)
- Verify bundles and chains in strict mode.
- Verify restoration proof for required fixtures and demos.
- Reject nondeterminism regressions.

## Token efficiency rules (hash-first operations)
Default interaction pattern
- Prefer: hashes + bounded previews + targeted dereference.
- Do not paste large file bodies into prompts.
- Use the expand-by-hash toolbelt once it exists:
  - `catalytic hash read`
  - `catalytic hash grep`
  - `catalytic hash ast`
  - `catalytic hash describe`

Bounded outputs (mandatory)
- Every “read” or “grep” must enforce explicit bounds:
  - max bytes
  - max matches
  - explicit ranges
- Any tool that can dump unbounded content is a violation and must be fixed.

Dereference protocol
- Only dereference by hash when needed for:
  - correctness of an acceptance decision
  - resolving an ambiguity
  - producing final outputs that require exact bytes
- Log dereference events into receipts where feasible (hash requested, byte bounds returned).

## Memoization (never pay twice)
Caching is required once the cache is implemented.
- Cache key MUST include identity and inputs:
  - job_hash
  - input_hash
  - toolchain_hash
  - model_hash (or validator_build_id when model is external)
- Cache hit must still produce verifiable receipts and proofs.
- Cache miss must not bypass verification.

## Determinism discipline
- Do not introduce timestamps or nondeterministic ordering into proofs, roots, manifests, or receipts unless explicitly allowed and documented.
- Normalize paths and sort deterministically before hashing or generating roots.
- When determinism is expected, add a test that runs twice and asserts identical roots and artifacts.

## Work style (how agents should operate)
Implementation workflow
1) Read `ROADMAP_V2.2.md` and pick the next smallest unchecked item.
2) Implement the minimum code to satisfy that item.
3) Add fixtures and tests that prove it.
4) Run tests locally.
5) Update docs only to reflect reality (no aspirational docs).
6) Commit in small, reviewable changes.

Tests are not optional
- Any new contract or enforcement rule requires tests.
- Any bug fix requires a regression test.
- Any security boundary change requires an adversarial fixture.

No silent behavior
- If the system does something, it must be documented.
- If docs claim behavior, tests must verify it.

## Roadmap navigation rules
- Do not “merge phases” unless the roadmap explicitly says so.
- If you discover the roadmap is wrong relative to reality:
  - do not rewrite history
  - propose a new versioned roadmap (v2.2, v2.3) or add a small patch file
  - keep prior versions intact

## Safety rail for edits
- Do not modify repo-root AGENTS.md, CANON, or DECISIONS unless explicitly instructed.
- Prefer adding scoped docs under CAT-DPT directories instead of changing global law.
- If a contradiction is found, report it and propose a minimal reconciliation.

## Architectural Boundaries
Kernel vs. LAB
- The root `CATALYTIC-DPT/` directory contains the **Kernel** (schemas, primitives, verified skills, and core testbenches).
- The `CATALYTIC-DPT/LAB/` directory contains **Experimental Scaffolds** (research, archived roadmaps, and unstable/phased components).
- **Rule**: Kernel code MUST NOT import from `LAB/`.
- **Rule**: `LAB/` code may import from the Kernel.

Test Gating
- Default `pytest` execution excludes the `LAB/` directory to ensure kernel stability and speed.
- To include `LAB` tests, set the environment variable: `CATDPT_LAB=1`.
- Example: `CATDPT_LAB=1 python -m pytest`


End state definition
CAT-DPT is considered “operational” when:
- Phase 0 schemas and fixtures exist and pass
- Phase 1 kernel (CAS, Merkle, ledger, restore proof) passes determinism and adversarial fixtures
- Expand-by-hash toolbelt exists and is bounded and deterministic
- Bundle and chain verification run in strict mode in CI
- At least one end-to-end demo proves:
  - resume without history
  - proof-gated acceptance
  - measurable token reduction at the interface

<!-- CONTENT_HASH: 70ce4fec9997fecf640dfd602a59a72cda8bec5f4d9344a0fe422210e3d9c80f -->

# Phase 7 Handoff (Freeze, Spec, Golden Demo)

Model: Codex (or Claude Sonnet)  
Reasoning: Low  
Repo root: `D:\CCC 2.0\AI\agent-governance-system`  
Workdir: repo root

## Environment (Windows / PowerShell)
```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH="THOUGHT\LAB\CAT_CHAT"
python -m catalytic_chat.cli --help
```

## Objective
Phase 7: Consolidation and Freeze

Produce canonical specs, a PowerShell runbook, and a golden end-to-end demo test. Do not change semantics unless an existing invariant is currently violated by code or tests.

## Hard requirements
- No semantic changes to verification or execution unless strictly required to match existing invariants.
- Deterministic outputs must remain byte-identical for identical inputs.
- Boundedness stays strict: only referenced artifacts, exact slices, no “ALL” sentinel.
- Fail-closed stays strict: any mismatch or forbidden content must hard fail.
- Minimal diffs. Prefer docs and fixtures over code changes.

## Deliverables

### 1) Spec documents (law)
Create these files:
- `THOUGHT/LAB/CAT_CHAT/SPEC/BUNDLE_PROTOCOL.md`
- `THOUGHT/LAB/CAT_CHAT/SPEC/RECEIPTS_AND_CHAIN.md`
- `THOUGHT/LAB/CAT_CHAT/SPEC/TRUST_AND_IDENTITY.md`

Each spec must include exact algorithms and rules. No prose fluff. Include:
- Canonical JSON rules (exact encoder rules).
- bundle_id algorithm (pre-manifest with `bundle_id=""`).
- root_hash algorithm.
- Ordering rules (steps, artifacts, inputs).
- Completeness gate definition (COMMITTED + exactly one receipt per step).
- Forbidden content rules (timestamps, absolute paths, ALL slice).

Receipts and chain spec must include:
- receipt_canonical_bytes vs receipt_signed_bytes roles.
- receipt_hash definition (excludes attestation).
- Chain linkage rules and failure modes.
- Merkle root computation algorithm.
- Explicit ordering and ambiguity rejection rules.

Trust and identity spec must include:
- Trust policy schema semantics.
- validator_id primary lookup rule.
- build_id pinning rule and what is signed.
- Strict-trust vs strict-identity behavior.
- Exit codes and machine-readable output behavior (if implemented).

### 2) Runbook (copy-paste runnable)
Create:
- `THOUGHT/LAB/CAT_CHAT/RUNBOOK.md`

Requirements:
- Windows PowerShell commands only.
- Must show how to capture run_id/job_id values from CLI output.
- Must include a full “happy path” from plan request to bundle build/verify/run to verify-chain and print-merkle or attest-merkle.
- No placeholders like `<JOB_ID>` in commands. Show exact PowerShell variable capture patterns.

### 3) Golden end-to-end demo fixture + test
Add minimal fixture(s) under:
- `THOUGHT/LAB/CAT_CHAT/tests/fixtures/`

Add:
- `THOUGHT/LAB/CAT_CHAT/tests/test_golden_demo_e2e.py`

Test must:
1) Create a plan request deterministically.
2) Execute minimal supported steps.
3) Build a bundle.
4) Verify the bundle.
5) Run the bundle with verify-chain and print-merkle OR attest-merkle.
6) Repeat the entire flow twice and assert:
   - `bundle.json` bytes identical.
   - All artifact file bytes identical.
   - Execution result JSON bytes identical (if emitted).
   - Merkle root identical.
Use `tmp_path`. No absolute Windows paths hardcoded.

If a completed-job fixture is required, create it inside the test using the existing CLI or internal entrypoints. Prefer internal entrypoints for speed and determinism.

## Verification (paste outputs)
From repo root:
```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH="THOUGHT\LAB\CAT_CHAT"

python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests
```

## Deliverable
- Commit-ready changes.
- Brief summary: changed files and what each doc/test covers.
- Confirm that specs match actual implemented behavior (no drift).

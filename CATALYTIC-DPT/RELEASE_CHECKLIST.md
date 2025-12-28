# CAT-DPT Release Checklist (Law Changes)

Use this checklist when changing schemas, validator identity, or capability semantics.

## 1. Governance
- [ ] ADR drafted and accepted in `CONTEXT/decisions/`.
- [ ] `CANON/CHANGELOG.md` updated with descriptive entry.
- [ ] `CANON/VERSIONING.md` bumped (Minor for additive, Major for breaking).

## 2. Schema Hardening
- [ ] New schema versioned correctly (if breaking).
- [ ] `additionalProperties: false` set where appropriate.
- [ ] Schema examples updated in `SCHEMAS/examples/`.

## 3. Tooling Compatibility
- [ ] `PipelineRuntime` updated with new `validator_semver`.
- [ ] `ags.py` plan/route/run commands verified against new schemas.
- [ ] `catalytic.py` hash/pipeline commands verified.

## 4. Regression & historical Verification
- [ ] Run `python -m pytest CATALYTIC-DPT/TESTBENCH/`.
- [ ] Verify that at least one historical run (e.g., from `FIXTURES/`) still passes verification.
- [ ] Test a "resume" scenario where a pipeline started with old law is finished under new law (if applicable).

## 5. Artifact Consistency
- [ ] Ensure all required artifacts (`PROOF.json`, `LEDGER.jsonl`, etc.) are byte-identical for identical inputs.
- [ ] Verify `POLICY.json` captures all relevant snapshots.

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

## 6. Known Issues / Investigations (Post-v2.12.0)
- [ ] **Packer Hygiene:** `test_packer_determinism_catalytic_dpt` fails with hash mismatch on Windows. Investigate CRLF/normalization in `packer.make_pack`.
- [ ] **Packer Tamper Check:** `test_verify_manifest_detects_tamper` assertion failure. Verify `verify_manifest` logic in `core.py`.
- [ ] **Swarm Reuse:** `test_swarm_execution_elision` fails with `DAG_DEP_MISSING`. Investigate artifact presence check in `pipeline_dag.py`.
- [ ] **Preflight in Tests:** Integration tests using `ags run` fail on dirty repo state. Consider mocking `preflight` for these tests or ensuring clean state in CI.
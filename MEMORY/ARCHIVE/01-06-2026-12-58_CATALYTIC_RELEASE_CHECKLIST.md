---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 Catalytic Release Checklist
section: archive
bucket: MEMORY/ARCHIVE
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: b156a5ae3af4ad3313886187fc925ac638c104ba28c14b700a9739da2aadf848 -->

# CAT-DPT Release Checklist (Law Changes)

Use this checklist when changing schemas, validator identity, or capability semantics.

## 1. Governance
- [x] ADR drafted and accepted in `CONTEXT/decisions/` (Covered by Roadmap Phase 7 spec).
- [x] `CHANGELOG.md` updated with descriptive entry (v1.61.0).
- [x] `CANON/VERSIONING.md` bumped (Minor for additive, Major for breaking).

## 2. Schema Hardening
- [x] New schema versioned correctly (if breaking).
- [x] `additionalProperties: false` set where appropriate.
- [x] Schema examples updated in `SCHEMAS/examples/`.

## 3. Tooling Compatibility
- [x] `PipelineRuntime` updated with new `validator_semver`.
- [x] `ags.py` plan/route/run commands verified against new schemas.
- [x] `catalytic.py` hash/pipeline commands verified.

## 4. Regression & historical Verification
- [x] Run `python -m pytest CATALYTIC-DPT/TESTBENCH/`.
- [x] Verify that at least one historical run (e.g., from `FIXTURES/`) still passes verification.
- [x] Test a "resume" scenario where a pipeline started with old law is finished under new law (if applicable).

## 5. Artifact Consistency
- [x] Ensure all required artifacts (`PROOF.json`, `LEDGER.jsonl`, etc.) are byte-identical for identical inputs.
- [x] Verify `POLICY.json` captures all relevant snapshots.

## 6. Known Issues / Investigations (Post-v2.12.0)
- [x] **Packer Hygiene:** `test_packer_determinism_catalytic_dpt` fails with hash mismatch on Windows. Investigate CRLF/normalization in `packer.make_pack`. **(FIXED via timestamp pinning)**
- [ ] **Swarm Terminal Instability:** External terminals are flaky. Avoid `use_terminal=True` for now.

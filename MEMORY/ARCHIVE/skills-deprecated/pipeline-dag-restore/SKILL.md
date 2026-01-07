<!-- CONTENT_HASH: 419edd6bb0ec051bbe08e2de9bd01ec57bcba07908dbe2ace02201659ae10fd0 -->

**required_canon_version:** >=3.0.0


# Skill: pipeline-dag-restore

**Version:** 0.1.0

**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `pipeline-toolkit`.
> Use `{"operation": "restore", ...}` with the pipeline-toolkit instead.



## Trigger

Use when implementing or modifying the pipeline DAG restore runner, including receipt-gated recovery, CLI wiring, tests, and roadmap/changelog updates.

## Inputs

- DAG runtime files under `CATALYTIC-DPT/PIPELINES/`
- CLI surface in `TOOLS/catalytic.py`
- Tests in `CATALYTIC-DPT/TESTBENCH/`
- Governance docs (`CATALYTIC-DPT/ROADMAP_V2.3.md`, `CATALYTIC-DPT/CHANGELOG.md`)

## Outputs

- Deterministic restore behavior (receipt-gated, idempotent)
- Fail-closed verification checks
- Updated tests and governance docs

## Constraints

- Determinism only; no timestamps or random IDs.
- Fail closed with stable error codes.
- Artifact-only dependencies; no implicit data flow.
- Do not touch out-of-scope files.

## Fixtures

- `fixtures/basic_ok`
- `fixtures/tamper_reject`

**required_canon_version:** >=3.0.0


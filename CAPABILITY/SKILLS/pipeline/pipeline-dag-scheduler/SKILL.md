<!-- CONTENT_HASH: 8fc47145aff0ac6bdeeacfc09d9f8c7cfd383263c054668c48152704a6e646de -->

**required_canon_version:** >=3.0.0


# Skill: pipeline-dag-scheduler

**Version:** 0.1.0

**Status:** Draft



## Trigger

When the agent is asked to implement deterministic pipeline DAG scheduling (Phase 7.0) that composes multiple existing pipelines into a DAG with artifact-only dependencies, resume safety, and fail-closed verification.

## Inputs

JSON object:
- `dag_spec_path` (string, required): Repo-relative path to a DAG spec JSON file.
- `runs_root` (string, optional): Runs root directory; default `"CONTRACTS/_runs"`.
- `project_root` (string, optional): Repo root path; default inferred.

## Outputs

JSON object:
- `ok` (boolean)
- `code` (string)
- `details` (object)

Side effects:
- Creates or updates deterministic DAG artifacts under `<runs_root>/_pipelines/_dags/<dag_id>/` only.
- Does not write artifacts outside allowed roots.

## Constraints

- Must be deterministic and idempotent.
- Must not modify CANON or CONTEXT.
- Must not import from `CATALYTIC-DPT/LAB/`.
- Must fail closed on any schema/validation ambiguity.

## Fixtures

- `fixtures/basic_ok`: a 2-node DAG executes deterministically and is verifiable.
- `fixtures/cycle_reject`: a cycle is rejected with a stable error code.

**required_canon_version:** >=3.0.0


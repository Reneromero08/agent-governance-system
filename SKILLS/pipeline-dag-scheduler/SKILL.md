# Skill: pipeline-dag-scheduler

**Version:** 0.1.0

**Status:** Draft

**required_canon_version:** ">=2.5.1 <3.0.0"

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

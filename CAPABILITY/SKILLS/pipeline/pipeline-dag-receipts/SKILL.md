# Skill: pipeline-dag-receipts

**Version:** 0.1.0

**Status:** Draft

**required_canon_version:** ">=2.5.1 <3.0.0"

## Trigger

When the agent is asked to implement distributed execution receipts for pipeline DAG nodes (Phase 7.1).

## Inputs

JSON object:
- `dag_spec_path` (string, required): Repo-relative path to a DAG spec JSON file.
- `runs_root` (string, optional): Runs root directory; default `"CONTRACTS/_runs"`.

## Outputs

JSON object:
- `ok` (boolean)
- `code` (string)
- `details` (object)

## Constraints

- Must be deterministic and idempotent.
- Must not modify CANON or CONTEXT directly.
- Must not import from `CATALYTIC-DPT/LAB/`.

## Fixtures

- `fixtures/basic_ok`: placeholder runner output (governance-only).
- `fixtures/tamper_reject`: placeholder runner output (governance-only).

<!-- CONTENT_HASH: 34e78399fa67a452129b5d956d27c8765382dd89943630e6221db8456c9ba80b -->

**required_canon_version:** >=3.0.0


# Skill: pipeline-dag-receipts

**Version:** 0.1.0

**Status:** Draft



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

**required_canon_version:** >=3.0.0


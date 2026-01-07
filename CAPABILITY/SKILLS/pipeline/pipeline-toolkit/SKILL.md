**required_canon_version:** >=3.0.0

# Skill: pipeline-toolkit

**Version:** 1.0.0

**Status:** Active

## Purpose

Unified toolkit for pipeline DAG operations including scheduling, receipt generation, and restoration.

## Trigger

Use when performing any pipeline DAG operation:
- Scheduling deterministic pipeline DAG execution
- Generating distributed execution receipts
- Restoring pipelines with receipt-gated recovery

## Operations

| Operation | Description |
|-----------|-------------|
| `schedule` | Deterministic DAG scheduling (Phase 7.0) |
| `receipts` | Distributed execution receipts (Phase 7.1) |
| `restore` | Receipt-gated DAG restoration |

## Inputs

`input.json` fields:

**Common:**
- `operation` (string, required): One of `schedule`, `receipts`, `restore`
- `dag_spec_path` (string): Repo-relative path to DAG spec JSON
- `runs_root` (string): Runs root directory (default: `CONTRACTS/_runs`)

**For `schedule`:**
- `project_root` (string, optional): Repo root path

**For `receipts`:**
- (uses dag_spec_path and runs_root)

**For `restore`:**
- (uses dag_spec_path and runs_root)

## Outputs

All operations return:
- `ok` (boolean): Success status
- `code` (string): Status code
- `details` (object): Additional details

## Constraints

- Deterministic and idempotent
- Must not modify CANON or CONTEXT
- Creates DAG artifacts under `<runs_root>/_pipelines/_dags/<dag_id>/`
- Fail closed on schema/validation ambiguity
- Must not import from `CATALYTIC-DPT/LAB/`

## Implementation Status

This skill is a governance placeholder. Actual implementation is in:
- `CATALYTIC-DPT/PIPELINES/` - DAG runtime files
- `TOOLS/catalytic.py` - CLI surface
- `CATALYTIC-DPT/TESTBENCH/` - Tests

## Fixtures

- `fixtures/basic/` - Basic governance test

**required_canon_version:** >=3.0.0

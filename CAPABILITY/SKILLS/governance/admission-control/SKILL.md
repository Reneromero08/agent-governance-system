<!-- CONTENT_HASH: 163bb0800b06a8adabb867a11a091bef3d9c250e22335393f5a4ae6411c65e5f -->

**required_canon_version:** >=3.0.0


# Skill: admission-control

**Version:** 0.1.0

**Status:** Active



## Purpose

Validate the `ags admit` admission control gate using deterministic fixtures.

## Trigger

Use when you need to verify admission control policy behavior via fixtures.

## Inputs

- `input.json`: admission intent JSON.

## Outputs

- Writes `actual.json` containing:
  - `rc` (int): exit code from `ags admit`.
  - `result` (object|null): parsed JSON output from `ags admit`.

## Constraints

- Must be deterministic.
- Must not modify repo files (reads only).

## Fixtures

- `fixtures/read_only_write_block/`
- `fixtures/artifact_only_outside_block/`
- `fixtures/repo_write_flag_required/`
- `fixtures/artifact_only_allow/`

**required_canon_version:** >=3.0.0


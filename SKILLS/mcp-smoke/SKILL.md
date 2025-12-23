# Skill: mcp-smoke

**Version:** 0.1.0

**Status:** Active

**required_canon_version:** ">=2.5.1 <3.0.0"

## Trigger

Use when asked to verify that the AGS MCP server is runnable (stdio mode) and to perform a quick smoke test.

## Inputs

- `input.json` with:
  - `entrypoint_substring` (string, optional): Relative path to the MCP entrypoint.
  - `args` (array, optional): Arguments to pass to the entrypoint (default: `["--test"]`).

## Outputs

- Writes `actual.json` with:
  - `ok` (boolean)
  - `returncode` (number)
  - `entrypoint` (string, relative path)
  - `args` (array)

## Constraints

- Must use `CORTEX/query.py` for file discovery checks.
- Must not scan the filesystem directly.
- Must not modify canon or context.
- Any runtime artifacts must stay under allowed roots.

## Fixtures

- `fixtures/basic/`

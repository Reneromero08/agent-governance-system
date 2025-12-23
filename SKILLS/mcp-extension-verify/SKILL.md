# Skill: mcp-extension-verify

**Version:** 0.1.0

**Status:** Active

**required_canon_version:** ">=2.5.1 <3.0.0"

## Trigger

Use when asked to verify the AGS MCP server inside any IDE extension or MCP-compatible client.

## Inputs

- `input.json` with:
  - `client` (string, optional): `"vscode"`, `"claude"`, or `"generic"` (default).
  - `entrypoint_substring` (string, optional): Relative path to the MCP entrypoint.
  - `args` (array, optional): Arguments to pass to the entrypoint (default: `["--test"]`).

## Outputs

- Writes `actual.json` with:
  - `ok` (boolean)
  - `returncode` (number)
  - `entrypoint` (string, relative path)
  - `args` (array)
  - `client` (string)
  - `instructions` (array of strings)

## Constraints

- Must use `CORTEX/query.py` for file discovery checks.
- Must not scan the filesystem directly.
- Must not modify canon or context.
- Any runtime artifacts must stay under allowed roots.

## Fixtures

- `fixtures/basic/`

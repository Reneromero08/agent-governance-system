<!-- CONTENT_HASH: 3500ba9ad1d99cf179c71ea71d756c5d30401724592a3accda58abb61a45d26f -->

**required_canon_version:** >=3.0.0


# Skill: mcp-smoke

**Version:** 0.1.0

**Status:** Active



## Trigger

Use when asked to verify that the AGS MCP server is runnable (stdio mode) and to perform a quick smoke test.

## Inputs

- `input.json` with:
  - `entrypoint_substring` (string, optional): Relative path to the MCP entrypoint.
  - `args` (array, optional): Arguments to pass to the entrypoint (default: `["--test"]`).
  - `bridge_smoke` (object, optional):
    - `enabled` (boolean): If true, call the local terminal bridge after the server check.
    - `command` (string, optional): Command to execute via the bridge.
    - `cwd` (string, optional): Working directory for the bridge command.
    - `timeout_seconds` (integer, optional): Bridge request timeout (default: 30).

## Outputs

- Writes `actual.json` with:
  - `ok` (boolean)
  - `returncode` (number)
  - `entrypoint` (string, relative path)
  - `args` (array)
  - `bridge_smoke` (object):
    - `enabled` (boolean)
    - `ok` (boolean)
    - `exit_code` (number, optional)
    - `error` (string, optional)

## Constraints

- Must use `CORTEX/query.py` for file discovery checks.
- Must not scan the filesystem directly.
- Must not modify canon or context.
- Any runtime artifacts must stay under allowed roots.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0


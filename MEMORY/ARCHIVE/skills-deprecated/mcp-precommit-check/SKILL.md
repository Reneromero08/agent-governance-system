<!-- CONTENT_HASH: 49074ec22830377e0417793f0c711f975a598826378c3ad82c421e3e099e94a0 -->

**required_canon_version:** >=3.0.0


# Skill: mcp-precommit-check

**Version:** 0.1.0

**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `mcp-toolkit`.
> Use `{"operation": "precommit", ...}` with the mcp-toolkit instead.



## Trigger

Use when asked to verify MCP is healthy before commit, including entrypoint tests,
server running, and Windows autostart status.

## Inputs

- `input.json` with:
  - `entrypoint` (string, optional): Relative path to MCP entrypoint (default: `LAW/CONTRACTS/ags_mcp_entrypoint.py`).
  - `auto_entrypoint` (string, optional): Relative path to auto entrypoint (default: `LAW/CONTRACTS/_runs/ags_mcp_auto.py`).
  - `args` (array, optional): Args for entrypoint (default: `["--test"]`).
  - `auto_args` (array, optional): Args for auto entrypoint (default: `["--test"]`).
  - `require_running` (boolean, optional): Require a running server PID (default: `true`).
  - `require_autostart` (boolean, optional): Require autostart enabled on Windows (default: `true`).
  - `dry_run` (boolean, optional): If true, skip execution and report success (default: `false`).
  - `bridge_config` (string, optional): Path to PowerShell bridge config for non-Windows checks.
  - `bridge_timeout_seconds` (integer, optional): Bridge request timeout (default: 30).

## Outputs

- Writes `actual.json` with:
  - `ok` (boolean)
  - `checks` (object):
    - `entrypoint` (object)
    - `auto_entrypoint` (object)
    - `running` (object)
    - `autostart` (object)

## Constraints

- Must not modify canon or context.
- Must not write artifacts outside allowed roots.
- On non-Windows, running/autostart checks use `powershell.exe` when available and fall back to the PowerShell bridge.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0


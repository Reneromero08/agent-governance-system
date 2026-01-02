<!-- CONTENT_HASH: fae1d6a08e7c5deaa0171dd9199751aa4c806741dda235d1172b845a784faa00 -->

**required_canon_version:** >=3.0.0


# Skill: powershell-bridge

**Version:** 0.1.0

**Status:** Active



## Trigger

Use when asked to set up a local PowerShell bridge for controlled command execution on Windows.

## Inputs

- `input.json` with:
  - `repo_root` (string, optional): Repository root path override.

## Outputs

- Writes `output.json` with:
  - `ok` (boolean)
  - `paths` (object): Relevant file locations.
  - `instructions` (array of strings): Setup and usage steps.

## Constraints

- Must not execute PowerShell or external commands.
- Must only emit deterministic instructions.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0


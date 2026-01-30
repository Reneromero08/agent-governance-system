---
name: critic-run
version: "0.1.0"
description: Run TOOLS/governance/critic.py to verify governance compliance before making changes
compatibility: all
---

**required_canon_version:** >=3.0.0

# Skill: critic-run

**Version:** 0.1.0

**Status:** Active

## Purpose

Runs the governance critic tool to verify that proposed changes comply with AGS governance rules. This is a critical pre-commit check that ensures canon integrity, contract compliance, and architectural decision alignment.

## Trigger

Use when:
- Before committing changes to the repository
- When validating governance compliance during CI
- As part of the commit ceremony workflow

## Inputs

JSON object with optional fields:
- `verbose` (boolean, optional): Enable verbose output (default: false)

## Outputs

JSON object:
- `passed` (boolean): Whether all governance checks passed
- `output` (string): Combined stdout and stderr from critic.py
- `exit_code` (integer): Process exit code (0 = pass)

## Constraints

- Read-only operation (does not modify files)
- Must be run from repository root
- Requires Python 3.8+

## Implementation

Wrapper around `CAPABILITY/TOOLS/governance/critic.py` that:
1. Executes the critic script via subprocess
2. Captures output and exit code
3. Returns structured JSON result

## Fixtures

- `fixtures/basic/` - Basic invocation test

**required_canon_version:** >=3.0.0

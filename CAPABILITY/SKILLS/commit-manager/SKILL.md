---
name: commit-manager
description: "Unified manager for commit-related operations including commit queue management, commit summary logging, and artifact escape hatch validation."
---
**required_canon_version:** >=3.0.0

# Skill: commit-manager

**Version:** 1.0.0

**Status:** Active

## Purpose

Unified manager for commit-related operations including commit queue management, commit summary logging, and artifact escape hatch validation.

## Trigger

Use when performing any commit-related operation:
- Managing a queued commit workflow (enqueue/list/process)
- Recording structured commit summaries
- Generating governance-friendly commit message templates
- Validating no artifacts have escaped allowed output roots

## Operations

| Operation | Description |
|-----------|-------------|
| `queue` | Manage commit queue (enqueue/list/process) |
| `summarize` | Generate commit summaries or message templates |
| `recover` | Emergency artifact recovery (escape hatch check) |

## Inputs

`input.json` fields:

**Common:**
- `operation` (string, required): One of `queue`, `summarize`, `recover`

**For `queue`:**
- `action` (string): `enqueue`, `list`, or `process`
- `queue_id` (string): Queue identifier (default: `default`)
- `entry` (object): For enqueue - message, files, author, notes, created_at
- `max_items` (int): For list - max entries to return
- `dry_run` (bool): For process - skip actual staging

**For `summarize`:**
- `action` (string): `log` or `template`
- For `log`:
  - `mode` (string): `git` or `manual`
  - `commit` (string): Commit ref for git mode
  - `entry` (object): Complete entry for manual mode
  - `append` (bool): Append to log file
  - `include_body` (bool): Include commit body
- For `template`:
  - `type` (string): feat/fix/docs/chore/refactor/test
  - `scope` (string): Short scope identifier
  - `subject` (string): Commit subject line

**For `recover`:**
- `check_type` (string): Type of check (default: `artifact-escape-hatch`)
- `description` (string): Description of the check

## Outputs

Output varies by operation.

**For `queue`:**
- `ok` (bool), `queue_id`, `action`, `entries` (for list), `status`, `error`

**For `summarize`:**
- `ok` (bool), `entry` (for log), `message` and `warnings` (for template)

**For `recover`:**
- `escaped_artifacts` (array), `escape_check_passed` (bool)

## Constraints

- Queue stored at `CONTRACTS/_runs/commit_queue/<queue_id>.jsonl`
- Logs stored at `CONTRACTS/_runs/commit_logs/`
- No automatic git commit (commits remain manual)
- Artifact check scans untracked files only

## Fixtures

- `fixtures/basic/` - Basic operation tests

**required_canon_version:** >=3.0.0

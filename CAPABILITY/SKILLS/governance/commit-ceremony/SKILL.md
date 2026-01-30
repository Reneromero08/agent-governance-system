---
name: commit-ceremony
version: "0.1.0"
description: Run failsafe checks and return ceremony checklist with staged files
compatibility: all
---

**required_canon_version:** >=3.0.0

# Skill: commit-ceremony

**Version:** 0.1.0

**Status:** Active

## Purpose

Orchestrates the pre-commit ceremony workflow by running failsafe checks (critic.py and contract runner) and returning a checklist with staged files. This ensures all governance requirements are met before committing changes.

## Trigger

Use when:
- Preparing to commit changes
- Validating that the repository is ready for a commit
- Running the "Chunked Commit Ceremony" workflow

## Inputs

JSON object with optional fields:
- `dry_run` (boolean, optional): If true, returns mock data for deterministic testing

For deterministic testing (dry_run mode), additional fields:
- `_mock_critic_passed` (boolean): Mock critic result
- `_mock_runner_passed` (boolean): Mock runner result
- `_mock_staged_files` (array of strings): Mock staged files list
- `_mock_git_status` (string): Mock git status output

## Outputs

JSON object with ceremony checklist:
- `checklist` (object):
  - `1_failsafe_critic` (object): Critic check result with `passed`, `tool`, `output`
  - `2_failsafe_runner` (object): Contract runner result with `passed`, `tool`, `output`
  - `3_files_staged` (boolean): Whether any files are staged
  - `4_ready_for_commit` (boolean): All checks passed and files staged
- `staged_files` (array): List of staged file paths
- `staged_count` (integer): Number of staged files
- `git_status` (string): Short git status output
- `ceremony_prompt` (string): Human-readable prompt for next steps

## Constraints

- Read-only for governance checks
- Requires git to be installed
- Runs TOOLS/governance/critic.py
- Runs LAW/CONTRACTS/runner.py

## Implementation

1. Runs critic.py to check governance compliance
2. Runs contract runner to verify fixtures
3. Gets staged files via git diff --cached
4. Gets git status
5. Compiles checklist and ceremony prompt

## Fixtures

- `fixtures/basic/` - Deterministic test with mocked values

**required_canon_version:** >=3.0.0

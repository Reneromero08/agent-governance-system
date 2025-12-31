# Skill: commit-summary-log

**Version:** 0.2.0

**Status:** Active

**required_canon_version:** ">=3.0.0 <4.0.0"

## Purpose

1) Write a deterministic, per-commit summary entry to a JSONL log under `CONTRACTS/_runs/commit_logs/`.

2) Generate a single-line, governance-friendly Git commit message template for the current commit.

This is intended to be run during/after the commit ceremony so you have a durable, machine-readable record of what each commit contained.

## Trigger

Use when you want to:
- Record a structured summary of a commit (hash, subject, files, dates).
- Maintain an append-only commit log under `CONTRACTS/_runs/` without modifying CANON.
- Generate a compliant commit message line (no body).

## Inputs

`input.json` fields:

- `action` (string): `"log"` or `"template"` (default: `"log"`).

For `action:"log"`:
- `mode` (string): `"git"` or `"manual"` (default: `"git"`).
- `commit` (string, required for `mode:"git"`): Commit hash or ref to summarize (example: `"HEAD"`).
- `note` (string, optional): Additional short note to attach to the log entry.
- `include_body` (boolean, optional): Include commit body in entry (default: `false`).
- `append` (boolean, optional): Append to the log file (default: `true`).
- `log_path` (string, optional): Log file path (default: `CONTRACTS/_runs/commit_logs/commit_summaries.jsonl`).

For `mode:"manual"`:
- `entry` (object, required): A complete entry payload to use as-is.

For `action:"template"`:
- `type` (string, required): Conventional commit-ish type (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`).
- `scope` (string, required): Short scope (examples: `canon`, `llm-packer`, `skills`).
- `subject` (string, required): Imperative, present tense; no capitalization; no trailing period; max 50 chars.
- `normalize` (boolean, optional): If true, lowercases and strips a trailing period (default: `true`).
- `warn_if_changelog_missing` (boolean, optional): If true, warn when there are staged changes but no staged `CANON/CHANGELOG.md` (default: `true`).

## Outputs

For `action:"log"`:
- Writes `output.json` containing:
  - `ok` (boolean)
  - `append` (boolean)
  - `log_path` (string)
  - `entry` (object)
- If `append:true`, also appends `entry` as one JSON line to `log_path`.

For `action:"template"`:
- Writes `output.json` containing:
  - `ok` (boolean)
  - `message` (string)
  - `warnings` (array of strings)

## Constraints

- Must be deterministic (no wall-clock timestamps). Dates must come from git metadata or explicit input.
- Must only write logs under `CONTRACTS/_runs/`.
- Must not modify CANON or CONTEXT.

## Commit Message Template Prompt

Create a structured Git commit message with the following format:

Format:
`<type>(<scope>): <subject>`

Rules:
- Check to see what changed in the Changelog and Git history
- Update the changelog if necessary
- Only read other documents if necessary
- One logical change per commit
- Use imperative, present tense
- No capitalization or trailing period
- Max 50 chars
- Details belong in the changelog, not the commit body

## Fixtures

- `fixtures/basic/`

<!-- CONTENT_HASH: 2acfc1d9ccc3d523edf7118e48beccae7b879b278e946f016b310644793d5539 -->

**required_canon_version:** >=3.0.0


# Skill: commit-queue

**Version:** 0.1.0

**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `commit-manager`.
> Use `{"operation": "queue", ...}` with the commit-manager instead.



## Trigger

Use when managing a queued commit workflow (enqueue, list, process) that coordinates multiple local agents and produces deterministic queue artifacts.

## Overview

Provide a deterministic commit queue stored under `CONTRACTS/_runs/commit_queue/` and a process step that stages files for the next queued entry without performing `git commit`.

## Workflow

1) Enqueue a commit request with `action: "enqueue"` and required metadata.
2) List the queue with `action: "list"`.
3) Process the next entry with `action: "process"` to stage files and emit a status payload.
4) If staging succeeds, follow the normal commit ceremony manually.

## Input schema (run.py)

- `action`: "enqueue" | "list" | "process"
- `queue_id`: optional string (default: "default")
- `entry`: object for enqueue:
  - `message`: commit message
  - `files`: array of repo-relative paths
  - `author`: optional string
  - `notes`: optional string
  - `created_at`: required string timestamp
- `max_items`: integer for list

## Output schema

- `ok`: boolean
- `queue_id`: string
- `action`: string
- `entries`: array (for list)
- `staged`: array (for process)
- `status`: string ("pending" | "staged" | "error")
- `error`: string (if any)

## Constraints

- Append-only queue at `CONTRACTS/_runs/commit_queue/<queue_id>.jsonl`.
- No automatic `git commit` inside the skill; commits remain manual and ceremony-gated.
- Repo-relative paths only.

**required_canon_version:** >=3.0.0


---
name: master-override
description: "Audit logging and gated access for MASTER_OVERRIDE usage."
---
<!-- CONTENT_HASH: 53873221b9c1bef59f0127203848959b23baebe62b5639efe005e3867bdf9c8b -->

**required_canon_version:** >=3.0.0


# Skill: master-override

**Version:** 0.1.0

**Status:** Active



## Purpose

Audit logging and gated access for `MASTER_OVERRIDE` usage.

## Trigger

Use when a user prompt contains `MASTER_OVERRIDE` (before executing any overridden action), or when the Sovereign explicitly asks to view override logs with `MASTER_OVERRIDE`.

## Inputs

- `action`: `"log"` or `"read"`
- `token`: must equal `MASTER_OVERRIDE`
- `note` (optional): short reason/context for the override
- `limit` (optional, `read` only): number of most recent entries to return (default: 20)

## Outputs

- For `log`: `{ ok, action, log_path }`
- For `read`: `{ ok, action, log_path, entries }` (only when authorized)

## Constraints

- Logs must be written under `CONTRACTS/_runs/override_logs/`.
- Must not read, quote, or summarize override logs unless authorized.
- Must not modify CANON or CONTEXT.

## Fixtures

- `fixtures/basic/`
- `fixtures/unauthorized-read/`

**required_canon_version:** >=3.0.0


<!-- CONTENT_HASH: 7e9a19ab3465161e49b31bc2a5ea38ad9591d82fc6cf02410938122840ea77e9 -->

**required_canon_version:** >=3.0.0


# Skill: mcp-message-board

**Version:** 0.1.0

**Status:** Active



## Trigger

Use when adding or modifying the MCP message board toolset (schema, server handlers, storage, roles, fixtures, ADR, changelog/version).

## Overview

Implement the MCP message board tools and their governance artifacts without changing unrelated MCP logic.

## Workflow

1) Add/confirm tool schema definitions in `MCP/schemas/tools.json`.
2) Implement tool handlers in `MCP/server.py`:
   - Read: `message_board_list`
   - Write: `message_board_write` (post/pin/unpin/delete/purge)
3) Store events under `CONTRACTS/_runs/message_board/<board>.jsonl` with append-only records.
4) Enforce roles:
   - Default role: `poster` (post + list)
   - `moderator` (post + list + pin/unpin + delete)
   - `admin` (all + purge)
   - Role allowlist file at `CAPABILITY/MCP/board_roles.json` keyed by `session_id`.
5) Add fixtures for the skill and any new governance expectations.
6) Add ADR under `CONTEXT/decisions/` and update `CANON/CHANGELOG.md` + `CANON/VERSIONING.md`.

## Output format (events)

Each event record is a JSON object with:
- `id` (uuid or hash)
- `board`
- `author_session_id`
- `role`
- `type` (`post|pin|unpin|delete|purge`)
- `message` (for posts)
- `ref_id` (for pin/delete)
- `created_at` (ISO timestamp)

## Constraints

- Use only standard library.
- Keep logic deterministic where possible.
- Write artifacts only under allowed roots.

**required_canon_version:** >=3.0.0


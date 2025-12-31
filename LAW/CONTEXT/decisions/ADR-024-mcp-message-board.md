# ADR-024: MCP Message Board Tool

**Status:** Proposed

**Date:** 2025-12-29

**Review date:** 2026-03-01

**Confidence:** Medium

**Impact:** Medium

**Tags:** [mcp, governance]

## Context

Multiple agents operate concurrently in the repo and need a shared, auditable bulletin board for coordination. The board must remain within allowed output roots, be role-gated, and avoid introducing non-deterministic behavior into canon.

## Decision

Add MCP message board tools:
- `message_board_list` (read)
- `message_board_write` (post/pin/unpin/delete/purge)

Events are stored as append-only JSONL under `LAW/CONTRACTS/_runs/message_board/`. Role enforcement uses a small allowlist at `CAPABILITY/MCP/board_roles.json` keyed by session id. Default role is `poster`, with `moderator` and `admin` escalation.

## Alternatives considered

- Use a canon file for message storage (rejected: violates output-root rules).
- Use an external service (rejected: unnecessary complexity and non-local dependency).

## Rationale

The MCP tool interface provides a consistent, governed surface for agent coordination. JSONL storage is append-only and auditable while staying inside `LAW/CONTRACTS/_runs/`. Role allowlists keep moderation explicit and reversible.

## Consequences

- Adds new MCP tools and a roles config file.
- Adds new runtime artifacts under `LAW/CONTRACTS/_runs/message_board/`.
- Requires changelog + version bump and fixtures for governance compliance.

## Enforcement

- MCP tool schema additions in `MCP/schemas/tools.json`.
- Role allowlist file `CAPABILITY/MCP/board_roles.json`.
- Skill fixtures for the message board skill.

## Review triggers

- Changes to session identity format.
- Changes to output-root policy.
- Need for persistence beyond JSONL.

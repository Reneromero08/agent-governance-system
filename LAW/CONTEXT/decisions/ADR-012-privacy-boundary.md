---
id: "ADR-012"
title: "Privacy Boundary for Agent File Access"
status: "Accepted"
date: "2025-12-23"
confidence: "High"
impact: "High"
tags: ["governance", "privacy", "access"]
---

<!-- CONTENT_HASH: 4b2de9898635920a9e35ee8143474df6c567a959ecd8d5ff38b85fff3237a7cf -->

# ADR-012: Privacy Boundary for Agent File Access

## Context

Agents can operate with broad filesystem visibility in some environments. Without
explicit guardrails, they may traverse or inspect paths outside the repository,
including personal directories and OS-level locations. This is a privacy risk and
violates user expectations.

## Decision

Establish a privacy boundary rule: agents must restrict file access to the repo
root unless the user explicitly requests access to specific external paths in the
same prompt.

## Requirements

1. **Repo-only by default**
   - Do not access or search outside the repo root without explicit instruction.
2. **Explicit permission for external paths**
   - If a task requires paths outside the repo, request confirmation and list the
     exact paths to be accessed.
3. **No broad personal scans**
   - Never scan user profile, OS, or other personal directories unless the user
     explicitly requests those paths.

## Alternatives considered

- Relying on sandbox settings alone.
- Allowing broad search with post-hoc consent.

## Rationale

User privacy and intent boundaries must be explicit. This rule provides a clear
default scope, reduces accidental exposure, and aligns agent behavior with the
user's expectations.

## Consequences

- Agents must ask before accessing paths outside the repo.
- Some troubleshooting steps will require explicit user direction.

## Acknowledgement

The agent accessed out-of-repo paths without explicit permission. That was a
privacy breach. I’m sorry for the impact and for the confusion and loss caused
during the search for the missing roadmap file.

## Enforcement

- Update `CANON/CONTRACT.md` and `AGENTS.md` with the privacy boundary rule.
- Update `MAPS/ENTRYPOINTS.md` for discoverability.
- Add a governance fixture documenting the rule.

## Review triggers

- Any privacy incident or near-miss involving out-of-repo access.
- Changes to sandbox defaults or multi-repo workflows.
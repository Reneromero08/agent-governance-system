# Deprecation Policy

This document defines how rules, tokens, and capabilities are deprecated and eventually removed from the Agent Governance System. It ensures breaking changes are predictable and migratable.

## Scope

This policy applies to:
- Canon rules (CONTRACT, INVARIANTS, VERSIONING, ARBITRATION)
- Token grammar (stable identifiers referenced in prompts and code)
- Skills (capabilities in SKILLS/)
- Cortex schema (entity types, fields, query patterns)

## Deprecation Windows

| Item Type | Minimum Window | Notes |
|-----------|----------------|-------|
| Canon rules | 2 major versions OR 90 days | Whichever is longer |
| Token grammar | 1 major version OR 30 days | Tokens are API; stability is critical |
| Skills | 1 minor version OR 14 days | Skills can be replaced faster |
| Cortex schema | 1 major version OR 30 days | Agents depend on stable queries |

During the deprecation window, both old and new paths MUST work.

## Deprecation Ceremony

To deprecate an item:

### 1. Create an ADR

Draft `CONTEXT/decisions/ADR-xxx-deprecate-*.md` explaining:
- What is being deprecated
- Why it is being deprecated
- The replacement (if any)
- The deprecation window and removal target date

### 2. Mark the Item as Deprecated

Add a deprecation notice to the item itself:

**For canon rules:**
```markdown
> [!WARNING]
> **DEPRECATED** (as of v1.2.0): This rule will be removed in v2.0.0.
> Replacement: See [new rule location].
> ADR: CONTEXT/decisions/ADR-xxx-deprecate-*.md
```

**For tokens:**
Add to `CANON/GLOSSARY.md`:
```markdown
- `old_token` — **DEPRECATED**: Use `new_token` instead. Removal: v2.0.0.
```

**For skills:**
Add to the skill's `SKILL.md`:
```markdown
**Status:** Deprecated (removal: v2.0.0)
**Replacement:** SKILLS/new-skill/
```

### 3. Create Migration Artifacts

- **Migration skill** (if applicable): `SKILLS/migrate-*/` that converts old format to new.
- **Fixtures**: Add fixtures proving both old and new paths work during the window.
- **Warnings**: If programmatic, emit warnings when deprecated items are used.

### 4. Update CHANGELOG

Add an entry under the version where deprecation is announced:
```markdown
### Deprecated
- `old_item`: Deprecated in favor of `new_item`. Removal target: v2.0.0.
```

## Removal Ceremony

To remove a deprecated item after the window expires:

### 1. Verify Window Has Passed

Confirm the deprecation window (version OR time) has elapsed.

### 2. Remove the Item

- Delete or archive the deprecated rule/token/skill.
- Remove the deprecation notice.
- Update all references.

### 3. Update CHANGELOG

```markdown
### Removed
- `old_item`: Removed as announced in v1.2.0. See ADR-xxx.
```

### 4. Major Version Bump

Removing a deprecated item is a **breaking change**. Increment the major version in `CANON/VERSIONING.md`.

## Early Removal (Emergency)

In exceptional cases (security vulnerability, critical bug), an item may be removed before the window expires:

1. **Document the emergency** in an ADR with clear justification.
2. **Notify users/agents** prominently (CHANGELOG, README).
3. **Provide migration path** even if rushed.
4. **Still bump major version**.

Early removal should be rare. The governance system's value depends on predictability.

## Sunset Archive

Removed items are not deleted from history. They are:
- Preserved in git history.
- Optionally moved to `CANON/archive/` or `SKILLS/archive/` for reference.
- Referenced in the removal ADR.

## Status

**Active**
Added: 2025-12-21

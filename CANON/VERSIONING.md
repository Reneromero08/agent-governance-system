# Versioning

This file defines the versioning policy for the Agent Governance System.  It tracks changes to the canon and describes the compatibility guarantees provided by each version.

## Canon version

```
canon_version: 2.10.0
```

The version consists of three numbers:

- **Major** - Incremented when breaking changes are introduced (e.g. removing or renaming tokens, changing invariants).
- **Minor** - Incremented when new, backward-compatible rules or tokens are added.
- **Patch** - Incremented for clarifications or fixes that do not affect behavior.

## Compatibility contracts

- **Tokens** - Within a minor version series, existing tokens remain valid.  Tokens may be added but not removed.  Breaking changes require a major version bump and a migration strategy.
- **Skills** - Each skill declares a `required_canon_version` range in its manifest.  A skill must check that the loaded canon version falls within this range before running.

## Deprecation policy

When deprecating a token or rule:

1. Add a note in this file documenting the deprecation and its replacement.
2. Provide migration scripts or skills in `TOOLS/` to help update content.
3. Maintain compatibility for at least one minor version.

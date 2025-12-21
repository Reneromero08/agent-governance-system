# Invariant Freeze

## Purpose

Verifies that all core invariants (INV-001 through INV-008) exist in `CANON/INVARIANTS.md`. This ensures that invariants are not removed without a major version bump, maintaining the v1.0 stability guarantee.

## Triggers

- Manual verification of governance stability.
- CI check before releases.

## Inputs

- `CANON/INVARIANTS.md` (source of truth)
- `input.json` with `expected_invariants` list.

## Outputs

- `actual.json` with:
  - `found_invariants`: list of matches.
  - `missing`: list of missing expected invariants.
  - `valid`: boolean status.

## Constraints

- Must search for exact `[INV-XXX]` tags.
- Must not allow gaps or removals from the frozen list.

<!-- CONTENT_HASH: bd4db62bbe253d888eafe3dddb4f5851b18a79f00cf3b90c0f5db7af6a850164 -->

**required_canon_version:** >=3.0.0


# Skill: example-echo

**Version:** 0.1.0

**Status:** Reference



## Trigger

Use when a minimal, deterministic example skill is needed to verify the contract runner.

## Inputs

- `input.json` with any JSON value.

## Outputs

- Writes `actual.json` containing the same JSON value as the input.

## Constraints

- Deterministic and side-effect free outside the provided output path (runner writes fixture outputs under `CONTRACTS/_runs/`).
- Must not modify canon or context.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0


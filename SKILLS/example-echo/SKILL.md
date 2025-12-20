# Skill: example-echo

**Version:** 0.1.0

**Status:** Reference

**required_canon_version:** ">=0.1.2 <1.0.0"

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

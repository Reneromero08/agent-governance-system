# Skill: skill_name

**Version:** 0.1.0

**Status:** Draft

**required_canon_version:** ">=0.1.0 <1.0.0"

## Trigger

Describe the user intent or condition that should trigger this skill. For example, "when the agent receives a request to create a new blog post".

## Inputs

List the inputs expected by the skill. Include types, defaults and whether each input is required.

## Outputs

Describe what the skill returns or modifies. Specify the output format (e.g. JSON, Markdown) and any side effects.

## Constraints

- Must not modify the canon or context directly.
- Must use the cortex to locate files.
- Must be deterministic and idempotent.

## Fixtures

List the fixtures (under `fixtures/`) that exercise this skill. Each fixture should capture a typical use case and an edge case.

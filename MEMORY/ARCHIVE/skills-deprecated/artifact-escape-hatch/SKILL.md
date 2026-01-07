<!-- CONTENT_HASH: 6cd3be5be3af699574e28b5c93e12683f8435c5e43cbe5bd5da88de25bae4a66 -->

**required_canon_version:** >=3.0.0


# Skill: artifact-escape-hatch

**Version:** 0.1.0

**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `commit-manager`.
> Use `{"operation": "recover", ...}` with the commit-manager instead.



## Trigger

Run this skill as part of CI or pre-commit to verify no generated files have been written outside allowed artifact roots.

## Inputs

- `check_type`: "artifact-escape-hatch"
- `description`: Human-readable description of the check

## Outputs

- `escaped_artifacts`: List of paths that violate INV-006
- `escape_check_passed`: Boolean indicating if check passed

## Constraints

- Must not modify any files.
- Scans CONTRACTS, CORTEX, MEMORY, SKILLS directories.
- Allowed roots: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/`, `BUILD/`.

## Fixtures

- `fixtures/basic/` - Verifies the check passes on a clean repo.

**required_canon_version:** >=3.0.0


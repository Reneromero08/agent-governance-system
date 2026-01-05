**required_canon_version:** >=3.0.0

# Cortex Build Skill

**Version:** 0.1.0

**Status:** Active

Rebuild the Cortex index and SECTION_INDEX, then verify expected paths appear in SECTION_INDEX.

## Usage

```bash
python run.py input.json output.json
```

## Inputs (input.json)

- expected_paths: list of repo-relative paths to verify in SECTION_INDEX
- timeout_sec: int (default 120)
- build_script: optional repo-relative path to cortex.build.py
- section_index_path: optional repo-relative path to SECTION_INDEX.json

## Outputs (output.json)

- ok: true|false
- returncode: int
- section_index_path: repo-relative path
- missing_paths: list of expected paths missing from SECTION_INDEX
- errors: list of error strings (empty when ok)

## Constraints

- Writes only to allowed output roots via the underlying builder.
- Deterministic env: sets PYTHONHASHSEED and CORTEX_BUILD_TIMESTAMP (git head).
- No network access.

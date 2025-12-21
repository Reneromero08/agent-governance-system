# Skill: pack-validate

**Version:** 0.1.0

**Status:** Active

**required_canon_version:** ">=0.1.0"

## Purpose

Validates that a pack is complete, correctly structured, and navigable.

## Checks Performed

1. **Structure Validation**
   - `meta/` directory exists
   - `repo/` directory exists
   - Required meta files present (PACK_INFO.json, REPO_STATE.json, FILE_INDEX.json)

2. **Manifest Integrity**
   - All files in manifest exist in pack
   - File hashes match manifest

3. **Navigation Validation**
   - START_HERE.md or ENTRYPOINTS.md accessible
   - Split files have correct naming (AGS-00_INDEX.md, etc.)

4. **Token Validation**
   - CONTEXT.txt exists
   - Token warnings noted

## Inputs

- `pack_path`: Path to the pack directory to validate

## Outputs

- `valid`: Boolean - whether pack passes all checks
- `errors`: List of validation errors
- `warnings`: List of warnings (non-fatal)
- `stats`: Pack statistics (file count, bytes, tokens)

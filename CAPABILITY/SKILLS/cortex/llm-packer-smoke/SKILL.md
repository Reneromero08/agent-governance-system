<!-- CONTENT_HASH: d833438867b3cbbf8ae15f98c1fe9e9caf7ee49eda2ca3fb1332f76479aaf02f -->

**required_canon_version:** >=3.0.0


# Skill: llm-packer-smoke

**Version:** 0.1.0

**Status:** Reference



## Trigger

Use to verify that `MEMORY/LLM_PACKER/Engine/packer.py` runs and produces a minimal pack skeleton under `MEMORY/LLM_PACKER/_packs/` (fixture outputs should go under `_packs/_system/`).

## Inputs

- `input.json`:
  - `scope` (string): `ags` (default) or `lab` (THOUGHT/LAB only).
  - `out_dir` (string): output directory for the pack, relative to the repo root and under `MEMORY/LLM_PACKER/_packs/`.
  - `mode` (string): `full` or `delta`.
  - `profile` (string): `full` or `lite`.
  - `combined` (bool): whether to generate `FULL/` outputs.
  - `stamp` (string): stamp for timestamped `FULL/` outputs.
  - `split_lite` (bool): whether to generate `LITE/` outputs.
  - `zip` (bool): whether to generate a zip archive.
  - `emit_pruned` (bool): whether to generate `PRUNED/` output (reduced planning context).

## Outputs

- Writes `actual.json` containing the verified pack directory and a list of required files that were found.

## Constraints

- Must only write generated artifacts under `MEMORY/LLM_PACKER/_packs/` (and must not write system artifacts to `BUILD/`).
- Deterministic and self-contained.
- When `emit_pruned` is OFF: PRUNED/ must not exist in the pack.
- When `emit_pruned` is ON: PRUNED/ must exist with valid manifest and rules files.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0


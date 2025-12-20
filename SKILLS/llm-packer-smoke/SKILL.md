# Skill: llm-packer-smoke

**Version:** 0.1.0

**Status:** Reference

**required_canon_version:** ">=0.1.2 <1.0.0"

## Trigger

Use to verify that `MEMORY/packer.py` runs and produces a minimal pack skeleton under `MEMORY/LLM-PACKER-1.0/_packs/`.

## Inputs

- `input.json`:
  - `out_dir` (string): output directory for the pack, relative to the repo root and under `MEMORY/LLM-PACKER-1.0/_packs/`.
  - `mode` (string): `full` or `delta`.
  - `combined` (bool): whether to generate `COMBINED/` outputs.
  - `zip` (bool): whether to generate a zip archive.

## Outputs

- Writes `actual.json` containing the verified pack directory and a list of required files that were found.

## Constraints

- Must only write generated artifacts under `MEMORY/LLM-PACKER-1.0/_packs/` (and must not write system artifacts to `BUILD/`).
- Deterministic and self-contained.

## Fixtures

- `fixtures/basic/`

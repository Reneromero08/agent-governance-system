---
name: doc-merge-batch-skill
version: "1.0.0"
description: Deterministic document merge utility (verify/apply) with JSON-in/JSON-out skill wrapper
compatibility: all
---

<!-- CONTENT_HASH: GENERATED -->

**required_canon_version:** >=3.0.0

# Skill: doc-merge-batch-skill

**Version:** 1.0.0
**Status:** Active

## Purpose
Wraps the `doc_merge_batch` module as an AGS skill with deterministic JSON I/O.

## Inputs
`input.json`:
- `mode`: `"verify"` (recommended) or `"apply"`
- `pairs`: list of `{ "a": "...", "b": "..." }` relative file paths
- `out_dir`: output directory (relative path)

## Outputs
Writes a JSON report to the provided `output.json` path:
- `ok`: boolean
- `mode`, `out_dir`, `pairs`
- `report_path`: path to the underlying tool's `report.json` if present

## Constraints
- Writes outputs only under the provided `out_dir`.
*** End Patch"}]}commentary to=functions.apply_patch  玩彩神争霸Exit code: 0

---
name: cortex-toolkit
description: "Unified toolkit for CORTEX operations including CAS integrity verification, summary generation, and LLM packer smoke testing."
---
**required_canon_version:** >=3.0.0

# Skill: cortex-toolkit

**Version:** 2.0.0

**Status:** Active

## Purpose

Unified toolkit for CORTEX operations including CAS integrity verification, summary generation, and LLM packer smoke testing.

Note: `build` and `verify_system1` operations removed in v2.0 - semantic search is now handled by the cassette network (`NAVIGATION/CORTEX/cassettes/`).

## Trigger

Use when performing any CORTEX-related operation:
- Verifying CAS blob integrity
- Generating section summaries
- Running LLM packer smoke tests

## Operations

| Operation | Description |
|-----------|-------------|
| `verify_cas` | Check CAS directory integrity (SHA-256 hash verification) |
| `summarize` | Generate deterministic section summaries |
| `smoke_test` | Run LLM Packer smoke tests |

## Inputs

`input.json` fields:

**Common:**
- `operation` (string, required): One of `verify_cas`, `summarize`, `smoke_test`

**For `verify_cas`:**
- `cas_root` (string, required): Path to CAS root directory

**For `summarize`:**
- `record` (object): SECTION_INDEX-style record with `section_id`, `heading`, etc.
- `slice_text` (string): Exact section slice text

**For `smoke_test`:**
- `scope` (string): `ags` or `lab`
- `out_dir` (string): Output directory under `MEMORY/LLM_PACKER/_packs/`
- `mode` (string): `full` or `delta`
- `profile` (string): `full` or `lite`
- `combined` (bool): Generate `FULL/` outputs
- `stamp` (string): Timestamp stamp for outputs
- `split_lite` (bool): Generate `LITE/` outputs
- `zip` (bool): Generate zip archive
- `emit_pruned` (bool): Generate `PRUNED/` output

## Outputs

`output.json` fields vary by operation:

**For `verify_cas`:**
- `status` (string): `success` or `failure`
- `total_blobs` (int): Number of blobs checked
- `corrupt_blobs` (array): List of corrupt blob details
- `cas_root` (string): Resolved CAS path

**For `summarize`:**
- `safe_filename` (string): Safe filename for summary
- `summary_md` (string): Generated summary markdown
- `summary_sha256` (string): Hash of summary

**For `smoke_test`:**
- `pack_dir` (string): Output pack directory
- `stamp` (string): Pack timestamp
- `verified` (array): List of verified files
- `emit_pruned` (bool): Whether pruned output was generated

## Constraints

- Writes only to allowed output roots via GuardedWriter
- Deterministic: sets PYTHONHASHSEED, uses git timestamps
- No network access
- Read-only operations for verify_cas
- No LLM calls for summarize operation

## Fixtures

- `fixtures/basic/` - Basic operation tests for each operation type

**required_canon_version:** >=3.0.0

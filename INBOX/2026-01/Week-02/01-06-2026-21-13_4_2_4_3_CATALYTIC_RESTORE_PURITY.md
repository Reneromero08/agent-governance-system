---
uuid: 00000000-0000-0000-0000-000000000000
title: Catalytic Restore Proof and Purity Checks
section: report
bucket: reports/v4/section_4_2 and 4_3
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: Catalytic Restore Proof and Purity Checks'
tags:
- catalytic
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: fde6a1034c0e9ee53b1ccd1bb1423bd82d3bc949fb8eb83e1f4051eb5f22498c -->

# Report: Catalytic Restore Proof and Purity Checks

Date: 2026-01-05
Roadmap Section: 4.2 and 4.3

## Goal
Upgrade catalysis from “protocol and audit” to “mandatory proof”:
- Every catalytic run must produce a machine-verifiable restore proof.
- Leakage outside declared roots must be blocked or detected fail closed.

This section binds Phase 1.5 mechanics into the catalytic architecture and into proofs.

## Core artifacts
### RESTORE_PROOF.json
Minimum fields:
- `pre_digest`
- `post_digest`
- `exclusions_spec_hash`
- `durable_roots`
- `tmp_roots`
- `verdict`
- `diff_summary` (deterministic on failure)

### PURITY_SCAN.json
Minimum fields:
- `verdict`
- `violations[]` with canonicalized paths and reasons
- `tmp_residue[]` if any
- `scan_version_hash`

## Binding requirements
- Restore proof must be emitted for every catalytic run.
- Restore proof must be included in run bundles.
- Public packs must treat restore/purity artifacts as protected if they reveal sensitive paths or content hints (sealed under CRYPTO_SAFE when required).

## Negative fixtures (required)
- Smuggled state outside durable roots.
- Hidden file in non-obvious location.
- Tmp residue left behind.
- Partial restore (file reverts but digest mismatch).

## Determinism requirements
- Canonical ordering of diffs and violations.
- Stable path normalization across OS boundaries (Windows vs WSL).

## Acceptance criteria
- Every catalytic run emits restore and purity receipts.
- Violations fail closed with minimal, deterministic leak reports.
- Proof artifacts can be verified offline from receipts.

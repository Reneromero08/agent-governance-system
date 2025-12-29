# Skill: cas-integrity-check

**Version:** 1.0.0

**Status:** Active

**required_canon_version:** ">=2.11.8 <3.0.0"

## Purpose

Verify the integrity of a content-addressed storage (CAS) directory by checking
that each blob's SHA-256 hash matches its filename.

## Trigger

Use when you want to verify that a CAS directory is not corrupted.

## Inputs

`input.json` fields:

- `cas_root` (string, required): Path to the CAS root directory (repo-relative recommended).

## Outputs

Writes `output.json` containing:

- `status` (string): `success` or `failure`
- `total_blobs` (int)
- `corrupt_blobs` (array): objects with `path` and `reason` (+ optional `expected`, `actual`)
- `cas_root` (string): resolved path checked

## Constraints

- Deterministic (no wall-clock usage).
- Read-only: must not modify repo files.

## Fixtures

- `fixtures/missing_root/`


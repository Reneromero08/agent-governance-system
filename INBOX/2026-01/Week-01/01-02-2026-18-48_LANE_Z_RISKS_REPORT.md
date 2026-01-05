---
title: "Lane Z Risks Report"
section: "report"
author: "Antigravity"
priority: "High"
created: "2026-01-02 18:48"
modified: "2026-01-02 18:48"
status: "Draft"
summary: "Risk analysis for Lane Z CAS integration and Z.3 roadmap"
tags: [lane-z, risks, cas, integration]
---
<!-- CONTENT_HASH: 9ae9babf0eb7f64d9e15dc8dd4dd134b88f7da95065f82cce823b1cbb89645e7 -->

# Lane Z Risks Report

## Highest-risk areas

### 1) Root correctness and completeness
Risk:
- If roots are incomplete, GC can delete blobs required for reproducibility.

Mitigations:
- Require packer to emit TASK_SPEC, OUTPUT_HASHES, final STATUS as roots.
- Add a test that a successful run is never GC-vulnerable.
- Add a root audit command that enumerates reachable set and asserts required artifacts are reachable.

### 2) Accidental full sweep
Risk:
- allow_empty_roots misuse causes mass deletion.

Mitigations:
- Keep Policy B default (fail-closed).
- Require an explicit override switch and operator acknowledgement text.
- Log the override in receipts and any ledger.

### 3) Concurrency and lock drift
Risk:
- GC sweep runs while packer is writing objects or before roots are committed.

Mitigations:
- Use a pack lock plus GC lock.
- In packer: commit roots before any success status is emitted.
- In GC: require stable root enumeration and lock acquisition before deletion.

### 4) Determinism erosion
Risk:
- Ordering, timestamps, or filesystem enumeration leak into OUTPUT_HASHES or receipts, breaking reproducibility.

Mitigations:
- Always sort enumerations.
- Canonical JSON encoding for records and receipts.
- Ban wall-clock fields from hashed content (allowed only in non-hashed metadata).

### 5) Windows path and case sensitivity edge cases
Risk:
- Case-insensitive collisions or separator differences cause inconsistent behavior between Windows and Linux.

Mitigations:
- Normalize paths used in any canonical representations.
- Avoid hashing paths. Hash bytes only.
- Add cross-platform tests if paths appear in TASK_SPEC.

### 6) Performance cliffs on CAS enumeration
Risk:
- Mark and sweep list many blobs and slow down on large stores.

Mitigations:
- Stream iteration.
- Consider optional indexes later without changing semantics.
- Keep GC explicit, not background.

### 7) Corruption handling policy
Risk:
- Corruption causes ambiguous behavior during traversal or sweep.

Mitigations:
- Fail-closed on any corruption in rooted blobs.
- For unrooted corruption, choose one policy (fail-closed or skip with reason) and test it.
- Always emit a receipt that records corruption findings.

### 8) Receipt and root growth
Risk:
- Receipts and run roots grow without bounds.

Mitigations:
- Keep receipts canonical and compact.
- Consider archiving receipts later, but do not delete roots casually.

### 9) Schema drift and versioning
Risk:
- Record format changes break older runs or reproducibility.

Mitigations:
- Version schemas explicitly in records.
- Keep loaders backward compatible.
- Add tests that old receipts still parse after upgrades.

### 10) Scope creep into heuristics
Risk:
- Integration introduces time-based cleanup or best-effort behavior that violates catalytic design.

Mitigations:
- Treat invariants as authority.
- Require mechanical tests for enforcement.
- Prefer fail-closed with report over best effort.

## Recommended next steps
- Add a root audit tool: list roots, compute reachable, compare to OUTPUT_HASHES, fail if mismatch.
- Add an end-to-end packer dry-run test proving GC safety.
- Keep allow_empty_roots guarded behind explicit operator intent.

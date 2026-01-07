---
uuid: 00000000-0000-0000-0000-000000000000
title: Catalytic IO Guardrails (Write Firewall + Repo Digest + Purity Scan)
section: report
bucket: reports/v4/section_1_5
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: Catalytic IO Guardrails (Write Firewall + Repo Digest
  + Purity Scan)'
tags:
- catalytic
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: f085509f7b4eadd9ad834c016bf951892e6a90d2c6db71fdeff833ed7fbf1c33 -->

# Report: Catalytic IO Guardrails (Write Firewall + Repo Digest + Purity Scan)

Date: 2026-01-05
Roadmap Section: 1.5

## Why this section exists
Your catalytic guarantees currently rely on protocols, audits, and operator discipline. This section makes catalysis *mechanical* at the filesystem boundary:
- Prevent out-of-domain writes during execution.
- Require deterministic pre/post repo state digests.
- Fail closed if any residue or leakage exists after restore.

This is the strongest engineering interpretation of catalytic restoration: the catalyst (repo and governance substrate) must return to a known state after each run, except for explicitly declared durable outputs.

## Scope
### In scope
- Runtime write firewall for all agent/tool writes.
- Deterministic repo digest primitive with declared exclusions.
- Post-run purity scan (new/modified files outside durable roots; tmp not cleaned).
- Tests with negative fixtures and deterministic receipts.

### Out of scope
- Cryptographic sealing (handled by CRYPTO_SAFE).
- GC deep traversal semantics (handled in 2.5).
- Vector substrate specifics (handled in 5.2).

## Definitions
- **Catalytic tmp roots**: directories where transient writes are allowed during the run.
- **Durable roots**: directories where committed outputs may exist after the run.
- **Exclusions**: an allowlist of paths that do not participate in the digest (example: OS-specific caches), declared and recorded in receipts.

## Proposed enforcement design
### A. Write firewall (runtime)
Implement a single IO policy layer used by all write operations:
- `write_tmp(path, bytes)` only permits paths under declared tmp roots.
- `write_durable(path, bytes)` only permits paths under durable roots and only after commit gate is open.
- `mkdir_tmp`, `mkdir_durable`, `rename`, `unlink` follow the same policy.
- Any violation raises a deterministic error and emits a failure receipt.

Implementation options (pick one and standardize):
1) Central wrapper module imported by every tool.
2) Patchpoint at the agent runner layer that intercepts file writes and blocks by path rules.

The wrapper approach is simpler to enforce. The runner interception approach is broader, but harder to implement correctly.

### B. Repo state digest (pre/post)
Goal: a canonical "tree hash" of the repo state to prove restoration.

Minimum viable digest:
- Enumerate files under repo root excluding declared durable roots and declared exclusions.
- For each file: hash its bytes (sha256) and include file size.
- Canonical ordering: lexicographic by normalized relative path.
- Tree digest: sha256 of the concatenation of `(path\0sha256\0bytes\n)` lines.

Emit:
- `PRE_DIGEST.json`
- `POST_DIGEST.json`
- `RESTORE_PROOF.json` with:
  - pre_digest
  - post_digest
  - exclusions spec hash
  - verdict PASS/FAIL
  - deterministic diff summary on FAIL (added/removed/changed paths)

### C. Purity scan (post-run)
Checks:
- Tmp roots must be empty after restore (or explicitly allowlisted residuals, discouraged).
- No new or modified files outside durable roots and declared exclusions.
- Optional hardening: detect touched-but-restored patterns by comparing pre/post metadata (keep simple at first).

Output:
- `PURITY_SCAN.json` receipt with verdict and the minimal leak report.

## Integration points
- Agent runners must use the IO policy layer.
- Packer and proof runners must emit restore proofs as part of run bundles (see 4.2/4.3 and 6.4.10).
- If CRYPTO_SAFE is enabled for public packs, restore proofs may be sealed if they reveal protected details.

## Test plan
### Unit
- Out-of-domain write fails.
- Durable write before commit gate fails.
- Digest is deterministic across repeated runs with identical inputs.

### Integration fixtures
- Fixture that creates a forbidden file and ensures scan fails.
- Fixture that leaves tmp residue and ensures scan fails.
- Fixture that only writes to durable root and ensures pass.

## Acceptance criteria
- A run cannot write outside allowed domains.
- Every catalytic run emits restore proof and purity scan receipts.
- Violations fail closed with deterministic errors and minimal leak reports.

## Open questions to resolve
- Do you allow any non-empty tmp residue ever? Recommendation: default NO.
- Are durable roots per-task, per-tool, or global? Recommendation: declare per run in receipts.
- Do you include file permissions and mtimes in the digest? Recommendation: bytes-only at first, then expand if needed.

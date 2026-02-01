---
uuid: 00000000-0000-0000-0000-000000000000
title: GC and AUDIT Deep Traversal Unification, Locking, and Content-Hash Receipts
section: report
bucket: reports/v4/section_2_5
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: GC and AUDIT Deep Traversal Unification, Locking, and
  Content-Hash Receipts'
tags:
- audit
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: a4ba6a1e93dd059a9ae3e1656c5648647ed9f2d058eeaef9a1d41012993c9315 -->

# Report: GC and AUDIT Deep Traversal Unification, Locking, and Content-Hash Receipts

Date: 2026-01-05
Roadmap Section: 2.5

## Goal
Make catalytic reachability and restoration enforceable at the graph level, not just via shallow roots:
- GC and AUDIT must compute the same reachable-set over the same reference graph.
- Packer and GC must not race.
- Receipts become immutable, addressable artifacts (content-hash addressed).

## Why this matters
If GC traversal and audit traversal diverge, “restoration” becomes ambiguous. A catalytic system needs a single, canonical definition of reachability and a single enforcement surface.

## Scope
- Implement deep traversal for reference closure.
- Unify traversal primitive so GC and AUDIT share identical logic.
- Add strict locks to prevent concurrent mutation (packer vs GC).
- Canonicalize receipts under `sha256:` paths (receipt bodies deduplicated).

## Design requirements
### A. Shared traversal primitive
Create a single function that:
- accepts a set of roots
- iterates deterministic reference edges
- produces a canonical reachable-set
- is imported by both GC and AUDIT

Key properties:
- deterministic ordering
- stable visited-set semantics
- bounded recursion safeguards

### B. Regression equivalence test
Fixture-driven test:
- Build a tiny synthetic reference graph.
- Run GC mark and AUDIT traversal.
- Assert identical reachable-set (as a set, and as a canonical ordered list for hashing).

### C. Locking
Two exclusive locks:
- packer lock
- GC lock

Rules:
- If packer lock held, GC refuses to run.
- If GC lock held, packer refuses to run.
- Errors must be deterministic and receipt-backed.

### D. Content-hash addressed receipts
Receipt bytes are canonicalized:
- `sha256:<digest>.json` holds the receipt body.
- Run directories store pointers to receipt refs.
- This reduces duplication and improves auditability.

## Integration points
- Packer uses locks and records lock state in receipts.
- AUDIT includes reachable-set digest and traversal version hash.
- PROOFS bind to these receipts (compression/catalytic proofs include traversal digests).

## Tests
- Equivalence test as above.
- Lock test: simulate both lock orders.
- Receipt canonicalization test: identical receipt bytes resolve to same content hash path.

## Acceptance criteria
- GC and AUDIT produce identical reachable-set on fixtures.
- Races are mechanically prevented via locks.
- Receipts are immutable, content-addressable, and independently auditable.

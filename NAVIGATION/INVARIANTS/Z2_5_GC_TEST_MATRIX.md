<!-- CONTENT_HASH: 08a3c456d2956a4403e591e329888baf4c8a705f83ddb5e32eae94c9285fae95 -->

# Z.2.5 GC Test Matrix (Mechanical Proof Plan)

Status: CANONICAL (Z.2.5)
Last updated: 2026-01-02
Goal: Prove GC is conservative, deterministic, fail-closed, and auditable.

## Test harness assumptions
- Tests operate in a temp CAS storage root (fixture).
- Tests can create blobs via CAS put.
- Tests can create roots via run-record artifacts and optional pin file.
- Tests avoid nondeterminism (stable ordering, fixed seeds, no wall-clock dependence).

## Legend
- Rooted: referenced by declared roots
- Unrooted: no reference from roots
- Apply: deletion mode
- Dry-run: report only

## Core test cases

| ID | Category | Setup | Action | Expected |
|---|---|---|---|---|
| GC-01 | Reachability | Create 1 blob, include its hash in roots | Apply | Blob is not deleted |
| GC-02 | Garbage | Create 1 blob, no roots | Apply | Blob is deleted |
| GC-03 | Mixed | Create 3 blobs, root only 1 | Apply | Only unrooted blobs deleted |
| GC-04 | Determinism report | Fixed CAS + roots | Dry-run twice | Identical GCReport including deletion list order |
| GC-05 | Dry-run parity | Fixed CAS + roots | Dry-run then Apply | Dry-run candidates exactly match Apply deleted set |
| GC-06 | Invalid root ref | Root list contains invalid hash format | Apply | Fail-closed, no deletions, error reported |
| GC-07 | Missing root artifact | Root source references non-existent object | Apply | Fail-closed, no deletions |
| GC-08 | Corrupted blob referenced | Corrupt a referenced blob | Apply | Fail-closed, no deletions, corruption reported |
| GC-09 | Corrupted unreferenced blob | Corrupt an unreferenced blob | Apply | Either fail-closed (preferred) or skip with reason; must not delete anything else incorrectly |
| GC-10 | Pin file root | Create 2 blobs, pin 1 | Apply | Pinned blob not deleted, other deleted |
| GC-11 | Pin file invalid | Pin file malformed JSON | Apply | Fail-closed, no deletions |
| GC-12 | **POLICY B: Empty roots** | No roots, CAS has blobs | Apply | **With allow_empty_roots=False: FAIL-CLOSED, no deletions, error. With allow_empty_roots=True: Deletes all blobs deterministically.** |
| GC-13 | Empty CAS | No blobs | Apply | No deletions, report valid |
| GC-14 | Large set ordering | Many blobs, subset rooted | Dry-run | Deletion list order stable and correct |
| GC-15 | Duplicate roots | Same hash appears multiple times | Dry-run | Reachable set deduped, report stable |
| GC-16 | Locking | Simulate lock held | Apply | GC aborts cleanly, no deletions |
| GC-17 | Partial failure during sweep | Force delete failure for 1 blob | Apply | Skipped blob recorded with reason; behavior for others must be explicit and tested |
| GC-18 | Receipt canonicalization | Same inputs | Dry-run | Receipt bytes identical (canonical JSON) |
| GC-19 | Snapshot hash | Fixed CAS listing | Dry-run twice | Snapshot hash identical |
| GC-20 | Legacy refs | Roots include legacy paths (if supported) | Apply | Either forbidden (fail-closed) or normalized; behavior must be explicit and tested |

## Policy B Decision (LOCKED)

**POLICY LOCK (Choice B)**: Full sweep is FORBIDDEN by default.

### GC-12 MUST enforce the following:

**Scenario A: allow_empty_roots=False (default)**
- If root enumeration yields ZERO roots:
  - GC MUST FAIL-CLOSED
  - GC MUST perform ZERO deletions
  - GC MUST return error: "POLICY_LOCK: Empty roots detected and allow_empty_roots=False"

**Scenario B: allow_empty_roots=True (explicit override)**
- If root enumeration yields ZERO roots AND allow_empty_roots=True:
  - Full sweep is allowed
  - GC MAY delete all unreferenced blobs deterministically
  - No error is reported

## Minimum Definition of Done for Z.2.5
Z.2.5 is complete when:
- GC-01 through GC-11 pass.
- GC-04 and GC-05 prove determinism and dry-run parity.
- Receipt canonicalization is proven (GC-18).
- Locking safety is proven (GC-16).
- A clear policy decision is made and tested (GC-12).

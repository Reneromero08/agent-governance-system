---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 P2 Cas Packer Integration P0
section: archive
bucket: ARCHIVE/implemented
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: 5ac8770d047e687e93db6745a58b573aaf50c7ccc8e97340558f72c21b26fd4f -->
# P.2: CAS-backed Packer Outputs (P0)
**Objective:** Make the LLM Packer produce **CAS-addressed outputs** (content hashes, not file-path payloads) with **deterministic manifests** and **root-audit gating**, while preserving existing FULL/SPLIT behavior unless explicitly migrated.

This is the next substrate step after **P.1 6-bucket migration**.

---

## Non-negotiable principles
1) **Deterministic output**: identical inputs → byte-identical pack outputs + identical CAS refs.
2) **Fail-closed**: if CAS write/verify/audit fails, the pack run must fail and must not claim success.
3) **No legacy roots**: packer may not reference legacy directory names in code or docs.
4) **Minimal blast radius**: changes limited to packer engine + packer tests + packer docs (and any tiny, necessary import wiring).
5) **Audit before “done”**: any pack run that emits CAS outputs must pass `root_audit` Mode B.

---

## Repo anchors (discover, don’t assume)
Packer is expected under:
- `MEMORY/LLM_PACKER/Engine/packer/`

Locate the active packer modules in your working tree:
```bash
git ls-files "*MEMORY/LLM_PACKER/Engine/packer/*.py"
```

You should see at least:
- `core.py`, `split.py`, `lite.py`, possibly `cli.py`, `archive.py`

All changes below apply to those located paths.

---

## Definitions
### CAS ref format
Canonical content reference:
- `sha256:<64 lowercase hex>`

### Manifest
A deterministic JSON document describing pack contents **by path + CAS ref**, not embedded payload.

### Roots (GC safety contract)
Roots are **the only liveness anchors** under Policy B.
If something is not rooted, it is eligible for GC.
Therefore: **every artifact needed must appear in roots** (directly, not by implication).

---

## P.2.0 What changes at a glance
### Before
- LITE outputs contain lightweight indexes and sometimes text excerpts.
- FULL/SPLIT contain bodies.
- Outputs are addressed by file paths.

### After (P.2)
- LITE becomes **manifest-only**: indexes + manifest + receipts.
- File contents are stored in CAS, and LITE references them by `sha256:<hash>`.
- Packer emits immutable run artifacts:
  - `TASK_SPEC`
  - `OUTPUT_HASHES`
  - final `STATUS`
- Packer writes roots (RUN_ROOTS) that include **all required hashes** and must pass `root_audit` Mode B.

**Note:** FULL/SPLIT may remain body-embedded for now (unless you explicitly extend P.2 scope). LITE is the forced change.

---

## P.2.1 LITE becomes manifest-only
### Goal
Ensure LITE contains:
- A small human index (optional but fine)
- A deterministic JSON manifest
- Any deterministic receipts (hashes, provenance) that are not payloads

And **never** contains concatenated repo file bodies.

### Requirements
1) Remove any logic that writes repo file bodies into `LITE/`.
2) LITE must not include large content; only:
   - pointers (CAS refs)
   - counts, sizes, metadata
   - stable lists
3) Any “preview text” feature must be removed or gated behind an explicit non-default flag (but do not add new flags in P.2 unless already present).

### Mechanical check
Assert LITE has no raw repo bodies by:
- limiting LITE writer code to operate only on metadata/refs
- adding tests that fail if LITE files include known repo content strings from fixtures

---

## P.2.2 Write pack payloads into CAS using CAPABILITY/ARTIFACTS
### Goal
During pack build, for every included file in scope:
- read raw bytes
- store in CAS via artifact store
- record `path → sha256:<hash>` in the manifest

### Requirements
1) Use existing substrate:
- CAS: `CAPABILITY/CAS/cas.py`
- Artifact store: `CAPABILITY/ARTIFACTS/store.py`

Prefer:
- `store_file(path)` for file payloads
- `store_bytes(data)` only for generated/virtual documents (manifest itself, receipts, etc.)

2) Validate returned refs:
- must match `^sha256:[0-9a-f]{64}$`
- any invalid ref is FAIL.

3) Optional but recommended verification:
- immediately `load_bytes(ref)` and compare with original bytes for critical outputs
- or at minimum verify CAS object exists and is readable

Fail-closed on mismatch/corruption.

---

## P.2.3 Deterministic manifest format (strict)
### Manifest file
Write to:
- `LITE/PACK_MANIFEST.json` (or your existing name, but lock one canonical name)

### Manifest schema (minimum)
Stable top-level keys in stable order (canonical JSON encoding in tests):
- `version`: string (ex: `"P2.0"`)
- `scope`: string (ex: `"ags"`)
- `repo_state`: object (commit hash, branch optional)
- `buckets`: list of bucket roots included (stable order)
- `entries`: list of objects, each:
  - `path`: repo-relative path (posix-style)
  - `ref`: CAS ref `sha256:<hash>`
  - `bytes`: integer size
  - `ext`: file extension (optional)
  - `kind`: `"FILE"` (optional)

### Ordering rules
- `entries` must be sorted by `path` ascending
- deduplicate by `path` (no duplicates allowed)
- `buckets` must be stable order:
  - `LAW`, `CAPABILITY`, `NAVIGATION`, `DIRECTION`, `THOUGHT`, `MEMORY`, then `.github` if included

### Canonical JSON
- UTF-8
- `sort_keys=True`
- no whitespace variance
- stable newline at EOF (optional)

---

## P.2.4 Emit immutable run artifacts (TASK_SPEC / OUTPUT_HASHES / STATUS)
### Goal
Packer becomes a first-class run emitter using existing Z.2.3 run record primitives.

Use:
- `CAPABILITY/RUNS/records.py`
  - `put_task_spec(spec: dict) -> str`
  - `put_output_hashes(hashes: list[str]) -> str`
  - `put_status(status: dict) -> str`

### Task spec content (deterministic)
`TASK_SPEC` should include only deterministic inputs:
- `scope`
- packer version / git commit (from repo state)
- include/exclude settings (bucket list, excluded dirs)
- pack mode(s) produced (`FULL`, `SPLIT`, `LITE`)
- manifest schema/version

**Hard prohibition:** no timestamps, no machine-specific absolute paths.

### Output hashes (what goes in)
`OUTPUT_HASHES` list must include **every CAS ref required to reconstruct the pack**, minimum:
- CAS ref for `PACK_MANIFEST.json` (store_bytes of canonical JSON)
- CAS refs for every file payload referenced by manifest

Optionally include:
- CAS ref for the LITE human index markdown (if generated)
- CAS refs for any deterministic receipts you choose to store in CAS

### Final status record
`STATUS` must include:
- `"state": "COMPLETED"` (or your canonical finished state)
- `task_spec_ref`
- `output_hashes_ref`
- `manifest_ref`
- `cas_snapshot_hash` (if available)
- `verdict`: `"PASS"` / `"FAIL"` (status must never claim PASS if audits fail)

---

## P.2.5 Roots emission (Policy B compliant)
### Goal
Write/update roots so GC cannot remove required objects.

### Rule (P0 conservative)
Because GC/mark traversal may not chase references, **roots must include every required output hash explicitly**.

Therefore RUN_ROOTS must include:
- `task_spec_ref`
- `output_hashes_ref`
- `status_ref`
- `manifest_ref`
- every payload `ref` included in manifest entries

### Where roots live
Use the same `runs_dir` convention used by:
- `CAPABILITY/GC/gc.py`
- `CAPABILITY/AUDIT/root_audit.py`

Do NOT invent a new directory.
If uncertain, inspect those modules to find the default `runs_dir` and root filenames and follow them.

---

## P.2.6 Gate pack completion on `root_audit` Mode B
### Goal
Before declaring pack success:
1) write CAS objects
2) write run records
3) write roots
4) call:
   - `root_audit(output_hashes_record=<OUTPUT_HASHES_REF>, ...)`
5) if verdict != PASS → fail closed

### Requirement
- Packer must not emit a “success receipt” unless root audit passes.

---

## P.2.7 Update CLI/runner wiring minimally
Only if needed:
- Ensure existing packer CLI path triggers the new behavior automatically for LITE.
- Avoid adding new flags in P.2 (unless already existing and you are fixing an inconsistency).
- Preserve existing output directory structure:
  - `FULL/`, `SPLIT/`, `LITE/`, `archive/`

---

## P.2.8 Tests (must be added)
Add a new packer test suite under the packer test area (use repo conventions; do not scatter tests).

Minimum tests:
1) **Determinism**: two pack builds with same inputs yield:
   - identical `PACK_MANIFEST.json` bytes
   - identical `manifest_ref`
   - identical `output_hashes_ref`
   - identical root_audit receipt (if saved)
2) **LITE manifest-only**: verify LITE does not contain raw file bodies (fixture with recognizable content).
3) **Manifest ordering**: entries sorted, no dup paths.
4) **Ref validation**: any invalid ref fails closed.
5) **Root completeness**: if you remove one payload from roots, `root_audit` Mode B fails and pack fails.
6) **Backward compatibility**: FULL/SPLIT generation remains unchanged (smoke test only).

Hard constraint:
- Tests must be runnable on Windows and Linux without relying on absolute paths or filesystem case quirks.

---

## Mechanical verification checklist
Run:
```bash
pytest -q
```

Confirm no legacy literals in packer code:
```bash
git grep -nE "\b(CANON|CONTEXT|MAPS|SKILLS|CONTRACTS|CORTEX|TOOLS)/" -- MEMORY/LLM_PACKER/Engine/packer
```

Confirm deterministic manifest and refs:
- build pack twice
- compare `PACK_MANIFEST.json` byte equality
- compare recorded `manifest_ref` and `output_hashes_ref`

---

## Exit criteria (ship gate)
All must be true:
1) Packer builds LITE as manifest-only.
2) Manifest references payloads via `sha256:<hash>`.
3) `TASK_SPEC`, `OUTPUT_HASHES`, `STATUS` emitted via Z.2.3 primitives (immutable CAS-stored records).
4) Roots written (Policy B), including **all payload refs**.
5) `root_audit` Mode B PASS is enforced before success.
6) Determinism proof passes across two builds.
7) All tests pass.
8) No legacy root literals in packer sources.

---

## Suggested commit message
`P.2: CAS-addressed LITE manifests + root-audit gated pack completion`

---

## Guardrails (do not violate)
- Do not add GC features, pinning semantics changes, or traversal expansions.
- Do not refactor CAS/ARTIFACTS/RUNS/AUDIT beyond minimal imports.
- Do not embed timestamps or machine paths in any CAS-stored record.
- No “best effort” warnings: failures must stop the run.

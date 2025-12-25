# SPECTRUM-06: Restore Runner Semantics

**Version:** 1.0.1
**Status:** Frozen
**Created:** 2025-12-25
**Updated:** 2025-12-25
**Depends on:** SPECTRUM-05 v1.0.0

---

## 1. Purpose

SPECTRUM-06 defines the exact semantics for restoring durable outputs from a verified bundle or chain to a target filesystem location. Restore is a world-mutating operation and MUST be gated by SPECTRUM-05 acceptance.

This specification is LAW. Once frozen, no implementation may deviate. Any ambiguity rejects.

---

## 2. Scope

This specification defines:
- Restore input requirements and eligibility
- Restore target model and path safety
- Single-bundle restore semantics
- Chain restore semantics and conflict resolution
- Atomicity and failure handling

This specification does NOT define:
- Verification procedure (see SPECTRUM-05)
- Signing or identity (see SPECTRUM-04)
- Network protocols or remote restore
- Incremental or partial restore modes

---

## 3. Inputs and Eligibility (Normative)

### 3.1 Supported Inputs

SPECTRUM-06 accepts exactly two input modes:

| Mode | Input | Description |
|------|-------|-------------|
| **Single** | `run_dir` | Single run directory containing a verified bundle |
| **Chain** | `[run_dir, ...]` | Ordered list of run directories forming a verified chain |

### 3.2 Eligibility Rules

A bundle is **eligible for restore** if and only if ALL conditions are met:

| Condition | Requirement |
|-----------|-------------|
| **Verification** | Bundle MUST pass `verify_bundle_spectrum05(strict=True)` with `ok=True` |
| **Proof presence** | `PROOF.json` MUST exist in run directory |
| **Proof validity** | `PROOF.json.restoration_result.verified` MUST equal `true` (boolean) |
| **Hashes presence** | `OUTPUT_HASHES.json` MUST exist in run directory |
| **Hashes completeness** | `OUTPUT_HASHES.json.hashes` MUST contain at least one entry |

If ANY condition is not met: **REJECT** with `RESTORE_INELIGIBLE`.

### 3.3 Chain Eligibility

A chain is **eligible for restore** if and only if:
1. Chain passes `verify_chain_spectrum05(strict=True)` with `ok=True`
2. Every bundle in the chain is individually eligible per Section 3.2
3. Chain is non-empty

---

## 4. Restore Target Model (Normative)

### 4.1 Explicit Restore Root

The restore target is specified via an explicit `restore_root` path.

**Requirements:**
- `restore_root` MUST be provided by the caller
- `restore_root` MUST be an absolute path
- `restore_root` MUST exist as a directory before restore begins
- `restore_root` MUST be writable

If any requirement is not met: **REJECT** with `RESTORE_TARGET_INVALID`.

### 4.2 Path Safety Rules

All restored paths are subject to strict safety rules:

**Step 4.2.1:** For each path in `OUTPUT_HASHES.json.hashes`:
- Normalize path to POSIX format (forward slashes)
- Remove any leading slashes
- Resolve to absolute: `restore_root / normalized_path`

**Step 4.2.2:** Validate the resolved path:
- MUST be lexically under `restore_root` (no `..` traversal)
- MUST NOT contain null bytes
- MUST NOT be a symlink pointing outside `restore_root`

If any validation fails: **REJECT** with `PATH_ESCAPE_DETECTED`.

### 4.3 Path Normalization

All paths are normalized per these rules:
1. Convert backslashes to forward slashes
2. Collapse multiple consecutive slashes to single slash
3. Remove leading slash (paths are relative)
4. Reject empty paths
5. Reject paths containing only `.` or `..` components

---

## 5. Single-Bundle Restore Semantics (Normative)

### 5.1 Procedure Overview

Single-bundle restore executes in exactly this order:

1. **Preflight verification** (Section 5.2)
2. **Build restore plan** (Section 5.3)
3. **Execute restore** (Section 5.4)
4. **Post-restore verification** (Section 5.5)

Any failure at any step terminates restore with **REJECT**.

### 5.2 Phase 1: Preflight Verification

**Step 1.1:** Verify bundle eligibility per Section 3.2.
- If ineligible: REJECT with `RESTORE_INELIGIBLE`

**Step 1.2:** Verify `restore_root` per Section 4.1.
- If invalid: REJECT with `RESTORE_TARGET_INVALID`

**Step 1.3:** Build path safety map per Section 4.2.
- If any path unsafe: REJECT with `PATH_ESCAPE_DETECTED`

### 5.3 Phase 2: Build Restore Plan

**Step 2.1:** Extract `hashes` object from `OUTPUT_HASHES.json`.

**Step 2.2:** For each `(relative_path, expected_hash)` in `hashes`:
- Compute `source_path = project_root / relative_path`
- Compute `target_path = restore_root / relative_path`
- Add to restore plan: `{source, target, hash}`

**Step 2.3:** Validate restore plan:
- All source files MUST exist
- All source files MUST be regular files (not directories or symlinks)
- If any source missing: REJECT with `SOURCE_MISSING`

### 5.4 Phase 3: Execute Restore

**Overwrite Policy: REJECT-IF-EXISTS**

SPECTRUM-06 mandates reject-if-exists semantics:
- If any `target_path` already exists: REJECT with `TARGET_EXISTS`
- Restore MUST NOT overwrite existing files
- Caller must explicitly clear `restore_root` if overwrite is intended

**Atomicity Model: STAGING + RENAME**

Restore uses a staging directory for atomicity:

**Step 3.1:** Create staging directory: `restore_root/.spectrum06_staging_<uuid>`

**Step 3.2:** For each entry in restore plan (deterministic order by `relative_path`):
- Create parent directories in staging as needed
- Copy source file to staging: `staging_root / relative_path`
- Verify copy integrity: compute SHA-256 of staged file
- If hash mismatch: REJECT with `COPY_INTEGRITY_FAILED`

**Step 3.3:** If all copies succeed:
- Move all staged files to final locations atomically
- Remove staging directory

**Step 3.4:** If any copy fails:
- Remove staging directory completely
- REJECT with appropriate error

### 5.5 Phase 4: Post-Restore Verification

**Step 4.1:** For each entry in restore plan:
- Read target file from `restore_root / relative_path`
- Compute `actual_hash = "sha256:" + lowercase_hex(sha256(file_bytes))`
- Compare to `expected_hash`

**Step 4.2:** If all hashes match: **ACCEPT**

**Step 4.3:** If any hash mismatch:
- Log mismatch details
- REJECT with `RESTORE_VERIFICATION_FAILED`

---

## 6. Chain Restore Semantics (Normative)

### 6.1 Chain Order

Chain restore processes bundles in the exact order provided:
- First bundle in list → restored first
- Last bundle in list → restored last
- Order is caller-specified and deterministic

### 6.2 Restore Layout: Per-Run Subfolders

Chain restore uses per-run subfolder isolation:

```
restore_root/
├── <run_id_1>/
│   └── <outputs from bundle 1>
├── <run_id_2>/
│   └── <outputs from bundle 2>
└── <run_id_N>/
    └── <outputs from bundle N>
```

Where `run_id` is the final path component of each `run_dir`.

### 6.3 Conflict Handling

**Path Conflicts Within a Bundle:**
- Not possible (OUTPUT_HASHES keys are unique within a bundle)

**Run ID Conflicts Across Chain:**
- If two bundles have the same `run_id`: REJECT with `CHAIN_DUPLICATE_RUN`
- This is already enforced by SPECTRUM-05 chain verification

**Target Directory Conflicts:**
- If `restore_root / run_id` already exists: REJECT with `TARGET_EXISTS`

### 6.4 Chain Restore Procedure

**Step C.1:** Verify chain eligibility per Section 3.3.

**Step C.2:** Extract `run_id` list from chain.

**Step C.3:** Verify no duplicate `run_id` values.

**Step C.4:** For each bundle in chain order:
- Compute `bundle_restore_root = restore_root / run_id`
- Create `bundle_restore_root` directory
- Execute single-bundle restore (Sections 5.2-5.5) with `bundle_restore_root`
- If any bundle restore fails: REJECT entire chain

**Step C.5:** If all bundle restores succeed: **ACCEPT**

### 6.5 Chain Atomicity

Chain restore is atomic at the chain level:

**Step C.6:** Before any bundle restore:
- Create chain staging manifest: `restore_root/.spectrum06_chain_<uuid>.json`
- Record intended bundle restore order

**Step C.7:** If any bundle restore fails:
- Rollback all previously restored bundles (remove their directories)
- Remove chain manifest
- REJECT with `CHAIN_RESTORE_FAILED`

**Step C.8:** On full chain success:
- Remove chain manifest
- ACCEPT

---

## 7. Restore Result Artifacts (Normative)

This section freezes the exact restore result artifacts produced on **successful** restore completion. These artifacts are required to be byte-identical across independent implementations given identical inputs.

### 7.1 Artifact Set (Required)

On successful restore completion, the Restore Runner MUST write exactly these **result** artifacts:
- `RESTORE_MANIFEST.json`
- `RESTORE_REPORT.json`

No restore result artifacts other than the two above may be persisted after successful restore completion.

### 7.2 Artifact Write Location (Required)

Restore result artifacts MUST be written only under `restore_root`.

Define `result_root` as:
- **Single mode:** `result_root = restore_root`
- **Chain mode:** for each restored bundle, `result_root = restore_root / run_id` (Section 6.2)

Write locations (exact filenames):
- `result_root/RESTORE_MANIFEST.json`
- `result_root/RESTORE_REPORT.json`

The Restore Runner MUST NOT write restore result artifacts outside `restore_root` under any circumstances.

### 7.3 Canonical JSON Serialization (Required)

Both artifacts MUST be serialized as canonical JSON bytes with all of the following properties:
- UTF-8 encoding
- No UTF-8 BOM
- No trailing whitespace
- No trailing newline
- Objects serialized with keys sorted lexicographically by UTF-8 byte value
- Arrays serialized in the exact order defined by this spec
- No insignificant whitespace (equivalent to JSON separators `,` and `:` only)

Any deviation is ambiguity and MUST be rejected.

### 7.4 RESTORE_MANIFEST.json (Required)

#### 7.4.1 Purpose

`RESTORE_MANIFEST.json` is the authoritative list of files restored into `result_root` for a single restored bundle.

#### 7.4.2 Definition: “restored file”

A “restored file” is a regular file that:
1. Was copied into `result_root` as part of restore execution (Section 5.4), and
2. Has a corresponding entry in the restored bundle’s `OUTPUT_HASHES.json.hashes` map, and
3. Exists at `result_root / relative_path` after successful restore completion.

No other files are considered restored files for purposes of the manifest.

#### 7.4.3 Top-Level Schema (Required)

`RESTORE_MANIFEST.json` MUST be a JSON object with exactly these required fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `entries` | array of object | **Normative.** Ordered list of restored files. |

No other top-level fields are permitted.

#### 7.4.4 Entry Schema (Required)

Each element of `entries` MUST be a JSON object with exactly these required fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `relative_path` | string | **Normative.** Output-relative path using forward slashes; MUST match a key in `OUTPUT_HASHES.json.hashes`. |
| `sha256` | string | **Normative.** MUST equal the expected hash string from `OUTPUT_HASHES.json.hashes[relative_path]` (including the `sha256:` prefix and lowercase hex). |
| `bytes` | integer | **Normative.** Exact byte length of the restored file at `result_root / relative_path`. |

No other entry fields are permitted.

#### 7.4.5 Entry Ordering (Required)

`entries` MUST be ordered by ascending `relative_path`, compared lexicographically by UTF-8 byte value.

#### 7.4.6 Manifest Production Requirement (Required)

Successful restore completion requires producing a valid `RESTORE_MANIFEST.json`.

If `RESTORE_MANIFEST.json` cannot be produced deterministically and completely (for any reason), then the restore MUST NOT be considered successful.

### 7.5 RESTORE_REPORT.json (Required)

#### 7.5.1 Purpose

`RESTORE_REPORT.json` is the required, minimal summary of a successful restore into `result_root`.

#### 7.5.2 Top-Level Schema (Required)

`RESTORE_REPORT.json` MUST be a JSON object with exactly these required fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `ok` | boolean | **Normative.** MUST equal `true`. |
| `restored_files_count` | integer | **Normative.** MUST equal the number of manifest entries. |
| `restored_bytes` | integer | **Normative.** MUST equal the sum of `bytes` across manifest entries. |
| `bundle_roots` | array of string | **Normative.** MUST contain exactly one bundle root (64 lowercase hex) for the restored bundle. |
| `chain_root` | string or null | **Normative.** If the restore invocation was chain mode, MUST equal the chain root computed by SPECTRUM-05 chain verification; otherwise MUST be `null`. |

No other report fields are permitted.

#### 7.5.3 Informational vs Normative Values

All fields in `RESTORE_REPORT.json` are normative.

### 7.6 Cross-Artifact Invariants (Required)

On successful restore completion, all invariants below MUST hold:

1. `RESTORE_REPORT.json.restored_files_count` MUST equal `len(RESTORE_MANIFEST.json.entries)`.
2. `RESTORE_REPORT.json.restored_bytes` MUST equal `sum(entry.bytes for entry in RESTORE_MANIFEST.json.entries)`.
3. For every `entry` in `RESTORE_MANIFEST.json.entries`:
   - `entry.relative_path` MUST be a key in the restored bundle’s `OUTPUT_HASHES.json.hashes`.
   - `entry.sha256` MUST equal the corresponding `OUTPUT_HASHES.json.hashes[entry.relative_path]` value exactly.
   - The SHA-256 of the restored file bytes at `result_root / entry.relative_path` MUST equal `entry.sha256` exactly.

If any invariant does not hold, the restore MUST NOT be considered successful.

---

## 7. Error Codes (Normative)

| Error Code | Condition |
|------------|-----------|
| `RESTORE_INELIGIBLE` | Bundle does not meet eligibility requirements |
| `RESTORE_TARGET_INVALID` | restore_root missing, not directory, or not writable |
| `PATH_ESCAPE_DETECTED` | Restored path would escape restore_root |
| `SOURCE_MISSING` | Source file from OUTPUT_HASHES does not exist |
| `TARGET_EXISTS` | Target file already exists (reject-if-exists policy) |
| `COPY_INTEGRITY_FAILED` | Staged file hash does not match during copy |
| `RESTORE_VERIFICATION_FAILED` | Post-restore hash verification failed |
| `CHAIN_RESTORE_FAILED` | One or more bundles in chain failed restore |

---

## 8. Determinism Requirements (Normative)

### 8.1 Deterministic Outcomes

Given identical inputs:
- Same `run_dir` or chain
- Same `restore_root`
- Same filesystem state

Restore MUST produce:
- Identical file contents at identical relative paths
- Identical success/failure outcome
- Identical error code on failure

### 8.2 No Interpretation

All rules in this specification are:
- Explicit
- Unambiguous
- Complete
- Testable

If any rule appears to require interpretation, that is a defect in this specification.

---

## 9. References

- SPECTRUM-04 v1.1.0: Validator Identity and Signing Law
- SPECTRUM-05 v1.0.0: Verification and Threat Law

---

## 10. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-25 | Initial frozen specification |

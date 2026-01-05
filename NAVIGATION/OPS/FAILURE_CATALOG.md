# Failure Catalog

This document catalogs expected fail-closed errors by subsystem, including failure codes, trigger conditions, detection signals, and safe recovery steps.

## CAS (Content-Addressable Storage)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `CAS_INVALID_HASH` | Invalid Hash Format | Hash string is not 64 lowercase hex characters | `InvalidHashException` with message "Invalid hash format" | Verify hash source and regenerate if corrupted; ensure lowercase hex format |
| `CAS_OBJECT_NOT_FOUND` | Object Not Found | Attempted to retrieve non-existent CAS object | `ObjectNotFoundException` with message "Object not found: <hash>" | Check if GC deleted unrooted object; restore from backup or source material |
| `CAS_CORRUPT_OBJECT` | Corrupt Object | Stored data does not match its hash | `CorruptObjectException` with message "Corruption detected for hash: <hash>" or "Stored data verification failed" | Delete corrupted object; re-store from source; run integrity check on related blobs |
| `CAS_IO_ERROR` | I/O Error | Filesystem error during read/write | `CASException` with message "Failed to read/write object" | Check disk space and permissions; retry operation; check filesystem integrity |

## ARTIFACTS (Artifact Store)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `ARTIFACT_INVALID_REF` | Invalid Reference Format | Reference string does not match `sha256:<64hex>` or is not a valid file path | `InvalidReferenceException` with message "Invalid CAS reference format" or "Invalid file path" | Verify reference source; ensure CAS refs use exact `sha256:` prefix format |
| `ARTIFACT_OBJECT_NOT_FOUND` | Artifact Not Found | CAS-referenced object does not exist or file path missing | `ObjectNotFoundException` with message "File not found: <path>" | For CAS refs: check if object exists in CAS or was GC'd; for file paths: verify file exists |
| `ARTIFACT_NOT_A_FILE` | Path Is Not File | Reference path points to directory or does not exist | `InvalidReferenceException` with message "Path is not a file: <path>" | Verify correct path; use file path, not directory path |

## RUNS (Run Records)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `RUNS_INVALID_INPUT` | Invalid Input Data | Task spec, status, or output hashes are not JSON-serializable or malformed | `InvalidInputException` with message "not JSON-serializable" or "must be a dict/list" | Validate input structure; ensure proper JSON serialization; fix malformed data |
| `RUNS_EMPTY_TASK_SPEC` | Empty Task Spec | Attempted to store empty task specification | `InvalidInputException` with message "Task spec cannot be empty" | Ensure task spec has required fields; add missing data before storing |
| `RUNS_DECODE_ERROR` | JSON Decode Error | Stored JSON bytes cannot be decoded | `RunRecordException` with message "Failed to decode JSON" | Verify CAS object integrity; re-store with valid JSON; check for encoding issues |
| `RUNS_INVALID_HASH_FORMAT` | Invalid Run Hash | Hash string is not valid for run record lookup | `InvalidInputException` with error about hash format | Ensure 64-character lowercase hex hash; check for typo or corruption |

## GC (Garbage Collection)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `GC_ROOT_ENUM_FAILED` | Root Enumeration Failed | RUN_ROOTS.json or GC_PINS.json is malformed or missing | `RootEnumerationException` with message about invalid JSON or hash format | Validate JSON syntax; ensure list of strings; check hash format; restore from backup |
| `GC_LOCK_FAILED` | Lock Acquisition Failed | Cannot acquire GC lock (another GC in progress) | `LockException` with message about lock | Wait for other GC to complete; check for stale lock file; manually remove if no active GC |
| `GC_EMPTY_ROOTS` | Empty Roots Without Permission | Root enumeration returns zero roots but `allow_empty_roots=False` | GC fails with policy error before any deletions | Do NOT run GC without explicit `allow_empty_roots=True`; check RUN_ROOTS.json exists; verify root tracking is working |

## AUDIT (Root Audit)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `AUDIT_RUN_ROOTS_INVALID` | RUN_ROOTS.json Invalid | JSON malformed or not a list | Audit returns error list: "RUN_ROOTS: Invalid JSON" or "RUN_ROOTS: Must be a list" | Fix JSON syntax; ensure top-level list; restore from backup |
| `AUDIT_RUN_ROOTS_HASH_INVALID` | RUN_ROOTS Hash Invalid | Root entry is not a valid 64-char hex hash | Audit returns error: "RUN_ROOTS[{i}]: Invalid hash format" | Validate each hash string; correct format; remove invalid entries |
| `AUDIT_GC_PINS_INVALID` | GC_PINS.json Invalid | JSON malformed or not a list | Audit returns error: "GC_PINS: Invalid JSON" or "GC_PINS: Must be a list" | Fix JSON syntax; ensure top-level list; restore from backup |
| `AUDIT_PIN_HASH_INVALID` | GC Pin Hash Invalid | Pin entry is not a valid 64-char hex hash | Audit returns error: "GC_PINS[{i}]: Invalid hash format" | Validate each hash string; correct format; remove invalid entries |
| `AUDIT_UNREACHABLE_OUTPUT` | Unreachable Output | Run's output hashes not in reachable set | Audit error: "OUTPUT_HASHES not reachable for run: <run_id>" | Check if run is rooted; verify reference chain; re-run to regenerate if lost |

## SKILL_RUNTIME (Skill Execution)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `SKILL_INPUT_READ_ERROR` | Input Read Failed | Cannot read or parse input.json for skill | Skill run.py returns exit code 1 with "Error reading input JSON" | Verify input.json exists and is valid JSON; check permissions; fix malformed JSON |
| `SKILL_CANON_INCOMPAT` | Canon Version Incompatible | Loaded canon version does not match skill's required range | `ensure_canon_compat()` returns False, exit code 1 | Update skill's required_canon_version or upgrade canon; check VERSIONING.md |
| `SKILL_FIXTURE_FAILED` | Fixture Validation Failed | Actual output does not match expected output | validate.py returns exit code 1 with diff output | Review fixture diff; fix implementation or update expected.json if behavior changed |
| `SKILL_RUNTIME_ERROR` | Runtime Error | Skill execution threw unhandled exception | Exit code 1 with Python traceback | Check traceback for root cause; fix skill logic; add proper exception handling |

## PACKER (Pack Consumption)

| Failure Code | Name | Trigger Condition | Detection Signal | Safe Recovery |
|-------------|------|------------------|-----------------|--------------|
| `PACK_CONSUME_INVALID_REF` | Invalid Manifest Reference | Manifest ref does not start with `sha256:` or has invalid hash | `ValueError` with message "PACK_CONSUME_INVALID_REF" or "PACK_CONSUME_INVALID_HASH" | Verify manifest ref format; ensure 64-char lowercase hex after `sha256:` |
| `PACK_CONSUME_MISSING_FIELD` | Missing Manifest Field | Manifest missing required top-level fields | `ValueError` with message "PACK_CONSUME_MISSING_FIELD: <field>" | Add missing field(s): version, scope, entries |
| `PACK_CONSUME_VERSION_MISMATCH` | Version Mismatch | Manifest version not P2.0 | `ValueError` with message "PACK_CONSUME_VERSION_MISMATCH" | Update manifest to version P2.0 or upgrade packer to support version |
| `PACK_CONSUME_INVALID_SCOPE` | Invalid Scope | Manifest scope not one of: ags, lab, cat | `ValueError` with message "PACK_CONSUME_INVALID_SCOPE" | Set valid scope: ags, lab, or cat |
| `PACK_CONSUME_INVALID_ENTRIES` | Invalid Entries List | Manifest entries is not a list | `ValueError` with message "PACK_CONSUME_INVALID_ENTRIES" | Ensure entries field is a list of entry objects |
| `PACK_CONSUME_ENTRY_MISSING_FIELD` | Entry Missing Field | Entry missing required field: path, ref, bytes, kind | `ValueError` with message "PACK_CONSUME_ENTRY_MISSING_FIELD" | Add missing fields to entry |
| `PACK_CONSUME_ABSOLUTE_PATH` | Absolute Path Detected | Entry path is absolute (starts with /) | `ValueError` with message "PACK_CONSUME_ABSOLUTE_PATH" | Use relative path only; remove leading `/` |
| `PACK_CONSUME_PATH_TRAVERSAL` | Path Traversal Detected | Entry path contains `..` component | `ValueError` with message "PACK_CONSUME_PATH_TRAVERSAL" | Remove `..` from path; use clean relative path |
| `PACK_CONSUME_MISSING_BLOB` | Missing CAS Blob | Referenced blob does not exist in CAS | pack_consume errors or ConsumptionReceipt shows missing blobs | Restore missing blobs from backup or source; re-run pack creation |

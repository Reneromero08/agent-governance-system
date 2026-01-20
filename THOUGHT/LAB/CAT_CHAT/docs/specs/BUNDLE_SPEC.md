# Bundle Protocol Specification v5.0.0

## 1. Overview

A **bundle** is a deterministic, bounded, self-contained execution package built from a completed job. Bundles enable offline replay and verification without access to the original repository.

**Key Properties:**
- Deterministic: Same job produces byte-identical bundle
- Bounded: No unbounded slices (`slice=ALL` forbidden)
- Self-Contained: All required artifacts included
- Verifiable: Hash integrity at every level

## 2. Bundle Structure

### 2.1 Directory Layout

```
bundle_dir/
  bundle.json           # Manifest (canonical JSON)
  artifacts/
    <artifact_id>.txt   # Content files (UTF-8, trailing newline)
```

### 2.2 Manifest Schema

```json
{
  "bundle_version": "5.0.0",
  "bundle_id": "<sha256>",
  "run_id": "<string>",
  "job_id": "<string>",
  "message_id": "<string>",
  "plan_hash": "<sha256>",
  "steps": [...],
  "inputs": {
    "symbols": [...],
    "files": [...],
    "slices": [...]
  },
  "artifacts": [...],
  "hashes": {
    "root_hash": "<sha256>"
  },
  "provenance": {}
}
```

### 2.3 Step Schema

Steps are ordered by `(ordinal ASC, step_id ASC)`:

```json
{
  "step_id": "<string>",
  "ordinal": <integer>,
  "op": "READ_SYMBOL" | "READ_SECTION",
  "refs": {
    "symbol_id": "<string>",     // for READ_SYMBOL
    "section_id": "<string>"     // for READ_SECTION
  },
  "constraints": {
    "slice": "<slice_expr>"      // e.g., "lines[0:100]", "head(50)"
  },
  "expected_outputs": {}
}
```

### 2.4 Artifact Schema

Artifacts are ordered by `artifact_id ASC`:

```json
{
  "artifact_id": "<sha256_prefix_16>",
  "kind": "SYMBOL_SLICE" | "SECTION_SLICE",
  "ref": "<symbol_id or section_id>",
  "slice": "<applied_slice>",
  "path": "artifacts/<artifact_id>.txt",
  "sha256": "<content_hash>",
  "bytes": <integer>
}
```

## 3. Artifact Constraints

### 3.1 Boundedness Gate

- `slice=ALL` is **FORBIDDEN** for any artifact
- Every artifact MUST be referenced by at least one step
- Artifacts not referenced by steps cause build failure

### 3.2 Content Hash

Content hash is computed as:
```
SHA256(content.encode('utf-8'))
```

Where `content` includes the trailing newline (enforced on all artifact files).

### 3.3 Trailing Newline

All artifact files MUST end with exactly one newline character (`\n`). If original content lacks a trailing newline, one is appended during bundle build.

## 4. Hash Computation

### 4.1 Root Hash

Computed from artifacts sorted by `artifact_id`:

```python
sorted_artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
hash_strings = []
for artifact in sorted_artifacts:
    hash_strings.append(f"{artifact['artifact_id']}:{artifact['sha256']}")
combined = "\n".join(hash_strings) + "\n"
root_hash = SHA256(combined.encode('utf-8'))
```

### 4.2 Bundle ID

Computed from pre-manifest with empty `bundle_id` and `root_hash`:

```python
pre_manifest = manifest.copy()
pre_manifest["bundle_id"] = ""
pre_manifest["hashes"]["root_hash"] = ""
pre_manifest_json = canonical_json(pre_manifest)
bundle_id = SHA256(pre_manifest_json.encode('utf-8'))
```

### 4.3 Plan Hash

Computed from canonical steps:

```python
sorted_steps = sorted(steps, key=lambda x: (x["ordinal"], x["step_id"]))
canonical_steps = []
for step in sorted_steps:
    step_for_hash = {
        "step_id": step["step_id"],
        "ordinal": step["ordinal"],
        "op": step["op"],
        "refs": step.get("refs", {}),
        "constraints": step.get("constraints", {}),
        "expected_outputs": step.get("expected_outputs", {})
    }
    canonical_steps.append(step_for_hash)
canonical_plan = canonical_json({"run_id": run_id, "steps": canonical_steps})
plan_hash = SHA256(canonical_plan.encode('utf-8'))
```

## 5. Completeness Gate

Bundle build **FAILS** unless:

1. **All steps COMMITTED**: Every step in the job has `status = "COMMITTED"`
2. **Exactly one receipt per step**: Each step has exactly one receipt in `cassette_receipts`

```python
def check_job_complete(run_id, job_id):
    for step in get_steps(run_id, job_id):
        if step.status != "COMMITTED":
            return False, f"Step {step.step_id} not COMMITTED"
        if count_receipts(step.step_id) != 1:
            return False, f"Step {step.step_id} has wrong receipt count"
    return True, "Job complete"
```

## 6. Canonical JSON Rules

All JSON in bundles uses canonical format:

```python
json.dumps(data, sort_keys=True, separators=(",", ":"))
```

- **No whitespace**: Use `separators=(",", ":")` (no spaces)
- **Keys sorted**: Use `sort_keys=True`
- **UTF-8 encoding**: All strings UTF-8 encoded
- **Single trailing newline**: File ends with exactly one `\n`

## 7. Forbidden Fields

The following fields are **FORBIDDEN** at the manifest top level:

- `timestamp`
- `created_at`
- `updated_at`
- `cwd`
- `os`
- `locale`

Artifact paths must:
- Use forward slashes only (`/`)
- Be relative (no leading `/`)
- Contain no backslashes (`\`)

## 8. Verification Protocol

To verify a bundle:

1. **Load manifest**: Parse `bundle.json`
2. **Validate schema**: Check all required fields present
3. **Verify ordering**: Steps by `(ordinal, step_id)`, artifacts by `artifact_id`
4. **Verify artifact hashes**: Each artifact content matches declared SHA256
5. **Verify artifact sizes**: Each artifact byte count matches declared `bytes`
6. **Verify trailing newlines**: Each artifact ends with `\n`
7. **Recompute root_hash**: Compare against `hashes.root_hash`
8. **Recompute bundle_id**: Compare against `bundle_id`
9. **Validate boundedness**: No `ALL` slices, all artifacts referenced by steps
10. **Reject forbidden fields**: Check for forbidden top-level fields

If ANY check fails, verification **FAILS**. Fail-closed.

## 9. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Verification passed |
| 1 | Verification failed (hash, ordering, bounds) |
| 2 | Invalid input (missing file, bad JSON, schema invalid) |
| 3 | Internal error (unexpected exception) |

## 10. Implementation Reference

- Source: `catalytic_chat/bundle.py`
- Classes: `BundleBuilder`, `BundleVerifier`
- Schema: `SCHEMAS/bundle.schema.json`

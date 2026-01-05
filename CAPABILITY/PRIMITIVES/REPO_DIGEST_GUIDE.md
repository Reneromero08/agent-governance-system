# Phase 1.5B: Repo Digest, Purity Scan, and Restore Proof

## Overview

This module implements deterministic repo-state proofs that make catalysis measurable:
- **Pre/Post Repo Digest**: Tree hash with declared exclusions
- **RESTORE_PROOF**: PASS/FAIL receipt with diff summary
- **PURITY_SCAN**: Verifies no new/modified files outside durable roots; tmp roots empty

## Hard Invariants

1. **Never mutate original user content** as part of the scan
2. **Fail closed**: If digest or scan cannot be computed deterministically, emit error receipt and exit nonzero
3. **Canonical ordering everywhere**: paths, lists, diffs
4. **No crypto sealing here**: Handled by CRYPTO_SAFE phase

## Module Location

- **Implementation**: `CAPABILITY/PRIMITIVES/repo_digest.py`
- **Tests**: `CAPABILITY/TESTBENCH/integration/test_phase_1_5b_repo_digest.py`

## Usage

### Command Line Interface

```bash
# Generate PRE_DIGEST
python repo_digest.py \
  --repo-root /path/to/repo \
  --pre-digest PRE_DIGEST.json \
  --exclusions ".git,__pycache__,.mypy_cache" \
  --durable-roots "LAW/CONTRACTS/_runs,outputs" \
  --tmp-roots "_tmp,_scratch"

# Generate POST_DIGEST
python repo_digest.py \
  --repo-root /path/to/repo \
  --post-digest POST_DIGEST.json \
  --exclusions ".git,__pycache__,.mypy_cache" \
  --durable-roots "LAW/CONTRACTS/_runs,outputs" \
  --tmp-roots "_tmp,_scratch"

# Generate PURITY_SCAN
python repo_digest.py \
  --repo-root /path/to/repo \
  --purity-scan PRE_DIGEST.json POST_DIGEST.json PURITY_SCAN.json \
  --tmp-roots "_tmp,_scratch"

# Generate RESTORE_PROOF
python repo_digest.py \
  --repo-root /path/to/repo \
  --restore-proof PRE_DIGEST.json POST_DIGEST.json PURITY_SCAN.json RESTORE_PROOF.json \
  --exclusions ".git,__pycache__" \
  --durable-roots "LAW/CONTRACTS/_runs,outputs" \
  --tmp-roots "_tmp,_scratch"
```

### Programmatic Usage

```python
from pathlib import Path
from CAPABILITY.PRIMITIVES.repo_digest import (
    DigestSpec,
    RepoDigest,
    PurityScan,
    RestoreProof,
)

# Define spec
spec = DigestSpec(
    repo_root=Path("/path/to/repo"),
    exclusions=[".git", "__pycache__"],
    durable_roots=["LAW/CONTRACTS/_runs", "outputs"],
    tmp_roots=["_tmp", "_scratch"],
)

# Compute PRE_DIGEST
digest_pre = RepoDigest(spec)
pre_receipt = digest_pre.compute_digest()

# ... run catalytic operation ...

# Compute POST_DIGEST
digest_post = RepoDigest(spec)
post_receipt = digest_post.compute_digest()

# Perform PURITY_SCAN
scanner = PurityScan(spec)
purity_receipt = scanner.scan(pre_receipt, post_receipt)

# Generate RESTORE_PROOF
prover = RestoreProof(spec)
proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)

# Check verdict
if proof["verdict"] == "PASS":
    print("✓ Restoration verified: repo state unchanged")
else:
    print(f"✗ Restoration failed: {proof['diff_summary']}")
```

## Receipt Formats

### PRE_DIGEST.json / POST_DIGEST.json

```json
{
  "digest": "abc123...",
  "file_count": 1234,
  "file_manifest": {
    "path/to/file1.txt": "hash1...",
    "path/to/file2.txt": "hash2..."
  },
  "exclusions_spec_hash": "def456...",
  "module_version_hash": "ghi789...",
  "module_version": "1.5b.0"
}
```

**Fields**:
- `digest`: SHA-256 tree digest of all tracked files (deterministic)
- `file_count`: Number of files included in digest
- `file_manifest`: Map of {path: sha256_hash} for all tracked files
- `exclusions_spec_hash`: Hash of canonical exclusions spec (durable_roots + tmp_roots + exclusions)
- `module_version_hash`: Hash of module version string (for deterministic tracking)
- `module_version`: Module version (e.g., "1.5b.0")

**Determinism Guarantee**: Running digest twice on identical repo state produces identical digest.

### PURITY_SCAN.json

```json
{
  "verdict": "PASS",
  "violations": [],
  "tmp_residue": [],
  "scan_module_version_hash": "ghi789...",
  "module_version": "1.5b.0"
}
```

**PASS Example**:
```json
{
  "verdict": "PASS",
  "violations": [],
  "tmp_residue": [],
  "scan_module_version_hash": "abc...",
  "module_version": "1.5b.0"
}
```

**FAIL Example (tmp residue)**:
```json
{
  "verdict": "FAIL",
  "violations": [],
  "tmp_residue": [
    "_tmp/residue1.txt",
    "_tmp/residue2.txt"
  ],
  "scan_module_version_hash": "abc...",
  "module_version": "1.5b.0"
}
```

**Fields**:
- `verdict`: "PASS" or "FAIL"
- `violations`: Reserved for future use (currently empty)
- `tmp_residue`: List of files remaining in tmp_roots (sorted, canonical ordering)
- `scan_module_version_hash`: Hash of module version
- `module_version`: Module version

**Verdict Logic**:
- PASS: tmp_roots empty AND digest unchanged
- FAIL: tmp_roots non-empty OR digest changed

### RESTORE_PROOF.json

**PASS Example**:
```json
{
  "verdict": "PASS",
  "pre_digest": "abc123...",
  "post_digest": "abc123...",
  "tmp_roots": ["_tmp"],
  "durable_roots": ["outputs"],
  "exclusions": [".git"],
  "exclusions_spec_hash": "def456...",
  "proof_module_version_hash": "ghi789...",
  "module_version": "1.5b.0"
}
```

**FAIL Example (with diff)**:
```json
{
  "verdict": "FAIL",
  "pre_digest": "abc123...",
  "post_digest": "xyz789...",
  "tmp_roots": ["_tmp"],
  "durable_roots": ["outputs"],
  "exclusions": [".git"],
  "exclusions_spec_hash": "def456...",
  "proof_module_version_hash": "ghi789...",
  "module_version": "1.5b.0",
  "diff_summary": {
    "added": ["rogue.txt"],
    "removed": ["deleted.txt"],
    "changed": ["modified.txt"]
  }
}
```

**Fields**:
- `verdict`: "PASS" or "FAIL"
- `pre_digest`: Digest before catalytic operation
- `post_digest`: Digest after catalytic operation
- `tmp_roots`: List of tmp root paths (sorted)
- `durable_roots`: List of durable root paths (sorted)
- `exclusions`: List of exclusion paths (sorted)
- `exclusions_spec_hash`: Hash of canonical exclusions spec
- `proof_module_version_hash`: Hash of module version
- `module_version`: Module version
- `diff_summary` (only on FAIL): Deterministic diff with canonical ordering
  - `added`: Files added outside durable roots (sorted)
  - `removed`: Files removed outside durable roots (sorted)
  - `changed`: Files modified outside durable roots (sorted)

**Verdict Logic**:
- PASS: pre_digest == post_digest AND purity_scan.verdict == "PASS"
- FAIL: pre_digest != post_digest OR purity_scan.verdict == "FAIL"

## Interpreting Receipts

### ✓ Restoration Verified (PASS)

```
RESTORE_PROOF.json:
  verdict: PASS
  pre_digest == post_digest

PURITY_SCAN.json:
  verdict: PASS
  tmp_residue: []
```

**Interpretation**:
- Repo state unchanged outside durable roots
- No tmp residue
- Catalytic operation completed cleanly

### ✗ Restoration Failed: Files Added

```
RESTORE_PROOF.json:
  verdict: FAIL
  diff_summary.added: ["rogue.txt"]
```

**Interpretation**:
- File "rogue.txt" added outside durable roots
- Violation of catalytic purity
- Action: Review rogue.txt, move to durable root or delete

### ✗ Restoration Failed: Files Modified

```
RESTORE_PROOF.json:
  verdict: FAIL
  diff_summary.changed: ["source.py"]
```

**Interpretation**:
- File "source.py" modified outside durable roots
- Violation of catalytic purity (source mutation)
- Action: Restore source.py to pre-state or move to durable root

### ✗ Restoration Failed: Tmp Residue

```
PURITY_SCAN.json:
  verdict: FAIL
  tmp_residue: ["_tmp/work.txt"]

RESTORE_PROOF.json:
  verdict: FAIL
```

**Interpretation**:
- Tmp root "_tmp" not cleaned up
- File "_tmp/work.txt" remains after operation
- Action: Clean tmp roots, ensure cleanup in script

## Failure Mode Reference

| Failure Mode | PURITY_SCAN | RESTORE_PROOF | diff_summary | tmp_residue |
|--------------|-------------|---------------|--------------|-------------|
| File added outside durable roots | FAIL | FAIL | added: [path] | [] |
| File modified outside durable roots | FAIL | FAIL | changed: [path] | [] |
| File removed outside durable roots | FAIL | FAIL | removed: [path] | [] |
| Tmp residue present | FAIL | FAIL | - | [paths] |
| Durable-only writes | PASS | PASS | - | [] |
| No changes | PASS | PASS | - | [] |

## Determinism Guarantees

### Canonical Ordering

All lists in receipts are sorted:
- `file_manifest` keys (paths)
- `tmp_residue` paths
- `diff_summary.added`, `diff_summary.removed`, `diff_summary.changed`
- `tmp_roots`, `durable_roots`, `exclusions`

### Hash Computation

1. **Tree Digest**: SHA-256 of canonical file records:
   ```
   path1:hash1\n
   path2:hash2\n
   ...
   ```

2. **Exclusions Spec Hash**: SHA-256 of canonical JSON:
   ```json
   ["exclusion1", "exclusion2", ...]
   ```

3. **Module Version Hash**: SHA-256 of module version string

### Repeated Digest Guarantee

Running `compute_digest()` twice on identical repo state produces:
- Identical `digest`
- Identical `file_count`
- Identical `exclusions_spec_hash`
- Identical `module_version_hash`

## Testing

Run all fixture-backed tests:

```bash
pytest CAPABILITY/TESTBENCH/integration/test_phase_1_5b_repo_digest.py -v
```

**Test Coverage**:
- ✓ Deterministic digest (repeated -> same digest)
- ✓ New file outside durable roots -> purity FAIL + restore FAIL
- ✓ Modified file outside durable roots -> purity FAIL + restore FAIL
- ✓ Tmp residue -> purity FAIL
- ✓ Durable-only writes -> purity PASS + restore PASS
- ✓ Canonical ordering of paths in diff summaries
- ✓ Exclusions respected in digest computation
- ✓ Path normalization (forward-slash format)
- ✓ Canonical JSON determinism
- ✓ Empty repo digest
- ✓ Module version hash in all receipts

## Integration with Catalytic Runtime

To integrate with catalytic pipeline:

```python
# Before run: capture PRE_DIGEST
spec = DigestSpec(
    repo_root=REPO_ROOT,
    exclusions=[".git", "__pycache__"],
    durable_roots=["LAW/CONTRACTS/_runs"],
    tmp_roots=["_tmp"],
)
digest_pre = RepoDigest(spec)
pre_receipt = digest_pre.compute_digest()

# Write PRE_DIGEST to run directory
run_dir = REPO_ROOT / "LAW/CONTRACTS/_runs" / run_id
write_receipt(run_dir / "PRE_DIGEST.json", pre_receipt)

# ... execute catalytic operation ...

# After run: capture POST_DIGEST
digest_post = RepoDigest(spec)
post_receipt = digest_post.compute_digest()
write_receipt(run_dir / "POST_DIGEST.json", post_receipt)

# Perform PURITY_SCAN
scanner = PurityScan(spec)
purity_receipt = scanner.scan(pre_receipt, post_receipt)
write_receipt(run_dir / "PURITY_SCAN.json", purity_receipt)

# Generate RESTORE_PROOF
prover = RestoreProof(spec)
proof = prover.generate_proof(pre_receipt, post_receipt, purity_receipt)
write_receipt(run_dir / "RESTORE_PROOF.json", proof)

# Exit nonzero if proof failed
if proof["verdict"] == "FAIL":
    sys.exit(1)
```

## Exit Codes

- `0`: Success (PASS)
- `1`: Restoration failed (FAIL)
- `2`: Error (digest computation failed, invalid args, etc.)

## Path Normalization and Symlink Policy

### Normalization Rules

All paths in receipts are normalized to canonical form:
1. **Separator normalization**: Windows backslashes (`\`) → forward slashes (`/`)
2. **Trailing slash removal**: `path/to/dir/` → `path/to/dir`
3. **Relative-to-repo-root**: All paths stored as repo-relative (e.g., `CAPABILITY/PRIMITIVES/file.py`)

### Symlink and Junction Handling

**Current behavior** (Phase 1.5B):
- **File enumeration**: `os.walk()` follows symlinks by default
- **Risk**: Symlinks escaping repo root may cause digest to include external files
- **Mitigation**: Exclusion check rejects files outside repo root (via `try/except ValueError` on `relative_to()`)

**Symlink policy** (defense-in-depth):
1. **No escape via symlinks**: If a symlink points outside `repo_root`, the file is excluded (path normalization fails `relative_to()` check)
2. **Symlinks within repo**: Followed normally, hashed at target location
3. **Circular symlinks**: May cause issues; avoid circular symlink structures in repo

**Future hardening** (not in 1.5B):
- Explicit symlink detection with `Path.is_symlink()`
- `os.walk(followlinks=False)` to prevent following any symlinks
- Separate handling for symlinks-as-metadata vs symlinks-as-files

### Change Detection Semantics

**Bytes-only change detection**:
- Files compared by **SHA-256 hash of file content bytes only**
- **Metadata NOT included**: mtime, permissions, ownership, xattrs ignored
- **Determinism**: Identical file bytes → identical hash, regardless of metadata

**What triggers "changed" verdict**:
- File content bytes differ (different SHA-256 hash)

**What does NOT trigger "changed"**:
- Modification time (`mtime`) changed
- File permissions changed (e.g., `chmod +x`)
- File ownership changed
- Extended attributes changed

**Rationale**: Content-only comparison ensures determinism across platforms and file copies.

### Cross-Platform Behavior

**Windows vs Linux**:
- **Path separators**: Normalized to forward slashes on all platforms
- **Case sensitivity**: Paths stored as-is; comparison is platform-dependent (case-insensitive on Windows, case-sensitive on Linux)
- **Symlinks on Windows**: Requires admin privileges or Developer Mode; if unavailable, symlinks treated as regular files
- **Junctions on Windows**: Followed by `os.walk()` like symlinks

**WSL interop**:
- Paths with Windows backslashes normalized to forward slashes
- Works correctly in WSL accessing Windows filesystems (`/mnt/c/...`)

## Error Handling and Receipts

### Error Codes (Frozen)

Error codes are **append-only** and **never reused**.

| Code | Meaning | Introduced |
|------|---------|------------|
| `DIGEST_COMPUTATION_FAILED` | Digest computation failed (exception during file enumeration or hashing) | v1.5b.0 |
| `HASH_FAILED` | File hash computation failed (unreadable file, I/O error) | v1.5b.0 |
| `PURITY_SCAN_FAILED` | Purity scan failed (exception during scan) | v1.5b.0 |
| `RESTORE_PROOF_GENERATION_FAILED` | Restore proof generation failed (exception during proof) | v1.5b.0 |

**Freeze Rules**:
1. **Never reuse error codes**: Retired codes are reserved permanently
2. **Never change meanings**: Semantic meaning is immutable once defined
3. **Append-only**: New codes may be added with version annotations

### CLI Error Receipts

**Current behavior** (Phase 1.5B):
- CLI prints error to stderr and exits with code 2
- **Missing**: No error receipt written to disk on unexpected exceptions

**Required behavior** (Phase 1.5 Polish):
- On unexpected exception, CLI must:
  1. Write error receipt to a deterministic path (e.g., `ERROR_RECEIPT.json`)
  2. Include: operation attempted, exception type, error code, module version hash, config snapshot
  3. Exit with deterministic nonzero code (2 for errors, 1 for FAIL verdict)

## Future Work (Not Implemented in 1.5B)

- Crypto sealing (CRYPTO_SAFE phase)
- Cross-run proof chaining
- Incremental digest updates (for large repos)
- Compression of file_manifest for large repos
- Explicit symlink bypass protection (`followlinks=False` in `os.walk`)

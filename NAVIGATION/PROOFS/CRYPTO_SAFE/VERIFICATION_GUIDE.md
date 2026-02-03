# Release Verification Guide

This guide explains how to verify that an AGS release has not been tampered with.

## Overview

AGS releases are cryptographically sealed using Ed25519 signatures and SHA-256 file hashes. This creates a tamper-evident seal that proves "you broke my seal" without preventing access to the code.

**Key Files:**
- `RELEASE_MANIFEST.json` - Contains hashes of all tracked files
- `RELEASE_MANIFEST.json.sig` - Ed25519 signature of the manifest

## Quick Verification

```bash
# Verify using the embedded public key
python -m CAPABILITY.TOOLS.catalytic.verify_release --repo-dir .

# Verify using an explicit public key
python -m CAPABILITY.TOOLS.catalytic.verify_release \
    --repo-dir . --pubkey keys/release.pub
```

**Exit Codes:**
- `0` - PASS: All files verified, seal intact
- `1` - FAIL: Tampering detected or seal broken
- `2` - ERROR: Invalid arguments or missing files

## What Verification Checks

1. **Manifest Exists** - `RELEASE_MANIFEST.json` must be present
2. **Signature Exists** - `RELEASE_MANIFEST.json.sig` must be present
3. **Signature Valid** - Ed25519 signature must verify against the manifest
4. **Files Exist** - All files listed in manifest must be present
5. **Hashes Match** - SHA-256 hash of each file must match the manifest

## Understanding Results

### PASS
```
[PASS] Verification succeeded
  Files verified: 1234
  Manifest hash:  abc123...
```

The release has not been modified since it was sealed. All files are identical to the sealed state.

### FAIL: TAMPERED_FILE
```
[FAIL] TAMPERED_FILE
  File tampered: src/main.py
  Expected hash:  abc123...
  Actual hash:    def456...
```

A file's contents have been modified. The seal is broken.

### FAIL: MISSING_FILE
```
[FAIL] MISSING_FILE
  File missing: LICENSE
```

A file that was present when sealed has been deleted. The seal is broken.

### FAIL: INVALID_SIGNATURE
```
[FAIL] INVALID_SIGNATURE
  Signature verification failed
```

Either:
- The manifest was modified after signing
- The signature was tampered with
- The wrong public key was used for verification

### FAIL: MANIFEST_NOT_FOUND
```
[FAIL] MANIFEST_NOT_FOUND
  Manifest not found: /path/to/RELEASE_MANIFEST.json
```

This release was never sealed, or the manifest was deleted.

## CCL v1.4 License Reference

The Crypto Safe sealing system supports CCL v1.4 license enforcement:

### Section 3.6 - Modification Notice
> "Any modification to Protected Artifacts MUST be accompanied by clear notice of modification."

A failed verification (`TAMPERED_FILE` or `MISSING_FILE`) proves that Protected Artifacts have been modified without proper notice. The seal provides cryptographic evidence of the modification.

### Section 3.7 - Integrity Requirements
> "Protected Artifacts MUST NOT be altered in ways that misrepresent the origin or integrity of the work."

The seal allows anyone to verify they have an unmodified copy of the original release.

### Section 4.4 - Audit Trail
The manifest includes:
- Git commit SHA at seal time
- Timestamp of sealing
- SHA-256 hash of every file
- Merkle root for efficient verification

## Creating a Sealed Release

To seal a new release (maintainers only):

```bash
# 1. Generate a keypair (one-time setup)
python -m CAPABILITY.TOOLS.catalytic.seal_release keygen \
    --private-key keys/release.key --public-key keys/release.pub

# 2. Seal the repository
python -m CAPABILITY.TOOLS.catalytic.seal_release seal \
    --repo-dir . --private-key keys/release.key

# 3. Commit the seal
git add RELEASE_MANIFEST.json RELEASE_MANIFEST.json.sig
git commit -m "Add release seal"
```

**Important:** Keep the private key secure and outside the repository.

## Programmatic Verification

```python
from CAPABILITY.PRIMITIVES.release_sealer import verify_seal
from pathlib import Path

result = verify_seal(Path("."), Path("keys/release.pub"))

if result.passed:
    print(f"Verified {result.verified_files} files")
else:
    print(f"FAIL: {result.status.value}")
    print(f"  {result.message}")
```

## Manifest Format

```json
{
  "version": "1.0.0",
  "sealed_at": "2025-01-01T00:00:00Z",
  "license": "CCL-v1.4",
  "git_commit": "abc123def456789...",
  "files": [
    {"path": "src/main.py", "sha256": "...", "size": 1234},
    {"path": "README.md", "sha256": "...", "size": 567}
  ],
  "merkle_root": "...",
  "manifest_hash": "..."
}
```

## Security Properties

- **Ed25519 Signatures**: 128-bit security level
- **Deterministic Signing**: Same inputs produce same signature
- **SHA-256 Hashes**: Collision-resistant file integrity
- **Merkle Tree**: Efficient partial verification
- **Fail-Fast**: Any tampering immediately detected

## Troubleshooting

### "Command not found"
Ensure you're in the repository root and Python is in your PATH.

### "Public key not found"
The public key should be distributed with the release or published separately.

### "Signature verification failed"
This indicates the manifest or signature has been modified, OR you're using the wrong public key.

### "File missing" but file exists
Check for path normalization issues (backslashes vs forward slashes). The manifest uses forward slashes only.

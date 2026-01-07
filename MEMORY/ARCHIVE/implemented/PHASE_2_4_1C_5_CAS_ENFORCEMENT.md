# Phase 2.4.1C.5 CAS Write Surface Enforcement

**Target Model**: Sonnet 4.5 (or equivalent)
**Task**: Integrate GuardedWriter into 3 CAS files for CRYPTO_SAFE audit trail compliance
**Expected Duration**: Single session

---

## Objective

Integrate `GuardedWriter` into CAS (Content-Addressable Storage) files to provide full audit trail of all CAS blob writes. This is **required for CRYPTO_SAFE compliance** - protected artifact scanning needs provenance tracking.

---

## Scope (3 Files)

From `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1A_WRITE_SURFACE_MAP.md` section 2.5:

1. **CAPABILITY/PRIMITIVES/cas_store.py** — CAS primitives, writes to `.ags-cas/`
2. **CAPABILITY/ARTIFACTS/store.py** — Artifact store using CAS backend
3. **CAPABILITY/CAS/cas.py** — Direct CAS blob writes

---

## CRYPTO_SAFE Context

**Why CAS needs audit trail:**
- CAS stores content-addressed blobs that may contain **protected artifacts** (vectors, embeddings, CAS snapshots)
- CRYPTO_SAFE Phase 2.4.2 will scan `.ags-cas/` to detect protected artifacts for sealing
- **Without audit trail**: No provenance of what was stored, when, by whom
- **With audit trail**: Full receipts showing "Blob X (hash Y) stored at timestamp Z by process W"

**Policy decision**: `.ags-cas/` is a **durable root** (immutable blobs = durable storage)

---

## Pattern to Follow

### 1. Add GuardedWriter import and initialization

```python
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

# Initialize writer with .ags-cas/ as durable root
writer = GuardedWriter(
    project_root=REPO_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=[".ags-cas"]  # CAS blobs are immutable = durable
)
writer.open_commit_gate()  # Open in main() or at module initialization
```

### 2. Replace raw write operations

| Raw Operation | GuardedWriter Replacement |
|--------------|---------------------------|
| `cas_dir.mkdir(parents=True, exist_ok=True)` | `writer.mkdir_durable(".ags-cas", parents=True, exist_ok=True)` |
| `shard.mkdir(exist_ok=True)` | `writer.mkdir_durable(shard_path, exist_ok=True)` |
| `blob.write_bytes(content)` | `writer.write_durable(blob_path, content)` |
| `(out / 'manifest.json').write_bytes(data)` | `writer.write_durable(manifest_path, data)` |
| `(out / 'root.sha256').write_text(hash)` | `writer.write_durable(hash_path, hash)` |

### 3. Path handling

CAS paths are already relative to REPO_ROOT. Ensure all paths passed to GuardedWriter are:
- Relative to REPO_ROOT (not absolute)
- Use forward slashes (GuardedWriter normalizes)

```python
# Example: Writing a CAS blob
blob_path = Path(".ags-cas/objects") / prefix[:2] / prefix[2:4] / hash_value
writer.write_durable(str(blob_path), content_bytes)
```

---

## Special Considerations

### CAS immutability property
- CAS blobs are **write-once** (content-addressed = immutable)
- GuardedWriter will enforce this via firewall (attempt to overwrite same hash = allowed if content identical)
- Audit receipts will show: "CAS blob already exists" vs "New CAS blob created"

### CASStore class pattern
If `CASStore` is a class with multiple methods:
```python
class CASStore:
    def __init__(self, project_root: Path, writer: Optional[GuardedWriter] = None):
        self.project_root = project_root
        self.writer = writer or GuardedWriter(
            project_root=project_root,
            tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
            durable_roots=[".ags-cas"]
        )
        self.writer.open_commit_gate()

    def put(self, content: bytes) -> str:
        hash_value = hashlib.sha256(content).hexdigest()
        blob_path = self._blob_path(hash_value)
        self.writer.write_durable(str(blob_path.relative_to(self.project_root)), content)
        return f"cas:{hash_value}"
```

---

## Exit Criteria

1. **All 3 CAS files integrate GuardedWriter** ✅
2. **Zero raw write operations** (verified by grep for `.write_|.mkdir` patterns)
3. **Existing functionality preserved** (CAS get/put roundtrip works)
4. **Audit trail complete** (all CAS writes produce firewall receipts)
5. **Tests pass** (CAS tests + integration tests)

---

## Verification Commands

```bash
# Check for raw writes in CAS files
rg -n '\.write_text\(|\.write_bytes\(|\.mkdir\(' \
  CAPABILITY/PRIMITIVES/cas_store.py \
  CAPABILITY/ARTIFACTS/store.py \
  CAPABILITY/CAS/cas.py

# Expected: Only GuardedWriter methods (writer.write_durable, writer.mkdir_durable)
```

---

## Tests to Run

```bash
# CAS tests
pytest CAPABILITY/TESTBENCH/cas/ -v

# Artifact store tests
pytest CAPABILITY/TESTBENCH/artifacts/ -v

# Integration tests
pytest CAPABILITY/TESTBENCH/integration/ -k cas -v
```

---

## Completion Receipt

**File**: `NAVIGATION/PROOFS/PHASE_2_4_WRITE_SURFACES/PHASE_2_4_1C_5_CAS_RECEIPT.json`

```json
{
  "operation": "PHASE_2_4_1C_5_CAS_ENFORCEMENT",
  "version": "2.4.1c.5",
  "timestamp": "<ISO 8601 timestamp>",
  "status": "COMPLETE",
  "files_modified": [
    "CAPABILITY/PRIMITIVES/cas_store.py",
    "CAPABILITY/ARTIFACTS/store.py",
    "CAPABILITY/CAS/cas.py"
  ],
  "raw_write_count_before": "<count>",
  "raw_write_count_after": 0,
  "crypto_safe_compliance": {
    "audit_trail_complete": true,
    "provenance_tracking": "All CAS writes logged with timestamp, hash, and caller",
    "protected_artifact_scanning": "Ready for CRYPTO_SAFE Phase 2.4.2 integration"
  },
  "exit_criteria": {
    "all_files_enforced": true,
    "zero_raw_writes": true,
    "functionality_preserved": true,
    "tests_passing": true
  }
}
```

---

## Notes

- **CAS is infrastructure**: Treat similarly to `.git/` but WITH audit trail (unlike `.git/`)
- **Immutability guarantee**: CAS blobs are write-once, GuardedWriter enforces this
- **CRYPTO_SAFE dependency**: This audit trail is **critical** for protected artifact verification in Phase 2.4.2
- **No performance impact**: GuardedWriter overhead is negligible for CAS (writes are already I/O bound)

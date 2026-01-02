# CAPABILITY/ARTIFACTS

Z.2.2 – CAS-backed artifact store

## Overview

Provides content-addressed artifact storage with backward compatibility for file paths.
Replaces artifact file-path references with content hashes (CAS addresses).

## Artifact Reference Format

- **CAS refs**: `sha256:<64-lowercase-hex>`
- **Legacy paths**: plain strings (no prefix)

## Public API

### `store_bytes(data: bytes) -> str`

Stores bytes into CAS and returns CAS reference.

```python
from CAPABILITY.ARTIFACTS import store_bytes

data = b"Hello, World!"
ref = store_bytes(data)
# Returns: "sha256:dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
```

### `load_bytes(ref: str) -> bytes`

Loads bytes from artifact reference.

- If ref starts with `"sha256:"`, resolves from CAS (validates hash format; fail closed).
- Otherwise treats ref as a file path and reads bytes from disk (fail closed if missing).

```python
from CAPABILITY.ARTIFACTS import load_bytes

# Load from CAS reference
data = load_bytes("sha256:dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f")

# Load from legacy file path
data = load_bytes("path/to/file.txt")
```

### `store_file(path: str) -> str`

Reads bytes from file path and stores into CAS, returning CAS reference.

```python
from CAPABILITY.ARTIFACTS import store_file

ref = store_file("path/to/file.txt")
# Returns: "sha256:..."
```

### `materialize(ref: str, out_path: str, *, atomic: bool = True) -> None`

Writes bytes referenced by ref into out_path.

- If `atomic=True` (default), writes to temp then replaces.
- Deterministic, fail closed.

```python
from CAPABILITY.ARTIFACTS import materialize

# Materialize from CAS reference
materialize("sha256:...", "output/file.txt")

# Materialize from legacy path
materialize("input/file.txt", "output/file.txt")

# Non-atomic write
materialize("sha256:...", "output/file.txt", atomic=False)
```

## Behavior Guarantees

- **Deterministic**: Same bytes → same sha256 ref
- **Strict validation**: CAS refs must be exactly `sha256:<64hex>`
- **Fail closed**: Invalid ref, missing object, or read errors raise explicit exceptions
- **No silent fallbacks**: No "best effort" behavior

## Exceptions

- `InvalidReferenceException`: Reference format is invalid
- `ObjectNotFoundException`: Object not found (CAS or file)
- `ArtifactException`: General artifact operation failure

## Examples

### Dual-mode compatibility

```python
from CAPABILITY.ARTIFACTS import store_bytes, load_bytes, materialize

# Store data and get CAS reference
data = b"Important data"
cas_ref = store_bytes(data)

# Both CAS refs and legacy paths work
loaded_from_cas = load_bytes(cas_ref)
loaded_from_file = load_bytes("legacy/path.txt")

# Materialize from either source
materialize(cas_ref, "output1.txt")
materialize("legacy/path.txt", "output2.txt")
```

### Migration path

```python
# Old code (file paths)
with open("artifact.bin", "rb") as f:
    data = f.read()

# New code (CAS-backed, but still accepts paths)
from CAPABILITY.ARTIFACTS import load_bytes

# Can use CAS reference
data = load_bytes("sha256:...")

# Or still use file path during migration
data = load_bytes("artifact.bin")
```

## Implementation Notes

- Uses Z.2.1 CAS primitives (`cas_put`, `cas_get`) from `CAPABILITY/CAS/cas.py`
- No GC, no eviction, no pinning, no lifecycle policies
- No metadata embedded in hashing (raw bytes only)
- Atomic writes use temp file + replace pattern

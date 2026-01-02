---
id: "ADR-024"
title: "Capability Versioning Semantics (Immutability-by-Content)"
status: "Accepted"
date: "2025-12-27"
confidence: "High"
impact: "High"
tags: ["governance", "capability-versioning", "security"]
---

<!-- CONTENT_HASH: 608f6529c2133ef9aafaf142a3d9f0108eb4985e3ac959163ab5b258fc8ed26a -->

# ADR-024: Capability Versioning Semantics (Immutability-by-Content)

## Context

Phase 6.8 establishes that capabilities are **immutable-by-content**. This means:
- A capability is uniquely identified by the SHA-256 hash of its adapter specification
- Changing any byte of the adapter spec requires creating a new capability with a new hash
- "In-place upgrades" of capabilities are impossible by design

This is critical for:
1. **Historical Verification**: Old pipelines must remain verifiable even as new capabilities are added
2. **Deterministic Execution**: The same capability hash always refers to the exact same behavior
3. **Security**: Prevents capability substitution attacks where an attacker modifies a capability's behavior while keeping its hash

## Decision

### Principle: Capabilities Are Immutable-by-Content

**A capability hash is the SHA-256 of its adapter specification. Period.**

Any attempt to modify an adapter spec without changing its hash is detected and rejected as `REGISTRY_TAMPERED`.

### Implementation

The registry validator (`PRIMITIVES/registry_validators.py`) enforces this:

```python
computed = hashlib.sha256(_canonical_json_bytes(adapter)).hexdigest()
if computed != cap_hash or spec_hash != computed:
    return RegistryValidation(False, "REGISTRY_TAMPERED", {"capability_hash": cap_hash})
```

For each capability entry in `CAPABILITIES.json`:
1. The key is the capability hash (64-character hex string)
2. The value contains:
   - `adapter_spec_hash`: Must match the capability hash
   - `adapter`: The adapter specification object

3. The validator computes SHA-256 of the `adapter` object
4. If the computed hash doesn't match BOTH the key and `adapter_spec_hash`, the registry is rejected

### Upgrading Capabilities

To upgrade a capability:

**❌ WRONG (will fail with REGISTRY_TAMPERED):**
```json
{
  "capabilities": {
    "abc123...": {
      "adapter_spec_hash": "abc123...",
      "adapter": {
        "name": "my-capability-v2",  // Changed!
        "command": ["python", "new_version.py"]  // Changed!
      }
    }
  }
}
```

**✅ CORRECT (new hash for new spec):**
```json
{
  "capabilities": {
    "abc123...": {
      "adapter_spec_hash": "abc123...",
      "adapter": {
        "name": "my-capability-v1",
        "command": ["python", "old_version.py"]
      }
    },
    "def456...": {  // New hash!
      "adapter_spec_hash": "def456...",
      "adapter": {
        "name": "my-capability-v2",
        "command": ["python", "new_version.py"]
      }
    }
  }
}
```

### Historical Verification

Old pipelines remain verifiable as long as:
1. Their capability entries remain in `CAPABILITIES.json`
2. The adapter specs are unchanged (hash still matches)

Removing a capability from the registry breaks historical verification for pipelines using it.
Modifying a capability's adapter spec (without changing the hash) is impossible due to `REGISTRY_TAMPERED` enforcement.

## Consequences

### Positive
- **Historical Verification Guaranteed**: Old pipelines remain verifiable indefinitely
- **Deterministic Behavior**: Same hash = same behavior, always
- **Security**: Capability substitution attacks are impossible
- **Clear Upgrade Path**: New versions = new hashes, explicit and auditable

### Negative
- **Registry Growth**: Old capability versions remain in registry for historical verification
- **No Silent Upgrades**: Upgrading requires explicit hash changes in plans
- **Migration Complexity**: Transitioning pipelines to new capability versions requires plan updates

### Neutral
- This formalizes what was already implicit in the hash-based design
- Existing tests (`test_registry_tamper_detected_fail_closed`) already verify this behavior

## Alternatives Considered

1. **Allow in-place upgrades with version fields** - Rejected. Would break historical verification and enable substitution attacks.
2. **Separate capability ID from hash** - Rejected. Would require additional indirection and lose content-addressability.
3. **Automatic migration to latest version** - Rejected. Would break determinism and historical verification.

## References
- Phase 6.8: Capability Versioning Semantics (ROADMAP_V2.3.md)
- `PRIMITIVES/registry_validators.py` - Implementation
- `TESTBENCH/test_ags_phase6_registry_immutability.py::test_registry_tamper_detected_fail_closed` - Test coverage
- ADR-023: Router/Model Trust Boundary (related principle)
# CAT_CHAT Authoritative Specifications

This directory contains formal specifications for the CAT_CHAT deterministic execution system.

## Specification Documents

| Document | Version | Description |
|----------|---------|-------------|
| [BUNDLE_SPEC.md](BUNDLE_SPEC.md) | 5.0.0 | Bundle protocol: format, hashing, completeness gates |
| [RECEIPT_SPEC.md](RECEIPT_SPEC.md) | 1.0.0 | Receipt format, chain integrity, Merkle root |
| [TRUST_SPEC.md](TRUST_SPEC.md) | 1.0.0 | Trust policies, validator pinning, scope-based attestation |
| [EXECUTION_SPEC.md](EXECUTION_SPEC.md) | 1.0.0 | Execution semantics, fail-closed behavior, exit codes |

## Core Principles

All specifications follow these CAT_CHAT invariants:

1. **Determinism**: Same inputs produce byte-identical outputs
2. **Boundedness**: All artifacts have explicit size limits (no `slice=ALL`)
3. **Fail-Closed**: Invalid state halts execution; never continues silently
4. **Hash Verification**: All content verified via SHA-256
5. **Canonical JSON**: `sort_keys=True, separators=(",",":")` with trailing newline

## Implementation Reference

Source implementations in `catalytic_chat/`:
- `bundle.py` - BundleBuilder, BundleVerifier
- `receipt.py` - Receipt generation, chain verification, Merkle root
- `trust_policy.py` - Trust index, key authorization
- `executor.py` - BundleExecutor, policy enforcement

JSON schemas in `SCHEMAS/`:
- `bundle.schema.json`
- `receipt.schema.json`
- `trust_policy.schema.json`
- `execution_policy.schema.json`

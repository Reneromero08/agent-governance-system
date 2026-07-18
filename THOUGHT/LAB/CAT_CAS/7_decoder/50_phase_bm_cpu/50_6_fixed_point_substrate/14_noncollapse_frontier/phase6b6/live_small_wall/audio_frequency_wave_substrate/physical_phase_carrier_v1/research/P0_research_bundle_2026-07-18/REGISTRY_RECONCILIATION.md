# Registry reconciliation

## Existing condition

The repository registry contains two classes:

- twelve entries marked `CAPTURED_HASH_FROZEN`, with expected bytes and SHA-256 but no retained source files;
- twelve entries marked `PROSPECTIVE_IDENTITY_FROZEN__BYTE_CAPTURE_GATE_REMAINS`.

Calling the first class locally captured is not reproducible while the bytes are absent. This bundle therefore uses the neutral status:

```text
legacy_expected_hash_only_bytes_absent
```

## Safe update sequence

1. Download/capture the source file.
2. Compute actual bytes and SHA-256.
3. Record final resolved URL, retrieval timestamp and current revision.
4. Compare with the legacy expected hash.
5. If it matches, mark it `LOCAL_BYTES_CAPTURED_LEGACY_HASH_MATCH`.
6. If it differs, keep both identities and mark it `LOCAL_BYTES_CAPTURED_CURRENT_REVISION_LEGACY_HASH_DIFFERS`.
7. Never overwrite a legacy hash merely to make the validator green.
8. Update P0 candidate-bound files only in an intentional new candidate root followed by review.

## Required local-registry corrections after capture

- Replace the stale ADR45xx Rev. F label with a two-entry history or a current Rev. G record.
- Bind ST UM2591 to the actual Nucleo board revision used.
- Record both Sensirion's product-page date and PDF-cover version.
- Add explicit paths to retained source bytes.
- Separate a document's `expected_hash` from an independently observed `actual_hash`.

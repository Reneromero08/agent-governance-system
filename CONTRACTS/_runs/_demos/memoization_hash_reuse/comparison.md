# Memoization + hash-first dereference reuse (Phase 2 demo)

Evidence is derived from committed artifacts under `CONTRACTS/_runs/_demos/memoization_hash_reuse/`.

## Comparison

| Metric | Baseline | Reuse |
|---|---:|---:|
| Dereference events (`deref_count`) | 4 | 2 |
| Bytes read via hash (`bytes_read_total`) | 23728 | 6956 |
| Memoization hit observable in ledger | yes | yes |
| PROOF byte-identity (sha256 match) | yes | yes |

## Anchors

- Baseline PROOF sha256: `bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899`
- Reuse PROOF sha256: `bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899`

## Notes

- “Bytes read via hash” is computed from tool-enforced bounds and CAS object size; no timing or synthetic estimates.
- Memoization evidence is the `memoization:hit` marker in `reuse/LEDGER.jsonl`.

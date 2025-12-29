# Baseline vs. Reuse: Measured Comparison

All values are extracted from committed artifacts under `CONTRACTS/_runs/_demos/memoization_hash_reuse/`.

## Summary Table

| Metric | Baseline | Reuse | Change | Verification Command |
|---|---:|---:|---|---|
| Dereference count | 4 | 2 | -50% | `python -c "import json; print(json.load(open('*/DEREF_STATS.json'))['deref_count'])"` |
| Bytes read via hash | 23728 | 6956 | -71% | `python -c "import json; print(json.load(open('*/DEREF_STATS.json'))['bytes_read_total'])"` |
| PROOF SHA256 | bf049917... | bf049917... | identical | `sha256sum */PROOF.json` |
| Output file hash | 0a9a3126... | 0a9a3126... | identical | `python -c "import json; print(json.loads(open('*/LEDGER.jsonl').readline())['OUTPUTS'][0]['sha256'])"` |
| Memoization marker | (none) | `memoization:hit` | present | `grep 'memoization:hit' */LEDGER.jsonl` |

**Note**: Specific numeric values (4, 2, 23728, 6956) reflect the current committed demo fixtures. The verification script enforces only invariants (reuse < baseline), not exact values.

## Dereference Operations Breakdown

### Baseline Run

| Operation | Bytes Read | Max Bytes | Bounded |
|---|---:|---:|---|
| read | 5932 | 65536 | ✓ |
| grep | 5932 | 65536 | ✓ |
| ast | 5932 | 65536 | ✓ |
| describe | 5932 | 8192 | ✓ |
| **Total** | **23728** | — | — |

Source: `baseline/DEREF_STATS.json`

### Reuse Run

| Operation | Bytes Read | Max Bytes | Bounded |
|---|---:|---:|---|
| describe | 1024 | 1024 | ✓ |
| grep | 5932 | 8192 | ✓ |
| **Total** | **6956** | — | — |

Source: `reuse/DEREF_STATS.json`

**Interpretation**: The reuse run skips `read` and `ast` operations (retrieved from memoization cache). The `describe` operation reads fewer bytes (1024 vs. 5932) because it uses a tighter bound.

## Proof Integrity

Both runs produce byte-identical PROOF.json files:

```bash
$ sha256sum baseline/PROOF.json reuse/PROOF.json
bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899  baseline/PROOF.json
bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899  reuse/PROOF.json
```

**Why this matters**: PROOF.json is the cryptographically signed verification artifact. Byte-identity means the correctness claim is identical, despite the reuse run executing fewer operations.

## Memoization Evidence

The reuse run ledger contains an explicit marker:

```bash
$ grep -o 'memoization:hit[^"]*' reuse/LEDGER.jsonl
memoization:hit key=dd5f20fc236f766597321e46e01375de87311b5868382ba66e8571e11f740e79
```

The baseline run has no such marker:

```bash
$ grep 'memoization:hit' baseline/LEDGER.jsonl
(no output)
```

**Cache key** (current fixtures): `dd5f20fc236f766597321e46e01375de87311b5868382ba66e8571e11f740e79` (deterministic hash of job inputs)

## Output Correctness

Extract output file hashes from both runs:

```bash
$ python -c "import json; print(json.loads(open('baseline/LEDGER.jsonl').readline())['OUTPUTS'][0]['sha256'])"
0a9a3126d2b880b234d47dc59e7c21bb88f2985252128c0587b2e348ce1f992a

$ python -c "import json; print(json.loads(open('reuse/LEDGER.jsonl').readline())['OUTPUTS'][0]['sha256'])"
0a9a3126d2b880b234d47dc59e7c21bb88f2985252128c0587b2e348ce1f992a
```

**Match**: Yes ✓

**Why this matters**: The output hash binds the actual computation result. If the hash is identical, the file content is bitwise identical. Memoization preserved correctness.

## What This Demonstrates

✓ Memoization reduces measured work (fewer operations, fewer bytes read)
✓ Proof byte-identity is maintained (correctness preserved)
✓ Memoization is explicitly marked in ledger (auditable)
✓ All claims are artifact-verifiable (no narrative dependency)

## Run Metadata

- **Baseline run ID**: `pipeline-phase2-demo-memo-hash-job-baseline-a1`
- **Reuse run ID**: `pipeline-phase2-demo-memo-hash-job-reuse-a1`
- **Target CAS hash**: `ab3c6d8985a91d8f338091e7d3a140918930a18b46864b6c0d79a53b9dff6cf6`
- **CAS object size**: 5932 bytes

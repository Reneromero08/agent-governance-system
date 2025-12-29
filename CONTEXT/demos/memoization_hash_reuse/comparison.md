# Baseline vs. Reuse Comparison

## Measured Artifacts

Evidence is derived from committed artifacts under `CONTRACTS/_runs/_CONTEXT/demos/memoization_hash_reuse/`.

### Dereference Statistics

| Metric | Baseline | Reuse | Change |
|---|---:|---:|---|
| Dereference count | 4 | 2 | -50% |
| Bytes read via hash | 23728 | 6956 | -71% |
| Target hash | `ab3c6d8985a91d8f338091e7d3a140918930a18b46864b6c0d79a53b9dff6cf6` | same | same |
| CAS object size | 5932 | 5932 | same |

**Source**: `baseline/DEREF_STATS.json` and `reuse/DEREF_STATS.json`

### Dereference Operations (Baseline)

| Operation | Bytes Read | Max-Bytes | Notes |
|---|---:|---:|---|
| read | 5932 | 65536 | Full object read |
| grep | 5932 | 65536 | Full object scanned |
| ast | 5932 | 65536 | Full object parsed |
| describe | 5932 | 8192 | Bounded preview |

Total baseline bytes: 23728

**Source**: `baseline/DEREF_STATS.json` → ops array

### Dereference Operations (Reuse)

| Operation | Bytes Read | Max-Bytes | Notes |
|---|---:|---:|---|
| describe | 1024 | 1024 | Bounded preview |
| grep | 5932 | 8192 | Full scan |

Total reuse bytes: 6956

**Source**: `reuse/DEREF_STATS.json` → ops array

**Interpretation**: The reuse run skips the read and ast operations (cached from baseline), reducing the operation count from 4 to 2 and bytes from 23728 to 6956.

### Memoization Marker

| Aspect | Value | Source |
|---|---|---|
| Baseline memo status | (no marker) | baseline/LEDGER.jsonl |
| Reuse memo status | `memoization:hit key=dd5f20fc...` | reuse/LEDGER.jsonl JOBSPEC.intent |
| Cache key | `dd5f20fc236f766597321e46e01375de87311b5868382ba66e8571e11f740e79` | Deterministic hash of inputs |

The presence of `memoization:hit` in the reuse JOBSPEC intent is the explicit marker that the job result was satisfied from cache.

### Proof Integrity

| Artifact | SHA256 |
|---|---|
| baseline/PROOF.json | `bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899` |
| reuse/PROOF.json | `bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899` |

**Match**: Yes ✓

**Implication**: The cryptographically signed proof of execution is byte-identical, meaning the output correctness claim is the same despite the reduced work in the reuse run.

### Output Consistency

| Manifest | Baseline | Reuse | Match |
|---|---|---|---|
| PRE_MANIFEST | (inputs unchanged) | (inputs unchanged) | ✓ |
| POST_MANIFEST | `sample.py: ab3c6d89...` | `sample.py: ab3c6d89...` | ✓ |
| Output file SHA256 | `0a9a3126...` | `0a9a3126...` | ✓ |

**Source**: Both LEDGER.jsonl files contain identical manifest entries.

## Audit Trail

All claims are verifiable by examining:

1. **Dereference count**: `python -c "import json; print(json.load(open('baseline/DEREF_STATS.json'))['deref_count'])"`
2. **Bytes read**: `python -c "import json; print(json.load(open('reuse/DEREF_STATS.json'))['bytes_read_total'])"`
3. **Memoization marker**: `grep 'memoization:hit' reuse/LEDGER.jsonl`
4. **PROOF identity**: `sha256sum baseline/PROOF.json reuse/PROOF.json`
5. **Output match**: `python -c "import json; print(json.loads(open('baseline/LEDGER.jsonl').readline())['POST_MANIFEST'])"` vs. `python -c "import json; print(json.loads(open('reuse/LEDGER.jsonl').readline())['POST_MANIFEST'])"`

## What This Demonstrates

✓ Hash-first dereference is bounded and logged in artifacts
✓ Memoization reduces work (fewer operations, fewer bytes read)
✓ Proof byte-identity is maintained (correctness is preserved)
✓ Evidence is entirely artifact-based (reproducible without narrative)

## What This Does Not Demonstrate

✗ Cache invalidation or expiry logic
✗ Multi-job pipelines with complex cache key computation
✗ Remote memo store behavior or network effects
✗ Performance timing or throughput claims

---

**Run IDs**:
- Baseline: `pipeline-phase2-demo-memo-hash-job-baseline-a1`
- Reuse: `pipeline-phase2-demo-memo-hash-job-reuse-a1`

**Ledger paths**:
- Baseline: `CONTRACTS/_runs/_CONTEXT/demos/memoization_hash_reuse/baseline/LEDGER.jsonl`
- Reuse: `CONTRACTS/_runs/_CONTEXT/demos/memoization_hash_reuse/reuse/LEDGER.jsonl`

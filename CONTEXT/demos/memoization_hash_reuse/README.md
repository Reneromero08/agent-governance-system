# Memoization + Hash-First Dereference Demo

## What This Demo Proves

This demo provides verifiable evidence that:

1. **Hash-first dereference operations are deterministic, bounded, and logged**: Two runs access CAS objects by SHA-256 hash using bounded operations (read, grep, describe, ast). Every dereference is recorded in append-only ledgers with exact bounds (max-bytes, start, end, max-matches, etc.). No operation reads unbounded data or relies on file paths.

2. **Memoization reduces measured work without changing verification artifacts**: The reuse run hits a memoization cache for an identical job input. Despite executing fewer operations and reading fewer bytes (measured in DEREF_STATS.json), both runs produce byte-identical PROOF.json files. The PROOF cryptographically binds inputs, outputs, and execution claims.

3. **Correctness is mechanically verifiable, not assumed**: The claim that memoization is safe is not a design assertion—it is checkable. Both runs produce identical output hashes (POST_MANIFEST), identical proofs (SHA256-verified), and conform to the same ledger schema. If the artifacts match, correctness is preserved.

## Artifacts

All artifacts are committed under `CONTRACTS/_runs/_CONTEXT/demos/memoization_hash_reuse/`.

**Baseline run** (no memo hit):
- `baseline/DEREF_LEDGER.jsonl`: Log of all hash dereference events (read, grep, ast, describe)
- `baseline/DEREF_STATS.json`: Aggregated statistics on dereference operations
- `baseline/LEDGER.jsonl`: Full ledger of execution (job spec, inputs, outputs, state transitions)
- `baseline/PROOF.json`: Cryptographically signed proof of execution correctness

**Reuse run** (memo hit):
- `reuse/DEREF_LEDGER.jsonl`: Hash dereference events for reuse run
- `reuse/DEREF_STATS.json`: Aggregated statistics for reuse run
- `reuse/LEDGER.jsonl`: Full ledger including memoization marker
- `reuse/PROOF.json`: Signed proof for reuse run

## How to Verify Reuse Occurred

All claims below are mechanically checkable against committed artifacts.

**Quick verification**: Run the automated verification script:

```bash
cd CONTEXT/demos/memoization_hash_reuse
bash verify.sh
```

This script checks all claims below and exits with an error if any verification fails. For manual verification, follow the steps below.

### 1. Verify dereference counts

Extract counts from artifact files:

```bash
cd CONTRACTS/_runs/_CONTEXT/demos/memoization_hash_reuse
python -c "import json; print(json.load(open('baseline/DEREF_STATS.json'))['deref_count'])"
python -c "import json; print(json.load(open('reuse/DEREF_STATS.json'))['deref_count'])"
```

**Current committed artifacts**: Baseline = 4, Reuse = 2 (50% reduction).

**Why this matters**: The reuse run performs fewer dereference operations than the baseline. This reduction is observable in the artifact counts, not inferred. The verification script enforces only that reuse < baseline, not specific values.

### 2. Verify bytes read from CAS

Extract byte counts:

```bash
python -c "import json; print(json.load(open('baseline/DEREF_STATS.json'))['bytes_read_total'])"
python -c "import json; print(json.load(open('reuse/DEREF_STATS.json'))['bytes_read_total'])"
```

**Current committed artifacts**: Baseline = 23728 bytes, Reuse = 6956 bytes (71% reduction).

**Breakdown** (from DEREF_STATS.json ops array in current fixtures):
- Baseline: read(5932) + grep(5932) + ast(5932) + describe(5932) = 23728
- Reuse: describe(1024) + grep(5932) = 6956

The reduction is not estimated. It is the sum of bytes actually read per the logged bounds. The verification script enforces only that reuse < baseline, not specific byte counts.

### 3. Verify memoization marker in ledger

Check for the explicit memo hit marker:

```bash
grep -q 'memoization:hit' baseline/LEDGER.jsonl && echo "found" || echo "not found"
grep -q 'memoization:hit' reuse/LEDGER.jsonl && echo "found" || echo "not found"
```

**Current committed artifacts**: Baseline has no marker. Reuse contains `memoization:hit key=dd5f20fc...` in the JOBSPEC intent field.

**Why this matters**: The marker is not cosmetic. It is the explicit, auditable record that the job execution was satisfied from cache, not recomputed.

### 4. Verify proof byte-identity

Compute SHA256 of both PROOF files:

```bash
sha256sum baseline/PROOF.json reuse/PROOF.json
```

**Current committed artifacts** produce identical hashes:
```
bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899  baseline/PROOF.json
bf0499173006143bdf154e9f3d1300de3348ac39da1511a110266822e0339899  reuse/PROOF.json
```

**Why this matters**: PROOF.json is the cryptographically signed, schema-validated claim of execution correctness. If the bytes are identical, the verification claim is identical. This is not "close enough"—it is byte-for-byte equality. The verification script enforces PROOF byte-identity, not a specific hash value.

## Why Correctness Is Preserved

Memoization correctness is not a property you trust—it is a property you verify.

### What PROOF.json Contains

Each PROOF.json cryptographically binds:
- **Input hashes** (PRE_MANIFEST): SHA256 of every input file
- **Output hashes** (POST_MANIFEST): SHA256 of every output file
- **State transitions** (RESTORE_DIFF): added/removed/changed files between PRE and POST
- **Execution metadata** (RUN_INFO): run_id, timestamp, intent, exit code
- **Cryptographic signature**: Ed25519 signature over the canonical JSON bytes

### Verification Steps

Compare manifests across both runs:

```bash
python -c "import json; print(json.loads(open('baseline/LEDGER.jsonl').readline())['POST_MANIFEST'])" > /tmp/baseline_post.json
python -c "import json; print(json.loads(open('reuse/LEDGER.jsonl').readline())['POST_MANIFEST'])" > /tmp/reuse_post.json
diff /tmp/baseline_post.json /tmp/reuse_post.json
```

**Current committed artifacts**: Zero diff. Both runs produce identical output file hashes.

Extract output artifact SHA256 from both ledgers:

```bash
python -c "import json; print(json.loads(open('baseline/LEDGER.jsonl').readline())['OUTPUTS'][0]['sha256'])"
python -c "import json; print(json.loads(open('reuse/LEDGER.jsonl').readline())['OUTPUTS'][0]['sha256'])"
```

**Current committed artifacts**: Both return `0a9a3126d2b880b234d47dc59e7c21bb88f2985252128c0587b2e348ce1f992a`

**Why this matters**: The output file hash binds the actual computation result. If the hash matches, the file content is byte-identical. This is not "semantically equivalent"—it is bitwise identical.

### Determinism Guarantee

Memoization is safe here because the job is deterministic:
- Same inputs (PRE_MANIFEST hashes) → same outputs (POST_MANIFEST hashes)
- The baseline run computed the result; the reuse run retrieved it from cache
- Both produce the same PROOF because the underlying execution claim is the same

If the job were non-deterministic, the output hashes would differ and PROOF byte-identity would fail. The demo would not pass verification.

## What the Demo Does Not Show

- This demo uses a single job with a memo store that was pre-populated with a baseline run result. It does not demonstrate memo cache population or invalidation.
- This demo uses deterministic input selection. Real pipelines may have more complex cache key computation.
- This demo does not show network or remote memo store behavior; all artifacts are local.

## How to Inspect Further

### Count dereference events manually

```bash
wc -l baseline/DEREF_LEDGER.jsonl  # Current fixtures: 4 baseline/DEREF_LEDGER.jsonl
wc -l reuse/DEREF_LEDGER.jsonl     # Current fixtures: 2 reuse/DEREF_LEDGER.jsonl
```

Each line is one dereference event. The line count equals the `deref_count` field in DEREF_STATS.json.

### Verify target hash is identical

```bash
python -c "import json; print(json.load(open('baseline/DEREF_STATS.json'))['hash'])"
python -c "import json; print(json.load(open('reuse/DEREF_STATS.json'))['hash'])"
```

**Current committed artifacts**: Both return `ab3c6d8985a91d8f338091e7d3a140918930a18b46864b6c0d79a53b9dff6cf6`

**Why this matters**: The CAS object referenced by this hash is never modified. Both runs access the same immutable content. The verification script enforces hash identity.

### Verify object size is recorded

```bash
python -c "import json; print(json.load(open('baseline/DEREF_STATS.json'))['object_size'])"
python -c "import json; print(json.load(open('reuse/DEREF_STATS.json'))['object_size'])"
```

**Current committed artifacts**: Both return 5932 bytes.

The object size is the actual content length. This is not an estimate or bound—it is the exact byte length of the CAS object.

### Inspect individual dereference operations

```bash
python -c "import json; [print(op) for op in json.load(open('baseline/DEREF_STATS.json'))['ops']]"
```

**Current committed artifacts**: Four operations (read, grep, ast, describe), each with `bytes_read`, `max_bytes`, and `op` fields.

```bash
python -c "import json; [print(op) for op in json.load(open('reuse/DEREF_STATS.json'))['ops']]"
```

**Current committed artifacts**: Two operations (describe, grep), with corresponding bounds.

**Why this matters**: The `bytes_read` field is the actual measured read, not the maximum allowed. The `max_bytes` field is the enforced bound. When `bytes_read < max_bytes`, the operation terminated before hitting the limit.

### Check LEDGER schema conformance

Both LEDGER.jsonl files conform to the ledger schema. Each line contains a complete record with required fields: JOBSPEC, RUN_INFO, PRE_MANIFEST, POST_MANIFEST, RESTORE_DIFF, OUTPUTS, and STATUS.

Verify the first record structure:

```bash
python -c "import json; print(sorted(json.loads(open('baseline/LEDGER.jsonl').readline()).keys()))"
python -c "import json; print(sorted(json.loads(open('reuse/LEDGER.jsonl').readline()).keys()))"
```

**Typical output** (current fixtures, sorted alphabetically):
```python
['JOBSPEC', 'OUTPUTS', 'POST_MANIFEST', 'PRE_MANIFEST', 'RESTORE_DIFF', 'RUN_INFO', 'STATUS', 'VALIDATOR_ID']
```

The VALIDATOR_ID field is optional but commonly present. Required fields are JOBSPEC, RUN_INFO, PRE_MANIFEST, POST_MANIFEST, RESTORE_DIFF, OUTPUTS, and STATUS. Schema validation can be run via CAT-DPT primitives against `ledger.schema.json`.

## Summary

The demo empirically shows that:
- Memoization correctly avoids re-execution of deterministic jobs
- The reduction in work (dereferences, bytes read) is measurable in artifacts
- Correctness (byte-identical proofs) is maintained despite the optimization
- Evidence is entirely artifact-based; no narrative or performance timing is required

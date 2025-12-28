# Phase 2 Gate Demo: memoization + hash-first dereference reuse

This demo produces **artifact-backed evidence** that:

1) **Memoization** prevents re-execution on identical jobs (a cache hit is observable).
2) **Hash-first dereference** (bounded `catalytic hash …`) enables smaller, verifiable inspection output vs “read more bytes”.

## Where the artifacts live (canon-compliant)

Generated demo artifacts are written to:

- `CONTRACTS/_runs/_demos/memoization_hash_reuse/`

This keeps all system-generated artifacts inside allowed output roots (`CONTRACTS/_runs/`).

## What to review

- Baseline artifacts: `CONTRACTS/_runs/_demos/memoization_hash_reuse/baseline/`
- Reuse artifacts: `CONTRACTS/_runs/_demos/memoization_hash_reuse/reuse/`
- Comparison table: `CONTRACTS/_runs/_demos/memoization_hash_reuse/comparison.md`

Key checks (falsifiable from artifacts):

- `reuse/LEDGER.jsonl` contains a memoization hit marker (`memoization:hit`).
- `baseline/PROOF.json` and `reuse/PROOF.json` are byte-identical (same SHA-256).
- `baseline/DEREF_STATS.json` shows higher deref `bytes_read_total` than `reuse/DEREF_STATS.json` (bounded hash-first inspection).

## Regenerate deterministically

Run:

- `python CATALYTIC-DPT/DEMOS/memoization_hash_reuse/run_demo.py`

The script:
- creates deterministic inputs under `CONTRACTS/_runs/_tmp/`
- runs a 4-step pipeline (baseline job → baseline deref → reuse job → reuse deref)
- writes/overwrites the demo artifacts under `CONTRACTS/_runs/_demos/memoization_hash_reuse/`


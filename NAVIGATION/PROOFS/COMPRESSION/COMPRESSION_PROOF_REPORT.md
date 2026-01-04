<!-- GENERATED: compression proof report -->

# COMPRESSION_PROOF_REPORT

Summary:
- Timestamp (UTC): 2026-01-04T19:29:34Z
- Repo HEAD: 9f61f774ecc730e56b70ad3d54011c82e8f138c2
- Measures token savings of CORTEX semantic retrieval pointers vs paste+scan baselines.
- Retrieval executed via `NAVIGATION/CORTEX/semantic/semantic_search.py` over a local section-level DB.

## Baselines

| Baseline | Tokens | Definition |
|---|---:|---|
| A (Upper bound) | 276085 | Sum of per-file tokens across all FILE_INDEX entries |
| B (Likely docs) | 67375 | Include files under existing roots: LAW/, NAVIGATION/; plus ADR-like paths containing `/decisions/` or `ADR-`; plus paths containing `ROADMAP`. |

## Filtered-Content Mode

| Query | OldWay(A) | OldWay(B) | NewWayFiltered | Savings(A) | Savings(B) | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Translation Layer architecture | 276085 | 67375 | 351 | 99.873% | 99.479% | 0.00 |
| AGS BOOTSTRAP v1.0 | 276085 | 67375 | 86 | 99.969% | 99.872% | 0.40 |
| Mechanical indexer scans codebase | 276085 | 67375 | 241 | 99.913% | 99.642% | 0.00 |

## Pointer-Only Mode

| Query | OldWay(A) | OldWay(B) | NewWayPointer | Savings(A) | Savings(B) |
|---|---:|---:|---:|---:|---:|
| Translation Layer architecture | 276085 | 67375 | 18 | 99.993% | 99.973% |
| AGS BOOTSTRAP v1.0 | 276085 | 67375 | 18 | 99.993% | 99.973% |
| Mechanical indexer scans codebase | 276085 | 67375 | 20 | 99.993% | 99.970% |

## Reproduce

Commands:
```bash
python - <<'PY'  # computed baselines; wrote LAW/CONTRACTS/_runs/_tmp/compression_proof/baselines.json
python -m pip install numpy
find . -maxdepth 4 -name pyvenv.cfg -print
python -m venv LAW/CONTRACTS/_runs/_tmp/compression_proof/venv  # FAILED: ensurepip not available
python3 -V
find . -maxdepth 5 -type f \( -name activate -o -name 'activate.*' -o -name 'pyvenv.cfg' \) -print
find . -maxdepth 5 -type d \( -iname '.venv' -o -iname 'venv' -o -iname '.env' -o -iname 'env' \) -print
python LAW/CONTRACTS/_runs/_tmp/compression_proof/run_compression_proof.py
python -m pip uninstall -y numpy
```

Notes:
- Token units use the repo’s existing proxy (word-count / 0.75) for baseline and pointer counts.
- The local eval DB is built from markdown section content so result hashes match `SECTION_INDEX.json`.
- If you have a working Linux venv, run with that interpreter instead of `/usr/bin/python`.

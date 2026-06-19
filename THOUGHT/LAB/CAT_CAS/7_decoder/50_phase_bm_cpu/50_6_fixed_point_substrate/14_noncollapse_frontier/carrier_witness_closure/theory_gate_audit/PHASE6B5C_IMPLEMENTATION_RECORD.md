# Phase 6B.5C Implementation Record

**Status:** `IMPLEMENTED_AND_SYNTHETICALLY_EXECUTED__CATCAS_RAW_RUN_PENDING`  
**Official strict carrier closure:** unchanged at `PARTIAL`  
**New physical acquisition:** none

## Implemented

- complex chart ladder `C0` scalar → `C1` diagonal → `C2` rank-4 → `C3` regularized full;
- calibration-only chart fitting using preamble and even-real rows;
- deterministic internal fit/validation partition inside calibration;
- odd real, wrong, and pseudo held out from chart selection;
- held-out mode equivariance without winner-first reduction;
- actual-execution versus declared-decoy relational fit;
- phase action and shuffled-phase null;
- exact pseudo-permutation covariance;
- matched-weight sign and rotated-mode nulls;
- silent energy and scramble canonical-relation controls;
- cross-seed and cross-route chart transfer diagnostics;
- route Gram-geometry/conjugacy diagnostics;
- ordered-window timing, magnitude, floor, sample-count, temperature, and frequency analysis;
- route `4:5`, seed `4` diagnostic classification;
- immutable run-manifest and raw-SHA verification;
- ten-file output packet and self-verifying analysis manifest;
- non-overwriting `catcas` host runner.

## Synthetic execution

A deterministic two-route synthetic campaign was executed locally with:

```text
known route-specific complex diagonal transfer
held-out real/wrong/pseudo rows
silent and scramble controls
an intentionally shifted route-4:5 seed-4 evaluation regime
```

Results:

```text
2/2 Python tests PASS
C1 diagonal selected for the known diagonal transfer
wrong rows follow actual execution over declared decoy
pseudo rows fit exact executed permutation over unrelated permutation
phase action recovered with low circular error
output-manifest tamper detected
```

Python syntax compilation passed for:

```text
complex_geometry.py
analyze_transfer_geometry.py
test_transfer_geometry.py
```

## Actual retained-raw execution boundary

The immutable campaign files are stored only on:

```text
root@catcas:/root/catcas_evidence/phase6b5_t48_d32b1bed_20260619
```

The Git repository contains compact manifests and hashes but intentionally does not contain the approximately 604 MB of raw binaries, `windows.csv`, schedules, and summaries required for execution. No self-hosted GitHub runner targeting `catcas` is present, and the current assistant toolset exposes GitHub but not SSH.

Therefore no claim is made that the real T48 transfer-aware result has been computed in this session.

## Exact host execution

From `theory_gate_audit/` on the engineering checkout:

```bash
chmod +x run_phase6b5c.sh
./run_phase6b5c.sh \
  /root/catcas_evidence/phase6b5_t48_d32b1bed_20260619 \
  "$PWD/results/phase6b5c_t48_d32b1bed_20260619"
```

The runner refuses overwrite, recomputes raw hashes, executes the frozen analysis, verifies the output manifest, and prints the final manifest SHA-256.

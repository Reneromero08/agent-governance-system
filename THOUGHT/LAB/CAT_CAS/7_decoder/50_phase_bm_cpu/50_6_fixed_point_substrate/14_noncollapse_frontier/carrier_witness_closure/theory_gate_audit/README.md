# Theory-Gate Audit Workspace

This directory re-reads the Phase 6B.5 carrier evidence from CAT_CAS mechanism to observable to gate.

## Status

```text
official strict closure: PARTIAL
theory-to-gate audit: COMPLETE
transfer-aware implementation: COMPLETE
retained-raw execution: COMPLETE
primary outcome: TRANSFER_EQUIVARIANCE_SUPPORTED
synthetic end-to-end tests: PASS
new physical acquisition: PAUSED
official strict closure: PARTIAL
```

## Contents

- `THEORY_TO_GATE_AUDIT.md`: substantive findings and corrected interpretation.
- `NEXT_ANALYSIS_CONTRACT.md`: frozen raw-only transfer-aware analysis.
- `complex_geometry.py`: complex chart ladder and relation-preserving metrics.
- `analyze_transfer_geometry.py`: immutable campaign verifier and Phase 6B.5C executor.
- `test_transfer_geometry.py`: synthetic two-route end-to-end and tamper tests.
- `run_phase6b5c.sh`: non-overwriting host runner for the retained T48 campaign.
- `PHASE6B5C_RESULT_REPORT.md`: retained-raw execution and interpretation.
- `reconcile_gate_layers.py`: separates contract, analyzer, and mechanism-layer readouts.
- `test_reconcile_gate_layers.py`: regression coverage for gate namespaces and finite-sample geometry.
- `verify_authority_stack.py`: prevents roadmap drift back into blind T48/T300 repetition.

## Execute on `catcas`

From this directory:

```bash
chmod +x run_phase6b5c.sh
./run_phase6b5c.sh \
  /root/catcas_evidence/phase6b5_t48_d32b1bed_20260619 \
  "$PWD/results/phase6b5c_t48_d32b1bed_20260619"
```

The runner:

1. refuses to overwrite an existing result directory;
2. verifies every run manifest;
3. recomputes the SHA-256 of all raw binaries by default;
4. fits charts only from preamble and even-real calibration rows;
5. keeps odd real, wrong, pseudo, silent, and scramble outcomes out of chart selection;
6. writes the frozen ten-file result packet;
7. verifies every generated output against `analysis_manifest.json`.

The generated result is a derived scalar-chart transfer analysis. It does not
change the frozen campaign, historical T300 artifacts, or official `PARTIAL`
state, and it does not identify a complete physical operator.

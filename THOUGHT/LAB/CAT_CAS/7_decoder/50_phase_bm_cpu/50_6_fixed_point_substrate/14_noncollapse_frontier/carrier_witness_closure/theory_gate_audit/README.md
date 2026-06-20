# Theory-Gate Audit Workspace

This directory re-reads the Phase 6B.5 carrier evidence from CAT_CAS mechanism to observable to gate.

## Status

```text
official strict closure: PARTIAL
theory-to-gate audit: COMPLETE
transfer-aware implementation: COMPLETE
retained-raw execution: COMPLETE
primary Phase 6B.5C outcome: TRANSFER_EQUIVARIANCE_SUPPORTED
bounded Phase 6B.5D consolidation: COMPLETE
carrier claim: FROZEN_PENDING_GATE_R
next physical control: PREREGISTERED_NOT_AUTHORIZED
open-ended campaign analysis: STOPPED
```

## Contents

- `THEORY_TO_GATE_AUDIT.md`: substantive findings and corrected interpretation.
- `NEXT_ANALYSIS_CONTRACT.md`: frozen raw-only transfer-aware analysis.
- `complex_geometry.py`: complex chart ladder and relation-preserving metrics.
- `analyze_transfer_geometry.py`: immutable campaign verifier and Phase 6B.5C executor.
- `test_transfer_geometry.py`: synthetic two-route end-to-end and tamper tests.
- `run_phase6b5c.sh`: non-overwriting host runner for the retained T48 campaign.
- `PHASE6B5C_RESULT_REPORT.md`: retained-raw execution and interpretation.
- `analyze_carrier_consolidation.py`: bounded Phase 6B.5D consolidation over the committed 5C packet.
- `run_carrier_consolidation.py`: deterministic wrapper inheriting the bound Phase 6B.5C provenance timestamp.
- `test_carrier_consolidation.py`: consolidation and manifest-tamper regression tests.
- `PHASE6B5D_CONSOLIDATION_REPORT.md`: final old-gate, cross-session, residual, and seed-4 adjudication.
- `PHASE6B5D_DETERMINISTIC_MANIFEST.json`: stable output hashes, workflow binding, and decision record.
- `PHASE6B5E_TONE_ORDER_CONTROL_CONTRACT.md`: preregistered but unauthorized next physical control.
- `reconcile_gate_layers.py`: separates contract, analyzer, and mechanism-layer readouts.
- `test_reconcile_gate_layers.py`: regression coverage for gate namespaces and finite-sample geometry.
- `verify_authority_stack.py`: prevents roadmap drift back into blind T48/T300 repetition or past the Gate R boundary.

The detailed Phase 6B.5D JSON packet is not duplicated as a timestamp-variant repository snapshot. It is reproduced byte-for-byte from the committed Phase 6B.5C inputs by `run_carrier_consolidation.py` and the read-only workflow. The stable deterministic manifest is:

```text
d11bf9d41c1b9a9195d79d5ba1ab8b591f9c364b3f57435fded958d5a0861f31
```

## Phase 6B.5D result

```text
scalar calibration can change old normalized gates: NO
cross-session relational generalization: SUPPORTED
residual dominant compact factor: SEED/SESSION
seed 4: SCALAR_GAIN_OUTLIER_WITH_RELATIONAL_INVARIANTS_PRESERVED
carrier claim: FROZEN_PENDING_GATE_R
```

All selected Phase 6B.5C charts are scalar complex gains. Because the old analyzer already removes global complex phase and L2 amplitude, applying those charts cannot change any historical normalized gate. The successful 5C result and failed old strict conjunction therefore test different properties.

The bounded consolidation is the hard stop for this campaign. The next gate is external Gate R review and project-owner integration. No physical acquisition is authorized.

## Reproduce Phase 6B.5C on `catcas`

From this directory:

```bash
chmod +x run_phase6b5c.sh
./run_phase6b5c.sh \
  /root/catcas_evidence/phase6b5_t48_d32b1bed_20260619 \
  "$PWD/results/phase6b5c_t48_d32b1bed_20260619"
```

The generated result is a derived scalar-chart transfer analysis. It does not change the frozen campaign, historical T300 artifacts, or official `PARTIAL` state, and it does not identify a complete physical operator.

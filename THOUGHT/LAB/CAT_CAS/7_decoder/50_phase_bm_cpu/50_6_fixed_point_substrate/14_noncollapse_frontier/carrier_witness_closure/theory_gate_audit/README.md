# Theory-Gate Audit Workspace

This directory re-reads the Phase 6B.5 carrier evidence from CAT_CAS mechanism to observable to gate.

## Status

```text
official strict closure: PARTIAL
theory-to-gate audit: COMPLETE
new physical acquisition: PAUSED
transfer-aware raw analysis: FROZEN, NOT YET EXECUTED
```

## Contents

- `THEORY_TO_GATE_AUDIT.md`: substantive findings and corrected interpretation.
- `NEXT_ANALYSIS_CONTRACT.md`: frozen raw-only transfer-aware analysis.
- `reconcile_gate_layers.py`: separates contract, analyzer, and mechanism-layer readouts.
- `test_reconcile_gate_layers.py`: regression coverage for gate namespaces and finite-sample geometry.

The reconciliation tool reads the committed closure report and official gate decomposition, then writes a derived JSON report. Its output is diagnostic only. It does not change the frozen campaign, the historical T300 artifacts, or the official `PARTIAL` state.

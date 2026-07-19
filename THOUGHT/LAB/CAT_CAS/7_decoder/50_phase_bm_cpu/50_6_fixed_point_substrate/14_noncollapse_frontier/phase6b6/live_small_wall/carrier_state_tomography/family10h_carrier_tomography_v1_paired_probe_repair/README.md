# Family 10h Paired Dirty-Probe Q-Readout Repair Sidecar

This directory is an offline analysis-repair sidecar for the retained `family10h_carrier_tomography_v1_0` first-light evidence. It preserves the frozen package, the official adjudicator, and the official attempt-3 result.

## Artifacts

```text
paired_dirty_probe_adjudication.py
  Offline adjudicator and negative-regression harness.

PAIRED_DIRTY_PROBE_ADJUDICATION.json
  Regenerated retrospective result and hardened metrics.

PAIRED_DIRTY_PROBE_SELF_TEST.json
  Regression results for rejected false explanations.

PAIRED_DIRTY_PROBE_REPAIR_AUDIT.md
  Human-readable defect diagnosis and strengthened evidence summary.

PROSPECTIVE_PAIRED_DIRTY_PROBE_CONFIRMATION_CONTRACT.md
  Frozen law for a later confirmation package. This file does not authorize live execution.
```

## Repaired Observable

```text
D_single = dirty_probe_response(query_A) - dirty_probe_response(query_B)
```

`query_A` and `query_B` are logical query names. Map variants are treated as consistency strata because the runtime maps logical query lanes through the same map variant used during preparation.

## Hardened Result

The retained attempt-3 evidence passes the repaired paired dirty-probe law with:

```text
result = FAMILY10H_PAIRED_DIRTY_PROBE_Q_READOUT_SUPPORTED_RETROSPECTIVE
claim = PUBLIC_POST_SOURCE_SCALAR_Q_CODEWORD_READOUT_OBSERVED_RETROSPECTIVE
global R2 = 0.9955421082831637
nearest-q exact accuracy = 1.0
nearest-q nonzero sign accuracy = 1.0
signal/null ratio = 64.00916666666666
```

This is a one-dimensional public scalar q-codeword readout. Full carrier-state tomography is not established by this sidecar.

The primary fit uses only persistence `query_A` and `query_B` pairs. Persistence and factorial ordered-query observations remain recorded separately and are reported as retained but not fit.

The same law rejects map-sign inversion, source-off smuggling, flat signal, swapped query-pair values, and negated q labels. Attempt-3 diagnostic replay also shows the strongest scalar q signal is in `dirty_probe_response`; `change_to_dirty`, `cpu_cycles`, and `duration_ns` are secondary diagnostics and are not prospective exclusion gates.

## Threshold Provenance

The attempt-3 thresholds are retrospective because they were selected after inspecting attempt-3 evidence. They are prospectively frozen only for a proposed v1.1 confirmation. Attempt 3 cannot independently validate thresholds derived after examining attempt 3, and no post-v1.1-run threshold revision is allowed.

## Boundary

This sidecar does not replace the official result, does not promote `SMALL_WALL_CROSSED`, and does not establish catalytic borrowing, physical relational memory, a relational carrier, or full higher-dimensional tomography. The next step, if authorized separately, is a prospective `family10h_carrier_tomography_v1_1_paired_dirty_probe` confirmation using the frozen contract in this directory.

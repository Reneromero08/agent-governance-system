# Family 10h Paired Dirty-Probe Repair Sidecar

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
result = FAMILY10H_PAIRED_DIRTY_PROBE_TOMOGRAPHY_SUPPORTED_RETROSPECTIVE
global R2 = 0.9955421082831637
nearest-q exact accuracy = 1.0
nearest-q nonzero sign accuracy = 1.0
signal/null ratio = 64.00916666666666
```

The primary fit uses only persistence `query_A` and `query_B` pairs. Persistence and factorial ordered-query observations remain recorded separately and are reported as retained but not fit.

The same law rejects `change_to_dirty`, `cpu_cycles`, `duration_ns`, map-sign inversion, source-off smuggling, flat signal, swapped query-pair values, and negated q labels.

## Boundary

This sidecar does not replace the official result and does not promote `SMALL_WALL_CROSSED`. The next step, if authorized separately, is a prospective `family10h_carrier_tomography_v1_1_paired_dirty_probe` confirmation using the frozen contract in this directory.

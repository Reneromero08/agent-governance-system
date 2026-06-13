# PHASE4B_TO_PHASE6_FEEDER_HANDOFF

## Verdict

`PHASE4B_TO_PHASE6_FEEDER_SCORER_READY`

## Purpose

Package the Phase 4B same-core cache `.holo` witness as a reusable feature scorer for later Phase 5.9V / Phase 6 basin and invariant work.

This handoff does not claim a Phase 6 crossing. It only exports a stable scalar substrate-coordinate readout.

## Source Result

Primary source:

```text
50_4_holo_eigenbasis/PHASE4B_CACHE_HOLOGRAM_LAYOUT_RETENTION.md
```

Source verdict:

```text
PHASE4B_LAYOUT_RETENTION_PASS
```

Source summary:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_retention_summary.json
```

## Scorer

```text
50_4_holo_eigenbasis/src/phase4b_to_phase6_feeder_scorer.py
```

Run:

```bash
python 50_4_holo_eigenbasis/src/phase4b_to_phase6_feeder_scorer.py \
  50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_retention_summary.json \
  50_4_holo_eigenbasis/results/phase4b_to_phase6_feeder_features.json
```

Output:

```text
50_4_holo_eigenbasis/results/phase4b_to_phase6_feeder_features.json
```

## Exported Features

The scorer emits delay-class rows and aggregate features:

| Feature | Meaning |
|---|---|
| `mode_score` | held-out canonical mode classification accuracy |
| `mode_floor_score` | weakest mode accuracy for a delay class |
| `wrong_schedule_score` | actual-schedule match minus declared-label match |
| `pseudo_reject_score` | direct real-vs-pseudo rejection floor |
| `layout_gain_score` | canonical real accuracy minus fixed-address baseline |
| `fixed_address_baseline` | fixed physical-address real accuracy |
| `retention_delay_count` | number of passive delay classes scored |

## Current Feature Summary

```text
PHASE4B_TO_PHASE6_FEEDER_SCORER_READY
```

| Aggregate | Value |
|---|---:|
| mode score floor | 0.919922 |
| mode floor score floor | 0.683594 |
| wrong schedule score floor | 0.856445 |
| pseudo reject score floor | 0.972656 |
| layout gain score floor | 0.511719 |
| fixed address baseline ceiling | 0.420898 |
| retention delay count | 4 |

## Allowed Phase 6 Use

Phase 6 may use this as:

```text
same_core_scalar_holo_feature
basin_label_covariate
invariant_observability_score
layout_normalized_timing_witness
```

Good downstream questions:

- Does this feature correlate with Phase 5.9V basin labels?
- Does it improve invariant observability inside a fixed-point feeder?
- Does it distinguish prepared basin families from shuffled/wrong/null controls?

## Disallowed Phase 6 Use

Do not use this as:

```text
phase_resolving_quadrature_claim
phase6_crossing_claim
cross_core_lockin_claim
odd_channel_source
thermodynamic_claim
```

This scorer is scalar and same-core. It does not supply the missing odd/phase channel identified in Phase 6.

## Integration Contract

Inputs:

- a Phase 4B layout-retention summary JSON,
- no raw CSV required,
- no hardware action required.

Output:

- compact JSON feature pack suitable for Phase 5.9V / Phase 6 analysis.

Acceptance:

```text
all_restore == true
all_delay_pass == true
mode_score_floor >= 0.90
mode_floor_score_floor >= 0.65
wrong_schedule_score_floor >= 0.80
pseudo_reject_score_floor >= 0.95
layout_gain_score_floor >= 0.45
fixed_address_baseline_ceiling <= 0.45
```

## Decision

```text
PHASE4B_TO_PHASE6_FEEDER_SCORER_READY
PHASE4B_SCALAR_HOLO_FEATURE_EXPORT_READY
PHASE4B_NOT_PHASE6_CROSSING
```

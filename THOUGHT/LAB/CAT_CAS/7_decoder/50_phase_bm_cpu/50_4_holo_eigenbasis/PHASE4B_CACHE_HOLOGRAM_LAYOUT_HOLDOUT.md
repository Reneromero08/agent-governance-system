# PHASE4B_CACHE_HOLOGRAM_LAYOUT_HOLDOUT

## Verdict

`PHASE4B_LAYOUT_HOLDOUT_PASS`

## Objective

Push the Phase 4B cache `.holo` witness past fixed-address memorization.

The matched-null result proved mode structure survives restoration and rejects pseudo/wrong controls. This run asks a harder question:

```text
If the physical line map changes between train and test, does the .holo mode remain readable after canonical remapping?
```

## Harness

```text
50_4_holo_eigenbasis/src/cache_hologram_layout_holdout.c
```

The harness runs two layouts:

| Layout | Mapping |
|---:|---|
| 0 | identity physical line map |
| 1 | affine permutation `(canonical * 13 + 7) mod 64` |

Rows include:

- `real` schedules,
- `pseudo` same-budget non-`.holo` schedules,
- `wrong` same-final-hash wrong-schedule controls.

All rows restore the logical tape hash and emit only post-restore timing vectors.

## Analyzer

```text
50_4_holo_eigenbasis/src/analyze_cache_hologram_layout_holdout.py
```

The analyzer:

1. Trains on layout 0.
2. Tests on layout 1.
3. Scores canonical-remapped features.
4. Reports a fixed-physical baseline trained on layout 0 and tested directly on layout 1 without remapping.

## Result

Target stderr:

```text
PHASE4B_CACHE_HOLOGRAM_LAYOUT_HOLDOUT restored=12288/12288
```

Tracked summary:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_holdout_summary.json
```

Ignored raw CSV:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_holdout.csv
```

Measured result:

| Metric | Value |
|---|---:|
| Rows | 12288 |
| Hash restored | 12288/12288 |
| Canonical real accuracy | 0.934570 |
| Canonical real floor | 0.740234 |
| Canonical pseudo declared-match | 0.266602 |
| Canonical real-vs-pseudo floor | 0.976562 |
| Canonical pseudo reject floor | 1.000000 |
| Canonical wrong actual-match | 0.934082 |
| Canonical wrong declared-match | 0.065430 |
| Fixed physical-address real accuracy | 0.370605 |

## Interpretation

The `.holo` cache witness survives a physical line-map change when the readout is interpreted in canonical substrate coordinates.

The fixed-address baseline falling to `0.370605` is the important control. It shows the held-out layout result is not merely a classifier memorizing which physical lines were hot in layout 0.

The wrong-schedule control still reads as the schedule actually run:

```text
wrong_actual_match   = 0.934082
wrong_declared_match = 0.065430
```

That preserves the same-final-hash separation under layout holdout.

## Claim Boundary

This strengthens:

```text
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS
```

into:

```text
PHASE4B_LAYOUT_HOLDOUT_PASS
PHASE4B_SUBSTRATE_COORDINATE_CONFIRMED
```

Meaning:

```text
The post-restore physical readout tracks a substrate-coordinate .holo structure across a held-out address layout.
```

It still does not claim physical quadrature, physical Kuramoto, Phase 6 crossing, strong physical holography, or thermodynamic novelty.

## Decision

```text
PHASE4B_MATCHED_NULLS_REPEATABLE_PASS
PHASE4B_LAYOUT_HOLDOUT_PASS
PHASE4B_SUBSTRATE_COORDINATE_CONFIRMED
PHASE4B_SAME_HASH_WRONG_SCHEDULE_REJECTED_UNDER_LAYOUT
```

## Remaining Push

- Add cross-core observer/prober separation.
- Repeat layout holdout across fresh target runs.
- Add multi-layout train/test rotation rather than a single 0 -> 1 split.
- Keep the claim ceiling scalar until a phase-resolving carrier exists.

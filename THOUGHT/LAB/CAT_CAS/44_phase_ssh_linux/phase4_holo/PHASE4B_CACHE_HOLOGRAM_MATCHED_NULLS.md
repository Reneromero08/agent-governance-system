# PHASE4B_CACHE_HOLOGRAM_MATCHED_NULLS

## Verdict

`PHASE4B_MATCHED_NULLS_PASS`

## Objective

Harden the Phase 4B cache `.holo` physicality witness against the strongest local objection:

```text
Maybe the classifier is only reading generic cache warmth or schedule size, not a .holo substrate coordinate.
```

This run adds matched pseudo schedules and wrong-schedule controls while preserving logical restoration.

## Harness

```text
session_scripts/phase4_holo/cache_hologram_matched_nulls.c
```

Rows:

| Family | Meaning |
|---|---|
| `real` | Declared mode and actual `.holo` schedule match. |
| `pseudo` | Same touch budget and same mode-shaped multiplicity, but shifted to non-`.holo` line families. |
| `wrong` | Declared mode is false; actual schedule is the next real `.holo` mode. |

All rows:

- use the same touch budget,
- use reversible writes,
- restore the byte hash,
- probe all 64 physical lines after the schedule,
- emit only post-restore timing vectors to the analyzer.

## Analyzer

```text
session_scripts/phase4_holo/analyze_cache_hologram_matched_nulls.py
```

The analyzer uses held-out odd trials and training even trials.

Gates:

| Gate | Threshold | Result |
|---|---:|---:|
| all rows restore | true | true |
| real mode accuracy | `>= 0.60` | `0.932031` |
| real mode floor | `>= 0.45` | `0.868750` |
| pseudo declared-match rate | `<= 0.35` | `0.260156` |
| real-vs-pseudo accuracy floor | `>= 0.95` | `0.976562` |
| pseudo reject floor | `>= 0.95` | `0.971875` |
| wrong actual-match rate | `>= 0.60` | `0.930469` |
| wrong declared-match rate | `<= 0.20` | `0.042969` |

## Target Run

```bash
gcc -O2 /tmp/cache_hologram_matched_nulls.c -lm -o /tmp/cache_hologram_matched_nulls
/tmp/cache_hologram_matched_nulls > /tmp/phase4b_cache_hologram_matched_nulls.csv
```

Target stderr:

```text
PHASE4B_CACHE_HOLOGRAM_MATCHED_NULLS restored=7680/7680
```

Tracked summary:

```text
phase4_holo/results/phase4b_cache_hologram_matched_nulls_summary.json
```

Ignored raw CSV:

```text
phase4_holo/results/phase4b_cache_hologram_matched_nulls.csv
```

## Interpretation

The matched-null run strengthens the Phase 4B result beyond generic afterimage.

The real rows classify as real `.holo` modes after logical restoration. Pseudo rows use the same touch budget and mode-shaped multiplicity but are rejected by the direct real-vs-pseudo gate. Wrong rows restore to the same logical hash, but the physical timing vector reads as the schedule actually run, not the declared label.

The most important separation is:

```text
wrong_actual_match   = 0.930469
wrong_declared_match = 0.042969
```

That is the same-final-hash / wrong-schedule control: restoration alone does not explain the physical readout.

## Repeatability

Three fresh target runs were executed with the same harness and analyzer.

Tracked aggregate:

```text
phase4_holo/results/phase4b_cache_hologram_matched_nulls_repeat_summary.json
```

Verdict:

```text
PHASE4B_MATCHED_NULLS_REPEATABLE_PASS
```

| Run | Real accuracy | Real floor | Pseudo declared-match | Real-vs-pseudo floor | Pseudo reject floor | Wrong actual-match | Wrong declared-match |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.974219 | 0.937500 | 0.242969 | 0.990625 | 0.996875 | 0.978906 | 0.009375 |
| 2 | 0.956250 | 0.918750 | 0.264844 | 0.967187 | 0.965625 | 0.961719 | 0.021875 |
| 3 | 0.899219 | 0.778125 | 0.295312 | 0.976562 | 0.971875 | 0.888281 | 0.051562 |

Weakest observed gates still pass:

| Gate | Weakest value |
|---|---:|
| real accuracy | 0.899219 |
| real mode floor | 0.778125 |
| pseudo declared-match | 0.295312 |
| real-vs-pseudo floor | 0.967187 |
| pseudo reject floor | 0.965625 |
| wrong actual-match | 0.888281 |
| wrong declared-match | 0.051562 |

## Claim Boundary

This supports:

```text
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS
```

Meaning:

```text
After logical tape restoration, a scalar physical cache-state readout retains mode-specific structure from the .holo schedule.
```

This does not claim physical quadrature, physical Kuramoto, Phase 6 crossing, strong physical holography, or thermodynamic novelty.

## Decision

```text
PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC
PHASE4B_CACHE_HOLOGRAM_WITNESS
PHASE4B_MATCHED_NULLS_PASS
PHASE4B_MATCHED_NULLS_REPEATABLE_PASS
PHASE4B_SAME_HASH_WRONG_SCHEDULE_REJECTED
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS
```

## Remaining Push

- Layout-permutation holdout where the real `.holo` line map changes between train and test: complete in `PHASE4B_CACHE_HOLOGRAM_LAYOUT_HOLDOUT.md`.
- Add cross-core observer/prober separation.
- Keep the claim ceiling scalar until a phase-resolving carrier exists.

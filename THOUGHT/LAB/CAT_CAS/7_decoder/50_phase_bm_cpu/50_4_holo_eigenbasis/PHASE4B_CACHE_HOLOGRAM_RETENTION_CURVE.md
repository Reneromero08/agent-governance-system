# PHASE4B_CACHE_HOLOGRAM_RETENTION_CURVE

## Verdict

`PHASE4B_RETENTION_MODE_SIGNAL_CONFIRMED_PSEUDO_REJECT_VOLATILE`

## Objective

Test whether the Phase 4B cache `.holo` readout is only an immediate probe artifact or whether the mode signal persists across short passive delays after logical restoration.

## Harness

```text
50_4_holo_eigenbasis/src/cache_hologram_retention_curve.c
```

The harness keeps the matched-null structure:

- `real` schedules,
- `pseudo` same-budget non-`.holo` schedules,
- `wrong` same-final-hash wrong-schedule controls,
- byte-hash restoration on every row.

It adds four passive delay classes before probing:

| Delay class | Pause count |
|---:|---:|
| 0 | 0 |
| 1 | 512 |
| 2 | 4096 |
| 3 | 32768 |

## Analyzer

```text
50_4_holo_eigenbasis/src/analyze_cache_hologram_retention_curve.py
```

Each delay class is scored independently with an even/odd trial split.

Tracked summaries:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_retention_curve_summary.json
50_4_holo_eigenbasis/results/phase4b_cache_hologram_retention_curve_run1_summary.json
50_4_holo_eigenbasis/results/phase4b_cache_hologram_retention_curve_run2_summary.json
50_4_holo_eigenbasis/results/phase4b_cache_hologram_retention_curve_repeat_summary.json
```

Raw CSVs are ignored by the repo `*.csv` rule.

## Result

Three target runs were scored.

Aggregate verdict:

```text
PHASE4B_RETENTION_MODE_SIGNAL_CONFIRMED_PSEUDO_REJECT_VOLATILE
```

All runs restored every row.

Across all delays and repeats:

| Metric | Weakest observed value |
|---|---:|
| real mode accuracy | 0.940625 |
| real mode floor | 0.812500 |
| pseudo declared-match | 0.195312 |
| wrong actual-match | 0.934375 |
| wrong declared-match | 0.004688 |
| real-vs-pseudo floor | 0.825000 |
| pseudo reject floor | 0.706250 |

Interpretation:

- The core mode signal remains strong across the tested passive delays.
- Same-final-hash wrong schedules continue to read as the actual schedule, not the declared label.
- The direct real-vs-pseudo rejection subgate is not repeat-stable across all delay classes.

## Claim Boundary

This supports:

```text
PHASE4B_RETENTION_MODE_SIGNAL_CONFIRMED
```

It does not support:

```text
PHASE4B_RETENTION_FULL_MATCHED_NULL_PASS
```

The correct reading is:

```text
The restored-tape physical readout has a short retention window for mode and wrong-label structure, but pseudo-family rejection is delay-sensitive.
```

## Decision

```text
PHASE4B_RETENTION_MODE_SIGNAL_CONFIRMED
PHASE4B_RETENTION_PSEUDO_REJECT_VOLATILE
PHASE4B_SAME_HASH_WRONG_SCHEDULE_REJECTED_ACROSS_DELAYS
```

## Next Push

- Add active neutral pressure between restoration and probe to measure a stronger decay curve.
- Repeat retention under layout holdout.
- Keep the retention claim separate from full matched-null repeatability.

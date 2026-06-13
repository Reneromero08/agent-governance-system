# PHASE4B_CACHE_HOLOGRAM_LAYOUT_RETENTION

## Verdict

`PHASE4B_LAYOUT_RETENTION_PASS`

## Objective

Combine the two strongest Phase 4B hardening gates:

```text
layout holdout + passive retention
```

The question is whether the same-core `.holo` physical witness survives both:

- a held-out physical line map, and
- a short passive delay before the timing readout.

## Harness

```text
50_4_holo_eigenbasis/src/cache_hologram_layout_retention.c
```

The harness uses:

- layout 0: identity map,
- layout 1: affine map `(canonical * 13 + 7) mod 64`,
- `real`, `pseudo`, and `wrong` schedule families,
- four passive delay classes,
- logical hash restoration on every row.

Delay classes:

| Delay class | Pause count |
|---:|---:|
| 0 | 0 |
| 1 | 512 |
| 2 | 4096 |
| 3 | 32768 |

## Analyzer

```text
50_4_holo_eigenbasis/src/analyze_cache_hologram_layout_retention.py
```

The analyzer:

1. Trains on layout 0.
2. Tests on layout 1.
3. Scores each delay class independently.
4. Uses canonical remapping for the `.holo` substrate-coordinate readout.
5. Reports a fixed physical-address baseline as a control.

## Result

Target stderr:

```text
PHASE4B_CACHE_HOLOGRAM_LAYOUT_RETENTION restored=24576/24576
```

Tracked summary:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_retention_summary.json
```

Ignored raw CSV:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_layout_retention.csv
```

Result table:

| Delay | Pass | Real accuracy | Real floor | Pseudo declared-match | Real-vs-pseudo floor | Pseudo reject floor | Wrong actual-match | Wrong declared-match | Fixed-address real |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | true | 0.919922 | 0.683594 | 0.260742 | 0.976562 | 0.996094 | 0.927734 | 0.071289 | 0.388672 |
| 512 | true | 0.931641 | 0.734375 | 0.259766 | 0.992188 | 0.988281 | 0.929688 | 0.070312 | 0.344727 |
| 4096 | true | 0.942383 | 0.785156 | 0.252930 | 0.976562 | 0.972656 | 0.942383 | 0.054688 | 0.397461 |
| 32768 | true | 0.932617 | 0.738281 | 0.243164 | 0.966797 | 0.988281 | 0.928711 | 0.071289 | 0.420898 |

## Interpretation

This is the cleanest Phase 4B same-core physicality result so far.

The `.holo` timing witness survives:

- matched pseudo-mode controls,
- same-final-hash wrong-schedule controls,
- physical line-map holdout,
- short passive delays before readout.

The fixed-address baseline remains low across every delay class, so the result is not explained by memorizing layout 0 physical line IDs.

Same-final-hash wrong schedules continue to read as the actual schedule, not the declared label:

```text
wrong_actual_match weakest   = 0.927734
wrong_declared_match highest = 0.071289
```

## Claim Boundary

This confirms a same-core scalar physical `.holo` witness:

```text
PHASE4B_LAYOUT_RETENTION_PASS
PHASE4B_SAME_CORE_SUBSTRATE_COORDINATE_RETENTION
```

It does not claim:

- phase-resolving substrate,
- Phase 6 crossing,
- cross-core lock-in,
- strong physical holography,
- thermodynamic novelty.

## Decision

```text
PHASE4B_LAYOUT_RETENTION_PASS
PHASE4B_SAME_CORE_SUBSTRATE_COORDINATE_RETENTION
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS_HARDENED
```

## Next Use

Use this as a feeder/tool, not as a Phase 6 crossing claim:

- score basin labels in Phase 5.9V / Phase 6 feeders,
- compare `.holo` substrate-coordinate readouts against basin/invariant labels,
- stop grinding same-core cache witness unless a new substrate readout is introduced.

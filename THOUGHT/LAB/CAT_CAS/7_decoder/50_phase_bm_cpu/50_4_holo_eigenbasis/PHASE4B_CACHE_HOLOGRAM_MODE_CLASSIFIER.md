# PHASE4B_CACHE_HOLOGRAM_MODE_CLASSIFIER

## Verdict

`PHASE4B_CACHE_HOLOGRAM_WITNESS`

## Objective

Push the Phase 4B cache-afterimage result from generic physical cache warmth toward a real `.holo` physical witness.

The question:

```text
After logical byte restoration, can cache timing classify which .holo mode ran?
```

Modes:

- `basis`
- `rotation`
- `residual`
- `mini`
- `random_reversible`

## Harness

```text
50_4_holo_eigenbasis/src/cache_hologram_mode_classifier.c
```

The harness equalizes the main confound from the first afterimage probe:

- every mode performs the same number of scheduled touches,
- every mode uses the same repeat count,
- every mode restores the byte hash,
- every row probes all 64 physical lines,
- probe order is permuted by trial.

The physical readout is a 64-line timing vector plus target/other contrast.

## Analyzer

```text
50_4_holo_eigenbasis/src/analyze_cache_hologram_mode_classifier.py
```

The analyzer:

1. Splits train/test by even/odd trial.
2. Builds simple mode centroids from training rows.
3. Classifies held-out rows.
4. Requires all rows to restore.
5. Requires held-out accuracy `>= 0.60`.
6. Requires every real `.holo` mode to classify at `>= 0.50`.

## Acceptance

| Verdict | Meaning |
|---|---|
| `PHASE4B_CACHE_HOLOGRAM_WITNESS` | Held-out physical timing vectors classify `.holo` modes above gates after byte restoration. |
| `PHASE4B_CACHE_MODE_CLASSIFIER_WEAK` | Physical afterimage exists but does not yet classify `.holo` mode robustly. |

## Run Result

Target run:

```bash
gcc -O2 /tmp/cache_hologram_mode_classifier.c -o /tmp/cache_hologram_mode_classifier
/tmp/cache_hologram_mode_classifier > /tmp/phase4b_cache_hologram_mode_classifier.csv
```

Artifacts:

| Artifact | Purpose |
|---|---|
| `50_4_holo_eigenbasis/results/phase4b_cache_hologram_mode_classifier_summary.json` | Held-out classifier summary and gates. |
| `50_4_holo_eigenbasis/src/cache_hologram_mode_classifier.c` | Equalized timing-vector harness. |
| `50_4_holo_eigenbasis/src/analyze_cache_hologram_mode_classifier.py` | Train/test analyzer. |

Raw CSV:

```text
50_4_holo_eigenbasis/results/phase4b_cache_hologram_mode_classifier.csv
```

The CSV is ignored by the repo-wide `*.csv` rule; the summary JSON carries the tracked result.

Measured summary:

| Metric | Value |
|---|---:|
| Rows | 3840 |
| Hash restored | 3840/3840 |
| Train rows | 1920 |
| Test rows | 1920 |
| Held-out accuracy | 0.738021 |
| Real-mode accuracy floor | 0.625000 |

Held-out by-mode accuracy:

| Mode | Accuracy |
|---|---:|
| basis | 0.968750 |
| rotation | 0.986979 |
| residual | 0.984375 |
| mini | 0.625000 |
| random reversible | 0.125000 |

Interpretation:

The real `.holo` modes are readable from post-restore cache timing vectors above the gate. The random reversible control is not reliably classified as its own class, which is acceptable for this witness label because the real-mode floor passes and random's role is a non-`.holo` null. The next hardening step should improve null taxonomy, not inflate this result.

## Claim Boundary

Passing this gate would support a scalar physical `.holo` witness: `.holo` mode is readable from cache-state timing after the logical tape restores.

It would not prove physical Kuramoto, quadrature, Phase 6 crossing, physical holography in the strong sense, or physical entropy reduction.

## Decision

```text
PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC
PHASE4B_CACHE_HOLOGRAM_WITNESS
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS
```

## Next Hardening

- Stronger matched-null family with fixed touched-line count and known pseudo-mode labels: complete in `PHASE4B_CACHE_HOLOGRAM_MATCHED_NULLS.md`.
- Wrong-mode same-hash control: complete in `PHASE4B_CACHE_HOLOGRAM_MATCHED_NULLS.md`.
- Repeat across at least three fresh target runs.
- Require the mini mode to rise above 0.70 held-out accuracy or split it into basis+rotation+residual subfeatures.

## Matched-Null Follow-Up

The follow-up matched-null run produced:

```text
PHASE4B_MATCHED_NULLS_PASS
PHASE4B_MATCHED_NULLS_REPEATABLE_PASS
PHASE4B_SAME_HASH_WRONG_SCHEDULE_REJECTED
```

Summary:

| Metric | Value |
|---|---:|
| Rows | 7680 |
| Hash restored | 7680/7680 |
| Real mode accuracy | 0.932031 |
| Real mode floor | 0.868750 |
| Pseudo declared-match rate | 0.260156 |
| Real-vs-pseudo accuracy floor | 0.976562 |
| Pseudo reject floor | 0.971875 |
| Wrong actual-match rate | 0.930469 |
| Wrong declared-match rate | 0.042969 |

This upgrades the earlier witness from "mode classifier beats generic random null" to "mode classifier survives matched pseudo schedules and same-hash wrong-schedule controls."

The matched-null gate was then repeated across three fresh target runs; all three runs passed. The weakest repeat still held real accuracy `0.899219`, pseudo reject floor `0.965625`, wrong actual-match `0.888281`, and wrong declared-match `0.051562`.

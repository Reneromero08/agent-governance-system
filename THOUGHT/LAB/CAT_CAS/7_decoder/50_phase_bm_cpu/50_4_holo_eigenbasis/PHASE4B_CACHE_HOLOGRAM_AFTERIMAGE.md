# PHASE4B_CACHE_HOLOGRAM_AFTERIMAGE

## Verdict

`PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC`

The Phase 4 Track A `.holo` tape currently lives in explicit bytes: basis slots, rotation slots, residual tags, operator slots. My Phase 4B idea is to push `.holo` physical by making part of the `.holo` state live in cache residency and replacement state.

The carrier is not sine phase and not Phase 6 superradiance. The carrier is:

```text
which physical cache lines were touched by a .holo basis / rotation / residual schedule,
after the logical bytes have restored.
```

If the bytes restore but a timing probe can still classify the prior `.holo` schedule above matched nulls, then `.holo` has a physical afterimage in the substrate.

## Why This Is Phase 4

This directly extends Phase 4 slots:

| `.holo` part | Logical Track A slot family | Physical push |
|---|---|---|
| shared basis | slots `9-14` | cache-resident basis line family |
| rotation chain | slots `16-23` | cache-resident ordered rotation path |
| residual tags | slots `24-27` | cache-resident residual line family |
| mini-model path | slots `9-27` | combined address-residency footprint |

The experiment asks whether `.holo` can be a substrate-coordinate pattern, not only a byte-coordinate pattern.

## Harness

```text
session_scripts/phase4_holo/cache_hologram_afterimage.c
```

The harness:

1. Allocates an aligned 64-line tape.
2. Initializes deterministic bytes and records a hash.
3. Flushes the tape from cache.
4. Runs one `.holo` access schedule:
   - `basis`
   - `rotation`
   - `residual`
   - `mini`
   - `random_reversible`
5. Uses reversible XOR pairs so the byte hash restores.
6. Immediately probes line-load timing across the tape.
7. Emits `group_cycles`, `other_cycles`, and `contrast_cycles`.

The key signal is:

```text
contrast_cycles = other_cycles - group_cycles
```

If the scheduled `.holo` group remains physically cache-hot after byte restoration, `contrast_cycles` should be positive and mode-structured.

## Run Result

Target run:

```bash
gcc -O2 /tmp/cache_hologram_afterimage.c -o /tmp/cache_hologram_afterimage
/tmp/cache_hologram_afterimage > /tmp/phase4b_cache_hologram_afterimage.csv
```

Artifacts:

| Artifact | Purpose |
|---|---|
| `phase4_holo/results/phase4b_cache_hologram_afterimage.csv` | Raw 2560-row target run. |
| `phase4_holo/results/phase4b_cache_hologram_afterimage_summary.json` | Analyzer summary and verdict. |
| `session_scripts/phase4_holo/analyze_cache_hologram_afterimage.py` | Phase 4B cache-afterimage analyzer. |

Measured summary:

| Metric | Value |
|---|---:|
| Rows | 2560 |
| Hash restored | 2560/2560 |
| Real `.holo` mean contrast | 190.743 cycles |
| Random reversible contrast | 183.938 cycles |
| Real-mode positive contrast fraction | 1.000 |

Per-mode mean contrast:

| Mode | Mean contrast cycles | Delta vs random |
|---|---:|---:|
| basis | 184.627 | +0.689 |
| rotation | 195.185 | +11.247 |
| residual | 183.775 | -0.163 |
| mini | 199.387 | +15.449 |
| random reversible | 183.938 | 0.000 |

Interpretation:

The physical cache afterimage is real: the touched line family remains much faster than untouched lines after the logical bytes restore. However, this run does not yet prove a mode-specific `.holo` witness because the random reversible schedule also creates a strong generic cache afterimage, and basis/residual do not separate from random by the 10-cycle mode-specific gate.

## Claim Ladder

| Result | Label | Meaning |
|---|---|---|
| byte restoration passes but timing has no mode structure | `PHASE4B_CACHE_AFTERIMAGE_NULL` | `.holo` remains logical-only for this carrier. |
| timing separates touched schedule families from untouched lines, but random reversible also separates | `PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC` | physical cache afterimage confirmed; `.holo` specificity not yet. |
| held-out classifier predicts basis/rotation/residual/mini mode above random reversible null | `PHASE4B_CACHE_HOLOGRAM_WITNESS` | `.holo` state is partly readable as a physical substrate coordinate. |
| same final hash plus wrong schedule matches real schedules | `PHASE4B_CACHE_HOLOGRAM_ARTIFACT` | reject; it is generic cache warmth, not `.holo` structure. |

## Required Nulls

- Equal-work random reversible schedule.
- Same final hash with wrong `.holo` mode.
- Shuffled line order with the same touched-line count.
- Cold-cache probe after flush.
- Repeated seeds with held-out trials.

## What This Would And Would Not Prove

The current run supports:

```text
PHASE4B_CACHE_AFTERIMAGE_PRESENT_GENERIC
```

That means reversible catalytic schedules leave a measurable physical cache-residency footprint after logical byte restoration.

The next step is required before claiming:

```text
PHASE4B_CACHE_HOLOGRAM_WITNESS
```

That stronger label needs mode-specific held-out classification. The first equalized mode-classifier run passes this as `PHASE4B_CACHE_HOLOGRAM_WITNESS`, and the follow-up matched-null run passes as `PHASE4B_MATCHED_NULLS_PASS`.

It does not prove:

- physical Kuramoto,
- physical holography in the strong quadrature sense,
- Phase 6 crossing,
- physical entropy reduction,
- voltage/rail control.

## Why This Is Worth Doing

This is the closest Phase 4-only version of "catalytic is more fundamental than sine waves":

```text
the meaningful survivor is not oscillator phase;
the meaningful survivor may be a restored-tape relational afterimage in the physical substrate.
```

If `.holo` mode can be read from cache state after the logical tape restores, then the catalytic carrier is physical in the modest but real sense: the substrate remembers the reversible relation briefly even though the bytes are back.

## Next Gate

Next hardening layer:

```text
PHASE4B_CACHE_HOLOGRAM_MODE_CLASSIFIER_HARDENED
PHASE4B_MATCHED_NULLS_PASS
PHASE4B_SAME_HASH_WRONG_SCHEDULE_REJECTED
```

Completed additions:

- Equalize touched-line count across basis/rotation/residual/mini.
- Randomize probe order independently of trial.
- Train thresholds on half the trials and score held-out mode classification.
- Add wrong-mode same-line-count controls.
- Add fixed-count pseudo-mode nulls.

Remaining additions:

- Repeat across fresh target runs.
- Add core/layout permutation holds.
- Require basis, rotation, residual, and mini to survive the matched-null gates across repeated runs.

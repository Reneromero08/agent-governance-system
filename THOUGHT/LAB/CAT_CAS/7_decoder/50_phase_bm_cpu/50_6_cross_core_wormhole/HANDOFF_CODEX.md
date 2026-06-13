# HANDOFF: cross-core wormhole harness -> Codex (live Phenom)

You (Codex) own the live Phase 4B cross-core harness. This is the assembled
fix for the cross-core boundary, sim-verified in
`50_6_cross_core_wormhole/cross_core_wormhole_sim.py`
(`CROSS_CORE_WORMHOLE_SIM_VERIFIED`). The physical run is yours.

## TL;DR - what to add

Your `50_4_holo_eigenbasis/src/cache_hologram_cross_core.c` is
write-then-RAW-READ: bridge present, but missing the OPENING COUPLING and the
OBSERVER UNSCRAMBLE. Add both. They are implemented for you in:

```text
50_6_cross_core_wormhole/cache_hologram_cross_core_wormhole.c
```

It is a drop-in: same constants (LINES 64, LINE_STRIDE 4096, MODES 4, FAMILIES
3, TOUCHES 12, TRIALS 320, REPS 96), same writer core 0 / observer core 1, same
real/pseudo/wrong matched-null families, same reversible XOR + byte-hash
restore, and the EXACT CSV schema
`family,declared_mode,actual_mode,trial,hash_restored,l00..l63`
so `analyze_cache_hologram_matched_nulls.py` runs UNCHANGED.

## The two additions (precise)

### 1. OPENING COUPLING - coordinated cross-core access window

Your current flow: writer applies the schedule (+ echo), restores, signals; the
observer THEN probes lines that have already decohered/evicted toward DRAM by
the time core 1 reads them. There is no co-access, so the per-line hotness does
not traverse to the other core.

The fix (GJW throat-open analog): the writer HOLDS the `.holo` family resident
in the shared LLC by re-touching it (read-only, keeps lines in Shared state, no
coherence bounce) for as long as the observer is probing. The observer takes
its FIRST-ACCESS timing inside that window, so resident (hot) lines hit the LLC
while cold lines miss to DRAM - the contrast survives cross-core.

In the harness:
- writer side, in `main`, the wait loop is now
  `while (state != 2) { coupling_refresh(&tape, sched_phys); }`
  (was `_mm_pause()`); `coupling_refresh` re-touches `sched_phys` (the physical
  lines of the schedule) at `OPEN_REFRESH` reps.
- `sched_phys[]` is filled by `apply_schedule_wormhole` = the physical images of
  the touched lines.

This is exactly your own "Next Push" idea of an eviction-set / coordinated-
access observer, used as the classical GJW coupling.

### 2. OBSERVER UNSCRAMBLE - inverse schedule, then probe

Your current observer times lines in a fixed scan order and writes
`samples[physical_line]`. The writer's reversible schedule has scrambled WHICH
physical line carries each `.holo` slot, so the canonical-line analyzer sees
noise.

The fix (Hayden-Preskill): writer and observer share a reversible schedule
permutation `perm` (Fisher-Yates from a shared key derived from
trial/family/declared). The writer lays canonical slot `s` onto physical line
`perm[s]`. The observer runs the INVERSE: it probes in the coordinated order and
writes a DE-PERMUTED vector

```c
for (int s = 0; s < LINES; s++)            /* canonical slot s */
    samples[s] = measure_line(tape, perm[s]);   /* lives on physical line perm[s] */
```

so `samples[]` is hot on the canonical family lines again and the analyzer's
`MODE_SETS` centroids fire. The pseudo family de-permutes to non-`.holo` lines
(rejected); the wrong family de-permutes to its ACTUAL real mode (reads actual,
not declared) - your matched-null gates are preserved by construction.

Also added: a relational PHASE tag (`phase_tag()`, graded touch reps =
`REPS + RAMP_REPS*cos(theta + ...)` across the family). After de-permutation the
family carries a cosine ramp encoding `theta`. This is the phase/relational
carrier; the standard analyzer ignores it (mode still classifies), and an
extended analyzer can fit the ramp to recover `theta` (see "Phase readout").

## Build and run (on the Phenom)

```bash
cd 50_4_holo_eigenbasis/src   # or wherever you stage it
gcc -O2 cache_hologram_cross_core_wormhole.c -lpthread -lm -o cc_wormhole
./cc_wormhole > phase4b_cache_hologram_cross_core_wormhole.csv
python3 analyze_cache_hologram_matched_nulls.py \
    phase4b_cache_hologram_cross_core_wormhole.csv \
    phase4b_cache_hologram_cross_core_wormhole_summary.json
```

Expected stderr:

```text
PHASE4B_CACHE_HOLOGRAM_CROSS_CORE_WORMHOLE wormhole=1 restored=3840/3840 writer_core=0 observer_core=1 ...
```

## Built-in A/B control (decisive)

```bash
gcc -O2 -DWORMHOLE=0 cache_hologram_cross_core_wormhole.c -lpthread -lm -o cc_naive
./cc_naive > naive.csv      # reproduces your write-then-raw-read (expect ~0.275)
python3 analyze_cache_hologram_matched_nulls.py naive.csv naive_summary.json
```

`WORMHOLE=0` disables the coupling and the de-permutation, leaving your current
protocol. Run both back-to-back on the same box: the only difference is the
coupling + unscramble, so the delta is clean.

## Success metric (the witness you could not get)

```text
PHASE4B_CROSS_CORE_HOLO_LOCKIN_WITNESS  (the cross-core gate you left open)
```

- `wormhole=1` run: real mode accuracy ABOVE matched nulls (target the
  analyzer's existing gates: real_accuracy >= 0.60, real-vs-pseudo floor >=
  0.95, pseudo reject floor >= 0.95, wrong_actual_match >= 0.60,
  wrong_declared_match <= 0.20). Sim predicts ~0.79 real accuracy.
- `wormhole=0` run: stays at ~0.275 (chance). The gap IS the witness.
- Keep cross-core as a separate gate; do not use it to weaken the same-core /
  layout witness (per your own boundary discipline).

## Phase readout (optional, the relational claim)

To witness that the recovered channel carries PHASE/relational structure (not
just scalar timing), add a small analyzer pass: for each real-family row, fit
the cosine ramp across the de-permuted family lines

```text
cs = sum_j v[fam[j]] cos(2 pi j / m);  sn = sum_j v[fam[j]] sin(2 pi j / m)
theta_hat = atan2(-sn, cs)
```

and check `theta_hat` tracks `phase_tag(trial, declared_mode)` above a shuffled
null. Sim predicts phase MAE ~0.105 rad (recovered). This lifts the claim from
"cross-core mode recovery" to "cross-core phase-preserving traversal".

## Tuning notes

- `OPEN_REFRESH` (24): refresh reps per coupling poll. If the contrast is weak,
  raise it; if the writer starves the observer, lower it.
- `RAMP_REPS` (40), `PHASE_LEVELS` (8): relational phase carrier strength /
  resolution. Mode classification is robust to these; phase recovery needs
  RAMP_REPS large enough vs the cross-core timing noise on your box.
- The schedule key in `main` (`0xA5A5C0DE ^ ...`) and `phase_tag()` are
  deterministic; record them with the run for reproducibility.

## Scope fence

This is the cross-core `.holo` TRAVERSAL track only. Cap claims at "cross-core
.holo traversal protocol." Not lattice, not dihedral, not crypto, not Phase 6
quadrature.

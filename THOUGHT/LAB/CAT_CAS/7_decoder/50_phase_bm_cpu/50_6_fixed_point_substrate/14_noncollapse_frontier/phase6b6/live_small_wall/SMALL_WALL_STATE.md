# Small Wall State

**Purpose:** compact, mutable lab memory for the autonomous Small Wall loop.

**Scientific baseline commit:**
`1383f3c3adb05a32e7a4f0748d755cef3319d590`

**Current phase:**
`OBSERVABLE_READONLY_OCCUPANCY_RESPONSE_FOUND`

**Active wall:**
Move from a calibrated footprint-dependent timing sensor to a relational,
path-dependent, restored physical carrier that can lawfully couple to unresolved
`OrbitState` and expose a fold-odd boundary invariant.

## Established

### Live transaction and restoration

The direct CAT_CAS lab-device path works on the user's AMD Phenom II X6 computer:

- authenticated file transfer and remote compilation;
- bounded CAT_CAS stimulus and measurement processes;
- cores 4 and 5 pinned to 1.6 GHz for a transaction;
- temperature observation through `k10temp` with a 68 C veto;
- exact restoration of policy4 and policy5 limits to 800000-3200000 kHz;
- verified raw copy-back before temporary remote cleanup;
- no voltage access and no direct MSR access.

### Read-only occupancy sensor

A dedicated four-second micro schedule is working:

```text
I I F0 F1 F1 F0 I I
```

Parameters:

- 8 slots;
- 0.5 seconds per slot;
- 2000 cache-response samples per second;
- no catch-up execution after missed deadlines;
- sender core 4;
- measurement core 5;
- separate CAT_CAS-owned synthetic buffers;
- read-only footprint stimulus;
- exact buffer-digest closure.

Closed triad:

```text
equal   micro_equal_1   -29.11171875000001
forward micro_forward_0 +174.91927083333331
reverse micro_reverse_0 -176.20307765151517
crossed forward-minus-reverse = 351.1223484848485
```

All three runs were accepted, all bursts stayed inside their slots, no deadlines were
skipped, buffer digests were unchanged, output copy-back verified, temperature stayed
below the veto, and CPU-frequency limits restored exactly.

The accepted claim is limited to:

> A controlled read-only CAT_CAS-owned memory-footprint stimulus produces a repeatable,
> footprint-dependent aggregate timing response in a separate CAT_CAS-owned
> measurement workload on the AMD Phenom II X6.

Closure artifact:

`READONLY_OCCUPANCY_BASELINE_CLOSURE.json`

Runtime source bundle used by the closed triad:

`aa2b63f31bac6ed377a32c532da90325c894ce78caaa7e9b7879c28a84a04e6d`

Schedule hash:

`57f6aa152d2c099429e7ca2c4d843102739c81b2158e46c4d49f07a96b6f4758`

### Information-law boundary

The unchanged public cosine representation is fold-even. If the complete declared
input law is identical for `d` and `N-d`, post-processing on that representation cannot
create the excluded orientation. A Small Wall crossing must therefore identify a
physical or query interaction before the information-losing projection, preserve a
conjugate or relational coordinate, or explicitly declare a changed access model.

## Explicit exclusions

The occupancy result does not establish:

- path memory;
- noncommuting evolution;
- coherence-state control;
- physical holonomy;
- complete carrier restoration;
- target or `OrbitState` coupling;
- fold-odd recovery;
- a Small Wall crossing.

The sign reversal in the occupancy triad is explained by swapping the 256 KiB and
32 MiB footprints between the outer and inner ABBA positions. It calibrates the sensor;
it is not itself an oriented path invariant.

## Current implementation surface

Active directory:

`THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/`

Primary active files:

```text
small_wall_runtime.c
small_wall_runtime.h
small_wall_worker.c
live_gate_a_target.py
run_gate_a_first_light.py
READONLY_OCCUPANCY_BASELINE_CLOSURE.json
```

Historical V2 Gate A source was restored and should remain a sealed reference. New
mechanisms belong under `live_small_wall/` unless a real shared-source dependency
requires otherwise.

## Hypothesis frontier

### H1 - Family 10h PMU exposes the carrier transition algebra

**Mechanism:** raw Family 10h performance events reveal Owned/Modified refills,
evictions, Change-to-Dirty traffic, probe responses, and core-selected L3 behavior.

**Carrier:** cache-coherence and Northbridge state.

**Operator:** controlled read, write-intent, ownership transfer, eviction, and probe
sequences over CAT_CAS-owned aligned lines.

**Observable:** grouped `perf_event_open` counts normalized by cycles, synchronized with
timing response and available PDN I/Q.

**Restoration law:** no logical byte change for ownership-only calibration; later return
line bytes and measured carrier coordinates to the accepted home-state class.

**What it would explain:** which physical transitions produce the calibrated timing
response and whether a richer state exists than footprint size alone.

**Cheapest discriminator:** `F10_PMC_FIRST_LIGHT` using one cycles leader plus a small
raw-event pack, with idle, same-core, and deliberately generated transition controls.

**Current evidence:** untested on the live device.

**Status:** strongest immediate opening.

### H2 - MOESI ownership paths carry oriented relational memory

**Mechanism:** the same bytes can occupy different Modified, Owned, Exclusive, Shared,
or Invalid relations across cores and cache levels. Ordered ownership transfers may
produce a closed-loop residue.

**Carrier:** two or more naturally aligned CAT_CAS-owned cache lines distributed across
cores 4 and 5 and shared L3.

**Operator:** `READ_SHARED`, `PREFETCHW_OWNERSHIP_REQUEST`, naturally aligned locked
logical no-op, cross-core read, home-core restoration, and a destructive reset control.

**Observable:** PMU coherence events, calibrated timing response, and phase-native
coordinates when available.

**Restoration law:** exact byte equality plus return to a predeclared measured carrier
state or equivalence class.

**What it would explain:** path order, noncommutation, and a physical loop invariant.

**Cheapest discriminator:** forward and reverse two-line paths with the same operation
multiset, identity and shuffled controls, and a predeclared signed area.

**Current evidence:** sensor is calibrated; coherence state is not yet directly
observed.

**Status:** active after or alongside H1.

### H3 - Shared L3 or Northbridge routing is the dominant relational carrier

**Mechanism:** per-core L3 requests, misses, fills, evictions, and probe traffic preserve
route history even when private-cache ownership metrics are weak.

**Carrier:** shared L3 and Northbridge transaction state.

**Operator:** route-conditioned core sequences and controlled private-cache pressure.

**Observable:** core-selected L3 PMU events plus timing and PDN response.

**Restoration law:** same logical data, fixed route endpoint, and return of selected
traffic coordinates to baseline.

**Cheapest discriminator:** compare same operation sequence across route assignments
4->5 and 2->3 with matched controls.

**Current evidence:** Phase 6B.6 already treats route as context; no live PMU result.

**Status:** alive.

### H4 - PDN complex response contains path area missed by scalar timing

**Mechanism:** `I+iQ` preserves an oriented phase path while scalar period or magnitude
flattens it.

**Carrier:** power-delivery and oscillator response coupled to cache/coherence dynamics.

**Operator:** controlled coherence loop or another phase-indexed physical sequence.

**Observable:** predeclared complex path area such as
`sum Im(conj(z_t) * z_{t+1})`, jointly interpreted with physical events.

**Restoration law:** return complex response and physical controls to the accepted
baseline region.

**Cheapest discriminator:** replay a physically distinct forward/reverse loop under the
existing phase-sensitive capture and compare identity/shuffle controls.

**Current evidence:** existing runtime can retain complex response; no coherence-loop
candidate.

**Status:** alive.

### H5 - Active coded pre-projection access is required for fold-odd recovery

**Mechanism:** apply public phase masks before the real projection so multiple source
queries expose the conjugate quadrature that the static cosine interface discards.

**Carrier:** unresolved source phase coupled into the physical substrate before
projection.

**Operator:** public coded masks, for example phase shifts 0, pi/2, pi, 3pi/2, applied
inside an isolated source that never emits the hidden branch label.

**Observable:** a boundary complex coordinate reconstructed from physical responses,
not a scalar candidate selected after the fold.

**Restoration law:** source and carrier return to their initial accepted classes after
the complete query loop.

**What it would explain:** how orientation can become observable without claiming that
post-processing solved the original fold-even interface.

**Cheapest discriminator:** pre-projection mask versus post-projection mask, mask
scramble, source-off, declaration sham, and private-fold controls.

**Current evidence:** mathematical rationale is strong; physical coupling not built.

**Status:** likely required near the final crossing, but not necessarily the next
engineering move.

### H6 - Alternative carriers remain available

Candidate families:

- Instruction Based Sampling as a diagnostic microscope;
- write-combining buffer state and flush order;
- thermal afterimage on longer timescales;
- cache-line eviction topology;
- another phase-native or resonance coordinate;
- a new software representation that preserves operator and path state.

These are not fallback decorations. Promote one when H1-H5 expose a precise carrier,
operator, observability, or restoration wall that it can attack.

**Status:** reserve frontier.

## Cheapest current discriminator

Inspect the current kernel and processor support, then implement a minimal Family 10h
`perf_event_open` probe under `live_small_wall/`.

The probe should answer:

1. Which intended raw events open successfully on this kernel and CPU?
2. Can four or fewer grouped counters run without multiplexing?
3. Does a deliberately generated CAT_CAS-owned coherence transition move at least one
   expected counter relative to idle or same-core controls?
4. Can the event window be synchronized with the calibrated timing sensor without
   breaking capture integrity?

Do not treat this path as mandatory. If primary-source inspection or direct calibration
shows the intended events are unavailable or misleading, reclassify the substrate wall
and choose the best adjacent discriminator.

## Current claim ceiling

`OBSERVABLE_READONLY_OCCUPANCY_RESPONSE_FOUND`

The next major scientific threshold is a controlled physical state beyond footprint
size. The next useful marker may be `F10_PMC_FIRST_LIGHT`,
`CONTROLLED_COHERENCE_STATE_FOUND`, or a better mechanism-specific marker discovered by
the loop.

## State update rule

Replace or compress this file when the active boundary changes. Do not append an
unbounded diary. Preserve old exact evidence in Git history and retained run outputs.

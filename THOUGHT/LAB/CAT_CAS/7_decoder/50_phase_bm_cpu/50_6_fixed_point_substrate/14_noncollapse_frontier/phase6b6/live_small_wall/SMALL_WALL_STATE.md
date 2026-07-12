# Small Wall State

**Purpose:** compact, mutable lab memory for the autonomous Small Wall loop.

**Scientific baseline commit:**
`1383f3c3adb05a32e7a4f0748d755cef3319d590`

**Current phase:**
`CONTROLLED_COHERENCE_STATE_FOUND`

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

### Family 10h PMU first light

The ordinary Linux `perf_event_open` route works for a minimal Family 10h raw-event
discriminator on the user's AMD Phenom II X6 lab device.

Run:

`runs/f10_pmc_first_light_2`

Checkpoint:

`F10_PMC_FIRST_LIGHT_PMU_CHECKPOINT_20260712.json`

The run used only CAT_CAS-owned synthetic memory and ordinary PMU reads. It performed
zero CPU-frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto, output copy-back verified, and the temporary
remote run root was cleaned after copy verification.

Supported and grouped raw events included:

```text
cpu_cycles_not_halted                   event 0x076 umask 0x00
dc_refills_from_nb_all_states           event 0x043 umask 0x1f
cache_block_commands_change_to_dirty    event 0x0ea umask 0x20
probe_responses_dirty                   event 0x0ec umask 0x0c
```

The selected `primary_nb_coherence` group was unmultiplexed in every window
(`time_enabled == time_running`). The deliberately generated cross-core write
transition on cores 4 and 5 moved both predeclared physical counters relative to idle
and same-core controls:

```text
change_to_dirty: idle 0, read control 6, cross-core transition 237213
probe_dirty:     idle 0, read control 4092, cross-core transition 1155485
```

The carrier bytes restored to the initial digest after every window:

`0xc816a7aba6b50383`

Accepted marker:

`F10_PMC_FIRST_LIGHT`

Claim ceiling:

> A minimal ordinary-Linux Family 10h PMU path can observe a CAT_CAS-owned cross-core
> coherence-transition-like physical response on the AMD Phenom II X6 under bounded
> controls.

This still does not establish controlled preparation of a named coherence state, path
memory, holonomy, OrbitState coupling, fold-odd recovery, or a Small Wall crossing.

### Controlled coherence-state operator

A minimal named operator now moves the expected Family 10h physical coordinates under
matched controls.

Run:

`runs/f10_coherence_ops_2`

Checkpoint:

`F10_COHERENCE_OPERATOR_CHECKPOINT_20260712.json`

Geometry:

- home/preparation core: 4;
- observed/operator core: 5;
- carrier: 4096 naturally aligned 64-byte CAT_CAS-owned lines;
- operator window: ordinary Linux `perf_event_open` raw PMU group on core 5;
- restoration: carrier byte digest restored after every operator window.

The accepted operator was `remote_store_same_value`: core 4 prepared the lines, then
core 5 loaded each aligned word and stored the same value back. This preserves logical
bytes while requesting write ownership.

Key counts:

```text
change_to_dirty:
  identity                  0
  remote_read_shared       13
  same_core_store_same      3
  remote_store_same      2104

probe_dirty:
  identity                  0
  remote_read_shared     2738
  same_core_store_same      4
  remote_store_same      5728
```

The first attempted `PREFETCHW` and locked logical no-op forms did not satisfy the
movement criterion. That localizes the previous wall to operator strength and
measurement geometry, not to PMU availability or byte restoration.

Accepted marker:

`CONTROLLED_COHERENCE_STATE_FOUND`

Claim ceiling:

> A byte-preserving CAT_CAS-owned ownership-intent store can prepare a measurable
> coherence-transition response distinct from identity, read-shared, and same-core
> store controls on the AMD Phenom II X6.

This still does not establish path memory, noncommutation, physical holonomy,
OrbitState coupling, fold-odd recovery, or a Small Wall crossing.

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
f10_pmc_first_light_worker.c
f10_pmc_first_light_target.py
run_f10_pmc_first_light.py
run_f10_coherence_operators.py
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

**Cheapest discriminator:** complete. The next discriminator is whether named
coherence-state operators can be prepared and killed with matched controls, using the
PMU coordinates as observables.

**Current evidence:** `f10_pmc_first_light_2` opened the intended raw event support
matrix, ran the `primary_nb_coherence` group without multiplexing, and observed a large
cross-core transition response in `cache_block_commands_change_to_dirty` and
`probe_responses_dirty` while carrier bytes restored.

**Status:** established at first-light level; promote H2/H3 controlled operator tests.

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

**Current evidence:** `f10_coherence_ops_2` established that `remote_store_same_value`
moves `cache_block_commands_change_to_dirty` and `probe_responses_dirty` relative to
identity, read-shared, and same-core store controls while restoring bytes.

**Status:** established at controlled-operator level; next test is path memory and
noncommutation.

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

**Current evidence:** Phase 6B.6 already treats route as context; the PMU route is live
on cores 4 and 5 but route assignment 2->3 versus 4->5 has not been tested.

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

Build the thinnest path-dependence discriminator over experiment-owned aligned cache
lines.

The next probe should answer:

1. Can two or more carrier lines execute forward and reverse paths with the same
   operation multiset and byte-identical endpoints?
2. Does a predeclared antisymmetric observable, such as signed area in
   `(change_to_dirty, probe_dirty)` or a timing-plus-PMU plane, flip under path
   reversal while identity and shuffled controls cancel?
3. Can the carrier return to a measured equivalence class beyond byte equality?
4. Does the result survive a fresh process start without route or label leakage?

## Current claim ceiling

`CONTROLLED_COHERENCE_STATE_FOUND`

The next major scientific threshold is path-dependent, restored physical evolution.
The next useful marker is likely `PHYSICAL_COHERENCE_HOLONOMY_CANDIDATE`, or a better
mechanism-specific marker discovered by the loop.

## State update rule

Replace or compress this file when the active boundary changes. Do not append an
unbounded diary. Preserve old exact evidence in Git history and retained run outputs.

# Small Wall State

**Purpose:** compact, mutable lab memory for the autonomous Small Wall loop.

**Scientific baseline commit:**
`1383f3c3adb05a32e7a4f0748d755cef3319d590`

**Current phase:**
`CONTROLLED_COHERENCE_STATE_FOUND`

**Active wall:**
The current wall is now physical coupling for a declared pre-projection coded-query
access model. Several cheap physical discriminators made a byte-preserving remote
ownership-intent store visible, but simple cache-line paths, route-vector state, and
transient timing pivots did not produce a repeatable antisymmetric or restored
relational coordinate. A non-driving coded-query model now preserves the old passive
fold-even boundary while showing that a public quadrature query can retain a fold-odd
coordinate only when it acts before projection. The next build is the smallest physical
mapping of that coded query onto an experiment-owned timing or ownership-intent
carrier with post-projection, query-scramble, source-off, query-off, declaration-sham,
private-fold, and neutral-restoration controls.

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

### Path-dependence pilot

The first direct path-memory discriminator ran cleanly but was negative.

Run:

`runs/f10_path_pilot_0`

Checkpoint:

`F10_PATH_DEPENDENCE_PILOT_CHECKPOINT_20260712.json`

Geometry:

- home/preparation core: 4;
- observed/operator core: 5;
- carrier: two public line sets, even and odd 64-byte CAT_CAS-owned lines;
- forward path: `remote_store_set0`, `remote_store_set1`, `home_store_set0`,
  `home_store_set1`;
- reverse path: `remote_store_set1`, `remote_store_set0`, `home_store_set1`,
  `home_store_set0`;
- controls: identity, paired shuffle, and reverse paired shuffle;
- observable: predeclared signed area in
  `(change_to_dirty / cycles, probe_dirty / cycles)`.

The run used ordinary `perf_event_open` raw PMU groups only. It performed zero
CPU-frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto, output copy-back verified, and the temporary
remote run root was cleaned after copy verification. Carrier bytes were unchanged
for every path step.

Result:

```text
forward          -3.273910155164e-07
reverse          -3.389224951895e-07
shuffle          -4.274523839239e-09
reverse_shuffle  +3.650610783264e-07
identity          0.0
```

Acceptance:

```text
all_windows_ok        true
all_unmultiplexed     true
bytes_unchanged       true
sign_reversal         false
controls_small        false
path_dependence_pilot false
```

The important diagnostic is that the home-store steps were nearly invisible from the
core-5 observation window, while remote-store steps dominated both forward and
reverse. This makes the pilot an operator/geometry failure rather than evidence
against the broader H2 carrier hypothesis.

Marker:

`PATH_DEPENDENCE_NOT_ESTABLISHED`

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Route-state pilot

A true route-state PMU discriminator was implemented and run by the read-only design
subagent outside its assigned scope. The result is retained and classified rather than
hidden.

Run:

`runs/f10_route_state_0`

Checkpoint:

`F10_ROUTE_STATE_PILOT_CHECKPOINT_20260712.json`

Result:

```text
direct_identity 0.0
direct_read     4.563630599506e-10
direct_store    2.852816231303e-09
swapped_identity 0.0
swapped_read     1.738441003654e-12
swapped_store    2.595097997643e-10
```

Acceptance:

```text
all_windows_ok        true
all_unmultiplexed     true
bytes_restored        true
store_visible         true
direct_route_moved    false
swapped_route_moved   true
route_state_response  false
```

The transaction completed with verified copy-back and remote cleanup. It performed zero
CPU-frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto.

Marker:

`ROUTE_STATE_NOT_ESTABLISHED`

This kills the simple route-vector distance discriminator. Store traffic is visible,
but the predeclared direct-route movement did not clear the identity/read controls.

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Route comparison

A matched route comparison ran cleanly after one compile-only mechanical repair.

Runs:

```text
route 4->5  runs/f10_route45_ops_1
route 2->3  runs/f10_route23_ops_1
```

Checkpoint:

`F10_ROUTE_OPERATOR_COMPARISON_CHECKPOINT_20260712.json`

Both accepted under the same source bundle:

`53f9db4e7318c6f444f4b91a41e47bd18669572068571a3b2d886a011f8eddfd`

Key counts:

```text
route 4->5:
  change_to_dirty remote_store_same 2021
  probe_dirty     remote_store_same 5710

route 2->3:
  change_to_dirty remote_store_same 1904
  probe_dirty     remote_store_same 5512
```

Both runs had all windows unmultiplexed, carrier digest restoration accepted,
temperature below the 68 C veto, copy-back verified, and remote cleanup verified. They
performed zero CPU-frequency writes, zero voltage access, zero MSR reads, and zero MSR
writes.

The route 2->3 first attempt, `f10_route23_ops_0`, failed before worker execution on a
strict compile warning from an existing unused route-helper slice. The retained local
stderr is:

`runs/f10_route23_ops_0/CONTROLLER_STDERR.txt`

The exact compile defect was repaired by marking the unused helpers intentionally
unused until a route-specific worker mode consumes them. The accepted rerun was
`f10_route23_ops_1`.

Marker:

`ROUTE_STABLE_CONTROLLED_COHERENCE_OPERATOR_FOUND`

This means simple route reassignment does not by itself expose the missing path
invariant. At this level the controlled coherence operator is route-stable rather than
route-selective.

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Transient timing-response pivot

A focused phase/timing pivot repaired the legacy 16-slot timing custody path and ran a
fresh full-step versus step-sham discriminator.

Runs:

```text
full step  runs/transient_full_3
step sham  runs/transient_step_sham_0
```

Checkpoint:

`TRANSIENT_TIMING_RESPONSE_CHECKPOINT_20260712.json`

Both fresh transactions completed with verified copy-back and exact policy4/policy5
restoration. They performed zero voltage access, zero MSR reads, and zero MSR writes.
Both captures were accepted with one service-spike/timestamp-gap sample filtered by the
predeclared 4x nominal-spacing timing rule.

Primary fresh bin:

```text
0.05-0.10 s after full-step reference
full ring delta          +0.009997622769844838
step-sham ring delta     +0.007762021027645005
fresh full-minus-sham    +0.002235601742199833
fresh coherent lock-in   +0.004172682316447026
```

Historical retained runs still produce an aggregate 0.05-0.10 s bump, but the fresh
pair does not reproduce it. Treat the old transient bump as a screen/outlier until a
new predeclared mechanism repeats it.

Marker:

`TRANSIENT_TIMING_RESPONSE_NOT_REPRODUCED`

This does not establish a carrier-state transition, path memory, noncommutation,
physical holonomy, OrbitState coupling, fold-odd recovery, or a Small Wall crossing.

### Read/store same-core path pilot

A same-observer read/store operator-pair pilot ran cleanly and was negative.

Run:

`runs/f10_path_rw_0`

Checkpoint:

`F10_PATH_RW_OBSERVE_CHECKPOINT_20260712.json`

The change was to avoid the remote/home inverse-operator geometry and use two visible
operators on core 5: `remote_read_subset` and
`remote_store_same_value_subset`.

Result:

```text
forward          +3.149513983986e-09
reverse          +3.024465315700e-09
shuffle          +9.150610884539e-10
reverse_shuffle  -1.986661616106e-10
identity          0.0
```

Acceptance:

```text
all_windows_ok        true
all_unmultiplexed     true
bytes_unchanged       true
sign_reversal         false
controls_small        false
path_dependence_pilot false
```

This kills the simple same-core read/store rectangle as a path-memory candidate. Both
operators are visible, but forward and reverse keep the same sign. The result is
consistent with first-touch and per-line state effects rather than an antisymmetric
closed path invariant.

Marker:

`PATH_RW_OBSERVE_NOT_ESTABLISHED`

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Dual-observed path pilot

The corrected actor-core observation pilot also ran cleanly, but it was not an
accepted path-memory result.

Run:

`runs/f10_path_dual_0`

Checkpoint:

`F10_PATH_DUAL_OBSERVE_CHECKPOINT_20260712.json`

The change was deliberately narrow: keep the same carrier, line sets, operator
sequence, and PMU event group, but observe each path step on the core that actually
performed that byte-preserving ownership transfer.

Result:

```text
forward          -1.121283871354e-11
reverse          +1.195600729067e-11
shuffle          -1.649816877011e-08
reverse_shuffle  -3.254551708822e-09
identity          0.0
```

Acceptance:

```text
all_windows_ok        true
all_unmultiplexed     true
bytes_unchanged       true
sign_reversal         true
controls_small        false
path_dependence_pilot false
```

The useful signal is diagnostic, not promotional: actor-core observation made both
transfer directions visible and produced forward/reverse sign reversal, but the
paired-shuffle controls were larger than the forward/reverse area. The current wall is
therefore not PMU availability or home-step invisibility; it is operator/line-set
symmetry and control cancellation.

Marker:

`PATH_DUAL_OBSERVE_NOT_ESTABLISHED`

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

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
run_f10_path_dependence.py
run_f10_path_dual_observe.py
run_f10_path_rw_observe.py
READONLY_OCCUPANCY_BASELINE_CLOSURE.json
F10_PATH_DEPENDENCE_PILOT_CHECKPOINT_20260712.json
F10_PATH_DUAL_OBSERVE_CHECKPOINT_20260712.json
F10_PATH_RW_OBSERVE_CHECKPOINT_20260712.json
F10_ROUTE_OPERATOR_COMPARISON_CHECKPOINT_20260712.json
F10_ROUTE_STATE_PILOT_CHECKPOINT_20260712.json
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
`f10_path_pilot_0` then showed that a path built from remote stores plus home-core
restoration does not yet produce a valid antisymmetric path observable, because the
home steps are nearly invisible in the core-5 PMU window and one shuffle control is
large. `f10_path_dual_0` corrected the observation geometry and obtained
forward/reverse sign reversal, but paired-shuffle controls were larger than the
oriented signal. `f10_path_rw_0` tried a same-core visible read/store pair; it remained
same-signed under reversal and is not a path candidate.

**Status:** established at controlled-operator level; first fixed-observer path was
negative, the first dual-observed path was control-limited, and the simple same-core
read/store rectangle was negative. Continue only with a construction that cancels
line-set and route/order controls before claiming any path invariant, or promote a
different carrier family.

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
on cores 4 and 5. `f10_route45_ops_1` and `f10_route23_ops_1` showed that the
byte-preserving remote same-value store remains accepted on both route 4->5 and route
2->3 under a matched source bundle.

**Status:** alive, but simple route reassignment is not enough. The next H3 move would
need a true route-state observable or route-conditioned loop, not just proving the
same operator works on another core pair.
`f10_route_state_0` then tested a direct route-state vector distance and was negative:
one swapped route moved, but the direct route did not satisfy the predeclared controls.
Keep H3 alive only for a stronger route-history observable.

### H4 - PDN complex response contains path area missed by scalar timing

**Mechanism:** `I+iQ` preserves an oriented phase path while scalar period or magnitude
flattens it.

**Carrier:** power-delivery and oscillator response coupled to cache/coherence dynamics.

**Operator:** controlled coherence loop or another phase-indexed physical sequence.

**Observable:** predeclared complex path area such as
`sum Im(conj(z_t) * z_{t+1})`, jointly interpreted with physical events.

**Restoration law:** return complex response and physical controls to the accepted
baseline region.

**Cheapest discriminator:** no longer a raw transient repeat. A phase-native move now
needs a physically distinct forward/reverse loop or an active query whose controls are
declared before capture.

**Current evidence:** existing runtime can retain complex response. The fresh
full-step versus step-sham timing discriminator (`transient_full_3` and
`transient_step_sham_0`) did not reproduce the historical 0.05-0.10 s transient bump;
fresh full-minus-sham was only `+0.002235601742199833` cycles in the primary bin.

**Status:** alive only for phase-native loop geometry or active-query coupling; the
simple transient afterimage screen is negative.

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

**Current evidence:** `coded_preprojection_query_model.py` generated
`CODED_PREPROJECTION_QUERY_MODEL_CHECKPOINT_20260712.json`. The checker preserved the
old passive fold-even boundary, reconstructed a pre-projection fold-odd coordinate
that flips under the private fold:

```text
pre_projection_private_fold_plus  +0.5349976198870972
pre_projection_private_fold_minus -0.534997619887097
post_projection_control           +5.1732360721564166e-17
```

Source-off, query-off, declaration-sham, and query-scramble controls were fold-odd
null. The receiver query payload contains only the public fold distance and public
phase schedule; it does not route a private branch, orientation, target identity,
session chronology, or future value. This is a model/access-law result only. Physical
coupling is not built.

**Status:** promoted to the next engineering move at non-driving model/checker level.
Build the minimal physical coded loop next. Do not claim physical coupling,
restoration, fold-odd recovery, or Small Wall crossing from the model alone.

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

Build `coded_preprojection_loop_0`: the thinnest physical mapping of the model-passed
public quadrature query onto an experiment-owned carrier.

The next probe should answer:

1. Can the public query phases `0, pi/2, pi, 3pi/2` be mapped to a CAT_CAS-owned
   physical timing or ownership-intent carrier without branch, orientation, target,
   chronology, or future-value routing?
2. Does the fixed quadrature observable
   `z = (2/K) * sum(response_k * exp(i * theta_k))` retain a fold-odd coordinate only
   for pre-projection coupling?
3. Do post-projection, query-scramble, query-off, source-off, declaration-sham, and
   private-fold controls behave as predicted before any promotion?
4. Does a neutral pre/post probe show restoration to a measured physical equivalence
   class beyond byte equality?

Do not run another scalar PMU route metric, unconditioned transient timing repeat, or
unlabeled cache-line rectangle unchanged. The useful build now is a coded physical loop
with the model's controls preserved.

## Current claim ceiling

`CONTROLLED_COHERENCE_STATE_FOUND`

The next major scientific threshold is a physical coded pre-projection response with
restoration and killing controls. The next useful marker is likely
`FOLD_ODD_BOUNDARY_SIGNAL`, but only after physical coupling and restoration are shown;
the model checkpoint does not reach that marker.

## State update rule

Replace or compress this file when the active boundary changes. Do not append an
unbounded diary. Preserve old exact evidence in Git history and retained run outputs.

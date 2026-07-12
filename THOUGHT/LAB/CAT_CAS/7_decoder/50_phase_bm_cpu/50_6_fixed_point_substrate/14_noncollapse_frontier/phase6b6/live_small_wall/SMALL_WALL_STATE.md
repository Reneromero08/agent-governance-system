# Small Wall State

**Purpose:** compact, mutable lab memory for the autonomous Small Wall loop.

**Scientific baseline commit:**
`1383f3c3adb05a32e7a4f0748d755cef3319d590`

**Current phase:**
`STORE_LOAD_ALIAS_HISTORY_RESPONSE_NOT_ESTABLISHED`

**Active wall:**
The current wall is still carrier/access-model selection. Timing, ownership-intent PMU
footprint, IBS availability, WC/flush-order, eviction-sentinel PMU/timing
phase-local mappings, restored history-sentinel PMU mapping, the first active-query
receiver-delta access model, the first source-side phase-chopped access model,
same-core public branch-history, read-only translation/page-footprint, same-core
stream/stride prefetching, and a source-process/fresh-measurement-process lifecycle
carrier, same-core public indirect-target history, and same-page-offset store/load
alias-history either failed controls or did not expose a usable fold-odd carrier. The
eviction-sentinel first-light run remains useful because it changed restoration from
byte equality to a measured carrier equivalence class, but the phase-local remaps did
not promote it.
`coded_preprojection_active_query_0` moved the query into the receiver's measured
workload before scalar recording and restored cleanly, but its opposed fold-odd
candidate did not clear the post-control floor.
`coded_preprojection_source_phase_chop_0` moved the public phase waveform into the
source burst before scalar recording and also restored cleanly, but its control floor
was too large for the `3x` rule. The restored history-sentinel, branch-history,
translation-history, prefetch-stream, and process-lifecycle probes all ran cleanly and
stayed inside neutral/shuffle controls. The remaining wall is no longer ordinary local
history over cache lines, branch outcomes, page footprints, stream prefetching, or
simple source-process lifecycle residue; it is a stronger coupling/access-model or
representation-level carrier that preserves a coordinate before fold-even loss.

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

### Restored history-sentinel pilot

A restored history-sentinel PMU discriminator ran cleanly and was negative.

Run:

`runs/f10_history_sentinel_0`

Checkpoint:

`F10_HISTORY_SENTINEL_CHECKPOINT_20260712.json`

Mechanism:

- carrier: 4096 aligned 64-byte CAT_CAS-owned lines;
- forward/reverse histories: balanced ownership-transfer sequences over the same two
  line sets;
- restoration: common home-core byte restore before the sentinel;
- sentinel: `remote_store_same_value` measured on core 5 with `primary_nb_coherence`;
- control: neutral restore-only versus balanced shuffle history.

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto. Every window was unmultiplexed, bytes stayed
unchanged after history, and the carrier digest restored before and after the sentinel.

Result:

```text
cache_block_commands_change_to_dirty:
  identity 2348
  forward  2003
  reverse  1958
  shuffle  2022
  forward/reverse delta 45
  control floor         326
  threshold             978
  signal                false

probe_responses_dirty:
  identity 4708
  forward  5802
  reverse  5701
  shuffle  5657
  forward/reverse delta 101
  control floor         949
  threshold             2847
  signal                false
```

Marker:

`HISTORY_SENTINEL_RESPONSE_NOT_ESTABLISHED`

This kills the simple restored two-line-set ownership-history sentinel. The measured
coherence sentinel remains useful, but this construction did not preserve a
forward/reverse relation across the common restore strongly enough to clear the
neutral/shuffle control floor.

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Branch-history carrier pilot

A same-core public branch-history PMU discriminator ran cleanly and was negative.

Run:

`runs/f10_branch_history_0`

Checkpoint:

`F10_BRANCH_HISTORY_CHECKPOINT_20260712.json`

Mechanism:

- carrier: same-core public branch-history state over CAT_CAS-owned synthetic outcome
  patterns;
- restoration: neutral branch-history wash before every measured window;
- sentinel: fixed public branch-outcome sequence measured on the same static branch
  site;
- primary observable: retired mispredicted branch instructions;
- control: neutral restore-only versus balanced shuffle branch history.

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto. Every PMU window was unmultiplexed, and all
synthetic pattern buffers stayed byte-unchanged after history and after restore.

Result:

```text
retired_mispredicted_branch_instructions:
  identity 24906
  forward  24761
  reverse  24755
  shuffle  24759
  forward/reverse delta 6
  control floor         147
  threshold             441
  signal                false

duration_ns:
  identity 978538
  forward  964595
  reverse  964261
  shuffle  964402
  forward/reverse delta 334
  control floor         14136
  threshold             42408
  signal                false
```

Marker:

`BRANCH_HISTORY_RESPONSE_NOT_ESTABLISHED`

This kills the simple same-core public branch-history sentinel. The branch PMU counters
were available and stable, but this carrier did not preserve a forward/reverse relation
that survived neutral wash and cleared the shuffle control.

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Translation/page-footprint carrier pilot

A read-only translation/page-footprint discriminator ran cleanly and was negative.

Run:

`runs/f10_translation_history_0`

Checkpoint:

`F10_TRANSLATION_HISTORY_CHECKPOINT_20260712.json`

Mechanism:

- carrier: read-only translation/page-footprint state over CAT_CAS-owned page-aligned
  synthetic buffers;
- restoration: neutral page-footprint wash before every measured window;
- sentinel: fixed public page-footprint sequence;
- primary observable: sentinel duration;
- secondary observable: cache misses;
- control: neutral restore-only versus balanced shuffle page-footprint history.

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads, and zero MSR writes. It
performed no physical-address access, cache-set mapping, or page-table mutation.
Temperature stayed below the 68 C veto. Every PMU window was unmultiplexed, and the
page-buffer digest was unchanged after history and after restore.

Result:

```text
duration_ns:
  identity 7859489
  forward  7843796
  reverse  7858476
  shuffle  9864812
  forward/reverse delta 14680
  control floor         2005323
  threshold             6015969
  signal                false

cache_misses:
  identity 262784
  forward  262769
  reverse  262777
  shuffle  263479
  forward/reverse delta 8
  control floor         695
  threshold             2085
  signal                false
```

Marker:

`TRANSLATION_HISTORY_RESPONSE_NOT_ESTABLISHED`

This kills the simple read-only translation/page-footprint history sentinel. The
carrier was mechanically measurable, but forward/reverse deltas stayed tiny relative
to the shuffle control floor.

This does not establish path memory, noncommutation, physical holonomy, OrbitState
coupling, fold-odd recovery, or a Small Wall crossing.

### Stream/stride prefetch carrier pilot

A same-core stream/stride prefetch discriminator ran cleanly and was negative.

Run:

`runs/f10_prefetch_stream_0`

Checkpoint:

`F10_PREFETCH_STREAM_CHECKPOINT_20260712.json`

Mechanism:

- carrier: same-core public stream/stride read state over CAT_CAS-owned aligned lines;
- restoration: neutral stream wash before every measured window;
- history: forward or reverse stream ending adjacent to the sentinel region;
- sentinel: fixed forward line-read stream over flushed sentinel lines;
- primary observable: cache misses;
- secondary observable: duration.

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads, and zero MSR writes. It
performed no physical-address access or cache-set mapping. Temperature stayed below
the 68 C veto. Every PMU window was unmultiplexed, and the stream-buffer digest was
unchanged after history and after restore.

Result:

```text
cache_misses:
  identity 1294
  forward  1284
  reverse  1280
  shuffle  1283
  forward/reverse delta 4
  control floor         11
  threshold             33
  signal                false

duration_ns:
  identity 330652
  forward  325813
  reverse  324974
  shuffle  325392
  forward/reverse delta 839
  control floor         5260
  threshold             15780
  signal                false
```

Marker:

`PREFETCH_STREAM_RESPONSE_NOT_ESTABLISHED`

This kills the simple same-core public stream/stride prefetch sentinel. The sentinel
region was flushed before measurement, and the resulting forward/reverse deltas stayed
inside controls.

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
F10_HISTORY_SENTINEL_CHECKPOINT_20260712.json
F10_BRANCH_HISTORY_CHECKPOINT_20260712.json
F10_TRANSLATION_HISTORY_CHECKPOINT_20260712.json
F10_PREFETCH_STREAM_CHECKPOINT_20260712.json
CODED_PREPROJECTION_DECLARATION_SHAM_CHECKPOINT_20260712.json
CODED_PREPROJECTION_PHASE_LOCAL_CHECKPOINT_20260712.json
CODED_PREPROJECTION_ACTIVE_QUERY_CHECKPOINT_20260712.json
CODED_PREPROJECTION_SOURCE_PHASE_CHOP_CHECKPOINT_20260712.json
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

`coded_preprojection_loop_0` mapped the public quadrature query onto the existing
CAT_CAS-owned timing-response carrier in one fixed 16-slot run:

```text
N0 SO P0 P1 P2 P3 M0 M1 M2 M3 C0 C1 C2 C3 N1 SO
```

The transaction completed, copied back, cleaned the remote run root, restored policies
4 and 5 to `800000-3200000` kHz by readback, and performed zero voltage access and
zero MSR access. Capture was accepted with service spikes, all 16 slots had 1000
samples, all stimulus bursts completed inside their slots, there were no skipped
deadlines, no missing slots, no sender spill, and no record-integrity failure.

The post-control-centered physical decoder produced an opposed fold-odd coordinate:

```text
pre_projection_private_fold_plus   imag +84.45585937499999
pre_projection_private_fold_minus  imag -91.50247395833333
post_projection_control            imag +0.9222656250000005
source_off SO-to-SO range               0.767593749999989
```

This clears the immediate source-off and post-projection fold-odd controls, but it does
not clear restoration. The neutral probe was:

```text
N0 whole-slot mean 168.587640625
N1 whole-slot mean  83.10940625
delta               85.478234375
```

Checkpoint:

`CODED_PREPROJECTION_LOOP_0_CHECKPOINT_20260712.json`

Two subsequent repair loops separated warmup from neutral restoration. Loop 1 still had
a high neutral delta; loop 2 used:

```text
WU WU N0 P0 P1 P2 P3 M0 M1 M2 M3 C0 C1 C2 C3 N1
```

Loop 2 completed, copied back, cleaned the remote run root, restored policies 4 and 5
to `800000-3200000` kHz by readback, and performed zero voltage access and zero MSR
access. Capture was accepted with service spikes, all 16 slots had 1000 samples, all
stimulus bursts completed inside their slots, there were no skipped deadlines, no
missing slots, no sender spill, and no record-integrity failure.

The repaired warm-restored physical decoder produced:

```text
pre_projection_private_fold_plus   imag +86.49127604166668
pre_projection_private_fold_minus  imag -85.87754498106062
post_projection_control            imag  +2.519140625000005
source-off / neutral delta                0.4185156249999977
neutral tolerance                         5.0
```

Checkpoint:

`CODED_PREPROJECTION_RESTORED_CHECKPOINT_20260712.json`

The first query-scramble attempt failed before worker execution because the new pilot
variant was missing from `live_gate_a_target.py`'s CLI choices. The retained local
stderr is under:

`runs/coded_preprojection_warm_query_scramble_0_argparse_failed/`

Policy4 and policy5 were verified restored to `800000-3200000` kHz after that failure,
and no CAT_CAS worker process remained. After the exact CLI-choice defect was repaired,
the same run ID was executed once:

```text
run id        coded_preprojection_warm_query_scramble_0
schedule      WU WU N0 QS0 QS1 QS2 QS3 QM0 QM1 QM2 QM3 C0 C1 C2 C3 N1
source bundle c74a03e8f38cac8b9e554104dc80096ef3ec358f27e73f8a54d205acdfc124bc
schedule hash 88a93ac2a565f612a3a3789b515a187dbb1e4196519962d56a2be09df2eb0ca7
```

The query-scramble transaction completed, copied back, cleaned the remote run root,
restored policies 4 and 5 to `800000-3200000` kHz by readback, and performed zero
voltage access and zero MSR access. Capture was accepted with service spikes, all 16
slots had 1000 samples, all stimulus bursts completed inside their slots, there were
no skipped deadlines, no missing slots, no sender spill, and no record-integrity
failure.

The frozen quadrature decoder produced a null control:

```text
query-scramble plus  imag -0.9868607954545379
query-scramble minus imag +1.93072916666668
post control         imag +1.4726562499999991
null bound                 5.0
neutral delta              0.306984374999999
neutral tolerance          5.0
```

Checkpoint:

`CODED_PREPROJECTION_QUERY_SCRAMBLE_CHECKPOINT_20260712.json`

**Status:** restored physical fold-odd response candidate with query-scramble killed.
Do not claim `SMALL_WALL_CROSSED`: the repaired warm loop clears source-off,
post-projection, private-fold sign, neutral restoration, and query-scramble, but
equal-footprint baseline curvature still has to be canceled by construction before
promotion.

The query-off run used the same warm-restored geometry with equal footprint in the
query window:

```text
run id        coded_preprojection_warm_query_off_0
schedule      WU WU N0 QO0 QO1 QO2 QO3 QO4 QO5 QO6 QO7 C0 C1 C2 C3 N1
source bundle b8595447f1b4c42321ad39f925aa9c3a574fc8cee90afaf6f7a32d570571665e
schedule hash 95d25a543007bdfcdb002ff0ce36642e9f64ef2280d262261e5ea17557482137
```

The transaction completed, copied back, cleaned the remote run root, restored policies
4 and 5 to `800000-3200000` kHz by readback, and performed zero voltage access and
zero MSR access. Capture was accepted, all 16 slots had 1000 samples, all stimulus
bursts completed inside their slots, there were no skipped deadlines, no missing slots,
no sender spill, and no record-integrity failure.

The frozen decoder did not pass the query-off null bound:

```text
query-off plus   imag  +6.980078124999999
query-off minus  imag +29.004501488095237
post control     imag  +4.615625000000001
null bound             13.846875000000004
neutral delta           0.2818125000000009
neutral tolerance       5.0
```

This does not reproduce the restored candidate's opposed fold-odd signature because
both query-off imag components are positive. It does expose residual same-sign
baseline curvature in the query-off geometry.

Checkpoint:

`CODED_PREPROJECTION_QUERY_OFF_CHECKPOINT_20260712.json`

**Status:** H5 remains unresolved rather than confirmed. Query-scramble is killed;
query-off is not clean-null because equal-footprint query slots retain same-sign
curvature. It did not reproduce the restored candidate's opposed fold-odd signature.

The declaration-sham run kept the restored candidate's public P/M/C declaration and
timing, but forced every P, M, and C physical footprint to the equal footprint:

```text
run id        coded_preprojection_warm_declaration_sham_0
schedule      WU WU N0 P0 P1 P2 P3 M0 M1 M2 M3 C0 C1 C2 C3 N1
source bundle 168bf9320acd35efb15ca6366c92788b04fcdebdf588a95c048f40f28aed7f52
schedule hash 89e53ef27c3799cc9c319283821e728e304a8b36a92ac1a76088f28934992310
```

The transaction completed, copied back, cleaned the remote run root, restored policies
4 and 5 to `800000-3200000` kHz by readback, and performed zero voltage access and
zero MSR access. Capture was accepted with service spikes, all 16 slots had 1000
samples, all stimulus bursts completed inside their slots, there were no skipped
deadlines, no missing slots, no sender spill, and no record-integrity failure.

The frozen decoder again did not pass the strict null bound, but it stayed same-signed
instead of reproducing the restored candidate's opposed fold-odd coordinate:

```text
declaration-sham plus   imag  +7.241796874999999
declaration-sham minus  imag +17.296093750000004
post control            imag  +1.1906249999999974
null bound                    5.0
neutral delta                 0.47585937499999886
neutral tolerance             5.0
```

Checkpoint:

`CODED_PREPROJECTION_DECLARATION_SHAM_CHECKPOINT_20260712.json`

**Status before phase-local repair:** H5 remained unresolved rather than confirmed or
killed. Query-scramble was a clean null; query-off and declaration-sham exposed
equal-footprint same-sign curvature; declaration alone did not produce the opposed
response. That made phase-local baseline cancellation the next required discriminator.

The phase-local schedule used balanced P/C/M and M/C/P triples:

```text
WU WU N0 P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3 N1
```

The equal-footprint sham passed the null classification:

```text
run id        coded_preprojection_warm_phase_local_sham_0
source bundle 93eecef06f1fabf0e51a4f81601f214720b13c8e01f05e31564c921bc105036e
schedule hash 51f3fb66cd4f03dff2d3e9aab9196d4f94d85e221cd552b04eda4929669cca2e
sham plus     imag -23.801065340909098
sham minus    imag -29.318252840909096
post control  imag +31.972549715909096
null bound          95.91764914772729
null passed         true
neutral delta        0.6180312500000014
```

The matched physical P/M footprint candidate restored and ran cleanly, but did not
produce an opposed fold-odd coordinate:

```text
run id        coded_preprojection_warm_phase_local_0
source bundle 13cca32cbdeb5d06a72109d73901c880c83a40872ba93a5a08b8abec92333fa5
schedule hash 1144b929905e30f3da1261fdedf5e6393c30d31a41c9a4dd2dc39e8573f4cbc4
candidate plus   imag +93.25494115259741
candidate minus  imag  +5.027810470779222
post control     imag  -3.0811011904761902
fold_odd_opposed       false
fold_odd_signal        false
neutral delta           0.634421875000001
```

Both phase-local transactions completed with verified copy-back, remote cleanup,
policy4/policy5 restoration to `800000-3200000` kHz by readback, zero voltage access,
zero MSR reads/writes, no skipped deadlines, no missing slots, no sender spill, and no
record-integrity failure.

Checkpoint:

`CODED_PREPROJECTION_PHASE_LOCAL_CHECKPOINT_20260712.json`

**Status:** the current timing-carrier implementation of H5 is not established under
phase-local baseline cancellation. The earlier sequential restored candidate is
demoted as layout/baseline-sensitive. H5 remains alive only as an access-model family
that now needs a different carrier/observable or a stronger physical restoration law.

The follow-on PMU carrier discriminator implemented the same phase-local public query
shape on the established ownership-intent surface, using ordinary `perf_event_open`,
rotated CAT_CAS-owned line-bank spans, and the already accepted
`remote_store_same_value` operator:

```text
run id        f10_phase_local_pmu_0
source bundle 4adb43d6206daadcafa438062008a424fe73e7542afda515b441da6f8dbef9df
sequence      P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3
worker status PHASE_LOCAL_PMU_CODED_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads, and zero MSR writes.
Temperature stayed below the 68 C veto. Every PMU window completed, all windows were
unmultiplexed, and the carrier byte digest restored after every window.

The decoded PMU quadrature produced opposed candidate signs, but neither established
coherence counter cleared the sham floor:

```text
cache_block_commands_change_to_dirty:
  sham floor       1.228536697394e-05
  candidate plus   9.575332079936e-06
  candidate minus -1.644057619436e-06
  signal           false

probe_responses_dirty:
  sham floor       2.224560787476e-05
  candidate plus   1.129175296055e-05
  candidate minus -1.098277509839e-05
  signal           false
```

Checkpoint:

`F10_PHASE_LOCAL_PMU_CHECKPOINT_20260712.json`

**Status:** H5 remains alive as an access-model family, but the tested
ownership-intent PMU footprint carrier is not established. Do not repeat the same
timing or PMU footprint discriminator unchanged.

The active-query receiver-delta discriminator then moved the query basis into the
measurement workload before scalar recording. Instead of measuring a passive response
buffer and decoding later, the receiver alternated matched positive/negative
CAT_CAS-owned response sub-banks and recorded the balanced delta directly:

```text
run id        coded_preprojection_active_query_0
variant       coded-preprojection-active-query-loop
source bundle e04ea5136af49a170792ba8672555c8c3c94431510fe8f12e2f542dd0670aad3
schedule hash 5a0ac285435ba33a80a3272020f19c85004fc63949f1df4325a3ad90fdcd87f2
schedule      WU WU N0 P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3 N1
```

The transaction completed with verified copy-back and remote cleanup. It restored
policy4 and policy5 to `800000-3200000` kHz by readback, performed zero voltage
access and zero MSR access, stayed below the 68 C temperature veto, produced all
16,000 samples, completed every stimulus burst inside its slot, skipped no deadlines,
had no missing slot, no sender spill, and no record-integrity failure. Capture was
accepted with service spikes.

Decoded active-query observables:

```text
pre_projection_private_fold_plus:
  real +10.739075969827587
  imag  -0.5096472537878798
  abs  +10.751162402690317

pre_projection_private_fold_minus:
  real  +2.431522253787879
  imag  +0.12953361742424174
  abs   +2.434970108380942

post_projection_control:
  real  +0.14023437499999955
  imag  -3.407421875
  abs   +3.4103063666128524
```

The candidate signs were opposed, but the active-query fold-odd magnitude did not
clear the post-control floor:

```text
fold_odd_opposed          true
fold_odd_exceeds_controls false
fold_odd_signal_candidate false
neutral N0/N1 range       0.24153124999999998
```

Checkpoint:

`CODED_PREPROJECTION_ACTIVE_QUERY_CHECKPOINT_20260712.json`

**Status:** this exact active-query receiver-delta mapping is negative. It changes the
access model and restores cleanly, but does not establish a control-clean fold-odd
carrier. H5 remains alive only for a materially stronger carrier/query algebra, not a
repeat of this phase-local active-query delta schedule.

The source-phase-chop discriminator then kept the receiver passive and moved the
public phase waveform into the source burst itself. Each driven slot used four
source-side segments over CAT_CAS-owned occupancy buffers before scalar receiver
recording:

```text
run id        coded_preprojection_source_phase_chop_0
variant       coded-preprojection-source-phase-chop-loop
source bundle e9764a8163ca9d7635562ccc3d756264bf9afd3f5dc1e1b4d3b0aa79706271b3
schedule hash 0308e6518c6e8e4fd60862f3825750a3865d3f8cb4cbef59d140eefb6d2e0fb1
schedule      WU WU N0 P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3 N1
```

The transaction completed with verified copy-back and remote cleanup. It restored
policy4 and policy5 to `800000-3200000` kHz by readback, performed zero voltage
access and zero MSR access, stayed below the 68 C temperature veto, produced all
16,000 samples, completed every stimulus burst inside its slot, skipped no deadlines,
had no missing slot, no sender spill, and no record-integrity failure. Capture was
accepted with service spikes.

Primary source-phase lock-in:

```text
plus mean          +35.98411458333335
minus mean         -28.657747395833333
control mean        -0.4480468749999995
control floor      27.4125
three-times floor  82.2375
opposed sign        true
signal candidate    false
neutral delta        0.5588281249999909
```

Checkpoint:

`CODED_PREPROJECTION_SOURCE_PHASE_CHOP_CHECKPOINT_20260712.json`

**Status:** this exact source-side phase-chop mapping is negative. It produces opposed
signs and restores cleanly, but the control floor is too large for the `3x` rule. H5
remains alive only for a materially stronger source-owned carrier/query algebra, not a
repeat of this phase-local burst-segmentation schedule.

### H6 - Alternative carriers remain available

Candidate families:

- Instruction Based Sampling as a diagnostic microscope only if a later kernel/tool
  path exposes usable samples;
- write-combining buffer state and flush order;
- thermal afterimage on longer timescales;
- cache-line eviction topology;
- another phase-native or resonance coordinate;
- a new software representation that preserves operator and path state.

These are not fallback decorations. Promote one when H1-H5 expose a precise carrier,
operator, observability, or restoration wall that it can attack.

The first IBS availability probe was negative:

```text
run id        f10_ibs_first_light_1
source bundle 0a28ed323e162800e3df2970dc4bafee4fd353ce420e50b43a7574497bea5d01
worker status IBS_FIRST_LIGHT_NOT_AVAILABLE
```

The lab kernel exposed `ibs_fetch` type `10` and `ibs_op` type `11`, but all tested
ordinary `perf_event_open` forms failed with `EINVAL`:

```text
ibs_fetch_default   open_errno 22
ibs_fetch_rand_en   open_errno 22
ibs_op_default      open_errno 22
ibs_op_cnt_ctl      open_errno 22
raw_cycles_precise  open_errno 22
raw_uops_precise    open_errno 22
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, temperature below veto, and no
forbidden CAT_CAS process residue.

Checkpoint:

`F10_IBS_FIRST_LIGHT_CHECKPOINT_20260712.json`

**Status:** IBS is not a usable near-term carrier under the current ordinary
`perf_event_open` access path. Do not spend the next loop on IBS unless the mechanism
changes to a real sampler or the kernel/tool boundary changes.

The write-combining/flush-order probe used only ordinary user-space `clflush`,
non-temporal same-value stores, CAT_CAS-owned aligned lines, and the existing
`primary_nb_coherence` PMU group:

```text
run id        f10_wc_flush_order_0
source bundle 09275480cfa7139869b094068b188a3eb88f3329890959806169585845cbdef0
worker status WC_FLUSH_ORDER_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, temperature below veto, every PMU window unmultiplexed, and carrier
byte digest restored after every window.

The operators were measurable, but the order pair was not:

```text
change_to_dirty:
  identity                 0
  flush_only              21
  normal_store_same     1933
  nt_store_same          132
  flush_then_nt_store     62
  nt_store_then_flush     56
  order_delta              6
  control_range         1933

probe_dirty:
  identity                 0
  flush_only            4178
  normal_store_same     6019
  nt_store_same         4358
  flush_then_nt_store   4221
  nt_store_then_flush   4210
  order_delta             11
  control_range         6019
```

Checkpoint:

`F10_WC_FLUSH_ORDER_CHECKPOINT_20260712.json`

**Status:** user-space flush and non-temporal same-value store operators are visible,
but this order pair does not carry the needed relation. Do not repeat this WC/flush
order discriminator unchanged.

The eviction/topology first-light probe used a separate 16 MiB CAT_CAS-owned
eviction buffer as a predetermined physical preconditioner, then measured the
established byte-preserving remote same-value-store sentinel with the
`primary_nb_coherence` PMU group:

```text
run id        f10_eviction_sentinel_0
source bundle 50fd61fd5747474734761f1ba23590dc99f8486fa6a7b94b55e89f59eb73b79c
worker status EVICTION_SENTINEL_RESPONSE_FOUND
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, no unrelated-process observation, temperature below veto, every PMU
window unmultiplexed, and both carrier and eviction-buffer digests restored.

Baseline and sentinel movement:

```text
baseline change_to_dirty 2067, movement threshold 206
baseline probe_dirty     4476, movement threshold 447

home_read_eviction:
  change_to_dirty 1908, delta 159
  probe_dirty     5403, delta 927

home_write_eviction:
  change_to_dirty 1783, delta 284
  probe_dirty     5092, delta 616

remote_write_eviction:
  change_to_dirty 1892, delta 175
  probe_dirty     5355, delta 879

remote_then_home_read_eviction:
  change_to_dirty 2034, delta 33
  probe_dirty     5340, delta 864
```

Checkpoint:

`F10_EVICTION_SENTINEL_CHECKPOINT_20260712.json`

**Status:** a restoration-sentinel carrier is established. The useful next move is a
phase-local coded discriminator on this carrier, with an equal-prep sham and fixed
public quadrature weights. This still does not establish path memory, holonomy,
fold-odd recovery, or a Small Wall crossing.

The first phase-local eviction-sentinel discriminator used the established sentinel
carrier with public P/C/M phases, an all-control equal-prep sham, and a high/low
candidate split:

```text
run id        f10_eviction_phase_local_0
source bundle 46f9e54c27d59b0bf315c326054386f1cb0c8403c6954ada56953b338283e647
worker status EVICTION_PHASE_LOCAL_CODED_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, no unrelated-process observation, temperature below veto, every PMU
window unmultiplexed, and both carrier and eviction-buffer digests restored.

Decoded observables:

```text
cache_block_commands_change_to_dirty:
  sham floor       1.668580446326e-05
  candidate plus   4.686098717265e-07
  candidate minus -4.409157976248e-06
  opposed sign     true
  signal           false

probe_responses_dirty:
  sham floor       2.003894391992e-05
  candidate plus   3.147384793339e-06
  candidate minus  3.938122407007e-07
  opposed sign     false
  signal           false
```

Checkpoint:

`F10_EVICTION_PHASE_LOCAL_CHECKPOINT_20260712.json`

**Status:** this high/low phase-local mapping is negative. The carrier remains alive,
but the equal-prep sham exposed sequence-position curvature. The next discriminator
must cancel local drift by construction, for example by bracketing each P or M token
with immediate control-prep sentinel windows before decoding.

The bracketed phase-local discriminator then measured each public P/M token between
immediate control-prep sentinel windows:

```text
run id        f10_eviction_bracket_0
source bundle 5e3fc671e398fafe21d11bf4539a16b935f9c1275c7e97d0daf0ca8ea644c237
worker status EVICTION_PHASE_BRACKETED_CODED_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, no unrelated-process observation, temperature below veto, every PMU
window unmultiplexed, and both carrier and eviction-buffer digests restored.

Decoded observables:

```text
cache_block_commands_change_to_dirty:
  sham floor       4.272108439287e-06
  candidate plus   1.171144192425e-05
  candidate minus -9.952010687383e-06
  opposed sign     true
  signal           false

probe_responses_dirty:
  sham floor       2.939070290535e-06
  candidate plus   1.442723561986e-05
  candidate minus -7.041693848168e-07
  opposed sign     true
  signal           false
```

Checkpoint:

`F10_EVICTION_PHASE_BRACKETED_CHECKPOINT_20260712.json`

**Status:** bracketing helped but did not promote. Change-to-Dirty is now the active
coordinate: it had opposed signs and missed the `3 * sham_floor` rule by margin rather
than by sign. The next cheap discriminator is a stronger Change-to-Dirty contrast
using the same bracketing, not a repeat of the same high=`home_read_eviction`,
low=`home_write_eviction` mapping.

The stronger Change-to-Dirty bracketed contrast then used high=`none` and
low=`home_write_eviction` under the same control bracketing:

```text
run id        f10_eviction_bracket_c2d_0
source bundle d427bcd135a5714e6a67241c998f57fa9e768cfc68912c81a2fabb4c64cf26ca
worker status EVICTION_PHASE_BRACKETED_C2D_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, no unrelated-process observation, temperature below veto, every PMU
window unmultiplexed, and both carrier and eviction-buffer digests restored.

Decoded observables:

```text
cache_block_commands_change_to_dirty:
  sham floor       3.040213368253e-06
  candidate plus  -8.558015947985e-06
  candidate minus -1.610243223208e-06
  opposed sign     false
  signal           false

probe_responses_dirty:
  sham floor       3.430795076460e-06
  candidate plus  -2.453676655913e-05
  candidate minus -1.829742445338e-05
  opposed sign     false
  exceeds 3x sham  true
  signal           false
```

Checkpoint:

`F10_EVICTION_PHASE_BRACKETED_C2D_CHECKPOINT_20260712.json`

**Status:** the stronger bracketed C2D prep contrast is negative. It made a large
same-sign probe response and did not preserve the opposed Change-to-Dirty geometry.
Do not repeat this eviction-sentinel phase-local family unchanged.

The duration-primary bracketed discriminator then used the same no-prep versus
home-write centers and made bracketed duration the primary observable. The first
enlarged-buffer attempt, `f10_eviction_duration_0`, failed mechanically with worker
return `-11` before writing the worker result. It copied back `FINAL_RESULT.json`,
performed zero frequency writes, zero voltage access, zero MSR reads/writes,
temperature stayed below veto, and the remote run root was verified absent after
cleanup. The exact enlargement was removed before the single rerun.

The repaired standard-buffer duration run was:

```text
run id        f10_eviction_duration_1
source bundle 8d6a0d63213bcc479cdf0003a8363b24db12ecbc0b3a79079727b750dee97687
worker status EVICTION_PHASE_BRACKETED_DURATION_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup, zero frequency
writes, zero voltage access, zero MSR reads/writes, no physical-address access, no
cache-set mapping, no unrelated-process observation, temperature below veto, every PMU
window unmultiplexed, and both carrier and eviction-buffer digests restored.

Primary decoded duration:

```text
duration_ns:
  sham floor       7604.25
  candidate plus  -4132.5
  candidate minus  6568.75
  opposed sign     true
  signal           false
```

Checkpoint:

`F10_EVICTION_PHASE_BRACKETED_DURATION_CHECKPOINT_20260712.json`

**Status:** duration bracketing is negative. The eviction-sentinel carrier remains a
useful restoration-sentinel calibration, but the tested PMU and duration phase-local
uses do not establish a control-clean fold-odd response. The next move should change
carrier family or access model rather than continuing eviction-sentinel remaps.

The process-lifecycle discriminator then tested whether a source-owned runtime carrier
survives source-process exit strongly enough to perturb a fresh sentinel. A child
process on core 4 applied neutral, forward, reverse, or shuffle balanced read/store
history over CAT_CAS-owned shared anonymous memory and exited. A fresh measurement
process on core 5 then ran the byte-preserving `remote_store_same_value` sentinel under
the established `primary_nb_coherence` PMU group:

```text
run id        f10_process_lifecycle_0
source bundle 28ef9ca7752f7ee92981edb3c413992809b62f411e0bf1279d5b3978dee15e1a
worker status PROCESS_LIFECYCLE_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads/writes, stayed below the
68 C temperature veto, found no forbidden CAT_CAS process residue, all PMU windows
were unmultiplexed, both source and sentinel children exited cleanly, and the shared
buffer digest was unchanged after source and after restore.

Fresh-process sentinel contrasts:

```text
cache_block_commands_change_to_dirty:
  identity 2206
  forward  2194
  reverse  2181
  shuffle  2314
  delta      13
  floor     108
  threshold 324
  signal false

probe_responses_dirty:
  identity 3926
  forward  3956
  reverse  3834
  shuffle  4016
  delta     122
  floor      90
  threshold 270
  signal false

duration_ns:
  identity 2231595
  forward  2192989
  reverse  2185991
  shuffle  2183831
  delta       6998
  floor      47764
  threshold 143292
  signal false
```

Checkpoint:

`F10_PROCESS_LIFECYCLE_CHECKPOINT_20260712.json`

**Status:** this exact source-process lifecycle carrier is negative. It changes the
access model by forcing the source process to exit before the sentinel, but the
forward/reverse residuals remain inside neutral/shuffle controls. Do not rerun this
neutral/forward/reverse/shuffle child-process read-store history unchanged.

The indirect-target discriminator then changed the public branch carrier from
conditional branch outcomes to indirect target selection at one CAT_CAS-owned call
site. Core 5 trained neutral, forward, reverse, or shuffle balanced target sequences,
then measured a fixed sentinel target sequence under the established
`branch_history_group` PMU group:

```text
run id        f10_indirect_target_history_0
source bundle 091c2f6636081b09654bab944b297d619e8c503cdb1e7465e335048b96343308
worker status INDIRECT_TARGET_HISTORY_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads/writes, stayed below the
68 C temperature veto, found no forbidden CAT_CAS process residue, all PMU windows
were unmultiplexed, and the target-pattern digest was unchanged after history and
after neutral restore.

Indirect-target sentinel contrasts:

```text
retired_mispredicted_branch_instructions:
  identity 20804
  forward  20782
  reverse  20779
  shuffle  20778
  delta        3
  floor       26
  threshold   78
  signal false

duration_ns:
  identity 1793079
  forward  1791799
  reverse  1791670
  shuffle  1791491
  delta        129
  floor       1588
  threshold   4764
  signal false
```

Checkpoint:

`F10_INDIRECT_TARGET_HISTORY_CHECKPOINT_20260712.json`

**Status:** this exact public indirect-target history carrier is negative. It changes
the branch carrier from conditional outcome history to indirect target selection, but
the forward/reverse residuals remain inside neutral/shuffle controls. Do not rerun this
neutral/forward/reverse/shuffle indirect-target sequence unchanged.

The store/load alias-history discriminator then changed carrier family to the
memory-ordering surface: core 5 trained neutral, forward, reverse, or shuffle balanced
same-page-offset store/load histories over CAT_CAS-owned pages, then measured a fixed
same-offset store/load sentinel under the `translation_history_group` PMU group:

```text
run id        f10_store_load_alias_0
source bundle bb182c0bf6adb1cedbf82bd64881f9bfadffc1e81ce500d9badca027f139ff71
worker status STORE_LOAD_ALIAS_HISTORY_RESPONSE_NOT_ESTABLISHED
```

The transaction completed with verified copy-back and remote cleanup. It performed
zero frequency writes, zero voltage access, zero MSR reads/writes, stayed below the
68 C temperature veto, found no forbidden CAT_CAS process residue, all PMU windows
were unmultiplexed, and the pattern-plus-byte digest was unchanged after history and
after neutral restore.

Store/load alias sentinel contrasts:

```text
duration_ns:
  identity 883239
  forward  879674
  reverse  879728
  shuffle  879439
  delta        54
  floor      3800
  threshold 11400
  signal false

cache_misses:
  identity 1
  forward  0
  reverse  0
  shuffle  0
  delta    0
  floor    1
  threshold 32
  signal false
```

Checkpoint:

`F10_STORE_LOAD_ALIAS_CHECKPOINT_20260712.json`

**Status:** this exact same-page-offset store/load alias-history carrier is negative.
It changes the runtime access model away from branch/cache-line ownership and into
store/load alias history, but the forward/reverse residuals remain inside
neutral/shuffle controls. Do not rerun this neutral/forward/reverse/shuffle store/load
alias sequence unchanged.

## Cheapest current discriminator

Change carrier family or access model again, not another remap of the same
phase-local timing/PMU/eviction/active-query/source-phase-chop/restored-history,
simple branch-history/indirect-target-history/translation-footprint/prefetch-stream
geometry, same-page-offset store/load alias-history, or the current fresh
source-process lifecycle sentinel. The run must stay closed: CAT_CAS-owned buffers
only, predetermined geometry, no physical-address access, no cache-set mapping, no
unrelated-process observation, and no MSR or voltage access.

The next probe should answer:

1. Is there a source-owned runtime carrier whose state survives long enough to couple
   before the public fold-even projection?
2. Can the query algebra preserve a conjugate/source-side coordinate without turning
   into receiver-order service curvature or source-segment timing spread?
3. Does a killing control remove the response without relying on route, line-set,
   eviction-prep, active-query subbank order, source-segment timing spread, or
   sequence-position or process-lifecycle imbalance?
4. Does the carrier restore beyond byte equality under the new observable?

Do not rerun the same timing coded loop, another scalar PMU route metric,
unconditioned transient timing repeat, IBS availability probe, WC/flush-order pair, or
unlabeled cache-line rectangle unchanged. Do not rerun the same restored
history-sentinel ownership sequence unchanged. Do not rerun the same branch-history
training/sentinel discriminator unchanged. Do not rerun the same translation-history
page-footprint discriminator unchanged. Do not rerun the same prefetch-stream
sentinel discriminator unchanged. Do not rerun
`coded_preprojection_active_query_0` or the same active-query receiver-delta schedule
unchanged. Do not rerun `coded_preprojection_source_phase_chop_0` or the same
source-phase-chop schedule unchanged. Do not rerun `f10_process_lifecycle_0` or the
same neutral/forward/reverse/shuffle child-process read-store lifecycle sentinel
unchanged. Do not rerun `f10_indirect_target_history_0` or the same
neutral/forward/reverse/shuffle indirect-target sequence unchanged. Do not rerun
`f10_store_load_alias_0` or the same neutral/forward/reverse/shuffle same-offset
store/load alias sequence unchanged.

## Current claim ceiling

`EVICTION_SENTINEL_RESPONSE_FOUND`

The next major scientific threshold is a physical coded pre-projection response with
restoration and killing controls. `coded_preprojection_loop_2` is a restored physical
fold-odd response candidate, and `coded_preprojection_warm_query_scramble_0` killed the
physical query-scramble alternative. `coded_preprojection_warm_query_off_0` and
`coded_preprojection_warm_declaration_sham_0` exposed same-sign residual curvature.
`coded_preprojection_warm_phase_local_sham_0` passed the phase-local null rule, but
`coded_preprojection_warm_phase_local_0` did not produce an opposed fold-odd signal.
`f10_phase_local_pmu_0` then produced opposed PMU signs but stayed below the
three-times-sham-floor rule. `f10_ibs_first_light_1` showed that IBS is exposed in
sysfs but not usable through the tested ordinary `perf_event_open` forms.
`f10_wc_flush_order_0` made flush/non-temporal operators visible but not order-sensitive
under controls. `f10_eviction_sentinel_0` then established a measured
restoration-sentinel carrier by showing that predetermined eviction-buffer
preconditioning changes the later ownership-intent sentinel while both buffers restore.
`f10_eviction_phase_local_0` did not promote that carrier into a phase-local coded
response because the equal-prep sham curvature exceeded the candidate.
`f10_eviction_bracket_0` reduced the sham floor and produced opposed Change-to-Dirty
candidate signs, but did not exceed the three-times-sham threshold.
`f10_eviction_bracket_c2d_0` then showed that simply strengthening the bracketed prep
contrast collapses the sign. `f10_eviction_duration_1` extended the rejection to a
duration-primary observable. `coded_preprojection_active_query_0` then changed the
access model by measuring a balanced active-query receiver delta before scalar
recording, but the opposed candidate did not clear the post-control floor.
`coded_preprojection_source_phase_chop_0` then moved the public phase waveform into the
source burst before scalar recording, but its control floor was too large for the
`3x` rule. `f10_history_sentinel_0` then tested whether balanced ownership-transfer
history survives a common home-core restore strongly enough to perturb a later
remote-store sentinel; it restored cleanly, but forward/reverse deltas stayed below the
neutral/shuffle floor. `f10_branch_history_0` then changed carrier family to
same-core public branch-history state; it also ran cleanly, but branch-miss and
duration deltas stayed inside the neutral/shuffle controls. The current marker remains
short of `SMALL_WALL_CROSSED`. `f10_translation_history_0` then moved to read-only
translation/page-footprint state and stayed negative: forward/reverse timing and
cache-miss deltas were tiny against the shuffle floor. `f10_prefetch_stream_0` then
tested a flushed sentinel stream after adjacent forward/reverse stream training and
also stayed negative. `f10_process_lifecycle_0` then forced the source history into a
separate exited child process before a fresh remote-store sentinel and also stayed
negative. `f10_indirect_target_history_0` then changed the public branch carrier from
conditional outcomes to indirect target selection and also stayed negative.
`f10_store_load_alias_0` then changed carrier family to same-page-offset store/load
alias history and also stayed negative. The next move must change carrier family or
access model rather than keep remapping the same eviction-sentinel PMU/timing,
active-query phase-local, source-phase-chop, restored two-line-set ownership-history,
simple branch-history, indirect-target-history, translation-footprint, store/load
alias-history, prefetch-stream, or source-process lifecycle geometry.

## State update rule

Replace or compress this file when the active boundary changes. Do not append an
unbounded diary. Preserve old exact evidence in Git history and retained run outputs.

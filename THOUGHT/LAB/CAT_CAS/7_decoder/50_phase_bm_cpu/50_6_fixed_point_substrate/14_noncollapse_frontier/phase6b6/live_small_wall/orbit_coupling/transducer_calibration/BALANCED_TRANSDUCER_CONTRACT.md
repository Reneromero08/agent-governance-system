# Balanced Public Transducer Contract

Date: 2026-07-12

## Claim Ceiling

This lane contains no private OrbitState member, no private condition map, no
unblinding step, and no Small Wall crossing claim.

Allowed classifications:

```text
BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED
BALANCED_PHYSICAL_TRANSDUCER_PARTIAL
BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED
```

Forbidden classifications:

```text
ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE
SMALL_WALL_CROSSED
```

## Public Calibration Law

The signed public ladder is frozen:

```text
q = {-1536, -1024, -512, 0, +512, +1024, +1536}
M = 2048
positive_work = M + q
negative_work = M - q
total_work = 4096
```

The minimum measured-bank prefix is 512 lines. The maximum measured-bank prefix is
3584 lines. Both are strictly below the 4096-line bank size.

## Bank Geometry

Frozen bank geometry:

```text
resident virtual banks = A, B
bank lines = 4096
line bytes = 64
bank bytes = 262144
alignment = 4096 bytes
source core = 4
receiver core = 5
```

Both banks are experiment-owned, synthetic, and process-local resident virtual banks.
No physical-address analysis or cache-set mapping is used, so this contract does not
claim fresh physical-frame identity.

## Physical Baseline

Each trial begins from a controlled physical baseline:

1. Allocate resident virtual bank A and resident virtual bank B.
2. Touch and initialize every byte of both banks symmetrically with identical
   deterministic bytes.
3. Verify equal byte digests and direct byte comparison against the deterministic
   pattern.
4. Pin to receiver core 5 and perform a full-bank same-value-store sweep over A and B
   using the same public permutation.
5. Fence and verify that bytes remain unchanged.
6. Record a multi-region receiver-owned sentinel over both banks before source
   encoding.

The encoded source operation is never the first write to either mapping.

## Line Permutation

The source and receiver operators use the same public affine permutation:

```text
line(i) = (257 * i + 43) mod 4096
```

`257` is coprime with `4096`. The same `a` and `b` are used for both banks and both
logical roles. A prefix must never wrap, and every line in each prefix must be unique.

## Trial Schedule

The public trial schedule is frozen before live execution and stored as:

```text
PUBLIC_TRIAL_SCHEDULE.json
PUBLIC_TRIAL_SCHEDULE.sha256
```

Frozen seed:

```text
CAT_CAS_BALANCED_TRANSDUCER_PUBLIC_SEED_V1
```

Fresh process replicates:

```text
2
```

Within each fresh process:

```text
7 q values * 8 crossover pairs per q * 2 mapping legs = 112 paired leg records
```

Every q, mapping, source-order, and receiver measurement-order cell has exactly two
records per fresh process. Every crossover pair runs both physical mappings on one
A/B allocation before that allocation is freed. Mapping order is balanced. The
schedule is materialized locally before target contact, transferred with the source
bundle, and verified by hash before execution. The runtime receives the frozen
schedule TSV and records each executed row; it does not generate or adapt the schedule.

## Pointer-Swap Control

Mapping 0:

```text
logical positive -> physical A
logical negative -> physical B
```

Mapping 1:

```text
logical positive -> physical B
logical negative -> physical A
```

The runtime records both:

```text
logical differential = response(logical positive) - response(logical negative)
physical differential = response(physical A) - response(physical B)
```

Expected law:

```text
logical differential remains invariant under pointer swap
physical A-minus-B changes sign under pointer swap
```

Both mapping legs of a crossover share the same allocated A/B pair. The receiver
baseline is re-established and verified between legs.

## Source-Order Control

Source encoding order is a frozen public factor:

```text
positive-source-first
negative-source-first
```

It is crossed with q, mapping, and receiver measurement order. The logical response
must agree across source-order strata, and the odd transfer must be present inside
every mapping x source-order x receiver-order stratum.

## PMU Configuration

The receiver measures ordinary Linux `perf_event_open` raw events in a pinned,
process-scoped window:

```text
cpu_cycles_not_halted                event 0x076 umask 0x00 config 0x0076
cache_block_commands_change_to_dirty event 0x0ea umask 0x20 config 0x20ea
probe_responses_dirty                event 0x0ec umask 0x0c config 0x0cec
```

The PMU layer must:

- preflight-gate `AuthenticAMD` CPU family `16` and the expected Linux PMU
  `event`/`umask` format fields;
- set `exclude_kernel = 1`;
- set `exclude_hv = 1`;
- retain `time_enabled`;
- retain `time_running`;
- retain event IDs;
- reject partial group reads;
- reject `time_enabled != time_running`;
- reject event-order drift;
- report raw counts;
- report cycle-normalized Change-to-Dirty and dirty probes as secondary coordinates;
- perform no scaling of multiplexed counts.

## Observables

Every coordinate is adjudicated independently:

```text
change_to_dirty
probe_dirty
cycles
duration_ns
change_to_dirty_per_cycle
probe_dirty_per_cycle
```

`duration_ns` is retained in the response vector as a diagnostic coordinate because
it includes syscall/ioctl intervals that the PMU counters exclude. It cannot become
the primary calibrated coordinate.

The fixed primary-coordinate priority is:

```text
change_to_dirty
probe_dirty
change_to_dirty_per_cycle
probe_dirty_per_cycle
cycles
```

All eligible coordinates are retained. The primary coordinate is the first eligible
coordinate in the fixed priority order.

## Acceptance Laws

The complete public null ceiling for each observable is built from:

- repeated held-in `q = 0` logical responses;
- `q = 0` pointer-swap residuals;
- `q = 0` measurement-order residuals;
- `q = 0` source-order residuals;
- `q = 0` mapping-order residuals;
- pre/post restoration sentinel variation;
- fresh-process variation;
- an absolute floor of `1.0`.

Frozen tolerances:

```text
oddness_error <= 0.25
pointer_swap_relative_error <= 0.25
measurement_order_relative_error <= 0.25
restoration_sentinel_relative_error <= 0.25
restoration_sentinel_absolute_floor = coordinate-specific
gain_multiplier = 3.0
```

For each coordinate:

- Held-out `F(0)` rows must lie inside the complete public null region.
- For `q in {512, 1024, 1536}`, `F(-q)` must be odd with `F(q)`.
- All resolved nonzero q values must share one sign convention:
  `sign(F(q)) = sign(q)` or `sign(F(q)) = -sign(q)`.
- `min(abs(F(+1024)), abs(F(-1024)), abs(F(+1536)), abs(F(-1536)))`
  must be greater than `3 * complete_null_ceiling`.
- Positive and negative ladders must satisfy
  `|F(512)| < |F(1024)| < |F(1536)|`.
- Logical response must be invariant under pointer swap.
- Raw physical A-minus-B must reverse under pointer swap.
- Positive-first and negative-first logical responses must agree.
- Positive-source-first and negative-source-first logical responses must agree.
- Every mapping x source-order x receiver-order stratum must independently satisfy
  sign, oddness, and gain laws with the same convention.
- Both fresh processes must use the same non-`none` sign convention and agree per q
  within the frozen replicate tolerance.
- The same coordinate must pass sign, oddness, gain, pointer-swap,
  measurement-order, and restoration laws independently in both fresh processes and
  in the aggregate.
- Contradictory fresh processes cannot be promoted by averaging.
- Schedule/raw/sentinel integrity is a hard gate for calibrated status.

Every accepted trial requires:

- bytes unchanged;
- direct byte comparison against the deterministic pattern, not only a digest;
- source total work exactly 4096;
- unique-line prefix law satisfied;
- all PMU windows unmultiplexed with positive `time_enabled == time_running`;
- receiver windows still resident on core 5 before and after the measured operator;
- receiver restoration sentinel within tolerance for A, B, common mode, and
  differential mode across four disjoint 512-line sentinel regions;
- temperature below veto;
- no process residue;
- policy limits unchanged or exactly restored.

Classification:

- `BALANCED_PHYSICAL_TRANSDUCER_CALIBRATED` if at least one frozen coordinate passes
  every law independently in both fresh processes and in the aggregate.
- `BALANCED_PHYSICAL_TRANSDUCER_PARTIAL` if a repeatable odd or monotonic public
  transfer exists but one or more required control or restoration laws fail.
- `BALANCED_PHYSICAL_TRANSDUCER_NOT_ESTABLISHED` if no coordinate has a reproducible
  public odd transfer above the complete null ceiling.

## Live Budget

One bounded live calibration is authorized under the current prompt:

```text
host = root@192.168.137.100
remote base = /root/catcas_live_small_wall
runtime timeout = 210 seconds
temperature veto = 68 C
frequency writes = 0 unless a later explicit need is found
voltage writes = 0
MSR reads = 0
MSR writes = 0
physical address access = false
cache set mapping = false
```

No automatic retry is allowed.

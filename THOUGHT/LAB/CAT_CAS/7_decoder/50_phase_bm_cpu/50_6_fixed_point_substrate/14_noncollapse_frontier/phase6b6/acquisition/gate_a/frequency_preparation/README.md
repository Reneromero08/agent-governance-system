# Gate A Frequency Preparation and Restoration

**Status:** `SOURCE_IMPLEMENTED__LIVE_WRITES_NOT_AUTHORIZED`

## Evidence basis

The integrated read-only observation on cores 4 and 5 established:

```text
cpufreq driver = acpi-cpufreq
governor = schedutil
policy ownership = separate policy4 and policy5
cpuinfo bounds = 800000–3200000 kHz
scaling bounds = 800000–3200000 kHz
available states = 3200000, 2400000, 1600000, 800000 kHz
200/200 observed samples = 800000 kHz on both cores
```

The frozen Gate A run plan requires the observed frequency to already equal
`1600000 kHz`. The smoke cannot write frequency state. Preparation must
therefore remain a separate transaction with its own review, authority,
evidence, and restoration proof.

## Exact write surface

Only four sysfs files may ever be written:

```text
/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
/sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
/sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
/sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
```

No governor, boost, voltage, MSR, firmware, sender, capture, or Gate A source is
modified.

Preparation order for each policy:

```text
scaling_max_freq = 1600000
scaling_min_freq = 1600000
```

Restoration order for each policy:

```text
scaling_min_freq = snapshotted minimum
scaling_max_freq = snapshotted maximum
```

The observed target baseline is exactly `800000–3200000 kHz`. Any policy,
driver, governor, ownership, bound, available-state, identity, write, readback,
observation, or restoration mismatch fails closed.

## Transaction

A future preparation-only qualification will perform:

```text
snapshot policy4 and policy5
validate exact baseline and 1600000 support
pin both policies with four writes
verify policy identity before and after every write
verify every write by immediate readback
observe 200 paired samples at 10 ms
require every pair to equal 1600000
verify policy identity before and after every paired sample
restore both policies with four writes
verify restored identities and bounds
stop
```

Restoration is attempted after every write-bearing failure. A restoration
failure is terminal and blocks all later target work. If a policy identity
changes, the transaction refuses to write to the replacement path and reports
a terminal restoration failure.

The core permits no automatic retry and caps the complete transaction at eight
frequency write attempts: four preparation writes and four restoration writes.
It records write attempts separately from write calls that returned normally.

## Current authority boundary

The CLI supports synthetic roots only and refuses `/sys`. The transaction core
also refuses `/sys`; no executable live writer or target transport exists in this
source lane.

```text
live frequency preparation authorized = false
preparation-only target qualification authorized = false
third Gate A attempt authorized = false
Gate B authorized = false
```

No execution authority artifact is created in this source lane. A future live
qualification requires an exact reviewed source, a separate owner authority,
one bounded target transaction, complete write/readback custody, and no retry.

## Next boundary

```text
INDEPENDENT_EXACT_HEAD_REVIEW_FOR_GATE_A_FREQUENCY_PREPARATION_RESTORATION
```

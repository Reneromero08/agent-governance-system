# Gate A Frequency Preparation Authority and Transport

**Status:** `SOURCE_IMPLEMENTED__NO_AUTHORITY_ARTIFACT__NO_TARGET_CONTACT`

This package is the authority-gated layer above the integrated synthetic
preparation/restoration core. It does not authorize or execute a live target
transaction by itself.

## Closed future transaction

A future exact owner authority may permit one preparation-only transaction:

1. prove the authority-specific claim, execution, stage, and archive namespaces absent;
2. create one durable claim outside all cleanup roots;
3. stage the deterministic five-file target payload and exact authority;
4. revalidate the target identity as `catcas`, `x86_64`, AMD Phenom II X6 1090T;
5. perform one four-write pin, 200-pair static observation, and four-write restoration;
6. seal target evidence, copy it back, validate its inventory, and clean transient paths;
7. retain the durable claim and prohibit any retry.

Only these live sysfs files may be written:

```text
/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq
/sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq
/sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq
/sys/devices/system/cpu/cpufreq/policy5/scaling_min_freq
```

No governor, boost, voltage, MSR, firmware, sender, capture, Gate A smoke, Gate B,
scientific acquisition, target coupling, or Small Wall surface is present.

## Failure and restoration

The target runner consumes the durable claim before target identity validation.
Every write-bearing failure enters restoration. Signal and alarm handlers raise
an interrupt through the transaction so its `finally` restoration path executes.
The host never deletes the target namespace while the writer remains observable.
Cleanup occurs only after the writer is absent and copied evidence validates.

The durable claim is never part of cleanup. `automatic_retry=false`, the target
transaction count is one, and the complete write-attempt cap is eight.

## Current authority boundary

```text
authority artifact created = false
live preparation authorized = false
target contact authorized = false
third Gate A smoke authorized = false
Gate B authorized = false
```

The next boundary after exact-head source review is:

```text
PROJECT_OWNER_DECISION_FOR_ONE_NO_RETRY_FREQUENCY_PREPARATION_QUALIFICATION_AUTHORITY
```

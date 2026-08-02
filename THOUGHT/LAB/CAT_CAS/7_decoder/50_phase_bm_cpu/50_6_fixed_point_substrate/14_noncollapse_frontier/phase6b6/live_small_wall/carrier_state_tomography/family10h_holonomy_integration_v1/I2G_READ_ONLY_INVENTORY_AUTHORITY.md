# I2G Read-Only Target Inventory Authority Boundary

## Result

```text
I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_CONTRACT_COMPLETE
AUTHORITY_NOT_GRANTED
TARGET_CONTACT_NOT_PERFORMED
TARGET_INVENTORY_NOT_ACQUIRED
GITHUB_ONLY_WORK_COMPLETE
PHYSICAL_PACKAGE_NOT_FREEZE_READY
NO_LIVE_EXECUTION_AUTHORIZED
```

I2F completed the non-executing interface specification. The remaining unknowns are
facts about the current Family 10h target: core topology, route choices, affinity
capability, PMU availability, page constraints, sensor authority, and source-isolation
capability.

Those facts cannot be established through GitHub alone.

## Proposed inventory scope

A future read-only inventory may observe:

```text
target identity and operating system
CPU, cache, NUMA, and online-core topology
current affinity capability
perf_event_paranoid and PMU event availability
page-size and memory-allocation capability
temperature sensor identities and permissions
frequency governor and policy state
toolchain versions
existing package paths and hashes
process and network custody needed for source isolation
```

The inventory may not open scientific PMU measurements, change affinity or policy,
allocate a carrier, compile or execute a backend, copy a package to the target, modify
files, or reuse any historical authorization.

## Required authority record

Before contact, a new record must bind:

```text
explicit user grant
access method
target identity
workspace or remote root
hash of the read-only command allowlist
output destination
cleanup law
```

Every field is currently unset.

## Current custody

```text
explicit_user_grant_recorded = false
target_access_method_recorded = false
target_contact_authorized = false
read_only_inventory_authorized = false
live_experiment_authorized = false
write_attempt_count = 0
scientific_measurement_count = 0
```

The generic continuation of repository work does not silently become target-contact
authority. Inventory and live execution remain separate gates.

## Next gate

```text
I2G_AUTHORITY_GRANT_AND_TARGET_ACCESS_REQUIRED
```

After an explicit grant and access method are recorded, the allowed inventory can be
implemented as a separately hashed, read-only transaction. A successful inventory would
still not authorize a physical experiment.

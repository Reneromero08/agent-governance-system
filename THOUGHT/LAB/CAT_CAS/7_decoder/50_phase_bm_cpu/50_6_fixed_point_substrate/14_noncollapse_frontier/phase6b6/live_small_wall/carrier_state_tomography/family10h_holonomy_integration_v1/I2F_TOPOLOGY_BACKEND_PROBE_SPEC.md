# I2F Topology, Backend, and Nondestructive Probe Specification

## Result

```text
I2F_TOPOLOGY_BACKEND_AND_NONDESTRUCTIVE_PROBE_SPEC_COMPLETE
ALL_PHYSICAL_ASSIGNMENTS_UNSET
TARGET_INVENTORY_REQUIRED_NEXT
PHYSICAL_PACKAGE_NOT_FREEZE_READY
I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_REQUIRED
NO_LIVE_EXECUTION_AUTHORIZED
FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED
FAMILY10H_HOLONOMY_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
```

I2F specifies the physical interfaces needed to populate the qualified I2E harness. It
does not inspect the target, assign cores, choose line addresses, implement a backend,
freeze thresholds, or authorize execution.

## Required role topology

A future package must bind these roles from fresh read-only target inventory:

```text
preparation core
home or reclaim core
remote operator core A
remote operator core B
route-control core
```

The two remote cores must be distinct from each other and from the home/reclaim core.
The preparation and home roles may coincide only if explicitly declared. Historical route
labels cannot be reused as current topology evidence.

All role values remain `null` and the assignment is unfrozen.

## Carrier layout

One controller-owned allocation must persist through preparation, source death, receiver
word, output decoupling, inverse word, and restoration measurement.

Required manifests:

```text
allocation identity
page identity
line identity
line-set A
line-set B
support cardinality
intersection cardinality
logical initialization digest
```

The supports must be partially overlapping. Disjoint and identical supports are required
controls. No concrete support or cardinality is currently assigned.

## Source capability boundary

The source must run as an exec-based preparation-only process. Its payload may contain
only the public carrier preparation. It must not contain or inherit:

```text
challenge entropy
realized operator word
accumulator target
analysis label
```

Receiver entropy is generated only after successful `waitpid`, descriptor closure, zero
source IPC, and zero surviving source helpers. Fork-only code-path secrecy is forbidden.

## Physical backend ABI

The future backend must implement ten methods:

```text
bind topology
bind carrier
prepare public codeword
seal source death
apply directional operation
measure state checkpoint
decouple accumulator
record restoration
record environment
destroy carrier
```

Every directional operation receipt binds the acting core, expected prior-owner proxy,
destination-owner proxy, route, support, amplitude, work budget, logical digests, state
receipts, barriers, and backend receipt identity.

The backend currently has status:

```text
SPEC_ONLY_NOT_IMPLEMENTED
physical_execution_enabled = false
```

## Nondestructive probe interface

Every checkpoint requires four observer roles and four support strata:

```text
home core
remote core A
remote core B
route control

A-only
A intersection B
B-only
outside union
```

The probe channels are:

```text
logical byte digest
D_single
D_local
change-to-dirty
probe-dirty
timing cycles
probe coordinate
```

A future public calibration must measure repeated-probe disturbance, measurement-only
baselines, randomized probe order, and same-carrier identity. The disturbance metric,
ceiling, and deadline are all unset.

## Accumulator backend

A physical accumulator must couple to carrier transport and must not read the public word
or source class. Required nulls are carrier-off, word-only replay, reference-only, and
coupling removal. It must decouple before inverse restoration and retain its output
afterward.

No accumulator backend is implemented.

## Environment receipt

Every transaction must classify:

```text
cache and coherence
page identity
frequency policy
temperature
source process
receiver process
measurement registers
accumulator state
external bath
```

An unclassified field is a hard failure.

## Freeze blockers

Nine blockers remain:

```text
concrete topology unassigned
carrier support unassigned
physical backend absent
nondestructive probe absent
accumulator backend absent
disturbance metric and ceiling unset
state-equivalence metric and thresholds unset
target inventory authority absent
live execution authority absent
```

## Next boundary

```text
I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_REQUIRED
```

Further physical design needs fresh read-only inventory of the actual target topology,
CPU affinity capability, observer routes, PMU availability, page allocation constraints,
and sensor authority. I2F itself grants no permission to obtain that inventory.

# Causal Separation Contract

Status: `FROZEN_DESIGN_CONSTRAINT`

The source and receiver must be separated by both time and capability.

## Source Capability

Before and during physical preparation, the source may read:

```text
public relation identity
sealed preparation seed
public carrier layout
public schedule skeleton
own process-local preparation receipts
```

The source must be technically incapable of reading:

```text
receiver query
query phase
query index
query order
expected response
collapse outcome
receiver measurement
receiver feedback
```

The source must close before receiver query selection. Closure requires:

```text
source process exit or sealed suspension
source receipts finalized
source output namespace closed read-only
no IPC, pipe, file, socket, shared memory, or environment path to receiver query
```

## Receiver Capability

Before receiver feature freeze, the receiver may read:

```text
public schedule skeleton
public carrier layout
query seed selected after source closure
own PMU/timing/raw measurements
public restoration plan
```

The receiver must be technically incapable of reading:

```text
private relation labels
hidden branch
source work coordinates
private source receipts
expected geometry
adjudication thresholds selected from the result
```

## Query Timing

The receiver query is selected only after source preparation completion. The query
selection receipt must bind:

```text
source-closed timestamp/order marker
query seed
query basis
query order
query operator
receiver process identity
```

## Adjudication Timing

Private relation receipts and source receipts are opened only after:

```text
receiver raw capture exists
receiver features are frozen
feature hash is sealed
restoration measurements are sealed
copy-back/custody evidence is complete
```

## Invalid Custody Cases

The package must classify as custody-invalid, not positive evidence, when:

```text
source sees query
receiver sees private map
receiver sees expected response
features are mutated after unblinding
schedule is incomplete or contains unknown rows
any duplicate or missing run identity exists
```

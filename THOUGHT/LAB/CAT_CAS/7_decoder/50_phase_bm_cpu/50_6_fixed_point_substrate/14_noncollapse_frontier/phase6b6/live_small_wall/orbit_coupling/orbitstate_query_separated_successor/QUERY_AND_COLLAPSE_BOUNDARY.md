# Query And Collapse Boundary

Status: `FROZEN_DESIGN_CONSTRAINT`

The receiver query is the first operation allowed to select a basis for extracting
the relation. It occurs only after source preparation has closed.

## Preparation Boundary

Before source closure:

```text
relation object exists
both branches contribute to shared carrier
no receiver query exists in source-readable state
no fold-odd result is computed
```

The preparation may load carrier topology, path history, or relation basis, but not a
winning branch or query answer.

## Query Boundary

After source closure, the receiver selects:

```text
query seed
query basis
query order
query operator
measurement windows
```

The query is public after selection and must be recorded in a receiver-only receipt.

## Collapse Boundary

The collapse boundary is:

```text
receiver applies public query operator to the prepared carrier and freezes the
predeclared fold-odd observable
```

Only this boundary may expose the fold-odd invariant. Adjudication may not select an
observable after seeing results.

## Query Operators

Allowed query operators for the selected shared-pair topology:

```text
basis-specific ownership-intent probe
bank-resolved reversal probe
topology closure probe
carrier-off matched probe
geometry-null matched probe
```

Forbidden query operators:

```text
operators that read private source files
operators that replay source q values
operators selected from observed response quality
operators that depend on candidate label names
```

## Required Query Receipts

Each query receipt must bind:

```text
source-closure receipt hash
query seed
query basis
query order
query operator
receiver binary hash
receiver process identity
raw measurement window ids
```

The query receipt must not bind:

```text
private branch labels
expected response
target-derived thresholds
post-hoc inclusion/exclusion choices
```

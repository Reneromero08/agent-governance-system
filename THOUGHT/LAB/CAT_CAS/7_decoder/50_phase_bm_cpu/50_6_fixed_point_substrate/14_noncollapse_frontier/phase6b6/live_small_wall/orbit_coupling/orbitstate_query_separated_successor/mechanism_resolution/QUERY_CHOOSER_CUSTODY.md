# Query Chooser Custody

Status: `CUSTODY_LAW_FROZEN_FOR_FUTURE_PROTOCOL`

## Required Causal Mechanism

The query chooser must be a causal mechanism, not a timestamp claim.

Required sequence:

```text
source process death or irreversible capability revocation
monotonic source-closure commitment
fresh high-entropy query generated afterward
no shared PRNG state with source
query/operator/order sealed before measurement
measurement windows sealed before observation
receiver features frozen before private unblinding
```

## Invalid Cases

Classify as custody-invalid:

```text
preselected query
source-visible query seed
shared PRNG state
post-measurement window choice
query selected after inspecting raw windows
receiver has private source map before feature freeze
source can write receiver-readable query receipt after closure
```

## Relation To Identifiability

This law prevents source-side query visibility. It does not defeat the finite answer
cache when `Q` is public and finite. Both laws are required.


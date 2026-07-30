# M113 maximal-real-subfield Horner review

## Decision

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level   SEPARATE_REFERENCE_PARITY
restoration class    EXACT_ALGEBRAIC_RESTORATION
```

The reviewed production source has SHA-256
`2a03970a4f40b9dddcad743d9728774d8a3155ada114ba749822a85abea19266`.
The separate oracle has SHA-256
`27b89803db94b4d9175ae80a5fa6d74f6d0c42ec59e2e8a0707a6c47354ad8c9`.
Both were executed from the repository Linux virtual environment. No hosted
workflow reexecution is established.

## Independent reconstruction

Production multiplies real norm state in the integral basis
`(1,s1,...,s7)`. The oracle does not import that successor and instead uses
the power basis for `Q[y]` modulo

```text
y^8 + y^7 - 7y^6 - 6y^5 + 15y^4 + 10y^3 - 10y^2 - 4y + 1.
```

It independently reconstructs the two public operators, advances Horner
coefficients sequentially by `x mod q` rather than production binary
powering, and reexecutes forward state, boundary projection, inverse
rematerialization, restoration, and reuse. It reproduces all 98 direction
norm factors, all 9,604 pair products, trace values, all four boundaries,
complete named resource tuples, period-1 cross-family restored-carrier reuse,
and exact period-1 and period-64 restoration for both public families.

The oracle converts power-basis values to the declared integral basis for
payload parity. It is therefore independent in algebra and execution, not in
the choice of payload convention. Production table release and actual Python
object lifetimes remain source-audited facts; no whole-process peak is
claimed.

## Resource result

The accepted warm table is 3,407 logical payload bits:

```text
retained full unit generators       294
public direction descriptors        434
degree-8 real norm factors         2679
```

The one-time compiler transition is 11,861 bits and includes the 8,177-bit
predecessor direction table, 571 distinct predecessor singleton norm
factors, descriptors, and the new real factors. Aliased generator
multipliers are not counted twice.

The exact named-component totals are:

```text
period family    phase total    matched raw Horner    phase minus raw
1      PRIMARY        93,479                10,005            83,474
1      REUSE         101,169                10,097            91,072
64     PRIMARY     3,466,215             2,790,766           675,449
64     REUSE       3,840,016             2,901,994           938,022
```

The search accounting includes the conversion pair, persistent current real
norm and energy, power result/factor pair, trial norm, scalar energy pair,
final full certified action, relative alignment, projection, and inverse
rematerialization. Stale full-norm, trial-norm, and scalar bindings are
released before later search work. The totals are conservative sums of named
component maxima, not legal live intervals or simultaneous process peaks.
Python object overhead, allocator behavior, internal integer multiplication,
native-library storage, and whole-process memory are outside the bound.

Because this accounting is stricter than predecessor package accounting, the
total is not promoted as a clean M112-to-M113 performance comparison. The
valid advance is the exact degree-8 norm-search algebra and smaller retained
table. All four phase totals remain above matched raw Horner, and an identical
normalized real-subfield algorithm is available to classical software.

## Restoration and ceiling

Restoration is exact over the discrete payload, pi ledger, and unit ledger on
the original backing, with no baseline reload. Generation and lease metadata
advance and are observed bookkeeping only, so full carrier-object equality
and bounded repeated-use metadata are not claimed. Cross-family reuse is
established only at period 1.

The result is limited to Linux x86_64 repository Python, two fixed public F17
period-17 families, periods 1 and 64, seven unit generators, 49 declared
directions and their fixed caps, degree-8 real norm search, one final full
certified action, and one named resident Horner carrier.

It does not establish a full real-subfield carrier, elimination of initial
full autocorrelation or final full action, fixed residual width, asymptotic
height control, CATVM custody, a distinct phase resource, computational
advantage, a Small Wall crossing, catalytic inference, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.

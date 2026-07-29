# Independent Review: Period-17 Pi-Unit Embedding Balance

## Decision

```text
classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

This decision is bound to:

```text
production source
    7dab9a3eec4d66e1ef92ae27348ae191a75c549f372e5a5fb452c385de9019f2

oracle source
    0abe868f118e562956ce77d1f8887459f5ae9fc288619985e5959a27ac388eb2

production full result
    12b7e0b174bf09c698ba04d7eab30bb2a0e136919cbcd996fb04f3c22e3913a3

oracle result
    07ee1f95f32b887eb758cbccedd0458897578f8bdaebdfdf071b97025af5eb4f
```

## Verified Scope

The production mechanism represents each nonzero exact value as a residual
in `Z[zeta_17]`, a power-of-`pi` ledger for `pi = 1 - zeta_17`, and a
seven-entry ledger over the declared cyclotomic-unit generators. It selects
unit moves with the exact field trace of `a * conjugate(a)`. This objective
equals the sum of squared magnitudes over all sixteen embeddings, and its
comparisons use integers rather than floating point.

The separate oracle imports no production successor code. It recompiles both
public operators, verifies their complete annihilator identities, advances
recurrence coefficients by sequential multiplication by `x mod q`, and
implements its own exact trace-energy balancing. Production uses binary
polynomial powering. The oracle reproduces every declared boundary, forward
resource tuple, and independently rebuilt inverse-rematerialization resource
tuple for both public families at periods `1` and `64`.

The exact unit balance substantially reduces the pi-factored residual
payload. The comparison remains negative against the identical raw
recurrence at period 64:

```text
period family    raw       pi-only     balanced resident   declared live
1      PRIMARY   1,306,953  6,152,594            601,130       1,202,260
1      REUSE     1,332,125  6,213,406            603,011       1,206,022
64     PRIMARY   2,368,807 11,160,253          5,489,878      10,979,756
64     REUSE     2,447,532 11,433,973          5,699,705      11,399,410
```

The declared live-state count includes the complete public-topology
rematerialization alongside the carrier without applying Python aliasing
discounts. It is a conservative logical signed-integer payload count, not a
whole-process memory measurement. At period 64 it remains above the raw
recurrence by `8,610,949` bits for `PRIMARY` and `8,951,878` bits for
`REUSE`.

The valid inverse independently rematerializes the exact public-topology
state, subtracts output, coefficients, basis messages, seed, pi ledgers, and
unit ledgers from the actual carrier, and restores every payload and ledger
cell to zero on the original backing. The unrelated family then reuses that
backing at period 1 and agrees with a fresh carrier. A nonzero unit-ledger
mutation changes the boundary.

## Claim Ceiling

```text
LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_Q_ZETA17_PI_CONTENT_PLUS_SEVEN_DECLARED_UNIT_LEDGER_EXACT_TRACE_ENERGY_BALANCE_128_STEP_CAP_PERIODS1_AND64_RESIDENT_AND_DECLARED_DUPLICATE_REMATERIALIZATION_LIVE_PAYLOAD_DIAGNOSTIC_EXACT_SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY
```

## Required Limitations

- The greedy search is bounded to the seven declared unit generators and
  128 selected steps per balance call. A cap hit does not certify a local or
  global optimum.
- Multiplicative independence of the declared unit generators is not
  certified.
- Cross-family restored-carrier reuse is tested at period 1. The period-64
  cases restore separately.
- The accepted counters include basis-operator work, exact norm
  construction, candidate norm multiplication, selected unit moves,
  scalar-vector multiplication, unit-power materialization, resident
  payload, and duplicate rematerialization payload. Python object overhead,
  allocator behavior, internal convolution temporaries, native-library
  storage, and whole-process peak memory remain outside the claim.
- Generation and lease metadata advance, so full carrier-object equality and
  bounded repeated-use metadata are not claimed.
- Wrong and reordered inverses are detected, but rejected attempts do not
  establish failure-atomic rollback.
- The identical exact recurrence and exact balancing objective are available
  to compact classical software.
- No CATVM custody, distinct phase resource, computational advantage, Small
  Wall crossing, catalytic inference, physical waveform execution, physical
  replacement of bits, or unbounded computation is established.

## Next Obstruction

Exact trace-energy unit balancing repairs much of the pi-factored residual
inflation, but both resident and conservatively declared live payload remain
above the identical raw recurrence at period 64, many calls reach the
declared search cap, and the same balancer is classically available. A
successor must change that obstruction rather than add periods or families.

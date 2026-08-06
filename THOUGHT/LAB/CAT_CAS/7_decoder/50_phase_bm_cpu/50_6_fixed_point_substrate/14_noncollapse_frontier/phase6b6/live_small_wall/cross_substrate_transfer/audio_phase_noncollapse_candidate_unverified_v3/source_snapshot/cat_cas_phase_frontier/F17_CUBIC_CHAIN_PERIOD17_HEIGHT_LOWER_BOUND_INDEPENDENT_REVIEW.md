# Independent Review: Period-17 Exact Boundary Height Lower Bound

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
    defeb3df76bad9ebb0fe2fe00e75c50d4e06c6ccbded18602fb55c49dad35857

oracle source
    43eb72bcddcbc43fa32c951ab5e66fa04de134c0a3a1c28cc053491e98caef7c

production full result
    0d9334dbf57f416e056abbd3d6b55981d666d2141e5d47ec7a7af05a3b22f224

oracle result
    4590afb996a51980feee6d408b0074b8d07d8a78351c4b9fc5e4b3561de59003
```

## Verified Mechanism

Let `pi = 1 - zeta_17`. The exact coefficient-prefix division identity in
`Z[zeta_17]` correctly computes `v_pi`. The declared lower bound

```text
L(n) = CEIL((272*n + 16)/3)
```

satisfies `L(n+3) = L(n) + 272`. Exact characteristic-coefficient
valuations meet the three residue-class induction requirements for both
public period-17 cubic path families. Both complete characteristic
identities are independently checked and hard-gated.

The production implementation directly evaluates exact boundaries for
periods 1 through 17. After division by `pi^L(n)`, its 16-lag recurrence
over `F17` reproduces period 17 and enters these finite cycles:

```text
PRIMARY
    prefix                  3
    cycle length            1,632
    nonzero cycle outputs   1,555

REUSE
    prefix                  0
    cycle length            14,688
    nonzero cycle outputs   13,826
```

A nonzero normalized output on each eventual cycle recurs infinitely often.
At those periods, the exact boundary valuation equals the strictly increasing
`L(n)`. Therefore there are infinitely many exact nonzero boundaries with
infinitely many distinct `pi`-valuations.

The separate oracle imports no production phase module. It uses the
previously sealed independent descriptor/ring kernel, recompiles both public
operators, checks the supplied annihilators, independently repeats exact
division and induction, and uses Brent rather than Floyd cycle detection. It
reproduces both exact cycles, all initial valuations and residues, and the
exact cycle densities `1555/1632` and `6913/7344`.

The oracle also detects a one-`pi` weakening of an induction coefficient and
a normalized recurrence-coefficient perturbation.

## Exact Claim Ceiling

```text
LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_Q_ZETA17_PI_ADIC_BOUNDARY_HEIGHT_LOWER_BOUND_IDENTICAL_COMPACT_CLASSICAL_RECURRENCE_EXACT_SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY
```

For an injective encoding from which the exact boundary or its valuation is
decodable through horizon `N`, when the decoder is not supplied the period
index for free, the worst-case code width through `N` is
`Omega(log N)`. This is a horizon/cardinality statement:

```text
max(code_width(boundary_m), m <= N) = Omega(log N)
```

It is not a pointwise bound at every period and not a generic machine-memory
or online-space lower bound. The `pi` exponent alone has an `O(log N)` ledger,
and compact classical software has the identical recurrence and certificate.

## Restoration and Accounting Scope

At 16 periods, the PRIMARY transaction and unrelated REUSE transaction
restore message payloads exactly on the same mutable backing. Fresh and
restored REUSE boundaries agree. There is no retained inverse history or
baseline reload.

This is exact algebraic message-payload restoration, not full carrier-object
equality. Generation and lease advance to two and remain monotone metadata;
their repeated-use width is not bounded here.

Resource figures are named logical components, not an exact temporary or
whole-process peak. They include both compiled operators and characteristics,
two direct stream messages, stored initial exact boundaries, the normalized
coefficient table and seed, the narrow Floyd core, and the 1,632-integer-cell
restoration carrier. Python objects, SymPy internals, allocator peak, and
whole-process peak remain unbounded by the evidence.

## Rejected Interpretations

The result does not establish:

- a linear bit lower bound;
- a pointwise `Omega(log n)` code width at every period;
- a lower bound when an external period index or counter is supplied free;
- absence of a compact indexed generator or variable-length encoding;
- full carrier-object restoration;
- bounded repeated-use generation/lease metadata;
- a distinct phase resource or computational advantage;
- a Small Wall crossing, CATVM custody, catalytic inference, physical
  waveform execution, replacement of physical bits, or unbounded
  computation.

## Next Obstruction

The exact boundary needs an unbounded lossless discrete boundary alphabet,
but its `pi` content has a compact logarithmic exponent ledger, and the
identical compact classical recurrence remains. A successor must address the
residual unit-normalized boundary content or introduce a phase-owned update
law unavailable to the matched compact classical recurrence; enlarging the
same fixture does not resolve this obstruction.

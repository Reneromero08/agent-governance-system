# Exact Coordinate Descent Independent Review

Date: 2026-07-29

Scientific source parent:
`98c5e242becf54ff184218157da9f334aa163c3f`

Decision:

```text
INDEPENDENTLY_VERIFIED_STRICT_SCOPE
SEPARATE_REFERENCE_PARITY
EXACT_ALGEBRAIC_RESTORATION
```

The separate oracle imports the established reference recurrence and ring
kernels, not the production successor. It recompiles both public F17
period-17 families, advances recurrence coefficients by sequential
multiplication by `x mod q`, and independently implements the exact
49-direction bracket and discrete-difference search.

The production and oracle full results are bound by provenance hashes:

```text
production  5e92a30f545088dfc0e55a052e01854353187c8a69a6d63cce0bf9dd74f3f1f0
oracle      de0d33558c1731877bc0b0e415c57edc47ac3b1e1b4fae46fa55a1347789e3da
```

All four declared cases (`PRIMARY` and `REUSE`, periods 1 and 64) agree on
the final boundary, exact balance/resource tuple, inverse rematerialization
tuple, and declared local-minimum certificate. The review also confirms:

- exact power accounting records the live result/factor pair initially and
  after every multiply or square;
- accepted move scales, ledger-derived unit scales, trial norms, energy
  pairs, accepted vectors, and unit rematerializations are represented in
  the named maxima;
- the central resident, duplicate-live, period-1, and period-64 payload
  predicates are production hard gates;
- each period-1 and period-64 transaction restores the same carrier backing
  exactly, while unrelated cross-family restored-carrier reuse is claimed
  and checked only at period 1;
- no baseline reload or retained inverse history is used.

The repaired accounting strengthens the obstruction. Resident and
duplicate-live payloads remain below the raw recurrence in all four cases,
but the conservative named-component maxima sum is approximately
8.48--8.51 Mbit at period 64 and remains above the 2.37--2.45 Mbit raw
recurrence.

Claim ceiling:

```text
Linux x86_64 Python software;
two public F17 period-17 cubic-path families;
periods 1 and 64;
49 declared exact unit-search directions;
exact boundary and component-level resource parity;
exact per-case restoration at periods 1 and 64;
cross-family restored-carrier reuse parity at period 1.
```

The named maxima sum is not a simultaneous process peak or an upper bound
on internal ring-multiplication, Python-object, allocator, or whole-process
memory. The result does not establish a global unit-lattice optimum, a
fixed total footprint, a distinct phase resource, computational advantage,
Small Wall crossing, CATVM custody, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

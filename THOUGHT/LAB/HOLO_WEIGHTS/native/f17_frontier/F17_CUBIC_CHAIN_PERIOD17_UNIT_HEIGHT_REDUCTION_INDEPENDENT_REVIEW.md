# Independent Review: Period-17 Cyclotomic-Unit Height Reduction

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
    dde70223c7b5be234501e872ebad267b2811668ad0a5e98192ce8f7100d3ea52

oracle source
    f89b775aa967035ed6e5a62330cceeb26101d5a556d2f71dd04541b3f8ba5d7d

production full result
    da9edf6387bf375484f51fb22bd3351016afe15ef5f0a4c245d27233425efd1b

oracle result
    fb053f81d019cdc37fc61e7138ff4d93e84d31c4f30685e61c1e75f886a7cf8f
```

## Verified Scope

The separate oracle imports no production module. It independently compiles
both public period operators, checks each complete annihilator identity, uses
sequential multiplication by `x` modulo the monic degree-16 factor, and
reexecutes the seven-generator unit normalization.

The oracle exactly reproduces all ten boundaries, ledger-inclusive carrier
payloads, coefficient widths, normalization calls, candidate evaluations,
selected moves, cap hits, exact valid-path restoration, backing identity, and
PRIMARY-to-REUSE restored-carrier reuse. The production and oracle stderr
streams are empty.

For generator index `a` and `b = a^-1 mod 17`, the implemented inverse

```text
sum(zeta^(a*m), m=0..b-1)
```

multiplies the declared geometric-series unit to one in
`Z[zeta_17]`. The ledger directions preserve the represented vector exactly.
The period operator is linear over the commutative cyclotomic ring, so the
ledger scale commutes through basis advancement. Final contraction against
the 16 resident basis messages reproduces the declared period boundary.

All ten declared cases reduce counted payload by between 15,534 and 176,398
bits. At period 256, PRIMARY uses 5,767,292 payload bits with maximum signed
width 8,931; REUSE uses 5,935,644 bits with width 8,988. Four normalization
calls per family reach the 128-step cap.

## Claim Ceiling

```text
LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_Q_ZETA17_SEVEN_DECLARED_CYCLOTOMIC_UNIT_GENERATORS_DETERMINISTIC_128_STEP_PER_CALL_SEARCH_PERIODS1_4_16_64_256_DENSE_DIRECT_PARITY_THROUGH64_SEPARATE_SEQUENTIAL_RECURRENCE_PARITY_THROUGH256_EXACT_VALID_PATH_SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY
```

The result establishes a bounded, exact, ledger-accounted height reduction.
It does not establish fixed-width storage. The same normalization and
recurrence are available to compact classical software.

## Required Limitations

- Eight cap hits preclude a strict local-minimum claim for every call and a
  global optimality claim.
- Multiplicative independence of the seven generators is not certified.
- Wrong and reordered inverses are detected, but rejected attempts do not
  provide failure-atomic rollback.
- Dense-direct execution is compared through period 64. Period 256 uses the
  independently reexecuted recurrence and exact annihilator.
- Exact payload accounting covers declared carrier integers and ledgers, not
  Python objects, allocator peak, transient bit-operation peak, or
  whole-process storage.
- No CATVM custody, distinct phase resource, computational advantage, Small
  Wall crossing, physical waveform execution, physical replacement of bits,
  or unbounded computation is established.

## Next Obstruction

The unit gauge reduces constants but does not arrest exact height growth.
Any successor must lawfully remove scale information from the semantic
machine, rematerialize it without an equivalent growing ledger, or establish
a different phase-native invariant. Enlarging the same two fixtures does not
address that obstruction.

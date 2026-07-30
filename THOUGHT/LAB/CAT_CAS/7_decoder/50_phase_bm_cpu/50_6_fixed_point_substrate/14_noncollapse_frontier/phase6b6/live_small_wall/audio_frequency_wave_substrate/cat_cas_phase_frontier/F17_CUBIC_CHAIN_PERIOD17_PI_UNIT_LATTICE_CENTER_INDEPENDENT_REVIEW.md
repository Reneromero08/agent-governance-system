# Independent Review: Period-17 Pi-Unit Lattice-Center Proposal

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
    184333ae7d1ae6e9440313fb7a9f81bb914b3c522e3328ea548cd2234b888e26

oracle source
    9efeb97ee144aa61e98a84af71539e95918eae58d985015f81c833a9adebcacb

production full result
    bd74878cd275e8ba036fd764513fa12619002863024348472d28a97cdcb6ab44

oracle full result
    6388b751dfb185a3704fc1b4e7ef0a259f8b56564bfba2097df0a1dbe1b6a983
```

## Verified Scope

The production successor represents each nonzero exact value by a
`Z[zeta17]` residual, a `pi = 1 - zeta17` exponent, and seven declared
cyclotomic-unit exponents. For each nonzero normalization call, it evaluates
eight norm embeddings from a fixed 65,536-bit cosine table and proposes one
global integer unit-lattice move with a float64 least-squares center. The
proposal is approximate. The carrier changes only when an exact integer
field-trace comparison proves that the proposed residual has lower energy.

The alternate oracle imports no production successor. It recompiles both
public operators, verifies their complete annihilator identities, advances
recurrence coefficients by sequential multiplication by `x mod q`, and
computes its center through float64 normal equations rather than the
production least-squares call. It reproduces every proposal commitment,
exact boundary, integer precision/resource tuple, separately rebuilt inverse
tuple, mutation response, and restored-reuse result at periods `1` and `64`.

```text
period family    raw       pi-only     resident   duplicate-live   named-maxima sum
1      PRIMARY   1,306,953  6,152,594   430,790          861,580        1,701,036
1      REUSE     1,332,125  6,213,406   430,336          860,672        1,701,260
64     PRIMARY   2,368,807 11,160,253   773,662        1,547,324        5,891,350
64     REUSE     2,447,532 11,433,973   782,790        1,565,580        5,942,297
```

All resident and duplicate-rematerialization live payloads are below the
identical raw recurrence. All conservative named-component maxima sums are
above raw. The latter adds separately observed maxima and is not a measured
simultaneous process peak.

The inverse independently rematerializes the public-topology state and
subtracts output, coefficients, basis messages, seed, pi ledgers, and unit
ledgers from the actual carrier. Every exact payload and ledger cell returns
to zero on the original backing. The unrelated family then uses that backing
at period 1 and agrees with fresh execution. No baseline reload or retained
inverse history is used.

## Claim Ceiling

```text
LINUX_X86_64_PYTHON_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_PERIODS1_AND64_65536_BIT_LOG_EMBEDDING_CENTER_PROPOSAL_EXACT_INTEGER_TRACE_ACCEPTANCE_EXACT_BOUNDARY_PAYLOAD_WIDTH_PROPOSAL_COMMITMENT_INVERSE_AND_REUSE_PARITY_COMPONENT_LEVEL_ACCOUNTING_SOFTWARE_ONLY
```

## Required Limitations

- Neither embedding values nor the lattice-center proposal are certified
  exact, and no exact or approximate closest-vector optimality is established.
- The fixed cosine table contributes 589,824 nominal mantissa bits. The
  float64 matrix, solve scratch, fixed-precision scalar work, rolling
  commitment, exact norm/scale/trial temporaries, materialized unit values,
  and duplicate inverse rematerialization are declared.
- The named-component maxima sum combines maxima that need not coexist.
  Python objects, allocator state, native-library storage, internal
  multiplication peak, bit-operation cost, and whole-process RSS are not
  bounded.
- Generation and lease counters are descriptive direct-process bookkeeping;
  their enforcement or bounded repeated-use width is not claimed.
- The identical fixed-precision proposal and exact-acceptance recurrence are
  available to compact classical software.
- No CATVM custody, distinct phase resource, computational advantage, Small
  Wall crossing, catalytic inference, physical waveform execution, physical
  replacement of bits, or unbounded computation is established.

## Next Obstruction

The bounded proposal reduces carrier payload, but its fixed high-precision
table and exact temporary maxima leave the conservative declared sum above
raw, while the same method remains classically available. A successor must
remove or compact that cost rather than add periods or families.

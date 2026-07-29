# Independent Review: F17 Period-17 Executed Recurrence

## Decision

```text
classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification level
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION

scientific source commit
    7fa4ec7d5f7d19ed1c6072ecfe8f19a62596f282
```

The accepted ceiling is:

```text
LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_Q_ZETA17_CHARACTERISTIC_RECURRENCE_PERIODS1_4_16_64_FIXED18_RESIDENT_MESSAGE_SLOTS_PLUS16_CYCLOTOMIC_COEFFICIENT_REGISTERS_DIRECT_DENSE_BOUNDARY_PARITY_EXACT_SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY
```

## Independent reconstruction

The separate oracle copies no production code and imports no production
module. It independently compiles the two public descriptors and their
17-by-17 operators with tuple `Z[zeta17]` arithmetic, checks the supplied
whole-operator annihilators, and advances the degree-16 native-`K`
recurrence by sequential multiplication by `x` modulo the monic factor
`q`. Production instead uses binary polynomial powering.

For PRIMARY and REUSE at periods `1, 4, 16, 64`, the oracle independently
reconstructs the recurrence and dense boundaries. It also reproduces every
reported exact boundary payload, carrier peak payload, coefficient signed
width, nonzero message/register count, and direct two-message payload/width
tuple.

## Restoration and controls

The oracle separately executes exact subtractive cleanup of the output,
coefficient registers, 16 retained basis messages, and seed. It checks the
identity of the message container, all 18 resident message slots, and the
16-register coefficient container. PRIMARY restores before REUSE consumes
the same backing; generation and lease reach two, fresh REUSE agrees, all
state is zero, and no snapshot reload or separate inverse-operation log is
used.

Production controls reject or detect missing inverse, wrong inverse,
reordered inverse, null carrier, and semantic-family perturbation. The
wrong-inverse control is not a CATVM failure-atomicity result because it may
reject after partial subtraction.

## Resource finding

The fixed count is 18 resident message slots plus 16 cyclotomic coefficient
registers, not fixed total storage. The 16 basis messages retain 4,352
integer cells of forward, inverse-enabling state; seed plus basis occupies
4,624 cells. Both compiled family operators occupy 9,248 integer cells and
both characteristics occupy 576 cells during the full run.

At period 64, recurrence-carrier payload is 2,368,807 bits for PRIMARY and
2,447,532 bits for REUSE. The matched direct two-message execution uses
1,221,725 and 1,246,736 bits. Maximum signed widths are identical between
the recurrence and direct paths at that period: 2,266 and 2,313 bits.
Named logical-cell accounting excludes Python objects, SymPy internals,
allocator/native-library peaks, bit-operation peaks, and whole-process
peaks.

## Claim boundary

The algebra `p(x) = x q(x)` and
`A^n = A (x^(n-1) mod q)(A)` is exact over
`K = Q(zeta17)`. It is not a scalar-rational recurrence or a minimal-order
result. PRIMARY has zero `q(0)`; REUSE has nonzero `q(0)`, but neither has
certified cyclotomic-unit or integral-inverse status. An integrally
reversible rolling window is therefore not established.

Compact classical software can execute the identical cyclotomic recurrence.
The package establishes no distinct phase resource, computational
advantage, Small Wall crossing, CATVM custody, catalytic inference,
physical waveform execution, physical bit replacement, or unbounded
computation.

# Growing Rotor Dihedral-Quotient Bosonic Fourier Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE` at
`INDEPENDENT_ORACLE_REEXECUTION`, with `NO_RESTORATION_CLAIM`.

The source momentum-bracelet carrier and the target zero-total-coordinate,
reflection-closed position sector each contain exactly 2,277 F103
coordinates. Equal dimension is genuine, but it does not make either tested
Fourier route compact in the required sense.

The existing exact Gaussian-elimination Fourier factorization contains 288
forward and 289 inverse elementary single-particle gates. Its first forward
gate is the shear `(15,16,10)`. Applied to the degree-six rotation orbit sum,
that gate creates a coefficient mismatch `6 * 10 = 60 mod 103` between
rotated monomials. It therefore leaves the fixed 2,277-coordinate rotation
quotient immediately. This is a statement about the existing 577-gate
factorization, not every possible structured transform.

The direct quotient-to-quotient kernel is also exact. A target with all six
particles at position zero gives a row with all 2,277 entries nonzero. A
source with all six particles at momentum zero gives a column whose 2,277
entries all equal 17. A generic kernel entry is a repeated `6 x 6` Fourier
minor permanent: the production 64-state subset recurrence and the
independent 720-assignment formula both return 65.

Retaining the full direct kernel would require 5,184,729 field cells. Streaming
every kernel entry avoids that storage but invokes 5,184,729 permanent
evaluations, with a declared maximum of 331,822,656 subset states. The
diagnostic deliberately executes only the exact full row, full column, and
generic-kernel certificates. It does not represent those route costs as a
universal lower bound, and it does not claim to have executed the complete
`2,277 x 2,277` transform.

## Independent reconstruction

The oracle imports no CAT_CAS module. It separately enumerates 74,613
exchange-symmetric occupations, 2,277 momentum bracelets, 4,389 zero-total
position occupations, and 2,277 reflection-closed target cells. It evaluates
the direct kernel by Ryser's formula rather than the production subset DP,
reconstructs the dense row and column commitments, independently derives the
first forward gate from the public order-17 Fourier matrix, and confirms the
generic permanent with all 720 assignments.

The wrong-total-coordinate control is zero after the complete source rotation
orbit sum. Dropping the source orbit factor changes the origin column from 17
to 1. Setting the first shear factor to zero removes the reported mismatch;
the actual nonzero factor discriminates.

## Resource and claim ceiling

The accepted diagnostic retains no dense direct kernel, full occupation
scratch, or transition plan. It does retain both 2,277-entry certificate
vectors, the source and target representative histograms, and 128 DP field
cells. Python containers, arbitrary-width integers, allocator/interpreter
memory, timing, and whole-process peaks are excluded rather than counted as
zero.

The strongest matched baselines are the identical direct F103 bosonic-kernel
recurrence, M201's identical 74,613-cell elementary bosonic transform, and
M204's identical eight-channel implicit-dihedral vector stream.

Exact ceiling:

```text
GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_STATIC_FORWARD_KERNEL_AND_COMPILED_GAUSSIAN_ELIMINATION_FOURIER_NETWORK_DIAGNOSTIC_ONLY
```

M204 execution, exact restoration, and reuse remain separately valid. This
diagnostic establishes neither a universal no-go for every structured
2,277-cell transform nor a full direct transform execution. It establishes
no CATVM custody, distinct phase resource, computational advantage, Small
Wall crossing, physical waveform execution, replacement of physical bits
with pi, or unbounded catalytic computation.

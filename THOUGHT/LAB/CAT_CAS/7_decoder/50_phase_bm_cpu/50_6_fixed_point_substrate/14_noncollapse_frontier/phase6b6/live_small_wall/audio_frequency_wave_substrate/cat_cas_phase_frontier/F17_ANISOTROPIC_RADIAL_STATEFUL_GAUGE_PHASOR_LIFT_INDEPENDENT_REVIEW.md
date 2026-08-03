# M146 independent review

Decision: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`.

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`.

Resident 51-angle carrier restoration:
`NUMERICAL_PHYSICAL_STATE_RESTORATION`.

Transient Cartesian, chart, comparison, and verification buffers:
`NO_RESTORATION_CLAIM`.

M146 repairs the M145 zero-fiber obstruction within a declared interior
magnitude envelope.  A two-phasor mean has an `S1` fiber at zero, whereas a
generic nonzero value has two ordered preimages.  For the tested two-cell
Givens coupling this prevents an injective fixed four-angle lift: a
zero/nonzero input fiber is uncountable while the generic/generic output fiber
has four elements.  The production zero probe also exhibits two distinct
input charts that collapse to the same legacy canonical output chart.

The repaired chart stores three phase angles per coordinate,

```text
z = epsilon exp(i g) + (1-epsilon)/2 (exp(i a) + exp(i b)),
```

with `epsilon=1/32`.  The declared base magnitude ceiling `15/16` guarantees
that the residual two-phasor chart remains representable for every gauge.
At base zero the residual radius is `1/31`, so the gauge is not discarded by
canonicalization.  Public phase modules rotate all three angles.  Each local
Givens update transports the two gauges by an invertible public
counter-rotation before recharting the two affected base amplitudes.

The accepted carrier is a fixed 51-angle `float64` array (408 resident bytes),
adding 17 angles and 136 resident bytes to M145.  It retains no inverse
history or restoration baseline.  The actual inverse is applied to the same
backing array, the generation advances exactly once, and the unrelated
depth-1537 program reuses that restored backing without snapshot reload.
Across the 21 declared cases through depth 4096, the exact-zero adversary, and
100 consecutive depth-64 reuse cycles, maximum recorded boundary error is
`7.313e-11` and maximum recorded restoration error is `1.766e-11`, within
the predeclared numerical tolerances.

The independent oracle imports neither production nor predecessor code.  It
reconstructs the weighted dense radial operator in long-double arithmetic,
checks symmetry, orthogonality, and involution independently, constructs a
separate `float64` QR factorization, and separately implements the 51-angle
forward/inverse path, zero-fiber collision and repair, mutation control, and
reuse controls.  Its 233 declared comparisons cover all 21 cases and the
sealed controls.  Two fresh no-write qualifier replays reproduced the
production and oracle JSON files byte-for-byte.

Frozen package hashes are:

```text
production source  27a147d45415e97de5fafd9436cfce7a3f1cb0ae0cbd3410189560282a760590
oracle source      eb236e860a0e503ebd15b1f981f25811351248fc0f75ec9694125640f394e233
production JSON    bcb917199e9d799ad53eba9cf692a30345162a9036a75f8c11d9ffaeeaf8dcac
oracle JSON        bfc03a9667d78bc07cdae7f8571443743955c90c9d2b5bb3bd31de1dc8a3a62b
qualifier          ffb1d7c67268cb06b5f739b8c86770f0325bcb1ef9bee3a7231ccea5fa0de172
```

All three declared compact classical comparisons execute in every production
case.  The identical local-plan 17-complex recurrence reaches 3,055 named
warm bytes; the matrix-free work frontier reaches 1,767 named warm bytes; and
the streamed real-kernel memory frontier reaches 983 named warm bytes.  The
accepted 51-angle path reaches 3,223 named warm bytes and performs additional
input-phasor, gauge, and chart trigonometry.  Python containers, allocator and
native-library storage, and whole-process peak memory are excluded from every
named accounting total.

Controls cover plan mutation, missing and wrong inverse, reordered inverse,
missing gauge inverse, premature projection, null carrier, out-of-envelope
input, and a phase-disabled path that is inapplicable to the declared
envelope.  The result is limited to the sealed direct-process Linux cases and
the declared interior envelope.

M146 establishes neither a global full-sphere chart, Cartesian-free updates,
exact algebraic restoration, CATVM custody, a distinct phase resource,
computational advantage, a Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, nor unbounded catalytic computation.
It leaves the gauge as additional phase-resident state while retaining local
Cartesian recharting and smaller executed complex recurrences.
